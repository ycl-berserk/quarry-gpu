// GPU 上的 LBVH 构建：Morton 码 → 基数排序 → 每个内部节点独立二分 → 自底向上合并 AABB。
//
// 每一步都是"每个线程只看自己那一份数据"的形态，没有任何一步需要递归或动态分配。
// 唯一的顺序依赖是 AABB 合并：父节点的包围盒必须等两个孩子都算完。孩子一定比父亲
// 更深（每层 LCP 严格增加），所以按深度从深到浅处理就是合法拓扑序。
// LCP 降序同样合法，但深度的桶数天然有界（<=64），实现更直接。

#include "gpu/gpu_bvh.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "gpu/gpu_node.h"

namespace {

constexpr int kBlock = 256;
constexpr int kDepthBins = 128;  // 每层 LCP 至少 +1，深度不会超过 64；留一倍余量

void cudaOk(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(err));
    }
}

template <typename T>
struct DevBuf {
    T* p = nullptr;
    explicit DevBuf(size_t count) {
        cudaOk(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)), "cudaMalloc");
    }
    ~DevBuf() {
        if (p) cudaFree(p);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

int blocksFor(int n) { return (n + kBlock - 1) / kBlock; }

struct StageClock {
    cudaEvent_t ev[8] = {};
    int marks = 0;

    StageClock() {
        for (cudaEvent_t& e : ev) cudaOk(cudaEventCreate(&e), "cudaEventCreate");
    }
    ~StageClock() {
        for (cudaEvent_t& e : ev) cudaEventDestroy(e);
    }
    void mark() { cudaOk(cudaEventRecord(ev[marks++]), "cudaEventRecord"); }
    double ms(int a, int b) const {
        float v = 0.0f;
        cudaEventElapsedTime(&v, ev[a], ev[b]);
        return static_cast<double>(v);
    }
};

// ---------------------------------------------------------------------------
// 浮点数的全序表示：把 IEEE-754 位模式映射成单调的 uint32，
// 这样 atomicMin/atomicMax 就能直接用在浮点上（先翻转符号位，负数整体取反）。
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned floatOrdered(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float floatFromOrdered(unsigned u) {
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
}

// 每个 block 归约一次，最后对 3 个全局计数器做 atomic —— 不是每个线程都 atomic。
__device__ unsigned gCentroidLo[3];
__device__ unsigned gCentroidHi[3];

__global__ void resetCentroidBoundsKernel() {
    if (threadIdx.x < 3) {
        gCentroidLo[threadIdx.x] = floatOrdered(CUDART_INF_F);
        gCentroidHi[threadIdx.x] = floatOrdered(-CUDART_INF_F);
    }
}

__global__ void centroidBoundsKernel(const Tri* __restrict__ tris, int n) {
    __shared__ float s[6][kBlock];
    const int tid = threadIdx.x;

    float lo0 = CUDART_INF_F, lo1 = CUDART_INF_F, lo2 = CUDART_INF_F;
    float hi0 = -CUDART_INF_F, hi1 = -CUDART_INF_F, hi2 = -CUDART_INF_F;

    for (int i = blockIdx.x * kBlock + tid; i < n; i += kBlock * gridDim.x) {
        const Tri t = tris[i];
        // 与 CPU 完全相同的重心公式（先加后除，三步顺序一致）
        const float cx = (t.a.x + t.b.x + t.c.x) / 3.0f;
        const float cy = (t.a.y + t.b.y + t.c.y) / 3.0f;
        const float cz = (t.a.z + t.b.z + t.c.z) / 3.0f;
        lo0 = fminf(lo0, cx);
        lo1 = fminf(lo1, cy);
        lo2 = fminf(lo2, cz);
        hi0 = fmaxf(hi0, cx);
        hi1 = fmaxf(hi1, cy);
        hi2 = fmaxf(hi2, cz);
    }

    s[0][tid] = lo0;
    s[1][tid] = lo1;
    s[2][tid] = lo2;
    s[3][tid] = hi0;
    s[4][tid] = hi1;
    s[5][tid] = hi2;
    __syncthreads();

    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int c = 0; c < 3; ++c) {
                s[c][tid] = fminf(s[c][tid], s[c][tid + stride]);
                s[c + 3][tid] = fmaxf(s[c + 3][tid], s[c + 3][tid + stride]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int c = 0; c < 3; ++c) {
            atomicMin(&gCentroidLo[c], floatOrdered(s[c][0]));
            atomicMax(&gCentroidHi[c], floatOrdered(s[c + 3][0]));
        }
    }
}

__device__ __forceinline__ uint32_t expand10(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

__device__ __forceinline__ uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    return expand10(x) | (expand10(y) << 1) | (expand10(z) << 2);
}

__device__ __forceinline__ uint32_t quantize(float v, float lo, float extent) {
    if (extent <= 0.0f) return 0;
    const float q = (v - lo) / extent * 1023.0f + 0.5f;
    const int i = static_cast<int>(fminf(1023.0f, fmaxf(0.0f, q)));
    return static_cast<uint32_t>(i);
}

// 排序键 = 30 位 Morton 码 << 32 | 原始索引。低 32 位让键唯一，
// 否则同一量化格子里的三角形无法区分，树的形状也就不唯一。
__global__ void mortonKernel(const Tri* __restrict__ tris, int n, uint64_t* __restrict__ keys) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= n) return;

    const float lo[3] = {floatFromOrdered(gCentroidLo[0]), floatFromOrdered(gCentroidLo[1]),
                         floatFromOrdered(gCentroidLo[2])};
    const float hi[3] = {floatFromOrdered(gCentroidHi[0]), floatFromOrdered(gCentroidHi[1]),
                         floatFromOrdered(gCentroidHi[2])};

    const Tri t = tris[i];
    const float c[3] = {(t.a.x + t.b.x + t.c.x) / 3.0f, (t.a.y + t.b.y + t.c.y) / 3.0f,
                        (t.a.z + t.b.z + t.c.z) / 3.0f};

    const uint32_t code = morton3D(quantize(c[0], lo[0], hi[0] - lo[0]),
                                   quantize(c[1], lo[1], hi[1] - lo[1]),
                                   quantize(c[2], lo[2], hi[2] - lo[2]));
    keys[i] = (static_cast<uint64_t>(code) << 32) | static_cast<uint32_t>(i);
}

__global__ void iotaKernel(int* __restrict__ v, int n) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < n) v[i] = i;
}

__global__ void scatterTrisKernel(const Tri* __restrict__ src, const int* __restrict__ order,
                                  int n, Tri* __restrict__ dst, int* __restrict__ primIds) {
    const int k = blockIdx.x * kBlock + threadIdx.x;
    if (k >= n) return;
    const int slot = order[k];
    dst[k] = src[slot];
    primIds[k] = slot;
}

__global__ void internalInitKernel(GpuBVHNode* __restrict__ nodes, int internalCount) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= internalCount) return;
    nodes[i].mnx = CUDART_INF_F;
    nodes[i].mny = CUDART_INF_F;
    nodes[i].mnz = CUDART_INF_F;
    nodes[i].mxx = -CUDART_INF_F;
    nodes[i].mxy = -CUDART_INF_F;
    nodes[i].mxz = -CUDART_INF_F;
}

__global__ void leafInitKernel(GpuBVHNode* __restrict__ nodes, const Tri* __restrict__ sortedTris,
                               int n) {
    const int k = blockIdx.x * kBlock + threadIdx.x;
    if (k >= n) return;
    GpuBVHNode& leaf = nodes[n - 1 + k];
    setNodeBox(leaf, sortedTris[k].bounds());
    leaf.left = -(k + 1);
    leaf.right = 1;
}

// refit 的第一步：调用方给的是场景顺序的三角形，树里存的是叶子顺序。
// primIds 是建树时定下的叶子槽 -> 场景索引，所以重排不需要在主机上做置换。
__global__ void gatherTrisKernel(const Tri* __restrict__ src, const int* __restrict__ primIds,
                                 int n, Tri* __restrict__ dst) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < n) dst[i] = src[primIds[i]];
}

// Karras 2012 的二叉基数树：内部节点 i 自己二分出自己的叶子区间和切分点。
// 每个线程只写 nodes[i]（以及 parent[child]），互不重叠。
__global__ void buildInternalKernel(const uint64_t* __restrict__ keys, int n,
                                    GpuBVHNode* __restrict__ nodes, int* __restrict__ parent) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= n - 1) return;

    auto delta = [&](int a, int b) -> int {
        if (b < 0 || b >= n) return -1;
        const uint64_t x = keys[a] ^ keys[b];
        return x == 0 ? 64 : __clzll(x);
    };

    const int dir = (delta(i, i + 1) > delta(i, i - 1)) ? 1 : -1;
    const int sigmaMin = delta(i, i - dir);

    int lMax = 2;
    while (delta(i, i + lMax * dir) > sigmaMin) lMax *= 2;

    int l = 0;
    for (int t = lMax / 2; t > 0; t /= 2) {
        if (delta(i, i + (l + t) * dir) > sigmaMin) l += t;
    }

    const int j = i + l * dir;
    const int lower = (i < j) ? i : j;
    const int upper = (i < j) ? j : i;

    const int deltaNode = delta(i, j);
    int gamma = lower;
    int p = lower, q = upper;
    while (p <= q) {
        const int mid = p + (q - p) / 2;
        if (delta(lower, mid) > deltaNode) {
            p = mid + 1;
        } else {
            gamma = mid - 1;
            q = mid - 1;
        }
    }

    const int leftId = (lower == gamma) ? (n - 1 + lower) : gamma;
    const int rightId = (upper == gamma + 1) ? (n - 1 + upper) : gamma + 1;

    nodes[i].left = leftId;
    nodes[i].right = rightId;
    parent[leftId] = i;
    parent[rightId] = i;
}

// 每个节点的深度 = 沿 parent 链走到根的距离。链长最多 64（每层 LCP 严格增加）。
// 选深度而不是直接按 LCP 分桶，是因为深度天然有界（<=64 个桶），且节点编号本身
// 不保证孩子比父亲小，不能直接按编号顺序算。
__global__ void depthKernel(const int* __restrict__ parent, int nodeCount,
                            int* __restrict__ depth, int* __restrict__ maxDepth) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= nodeCount) return;
    int d = 0;
    int p = i;
    while (p != 0 && d < nodeCount) {
        p = parent[p];
        ++d;
    }
    depth[i] = d;
    atomicMax(maxDepth, d);
}

// 按深度排好的桶里，每层的起点 = 该深度在有序键里的 lower_bound。
// 只有 <=64 个桶，一个 block 一次二分查完，比在主机上按计数做前缀省一次
// 大数组回传。
__global__ void depthOffsetsKernel(const uint32_t* __restrict__ sortedKeys, int nodeCount,
                                   int bins, int* __restrict__ offsets) {
    const int d = threadIdx.x;
    if (d > bins) return;
    int lo = 0;
    int hi = nodeCount;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (sortedKeys[mid] < static_cast<uint32_t>(d)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    offsets[d] = lo;
}

// 同一层里没有父子关系，可以整层并行；孩子的层已经在前一批 kernel 里算完。
// 叶子盒在 leafInitKernel 里就定稿了，这里只处理内部节点。
//
// 每个线程一次处理 kRefitUnroll 个节点：子节点下标是散的，单条链上两个 32B
// 读要串行等显存；把 kRefitUnroll 份摊开，8 次读能同时挂在飞行队列里。
constexpr int kRefitUnroll = 4;

__global__ void refitKernel(GpuBVHNode* __restrict__ nodes, const int* __restrict__ bucket,
                            int count) {
    const int base = blockIdx.x * (kBlock * kRefitUnroll) + threadIdx.x;

    int ids[kRefitUnroll];
    GpuBVHNode l[kRefitUnroll], r[kRefitUnroll];
    bool ok[kRefitUnroll];

#pragma unroll
    for (int u = 0; u < kRefitUnroll; ++u) {
        const int t = base + u * kBlock;
        ok[u] = false;
        if (t < count) {
            const int i = bucket[t];
            const GpuBVHNode nd = nodes[i];
            if (nd.left >= 0) {
                ok[u] = true;
                ids[u] = i;
                l[u] = nodes[nd.left];
                r[u] = nodes[nd.right];
            }
        }
    }

#pragma unroll
    for (int u = 0; u < kRefitUnroll; ++u) {
        if (!ok[u]) continue;
        GpuBVHNode& nd = nodes[ids[u]];
        nd.mnx = l[u].mnx < r[u].mnx ? l[u].mnx : r[u].mnx;
        nd.mny = l[u].mny < r[u].mny ? l[u].mny : r[u].mny;
        nd.mnz = l[u].mnz < r[u].mnz ? l[u].mnz : r[u].mnz;
        nd.mxx = l[u].mxx > r[u].mxx ? l[u].mxx : r[u].mxx;
        nd.mxy = l[u].mxy > r[u].mxy ? l[u].mxy : r[u].mxy;
        nd.mxz = l[u].mxz > r[u].mxz ? l[u].mxz : r[u].mxz;
    }
}

// ---------------------------------------------------------------------------
// 结构校验：父盒必须逐位等于两个子盒的并，叶子盒必须逐位等于三角形包围盒，
// 每个非根节点必须被恰好引用一次。
// ---------------------------------------------------------------------------
__global__ void checkNodesKernel(const GpuBVHNode* __restrict__ nodes,
                                 const Tri* __restrict__ sortedTris, int n, int nodeCount,
                                 int* __restrict__ visited, int* __restrict__ errors) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= nodeCount) return;

    const GpuBVHNode& nd = nodes[i];
    if (i < n - 1) {
        if (nd.left < 0 || nd.left >= nodeCount || nd.right < 0 || nd.right >= nodeCount) {
            atomicAdd(&errors[0], 1);
            return;
        }
        atomicAdd(&visited[nd.left], 1);
        atomicAdd(&visited[nd.right], 1);

        const AABB mine = nodeBox(nd);
        const AABB l = nodeBox(nodes[nd.left]);
        const AABB r = nodeBox(nodes[nd.right]);
        AABB uni;
        uni.extend(l);
        uni.extend(r);
        if (mine.mn.x != uni.mn.x || mine.mn.y != uni.mn.y || mine.mn.z != uni.mn.z ||
            mine.mx.x != uni.mx.x || mine.mx.y != uni.mx.y || mine.mx.z != uni.mx.z) {
            atomicAdd(&errors[0], 1);
        }
    } else {
        const int slot = -nd.left - 1;
        if (slot < 0 || slot >= n) {
            atomicAdd(&errors[0], 1);
            return;
        }
        const AABB mine = nodeBox(nd);
        const AABB tri = sortedTris[slot].bounds();
        if (mine.mn.x != tri.mn.x || mine.mn.y != tri.mn.y || mine.mn.z != tri.mn.z ||
            mine.mx.x != tri.mx.x || mine.mx.y != tri.mx.y || mine.mx.z != tri.mx.z) {
            atomicAdd(&errors[1], 1);
        }
    }
}

__global__ void checkRefsKernel(const int* __restrict__ visited, int nodeCount,
                                int* __restrict__ errors) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= nodeCount) return;
    const int want = (i == 0) ? 0 : 1;
    if (visited[i] != want) atomicAdd(&errors[2], 1);
}

}  // namespace

bool gpuDescribe(std::string* out) {
    int devices = 0;
    const cudaError_t err = cudaGetDeviceCount(&devices);
    if (err != cudaSuccess) {
        if (out) *out = cudaGetErrorString(err);
        return false;
    }
    if (devices == 0) {
        if (out) *out = "no CUDA device";
        return false;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
    if (out) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s sm_%d%d (%d SM, %.1f GiB)",
                      prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                      static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
        *out = buf;
    }
    return true;
}

void warmupBuildGPU() {
    resetCentroidBoundsKernel<<<1, 32>>>();
    cudaOk(cudaGetLastError(), "warmup");
    cudaOk(cudaDeviceSynchronize(), "warmup sync");
}

GpuBVH buildBVHGPU(const Scene& scene, GpuBuildTimings* timings) {
    const int n = static_cast<int>(scene.tris.size());
    if (n < 2) throw std::runtime_error("buildBVHGPU: at least 2 triangles required");

    const int nodeCount = 2 * n - 1;
    const int internalCount = n - 1;

    GpuBVH bvh;
    bvh.triCount = n;
    bvh.nodeCount = nodeCount;
    cudaOk(cudaMalloc(&bvh.d_nodes, static_cast<size_t>(nodeCount) * sizeof(GpuBVHNode)), "nodes");
    cudaOk(cudaMalloc(&bvh.d_tris, static_cast<size_t>(n) * sizeof(Tri)), "tris");
    cudaOk(cudaMalloc(&bvh.d_primIds, static_cast<size_t>(n) * sizeof(int)), "primIds");

    auto* nodes = static_cast<GpuBVHNode*>(bvh.d_nodes);
    auto* sortedTris = static_cast<Tri*>(bvh.d_tris);
    auto* primIds = static_cast<int*>(bvh.d_primIds);

    try {
        DevBuf<Tri> dSrc(n);
        cudaOk(cudaMemcpy(dSrc.p, scene.tris.data(), static_cast<size_t>(n) * sizeof(Tri),
                          cudaMemcpyHostToDevice),
               "H2D tris");

        DevBuf<uint64_t> dKeys(n), dKeysSorted(n);
        DevBuf<int> dOrder(n), dOrderSorted(n);
        DevBuf<int> dParent(nodeCount);
        DevBuf<int> dDepth(nodeCount);
        DevBuf<uint32_t> dDepthKeySorted(nodeCount);
        DevBuf<int> dDepthVal(nodeCount), dDepthValSorted(nodeCount);
        DevBuf<int> dOffsets(kDepthBins + 1);
        DevBuf<int> dMaxDepth(1);

        StageClock clock;
        clock.mark();  // 0

        resetCentroidBoundsKernel<<<1, 32>>>();
        centroidBoundsKernel<<<blocksFor(n), kBlock>>>(dSrc.p, n);
        mortonKernel<<<blocksFor(n), kBlock>>>(dSrc.p, n, dKeys.p);
        cudaOk(cudaGetLastError(), "morton");
        clock.mark();  // 1

        std::size_t sortBytes = 0;
        iotaKernel<<<blocksFor(n), kBlock>>>(dOrder.p, n);
        cudaOk(cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, dKeys.p, dKeysSorted.p,
                                               dOrder.p, dOrderSorted.p, n),
               "sort(probe)");
        DevBuf<unsigned char> dSortTemp(sortBytes);
        cudaOk(cub::DeviceRadixSort::SortPairs(dSortTemp.p, sortBytes, dKeys.p, dKeysSorted.p,
                                               dOrder.p, dOrderSorted.p, n),
               "sort");
        clock.mark();  // 2

        scatterTrisKernel<<<blocksFor(n), kBlock>>>(dSrc.p, dOrderSorted.p, n, sortedTris, primIds);
        cudaOk(cudaGetLastError(), "scatter tris");
        clock.mark();  // 3

        cudaOk(cudaMemset(dParent.p, 0xff, static_cast<size_t>(nodeCount) * sizeof(int)), "parent");
        internalInitKernel<<<blocksFor(internalCount), kBlock>>>(nodes, internalCount);
        buildInternalKernel<<<blocksFor(internalCount), kBlock>>>(dKeysSorted.p, n, nodes,
                                                                  dParent.p);
        cudaOk(cudaGetLastError(), "build internal");
        cudaOk(cudaMemset(dMaxDepth.p, 0, sizeof(int)), "maxDepth");
        depthKernel<<<blocksFor(nodeCount), kBlock>>>(dParent.p, nodeCount, dDepth.p, dMaxDepth.p);
        cudaOk(cudaMemcpy(&bvh.depth, dMaxDepth.p, sizeof(int), cudaMemcpyDeviceToHost), "depth");
        if (bvh.depth >= kDepthBins) throw std::runtime_error("buildBVHGPU: depth exceeds bin table");
        clock.mark();  // 4

        leafInitKernel<<<blocksFor(n), kBlock>>>(nodes, sortedTris, n);
        iotaKernel<<<blocksFor(internalCount), kBlock>>>(dDepthVal.p, internalCount);

        // 内部节点编号恰好是 0..n-2，叶子不会进桶——refit 少处理一半的节点。
        // 深度取值 <=64，7 位基数排序就够一次排完。稳定排序让同层节点按编号升序
        // 排在一起，refit 时相邻线程读相邻节点；也避开了 2M 线程抢 128 个计数器的
        // 全局 atomic（实测分布 + 落桶占了 2M 构建里 8.4 ms）。
        std::size_t depthSortBytes = 0;
        cudaOk(cub::DeviceRadixSort::SortPairs(nullptr, depthSortBytes,
                                               reinterpret_cast<const uint32_t*>(dDepth.p),
                                               dDepthKeySorted.p, dDepthVal.p, dDepthValSorted.p,
                                               internalCount, 0, 7),
               "depth sort(probe)");
        DevBuf<unsigned char> dDepthTemp(depthSortBytes);
        cudaOk(cub::DeviceRadixSort::SortPairs(dDepthTemp.p, depthSortBytes,
                                               reinterpret_cast<const uint32_t*>(dDepth.p),
                                               dDepthKeySorted.p, dDepthVal.p, dDepthValSorted.p,
                                               internalCount, 0, 7),
               "depth sort");

        const int bins = bvh.depth + 1;
        depthOffsetsKernel<<<1, kDepthBins + 1>>>(dDepthKeySorted.p, internalCount, bins,
                                                  dOffsets.p);
        cudaOk(cudaGetLastError(), "depth offsets");

        int offsets[kDepthBins + 1] = {};
        cudaOk(cudaMemcpy(offsets, dOffsets.p, static_cast<size_t>(bins + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "D2H offsets");
        if (offsets[bins] != internalCount) {
            throw std::runtime_error("buildBVHGPU: depth bucketing lost nodes");
        }

        // 拓扑不变，这个 (深度, 编号) 顺序也不变：留给 refit 逐层重放，
        // 省掉每次更新重算 parent/depth + 一次 7 位基数排序。
        cudaOk(cudaMalloc(&bvh.d_refitOrder, static_cast<size_t>(internalCount) * sizeof(int)),
               "refit order");
        cudaOk(cudaMemcpy(bvh.d_refitOrder, dDepthValSorted.p,
                          static_cast<size_t>(internalCount) * sizeof(int),
                          cudaMemcpyDeviceToDevice),
               "D2D refit order");
        bvh.refitOffsets.assign(offsets, offsets + bins + 1);

        // 从最深层往上：第 d 层的节点只读第 d+1 层，天然无竞争。
        for (int d = bins - 1; d >= 0; --d) {
            const int count = offsets[d + 1] - offsets[d];
            if (count == 0) continue;
            const int blocks = (count + kBlock * kRefitUnroll - 1) / (kBlock * kRefitUnroll);
            refitKernel<<<blocks, kBlock>>>(nodes, dDepthValSorted.p + offsets[d], count);
        }
        cudaOk(cudaGetLastError(), "refit");
        clock.mark();  // 5

        cudaOk(cudaEventSynchronize(clock.ev[clock.marks - 1]), "sync");

        if (timings) {
            timings->mortonMs = clock.ms(0, 1);
            timings->sortMs = clock.ms(1, 2);
            timings->scatterMs = clock.ms(2, 3);
            timings->treeMs = clock.ms(3, 4);
            timings->refitMs = clock.ms(4, 5);
            timings->totalMs = clock.ms(0, 5);
        }
    } catch (...) {
        destroyBVHGPU(bvh);
        throw;
    }

    return bvh;
}

void destroyBVHGPU(GpuBVH& bvh) {
    if (bvh.d_nodes) cudaFree(bvh.d_nodes);
    if (bvh.d_tris) cudaFree(bvh.d_tris);
    if (bvh.d_primIds) cudaFree(bvh.d_primIds);
    if (bvh.d_refitOrder) cudaFree(bvh.d_refitOrder);
    if (bvh.d_stage) cudaFree(bvh.d_stage);
    bvh = GpuBVH{};
}

// 只上传改动的三角形：主机侧已经按 dirty 顺序打包好，slot 是它的叶子槽。
// 写入 + 重算叶子盒在同一个线程里做，省一次 kernel。
__global__ void scatterLeafKernel(const int* __restrict__ slots, const Tri* __restrict__ src,
                                  int count, Tri* __restrict__ sortedTris,
                                  GpuBVHNode* __restrict__ nodes, int n) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count) return;
    const int slot = slots[i];
    const Tri t = src[i];
    sortedTris[slot] = t;
    GpuBVHNode& leaf = nodes[n - 1 + slot];
    setNodeBox(leaf, t.bounds());
}

double refitImpl(GpuBVH& bvh, const Scene& scene, const std::vector<int>* dirty,
                 GpuRefitTimings* timings) {    const int n = bvh.triCount;
    if (n < 2 || bvh.d_nodes == nullptr) throw std::runtime_error("refitBVHGPU: BVH not built");
    if (bvh.d_refitOrder == nullptr) throw std::runtime_error("refitBVHGPU: missing refit order");
    if (static_cast<int>(scene.tris.size()) != n) {
        throw std::runtime_error("refitBVHGPU: triangle count changed, rebuild instead");
    }

    auto* nodes = static_cast<GpuBVHNode*>(bvh.d_nodes);
    auto* sortedTris = static_cast<Tri*>(bvh.d_tris);
    const auto* primIds = static_cast<const int*>(bvh.d_primIds);
    const auto* order = static_cast<const int*>(bvh.d_refitOrder);

    if (dirty && bvh.primSlot.empty()) {
        // 反向映射只在第一次增量更新时建：建树路径不替不 refit 的调用方付这次回传。
        std::vector<int> hostPrimIds(static_cast<size_t>(n));
        cudaOk(cudaMemcpy(hostPrimIds.data(), primIds, static_cast<size_t>(n) * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "D2H primIds");
        bvh.primSlot.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) bvh.primSlot[static_cast<size_t>(hostPrimIds[i])] = i;
    }

    StageClock clock;
    clock.mark();  // 0

    if (dirty) {
        const int count = static_cast<int>(dirty->size());
        // 主机侧 gather：改动量通常远小于 n，拿到叶子序的紧凑数组再传一次。
        std::vector<int> slots(static_cast<size_t>(count));
        std::vector<Tri> moved(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            const int id = (*dirty)[static_cast<size_t>(i)];
            if (id < 0 || id >= n) throw std::runtime_error("refitBVHGPU: dirty index out of range");
            slots[static_cast<size_t>(i)] = bvh.primSlot[static_cast<size_t>(id)];
            moved[static_cast<size_t>(i)] = scene.tris[static_cast<size_t>(id)];
        }

        DevBuf<int> dSlots(std::max(1, count));
        DevBuf<Tri> dMoved(std::max(1, count));
        if (count > 0) {
            cudaOk(cudaMemcpy(dSlots.p, slots.data(), static_cast<size_t>(count) * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "refit H2D slots");
            cudaOk(cudaMemcpy(dMoved.p, moved.data(), static_cast<size_t>(count) * sizeof(Tri),
                              cudaMemcpyHostToDevice),
                   "refit H2D tris");
        }
        clock.mark();  // 1

        if (count > 0) {
            scatterLeafKernel<<<blocksFor(count), kBlock>>>(dSlots.p, dMoved.p, count, sortedTris,
                                                            nodes, n);
        }
    } else {
        // 暂存区首次用到才分配：不 refit 的调用方不付这块显存。
        if (bvh.d_stage == nullptr || bvh.stageCount < n) {
            if (bvh.d_stage) cudaFree(bvh.d_stage);
            cudaOk(cudaMalloc(&bvh.d_stage, static_cast<size_t>(n) * sizeof(Tri)), "refit stage");
            bvh.stageCount = n;
        }
        auto* dSrc = static_cast<Tri*>(bvh.d_stage);
        cudaOk(cudaMemcpy(dSrc, scene.tris.data(), static_cast<size_t>(n) * sizeof(Tri),
                          cudaMemcpyHostToDevice),
               "refit H2D");
        clock.mark();  // 1

        gatherTrisKernel<<<blocksFor(n), kBlock>>>(dSrc, primIds, n, sortedTris);
        leafInitKernel<<<blocksFor(n), kBlock>>>(nodes, sortedTris, n);
    }
    cudaOk(cudaGetLastError(), "refit scatter");
    clock.mark();  // 2

    // 叶子盒已经定稿，内部节点按缓存的层序从深到浅合并：第 d 层只读 d+1 层。
    const int bins = static_cast<int>(bvh.refitOffsets.size()) - 1;
    for (int d = bins - 1; d >= 0; --d) {
        const int count = bvh.refitOffsets[d + 1] - bvh.refitOffsets[d];
        if (count == 0) continue;
        const int blocks = (count + kBlock * kRefitUnroll - 1) / (kBlock * kRefitUnroll);
        refitKernel<<<blocks, kBlock>>>(nodes, order + bvh.refitOffsets[d], count);
    }
    cudaOk(cudaGetLastError(), "refit tree");
    clock.mark();  // 3
    cudaOk(cudaEventSynchronize(clock.ev[clock.marks - 1]), "refit sync");

    const double total = clock.ms(0, 3);
    if (timings) {
        timings->uploadMs = clock.ms(0, 1);
        timings->gatherMs = clock.ms(1, 2);
        timings->treeMs = clock.ms(2, 3);
        timings->totalMs = total;
        timings->updated = dirty ? static_cast<int>(dirty->size()) : n;
    }
    return total;
}

double refitBVHGPU(GpuBVH& bvh, const Scene& scene, GpuRefitTimings* timings) {
    return refitImpl(bvh, scene, nullptr, timings);
}

double refitBVHGPU(GpuBVH& bvh, const Scene& scene, const std::vector<int>& dirty,
                   GpuRefitTimings* timings) {
    return refitImpl(bvh, scene, &dirty, timings);
}

std::string validateBVHGPU(const GpuBVH& bvh, const Scene& scene) {
    const int n = bvh.triCount;
    const int nodeCount = bvh.nodeCount;
    if (nodeCount != 2 * n - 1) return "nodeCount != 2n-1";

    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);
    const auto* sortedTris = static_cast<const Tri*>(bvh.d_tris);

    DevBuf<int> dVisited(nodeCount);
    DevBuf<int> dErrors(4);
    cudaOk(cudaMemset(dVisited.p, 0, static_cast<size_t>(nodeCount) * sizeof(int)), "visited");
    cudaOk(cudaMemset(dErrors.p, 0, 4 * sizeof(int)), "errors");

    checkNodesKernel<<<blocksFor(nodeCount), kBlock>>>(nodes, sortedTris, n, nodeCount, dVisited.p,
                                                       dErrors.p);
    checkRefsKernel<<<blocksFor(nodeCount), kBlock>>>(dVisited.p, nodeCount, dErrors.p);

    int errors[4] = {};
    cudaOk(cudaMemcpy(errors, dErrors.p, sizeof(errors), cudaMemcpyDeviceToHost), "D2H errors");

    GpuBVHNode root{};
    cudaOk(cudaMemcpy(&root, nodes, sizeof(GpuBVHNode), cudaMemcpyDeviceToHost), "root");
    const AABB rootBox = nodeBox(root);
    const bool badRoot = rootBox.mn.x != scene.bounds.mn.x || rootBox.mn.y != scene.bounds.mn.y ||
                         rootBox.mn.z != scene.bounds.mn.z || rootBox.mx.x != scene.bounds.mx.x ||
                         rootBox.mx.y != scene.bounds.mx.y || rootBox.mx.z != scene.bounds.mx.z;

    std::string why;
    char buf[160];
    auto add = [&](const char* name, long long value) {
        std::snprintf(buf, sizeof(buf), "%s%s=%lld", why.empty() ? "" : " ", name, value);
        why += buf;
    };
    if (errors[0]) add("bad_children_or_box", errors[0]);
    if (errors[1]) add("bad_leaf_box", errors[1]);
    if (errors[2]) add("bad_refcount", errors[2]);
    if (badRoot) add("root_box_mismatch", 1);
    return why;
}
