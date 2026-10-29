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
#include "gpu/prim_ops.h"

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
    T* release() {
        T* q = p;
        p = nullptr;
        return q;
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

template <typename Ops>
__global__ void centroidBoundsKernel(const typename Ops::Item* __restrict__ items, int n) {
    __shared__ float s[6][kBlock];
    const int tid = threadIdx.x;

    float lo0 = CUDART_INF_F, lo1 = CUDART_INF_F, lo2 = CUDART_INF_F;
    float hi0 = -CUDART_INF_F, hi1 = -CUDART_INF_F, hi2 = -CUDART_INF_F;

    for (int i = blockIdx.x * kBlock + tid; i < n; i += kBlock * gridDim.x) {
        const Vec3 c = Ops::centroid(items[i]);
        lo0 = fminf(lo0, c.x);
        lo1 = fminf(lo1, c.y);
        lo2 = fminf(lo2, c.z);
        hi0 = fmaxf(hi0, c.x);
        hi1 = fmaxf(hi1, c.y);
        hi2 = fmaxf(hi2, c.z);
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
// 否则同一量化格子里的图元无法区分，树的形状也就不唯一。
template <typename Ops>
__global__ void mortonKernel(const typename Ops::Item* __restrict__ items, int n,
                             uint64_t* __restrict__ keys) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= n) return;

    const float lo[3] = {floatFromOrdered(gCentroidLo[0]), floatFromOrdered(gCentroidLo[1]),
                         floatFromOrdered(gCentroidLo[2])};
    const float hi[3] = {floatFromOrdered(gCentroidHi[0]), floatFromOrdered(gCentroidHi[1]),
                         floatFromOrdered(gCentroidHi[2])};

    const Vec3 c = Ops::centroid(items[i]);

    const uint32_t code = morton3D(quantize(c.x, lo[0], hi[0] - lo[0]),
                                   quantize(c.y, lo[1], hi[1] - lo[1]),
                                   quantize(c.z, lo[2], hi[2] - lo[2]));
    keys[i] = (static_cast<uint64_t>(code) << 32) | static_cast<uint32_t>(i);
}

__global__ void iotaKernel(int* __restrict__ v, int n) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < n) v[i] = i;
}

template <typename Ops>
__global__ void scatterTrisKernel(const typename Ops::Item* __restrict__ src,
                                  const int* __restrict__ order, int n,
                                  typename Ops::Item* __restrict__ dst,
                                  int* __restrict__ primIds) {
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

template <typename Ops>
__global__ void leafInitKernel(GpuBVHNode* __restrict__ nodes,
                               const typename Ops::Item* __restrict__ sortedItems, int n) {
    const int k = blockIdx.x * kBlock + threadIdx.x;
    if (k >= n) return;
    GpuBVHNode& leaf = nodes[n - 1 + k];
    setNodeBox(leaf, Ops::bounds(sortedItems[k]));
    leaf.left = -(k + 1);
    leaf.right = 1;
}

// refit 的第一步：调用方给的是场景顺序的图元，树里存的是叶子顺序。
// primIds 是建树时定下的叶子槽 -> 场景索引，所以重排不需要在主机上做置换。
template <typename Ops>
__global__ void gatherTrisKernel(const typename Ops::Item* __restrict__ src,
                                 const int* __restrict__ primIds, int n,
                                 typename Ops::Item* __restrict__ dst) {
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

// 增量路径：每层只合并脏叶子的祖先。发射容量取 min(脏叶子数, 该层原节点数)，
// 实际条数从显存读——省掉每层一次 30–60 µs 的 D2H 同步。
__global__ void refitAncestorKernel(GpuBVHNode* __restrict__ nodes,
                                    const int* __restrict__ bucket,
                                    const int* __restrict__ count, int capacity) {
    const int t = blockIdx.x * kBlock + threadIdx.x;
    if (t >= capacity || t >= *count) return;
    const int i = bucket[t];
    const GpuBVHNode node = nodes[i];
    const GpuBVHNode l = nodes[node.left];
    const GpuBVHNode r = nodes[node.right];
    GpuBVHNode& out = nodes[i];
    out.mnx = l.mnx < r.mnx ? l.mnx : r.mnx;
    out.mny = l.mny < r.mny ? l.mny : r.mny;
    out.mnz = l.mnz < r.mnz ? l.mnz : r.mnz;
    out.mxx = l.mxx > r.mxx ? l.mxx : r.mxx;
    out.mxy = l.mxy > r.mxy ? l.mxy : r.mxy;
    out.mxz = l.mxz > r.mxz ? l.mxz : r.mxz;
}

// ---------------------------------------------------------------------------
// 结构校验：父盒必须逐位等于两个子盒的并，叶子盒必须逐位等于三角形包围盒，
// 每个非根节点必须被恰好引用一次。
// ---------------------------------------------------------------------------
template <typename Ops>
__global__ void checkNodesKernel(const GpuBVHNode* __restrict__ nodes,
                                 const typename Ops::Item* __restrict__ sortedItems, int n,
                                 int nodeCount, int* __restrict__ visited,
                                 int* __restrict__ errors) {
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
        const AABB prim = Ops::bounds(sortedItems[slot]);
        if (mine.mn.x != prim.mn.x || mine.mn.y != prim.mn.y || mine.mn.z != prim.mn.z ||
            mine.mx.x != prim.mx.x || mine.mx.y != prim.mx.y || mine.mx.z != prim.mx.z) {
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

// ---------------------------------------------------------------------------
// 4 宽折叠。偶数深度的内部节点各生成一个 128B 节点，四个槽填"孩子 + 孙子"：
//   孩子是叶子  → 直接占一个槽；
//   孩子是内部  → 把它的两个孩子（孙辈）各占一个槽；
//   孩子是叶子、兄弟是内部 → 剩下的槽留空（空盒 + kWideEmpty）。
// 叶子的深度非奇即偶，这两条规则覆盖了全部叶子；偶数深度的内部孙辈自己也是
// wide 节点的根，所以用前缀和算出的 wideId 互相引用。每个线程只写自己的节点。
// ---------------------------------------------------------------------------
__global__ void wideCountKernel(const int* __restrict__ depth, int internalCount,
                                int* __restrict__ flags) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= internalCount) return;
    flags[i] = (depth[i] & 1) == 0 ? 1 : 0;
}

__device__ __forceinline__ void clearWideEntry(GpuBVH4Entry& e) {
    e.mnx = CUDART_INF_F;
    e.mny = CUDART_INF_F;
    e.mnz = CUDART_INF_F;
    e.mxx = -CUDART_INF_F;
    e.mxy = -CUDART_INF_F;
    e.mxz = -CUDART_INF_F;
    e.child = kWideEmpty;
    e.pad = 0;
}

__device__ __forceinline__ void fillWideEntry(GpuBVH4Entry& e, const GpuBVHNode& box,
                                              int child) {
    e.mnx = box.mnx;
    e.mny = box.mny;
    e.mnz = box.mnz;
    e.mxx = box.mxx;
    e.mxy = box.mxy;
    e.mxz = box.mxz;
    e.child = child;
    e.pad = 0;
}

__device__ __forceinline__ void fillWideNode(const GpuBVHNode* __restrict__ nodes, int n,
                                             const int* __restrict__ wideId, int i,
                                             GpuBVH4Node& out) {
#pragma unroll
    for (int s = 0; s < 4; ++s) clearWideEntry(out.c[s]);

    const GpuBVHNode nd = nodes[i];
    const int kids[2] = {nd.left, nd.right};
    int slot = 0;
    for (int k = 0; k < 2; ++k) {
        const int c = kids[k];
        if (c >= n - 1) {
            fillWideEntry(out.c[slot++], nodes[c], nodes[c].left);
            continue;
        }
        const GpuBVHNode mid = nodes[c];
        const int gkids[2] = {mid.left, mid.right};
        for (int g = 0; g < 2; ++g) {
            const int gc = gkids[g];
            if (gc >= n - 1) {
                fillWideEntry(out.c[slot++], nodes[gc], nodes[gc].left);
            } else {
                fillWideEntry(out.c[slot++], nodes[gc], wideId[gc]);
            }
        }
    }
}

__global__ void wideEmitKernel(const GpuBVHNode* __restrict__ nodes, int n,
                               const int* __restrict__ depth, const int* __restrict__ wideId,
                               GpuBVH4Node* __restrict__ wide) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= n - 1) return;
    if ((depth[i] & 1) != 0) return;

    GpuBVH4Node out;
    fillWideNode(nodes, n, wideId, i, out);
    wide[wideId[i]] = out;
}

// refit 之后：一次只重发本层被合并过的祖先（dirtyOrder 里只可能是内部节点），
// 按层交错执行时更深层的盒子已经定稿，所以宽节点读到的是最终值。
// 计数留在显存里，kernel 自己读，省掉每层一次阻塞主机的 D2H。
__global__ void wideEmitDirtyKernel(const GpuBVHNode* __restrict__ nodes, int n,
                                    const int* __restrict__ depth,
                                    const int* __restrict__ wideId,
                                    const int* __restrict__ dirtyOrder,
                                    const int* __restrict__ count, GpuBVH4Node* __restrict__ wide) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= *count) return;
    const int id = dirtyOrder[i];
    if ((depth[id] & 1) != 0) return;

    GpuBVH4Node out;
    fillWideNode(nodes, n, wideId, id, out);
    wide[wideId[id]] = out;
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

template <typename Ops>
GpuBVH buildLBVH(const typename Ops::Item* hostItems, int n, GpuBuildTimings* timings) {
    using Item = typename Ops::Item;
    if (n < 2) throw std::runtime_error("buildLBVH: at least 2 primitives required");

    const int nodeCount = 2 * n - 1;
    const int internalCount = n - 1;

    GpuBVH bvh;
    bvh.primCount = n;
    bvh.nodeCount = nodeCount;
    cudaOk(cudaMalloc(&bvh.d_nodes, static_cast<size_t>(nodeCount) * sizeof(GpuBVHNode)), "nodes");
    cudaOk(cudaMalloc(&bvh.d_tris, static_cast<size_t>(n) * sizeof(Item)), "tris");
    cudaOk(cudaMalloc(&bvh.d_primIds, static_cast<size_t>(n) * sizeof(int)), "primIds");

    auto* nodes = static_cast<GpuBVHNode*>(bvh.d_nodes);
    auto* sortedTris = static_cast<Item*>(bvh.d_tris);
    auto* primIds = static_cast<int*>(bvh.d_primIds);

    try {
        DevBuf<Item> dSrc(n);
        cudaOk(cudaMemcpy(dSrc.p, hostItems, static_cast<size_t>(n) * sizeof(Item),
                          cudaMemcpyHostToDevice),
               "H2D prims");

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
        centroidBoundsKernel<Ops><<<blocksFor(n), kBlock>>>(dSrc.p, n);
        mortonKernel<Ops><<<blocksFor(n), kBlock>>>(dSrc.p, n, dKeys.p);
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

        scatterTrisKernel<Ops><<<blocksFor(n), kBlock>>>(dSrc.p, dOrderSorted.p, n, sortedTris,
                                                         primIds);
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
        if (bvh.depth >= kDepthBins) throw std::runtime_error("buildLBVH: depth exceeds bin table");
        clock.mark();  // 4

        leafInitKernel<Ops><<<blocksFor(n), kBlock>>>(nodes, sortedTris, n);
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
            throw std::runtime_error("buildLBVH: depth bucketing lost nodes");
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

        // 只合并脏叶子的祖先要用：parent / depth 建树时已经算过，留一份免得重算。
        const size_t ints = static_cast<size_t>(nodeCount) * sizeof(int);
        cudaOk(cudaMalloc(&bvh.d_parent, ints), "parent keep");
        cudaOk(cudaMemcpy(bvh.d_parent, dParent.p, ints, cudaMemcpyDeviceToDevice), "parent D2D");
        cudaOk(cudaMalloc(&bvh.d_depth, ints), "depth keep");
        cudaOk(cudaMemcpy(bvh.d_depth, dDepth.p, ints, cudaMemcpyDeviceToDevice), "depth D2D");
        cudaOk(cudaMalloc(&bvh.d_mark, ints), "mark");
        cudaOk(cudaMemset(bvh.d_mark, 0, ints), "mark clear");
        cudaOk(cudaMalloc(&bvh.d_dirtyOrder,
                          static_cast<size_t>(internalCount) * sizeof(int)),
               "dirty order");
        cudaOk(cudaMalloc(&bvh.d_refitOffsets, static_cast<size_t>(bins + 1) * sizeof(int)),
               "refit offsets");
        cudaOk(cudaMemcpy(bvh.d_refitOffsets, offsets, static_cast<size_t>(bins + 1) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "refit offsets H2D");
        cudaOk(cudaMalloc(&bvh.d_levelCounts, static_cast<size_t>(bins) * sizeof(int)),
               "level counts");

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

GpuBVH buildBVHGPU(const Scene& scene, GpuBuildTimings* timings) {
    return buildLBVH<TriOps>(scene.tris.data(), static_cast<int>(scene.tris.size()), timings);
}

GpuBVH buildPointBVHGPU(const PointCloud& cloud, GpuBuildTimings* timings) {
    return buildLBVH<PointOps>(cloud.pts.data(), static_cast<int>(cloud.pts.size()), timings);
}

GpuBVH buildGaussianBVHGPU(const GaussianCloud& cloud, GpuBuildTimings* timings) {
    return buildLBVH<GaussianOps>(cloud.gaussians.data(), static_cast<int>(cloud.gaussians.size()),
                                  timings);
}

double buildWideNodesGPU(GpuBVH& bvh) {
    if (bvh.nodeCount <= 0 || bvh.d_depth == nullptr) {
        throw std::runtime_error("buildWideNodesGPU: tree has no depth table");
    }
    if (bvh.d_wideNodes) {
        cudaFree(bvh.d_wideNodes);
        bvh.d_wideNodes = nullptr;
    }
    if (bvh.d_wideId) {
        cudaFree(bvh.d_wideId);
        bvh.d_wideId = nullptr;
    }
    bvh.wideCount = 0;

    const int n = bvh.primCount;
    const int internal = n - 1;
    const int nodeCount = bvh.nodeCount;
    const int* depth = static_cast<const int*>(bvh.d_depth);
    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);

    DevBuf<int> dFlags(internal);
    DevBuf<int> dWideId(nodeCount);
    wideCountKernel<<<blocksFor(internal), kBlock>>>(depth, internal, dFlags.p);
    cudaOk(cudaGetLastError(), "wide flags");

    size_t scanBytes = 0;
    cudaOk(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, dFlags.p, dWideId.p, internal),
           "wide scan(probe)");
    DevBuf<unsigned char> dScanTemp(scanBytes > 0 ? scanBytes : 1);
    cudaOk(cub::DeviceScan::ExclusiveSum(dScanTemp.p, scanBytes, dFlags.p, dWideId.p, internal),
           "wide scan");

    // 前缀和只到 internal-1，总数要单独加最后一项的 flag。
    int tail[2] = {0, 0};
    cudaOk(cudaMemcpy(&tail[0], dWideId.p + (internal - 1), sizeof(int), cudaMemcpyDeviceToHost),
           "D2H prefix");
    cudaOk(cudaMemcpy(&tail[1], dFlags.p + (internal - 1), sizeof(int), cudaMemcpyDeviceToHost),
           "D2H flag");
    const int wideCount = tail[0] + tail[1];
    if (wideCount <= 0) throw std::runtime_error("buildWideNodesGPU: no wide nodes");

    cudaOk(cudaMalloc(&bvh.d_wideNodes, static_cast<size_t>(wideCount) * sizeof(GpuBVH4Node)),
           "wide nodes");
    bvh.wideCount = wideCount;
    bvh.d_wideId = dWideId.release();

    StageClock clock;
    clock.mark();  // 0
    wideEmitKernel<<<blocksFor(internal), kBlock>>>(nodes, n, depth,
                                                    static_cast<const int*>(bvh.d_wideId),
                                                    static_cast<GpuBVH4Node*>(bvh.d_wideNodes));
    cudaOk(cudaGetLastError(), "wide emit");
    clock.mark();  // 1
    cudaOk(cudaEventSynchronize(clock.ev[clock.marks - 1]), "sync");
    return clock.ms(0, 1);
}

void destroyBVHGPU(GpuBVH& bvh) {
    if (bvh.d_nodes) cudaFree(bvh.d_nodes);
    if (bvh.d_tris) cudaFree(bvh.d_tris);
    if (bvh.d_primIds) cudaFree(bvh.d_primIds);
    if (bvh.d_refitOrder) cudaFree(bvh.d_refitOrder);
    if (bvh.d_stage) cudaFree(bvh.d_stage);
    if (bvh.d_dirtySlots) cudaFree(bvh.d_dirtySlots);
    if (bvh.d_dirtyTris) cudaFree(bvh.d_dirtyTris);
    if (bvh.d_parent) cudaFree(bvh.d_parent);
    if (bvh.d_depth) cudaFree(bvh.d_depth);
    if (bvh.d_mark) cudaFree(bvh.d_mark);
    if (bvh.d_dirtyOrder) cudaFree(bvh.d_dirtyOrder);
    if (bvh.d_refitOffsets) cudaFree(bvh.d_refitOffsets);
    if (bvh.d_levelCounts) cudaFree(bvh.d_levelCounts);
    if (bvh.d_wideNodes) cudaFree(bvh.d_wideNodes);
    if (bvh.d_wideId) cudaFree(bvh.d_wideId);
    bvh = GpuBVH{};
}

// 只上传改动的图元：主机侧已经按 dirty 顺序打包好，slot 是它的叶子槽。
// 写入 + 重算叶子盒在同一个线程里做，省一次 kernel。
template <typename Ops>
__global__ void scatterLeafKernel(const int* __restrict__ slots,
                                  const typename Ops::Item* __restrict__ src, int count,
                                  typename Ops::Item* __restrict__ sortedTris,
                                  GpuBVHNode* __restrict__ nodes, int n) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count) return;
    const int slot = slots[i];
    const auto item = src[i];
    sortedTris[slot] = item;
    GpuBVHNode& leaf = nodes[n - 1 + slot];
    setNodeBox(leaf, Ops::bounds(item));
}

// 脏叶子 → 祖先链：从父节点一路走到根。epoch 标记顺带做去重——同一棵子树被多个
// 脏叶子共享时只合并一次；写入位置复用缓存层序的同一段区间，每个节点最多写一次，
// 所以每层不会越界到下一层。
__global__ void markAncestorsKernel(const int* __restrict__ slots, int count,
                                    const int* __restrict__ parent,
                                    const int* __restrict__ depth,
                                    int* __restrict__ mark, int epoch,
                                    int* __restrict__ levelCounts,
                                    const int* __restrict__ levelOffsets,
                                    int* __restrict__ dirtyOrder, int n) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count) return;
    int node = parent[n - 1 + slots[i]];
    while (node >= 0) {
        if (atomicExch(&mark[node], epoch) == epoch) break;
        const int d = depth[node];
        dirtyOrder[levelOffsets[d] + atomicAdd(&levelCounts[d], 1)] = node;
        node = parent[node];
    }
}

template <typename Ops>
double refitImpl(GpuBVH& bvh, const std::vector<typename Ops::Item>& items,
                 const std::vector<int>* dirty, GpuRefitTimings* timings) {
    using Item = typename Ops::Item;
    const int n = bvh.primCount;
    if (n < 2 || bvh.d_nodes == nullptr) throw std::runtime_error("refitBVHGPU: BVH not built");
    if (bvh.d_refitOrder == nullptr) throw std::runtime_error("refitBVHGPU: missing refit order");
    if (dirty && bvh.d_parent == nullptr) {
        throw std::runtime_error("refitBVHGPU: missing parent links");
    }
    if (static_cast<int>(items.size()) != n) {
        throw std::runtime_error("refitBVHGPU: primitive count changed, rebuild instead");
    }

    auto* nodes = static_cast<GpuBVHNode*>(bvh.d_nodes);
    auto* sortedTris = static_cast<Item*>(bvh.d_tris);
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

    const int bins = static_cast<int>(bvh.refitOffsets.size()) - 1;
    int dirtyCount = 0;

    if (dirty) {
        dirtyCount = static_cast<int>(dirty->size());
        // 主机侧 gather：改动量通常远小于 n，拿到叶子序的紧凑数组再传一次。
        std::vector<int> slots(static_cast<size_t>(dirtyCount));
        std::vector<Item> moved(static_cast<size_t>(dirtyCount));
        for (int i = 0; i < dirtyCount; ++i) {
            const int id = (*dirty)[static_cast<size_t>(i)];
            if (id < 0 || id >= n) throw std::runtime_error("refitBVHGPU: dirty index out of range");
            slots[static_cast<size_t>(i)] = bvh.primSlot[static_cast<size_t>(id)];
            moved[static_cast<size_t>(i)] = items[static_cast<size_t>(id)];
        }

        // 暂存区按需增长：每次 cudaMalloc 的固定开销比内容本身还贵。
        if (bvh.dirtyCapacity < dirtyCount) {
            if (bvh.d_dirtySlots) cudaFree(bvh.d_dirtySlots);
            if (bvh.d_dirtyTris) cudaFree(bvh.d_dirtyTris);
            cudaOk(cudaMalloc(&bvh.d_dirtySlots, static_cast<size_t>(dirtyCount) * sizeof(int)),
                   "dirty slots");
            cudaOk(cudaMalloc(&bvh.d_dirtyTris, static_cast<size_t>(dirtyCount) * sizeof(Item)),
                   "dirty prims");
            bvh.dirtyCapacity = dirtyCount;
        }
        auto* dSlots = static_cast<int*>(bvh.d_dirtySlots);
        auto* dMoved = static_cast<Item*>(bvh.d_dirtyTris);
        if (dirtyCount > 0) {
            cudaOk(cudaMemcpy(dSlots, slots.data(), static_cast<size_t>(dirtyCount) * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "refit H2D slots");
            cudaOk(cudaMemcpy(dMoved, moved.data(), static_cast<size_t>(dirtyCount) * sizeof(Item),
                              cudaMemcpyHostToDevice),
                   "refit H2D prims");
        }
        clock.mark();  // 1

        if (dirtyCount > 0) {
            scatterLeafKernel<Ops><<<blocksFor(dirtyCount), kBlock>>>(dSlots, dMoved, dirtyCount,
                                                                      sortedTris, nodes, n);
        }
        cudaOk(cudaGetLastError(), "refit scatter");
        clock.mark();  // 2

        // 沿 parent 链标出要合并的祖先，按层计数留在显存：合并 kernel 自己读，
        // 每层省掉一次会阻塞主机的 D2H。
        if (dirtyCount > 0) {
            cudaOk(cudaMemset(bvh.d_levelCounts, 0, static_cast<size_t>(bins) * sizeof(int)),
                   "level counts");
            ++bvh.refitEpoch;
            markAncestorsKernel<<<blocksFor(dirtyCount), kBlock>>>(
                dSlots, dirtyCount, static_cast<const int*>(bvh.d_parent),
                static_cast<const int*>(bvh.d_depth), static_cast<int*>(bvh.d_mark),
                bvh.refitEpoch, static_cast<int*>(bvh.d_levelCounts),
                static_cast<const int*>(bvh.d_refitOffsets), static_cast<int*>(bvh.d_dirtyOrder),
                n);
            cudaOk(cudaGetLastError(), "refit mark");
        }
        clock.mark();  // 3
    } else {
        // 暂存区首次用到才分配：不 refit 的调用方不付这块显存。
        if (bvh.d_stage == nullptr || bvh.stageCount < n) {
            if (bvh.d_stage) cudaFree(bvh.d_stage);
            cudaOk(cudaMalloc(&bvh.d_stage, static_cast<size_t>(n) * sizeof(Item)), "refit stage");
            bvh.stageCount = n;
        }
        auto* dSrc = static_cast<Item*>(bvh.d_stage);
        cudaOk(cudaMemcpy(dSrc, items.data(), static_cast<size_t>(n) * sizeof(Item),
                          cudaMemcpyHostToDevice),
               "refit H2D");
        clock.mark();  // 1

        gatherTrisKernel<Ops><<<blocksFor(n), kBlock>>>(dSrc, primIds, n, sortedTris);
        leafInitKernel<Ops><<<blocksFor(n), kBlock>>>(nodes, sortedTris, n);
        cudaOk(cudaGetLastError(), "refit scatter");
        clock.mark();  // 2
    }

    // 叶子盒已经定稿，内部节点按缓存的层序从深到浅合并：第 d 层只读 d+1 层。
    // 增量路径只合并脏叶子的祖先；区间与全量层序共用，所以不会串层。
    // 有宽节点层时，合并完还要把受影响的宽节点重发一遍（拓扑与 wideId 都没变，
    // 只是盒子拷贝变旧了）：全量路径整体重发，增量路径只重发脏祖先里偶数深度的。
    double total = 0.0;
    const bool wideLayer = bvh.d_wideNodes != nullptr && bvh.d_wideId != nullptr;
    auto* wideNodes = static_cast<GpuBVH4Node*>(bvh.d_wideNodes);
    const auto* wideId = static_cast<const int*>(bvh.d_wideId);
    const auto* depths = static_cast<const int*>(bvh.d_depth);
    if (dirty) {
        auto* dirtyOrder = static_cast<int*>(bvh.d_dirtyOrder);
        auto* levelCounts = static_cast<int*>(bvh.d_levelCounts);
        for (int d = bins - 1; d >= 0; --d) {
            const int levelSize = bvh.refitOffsets[d + 1] - bvh.refitOffsets[d];
            // 每个脏叶子在每层最多贡献一个祖先，所以脏叶子数就是每层的容量上界。
            const int capacity = std::min(levelSize, dirtyCount);
            if (capacity == 0) continue;
            const int blocks = (capacity + kBlock - 1) / kBlock;
            refitAncestorKernel<<<blocks, kBlock>>>(nodes, dirtyOrder + bvh.refitOffsets[d],
                                                    levelCounts + d, capacity);
        }
        cudaOk(cudaGetLastError(), "refit tree");
        clock.mark();  // 4

        if (wideLayer && dirtyCount > 0) {
            // 按层重发要 launch 十几次小 kernel，小树上启动开销反而更高：
            // 实测 131k 按层 0.23–0.37 ms、整层 0.17 ms；2M 按层 0.38、整层 1.5 ms。
            // 所以小树或大部分叶子都脏时整层重发，其余按层。阈值取实测中点
            // （262144 个宽节点 ≈ 52 万图元）。
            const bool fullEmit = bvh.wideCount <= 262144 ||
                                  static_cast<long long>(dirtyCount) * 8 >= n;
            if (fullEmit) {
                wideEmitKernel<<<blocksFor(n - 1), kBlock>>>(nodes, n, depths, wideId, wideNodes);
            } else {
                for (int d = bins - 1; d >= 0; --d) {
                    if ((d & 1) != 0) continue;
                    const int levelSize = bvh.refitOffsets[d + 1] - bvh.refitOffsets[d];
                    const int capacity = std::min(levelSize, dirtyCount);
                    if (capacity == 0) continue;
                    wideEmitDirtyKernel<<<blocksFor(capacity), kBlock>>>(
                        nodes, n, depths, wideId, dirtyOrder + bvh.refitOffsets[d],
                        levelCounts + d, wideNodes);
                }
            }
            cudaOk(cudaGetLastError(), "wide refit");
            clock.mark();  // 5
        }

        const int last = clock.marks - 1;
        cudaOk(cudaEventSynchronize(clock.ev[last]), "refit sync");
        total = clock.ms(0, last);
        if (timings) {
            timings->markMs = clock.ms(2, 3);
            timings->mergeMs = clock.ms(3, 4);
            timings->wideMs = (last > 4) ? clock.ms(4, last) : 0.0;
            timings->treeMs = clock.ms(2, 4);
            timings->updated = dirtyCount;
        }
    } else {
        for (int d = bins - 1; d >= 0; --d) {
            const int count = bvh.refitOffsets[d + 1] - bvh.refitOffsets[d];
            if (count == 0) continue;
            const int blocks = (count + kBlock * kRefitUnroll - 1) / (kBlock * kRefitUnroll);
            refitKernel<<<blocks, kBlock>>>(nodes, order + bvh.refitOffsets[d], count);
        }
        cudaOk(cudaGetLastError(), "refit tree");
        clock.mark();  // 3

        if (wideLayer) {
            wideEmitKernel<<<blocksFor(n - 1), kBlock>>>(nodes, n, depths, wideId, wideNodes);
            cudaOk(cudaGetLastError(), "wide refit");
            clock.mark();  // 4
        }

        const int last = clock.marks - 1;
        cudaOk(cudaEventSynchronize(clock.ev[last]), "refit sync");
        total = clock.ms(0, last);
        if (timings) {
            timings->mergeMs = clock.ms(2, 3);
            timings->wideMs = (last > 3) ? clock.ms(3, last) : 0.0;
            timings->treeMs = timings->mergeMs;
            timings->updated = n;
        }
    }

    if (timings) {
        timings->uploadMs = clock.ms(0, 1);
        timings->gatherMs = clock.ms(1, 2);
        timings->totalMs = total;
    }
    return total;
}

double refitBVHGPU(GpuBVH& bvh, const Scene& scene, GpuRefitTimings* timings) {
    return refitImpl<TriOps>(bvh, scene.tris, nullptr, timings);
}

double refitBVHGPU(GpuBVH& bvh, const Scene& scene, const std::vector<int>& dirty,
                   GpuRefitTimings* timings) {
    return refitImpl<TriOps>(bvh, scene.tris, &dirty, timings);
}

double refitPointBVHGPU(GpuBVH& bvh, const PointCloud& cloud, GpuRefitTimings* timings) {
    return refitImpl<PointOps>(bvh, cloud.pts, nullptr, timings);
}

double refitPointBVHGPU(GpuBVH& bvh, const PointCloud& cloud, const std::vector<int>& dirty,
                        GpuRefitTimings* timings) {
    return refitImpl<PointOps>(bvh, cloud.pts, &dirty, timings);
}

double refitGaussianBVHGPU(GpuBVH& bvh, const GaussianCloud& cloud,
                           GpuRefitTimings* timings) {
    return refitImpl<GaussianOps>(bvh, cloud.gaussians, nullptr, timings);
}

double refitGaussianBVHGPU(GpuBVH& bvh, const GaussianCloud& cloud,
                           const std::vector<int>& dirty, GpuRefitTimings* timings) {
    return refitImpl<GaussianOps>(bvh, cloud.gaussians, &dirty, timings);
}

template <typename Ops>
std::string validateImpl(const GpuBVH& bvh, const std::vector<typename Ops::Item>& items,
                         const AABB& bounds) {
    const int n = bvh.primCount;
    const int nodeCount = bvh.nodeCount;
    if (nodeCount != 2 * n - 1) return "nodeCount != 2n-1";
    if (static_cast<int>(items.size()) != n) return "prim count mismatch";

    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);
    const auto* sortedTris = static_cast<const typename Ops::Item*>(bvh.d_tris);

    DevBuf<int> dVisited(nodeCount);
    DevBuf<int> dErrors(4);
    cudaOk(cudaMemset(dVisited.p, 0, static_cast<size_t>(nodeCount) * sizeof(int)), "visited");
    cudaOk(cudaMemset(dErrors.p, 0, 4 * sizeof(int)), "errors");

    checkNodesKernel<Ops><<<blocksFor(nodeCount), kBlock>>>(nodes, sortedTris, n, nodeCount,
                                                            dVisited.p, dErrors.p);
    checkRefsKernel<<<blocksFor(nodeCount), kBlock>>>(dVisited.p, nodeCount, dErrors.p);

    int errors[4] = {};
    cudaOk(cudaMemcpy(errors, dErrors.p, sizeof(errors), cudaMemcpyDeviceToHost), "D2H errors");

    GpuBVHNode root{};
    cudaOk(cudaMemcpy(&root, nodes, sizeof(GpuBVHNode), cudaMemcpyDeviceToHost), "root");
    const AABB rootBox = nodeBox(root);
    const bool badRoot = rootBox.mn.x != bounds.mn.x || rootBox.mn.y != bounds.mn.y ||
                         rootBox.mn.z != bounds.mn.z || rootBox.mx.x != bounds.mx.x ||
                         rootBox.mx.y != bounds.mx.y || rootBox.mx.z != bounds.mx.z;

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

std::string validateBVHGPU(const GpuBVH& bvh, const Scene& scene) {
    return validateImpl<TriOps>(bvh, scene.tris, scene.bounds);
}

std::string validatePointBVHGPU(const GpuBVH& bvh, const PointCloud& cloud) {
    return validateImpl<PointOps>(bvh, cloud.pts, cloud.bounds);
}

std::string validateGaussianBVHGPU(const GpuBVH& bvh, const GaussianCloud& cloud) {
    return validateImpl<GaussianOps>(bvh, cloud.gaussians, cloud.bounds);
}
