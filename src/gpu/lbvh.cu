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
#include <utility>

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

// 拓扑更新期间的设备内存池。更新每轮要几百 MB 临时块，直接 cudaMalloc/cudaFree
// 在 2M 上一轮就是 15 ms 量级（free 尤其贵）；这里把上一轮归还的块按容量缓存，
// 下一轮 best-fit 复用。只在 updateTopology 里激活，建树 / 查询不受影响。
thread_local GpuBVH* gTopoPool = nullptr;

constexpr size_t kTopoPoolCapBytes = size_t(2) << 30;

void* topoPoolAlloc(GpuBVH* bvh, size_t need, size_t* got) {
    auto& freeBlocks = bvh->topoFree;
    size_t best = SIZE_MAX;
    size_t bestIdx = freeBlocks.size();
    for (size_t i = 0; i < freeBlocks.size(); ++i) {
        if (freeBlocks[i].bytes >= need && freeBlocks[i].bytes < best) {
            best = freeBlocks[i].bytes;
            bestIdx = i;
        }
    }
    if (bestIdx < freeBlocks.size()) {
        void* p = freeBlocks[bestIdx].p;
        const size_t bytes = freeBlocks[bestIdx].bytes;
        freeBlocks[bestIdx] = freeBlocks.back();
        freeBlocks.pop_back();
        bvh->topoFreeBytes -= bytes;
        *got = bytes;
        return p;
    }
    void* p = nullptr;
    cudaOk(cudaMalloc(&p, need), "cudaMalloc");
    *got = need;
    return p;
}

void topoPoolRelease(GpuBVH* bvh, void* p, size_t bytes) {
    if (p == nullptr || bytes == 0) return;
    auto& freeBlocks = bvh->topoFree;
    freeBlocks.push_back({p, bytes});
    bvh->topoFreeBytes += bytes;
    // 容量只涨不降会在大 n 来回振荡时堆住多余的块；超上限先丢最小的。
    while (bvh->topoFreeBytes > kTopoPoolCapBytes && !freeBlocks.empty()) {
        size_t bi = 0;
        for (size_t i = 1; i < freeBlocks.size(); ++i) {
            if (freeBlocks[i].bytes < freeBlocks[bi].bytes) bi = i;
        }
        cudaOk(cudaFree(freeBlocks[bi].p), "pool evict");
        bvh->topoFreeBytes -= freeBlocks[bi].bytes;
        freeBlocks[bi] = freeBlocks.back();
        freeBlocks.pop_back();
    }
}

// 换树时旧数组不 cudaFree，而是放回池里给下一轮用。池不活跃（回退重建时的
// destroyBVHGPU）就走原来的 cudaFree。
void topoReleasePtr(GpuBVH& bvh, void* p, size_t bytes) {
    if (p == nullptr) return;
    // 尺寸未知（0）的块进池也没法再匹配，直接释放，避免泄漏。
    if (gTopoPool == &bvh && bytes > 0) {
        topoPoolRelease(&bvh, p, bytes);
    } else {
        cudaOk(cudaFree(p), "topo release");
    }
}

struct TopoPoolScope {
    GpuBVH* prev;
    explicit TopoPoolScope(GpuBVH* bvh) : prev(gTopoPool) { gTopoPool = bvh; }
    ~TopoPoolScope() { gTopoPool = prev; }
};

// 建树的持久数组（节点 / 图元 / parent / depth ...）：池活跃时先问池——回退重建
// 要的正是上一棵树还回来的同尺寸块——否则 cudaMalloc。池块在分配时已从池里摘掉，
// 之后 destroyBVHGPU 照常 cudaFree，不会重复释放。
void* topoAwareMalloc(size_t bytes) {
    if (!gTopoPool) {
        void* p = nullptr;
        cudaOk(cudaMalloc(&p, bytes), "cudaMalloc");
        return p;
    }
    size_t got = 0;
    return topoPoolAlloc(gTopoPool, bytes, &got);
}

template <typename T>
struct DevBuf {
    T* p = nullptr;
    size_t bytes = 0;
    GpuBVH* pool = nullptr;  // 非空表示归还到该 BVH 的池
    DevBuf() = default;
    explicit DevBuf(size_t count) {
        const size_t need = count * sizeof(T);
        if (need == 0) return;
        pool = gTopoPool;
        if (pool) {
            p = static_cast<T*>(topoPoolAlloc(pool, need, &bytes));
        } else {
            cudaOk(cudaMalloc(reinterpret_cast<void**>(&p), need), "cudaMalloc");
            bytes = need;
        }
    }
    ~DevBuf() {
        if (p == nullptr) return;
        if (pool) {
            topoPoolRelease(pool, p, bytes);
        } else {
            cudaFree(p);
        }
    }
    T* release() {
        T* q = p;
        p = nullptr;
        bytes = 0;
        pool = nullptr;
        return q;
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p(o.p), bytes(o.bytes), pool(o.pool) {
        o.p = nullptr;
        o.bytes = 0;
        o.pool = nullptr;
    }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) {
            if (p) {
                if (pool) topoPoolRelease(pool, p, bytes);
                else cudaFree(p);
            }
            p = o.p;
            bytes = o.bytes;
            pool = o.pool;
            o.p = nullptr;
            o.bytes = 0;
            o.pool = nullptr;
        }
        return *this;
    }
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

// SAH 每层要重复记时，事件数固定、不随层数增长。
struct QuadClock {
    cudaEvent_t ev[4] = {};

    QuadClock() {
        for (cudaEvent_t& e : ev) cudaOk(cudaEventCreate(&e), "cudaEventCreate");
    }
    ~QuadClock() {
        for (cudaEvent_t& e : ev) cudaEventDestroy(e);
    }
    void mark(int i) { cudaOk(cudaEventRecord(ev[i]), "cudaEventRecord"); }
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

// 与 depthOffsetsKernel 相同，只是键是 (深度 << 26) | 来源序 的打包形式；全 1 哨兵
// （深度 63）排在所有有效键之后，不会被算进任何一层。
__global__ void depthOffsetsPackedKernel(const uint32_t* __restrict__ sortedKeys, int nodeCount,
                                         int bins, int* __restrict__ offsets) {
    const int d = threadIdx.x;
    if (d > bins) return;
    int lo = 0;
    int hi = nodeCount;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (static_cast<int>(sortedKeys[mid] >> 26) < d) {
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

// 拓扑更新 E 阶段：新树的 mark 表是刚分配的，直接按 epoch 写就相当于清空。
// 临时森林的全部内部节点都要合并（建森林只定拓扑、不算盒子）；切分根和移动
// 叶子的祖先盒子变了，沿 parent 链标脏——其余旧内部节点原样沿用旧盒。
__global__ void markTempNodesKernel(const int* __restrict__ idMap, int base, int count,
                                    int* __restrict__ mark, int epoch) {
    const int t = blockIdx.x * kBlock + threadIdx.x;
    if (t < count) mark[idMap[base + t]] = epoch;
}

// starts 是"第一个要合并其祖先的节点"（切分根传森林根，从它的父节点开始爬）。
__global__ void markNodeAncestorsKernel(const int* __restrict__ starts, int count,
                                        const int* __restrict__ parent,
                                        int* __restrict__ mark, int epoch) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count) return;
    int node = parent[starts[i]];
    while (node >= 0) {
        if (atomicExch(&mark[node], epoch) == epoch) break;
        node = parent[node];
    }
}

// 移动叶子：先经 sceneSlot（新场景下标 -> 新叶子槽）找到叶子，再从父节点往上爬。
__global__ void markMovedAncestorsKernel(const int* __restrict__ sceneIds, int count,
                                         const int* __restrict__ sceneSlot,
                                         const int* __restrict__ parent, int nNew,
                                         int* __restrict__ mark, int epoch) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count) return;
    int node = parent[nNew - 1 + sceneSlot[sceneIds[i]]];
    while (node >= 0) {
        if (atomicExch(&mark[node], epoch) == epoch) break;
        node = parent[node];
    }
}

// 只在层区间 [begin, begin+count) 里合并标脏的节点。层内没有父子关系，
// 自深向浅逐层发射即可；干净节点连子盒都不读。
__global__ void refitDirtyKernel(GpuBVHNode* __restrict__ nodes, int begin, int count,
                                 const int* __restrict__ mark, int epoch) {
    const int base = blockIdx.x * (kBlock * kRefitUnroll) + threadIdx.x;

    int ids[kRefitUnroll];
    GpuBVHNode l[kRefitUnroll], r[kRefitUnroll];
    bool ok[kRefitUnroll];

#pragma unroll
    for (int u = 0; u < kRefitUnroll; ++u) {
        const int t = base + u * kBlock;
        ok[u] = false;
        if (t < count) {
            const int i = begin + t;
            if (mark[i] == epoch) {
                const GpuBVHNode nd = nodes[i];
                if (nd.left >= 0) {
                    ok[u] = true;
                    ids[u] = i;
                    l[u] = nodes[nd.left];
                    r[u] = nodes[nd.right];
                }
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
        // 叶子必须正好占 n-1+slot 且装一个图元：拓扑变更后的重编号要靠它，
        // 查询/refit 的叶子寻址也都默认这条。
        if (i != n - 1 + slot || nd.right != 1) atomicAdd(&errors[3], 1);
        const AABB mine = nodeBox(nd);
        const AABB prim = Ops::bounds(sortedItems[slot]);
        if (mine.mn.x != prim.mn.x || mine.mn.y != prim.mn.y || mine.mn.z != prim.mn.z ||
            mine.mx.x != prim.mx.x || mine.mx.y != prim.mx.y || mine.mx.z != prim.mx.z) {
            atomicAdd(&errors[1], 1);
        }
    }
}

// 深度必须与父子关系一致：重编号、层序合并和切分选择都靠它，而且查询栈只有 64 层。
__global__ void checkDepthKernel(const GpuBVHNode* __restrict__ nodes,
                                 const int* __restrict__ depth, int n, int nodeCount,
                                 int* __restrict__ errors) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= nodeCount) return;
    if (i == 0) {
        if (depth[i] != 0) atomicAdd(&errors[5], 1);
    }
    if (i < n - 1) {
        const int l = nodes[i].left;
        const int r = nodes[i].right;
        if (l >= 0 && l < nodeCount && depth[l] != depth[i] + 1) atomicAdd(&errors[5], 1);
        if (r >= 0 && r < nodeCount && depth[r] != depth[i] + 1) atomicAdd(&errors[5], 1);
    }
    if (depth[i] >= 63) atomicAdd(&errors[5], 1);
}

__global__ void checkRefsKernel(const int* __restrict__ visited, int nodeCount,
                                int* __restrict__ errors) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= nodeCount) return;
    const int want = (i == 0) ? 0 : 1;
    if (visited[i] != want) atomicAdd(&errors[2], 1);
}

// 叶子槽 -> 场景索引必须是一个排列：拓扑更新后这条比任何盒子检查都先崩。
__global__ void checkPermKernel(const int* __restrict__ primIds, int n,
                                int* __restrict__ hist, int* __restrict__ errors) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= n) return;
    const int id = primIds[i];
    if (id < 0 || id >= n) {
        atomicAdd(&errors[4], 1);
        return;
    }
    atomicAdd(&hist[id], 1);
}

__global__ void checkPermCountKernel(const int* __restrict__ hist, int n,
                                     int* __restrict__ errors) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= n) return;
    if (hist[i] != 1) atomicAdd(&errors[4], 1);
}

// 叶子载荷必须就是 primIds 指向的那个图元：d_tris 与 primIds 是同一个置换的两半，
// 两个都错但互相自洽时上面几项都查不出来（拓扑更新最容易在这里漂）。
template <typename Ops>
__global__ void checkPayloadKernel(const typename Ops::Item* __restrict__ sorted,
                                   const typename Ops::Item* __restrict__ items,
                                   const int* __restrict__ primIds, int n,
                                   int* __restrict__ errors) {
    const int s = blockIdx.x * kBlock + threadIdx.x;
    if (s >= n) return;
    const int id = primIds[s];
    if (id < 0 || id >= n) {
        atomicAdd(&errors[4], 1);
        return;
    }
    const typename Ops::Item a = sorted[s];
    const typename Ops::Item b = items[id];
    const unsigned char* pa = reinterpret_cast<const unsigned char*>(&a);
    const unsigned char* pb = reinterpret_cast<const unsigned char*>(&b);
    int diff = 0;
    for (size_t k = 0; k < sizeof(typename Ops::Item); ++k) diff |= pa[k] ^ pb[k];
    if (diff != 0) atomicAdd(&errors[4], 1);
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

// ---------------------------------------------------------------------------
// 自顶向下分箱 SAH（Wald 2007 的口径）：每层对还没定叶子的区间，在 3 根轴上
// 各分 16 个箱，选 area(左)*n左 + area(右)*n右 最小的切分；稳定分区后进入下一层。
// 节点编号与 LBVH 完全一致（内部 0..n-2、叶子 n-1..），所以查询、refit、宽节点
// 都不用改。区间小到 64 个图元就退回对象中位数，并在一个 kernel 里串行建完，
// 省掉深层那些"扫描整棵树也只切几个点"的层。
// ---------------------------------------------------------------------------
constexpr int kSahBins = 16;
constexpr int kSahMedian = 64;     // 区间 <= 这个数就对象中位数 + 串行建
constexpr int kSahChunk = 4096;    // 分箱按块并行，一块最多这么多槽位
constexpr int kSahMinSideDiv = 8;  // 切分两侧都至少占 1/8，给树深兜底
constexpr int kSahBinStride = 7;   // [count, lo0..2, hi0..2]
constexpr int kSahBinWords = kSahBins * 3 * kSahBinStride;

__device__ __forceinline__ float sahArea(const float* lo, const float* hi) {
    const float dx = hi[0] - lo[0];
    const float dy = hi[1] - lo[1];
    const float dz = hi[2] - lo[2];
    return dx * dy + dy * dz + dz * dx;  // 2×表面积，常数因子不影响比较
}

__device__ __forceinline__ int sahBinOf(float c, float lo, float hi) {
    const float ext = hi - lo;
    if (!(ext > 0.0f)) return 0;
    int b = __float2int_rd((c - lo) * (static_cast<float>(kSahBins) / ext));
    if (b < 0) b = 0;
    if (b >= kSahBins) b = kSahBins - 1;
    return b;
}

// 切分信息打包：mode(1b) | axis(2b) | 右半起始箱(4b)。mode=0 是对象中位数。
__device__ __forceinline__ bool sahIsLeft(int info, int slot, int begin, int leftCount,
                                          const float* cen, const float* lo, const float* hi) {
    if (info == 0) return (slot - begin) < leftCount;
    const int axis = (info >> 2) & 3;
    return sahBinOf(cen[axis], lo[axis], hi[axis]) < (info >> 4);
}

__global__ void sahInitKernel(int n, int* __restrict__ order, int* __restrict__ slotTask,
                              int* __restrict__ taskBegin, int* __restrict__ taskEnd,
                              int* __restrict__ taskNode) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < n) {
        order[i] = i;
        slotTask[i] = 0;
    }
    if (i == 0) {
        taskBegin[0] = 0;
        taskEnd[0] = n;
        taskNode[0] = 0;
    }
}

// 给大区间分配紧凑的箱下标和块列表；块的分配顺序取决于 atomic，但块之间只做
// 加法/最值合并，与顺序无关，所以结果仍然可复现。
__global__ void sahPrepKernel(int taskCount, const int* __restrict__ taskBegin,
                              const int* __restrict__ taskEnd, int* __restrict__ binId,
                              int* __restrict__ chunkTask, int* __restrict__ chunkBegin,
                              int* __restrict__ chunkEnd, uint32_t* __restrict__ cenLo,
                              uint32_t* __restrict__ cenHi, int* __restrict__ binCounter,
                              int* __restrict__ chunkCounter) {
    const int t = blockIdx.x * kBlock + threadIdx.x;
    if (t >= taskCount) return;
    binId[t] = -1;
    const int begin = taskBegin[t];
    const int count = taskEnd[t] - begin;
    if (count <= kSahMedian) return;

    binId[t] = atomicAdd(binCounter, 1);
    for (int a = 0; a < 3; ++a) {
        cenLo[t * 3 + a] = floatOrdered(CUDART_INF_F);
        cenHi[t * 3 + a] = floatOrdered(-CUDART_INF_F);
    }

    const int chunks = (count + kSahChunk - 1) / kSahChunk;
    const int base = atomicAdd(chunkCounter, chunks);
    for (int j = 0; j < chunks; ++j) {
        const int cb = begin + j * kSahChunk;
        chunkTask[base + j] = t;
        chunkBegin[base + j] = cb;
        chunkEnd[base + j] = cb + kSahChunk < taskEnd[t] ? cb + kSahChunk : taskEnd[t];
    }
}

__global__ void sahClearBinsKernel(uint32_t* __restrict__ bins, int words) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= words) return;
    const int comp = i % kSahBinStride;
    bins[i] = comp == 0 ? 0u
                        : (comp <= 3 ? floatOrdered(CUDART_INF_F) : floatOrdered(-CUDART_INF_F));
}

// 第一遍：只算重心包围盒。分箱需要它当坐标域，所以要单独一趟。
template <typename Ops>
__global__ void sahCenBoundsKernel(const typename Ops::Item* __restrict__ items,
                                   const int* __restrict__ order,
                                   const int* __restrict__ chunkTask,
                                   const int* __restrict__ chunkBegin,
                                   const int* __restrict__ chunkEnd, int chunkCount,
                                   uint32_t* __restrict__ cenLo, uint32_t* __restrict__ cenHi) {
    const int c = blockIdx.x;
    if (c >= chunkCount) return;
    const int tid = threadIdx.x;
    __shared__ float s[6][kBlock];

    float lo0 = CUDART_INF_F, lo1 = CUDART_INF_F, lo2 = CUDART_INF_F;
    float hi0 = -CUDART_INF_F, hi1 = -CUDART_INF_F, hi2 = -CUDART_INF_F;
    const int begin = chunkBegin[c];
    const int end = chunkEnd[c];
    for (int i = begin + tid; i < end; i += kBlock) {
        const Vec3 p = Ops::centroid(items[order[i]]);
        lo0 = fminf(lo0, p.x);
        lo1 = fminf(lo1, p.y);
        lo2 = fminf(lo2, p.z);
        hi0 = fmaxf(hi0, p.x);
        hi1 = fmaxf(hi1, p.y);
        hi2 = fmaxf(hi2, p.z);
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
            s[0][tid] = fminf(s[0][tid], s[0][tid + stride]);
            s[1][tid] = fminf(s[1][tid], s[1][tid + stride]);
            s[2][tid] = fminf(s[2][tid], s[2][tid + stride]);
            s[3][tid] = fmaxf(s[3][tid], s[3][tid + stride]);
            s[4][tid] = fmaxf(s[4][tid], s[4][tid + stride]);
            s[5][tid] = fmaxf(s[5][tid], s[5][tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        const int t = chunkTask[c];
        for (int a = 0; a < 3; ++a) {
            atomicMin(&cenLo[t * 3 + a], floatOrdered(s[a][0]));
            atomicMax(&cenHi[t * 3 + a], floatOrdered(s[3 + a][0]));
        }
    }
}

// 第二遍：每块在共享内存里攒自己的箱，再一次性合并进该区间的箱数组。
// 只有"块数 × 336" 次全局 atomic，不会出现 2M 线程抢 16 个计数的情形。
template <typename Ops>
__global__ void sahBinKernel(const typename Ops::Item* __restrict__ items,
                             const int* __restrict__ order, const int* __restrict__ chunkTask,
                             const int* __restrict__ chunkBegin,
                             const int* __restrict__ chunkEnd, int chunkCount,
                             const int* __restrict__ binId, const uint32_t* __restrict__ cenLo,
                             const uint32_t* __restrict__ cenHi, uint32_t* __restrict__ bins) {
    const int c = blockIdx.x;
    if (c >= chunkCount) return;
    const int tid = threadIdx.x;
    __shared__ int sCount[3][kSahBins];
    __shared__ uint32_t sLo[3][kSahBins][3];
    __shared__ uint32_t sHi[3][kSahBins][3];

    for (int i = tid; i < 3 * kSahBins; i += kBlock) {
        const int a = i / kSahBins;
        const int b = i % kSahBins;
        sCount[a][b] = 0;
        for (int k = 0; k < 3; ++k) {
            sLo[a][b][k] = floatOrdered(CUDART_INF_F);
            sHi[a][b][k] = floatOrdered(-CUDART_INF_F);
        }
    }
    __syncthreads();

    const int t = chunkTask[c];
    float nlo[3], nhi[3];
    for (int a = 0; a < 3; ++a) {
        nlo[a] = floatFromOrdered(cenLo[t * 3 + a]);
        nhi[a] = floatFromOrdered(cenHi[t * 3 + a]);
    }

    const int begin = chunkBegin[c];
    const int end = chunkEnd[c];
    for (int i = begin + tid; i < end; i += kBlock) {
        const int slot = order[i];
        const Vec3 cen = Ops::centroid(items[slot]);
        const AABB box = Ops::bounds(items[slot]);
        const float px[3] = {box.mn.x, box.mn.y, box.mn.z};
        const float qx[3] = {box.mx.x, box.mx.y, box.mx.z};
        const float cx[3] = {cen.x, cen.y, cen.z};
#pragma unroll
        for (int a = 0; a < 3; ++a) {
            const int b = sahBinOf(cx[a], nlo[a], nhi[a]);
            atomicAdd(&sCount[a][b], 1);
            atomicMin(&sLo[a][b][0], floatOrdered(px[0]));
            atomicMin(&sLo[a][b][1], floatOrdered(px[1]));
            atomicMin(&sLo[a][b][2], floatOrdered(px[2]));
            atomicMax(&sHi[a][b][0], floatOrdered(qx[0]));
            atomicMax(&sHi[a][b][1], floatOrdered(qx[1]));
            atomicMax(&sHi[a][b][2], floatOrdered(qx[2]));
        }
    }
    __syncthreads();

    uint32_t* dst = bins + static_cast<size_t>(binId[t]) * kSahBinWords;
    for (int i = tid; i < kSahBins * 3 * kSahBinStride; i += kBlock) {
        const int comp = i % kSahBinStride;
        const int ab = i / kSahBinStride;
        const int b = ab % kSahBins;
        const int a = ab / kSahBins;
        if (comp == 0) {
            atomicAdd(&dst[i], static_cast<uint32_t>(sCount[a][b]));
        } else if (comp <= 3) {
            atomicMin(&dst[i], sLo[a][b][comp - 1]);
        } else {
            atomicMax(&dst[i], sHi[a][b][comp - 4]);
        }
    }
}

__global__ void sahSplitKernel(int taskCount, const int* __restrict__ taskBegin,
                               const int* __restrict__ taskEnd, const int* __restrict__ binId,
                               const uint32_t* __restrict__ cenLo,
                               const uint32_t* __restrict__ cenHi,
                               const uint32_t* __restrict__ bins, int* __restrict__ splitInfo,
                               int* __restrict__ leftOut, int* __restrict__ childIntOut) {
    const int t = blockIdx.x * kBlock + threadIdx.x;
    if (t >= taskCount) return;
    const int count = taskEnd[t] - taskBegin[t];
    if (count <= kSahMedian) {
        const int lc = count / 2;
        splitInfo[t] = 0;
        leftOut[t] = lc;
        childIntOut[t] = (lc > 1) + (count - lc > 1);
        return;
    }

    const uint32_t* B = bins + static_cast<size_t>(binId[t]) * kSahBinWords;
    const int minSide = count / kSahMinSideDiv;
    float bestCost = CUDART_INF_F;
    int bestInfo = 0;
    int bestLeft = -1;

    for (int a = 0; a < 3; ++a) {
        const float olo = floatFromOrdered(cenLo[t * 3 + a]);
        const float ohi = floatFromOrdered(cenHi[t * 3 + a]);
        if (!(ohi > olo)) continue;

        // 后缀盒只依赖"从第 b 箱到第 15 箱"的并集，先倒序算一遍（16 项），
        // 再从左往右累加左半盒，切分扫描就从 O(箱²) 降成 O(箱)。
        float slo[kSahBins][3], shi[kSahBins][3];
        {
            float rlo[3] = {CUDART_INF_F, CUDART_INF_F, CUDART_INF_F};
            float rhi[3] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};
            for (int b = kSahBins - 1; b >= 0; --b) {
                const uint32_t* f = B + (a * kSahBins + b) * kSahBinStride;
                if (f[0] > 0) {
                    for (int k = 0; k < 3; ++k) {
                        rlo[k] = fminf(rlo[k], floatFromOrdered(f[1 + k]));
                        rhi[k] = fmaxf(rhi[k], floatFromOrdered(f[4 + k]));
                    }
                }
                for (int k = 0; k < 3; ++k) {
                    slo[b][k] = rlo[k];
                    shi[b][k] = rhi[k];
                }
            }
        }

        float llo[3] = {CUDART_INF_F, CUDART_INF_F, CUDART_INF_F};
        float lhi[3] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};
        int lc = 0;
        for (int b = 0; b < kSahBins - 1; ++b) {
            const uint32_t* e = B + (a * kSahBins + b) * kSahBinStride;
            const int cnt = static_cast<int>(e[0]);
            lc += cnt;
            if (cnt > 0) {
                for (int k = 0; k < 3; ++k) {
                    llo[k] = fminf(llo[k], floatFromOrdered(e[1 + k]));
                    lhi[k] = fmaxf(lhi[k], floatFromOrdered(e[4 + k]));
                }
            }
            if (lc == 0 || lc < minSide || count - lc < minSide) continue;

            const float rArea = sahArea(slo[b + 1], shi[b + 1]);
            const float cost = sahArea(llo, lhi) * lc + rArea * (count - lc);
            if (cost < bestCost) {
                bestCost = cost;
                bestInfo = 1 | (a << 2) | ((b + 1) << 4);
                bestLeft = lc;
            }
        }
    }

    if (bestInfo == 0) {  // 重心全重合或没有合法切分：退回对象中位数
        bestLeft = count / 2;
    }
    splitInfo[t] = bestInfo;
    leftOut[t] = bestLeft;
    childIntOut[t] = (bestLeft > 1) + (count - bestLeft > 1);
}

template <typename Ops>
__global__ void sahSideKernel(const typename Ops::Item* __restrict__ items,
                              const int* __restrict__ order, const int* __restrict__ slotTask,
                              const int* __restrict__ splitInfo,
                              const int* __restrict__ leftCount,
                              const int* __restrict__ taskBegin,
                              const uint32_t* __restrict__ cenLo,
                              const uint32_t* __restrict__ cenHi, int n,
                              int* __restrict__ leftFlag) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= n) return;
    const int t = slotTask[i];
    if (t < 0) {
        leftFlag[i] = 0;
        return;
    }
    const Vec3 cen = Ops::centroid(items[order[i]]);
    const float c[3] = {cen.x, cen.y, cen.z};
    float lo[3], hi[3];
    for (int a = 0; a < 3; ++a) {
        lo[a] = floatFromOrdered(cenLo[t * 3 + a]);
        hi[a] = floatFromOrdered(cenHi[t * 3 + a]);
    }
    leftFlag[i] =
        sahIsLeft(splitInfo[t], i, taskBegin[t], leftCount[t], c, lo, hi) ? 1 : 0;
}

// 稳定分区：左半在前、右半在后，两侧都保持原相对顺序。目的下标只依赖同一个
// 区间内的左计数前缀，所以每个写地址唯一、结果逐位可复现。左/右的判定直接读
// 上一趟的 leftFlag，不再重算重心与分箱。
__global__ void sahScatterKernel(const int* __restrict__ order,
                                 const int* __restrict__ slotTask,
                                 const int* __restrict__ leftFlag,
                                 const int* __restrict__ leftCount,
                                 const int* __restrict__ taskBegin,
                                 const int* __restrict__ taskEnd,
                                 const int* __restrict__ childBase,
                                 const int* __restrict__ leftPrefix, int n,
                                 int* __restrict__ orderOut, int* __restrict__ slotTaskOut) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= n) return;
    const int t = slotTask[i];
    if (t < 0) {  // 已经定成叶子的槽位不参与改动
        orderOut[i] = order[i];
        slotTaskOut[i] = -1;
        return;
    }
    const int begin = taskBegin[t];
    const int lc = leftCount[t];
    const int rc = taskEnd[t] - begin - lc;
    const bool left = leftFlag[i] != 0;
    const int localLeft = leftPrefix[i] - leftPrefix[begin];
    const int dest = left ? begin + localLeft : begin + lc + (i - begin - localLeft);
    orderOut[dest] = order[i];

    const int base = childBase[t];
    slotTaskOut[dest] = left ? (lc > 1 ? base : -1)
                             : (rc > 1 ? base + (lc > 1 ? 1 : 0) : -1);
}

// 叶子引用有两种编码：整棵构建时叶子就在同一片节点池里（n-1+slot）；森林模式只
// 重建几个区间，叶子没有临时 id，直接写 -(slot+1)——这正是查询侧期望的编码。
// posSlot 非空时区间下标是压缩后的局部序号，要查回真正的叶子槽。
__device__ __forceinline__ int leafRefOf(int leafBase, const int* posSlot, int pos) {
    if (posSlot) return -(posSlot[pos] + 1);
    return leafBase + pos;
}

__global__ void sahAssignKernel(int taskCount, const int* __restrict__ taskBegin,
                                const int* __restrict__ taskEnd, const int* __restrict__ taskNode,
                                const int* __restrict__ leftCount,
                                const int* __restrict__ childBase, int internalBase, int leafBase,
                                const int* __restrict__ posSlot, GpuBVHNode* __restrict__ nodes,
                                int* __restrict__ parent, int* __restrict__ nextBegin,
                                int* __restrict__ nextEnd, int* __restrict__ nextNode,
                                const int* __restrict__ taskDepth, int* __restrict__ nextDepth,
                                int* __restrict__ depthOut) {
    const int t = blockIdx.x * kBlock + threadIdx.x;
    if (t >= taskCount) return;
    const int begin = taskBegin[t];
    const int end = taskEnd[t];
    const int lc = leftCount[t];
    const int rc = end - begin - lc;
    const int p = childBase[t];
    const int node = taskNode[t];

    const int leftId = lc > 1 ? internalBase + p : leafRefOf(leafBase, posSlot, begin);
    const int rightId = rc > 1 ? internalBase + p + (lc > 1 ? 1 : 0)
                               : leafRefOf(leafBase, posSlot, begin + lc);
    nodes[node].left = leftId;
    nodes[node].right = rightId;
    if (leftId >= 0) parent[leftId] = node;
    if (rightId >= 0) parent[rightId] = node;

    const int childDepth = taskDepth ? taskDepth[t] + 1 : 0;
    if (depthOut) depthOut[node] = taskDepth[t];

    if (lc > 1) {
        nextBegin[p] = begin;
        nextEnd[p] = begin + lc;
        nextNode[p] = leftId;
        if (nextDepth) nextDepth[p] = childDepth;
    }
    if (rc > 1) {
        const int id = p + (lc > 1 ? 1 : 0);
        nextBegin[id] = begin + lc;
        nextEnd[id] = end;
        nextNode[id] = rightId;
        if (nextDepth) nextDepth[id] = childDepth;
    }
}

__global__ void sahSubtreeSizeKernel(int taskCount, const int* __restrict__ taskBegin,
                                     const int* __restrict__ taskEnd, int* __restrict__ out) {
    const int t = blockIdx.x * kBlock + threadIdx.x;
    if (t < taskCount) out[t] = taskEnd[t] - taskBegin[t] - 2;
}

// 小片区间的收尾：平衡中位数切分 + 串行建子树。区间 <=64，递归深度 <=6，
// 用 32 帧的显式栈就够；每个任务线程只写自己的节点。
// 帧里的 next 是"本节点后代"的分配起点（根帧就是本任务领到的起点，
// 内部帧是 childId+1——childId 自己已占掉一个号）。
__global__ void sahSerialBuildKernel(int taskCount, const int* __restrict__ taskBegin,
                                     const int* __restrict__ taskEnd,
                                     const int* __restrict__ taskNode,
                                     const int* __restrict__ childBase, int internalBase,
                                     int leafBase, const int* __restrict__ posSlot,
                                     GpuBVHNode* __restrict__ nodes, int* __restrict__ parent,
                                     const int* __restrict__ taskDepth, int* __restrict__ depthOut,
                                     int* __restrict__ errors) {
    const int t = blockIdx.x * kBlock + threadIdx.x;
    if (t >= taskCount) return;

    struct Frame {
        int begin, count, id, next, depth;
    };
    Frame st[32];
    int sp = 0;
    const int count = taskEnd[t] - taskBegin[t];
    if (count > kSahMedian) {
        atomicAdd(&errors[0], 1);
        return;
    }
    st[sp++] = {taskBegin[t], count, taskNode[t], internalBase + childBase[t],
                taskDepth ? taskDepth[t] : 0};
    while (sp > 0) {
        const Frame f = st[--sp];
        const int lc = f.count / 2;
        const int rc = f.count - lc;
        const int leftId = lc == 1 ? leafRefOf(leafBase, posSlot, f.begin) : f.next;
        const int rightId = rc == 1 ? leafRefOf(leafBase, posSlot, f.begin + lc)
                                    : f.next + (lc - 1);
        nodes[f.id].left = leftId;
        nodes[f.id].right = rightId;
        if (leftId >= 0) parent[leftId] = f.id;
        if (rightId >= 0) parent[rightId] = f.id;
        if (depthOut) depthOut[f.id] = f.depth;
        if (rc > 1 && sp < 32) st[sp++] = {f.begin + lc, rc, rightId, rightId + 1, f.depth + 1};
        if (lc > 1 && sp < 32) st[sp++] = {f.begin, lc, leftId, leftId + 1, f.depth + 1};
        if ((rc > 1 || lc > 1) && sp >= 32) {
            atomicAdd(&errors[0], 1);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// 拓扑变更：子树统计 → 切分选择 → 规范重编号。
// 树里每个内部节点的子树覆盖一段连续的叶子槽，这一点让"删了谁、剩多少、重建
// 哪一块"全部可以用区间算术表达，不需要在主机上还原整棵树。
// ---------------------------------------------------------------------------
constexpr int kCutFields = 5;  // oldBeg, oldEnd, newSize, depth, nodeId

// 叶子贡献 size=1，内部节点先清零；随后按深度自深向浅逐层相加（和 refit 一样，
// 第 d 层只读第 d+1 层）。minSlot 顺带算出子树的区间起点。
__global__ void subtreeInitKernel(int* __restrict__ size, int* __restrict__ minSlot, int n,
                                  int nodeCount) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= nodeCount) return;
    if (i >= n - 1) {
        size[i] = 1;
        minSlot[i] = i - (n - 1);
    } else {
        size[i] = 0;
        minSlot[i] = 0x7fffffff;
    }
}

__global__ void subtreeMergeKernel(const GpuBVHNode* __restrict__ nodes, int* __restrict__ size,
                                   int* __restrict__ minSlot, const int* __restrict__ level,
                                   int count) {
    const int t = blockIdx.x * kBlock + threadIdx.x;
    if (t >= count) return;
    const int i = level[t];
    const int l = nodes[i].left;
    const int r = nodes[i].right;
    size[i] = size[l] + size[r];
    minSlot[i] = minSlot[l] < minSlot[r] ? minSlot[l] : minSlot[r];
}

// 删除位图 + 它的前缀和：keptPrefix[s] 是 s 之前还剩多少个图元，也就是旧槽 s
// 在新场景里的下标。多留一格哨兵，扫描一次就能拿到总数。
__global__ void keptFlagsInitKernel(int n, int* __restrict__ flags) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i > n) return;
    flags[i] = i < n ? 1 : 0;
}

__global__ void clearKeptFlagsKernel(const int* __restrict__ deleted, int count,
                                     int* __restrict__ flags) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < count) flags[deleted[i]] = 0;
}

// 子树区间是"叶子槽"的区间，删除表却是场景下标，所以还要一张槽序的存活位图。
__global__ void slotKeptFlagsKernel(const int* __restrict__ primIds,
                                    const int* __restrict__ keptPrefix, int n,
                                    int* __restrict__ flags) {
    const int s = blockIdx.x * kBlock + threadIdx.x;
    if (s >= n) return;
    const int id = primIds[s];
    flags[s] = (keptPrefix[id + 1] != keptPrefix[id]) ? 1 : 0;
}

// primIds 的反向：场景下标 -> 叶子槽。只有建树时那次重排之后两者不同，拓扑
// 更新完又变回 iota，但每条编辑脚本都要从"场景下标"找到真正的叶子。
__global__ void invertPermKernel(const int* __restrict__ primIds, int n,
                                 int* __restrict__ primSlot) {
    const int s = blockIdx.x * kBlock + threadIdx.x;
    if (s < n) primSlot[primIds[s]] = s;
}

// 每个被删掉的叶子往上爬到第一个"够大、且自己能凑够两个新叶子"的祖先。
// 删除表是场景下标，先用 primSlot 换成叶子槽。新增的图元（中途插入 + 尾部追加）
// 在槽序里都接在最后一个旧叶子之后，只能落进拥有最后一个旧叶子的那棵子树，所以
// 尾部叶子也参与一次爬升，并且只有它可以靠新增块补足新叶子数。
// 同一棵子树会被多个删除叶子选中，epoch 标记顺手去重（谁先写都一样）。
//
// 尾部那份用单独的 minSize：新增块越大，切分根就得越高，否则每轮都只把新增
// 图元挂在一条越来越深的路径底下（R4 漂移）。要求重建区间里至少有一份与新增
// 块同样多的旧叶子，新子树的根才能稳定在上一次的位置。
__global__ void climbKernel(const int* __restrict__ deleted, const int* __restrict__ primSlot,
                            int delCount, int tailLeaf, int extra,
                            const int* __restrict__ parent, const int* __restrict__ size,
                            const int* __restrict__ minSlot, const int* __restrict__ keptPrefix,
                            int nOld, int minSize, int extraMinSize, int* __restrict__ mark,
                            int epoch, int* __restrict__ cand, int* __restrict__ flag) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    const int total = delCount + (extra > 0 ? 1 : 0);
    if (i >= total) return;
    const int slot = i < delCount ? primSlot[deleted[i]] : tailLeaf;
    const int need = i < delCount ? minSize : extraMinSize;
    int node = parent[nOld - 1 + slot];
    while (node > 0) {
        const int beg = minSlot[node];
        const int end = beg + size[node];
        const int kept = keptPrefix[end] - keptPrefix[beg];
        const int bonus = (extra > 0 && end == nOld) ? extra : 0;
        if (size[node] >= need && kept + bonus >= 2) break;
        node = parent[node];
    }
    if (node < 0) node = 0;  // 树比 minSize 还小：整棵当切分根
    cand[i] = node;
    flag[i] = (atomicExch(&mark[node], epoch) == epoch) ? 0 : 1;
}

// 区间起点升序；同起点时大的（祖先）在前，后面一趟扫描一次就能压掉被包含的候选。
__global__ void candKeyKernel(const int* __restrict__ ids, int count,
                              const int* __restrict__ minSlot, const int* __restrict__ size,
                              uint64_t* __restrict__ keys, int* __restrict__ vals) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count) return;
    const int id = ids[i];
    keys[i] = (static_cast<uint64_t>(static_cast<uint32_t>(minSlot[id])) << 32) |
              static_cast<uint32_t>(~size[id]);
    vals[i] = id;
}

// 候选按祖先关系构成一片森林：一个候选被压掉，当且仅当它有一个候选祖先。
// 每个候选从父节点往上爬，找到第一个候选祖先就说明自己被它包含，标记自己后停。
// 祖先候选自己的线程会继续往上找，所以每层只爬一趟。
__global__ void markDominatedKernel(const int* __restrict__ cands, int count,
                                    const int* __restrict__ parent,
                                    const int* __restrict__ candEpochMark, int epoch,
                                    int* __restrict__ dom) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count) return;
    int node = parent[cands[i]];
    while (node > 0) {
        if (candEpochMark[node] == epoch) {
            dom[cands[i]] = 1;
            return;
        }
        node = parent[node];
    }
}

__global__ void keepFromDomKernel(const int* __restrict__ cands, int count,
                                  const int* __restrict__ dom, int* __restrict__ keep) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < count) keep[i] = dom[cands[i]] ? 0 : 1;
}

__global__ void compactKernel(const int* __restrict__ src, const int* __restrict__ flag,
                              const int* __restrict__ off, int count, int* __restrict__ dst) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < count && flag[i]) dst[off[i]] = src[i];
}

__global__ void cutMetaKernel(const int* __restrict__ ids, const int* __restrict__ keep,
                              const int* __restrict__ keepOff, int count,
                              const int* __restrict__ minSlot, const int* __restrict__ size,
                              const int* __restrict__ depth, const int* __restrict__ keptSlotPrefix,
                              int nOld, int extra, int* __restrict__ meta) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count || !keep[i]) return;
    const int id = ids[i];
    const int beg = minSlot[id];
    const int end = beg + size[id];
    const int tail = (extra > 0 && end == nOld) ? extra : 0;
    int* out = meta + static_cast<size_t>(keepOff[i]) * kCutFields;
    out[0] = beg;
    out[1] = end;
    out[2] = keptSlotPrefix[end] - keptSlotPrefix[beg] + tail;  // 重建后这块有几个叶子
    out[3] = depth[id];
    out[4] = id;
}

// 森林只重建这些区间：把每个删除区间里的幸存者按槽序压进一张紧凑数组。区间
// 下标 pos 对应真正的新叶子槽 slotOf[pos]（槽序压缩下标），排序数组 order 也
// 存这个下标——森林内部只需要"紧凑位置连续"，叶引用可以指向任意新槽。
// 新增的图元接在尾区间的末尾（新槽就是 nSurv 之后），中途插入的在前、追加的在后。
__global__ void buildSlotOfKernel(int nOld, const int* __restrict__ keptSlotPrefix,
                                  const int* __restrict__ cutBeg, const int* __restrict__ cutEnd,
                                  int cutCount, const int* __restrict__ compactBeg, int nSurv,
                                  int extra, int compactLen, int* __restrict__ slotOf,
                                  int* __restrict__ order, int* __restrict__ slotTask) {
    const int s = blockIdx.x * kBlock + threadIdx.x;
    if (s < nOld) {
        if (keptSlotPrefix[s + 1] == keptSlotPrefix[s]) return;  // 被删掉的槽
        int lo = 0, hi = cutCount;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (cutBeg[mid] <= s) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        const int r = lo - 1;
        if (r < 0 || cutEnd[r] <= s) return;  // 不在任何重建区间里
        const int newSlot = keptSlotPrefix[s];
        const int pos = compactBeg[r] + (keptSlotPrefix[s] - keptSlotPrefix[cutBeg[r]]);
        slotOf[pos] = newSlot;
        order[pos] = newSlot;
        slotTask[pos] = r;
    } else if (s < nOld + extra) {
        const int k = s - nOld;
        const int pos = compactLen - extra + k;
        slotOf[pos] = nSurv + k;
        order[pos] = slotOf[pos];
        slotTask[pos] = cutCount - 1;
    }
}

// 新叶子槽 = 旧槽序里删掉删除项之后的压缩下标。载荷按槽序重排，场景顺序由
// d_primIds 单独记录——两者不能混：重建子树的叶子区间只在槽序里连续。
template <typename Ops>
__global__ void gatherSurvivorsKernel(const typename Ops::Item* __restrict__ oldTris,
                                      const int* __restrict__ slotPrefix, int nOld,
                                      typename Ops::Item* __restrict__ newTris) {
    const int s = blockIdx.x * kBlock + threadIdx.x;
    if (s >= nOld) return;
    const int dst = slotPrefix[s];
    if (slotPrefix[s + 1] != dst) newTris[dst] = oldTris[s];  // 删掉的位置不写
}

// 中途插入会把后面的幸存者往后推。设插入表为 i_j，位移 e_j = i_j - j（非降），
// 则场景序里第 k 个幸存者的位移恰好是 e_j <= k 的插入个数（插入位置只可能落到
// 已排好的幸存者之间）。一次二分，不必为每个槽建整张映射表。
__device__ inline int survivorScene(const int* __restrict__ shifts, int nIns, int k) {
    int lo = 0, hi = nIns;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (shifts[mid] <= k) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return k + lo;
}

// 新槽 -> 新场景下标：幸存者沿旧 primIds 先换成场景序的幸存者序号 k，再按插入
// 位移换成真正的新场景下标；新增块按 extraScene 取（先中途插入、后尾部追加）。
// 删除/插入/追加之后 primIds 一般不再是 iota，查询输出靠它回场景下标。
__global__ void remapPermKernel(const int* __restrict__ primIds,
                                const int* __restrict__ slotPrefix,
                                const int* __restrict__ scenePrefix,
                                const int* __restrict__ shifts, int nIns, int nOld, int nSurv,
                                int extra, const int* __restrict__ extraScene,
                                int* __restrict__ newPrimIds) {
    const int s = blockIdx.x * kBlock + threadIdx.x;
    if (s < nOld) {
        const int dst = slotPrefix[s];
        if (slotPrefix[s + 1] != dst) {
            const int k = scenePrefix[primIds[s]];
            newPrimIds[dst] = nIns > 0 ? survivorScene(shifts, nIns, k) : k;
        }
    } else if (s < nOld + extra) {
        newPrimIds[nSurv + (s - nOld)] = extraScene[s - nOld];
    }
}

// 移动图元按"新场景下标 -> 新叶子槽"散写。场景下标与槽位只有初始树相等，
// 一次拓扑更新之后这种对应就断了。
template <typename Ops>
__global__ void scatterItemsKernel(const int* __restrict__ sceneIds,
                                   const typename Ops::Item* __restrict__ src, int count,
                                   typename Ops::Item* __restrict__ dst,
                                   const int* __restrict__ sceneSlot) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < count) dst[sceneSlot[sceneIds[i]]] = src[i];
}

// 最终编号 = 按 (深度, 来源序) 的排名。键的高 6 位是深度、低 26 位是来源序，
// 死掉的旧节点用全 1 哨兵键排到最后——这样"保留哪些"不需要另一张标记表。
__global__ void relayoutKeyKernel(const int* __restrict__ minSlot, const int* __restrict__ size,
                                  const int* __restrict__ depth, const int* __restrict__ cutBeg,
                                  const int* __restrict__ cutEnd, int cutCount,
                                  int nOldInternal, const int* __restrict__ tempDepth, int want,
                                  uint32_t* __restrict__ keys, int* __restrict__ vals) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= nOldInternal + want) return;
    if (i < nOldInternal) {
        const int beg = minSlot[i];
        int lo = 0, hi = cutCount;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (cutBeg[mid] <= beg) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        const bool dead = lo > 0 && cutEnd[lo - 1] >= beg + size[i];
        keys[i] = dead ? 0xffffffffu
                       : (static_cast<uint32_t>(depth[i]) << 26) | static_cast<uint32_t>(i);
        vals[i] = i;
    } else {
        const int t = i - nOldInternal;
        keys[i] = (static_cast<uint32_t>(tempDepth[t]) << 26) | static_cast<uint32_t>(i);
        vals[i] = i;
    }
}

__global__ void rankKernel(const int* __restrict__ sortedVals, int count,
                           int* __restrict__ idMap) {
    const int p = blockIdx.x * kBlock + threadIdx.x;
    if (p < count) idMap[sortedVals[p]] = p;
}

// 三处重映射。旧节点的孩子有三种：另一个活着的旧内部节点（idMap）、旧叶子
// （节点 id = nOld-1+slot，新 id = nNew-1+slotPrefix[slot]）、以及被整块替换掉的
// 切分根（换成重建森林的根 idMap[nOldInternal + r]）。临时节点的孩子只有两种：
// 临时内部（idMap）和负号叶子引用 -(slot+1)。
__global__ void relayoutKernel(const int* __restrict__ sortedVals, int internalCount,
                               const GpuBVHNode* __restrict__ oldNodes,
                               const int* __restrict__ oldDepth, int nOldInternal,
                               const int* __restrict__ minSlot, const int* __restrict__ size,
                               const int* __restrict__ cutBeg, const int* __restrict__ cutEnd,
                               const int* __restrict__ slotPrefix,
                               const GpuBVHNode* __restrict__ tempNodes,
                               const int* __restrict__ tempDepth, const int* __restrict__ idMap,
                               int nNew, GpuBVHNode* __restrict__ newNodes,
                               int* __restrict__ newParent, int* __restrict__ newDepth,
                               int cutCount) {
    const int p = blockIdx.x * kBlock + threadIdx.x;
    if (p >= internalCount) return;
    const int src = sortedVals[p];
    GpuBVHNode out;
    int l = 0, r = 0, d = 0;
    if (src < nOldInternal) {
        out = oldNodes[src];
        auto remapOld = [&](int c) {
            if (c < 0) return nNew - 1 + slotPrefix[-c - 1];  // 旧式负号叶子引用
            if (c >= nOldInternal) {  // 叶子节点 id：旧槽经槽序前缀换成新槽
                return nNew - 1 + slotPrefix[c - nOldInternal];
            }
            // 内部节点：活着就换名，死掉的话它必然是某个切分根，换成森林的根。
            const int beg = minSlot[c];
            int lo = 0, hi = cutCount;
            while (lo < hi) {
                const int mid = (lo + hi) / 2;
                if (cutBeg[mid] <= beg) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            const bool dead = lo > 0 && cutEnd[lo - 1] >= beg + size[c];
            return dead ? idMap[nOldInternal + lo - 1] : idMap[c];
        };
        l = remapOld(out.left);
        r = remapOld(out.right);
        d = oldDepth[src];
    } else {
        const int t = src - nOldInternal;
        out = tempNodes[t];
        l = out.left >= 0 ? idMap[nOldInternal + out.left] : nNew - 2 - out.left;
        r = out.right >= 0 ? idMap[nOldInternal + out.right] : nNew - 2 - out.right;
        d = tempDepth[t];
    }
    out.left = l;
    out.right = r;
    newNodes[p] = out;
    newParent[l] = p;
    newParent[r] = p;
    newDepth[p] = d;
}

template <typename Ops>
__global__ void emitLeafKernel(const typename Ops::Item* __restrict__ newTris, int nNew,
                               GpuBVHNode* __restrict__ newNodes,
                               const int* __restrict__ newParent, int* __restrict__ newDepth) {
    const int j = blockIdx.x * kBlock + threadIdx.x;
    if (j >= nNew) return;
    GpuBVHNode& leaf = newNodes[nNew - 1 + j];
    setNodeBox(leaf, Ops::bounds(newTris[j]));
    leaf.left = -(j + 1);
    leaf.right = 1;
    newDepth[nNew - 1 + j] = newDepth[newParent[nNew - 1 + j]] + 1;
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

// 建树尾段：叶子盒 → 深度分桶 → 自底向上合并 → 留下 refit 的拓扑缓存。
// LBVH 与 SAH 共用；两者都保证"内部节点 0..n-2、叶子 n-1..2n-2"，所以这一段
// 不需要知道树是怎么划分出来的。bucketMs 含叶子盒、深度排序与偏移；boxMs 含
// refit 缓存分配和逐层合并。bvh.depth 必须由调用方先算好。
template <typename Ops>
void finalizeTree(GpuBVH& bvh, int n, GpuBVHNode* nodes, const typename Ops::Item* sortedTris,
                  DevBuf<int>& dParent, DevBuf<int>& dDepth, double* bucketMs, double* boxMs) {
    const int internalCount = n - 1;
    const int nodeCount = 2 * n - 1;

    StageClock clock;
    clock.mark();  // 0

    DevBuf<int> dDepthVal(internalCount), dDepthValSorted(internalCount);
    DevBuf<uint32_t> dDepthKeySorted(nodeCount);
    DevBuf<int> dOffsets(kDepthBins + 1);

    leafInitKernel<Ops><<<blocksFor(n), kBlock>>>(nodes, sortedTris, n);
    iotaKernel<<<blocksFor(internalCount), kBlock>>>(dDepthVal.p, internalCount);
    cudaOk(cudaGetLastError(), "leaf init");

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
    depthOffsetsKernel<<<1, kDepthBins + 1>>>(dDepthKeySorted.p, internalCount, bins, dOffsets.p);
    cudaOk(cudaGetLastError(), "depth offsets");

    int offsets[kDepthBins + 1] = {};
    cudaOk(cudaMemcpy(offsets, dOffsets.p, static_cast<size_t>(bins + 1) * sizeof(int),
                      cudaMemcpyDeviceToHost),
           "D2H offsets");
    if (offsets[bins] != internalCount) {
        throw std::runtime_error("build: depth bucketing lost nodes");
    }

    // 拓扑不变，这个 (深度, 编号) 顺序也不变：留给 refit 逐层重放，
    // 省掉每次更新重算 parent/depth + 一次 7 位基数排序。
    bvh.d_refitOrder = topoAwareMalloc(static_cast<size_t>(internalCount) * sizeof(int));
    cudaOk(cudaMemcpy(bvh.d_refitOrder, dDepthValSorted.p,
                      static_cast<size_t>(internalCount) * sizeof(int), cudaMemcpyDeviceToDevice),
           "D2D refit order");
    bvh.refitOffsets.assign(offsets, offsets + bins + 1);

    // 只合并脏叶子的祖先要用：parent / depth 建树时已经算过，留一份免得重算。
    const size_t ints = static_cast<size_t>(nodeCount) * sizeof(int);
    bvh.d_parent = topoAwareMalloc(ints);
    cudaOk(cudaMemcpy(bvh.d_parent, dParent.p, ints, cudaMemcpyDeviceToDevice), "parent D2D");
    bvh.d_depth = topoAwareMalloc(ints);
    cudaOk(cudaMemcpy(bvh.d_depth, dDepth.p, ints, cudaMemcpyDeviceToDevice), "depth D2D");
    bvh.d_mark = topoAwareMalloc(ints);
    cudaOk(cudaMemset(bvh.d_mark, 0, ints), "mark clear");
    bvh.d_dirtyOrder = topoAwareMalloc(static_cast<size_t>(internalCount) * sizeof(int));
    bvh.d_refitOffsets = topoAwareMalloc(static_cast<size_t>(bins + 1) * sizeof(int));
    cudaOk(cudaMemcpy(bvh.d_refitOffsets, offsets, static_cast<size_t>(bins + 1) * sizeof(int),
                      cudaMemcpyHostToDevice),
           "refit offsets H2D");
    bvh.d_levelCounts = topoAwareMalloc(static_cast<size_t>(bins) * sizeof(int));
    clock.mark();  // 1

    // 从最深层往上：第 d 层的节点只读第 d+1 层，天然无竞争。
    for (int d = bins - 1; d >= 0; --d) {
        const int count = offsets[d + 1] - offsets[d];
        if (count == 0) continue;
        const int blocks = (count + kBlock * kRefitUnroll - 1) / (kBlock * kRefitUnroll);
        refitKernel<<<blocks, kBlock>>>(nodes, dDepthValSorted.p + offsets[d], count);
    }
    cudaOk(cudaGetLastError(), "refit");
    clock.mark();  // 2
    cudaOk(cudaEventSynchronize(clock.ev[2]), "sync");

    if (bucketMs) *bucketMs = clock.ms(0, 1);
    if (boxMs) *boxMs = clock.ms(1, 2);
}

// SAH 层流水线的工作区。层间只换缓冲区、不重新分配；构建与增量更新用同一份。
struct SahWorkspace {
    DevBuf<int> orderA, orderB;
    DevBuf<int> slotA, slotB;
    DevBuf<int> beginA, endA, nodeA;
    DevBuf<int> beginB, endB, nodeB;
    DevBuf<int> depthA, depthB;  // 每个任务的局部深度（只有森林模式用）
    DevBuf<int> splitInfo, leftCount, childInt, childBase, subtree;
    DevBuf<int> leftFlag, leftPrefix;
    DevBuf<uint32_t> cenLo, cenHi;
    DevBuf<int> binId, chunkTask, chunkBegin, chunkEnd;
    DevBuf<uint32_t> bins;
    DevBuf<int> counters, errors;
    std::size_t scanBytes = 0;
    DevBuf<unsigned char> scanTemp;

    explicit SahWorkspace(int n)
        : orderA(n), orderB(n), slotA(n), slotB(n), beginA(n), endA(n), nodeA(n), beginB(n),
          endB(n), nodeB(n), depthA(n), depthB(n), splitInfo(n), leftCount(n), childInt(n),
          childBase(n), subtree(n), leftFlag(n), leftPrefix(n),
          cenLo(static_cast<size_t>(n) * 3), cenHi(static_cast<size_t>(n) * 3), binId(n),
          chunkTask(n / 32 + 16), chunkBegin(n / 32 + 16), chunkEnd(n / 32 + 16),
          bins(static_cast<size_t>(n / (kSahMedian + 1) + 2) * kSahBinWords), counters(2),
          errors(1) {
        std::size_t childBytes = 0;
        std::size_t leftBytes = 0;
        cudaOk(cub::DeviceScan::ExclusiveSum(nullptr, childBytes, childInt.p, childBase.p, n),
               "child scan(probe)");
        cudaOk(cub::DeviceScan::ExclusiveSum(nullptr, leftBytes, leftFlag.p, leftPrefix.p, n),
               "left scan(probe)");
        scanBytes = std::max(childBytes, leftBytes);
        scanTemp = DevBuf<unsigned char>(scanBytes > 0 ? scanBytes : 1);
    }
};

struct SahLevelTimings {
    double bin = 0.0;
    double split = 0.0;
    double part = 0.0;
};

// 层流水线本体：把一批已种好的任务（区间 + 目标节点 + 局部深度）自顶向下建完。
// 整棵构建是一个任务、根节点 0、叶子 n-1+slot；拓扑更新是 K 个任务、临时区节点
// 0..K-1、叶子写成 -(slot+1)。两者的记账规则相同：最终用掉的内部节点数必须正好
// 等于 want，否则说明某个任务的区间被算漏了。
template <typename Ops>
int* sahBuildForest(const typename Ops::Item* dItems, int n, int leafBase, const int* posSlot,
                    int internalBase0, int taskCount, SahWorkspace& ws, GpuBVHNode* nodes,
                    int* parent, int want, int* depthOut, SahLevelTimings* lt) {
    int* order = ws.orderA.p;
    int* orderOther = ws.orderB.p;
    int* slot = ws.slotA.p;
    int* slotOther = ws.slotB.p;
    int* begin = ws.beginA.p;
    int* beginOther = ws.beginB.p;
    int* end = ws.endA.p;
    int* endOther = ws.endB.p;
    int* nodeId = ws.nodeA.p;
    int* nodeIdOther = ws.nodeB.p;
    int* taskDepth = depthOut ? ws.depthA.p : nullptr;
    int* taskDepthOther = depthOut ? ws.depthB.p : nullptr;

    int internalBase = internalBase0;
    int level = 0;
    double binMs = 0.0, splitMs = 0.0, partMs = 0.0;
    QuadClock qc;

    while (taskCount > 0) {
        if (++level >= kDepthBins) throw std::runtime_error("buildSAH: depth limit");
        cudaOk(cudaMemset(ws.counters.p, 0, 2 * sizeof(int)), "counters");
        sahPrepKernel<<<blocksFor(taskCount), kBlock>>>(
            taskCount, begin, end, ws.binId.p, ws.chunkTask.p, ws.chunkBegin.p, ws.chunkEnd.p,
            ws.cenLo.p, ws.cenHi.p, ws.counters.p, ws.counters.p + 1);
        cudaOk(cudaGetLastError(), "sah prep");
        int hBin = 0, hChunk = 0;
        cudaOk(cudaMemcpy(&hBin, ws.counters.p, sizeof(int), cudaMemcpyDeviceToHost), "bin count");
        cudaOk(cudaMemcpy(&hChunk, ws.counters.p + 1, sizeof(int), cudaMemcpyDeviceToHost),
               "chunk count");
        if (hBin == 0) break;  // 全部区间都 <= kSahMedian，交给串行收尾

        qc.mark(0);
        sahClearBinsKernel<<<blocksFor(hBin * kSahBinWords), kBlock>>>(ws.bins.p,
                                                                      hBin * kSahBinWords);
        sahCenBoundsKernel<Ops><<<hChunk, kBlock>>>(dItems, order, ws.chunkTask.p,
                                                    ws.chunkBegin.p, ws.chunkEnd.p, hChunk,
                                                    ws.cenLo.p, ws.cenHi.p);
        sahBinKernel<Ops><<<hChunk, kBlock>>>(dItems, order, ws.chunkTask.p, ws.chunkBegin.p,
                                              ws.chunkEnd.p, hChunk, ws.binId.p, ws.cenLo.p,
                                              ws.cenHi.p, ws.bins.p);
        cudaOk(cudaGetLastError(), "sah bins");
        qc.mark(1);

        sahSplitKernel<<<blocksFor(taskCount), kBlock>>>(taskCount, begin, end, ws.binId.p,
                                                         ws.cenLo.p, ws.cenHi.p, ws.bins.p,
                                                         ws.splitInfo.p, ws.leftCount.p,
                                                         ws.childInt.p);
        cudaOk(cudaGetLastError(), "sah split");
        cudaOk(cub::DeviceScan::ExclusiveSum(ws.scanTemp.p, ws.scanBytes, ws.childInt.p,
                                             ws.childBase.p, taskCount),
               "child scan");
        sahSideKernel<Ops><<<blocksFor(n), kBlock>>>(dItems, order, slot, ws.splitInfo.p,
                                                     ws.leftCount.p, begin, ws.cenLo.p, ws.cenHi.p,
                                                     n, ws.leftFlag.p);
        cudaOk(cudaGetLastError(), "sah side");
        cudaOk(cub::DeviceScan::ExclusiveSum(ws.scanTemp.p, ws.scanBytes, ws.leftFlag.p,
                                             ws.leftPrefix.p, n),
               "left scan");
        qc.mark(2);

        sahScatterKernel<<<blocksFor(n), kBlock>>>(order, slot, ws.leftFlag.p, ws.leftCount.p,
                                                   begin, end, ws.childBase.p, ws.leftPrefix.p, n,
                                                   orderOther, slotOther);
        sahAssignKernel<<<blocksFor(taskCount), kBlock>>>(
            taskCount, begin, end, nodeId, ws.leftCount.p, ws.childBase.p, internalBase, leafBase,
            posSlot, nodes, parent, beginOther, endOther, nodeIdOther, taskDepth, taskDepthOther,
            depthOut);
        cudaOk(cudaGetLastError(), "sah partition");
        qc.mark(3);
        cudaOk(cudaEventSynchronize(qc.ev[3]), "sah level");
        binMs += qc.ms(0, 1);
        splitMs += qc.ms(1, 2);
        partMs += qc.ms(2, 3);

        int lastBase = 0, lastInt = 0;
        cudaOk(cudaMemcpy(&lastBase, ws.childBase.p + (taskCount - 1), sizeof(int),
                          cudaMemcpyDeviceToHost),
               "child base");
        cudaOk(cudaMemcpy(&lastInt, ws.childInt.p + (taskCount - 1), sizeof(int),
                          cudaMemcpyDeviceToHost),
               "child int");
        const int total = lastBase + lastInt;

        std::swap(order, orderOther);
        std::swap(slot, slotOther);
        std::swap(begin, beginOther);
        std::swap(end, endOther);
        std::swap(nodeId, nodeIdOther);
        std::swap(taskDepth, taskDepthOther);
        taskCount = total;
        internalBase += total;
    }

    if (taskCount > 0) {  // 收尾：每个小区间串行建平衡子树
        sahSubtreeSizeKernel<<<blocksFor(taskCount), kBlock>>>(taskCount, begin, end, ws.subtree.p);
        cudaOk(cub::DeviceScan::ExclusiveSum(ws.scanTemp.p, ws.scanBytes, ws.subtree.p,
                                             ws.childBase.p, taskCount),
               "subtree scan");
        cudaOk(cudaMemset(ws.errors.p, 0, sizeof(int)), "errors");
        sahSerialBuildKernel<<<blocksFor(taskCount), kBlock>>>(
            taskCount, begin, end, nodeId, ws.childBase.p, internalBase, leafBase, posSlot, nodes,
            parent, taskDepth, depthOut, ws.errors.p);
        cudaOk(cudaGetLastError(), "sah serial");

        int lastBase = 0, lastSize = 0, hErr = 0;
        cudaOk(cudaMemcpy(&lastBase, ws.childBase.p + (taskCount - 1), sizeof(int),
                          cudaMemcpyDeviceToHost),
               "subtree base");
        cudaOk(cudaMemcpy(&lastSize, ws.subtree.p + (taskCount - 1), sizeof(int),
                          cudaMemcpyDeviceToHost),
               "subtree size");
        cudaOk(cudaMemcpy(&hErr, ws.errors.p, sizeof(int), cudaMemcpyDeviceToHost), "serial err");
        if (hErr) throw std::runtime_error("buildSAH: serial subtree failed");
        if (internalBase + lastBase + lastSize != want) {
            throw std::runtime_error("buildSAH: internal node accounting mismatch");
        }
    } else if (internalBase != want) {
        throw std::runtime_error("buildSAH: internal node accounting mismatch");
    }

    if (lt) {
        lt->bin = binMs;
        lt->split = splitMs;
        lt->part = partMs;
    }
    return order;  // 层间会换缓冲区，最终顺序不一定在最初那张数组里
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
    bvh.sah = false;
    bvh.d_nodes = topoAwareMalloc(static_cast<size_t>(nodeCount) * sizeof(GpuBVHNode));
    bvh.d_tris = topoAwareMalloc(static_cast<size_t>(n) * sizeof(Item));
    bvh.d_primIds = topoAwareMalloc(static_cast<size_t>(n) * sizeof(int));

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

        finalizeTree<Ops>(bvh, n, nodes, sortedTris, dParent, dDepth, nullptr, nullptr);
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

// 自顶向下分箱 SAH 的主机侧调度：每层"准备 → 重心域 → 分箱 → 选切分 → 稳定分区 →
// 分配子节点"，装箱与块列表靠 atomic 分配但只做可交换合并，结果逐位可复现。
template <typename Ops>
GpuBVH buildSAH(const typename Ops::Item* hostItems, int n, GpuBuildTimings* timings) {
    using Item = typename Ops::Item;
    if (n < 2) throw std::runtime_error("buildSAH: at least 2 primitives required");

    const int nodeCount = 2 * n - 1;
    const int internalCount = n - 1;

    GpuBVH bvh;
    bvh.primCount = n;
    bvh.nodeCount = nodeCount;
    bvh.sah = true;
    bvh.d_nodes = topoAwareMalloc(static_cast<size_t>(nodeCount) * sizeof(GpuBVHNode));
    bvh.d_tris = topoAwareMalloc(static_cast<size_t>(n) * sizeof(Item));
    bvh.d_primIds = topoAwareMalloc(static_cast<size_t>(n) * sizeof(int));

    auto* nodes = static_cast<GpuBVHNode*>(bvh.d_nodes);
    auto* sortedTris = static_cast<Item*>(bvh.d_tris);
    auto* primIds = static_cast<int*>(bvh.d_primIds);

    try {
        DevBuf<Item> dSrc(n);
        cudaOk(cudaMemcpy(dSrc.p, hostItems, static_cast<size_t>(n) * sizeof(Item),
                          cudaMemcpyHostToDevice),
               "H2D prims");

        SahWorkspace ws(n);
        DevBuf<int> dParent(nodeCount);
        DevBuf<int> dDepth(nodeCount);
        DevBuf<int> dMaxDepth(1);

        StageClock clock;
        clock.mark();  // 0

        cudaOk(cudaMemset(dParent.p, 0xff, static_cast<size_t>(nodeCount) * sizeof(int)), "parent");
        sahInitKernel<<<blocksFor(n), kBlock>>>(n, ws.orderA.p, ws.slotA.p, ws.beginA.p,
                                                ws.endA.p, ws.nodeA.p);
        cudaOk(cudaGetLastError(), "sah init");

        SahLevelTimings lt;
        const int* finalOrder = sahBuildForest<Ops>(dSrc.p, n, n - 1, nullptr, 1, 1, ws, nodes,
                                                    dParent.p, internalCount, nullptr, &lt);
        const double binMs = lt.bin, splitMs = lt.split, partMs = lt.part;
        clock.mark();  // 1: 拓扑完成

        cudaOk(cudaMemset(dMaxDepth.p, 0, sizeof(int)), "maxDepth");
        depthKernel<<<blocksFor(nodeCount), kBlock>>>(dParent.p, nodeCount, dDepth.p, dMaxDepth.p);
        cudaOk(cudaMemcpy(&bvh.depth, dMaxDepth.p, sizeof(int), cudaMemcpyDeviceToHost), "depth");
        clock.mark();  // 2

        scatterTrisKernel<Ops><<<blocksFor(n), kBlock>>>(dSrc.p, finalOrder, n, sortedTris,
                                                         primIds);
        cudaOk(cudaGetLastError(), "scatter tris");
        clock.mark();  // 3

        double bucketMs = 0.0, refitMs = 0.0;
        finalizeTree<Ops>(bvh, n, nodes, sortedTris, dParent, dDepth, &bucketMs, &refitMs);
        clock.mark();  // 4
        cudaOk(cudaEventSynchronize(clock.ev[4]), "sync");

        if (timings) {
            timings->mortonMs = 0.0;
            timings->sortMs = 0.0;
            timings->scatterMs = clock.ms(2, 3);
            timings->treeMs = clock.ms(0, 1);
            timings->refitMs = clock.ms(3, 4);
            timings->sahBinMs = binMs;
            timings->sahSplitMs = splitMs;
            timings->sahPartMs = partMs;
            timings->totalMs = clock.ms(0, 4);
        }
    } catch (...) {
        destroyBVHGPU(bvh);
        throw;
    }

    return bvh;
}

GpuBVH buildBVHSahGPU(const Scene& scene, GpuBuildTimings* timings) {
    return buildSAH<TriOps>(scene.tris.data(), static_cast<int>(scene.tris.size()), timings);
}

GpuBVH buildPointBVHSahGPU(const PointCloud& cloud, GpuBuildTimings* timings) {
    return buildSAH<PointOps>(cloud.pts.data(), static_cast<int>(cloud.pts.size()), timings);
}

GpuBVH buildGaussianBVHSahGPU(const GaussianCloud& cloud, GpuBuildTimings* timings) {
    return buildSAH<GaussianOps>(cloud.gaussians.data(), static_cast<int>(cloud.gaussians.size()),
                                 timings);
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
    bvh.wideCount = 0;

    const int n = bvh.primCount;
    const int internal = n - 1;
    const int nodeCount = bvh.nodeCount;
    const int* depth = static_cast<const int*>(bvh.d_depth);
    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);

    // 两个数组按容量复用；拓扑更新每次 n 都变，但容量只会增长。
    if (bvh.wideIdCapacity < nodeCount) {
        if (bvh.d_wideId) cudaFree(bvh.d_wideId);
        const int cap = nodeCount + nodeCount / 8 + 1024;
        cudaOk(cudaMalloc(&bvh.d_wideId, static_cast<size_t>(cap) * sizeof(int)), "wide id");
        bvh.wideIdCapacity = cap;
    }
    int* wideId = static_cast<int*>(bvh.d_wideId);

    DevBuf<int> dFlags(internal);
    wideCountKernel<<<blocksFor(internal), kBlock>>>(depth, internal, dFlags.p);
    cudaOk(cudaGetLastError(), "wide flags");

    size_t scanBytes = 0;
    cudaOk(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, dFlags.p, wideId, internal),
           "wide scan(probe)");
    DevBuf<unsigned char> dScanTemp(scanBytes > 0 ? scanBytes : 1);
    cudaOk(cub::DeviceScan::ExclusiveSum(dScanTemp.p, scanBytes, dFlags.p, wideId, internal),
           "wide scan");

    // 前缀和只到 internal-1，总数要单独加最后一项的 flag。
    int tail[2] = {0, 0};
    cudaOk(cudaMemcpy(&tail[0], wideId + (internal - 1), sizeof(int), cudaMemcpyDeviceToHost),
           "D2H prefix");
    cudaOk(cudaMemcpy(&tail[1], dFlags.p + (internal - 1), sizeof(int), cudaMemcpyDeviceToHost),
           "D2H flag");
    const int wideCount = tail[0] + tail[1];
    if (wideCount <= 0) throw std::runtime_error("buildWideNodesGPU: no wide nodes");

    if (bvh.wideCapacity < wideCount) {
        if (bvh.d_wideNodes) cudaFree(bvh.d_wideNodes);
        const int cap = wideCount + wideCount / 8 + 1024;
        cudaOk(cudaMalloc(&bvh.d_wideNodes, static_cast<size_t>(cap) * sizeof(GpuBVH4Node)),
               "wide nodes");
        bvh.wideCapacity = cap;
    }
    bvh.wideCount = wideCount;

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
    auto free1 = [](void* p) {
        if (p) cudaOk(cudaFree(p), "destroy free");
    };
    free1(bvh.d_nodes);
    free1(bvh.d_tris);
    free1(bvh.d_primIds);
    free1(bvh.d_refitOrder);
    free1(bvh.d_stage);
    free1(bvh.d_dirtySlots);
    free1(bvh.d_dirtyTris);
    free1(bvh.d_parent);
    free1(bvh.d_depth);
    free1(bvh.d_mark);
    free1(bvh.d_dirtyOrder);
    free1(bvh.d_refitOffsets);
    free1(bvh.d_levelCounts);
    free1(bvh.d_wideNodes);
    free1(bvh.d_wideId);
    if (gTopoPool == &bvh) {
        // 回退重建正在这棵树上进行：池里的块还被本轮的 DevBuf 指着，不能释放，
        // 只清树字段，把池原样留给后续归还。
        auto keepFree = std::move(bvh.topoFree);
        const size_t keepBytes = bvh.topoFreeBytes;
        bvh = GpuBVH{};
        bvh.topoFree = std::move(keepFree);
        bvh.topoFreeBytes = keepBytes;
        return;
    }
    for (const GpuDevBlock& blk : bvh.topoFree) {
        if (blk.p) cudaOk(cudaFree(blk.p), "destroy pool");
    }
    bvh = GpuBVH{};
}

// 把旧树的换树数组全部还进池（尺寸按旧树算）并置空字段。调用方随后要么换成
// 新数组（install），要么整棵重建。d_stage / 脏区 / 宽节点不在这里处理。
template <typename Item>
void releaseTreeArrays(GpuBVH& bvh, int nOld) {
    const int nodeCount = bvh.nodeCount;
    const int internalOld = nOld - 1;
    const int binsOld = static_cast<int>(bvh.refitOffsets.size()) - 1;
    topoReleasePtr(bvh, bvh.d_nodes, static_cast<size_t>(nodeCount) * sizeof(GpuBVHNode));
    topoReleasePtr(bvh, bvh.d_tris, static_cast<size_t>(nOld) * sizeof(Item));
    topoReleasePtr(bvh, bvh.d_primIds, static_cast<size_t>(nOld) * sizeof(int));
    topoReleasePtr(bvh, bvh.d_refitOrder, static_cast<size_t>(internalOld) * sizeof(int));
    topoReleasePtr(bvh, bvh.d_parent, static_cast<size_t>(nodeCount) * sizeof(int));
    topoReleasePtr(bvh, bvh.d_depth, static_cast<size_t>(nodeCount) * sizeof(int));
    topoReleasePtr(bvh, bvh.d_mark, static_cast<size_t>(nodeCount) * sizeof(int));
    topoReleasePtr(bvh, bvh.d_dirtyOrder, static_cast<size_t>(internalOld) * sizeof(int));
    topoReleasePtr(bvh, bvh.d_refitOffsets, static_cast<size_t>(binsOld + 1) * sizeof(int));
    topoReleasePtr(bvh, bvh.d_levelCounts, static_cast<size_t>(binsOld) * sizeof(int));
    bvh.d_nodes = nullptr;
    bvh.d_tris = nullptr;
    bvh.d_primIds = nullptr;
    bvh.d_refitOrder = nullptr;
    bvh.d_parent = nullptr;
    bvh.d_depth = nullptr;
    bvh.d_mark = nullptr;
    bvh.d_dirtyOrder = nullptr;
    bvh.d_refitOffsets = nullptr;
    bvh.d_levelCounts = nullptr;
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

// 增量更新允许比"最近一次均衡构建"深多少层。追加永远落在槽尾，每轮都会给
// 那一侧加 log2(新增) 的深度；放宽太多会让查询退化，收紧则频繁整树重建。
constexpr int kTopoDepthSlack = 6;

// 拓扑更新。编辑脚本的语义写在 gpu_bvh.h 的 GpuTopoEdit 上：newScene 是幸存者
// （保序）+ 中途插入 + 末尾追加。槽序不跟随场景序：幸存者按旧槽序压缩在前，
// 新增块接在最后（先插入后追加），子树覆盖的槽区间因此仍然连续。
template <typename Ops>
static double updateTopology(GpuBVH& bvh, const std::vector<typename Ops::Item>& items,
                             const GpuTopoEdit& edit, GpuTopoUpdateTimings* timings) {
    using Item = typename Ops::Item;
    const int nOld = bvh.primCount;
    if (bvh.d_nodes == nullptr || nOld < 2) throw std::invalid_argument("updateBVHGPU: BVH not built");
    if (edit.appended < 0) throw std::invalid_argument("updateBVHGPU: appended must be >= 0");
    for (size_t i = 0; i < edit.deleted.size(); ++i) {
        const int id = edit.deleted[i];
        if (id < 0 || id >= nOld) throw std::invalid_argument("updateBVHGPU: deleted out of range");
        if (i > 0 && id <= edit.deleted[i - 1]) {
            throw std::invalid_argument("updateBVHGPU: deleted must be ascending and unique");
        }
    }
    const long long nNew = static_cast<long long>(nOld) - static_cast<long long>(edit.deleted.size()) +
                           static_cast<long long>(edit.inserted.size()) + edit.appended;
    if (nNew < 2) throw std::invalid_argument("updateBVHGPU: at least 2 primitives required");
    if (static_cast<long long>(items.size()) != nNew) {
        throw std::invalid_argument("updateBVHGPU: scene size does not match the edit");
    }
    const int nIns = static_cast<int>(edit.inserted.size());
    const long long tailBeg = nNew - edit.appended;  // 追加块从新场景的这个位置开始
    for (size_t i = 0; i < edit.inserted.size(); ++i) {
        const int id = edit.inserted[i];
        if (id < 0 || id >= tailBeg) {
            throw std::invalid_argument("updateBVHGPU: inserted index out of range");
        }
        if (i > 0 && id <= edit.inserted[i - 1]) {
            throw std::invalid_argument("updateBVHGPU: inserted must be ascending and unique");
        }
    }
    for (int id : edit.moved) {
        if (id < 0 || id >= static_cast<int>(nNew)) {
            throw std::invalid_argument("updateBVHGPU: moved index out of range");
        }
    }

    GpuTopoUpdateTimings local;
    GpuTopoUpdateTimings* t = timings ? timings : &local;
    *t = GpuTopoUpdateTimings{};
    t->deleted = static_cast<int>(edit.deleted.size());
    t->inserted = nIns;
    t->appended = edit.appended;
    t->newDepth = bvh.depth;
    if (bvh.topoBaseDepth <= 0) bvh.topoBaseDepth = bvh.depth;

    if (edit.deleted.empty() && nIns == 0 && edit.appended == 0) {
        // 纯移动：树形没变，走原来的增量 refit 快路径（上传只含改动的图元）。
        GpuRefitTimings rt;
        t->refitMs = refitImpl<Ops>(bvh, items, &edit.moved, &rt);
        t->moved = rt.updated;
        t->totalMs = t->refitMs;
        return t->totalMs;
    }

    // 从这里开始，本函数里所有 DevBuf 都从 bvh 的内存池取还，跨更新复用。
    // 纯移动在上面已经走 refit 快路径，不经过池。
    TopoPoolScope poolScope(&bvh);

    const bool wide = bvh.d_wideNodes != nullptr;
    const bool sah = bvh.sah;
    const int nNewI = static_cast<int>(nNew);
    const int delCount = static_cast<int>(edit.deleted.size());
    const int nSurv = nOld - delCount;
    const int extra = nIns + edit.appended;  // 新增块：槽尾连续的 extra 个叶子

    // 回退：整棵按同一套建树器重建。拓扑缓存的形状全变了，没有可复用的部分。
    // 旧树不 cudaFree 而是整棵还进池——重建要的正是同一批同尺寸的块，
    // 2M 上省掉一次 free + malloc（合计 15 ms 量级）。宽节点数组跨重建保留，
    // buildWideNodesGPU 按容量直接覆写；其余字段由 destroy 清空。
    auto rebuildAll = [&]() -> double {
        StageClock clock;
        clock.mark();  // 0
        void* wideNodes = bvh.d_wideNodes;
        void* wideIdArr = bvh.d_wideId;
        const int wideCap = bvh.wideCapacity;
        const int wideIdCap = bvh.wideIdCapacity;
        if (bvh.d_nodes || bvh.d_tris || bvh.d_parent) releaseTreeArrays<Item>(bvh, nOld);
        topoReleasePtr(bvh, bvh.d_stage, static_cast<size_t>(bvh.stageCount) * sizeof(Item));
        topoReleasePtr(bvh, bvh.d_dirtySlots,
                       static_cast<size_t>(bvh.dirtyCapacity) * sizeof(int));
        topoReleasePtr(bvh, bvh.d_dirtyTris,
                       static_cast<size_t>(bvh.dirtyCapacity) * sizeof(Item));
        bvh.d_stage = nullptr;
        bvh.d_dirtySlots = nullptr;
        bvh.d_dirtyTris = nullptr;
        // 宽节点数组由调用方保留复用，destroy 不能把它们释放掉。
        bvh.d_wideNodes = nullptr;
        bvh.d_wideId = nullptr;
        destroyBVHGPU(bvh);  // 池活跃：只清字段，池原样留住

        GpuBuildTimings bt;
        GpuBVH fresh =
            sah ? buildSAH<Ops>(items.data(), nNewI, &bt) : buildLBVH<Ops>(items.data(), nNewI, &bt);
        std::swap(bvh, fresh);
        // 换进来的新树自带空池；把旧池拿回来，fresh 留空析构。
        bvh.topoFree = std::move(fresh.topoFree);
        bvh.topoFreeBytes = fresh.topoFreeBytes;
        fresh.topoFreeBytes = 0;
        bvh.d_wideNodes = wideNodes;
        bvh.d_wideId = wideIdArr;
        bvh.wideCapacity = wideCap;
        bvh.wideIdCapacity = wideIdCap;
        clock.mark();  // 1
        double wideMs = 0.0;
        if (wide) {
            wideMs = buildWideNodesGPU(bvh);
            clock.mark();  // 2
        }
        cudaOk(cudaEventSynchronize(clock.ev[clock.marks - 1]), "topo sync");

        t->surgeryRoots = 1;
        t->rebuiltLeaves = nNewI;
        t->newDepth = bvh.depth;
        t->fullRebuild = true;
        t->wideMs = wideMs;
        t->totalMs = clock.ms(0, clock.marks - 1);
        t->surgeryMs = t->totalMs - wideMs;
        bvh.topoBaseDepth = bvh.depth;
        return t->totalMs;
    };

    if (bvh.d_parent == nullptr || bvh.d_depth == nullptr || bvh.d_refitOrder == nullptr) {
        return rebuildAll();
    }
    // 改动过半：局部重建的规划成本（一趟全树统计 + 一趟重编号）已经接近重建，
    // 而且最后大概率也要超过下面的重建叶子上限。
    if (delCount + extra > nOld / 2) return rebuildAll();

    const int nodeCount = bvh.nodeCount;
    const int internalOld = nOld - 1;
    const auto* oldNodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);
    const auto* oldTris = static_cast<const Item*>(bvh.d_tris);
    const int* oldParent = static_cast<const int*>(bvh.d_parent);
    const int* oldDepth = static_cast<const int*>(bvh.d_depth);
    const int* oldPrimIds = static_cast<const int*>(bvh.d_primIds);
    const int* refitOrder = static_cast<const int*>(bvh.d_refitOrder);
    int* mark = static_cast<int*>(bvh.d_mark);
    const int binsOld = static_cast<int>(bvh.refitOffsets.size()) - 1;

    StageClock clock;
    clock.mark();  // 0

    // ---- A 规划：子树大小 / 区间起点 / 删除前缀和 ----
    DevBuf<int> dSize(nodeCount), dMinSlot(nodeCount);
    subtreeInitKernel<<<blocksFor(nodeCount), kBlock>>>(dSize.p, dMinSlot.p, nOld, nodeCount);
    for (int d = binsOld - 1; d >= 0; --d) {
        const int count = bvh.refitOffsets[d + 1] - bvh.refitOffsets[d];
        if (count == 0) continue;
        subtreeMergeKernel<<<blocksFor(count), kBlock>>>(oldNodes, dSize.p, dMinSlot.p,
                                                         refitOrder + bvh.refitOffsets[d], count);
    }
    cudaOk(cudaGetLastError(), "topo subtree stats");

    DevBuf<int> dKeptFlag(nOld + 1), dKeptPrefix(nOld + 1);
    keptFlagsInitKernel<<<blocksFor(nOld + 1), kBlock>>>(nOld, dKeptFlag.p);
    DevBuf<int> dDeleted(delCount > 0 ? delCount : 1);
    if (delCount > 0) {
        cudaOk(cudaMemcpy(dDeleted.p, edit.deleted.data(), static_cast<size_t>(delCount) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "deleted H2D");
        clearKeptFlagsKernel<<<blocksFor(delCount), kBlock>>>(dDeleted.p, delCount, dKeptFlag.p);
    }
    cudaOk(cudaGetLastError(), "topo kept flags");
    std::size_t scanBytes = 0;
    cudaOk(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, dKeptFlag.p, dKeptPrefix.p, nOld + 1),
           "kept scan(probe)");
    DevBuf<unsigned char> dScan(scanBytes > 0 ? scanBytes : 1);
    cudaOk(cub::DeviceScan::ExclusiveSum(dScan.p, scanBytes, dKeptFlag.p, dKeptPrefix.p, nOld + 1),
           "kept scan");

    // 第二张前缀和：同一批删改在"叶子槽"顺序下的版本。子树区间是槽序的，删除表
    // 是场景序的，切分选择与重建区间里的幸存者计数都要用槽序这一份。
    DevBuf<int> dSlotFlag(nOld + 1), dSlotPrefix(nOld + 1);
    slotKeptFlagsKernel<<<blocksFor(nOld), kBlock>>>(oldPrimIds, dKeptPrefix.p, nOld, dSlotFlag.p);
    cudaOk(cudaMemset(dSlotFlag.p + nOld, 0, sizeof(int)), "slot flag tail");
    cudaOk(cudaGetLastError(), "slot flags");
    cudaOk(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, dSlotFlag.p, dSlotPrefix.p, nOld + 1),
           "slot scan(probe)");
    DevBuf<unsigned char> dScan1b(scanBytes > 0 ? scanBytes : 1);
    cudaOk(cub::DeviceScan::ExclusiveSum(dScan1b.p, scanBytes, dSlotFlag.p, dSlotPrefix.p,
                                         nOld + 1),
           "slot scan");

    // ---- 切分选择：每个删除点爬到一个够大的祖先，去重、排序、压掉被包含的候选 ----
    const int cabMax = delCount + (extra > 0 ? 1 : 0);
    const int budget = std::max(64, nNewI / 8);
    int minSize = budget / std::max(1, delCount);
    minSize = std::max(64, std::min(4096, minSize));
    // 新增区间至少吸收与新增块同样多的旧叶子，新子树的根才不会逐轮下沉。
    const int extraMinSize = extra > 0 ? std::max(minSize, std::min(4096, extra)) : minSize;

    DevBuf<int> dCand(cabMax), dCandFlag(cabMax + 1), dCandOff(cabMax + 1);
    DevBuf<int> dPrimSlot(nOld);
    invertPermKernel<<<blocksFor(nOld), kBlock>>>(oldPrimIds, nOld, dPrimSlot.p);
    cudaOk(cudaGetLastError(), "invert perm");
    ++bvh.refitEpoch;
    climbKernel<<<blocksFor(cabMax), kBlock>>>(dDeleted.p, dPrimSlot.p, delCount, nOld - 1, extra,
                                               oldParent, dSize.p, dMinSlot.p, dSlotPrefix.p, nOld,
                                               minSize, extraMinSize, mark, bvh.refitEpoch,
                                               dCand.p, dCandFlag.p);
    cudaOk(cudaGetLastError(), "topo climb");
    cudaOk(cudaMemset(dCandFlag.p + cabMax, 0, sizeof(int)), "cand tail");
    cudaOk(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, dCandFlag.p, dCandOff.p, cabMax + 1),
           "cand scan(probe)");
    DevBuf<unsigned char> dScan2(scanBytes > 0 ? scanBytes : 1);
    cudaOk(cub::DeviceScan::ExclusiveSum(dScan2.p, scanBytes, dCandFlag.p, dCandOff.p, cabMax + 1),
           "cand scan");
    DevBuf<int> dCandIds(cabMax);
    compactKernel<<<blocksFor(cabMax), kBlock>>>(dCand.p, dCandFlag.p, dCandOff.p, cabMax,
                                                 dCandIds.p);
    cudaOk(cudaGetLastError(), "topo compact");
    int candCount = 0;
    cudaOk(cudaMemcpy(&candCount, dCandOff.p + cabMax, sizeof(int), cudaMemcpyDeviceToHost),
           "cand count D2H");

    DevBuf<uint64_t> dCandKey(candCount), dCandKeySorted(candCount);
    DevBuf<int> dCandVal(candCount), dCandValSorted(candCount);
    candKeyKernel<<<blocksFor(candCount), kBlock>>>(dCandIds.p, candCount, dMinSlot.p, dSize.p,
                                                    dCandKey.p, dCandVal.p);
    cudaOk(cudaGetLastError(), "cand keys");
    std::size_t sortBytes = 0;
    cudaOk(cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, dCandKey.p, dCandKeySorted.p,
                                           dCandVal.p, dCandValSorted.p, candCount),
           "cand sort(probe)");
    DevBuf<unsigned char> dSortTemp(sortBytes > 0 ? sortBytes : 1);
    cudaOk(cub::DeviceRadixSort::SortPairs(dSortTemp.p, sortBytes, dCandKey.p, dCandKeySorted.p,
                                           dCandVal.p, dCandValSorted.p, candCount),
           "cand sort");

    // 压掉被祖先包住的候选：候选之间的包含关系就是祖先关系，所以一张"被包含"表
    // 加一趟爬链就够了，不需要再做一次前缀最大值扫描（cub 的 DeviceScan 只有求和）。
    DevBuf<int> dDomMark(nodeCount), dKeep(candCount + 1), dKeepOff(candCount + 1);
    cudaOk(cudaMemset(dDomMark.p, 0, static_cast<size_t>(nodeCount) * sizeof(int)), "dom clear");
    markDominatedKernel<<<blocksFor(candCount), kBlock>>>(dCandValSorted.p, candCount, oldParent,
                                                          mark, bvh.refitEpoch, dDomMark.p);
    cudaOk(cudaGetLastError(), "cand dominance");
    keepFromDomKernel<<<blocksFor(candCount), kBlock>>>(dCandValSorted.p, candCount, dDomMark.p,
                                                        dKeep.p);
    cudaOk(cudaGetLastError(), "cand keep");
    cudaOk(cudaMemset(dKeep.p + candCount, 0, sizeof(int)), "keep tail");
    cudaOk(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, dKeep.p, dKeepOff.p, candCount + 1),
           "keep scan(probe)");
    DevBuf<unsigned char> dScan4(scanBytes > 0 ? scanBytes : 1);
    cudaOk(cub::DeviceScan::ExclusiveSum(dScan4.p, scanBytes, dKeep.p, dKeepOff.p, candCount + 1),
           "keep scan");
    int cutCount = 0;
    cudaOk(cudaMemcpy(&cutCount, dKeepOff.p + candCount, sizeof(int), cudaMemcpyDeviceToHost),
           "cut count D2H");
    if (cutCount <= 0 || cutCount > 8192) return rebuildAll();

    DevBuf<int> dCutMeta(static_cast<size_t>(cutCount) * kCutFields);
    cutMetaKernel<<<blocksFor(candCount), kBlock>>>(dCandValSorted.p, dKeep.p, dKeepOff.p,
                                                    candCount, dMinSlot.p, dSize.p, oldDepth,
                                                    dSlotPrefix.p, nOld, extra, dCutMeta.p);
    cudaOk(cudaGetLastError(), "topo cut meta");
    std::vector<int> cuts(static_cast<size_t>(cutCount) * kCutFields);
    cudaOk(cudaMemcpy(cuts.data(), dCutMeta.p, cuts.size() * sizeof(int), cudaMemcpyDeviceToHost),
           "cut meta D2H");
    clock.mark();  // 1

    std::vector<int> dCutBeg(cutCount), dCutEnd(cutCount);
    std::vector<int> compactBeg(cutCount), seedBegin(cutCount), seedEnd(cutCount);
    std::vector<int> seedNode(cutCount), seedDepth(cutCount);
    long long sumOld = 0, sumNew = 0;
    int compactLen = 0;
    for (int r = 0; r < cutCount; ++r) {
        const int* c = cuts.data() + static_cast<size_t>(r) * kCutFields;
        dCutBeg[static_cast<size_t>(r)] = c[0];
        dCutEnd[static_cast<size_t>(r)] = c[1];
        compactBeg[static_cast<size_t>(r)] = compactLen;
        seedBegin[static_cast<size_t>(r)] = compactLen;      // 紧凑数组里的区间（不是叶子槽）
        seedEnd[static_cast<size_t>(r)] = compactLen + c[2];
        seedNode[static_cast<size_t>(r)] = r;
        seedDepth[static_cast<size_t>(r)] = c[3];
        sumOld += c[1] - c[0];
        sumNew += c[2];
        compactLen += c[2];
    }
    if (sumOld > nNewI / 4) return rebuildAll();
    const int want = compactLen - cutCount;

    // ---- B 暂存：幸存者按旧槽序拷贝，插入、追加和移动直接落新槽 ----
    // 新槽是槽序压缩下标（保持子树区间连续）；新场景下标由 primIds 单独记录。
    DevBuf<Item> dNewTris(nNewI);
    DevBuf<int> dNewPrimIds(nNewI);
    gatherSurvivorsKernel<Ops><<<blocksFor(nOld), kBlock>>>(oldTris, dSlotPrefix.p, nOld,
                                                            dNewTris.p);
    // 新增块：槽 [nSurv, nNew) 依次装中途插入的 nIns 个和尾部追加的 appended 个，
    // 对应场景下标记在 extraScene（inserted 升序在前、追加位置在后）。
    DevBuf<int> dExtraScene(extra > 0 ? extra : 1);
    if (extra > 0) {
        std::vector<int> extraScene(static_cast<size_t>(extra));
        std::vector<Item> extraItems(static_cast<size_t>(extra));
        for (int e = 0; e < nIns; ++e) extraScene[static_cast<size_t>(e)] = edit.inserted[e];
        for (int k = 0; k < edit.appended; ++k) {
            extraScene[static_cast<size_t>(nIns + k)] = nNewI - edit.appended + k;
        }
        for (int e = 0; e < extra; ++e) {
            extraItems[static_cast<size_t>(e)] = items[static_cast<size_t>(extraScene[e])];
        }
        cudaOk(cudaMemcpy(dExtraScene.p, extraScene.data(), static_cast<size_t>(extra) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "extra scenes H2D");
        cudaOk(cudaMemcpy(dNewTris.p + nSurv, extraItems.data(),
                          static_cast<size_t>(extra) * sizeof(Item), cudaMemcpyHostToDevice),
               "extra prims H2D");
    }
    // 中途插入才要位移表（形态是 i_j - j，非降，供核内二分）；没有插入时幸存者
    // 的新场景下标就是 keptPrefix 的值。
    DevBuf<int> dShifts;
    if (nIns > 0) {
        std::vector<int> shifts(static_cast<size_t>(nIns));
        for (int j = 0; j < nIns; ++j) {
            shifts[static_cast<size_t>(j)] = edit.inserted[static_cast<size_t>(j)] - j;
        }
        dShifts = DevBuf<int>(nIns);
        cudaOk(cudaMemcpy(dShifts.p, shifts.data(), static_cast<size_t>(nIns) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "insert shifts H2D");
    }
    remapPermKernel<<<blocksFor(nOld + extra), kBlock>>>(
        oldPrimIds, dSlotPrefix.p, dKeptPrefix.p, dShifts.p, nIns, nOld, nSurv, extra,
        dExtraScene.p, dNewPrimIds.p);
    cudaOk(cudaGetLastError(), "topo stage");
    // E 阶段还要用移动表标脏，所以这两个缓冲区留到函数末尾。
    const int mvCount = static_cast<int>(edit.moved.size());
    DevBuf<int> dScenesMv, dSceneSlotMv;
    DevBuf<Item> dMovedMv;
    if (mvCount > 0) {
        // 移动表给的是新场景下标，散写前先换成新叶子槽（primIds 一般不再是 iota）。
        std::vector<int> scenes(static_cast<size_t>(mvCount));
        std::vector<Item> moved(static_cast<size_t>(mvCount));
        for (int i = 0; i < mvCount; ++i) {
            scenes[static_cast<size_t>(i)] = edit.moved[static_cast<size_t>(i)];
            moved[static_cast<size_t>(i)] =
                items[static_cast<size_t>(edit.moved[static_cast<size_t>(i)])];
        }
        dScenesMv = DevBuf<int>(mvCount);
        dSceneSlotMv = DevBuf<int>(nNewI);
        dMovedMv = DevBuf<Item>(mvCount);
        cudaOk(cudaMemcpy(dScenesMv.p, scenes.data(), static_cast<size_t>(mvCount) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "moved scenes H2D");
        cudaOk(cudaMemcpy(dMovedMv.p, moved.data(), static_cast<size_t>(mvCount) * sizeof(Item),
                          cudaMemcpyHostToDevice),
               "moved prims H2D");
        invertPermKernel<<<blocksFor(nNewI), kBlock>>>(dNewPrimIds.p, nNewI, dSceneSlotMv.p);
        cudaOk(cudaGetLastError(), "moved invert");
        scatterItemsKernel<Ops><<<blocksFor(mvCount), kBlock>>>(dScenesMv.p, dMovedMv.p, mvCount,
                                                                dNewTris.p, dSceneSlotMv.p);
        cudaOk(cudaGetLastError(), "moved scatter");
    }

    // ---- C 森林 SAH：每个切分区间独立建一棵子树，节点写在临时区 ----
    DevBuf<GpuBVHNode> dTempNodes(want);
    DevBuf<int> dTempParent(want), dTempDepth(want);
    internalInitKernel<<<blocksFor(want), kBlock>>>(dTempNodes.p, want);
    cudaOk(cudaMemset(dTempParent.p, 0xff, static_cast<size_t>(want) * sizeof(int)), "temp parent");

    DevBuf<int> dCompactBeg(cutCount), dSlotOf(compactLen);
    cudaOk(cudaMemcpy(dCompactBeg.p, compactBeg.data(), static_cast<size_t>(cutCount) * sizeof(int),
                      cudaMemcpyHostToDevice),
           "compact beg H2D");
    // 切分区间要在 D 之前就拿到设备副本：叶子映射与重编号都要用。
    DevBuf<int> dCutBegDev(cutCount), dCutEndDev(cutCount);
    cudaOk(cudaMemcpy(dCutBegDev.p, dCutBeg.data(), static_cast<size_t>(cutCount) * sizeof(int),
                      cudaMemcpyHostToDevice),
           "cut beg H2D");
    cudaOk(cudaMemcpy(dCutEndDev.p, dCutEnd.data(), static_cast<size_t>(cutCount) * sizeof(int),
                      cudaMemcpyHostToDevice),
           "cut end H2D");
    SahWorkspace ws(compactLen);
    buildSlotOfKernel<<<blocksFor(nOld + extra), kBlock>>>(
        nOld, dSlotPrefix.p, dCutBegDev.p, dCutEndDev.p, cutCount, dCompactBeg.p, nSurv, extra,
        compactLen, dSlotOf.p, ws.orderA.p, ws.slotA.p);
    cudaOk(cudaGetLastError(), "compact slots");

    cudaOk(cudaMemcpy(ws.beginA.p, seedBegin.data(), static_cast<size_t>(cutCount) * sizeof(int),
                      cudaMemcpyHostToDevice),
           "seed begin H2D");
    cudaOk(cudaMemcpy(ws.endA.p, seedEnd.data(), static_cast<size_t>(cutCount) * sizeof(int),
                      cudaMemcpyHostToDevice),
           "seed end H2D");
    cudaOk(cudaMemcpy(ws.nodeA.p, seedNode.data(), static_cast<size_t>(cutCount) * sizeof(int),
                      cudaMemcpyHostToDevice),
           "seed node H2D");
    cudaOk(cudaMemcpy(ws.depthA.p, seedDepth.data(), static_cast<size_t>(cutCount) * sizeof(int),
                      cudaMemcpyHostToDevice),
           "seed depth H2D");

    SahLevelTimings lt;
    sahBuildForest<Ops>(dNewTris.p, compactLen, -1, dSlotOf.p, cutCount, cutCount, ws,
                        dTempNodes.p, dTempParent.p, want, dTempDepth.p, &lt);
    clock.mark();  // 2

    // ---- D 规范重编号：活着的旧内部 + 临时内部按 (深度, 来源序) 排名 ----
    const int internalNew = nNewI - 1;
    const int numSrc = internalOld + want;

    DevBuf<uint32_t> dKeys(numSrc), dKeysSorted(numSrc);
    DevBuf<int> dVals(numSrc), dValsSorted(numSrc), dIdMap(numSrc);
    relayoutKeyKernel<<<blocksFor(numSrc), kBlock>>>(dMinSlot.p, dSize.p, oldDepth, dCutBegDev.p,
                                                     dCutEndDev.p, cutCount, internalOld,
                                                     dTempDepth.p, want, dKeys.p, dVals.p);
    cudaOk(cudaGetLastError(), "relayout keys");
    cudaOk(cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, dKeys.p, dKeysSorted.p, dVals.p,
                                           dValsSorted.p, numSrc, 26, 32),
           "relayout sort(probe)");
    DevBuf<unsigned char> dSortTemp2(sortBytes > 0 ? sortBytes : 1);
    cudaOk(cub::DeviceRadixSort::SortPairs(dSortTemp2.p, sortBytes, dKeys.p, dKeysSorted.p,
                                           dVals.p, dValsSorted.p, numSrc, 26, 32),
           "relayout sort");

    uint32_t maxKey = 0;
    cudaOk(cudaMemcpy(&maxKey, dKeysSorted.p + (internalNew - 1), sizeof(uint32_t),
                      cudaMemcpyDeviceToHost),
           "max depth D2H");
    const int maxInternalDepth = static_cast<int>(maxKey >> 26);
    // 查询栈 64 层是硬约束。连续同向追加会让某条路径逐轮变深——清洁子树永远
    // 不重切，靠增量更新自己回不到浅处，所以超过基深度 kTopoDepthSlack 层就
    // 整树重建重新平衡（R4）。
    if (maxInternalDepth >= 63 ||
        maxInternalDepth + 1 > bvh.topoBaseDepth + kTopoDepthSlack) {
        return rebuildAll();
    }
    const int newDepthTotal = maxInternalDepth + 1;

    const int binsNew = newDepthTotal;
    DevBuf<int> dOffsetsDev(binsNew + 1);
    depthOffsetsPackedKernel<<<1, kDepthBins + 1>>>(dKeysSorted.p, internalNew, binsNew,
                                                    dOffsetsDev.p);
    cudaOk(cudaGetLastError(), "depth offsets");
    int offsets[kDepthBins + 1] = {};
    cudaOk(cudaMemcpy(offsets, dOffsetsDev.p, static_cast<size_t>(binsNew + 1) * sizeof(int),
                      cudaMemcpyDeviceToHost),
           "offsets D2H");
    if (offsets[binsNew] != internalNew) {
        throw std::runtime_error("topo update: depth bucketing lost nodes");
    }

    const int totalNewNodes = 2 * nNewI - 1;
    DevBuf<GpuBVHNode> dNewNodes(totalNewNodes);
    DevBuf<int> dNewParent(totalNewNodes);
    DevBuf<int> dNewDepth(totalNewNodes);
    cudaOk(cudaMemset(dNewParent.p, 0xff, static_cast<size_t>(totalNewNodes) * sizeof(int)),
           "new parent");
    rankKernel<<<blocksFor(internalNew), kBlock>>>(dValsSorted.p, internalNew, dIdMap.p);
    relayoutKernel<<<blocksFor(internalNew), kBlock>>>(
        dValsSorted.p, internalNew, oldNodes, oldDepth, internalOld, dMinSlot.p, dSize.p,
        dCutBegDev.p, dCutEndDev.p, dSlotPrefix.p, dTempNodes.p, dTempDepth.p, dIdMap.p, nNewI,
        dNewNodes.p, dNewParent.p, dNewDepth.p, cutCount);
    emitLeafKernel<Ops><<<blocksFor(nNewI), kBlock>>>(dNewTris.p, nNewI, dNewNodes.p, dNewParent.p,
                                                      dNewDepth.p);
    cudaOk(cudaGetLastError(), "relayout");
    clock.mark();  // 3


    // ---- E 合并：只合并本次新算过盒子的节点 ----
    // 临时森林的内部节点还没有盒子（建森林只定拓扑），全部标脏；切分根与移动
    // 叶子在新树里的祖先盒子变了，沿 parent 链标脏。调用方没碰过的旧内部节点
    // 直接沿用旧盒，所以合并量是"重建叶数 + 树高"，与场景大小无关。
    DevBuf<int> dMarkNew(totalNewNodes);
    cudaOk(cudaMemset(dMarkNew.p, 0, static_cast<size_t>(totalNewNodes) * sizeof(int)),
           "mark clear");
    ++bvh.refitEpoch;  // 新表的坑位全 0，这个 epoch 不可能撞上
    markTempNodesKernel<<<blocksFor(want), kBlock>>>(dIdMap.p, internalOld, want, dMarkNew.p,
                                                     bvh.refitEpoch);
    markNodeAncestorsKernel<<<blocksFor(cutCount), kBlock>>>(
        dIdMap.p + internalOld, cutCount, dNewParent.p, dMarkNew.p, bvh.refitEpoch);
    if (mvCount > 0) {
        markMovedAncestorsKernel<<<blocksFor(mvCount), kBlock>>>(
            dScenesMv.p, mvCount, dSceneSlotMv.p, dNewParent.p, nNewI, dMarkNew.p, bvh.refitEpoch);
    }
    cudaOk(cudaGetLastError(), "topo dirty");
    for (int d = binsNew - 1; d >= 0; --d) {
        const int count = offsets[d + 1] - offsets[d];
        if (count == 0) continue;
        const int blocks = (count + kBlock * kRefitUnroll - 1) / (kBlock * kRefitUnroll);
        refitDirtyKernel<<<blocks, kBlock>>>(dNewNodes.p, offsets[d], count, dMarkNew.p,
                                             bvh.refitEpoch);
    }
    cudaOk(cudaGetLastError(), "topo merge");
    clock.mark();  // 4

    DevBuf<int> dRefitOrderNew(internalNew);
    iotaKernel<<<blocksFor(internalNew), kBlock>>>(dRefitOrderNew.p, internalNew);
    DevBuf<int> dDirtyOrderNew(internalNew), dLevelCountsNew(binsNew);
    DevBuf<int> dRefitOffsetsNew(binsNew + 1);
    cudaOk(cudaMemcpy(dRefitOffsetsNew.p, offsets, static_cast<size_t>(binsNew + 1) * sizeof(int),
                      cudaMemcpyHostToDevice),
           "refit offsets H2D");

    // ---- 整体换树：到这里以前旧树一个字节都没动过，失败可以直接原地回退 ----
    // 旧数组不 cudaFree，放回内存池给下一轮复用（大小按旧树的计数算）。
    releaseTreeArrays<Item>(bvh, nOld);
    bvh.d_nodes = dNewNodes.release();
    bvh.d_tris = dNewTris.release();
    bvh.d_primIds = dNewPrimIds.release();
    bvh.d_refitOrder = dRefitOrderNew.release();
    bvh.d_parent = dNewParent.release();
    bvh.d_depth = dNewDepth.release();
    bvh.d_mark = dMarkNew.release();
    bvh.d_dirtyOrder = dDirtyOrderNew.release();
    bvh.d_refitOffsets = dRefitOffsetsNew.release();
    bvh.d_levelCounts = dLevelCountsNew.release();
    bvh.refitOffsets.assign(offsets, offsets + binsNew + 1);
    bvh.primSlot.clear();  // 反向映射变了，局部 refit 下次用到时按新 primIds 重建
    bvh.primCount = nNewI;
    bvh.nodeCount = totalNewNodes;
    bvh.depth = newDepthTotal;
    // 只在树变浅时下调基深度（删除会让整个场景变小）；变深由上面的上限管。
    if (newDepthTotal < bvh.topoBaseDepth) bvh.topoBaseDepth = newDepthTotal;

    double wideMs = 0.0;
    if (wide) {
        wideMs = buildWideNodesGPU(bvh);
        clock.mark();  // 5
    }
    cudaOk(cudaEventSynchronize(clock.ev[clock.marks - 1]), "topo sync");

    t->surgeryRoots = cutCount;
    t->rebuiltLeaves = static_cast<int>(sumOld);
    t->newDepth = bvh.depth;
    t->fullRebuild = false;
    t->wideMs = wideMs;
    t->planMs = clock.ms(0, 1);
    t->surgeryMs = clock.ms(1, 2);
    t->relayoutMs = clock.ms(2, 3);
    t->mergeMs = clock.ms(3, 4);
    t->uploadMs = 0.0;
    t->moved = static_cast<int>(edit.moved.size());
    t->totalMs = clock.ms(0, clock.marks - 1);
    return t->totalMs;
}

double updateBVHGPU(GpuBVH& bvh, const Scene& scene, const GpuTopoEdit& edit,
                    GpuTopoUpdateTimings* timings) {
    return updateTopology<TriOps>(bvh, scene.tris, edit, timings);
}

double updatePointBVHGPU(GpuBVH& bvh, const PointCloud& cloud, const GpuTopoEdit& edit,
                         GpuTopoUpdateTimings* timings) {
    return updateTopology<PointOps>(bvh, cloud.pts, edit, timings);
}

double updateGaussianBVHGPU(GpuBVH& bvh, const GaussianCloud& cloud, const GpuTopoEdit& edit,
                            GpuTopoUpdateTimings* timings) {
    return updateTopology<GaussianOps>(bvh, cloud.gaussians, edit, timings);
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
    DevBuf<int> dErrors(6);
    DevBuf<int> dHist(static_cast<size_t>(n));
    DevBuf<typename Ops::Item> dItems(static_cast<size_t>(n));
    cudaOk(cudaMemcpy(dItems.p, items.data(), static_cast<size_t>(n) * sizeof(typename Ops::Item),
                      cudaMemcpyHostToDevice),
           "validate H2D items");
    cudaOk(cudaMemset(dVisited.p, 0, static_cast<size_t>(nodeCount) * sizeof(int)), "visited");
    cudaOk(cudaMemset(dErrors.p, 0, 6 * sizeof(int)), "errors");
    cudaOk(cudaMemset(dHist.p, 0, static_cast<size_t>(n) * sizeof(int)), "prim hist");

    checkNodesKernel<Ops><<<blocksFor(nodeCount), kBlock>>>(nodes, sortedTris, n, nodeCount,
                                                            dVisited.p, dErrors.p);
    checkRefsKernel<<<blocksFor(nodeCount), kBlock>>>(dVisited.p, nodeCount, dErrors.p);
    checkPermKernel<<<blocksFor(n), kBlock>>>(static_cast<const int*>(bvh.d_primIds), n, dHist.p,
                                              dErrors.p);
    checkPermCountKernel<<<blocksFor(n), kBlock>>>(dHist.p, n, dErrors.p);
    checkPayloadKernel<Ops><<<blocksFor(n), kBlock>>>(sortedTris, dItems.p,
                                                     static_cast<const int*>(bvh.d_primIds), n,
                                                     dErrors.p);
    checkDepthKernel<<<blocksFor(nodeCount), kBlock>>>(nodes,
                                                       static_cast<const int*>(bvh.d_depth), n,
                                                       nodeCount, dErrors.p);

    int errors[6] = {};
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
    if (errors[3]) add("bad_leaf_id", errors[3]);
    if (errors[4]) add("bad_prim_map", errors[4]);
    if (errors[5]) add("bad_depth", errors[5]);
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

uint64_t treeHashGPU(const GpuBVH& bvh) {
    // 只看节点数组：盒按位、子引用、叶子区间。同样的输入两次更新必须同值，
    // 否则说明某条路径依赖了不确定的顺序（原子竞争、未初始化槽等）。
    std::vector<unsigned char> bytes(static_cast<size_t>(bvh.nodeCount) * sizeof(GpuBVHNode));
    if (!bytes.empty()) {
        cudaOk(cudaMemcpy(bytes.data(), bvh.d_nodes, bytes.size(), cudaMemcpyDeviceToHost),
               "tree hash D2H");
    }
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : bytes) {
        h ^= static_cast<uint64_t>(c);
        h *= 0x100000001b3ull;
    }
    return h;
}
