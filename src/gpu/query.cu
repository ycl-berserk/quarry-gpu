// GPU 点查询：每线程一个查询点，显式栈做深度优先，当前最优距离当剪枝阈值。
//
// 与 raycast.cu 的遍历骨架相同，差别在"进节点"的判据：射线是盒入口 t < bestT，
// 最近点是点到盒的平方距离 <= bestDist2。用 <= 而不是 < 是为了不漏掉距离并列、
// 但场景索引更小的三角形——CPU 参考实现同样用 <=，两边才能逐位对上。
//
// 这里三条路径共用同一套"每线程一个显式栈"的遍历骨架：
//   closest：一个最优值当剪枝阈值；
//   kNN：大小 k 的有序小表，第 k 近当剪枝阈值；
//   radius：固定半径的球-AABB 剔除，结果条数不定，所以走两遍遍历 + CSR。

#include "gpu/gpu_bvh.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu/gpu_node.h"
#include "gpu/morton.h"
#include "gpu/prim_ops.h"

namespace {

constexpr int kBlock = 256;
constexpr int kStack = 64;  // 每层最多压一个远端子节点，depth<=63 的树不会溢出
constexpr int kMaxKnn = 32; // 每线程定长候选表的上限，超了在主机侧报错

void cudaOk(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(err));
    }
}

__device__ unsigned long long gNodeTests = 0;
__device__ unsigned long long gTriTests = 0;
__device__ unsigned long long gOverflows = 0;

__global__ void resetStatsKernel() {
    gNodeTests = 0;
    gTriTests = 0;
    gOverflows = 0;
}

template <typename Ops, bool kStats>
__global__ __launch_bounds__(kBlock) void closestPointKernel(
    const GpuBVHNode* __restrict__ nodes, const typename Ops::Item* __restrict__ prims,
    const int* __restrict__ primIds, const Vec3* __restrict__ points, int count,
    GpuClosestResult* __restrict__ out) {
    const int idx = blockIdx.x * kBlock + threadIdx.x;
    const bool active = idx < count;

    GpuClosestResult result;
    int nodeTests = 0;
    int triTests = 0;
    bool overflow = false;

    if (active) {
        const Vec3 p = points[idx];
        float best2 = CUDART_INF_F;
        int bestPrim = -1;

        int stack[kStack];
        int sp = 0;
        stack[sp++] = 0;

        while (sp > 0) {
            const int id = stack[--sp];
            const GpuBVHNode nd = nodes[id];
            ++nodeTests;

            if (nd.left < 0) {
                const int slot = -nd.left - 1;
                ++triTests;
                const float d2 = Ops::dist2(p, prims[slot]);
                const int prim = primIds[slot];
                // 与 closerHit 等价：(距离, 场景索引) 字典序严格更小才替换。
                if (bestPrim < 0 || d2 < best2 || (d2 == best2 && prim < bestPrim)) {
                    best2 = d2;
                    bestPrim = prim;
                }
                continue;
            }

            const float dl = distanceSquaredToAABB(p, nodeBox(nodes[nd.left]));
            const float dr = distanceSquaredToAABB(p, nodeBox(nodes[nd.right]));
            // 下界等于当前最优时仍可能藏着并列但索引更小的解，所以用 <=。
            const bool okL = bestPrim < 0 || dl <= best2;
            const bool okR = bestPrim < 0 || dr <= best2;

            if (sp + (okL ? 1 : 0) + (okR ? 1 : 0) > kStack) {
                overflow = true;
                break;
            }
            // 近的孩子后压先出：先把 best 收紧，远的那支更可能被整枝剪掉。
            if (okL && okR) {
                if (dl <= dr) {
                    stack[sp++] = nd.right;
                    stack[sp++] = nd.left;
                } else {
                    stack[sp++] = nd.left;
                    stack[sp++] = nd.right;
                }
            } else if (okL) {
                stack[sp++] = nd.left;
            } else if (okR) {
                stack[sp++] = nd.right;
            }
        }

        if (bestPrim >= 0) {
            result.dist2Bits = __float_as_uint(best2);
            result.triangle = bestPrim;
        }
        out[idx] = result;
    }

    if (kStats) {
        const unsigned mask = __ballot_sync(0xffffffffu, active);
        const int sumNodes = __reduce_add_sync(mask, nodeTests);
        const int sumTris = __reduce_add_sync(mask, triTests);
        if ((threadIdx.x & 31) == 0) {
            if (sumNodes) atomicAdd(&gNodeTests, static_cast<unsigned long long>(sumNodes));
            if (sumTris) atomicAdd(&gTriTests, static_cast<unsigned long long>(sumTris));
            if (overflow) atomicAdd(&gOverflows, 1ull);
        }
    }
}

GpuQueryStats gLastStats;

// kNN 的候选表项：只留排序需要的两个字段，32 项也只占 256B 本地内存。
struct KnnEntry {
    float dist2;
    int tri;
};

__device__ __forceinline__ bool knnBetter(const KnnEntry& a, const KnnEntry& b) {
    return a.dist2 < b.dist2 || (a.dist2 == b.dist2 && a.tri < b.tri);
}

// 和 CPU 的大顶堆同语义：满了以后只有严格优于第 k 近的候选能进表。
// 表内保持 (dist2, tri) 升序，插入是 O(k) 的搬移，k<=32 且命中很少，成本可忽略。
__device__ __forceinline__ void knnInsert(KnnEntry* best, int& n, int k, const KnnEntry& h) {
    int pos = n;
    while (pos > 0 && knnBetter(h, best[pos - 1])) --pos;
    if (n == k && pos == k) return;
    if (n < k) ++n;
    for (int j = n - 1; j > pos; --j) best[j] = best[j - 1];
    best[pos] = h;
}

template <typename Ops, bool kStats>
__global__ __launch_bounds__(kBlock) void kNearestKernel(
    const GpuBVHNode* __restrict__ nodes, const typename Ops::Item* __restrict__ prims,
    const int* __restrict__ primIds, const Vec3* __restrict__ points, int count, int k,
    GpuClosestResult* __restrict__ out) {
    const int idx = blockIdx.x * kBlock + threadIdx.x;
    const bool active = idx < count;

    int nodeTests = 0;
    int triTests = 0;
    bool overflow = false;

    if (active) {
        const Vec3 p = points[idx];
        KnnEntry best[kMaxKnn];
        int n = 0;

        int stack[kStack];
        int sp = 0;
        stack[sp++] = 0;

        while (sp > 0) {
            const int id = stack[--sp];
            const GpuBVHNode nd = nodes[id];
            ++nodeTests;

            if (nd.left < 0) {
                const int slot = -nd.left - 1;
                ++triTests;
                const KnnEntry e{Ops::dist2(p, prims[slot]), primIds[slot]};
                if (n == k && !knnBetter(e, best[k - 1])) continue;
                knnInsert(best, n, k, e);
                continue;
            }

            const float dl = distanceSquaredToAABB(p, nodeBox(nodes[nd.left]));
            const float dr = distanceSquaredToAABB(p, nodeBox(nodes[nd.right]));
            const bool full = n == k;
            const bool okL = !full || dl <= best[k - 1].dist2;
            const bool okR = !full || dr <= best[k - 1].dist2;

            if (sp + (okL ? 1 : 0) + (okR ? 1 : 0) > kStack) {
                overflow = true;
                break;
            }
            if (okL && okR) {
                if (dl <= dr) {
                    stack[sp++] = nd.right;
                    stack[sp++] = nd.left;
                } else {
                    stack[sp++] = nd.left;
                    stack[sp++] = nd.right;
                }
            } else if (okL) {
                stack[sp++] = nd.left;
            } else if (okR) {
                stack[sp++] = nd.right;
            }
        }

        GpuClosestResult* dst = out + static_cast<size_t>(idx) * k;
        for (int i = 0; i < n; ++i) {
            dst[i].dist2Bits = __float_as_uint(best[i].dist2);
            dst[i].triangle = best[i].tri;
        }
        for (int i = n; i < k; ++i) {
            dst[i].dist2Bits = 0;
            dst[i].triangle = -1;
        }
    }

    if (kStats) {
        const unsigned mask = __ballot_sync(0xffffffffu, active);
        const int sumNodes = __reduce_add_sync(mask, nodeTests);
        const int sumTris = __reduce_add_sync(mask, triTests);
        if ((threadIdx.x & 31) == 0) {
            if (sumNodes) atomicAdd(&gNodeTests, static_cast<unsigned long long>(sumNodes));
            if (sumTris) atomicAdd(&gTriTests, static_cast<unsigned long long>(sumTris));
            if (overflow) atomicAdd(&gOverflows, 1ull);
        }
    }
}

// 半径查询的两遍遍历：第一遍只数条数，第二遍按前缀和写进 CSR 段。
// 遍历规则与 CPU 的 radiusBVH 一致：AABB 下界 > r2 的节点不进，三角形
// dist2 > r2 的丢掉（含相等），所以两边的结果集合逐条相同。
__device__ __forceinline__ bool radiusVisit(const Vec3& p, const GpuBVHNode& nd, float r2) {
    return distanceSquaredToAABB(p, nodeBox(nd)) <= r2;
}

template <typename Ops, bool kStats>
__global__ __launch_bounds__(kBlock) void radiusCountKernel(
    const GpuBVHNode* __restrict__ nodes, const typename Ops::Item* __restrict__ prims,
    const Vec3* __restrict__ points, int count, float r2, int* __restrict__ counts) {
    const int idx = blockIdx.x * kBlock + threadIdx.x;
    const bool active = idx < count;

    int nodeTests = 0;
    int triTests = 0;
    bool overflow = false;

    if (active) {
        const Vec3 p = points[idx];
        int found = 0;

        int stack[kStack];
        int sp = 0;
        stack[sp++] = 0;

        while (sp > 0) {
            const int id = stack[--sp];
            const GpuBVHNode nd = nodes[id];
            if (!radiusVisit(p, nd, r2)) continue;
            ++nodeTests;

            if (nd.left < 0) {
                const int slot = -nd.left - 1;
                ++triTests;
                if (Ops::dist2(p, prims[slot]) <= r2) ++found;
                continue;
            }

            if (sp + 2 > kStack) {
                overflow = true;
                break;
            }
            stack[sp++] = nd.right;
            stack[sp++] = nd.left;
        }
        counts[idx] = found;
    }

    if (kStats) {
        const unsigned mask = __ballot_sync(0xffffffffu, active);
        const int sumNodes = __reduce_add_sync(mask, nodeTests);
        const int sumTris = __reduce_add_sync(mask, triTests);
        if ((threadIdx.x & 31) == 0) {
            if (sumNodes) atomicAdd(&gNodeTests, static_cast<unsigned long long>(sumNodes));
            if (sumTris) atomicAdd(&gTriTests, static_cast<unsigned long long>(sumTris));
            if (overflow) atomicAdd(&gOverflows, 1ull);
        }
    }
}

template <typename Ops>
__global__ __launch_bounds__(kBlock) void radiusFillKernel(
    const GpuBVHNode* __restrict__ nodes, const typename Ops::Item* __restrict__ prims,
    const int* __restrict__ primIds, const Vec3* __restrict__ points, int count, float r2,
    const int* __restrict__ offsets, unsigned long long* __restrict__ keys) {
    const int idx = blockIdx.x * kBlock + threadIdx.x;
    if (idx >= count) return;

    const Vec3 p = points[idx];
    int write = offsets[idx];

    int stack[kStack];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const int id = stack[--sp];
        const GpuBVHNode nd = nodes[id];
        if (!radiusVisit(p, nd, r2)) continue;

        if (nd.left < 0) {
            const int slot = -nd.left - 1;
            const float d2 = Ops::dist2(p, prims[slot]);
            if (d2 > r2) continue;
            const int prim = primIds[slot];
            // 非负浮点的位模式就是升序整数，拼上场景索引后按 64 位键升序排
            // 就等价于 closerHit 的 (dist2, 索引) 字典序。图元索引随键一起排，
            // 不用再维护第二个数组。
            keys[write] = (static_cast<unsigned long long>(__float_as_uint(d2)) << 32) |
                          static_cast<unsigned int>(prim);
            ++write;
        } else {
            // 和计数那一遍一样在溢出处停下，否则写指针会越出本查询点的段。
            if (sp + 2 > kStack) break;
            stack[sp++] = nd.right;
            stack[sp++] = nd.left;
        }
    }
}

// 半径查询每段长度通常只有几条到几十条，但长尾能有几百条。短段用一个 warp
// 在共享内存里做双调排序：cap 取到 2 的幂，补 UINT64_MAX 到尾部，随段长缩放
// 网络层数（m=10 只跑 16 元网络）。超过 256 条的段留给插入排序兜底。
constexpr int kRadiusSortCap = 256;
constexpr int kWarpsPerBlock = kBlock / 32;

__global__ __launch_bounds__(kBlock) void radiusSortWarpKernel(const int* __restrict__ offsets,
                                                               unsigned long long* __restrict__ keys,
                                                               int count) {
    extern __shared__ unsigned long long smem[];
    const int lane = threadIdx.x & 31;
    const int warpInBlock = threadIdx.x >> 5;
    const int qid = blockIdx.x * kWarpsPerBlock + warpInBlock;
    if (qid >= count) return;

    const int begin = offsets[qid];
    const int m = offsets[qid + 1] - begin;
    if (m > kRadiusSortCap) return;

    int cap = 1;
    while (cap < m) cap <<= 1;

    unsigned long long* buf = smem + warpInBlock * kRadiusSortCap;
    for (int i = lane; i < cap; i += 32) buf[i] = (i < m) ? keys[begin + i] : ~0ull;
    __syncwarp();

    for (int size = 2; size <= cap; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < cap; i += 32) {
                const int partner = i ^ stride;
                if (partner < i) continue;  // 每对只处理一次
                const unsigned long long a = buf[i];
                const unsigned long long b = buf[partner];
                const bool ascending = (i & size) == 0;
                if (ascending ? (a > b) : (a < b)) {
                    buf[i] = b;
                    buf[partner] = a;
                }
            }
            __syncwarp();
        }
    }

    for (int i = lane; i < m; i += 32) keys[begin + i] = buf[i];
}

}  // namespace

// ---------------------------------------------------------------------------
// 查询调度：Morton 序重排查询点。每条查询相互独立，重排只改变执行顺序——
// 相邻下标的查询点落在同一块空间，同一 warp 的遍历路径重合，节点/叶子从
// HBM 随机读变成 L1/L2 命中。结果按原下标散回，逐点输出与顺序无关。
// ---------------------------------------------------------------------------
namespace {

struct QuerySortScratch {
    uint32_t* keysIn = nullptr;
    uint32_t* keysOut = nullptr;
    int* orderIn = nullptr;
    int* orderOut = nullptr;
    Vec3* sortedPoints = nullptr;
    GpuClosestResult* sortedResults = nullptr;
    unsigned char* temp = nullptr;
    size_t tempBytes = 0;
};

void destroyQuerySortScratch(void* p) {
    if (!p) return;
    auto* s = static_cast<QuerySortScratch*>(p);
    if (s->keysIn) cudaFree(s->keysIn);
    if (s->keysOut) cudaFree(s->keysOut);
    if (s->orderIn) cudaFree(s->orderIn);
    if (s->orderOut) cudaFree(s->orderOut);
    if (s->sortedPoints) cudaFree(s->sortedPoints);
    if (s->sortedResults) cudaFree(s->sortedResults);
    if (s->temp) cudaFree(s->temp);
    delete s;
}

// 容量跟查询点数绑定，首次请求 Morton 序时才分配。
QuerySortScratch* ensureQuerySortScratch(GpuPointBuffer& buf) {
    if (buf.d_sortScratch) return static_cast<QuerySortScratch*>(buf.d_sortScratch);

    auto* s = new QuerySortScratch{};
    try {
        auto alloc = [](auto*& p, size_t n) {
            cudaOk(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(*p)), "query sort scratch");
        };
        const size_t n = static_cast<size_t>(buf.count);
        alloc(s->keysIn, n);
        alloc(s->keysOut, n);
        alloc(s->orderIn, n);
        alloc(s->orderOut, n);
        alloc(s->sortedPoints, n);
        cub::DeviceRadixSort::SortPairs(nullptr, s->tempBytes, s->keysIn, s->keysOut, s->orderIn,
                                        s->orderOut, buf.count, 0, 30);
        alloc(s->temp, s->tempBytes > 0 ? s->tempBytes : 1);
    } catch (...) {
        destroyQuerySortScratch(s);
        throw;
    }
    buf.d_sortScratch = s;
    return s;
}

// 半径查询用不到排序后的结果缓冲（CSR 有自己的散射路径），按需分配。
GpuClosestResult* ensureSortedResults(GpuPointBuffer& buf, QuerySortScratch* s) {
    if (!s->sortedResults) {
        cudaOk(cudaMalloc(reinterpret_cast<void**>(&s->sortedResults),
                          static_cast<size_t>(buf.count) * static_cast<size_t>(buf.stride) *
                              sizeof(GpuClosestResult)),
               "query sort results");
    }
    return s->sortedResults;
}

__global__ void mortonPointKeysKernel(const Vec3* __restrict__ points, int count, AABB bounds,
                                      uint32_t* __restrict__ keys, int* __restrict__ order) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count) return;
    const Vec3 p = points[i];
    // 量化域是查询点自己的包围盒：查询点可能整片落在场景盒外或只覆盖一小块，
    // 拿场景盒做域会把这些点钳到同一格里，排序就失去意义。
    keys[i] = morton3D10(p.x, p.y, p.z, bounds.mn.x, bounds.mx.x, bounds.mn.y, bounds.mx.y,
                         bounds.mn.z, bounds.mx.z);
    order[i] = i;
}

__global__ void gatherPointsKernel(const Vec3* __restrict__ points, const int* __restrict__ order,
                                   int count, Vec3* __restrict__ out) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < count) out[i] = points[order[i]];
}

__global__ void scatterQueryResultsKernel(const GpuClosestResult* __restrict__ sortedOut,
                                          const int* __restrict__ order, int count, int stride,
                                          GpuClosestResult* __restrict__ out) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count) return;
    const GpuClosestResult* src = sortedOut + static_cast<size_t>(i) * stride;
    GpuClosestResult* dst = out + static_cast<size_t>(order[i]) * stride;
    for (int j = 0; j < stride; ++j) dst[j] = src[j];
}

// 半径的 CSR 是变长段：先把每段长度散回原下标，再做一次前缀和，最后逐段把
// 已排好序的键拷进原序的偏移。
__global__ void scatterCountsKernel(const int* __restrict__ counts, const int* __restrict__ order,
                                    int count, int* __restrict__ out) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < count) out[order[i]] = counts[i];
}

__global__ void scatterRadiusSegmentsKernel(const uint64_t* __restrict__ keysSorted,
                                            const int* __restrict__ offsetsSorted,
                                            const int* __restrict__ offsets,
                                            const int* __restrict__ order, int count,
                                            uint64_t* __restrict__ out) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count) return;
    const int begin = offsetsSorted[i];
    const int len = offsetsSorted[i + 1] - begin;
    uint64_t* dst = out + offsets[order[i]];
    for (int j = 0; j < len; ++j) dst[j] = keysSorted[begin + j];
}

// 建码 → 排序 → 收集查询点。返回排序后的点数组；计时由调用方的 event 覆盖。
const Vec3* reorderQueryPoints(GpuPointBuffer& buf, QuerySortScratch* s, int blocks) {
    const auto* points = static_cast<const Vec3*>(buf.d_points);
    mortonPointKeysKernel<<<blocks, kBlock>>>(points, buf.count, buf.bounds, s->keysIn,
                                              s->orderIn);
    cudaOk(cudaGetLastError(), "morton points");

    cub::DeviceRadixSort::SortPairs(s->temp, s->tempBytes, s->keysIn, s->keysOut, s->orderIn,
                                    s->orderOut, buf.count, 0, 30);
    cudaOk(cudaGetLastError(), "sort points");

    gatherPointsKernel<<<blocks, kBlock>>>(points, s->orderOut, buf.count, s->sortedPoints);
    cudaOk(cudaGetLastError(), "gather points");
    return s->sortedPoints;
}

}  // namespace

GpuPointBuffer uploadPointsGPU(const std::vector<Vec3>& points, int resultsPerPoint) {
    GpuPointBuffer buf;
    buf.count = static_cast<int>(points.size());
    buf.stride = resultsPerPoint > 0 ? resultsPerPoint : 1;
    if (buf.count == 0) return buf;

    buf.bounds.mn = points[0];
    buf.bounds.mx = points[0];
    for (const Vec3& p : points) {
        buf.bounds.mn.x = std::min(buf.bounds.mn.x, p.x);
        buf.bounds.mn.y = std::min(buf.bounds.mn.y, p.y);
        buf.bounds.mn.z = std::min(buf.bounds.mn.z, p.z);
        buf.bounds.mx.x = std::max(buf.bounds.mx.x, p.x);
        buf.bounds.mx.y = std::max(buf.bounds.mx.y, p.y);
        buf.bounds.mx.z = std::max(buf.bounds.mx.z, p.z);
    }

    cudaOk(cudaMalloc(&buf.d_points, static_cast<size_t>(buf.count) * sizeof(Vec3)), "points");
    cudaOk(cudaMalloc(&buf.d_results,
                      static_cast<size_t>(buf.count) * buf.stride * sizeof(GpuClosestResult)),
           "query results");
    cudaOk(cudaMemcpy(buf.d_points, points.data(), static_cast<size_t>(buf.count) * sizeof(Vec3),
                      cudaMemcpyHostToDevice),
           "H2D points");
    return buf;
}

void destroyPointBufferGPU(GpuPointBuffer& buf) {
    if (buf.d_points) cudaFree(buf.d_points);
    if (buf.d_results) cudaFree(buf.d_results);
    destroyQuerySortScratch(buf.d_sortScratch);
    buf = GpuPointBuffer{};
}

template <typename Ops>
double closestImpl(const GpuBVH& bvh, GpuPointBuffer& buf, bool collectStats, QueryOrder order) {
    if (buf.count <= 0 || bvh.nodeCount <= 0) return 0.0;

    gLastStats = GpuQueryStats{};
    if (collectStats) {
        resetStatsKernel<<<1, 1>>>();
        cudaOk(cudaGetLastError(), "reset query stats");
    }

    cudaEvent_t start, mark, mark2, stop;
    cudaOk(cudaEventCreate(&start), "event");
    cudaOk(cudaEventCreate(&mark), "event");
    cudaOk(cudaEventCreate(&mark2), "event");
    cudaOk(cudaEventCreate(&stop), "event");
    cudaOk(cudaEventRecord(start), "record start");

    const int blocks = (buf.count + kBlock - 1) / kBlock;
    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);
    const auto* prims = static_cast<const typename Ops::Item*>(bvh.d_tris);
    const auto* primIds = static_cast<const int*>(bvh.d_primIds);
    auto* out = static_cast<GpuClosestResult*>(buf.d_results);

    QuerySortScratch* sort = nullptr;
    const Vec3* points = static_cast<const Vec3*>(buf.d_points);
    GpuClosestResult* sortedOut = out;
    if (order == QueryOrder::Morton) {
        sort = ensureQuerySortScratch(buf);
        points = reorderQueryPoints(buf, sort, blocks);
        sortedOut = ensureSortedResults(buf, sort);
        cudaOk(cudaEventRecord(mark), "record mark");
    }

    if (collectStats) {
        closestPointKernel<Ops, true><<<blocks, kBlock>>>(nodes, prims, primIds, points, buf.count,
                                                          sortedOut);
    } else {
        closestPointKernel<Ops, false><<<blocks, kBlock>>>(nodes, prims, primIds, points, buf.count,
                                                           sortedOut);
    }
    cudaOk(cudaGetLastError(), "closest point");
    cudaOk(cudaEventRecord(mark2), "record mark2");

    if (sort) {
        scatterQueryResultsKernel<<<blocks, kBlock>>>(sortedOut, sort->orderOut, buf.count,
                                                      buf.stride, out);
        cudaOk(cudaGetLastError(), "scatter query results");
    }

    cudaOk(cudaEventRecord(stop), "record stop");
    cudaOk(cudaEventSynchronize(stop), "sync");
    float ms = 0.0f, sortMs = 0.0f, scatterMs = 0.0f;
    cudaOk(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    if (sort) {
        cudaOk(cudaEventElapsedTime(&sortMs, start, mark), "sort elapsed");
        cudaOk(cudaEventElapsedTime(&scatterMs, mark2, stop), "scatter elapsed");
        gLastStats.sortMs = static_cast<double>(sortMs + scatterMs);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(mark);
    cudaEventDestroy(mark2);
    cudaEventDestroy(stop);

    if (collectStats) {
        cudaOk(cudaMemcpyFromSymbol(&gLastStats.nodeTests, gNodeTests, sizeof(gNodeTests)),
               "copy nodeTests");
        cudaOk(cudaMemcpyFromSymbol(&gLastStats.triTests, gTriTests, sizeof(gTriTests)),
               "copy triTests");
        cudaOk(cudaMemcpyFromSymbol(&gLastStats.stackOverflows, gOverflows, sizeof(gOverflows)),
               "copy overflows");
    }
    return static_cast<double>(ms);
}

double closestPointGPU(const GpuBVH& bvh, GpuPointBuffer& buf, bool collectStats,
                       QueryOrder order) {
    return closestImpl<TriOps>(bvh, buf, collectStats, order);
}

// 点云最近点：同一条遍历，只是叶子载荷换成 Vec3、距离换成点距。
// 输出的 triangle 字段装的是点索引。
double closestVertexGPU(const GpuBVH& bvh, GpuPointBuffer& buf, bool collectStats,
                        QueryOrder order) {
    return closestImpl<PointOps>(bvh, buf, collectStats, order);
}

// 高斯最近点：载荷换成 Gaussian，叶子距离是到椭球实体的平方距离。
double closestGaussianGPU(const GpuBVH& bvh, GpuPointBuffer& buf, bool collectStats,
                          QueryOrder order) {
    return closestImpl<GaussianOps>(bvh, buf, collectStats, order);
}

template <typename Ops>
double kNearestImpl(const GpuBVH& bvh, GpuPointBuffer& buf, int k, bool collectStats,
                    QueryOrder order) {
    if (buf.count <= 0 || bvh.nodeCount <= 0) return 0.0;
    if (k < 1 || k > kMaxKnn) {
        throw std::runtime_error("kNearestGPU: k must be in [1, 32]");
    }
    if (buf.stride < k) {
        throw std::runtime_error("kNearestGPU: result buffer was uploaded with a smaller stride");
    }

    gLastStats = GpuQueryStats{};
    if (collectStats) {
        resetStatsKernel<<<1, 1>>>();
        cudaOk(cudaGetLastError(), "reset query stats");
    }

    cudaEvent_t start, mark, mark2, stop;
    cudaOk(cudaEventCreate(&start), "event");
    cudaOk(cudaEventCreate(&mark), "event");
    cudaOk(cudaEventCreate(&mark2), "event");
    cudaOk(cudaEventCreate(&stop), "event");
    cudaOk(cudaEventRecord(start), "record start");

    const int blocks = (buf.count + kBlock - 1) / kBlock;
    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);
    const auto* prims = static_cast<const typename Ops::Item*>(bvh.d_tris);
    const auto* primIds = static_cast<const int*>(bvh.d_primIds);
    auto* out = static_cast<GpuClosestResult*>(buf.d_results);

    QuerySortScratch* sort = nullptr;
    const Vec3* points = static_cast<const Vec3*>(buf.d_points);
    GpuClosestResult* sortedOut = out;
    if (order == QueryOrder::Morton) {
        sort = ensureQuerySortScratch(buf);
        points = reorderQueryPoints(buf, sort, blocks);
        sortedOut = ensureSortedResults(buf, sort);
        cudaOk(cudaEventRecord(mark), "record mark");
    }

    if (collectStats) {
        kNearestKernel<Ops, true><<<blocks, kBlock>>>(nodes, prims, primIds, points, buf.count, k,
                                                      sortedOut);
    } else {
        kNearestKernel<Ops, false><<<blocks, kBlock>>>(nodes, prims, primIds, points, buf.count, k,
                                                       sortedOut);
    }
    cudaOk(cudaGetLastError(), "k nearest");
    cudaOk(cudaEventRecord(mark2), "record mark2");

    if (sort) {
        scatterQueryResultsKernel<<<blocks, kBlock>>>(sortedOut, sort->orderOut, buf.count,
                                                      buf.stride, out);
        cudaOk(cudaGetLastError(), "scatter query results");
    }

    cudaOk(cudaEventRecord(stop), "record stop");
    cudaOk(cudaEventSynchronize(stop), "sync");
    float ms = 0.0f, sortMs = 0.0f, scatterMs = 0.0f;
    cudaOk(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    if (sort) {
        cudaOk(cudaEventElapsedTime(&sortMs, start, mark), "sort elapsed");
        cudaOk(cudaEventElapsedTime(&scatterMs, mark2, stop), "scatter elapsed");
        gLastStats.sortMs = static_cast<double>(sortMs + scatterMs);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(mark);
    cudaEventDestroy(mark2);
    cudaEventDestroy(stop);

    if (collectStats) {
        cudaOk(cudaMemcpyFromSymbol(&gLastStats.nodeTests, gNodeTests, sizeof(gNodeTests)),
               "copy nodeTests");
        cudaOk(cudaMemcpyFromSymbol(&gLastStats.triTests, gTriTests, sizeof(gTriTests)),
               "copy triTests");
        cudaOk(cudaMemcpyFromSymbol(&gLastStats.stackOverflows, gOverflows, sizeof(gOverflows)),
               "copy overflows");
    }
    return static_cast<double>(ms);
}

double kNearestGPU(const GpuBVH& bvh, GpuPointBuffer& buf, int k, bool collectStats,
                   QueryOrder order) {
    return kNearestImpl<TriOps>(bvh, buf, k, collectStats, order);
}

double kNearestVertexGPU(const GpuBVH& bvh, GpuPointBuffer& buf, int k, bool collectStats,
                         QueryOrder order) {
    return kNearestImpl<PointOps>(bvh, buf, k, collectStats, order);
}

double kNearestGaussianGPU(const GpuBVH& bvh, GpuPointBuffer& buf, int k, bool collectStats,
                           QueryOrder order) {
    return kNearestImpl<GaussianOps>(bvh, buf, k, collectStats, order);
}

template <typename Ops>
double radiusQueryImpl(const GpuBVH& bvh, GpuPointBuffer& buf, float radius, GpuRadiusBuffer& out,
                       bool collectStats, QueryOrder order) {
    out = GpuRadiusBuffer{};
    if (buf.count <= 0 || bvh.nodeCount <= 0) return 0.0;

    const float r2 = radius * radius;
    const int n = buf.count;

    int* dCounts = nullptr;
    int* dOffsets = nullptr;
    cudaOk(cudaMalloc(&dCounts, static_cast<size_t>(n) * sizeof(int)), "radius counts");
    cudaOk(cudaMalloc(&dOffsets, static_cast<size_t>(n + 1) * sizeof(int)), "radius offsets");

    size_t scanBytes = 0;
    cudaOk(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, dCounts, dOffsets, n),
           "scan probe");
    void* dScanTemp = nullptr;
    cudaOk(cudaMalloc(&dScanTemp, scanBytes), "scan temp");

    gLastStats = GpuQueryStats{};
    if (collectStats) {
        resetStatsKernel<<<1, 1>>>();
        cudaOk(cudaGetLastError(), "reset query stats");
    }

    cudaEvent_t start, mark, mark2, stop;
    cudaOk(cudaEventCreate(&start), "event");
    cudaOk(cudaEventCreate(&mark), "event");
    cudaOk(cudaEventCreate(&mark2), "event");
    cudaOk(cudaEventCreate(&stop), "event");
    cudaOk(cudaEventRecord(start), "record start");

    const int blocks = (n + kBlock - 1) / kBlock;
    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);
    const auto* prims = static_cast<const typename Ops::Item*>(bvh.d_tris);
    const auto* primIds = static_cast<const int*>(bvh.d_primIds);

    QuerySortScratch* sort = nullptr;
    const Vec3* points = static_cast<const Vec3*>(buf.d_points);
    if (order == QueryOrder::Morton) {
        sort = ensureQuerySortScratch(buf);
        points = reorderQueryPoints(buf, sort, blocks);
        cudaOk(cudaEventRecord(mark), "record mark");
    }

    if (collectStats) {
        radiusCountKernel<Ops, true><<<blocks, kBlock>>>(nodes, prims, points, n, r2, dCounts);
    } else {
        radiusCountKernel<Ops, false><<<blocks, kBlock>>>(nodes, prims, points, n, r2, dCounts);
    }
    cudaOk(cudaGetLastError(), "radius count");

    cudaOk(cub::DeviceScan::ExclusiveSum(dScanTemp, scanBytes, dCounts, dOffsets, n), "scan");

    // 扫描结果整段拷回：主机需要 offsets 的末项来算 total，也需要最长段来决定
    // 排序策略。CSR 的长度可变，缓冲没法提前静态分配，这次同步省不掉。
    std::vector<int> hOffsets(static_cast<size_t>(n) + 1, 0);
    cudaOk(cudaMemcpy(hOffsets.data(), dOffsets, static_cast<size_t>(n) * sizeof(int),
                      cudaMemcpyDeviceToHost),
           "offsets");
    int lastCount = 0;
    cudaOk(cudaMemcpy(&lastCount, dCounts + (n - 1), sizeof(int), cudaMemcpyDeviceToHost),
           "tail count");
    const int total = hOffsets[n - 1] + lastCount;
    hOffsets[n] = total;
    cudaOk(cudaMemcpy(dOffsets + n, &total, sizeof(int), cudaMemcpyHostToDevice), "tail total");

    int maxSeg = 0;
    for (int i = 0; i < n; ++i) {
        const int m = hOffsets[i + 1] - hOffsets[i];
        if (m > maxSeg) maxSeg = m;
    }

    uint64_t* dKeys = nullptr;
    if (total > 0) {
        const size_t bytes = static_cast<size_t>(total);
        cudaOk(cudaMalloc(&dKeys, bytes * sizeof(uint64_t)), "radius keys");

        radiusFillKernel<Ops><<<blocks, kBlock>>>(nodes, prims, primIds, points, n, r2, dOffsets,
                                                  dKeys);
        cudaOk(cudaGetLastError(), "radius fill");

        if (maxSeg <= kRadiusSortCap) {
            const int warpBlocks = (n + kWarpsPerBlock - 1) / kWarpsPerBlock;
            radiusSortWarpKernel<<<warpBlocks, kBlock,
                                   kWarpsPerBlock * kRadiusSortCap * sizeof(uint64_t)>>>(
                dOffsets, dKeys, n);
            cudaOk(cudaGetLastError(), "radius sort");
        } else {
            // 有超过 256 条的长段（半径很大或点云极密）时退回 CUB 分段排序：
            // 它在两万个短段上要 13 ms，但对任意段长都正确，作为罕见路径足够。
            uint64_t* dKeysSorted = nullptr;
            cudaOk(cudaMalloc(&dKeysSorted, bytes * sizeof(uint64_t)), "radius keys sorted");
            size_t sortBytes = 0;
            cudaOk(cub::DeviceSegmentedRadixSort::SortKeys(
                       nullptr, sortBytes, dKeys, dKeysSorted, total, n, dOffsets, dOffsets + 1, 0,
                       64),
                   "segmented sort probe");
            void* dSortTemp = nullptr;
            cudaOk(cudaMalloc(&dSortTemp, sortBytes), "sort temp");
            cudaOk(cub::DeviceSegmentedRadixSort::SortKeys(
                       dSortTemp, sortBytes, dKeys, dKeysSorted, total, n, dOffsets, dOffsets + 1,
                       0, 64),
                   "segmented sort");
            cudaFree(dKeys);
            cudaFree(dSortTemp);
            dKeys = dKeysSorted;
        }
    }

    if (sort) {
        // 段长散回原序 → 前缀和 → 逐段拷键。CSR 是变长的，这一步没法像
        // closest/kNN 那样按固定 stride 一次散射。
        int* dCountsOrig = nullptr;
        int* dOffsetsOrig = nullptr;
        cudaOk(cudaMalloc(&dCountsOrig, static_cast<size_t>(n) * sizeof(int)),
               "radius counts orig");
        cudaOk(cudaMalloc(&dOffsetsOrig, static_cast<size_t>(n + 1) * sizeof(int)),
               "radius offsets orig");
        scatterCountsKernel<<<blocks, kBlock>>>(dCounts, sort->orderOut, n, dCountsOrig);
        cudaOk(cudaGetLastError(), "scatter radius counts");
        cudaOk(cub::DeviceScan::ExclusiveSum(dScanTemp, scanBytes, dCountsOrig, dOffsetsOrig, n),
               "scan orig");
        cudaOk(cudaMemcpy(dOffsetsOrig + n, &total, sizeof(int), cudaMemcpyHostToDevice),
               "tail total orig");

        uint64_t* dKeysOrig = nullptr;
        if (total > 0) {
            cudaOk(cudaMalloc(&dKeysOrig, static_cast<size_t>(total) * sizeof(uint64_t)),
                   "radius keys orig");
            scatterRadiusSegmentsKernel<<<blocks, kBlock>>>(dKeys, dOffsets, dOffsetsOrig,
                                                            sort->orderOut, n, dKeysOrig);
            cudaOk(cudaGetLastError(), "scatter radius segments");
        }
        cudaFree(dKeys);
        cudaFree(dOffsets);
        cudaFree(dCountsOrig);
        dKeys = dKeysOrig;
        dOffsets = dOffsetsOrig;
    }

    cudaOk(cudaEventRecord(mark2), "record mark2");
    cudaOk(cudaEventRecord(stop), "record stop");
    cudaOk(cudaEventSynchronize(stop), "sync");
    float ms = 0.0f, sortMs = 0.0f, scatterMs = 0.0f;
    cudaOk(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    if (sort) {
        cudaOk(cudaEventElapsedTime(&sortMs, start, mark), "sort elapsed");
        cudaOk(cudaEventElapsedTime(&scatterMs, mark2, stop), "scatter elapsed");
        gLastStats.sortMs = static_cast<double>(sortMs + scatterMs);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(mark);
    cudaEventDestroy(mark2);
    cudaEventDestroy(stop);

    if (collectStats) {
        cudaOk(cudaMemcpyFromSymbol(&gLastStats.nodeTests, gNodeTests, sizeof(gNodeTests)),
               "copy nodeTests");
        cudaOk(cudaMemcpyFromSymbol(&gLastStats.triTests, gTriTests, sizeof(gTriTests)),
               "copy triTests");
        cudaOk(cudaMemcpyFromSymbol(&gLastStats.stackOverflows, gOverflows, sizeof(gOverflows)),
               "copy overflows");
    }

    cudaFree(dCounts);
    cudaFree(dScanTemp);

    out.d_offsets = dOffsets;
    out.d_keys = dKeys;
    out.count = n;
    out.total = total;
    return static_cast<double>(ms);
}

double radiusQueryGPU(const GpuBVH& bvh, GpuPointBuffer& buf, float radius, GpuRadiusBuffer& out,
                      bool collectStats, QueryOrder order) {
    return radiusQueryImpl<TriOps>(bvh, buf, radius, out, collectStats, order);
}

// 点云的半径查询：CSR 的键里装点索引，短段的 warp 双调排序和长段的 CUB
// 分段排序都与三角形路径共用。
double radiusQueryVertexGPU(const GpuBVH& bvh, GpuPointBuffer& buf, float radius,
                            GpuRadiusBuffer& out, bool collectStats, QueryOrder order) {
    return radiusQueryImpl<PointOps>(bvh, buf, radius, out, collectStats, order);
}

double radiusQueryGaussianGPU(const GpuBVH& bvh, GpuPointBuffer& buf, float radius,
                              GpuRadiusBuffer& out, bool collectStats, QueryOrder order) {
    return radiusQueryImpl<GaussianOps>(bvh, buf, radius, out, collectStats, order);
}

void destroyRadiusBufferGPU(GpuRadiusBuffer& buf) {
    if (buf.d_offsets) cudaFree(buf.d_offsets);
    if (buf.d_keys) cudaFree(buf.d_keys);
    buf = GpuRadiusBuffer{};
}

std::vector<std::vector<ClosestHit>> readRadiusResultsGPU(const GpuRadiusBuffer& buf) {
    std::vector<std::vector<ClosestHit>> out;
    if (buf.count <= 0) return out;
    out.resize(static_cast<size_t>(buf.count));

    std::vector<int> offsets(static_cast<size_t>(buf.count) + 1);
    cudaOk(cudaMemcpy(offsets.data(), buf.d_offsets, offsets.size() * sizeof(int),
                      cudaMemcpyDeviceToHost),
           "D2H radius offsets");

    std::vector<uint64_t> keys(static_cast<size_t>(buf.total));
    if (buf.total > 0) {
        cudaOk(cudaMemcpy(keys.data(), buf.d_keys, keys.size() * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost),
               "D2H radius keys");
    }

    for (int i = 0; i < buf.count; ++i) {
        std::vector<ClosestHit>& list = out[static_cast<size_t>(i)];
        list.reserve(static_cast<size_t>(offsets[i + 1] - offsets[i]));
        for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
            ClosestHit h;
            h.found = true;
            const uint64_t key = keys[static_cast<size_t>(j)];
            const uint32_t bits = static_cast<uint32_t>(key >> 32);
            std::memcpy(&h.dist2, &bits, sizeof(h.dist2));
            h.triangle = static_cast<int>(static_cast<uint32_t>(key));
            list.push_back(h);
        }
    }
    return out;
}

std::vector<GpuClosestResult> readQueryResultsGPU(const GpuPointBuffer& buf) {
    std::vector<GpuClosestResult> host(static_cast<size_t>(buf.count) * buf.stride);
    if (buf.count <= 0) return host;
    cudaOk(cudaMemcpy(host.data(), buf.d_results, host.size() * sizeof(GpuClosestResult),
                      cudaMemcpyDeviceToHost),
           "D2H query results");
    return host;
}

GpuQueryStats lastQueryStatsGPU() { return gLastStats; }

void warmupQueryGPU() {
    resetStatsKernel<<<1, 1>>>();
    cudaOk(cudaGetLastError(), "warmup query");
    cudaOk(cudaDeviceSynchronize(), "warmup query sync");
}
