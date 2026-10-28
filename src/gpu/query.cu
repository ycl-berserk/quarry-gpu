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

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu/gpu_node.h"

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

template <bool kStats>
__global__ __launch_bounds__(kBlock) void closestPointKernel(
    const GpuBVHNode* __restrict__ nodes, const Tri* __restrict__ tris,
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
        int bestTri = -1;

        int stack[kStack];
        int sp = 0;
        stack[sp++] = 0;

        while (sp > 0) {
            const int id = stack[--sp];
            const GpuBVHNode nd = nodes[id];
            ++nodeTests;

            if (nd.left < 0) {
                // 叶子是一个连续区间 [start, start+count)，count <= leafSize。
                const int start = -nd.left - 1;
                triTests += nd.right;
                for (int s = start, e = start + nd.right; s < e; ++s) {
                    const ClosestHit h = closestPointOnTriangle(p, tris[s]);
                    const int tri = primIds[s];
                    // 与 closerHit 等价：(距离, 场景索引) 字典序严格更小才替换。
                    if (bestTri < 0 || h.dist2 < best2 || (h.dist2 == best2 && tri < bestTri)) {
                        best2 = h.dist2;
                        bestTri = tri;
                    }
                }
                continue;
            }

            const float dl = distanceSquaredToAABB(p, nodeBox(nodes[nd.left]));
            const float dr = distanceSquaredToAABB(p, nodeBox(nodes[nd.right]));
            // 下界等于当前最优时仍可能藏着并列但索引更小的解，所以用 <=。
            const bool okL = bestTri < 0 || dl <= best2;
            const bool okR = bestTri < 0 || dr <= best2;

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

        if (bestTri >= 0) {
            result.dist2Bits = __float_as_uint(best2);
            result.triangle = bestTri;
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

template <bool kStats>
__global__ __launch_bounds__(kBlock) void kNearestKernel(
    const GpuBVHNode* __restrict__ nodes, const Tri* __restrict__ tris,
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
                const int start = -nd.left - 1;
                triTests += nd.right;
                for (int s = start, send = start + nd.right; s < send; ++s) {
                    const ClosestHit h = closestPointOnTriangle(p, tris[s]);
                    const KnnEntry e{h.dist2, primIds[s]};
                    if (n == k && !knnBetter(e, best[k - 1])) continue;
                    knnInsert(best, n, k, e);
                }
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

template <bool kStats>
__global__ __launch_bounds__(kBlock) void radiusCountKernel(
    const GpuBVHNode* __restrict__ nodes, const Tri* __restrict__ tris,
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
                const int start = -nd.left - 1;
                triTests += nd.right;
                for (int s = start, send = start + nd.right; s < send; ++s) {
                    if (closestPointOnTriangle(p, tris[s]).dist2 <= r2) ++found;
                }
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

__global__ __launch_bounds__(kBlock) void radiusFillKernel(
    const GpuBVHNode* __restrict__ nodes, const Tri* __restrict__ tris,
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
            const int start = -nd.left - 1;
            for (int s = start, send = start + nd.right; s < send; ++s) {
                const ClosestHit h = closestPointOnTriangle(p, tris[s]);
                if (h.dist2 > r2) continue;
                const int tri = primIds[s];
                // 非负浮点的位模式就是升序整数，拼上场景索引后按 64 位键升序排
                // 就等价于 closerHit 的 (dist2, 索引) 字典序。三角形索引随键一起排，
                // 不用再维护第二个数组。
                keys[write] = (static_cast<unsigned long long>(__float_as_uint(h.dist2)) << 32) |
                              static_cast<unsigned int>(tri);
                ++write;
            }
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

GpuPointBuffer uploadPointsGPU(const std::vector<Vec3>& points, int resultsPerPoint) {
    GpuPointBuffer buf;
    buf.count = static_cast<int>(points.size());
    buf.stride = resultsPerPoint > 0 ? resultsPerPoint : 1;
    if (buf.count == 0) return buf;

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
    buf = GpuPointBuffer{};
}

double closestPointGPU(const GpuBVH& bvh, GpuPointBuffer& buf, bool collectStats) {
    if (buf.count <= 0 || bvh.nodeCount <= 0) return 0.0;

    gLastStats = GpuQueryStats{};
    if (collectStats) {
        resetStatsKernel<<<1, 1>>>();
        cudaOk(cudaGetLastError(), "reset query stats");
    }

    cudaEvent_t start, stop;
    cudaOk(cudaEventCreate(&start), "event");
    cudaOk(cudaEventCreate(&stop), "event");
    cudaOk(cudaEventRecord(start), "record start");

    const int blocks = (buf.count + kBlock - 1) / kBlock;
    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);
    const auto* tris = static_cast<const Tri*>(bvh.d_tris);
    const auto* primIds = static_cast<const int*>(bvh.d_primIds);
    const auto* points = static_cast<const Vec3*>(buf.d_points);
    auto* out = static_cast<GpuClosestResult*>(buf.d_results);

    if (collectStats) {
        closestPointKernel<true><<<blocks, kBlock>>>(nodes, tris, primIds, points, buf.count, out);
    } else {
        closestPointKernel<false><<<blocks, kBlock>>>(nodes, tris, primIds, points, buf.count, out);
    }

    cudaOk(cudaGetLastError(), "closest point");
    cudaOk(cudaEventRecord(stop), "record stop");
    cudaOk(cudaEventSynchronize(stop), "sync");
    float ms = 0.0f;
    cudaOk(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start);
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

double kNearestGPU(const GpuBVH& bvh, GpuPointBuffer& buf, int k, bool collectStats) {
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

    cudaEvent_t start, stop;
    cudaOk(cudaEventCreate(&start), "event");
    cudaOk(cudaEventCreate(&stop), "event");
    cudaOk(cudaEventRecord(start), "record start");

    const int blocks = (buf.count + kBlock - 1) / kBlock;
    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);
    const auto* tris = static_cast<const Tri*>(bvh.d_tris);
    const auto* primIds = static_cast<const int*>(bvh.d_primIds);
    const auto* points = static_cast<const Vec3*>(buf.d_points);
    auto* out = static_cast<GpuClosestResult*>(buf.d_results);

    if (collectStats) {
        kNearestKernel<true><<<blocks, kBlock>>>(nodes, tris, primIds, points, buf.count, k, out);
    } else {
        kNearestKernel<false><<<blocks, kBlock>>>(nodes, tris, primIds, points, buf.count, k, out);
    }

    cudaOk(cudaGetLastError(), "k nearest");
    cudaOk(cudaEventRecord(stop), "record stop");
    cudaOk(cudaEventSynchronize(stop), "sync");
    float ms = 0.0f;
    cudaOk(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start);
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

double radiusQueryGPU(const GpuBVH& bvh, GpuPointBuffer& buf, float radius, GpuRadiusBuffer& out,
                      bool collectStats) {
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

    cudaEvent_t start, stop;
    cudaOk(cudaEventCreate(&start), "event");
    cudaOk(cudaEventCreate(&stop), "event");
    cudaOk(cudaEventRecord(start), "record start");

    const int blocks = (n + kBlock - 1) / kBlock;
    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);
    const auto* tris = static_cast<const Tri*>(bvh.d_tris);
    const auto* primIds = static_cast<const int*>(bvh.d_primIds);
    const auto* points = static_cast<const Vec3*>(buf.d_points);

    if (collectStats) {
        radiusCountKernel<true><<<blocks, kBlock>>>(nodes, tris, points, n, r2, dCounts);
    } else {
        radiusCountKernel<false><<<blocks, kBlock>>>(nodes, tris, points, n, r2, dCounts);
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

        radiusFillKernel<<<blocks, kBlock>>>(nodes, tris, primIds, points, n, r2, dOffsets, dKeys);
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

    cudaOk(cudaEventRecord(stop), "record stop");
    cudaOk(cudaEventSynchronize(stop), "sync");
    float ms = 0.0f;
    cudaOk(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start);
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
