// GPU 上的射线遍历：每个线程一条射线，显式栈做深度优先，最近命中距离 best 当剪枝阈值。
// 关键点是"一个线程一条射线"——一万条射线的延迟同时挂起来，靠并发掩盖显存延迟，
// 这正是 CPU 单线程做不到的（见 docs/baseline.md 的 Embree 32 线程一行）。

#include "gpu/gpu_bvh.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gpu/gpu_node.h"
#include "gpu/morton.h"

namespace {

constexpr int kBlock = 512;
constexpr int kStack = 64;  // 每层最多压一个远端子节点，depth<=63 的树不会溢出

void cudaOk(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// 射线排序：把原点量化到 10 位/轴，交错成 30 位 Morton 码（编码在 morton.h，
// 点查询排序共用）。相邻下标的射线因此落在同一块空间里，一个 warp 的遍历
// 路径高度重合，节点访问从 HBM 随机读变成 L1/L2 命中。排序只改变执行顺序，
// 结果按原下标散回。
// ---------------------------------------------------------------------------
__global__ void mortonKeysKernel(const Ray* __restrict__ rays, int count, AABB bounds,
                                 int keyMode, uint32_t* __restrict__ keys,
                                 int* __restrict__ order) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i >= count) return;
    const Ray r = rays[i];
    // keyMode 0：原点（量化域是射线原点包围盒，不能拿场景盒——相机和球面原点
    // 都会落到盒外被 clamp 成一团）；keyMode 1：方向（域固定 [-1,1]）。
    const Vec3 p = keyMode == 0 ? r.origin : r.dir;
    keys[i] = morton3D10(p.x, p.y, p.z, bounds.mn.x, bounds.mx.x, bounds.mn.y, bounds.mx.y,
                         bounds.mn.z, bounds.mx.z);
    order[i] = i;
}

__global__ void gatherRaysKernel(const Ray* __restrict__ rays, const int* __restrict__ order,
                                 int count, Ray* __restrict__ out) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < count) out[i] = rays[order[i]];
}

__global__ void scatterResultsKernel(const uint64_t* __restrict__ sortedOut,
                                     const int* __restrict__ order, int count,
                                     uint64_t* __restrict__ out) {
    const int i = blockIdx.x * kBlock + threadIdx.x;
    if (i < count) out[order[i]] = sortedOut[i];
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
__global__ __launch_bounds__(kBlock) void raycastKernel(
    const GpuBVHNode* __restrict__ nodes, const Tri* __restrict__ tris,
    const Ray* __restrict__ rays, int count, uint64_t* __restrict__ out) {
    const int idx = blockIdx.x * kBlock + threadIdx.x;
    const bool active = idx < count;

    float best = 0.0f;
    bool any = false;
    int nodeTests = 0;
    int triTests = 0;
    bool overflow = false;

    if (active) {
        const Ray ray = rays[idx];
        best = ray.tMax;

        int stack[kStack];
        int sp = 0;

        float tEnter;
        if (intersectAABB(ray, nodeBox(nodes[0]), best, tEnter)) {
            ++nodeTests;
            stack[sp++] = 0;

            while (sp > 0) {
                const int id = stack[--sp];
                const GpuBVHNode nd = nodes[id];

                if (nd.left < 0) {
                    const Tri& tri = tris[-nd.left - 1];
                    ++triTests;
                    float t;
                    if (intersectTriangle(ray, tri, t) && t < best) {
                        best = t;
                        any = true;
                    }
                    continue;
                }

                float tl, tr;
                const bool hl = intersectAABB(ray, nodeBox(nodes[nd.left]), best, tl);
                const bool hr = intersectAABB(ray, nodeBox(nodes[nd.right]), best, tr);
                if (hl) ++nodeTests;
                if (hr) ++nodeTests;

                if (sp + 2 > kStack) {
                    overflow = true;  // 树比栈深，结果不可信，交给主机报错
                    break;
                }
                // 近的孩子后压先出：先收紧 best，远的那支更可能被整枝剪掉。
                if (hl && hr) {
                    if (tl <= tr) {
                        stack[sp++] = nd.right;
                        stack[sp++] = nd.left;
                    } else {
                        stack[sp++] = nd.left;
                        stack[sp++] = nd.right;
                    }
                } else if (hl) {
                    stack[sp++] = nd.left;
                } else if (hr) {
                    stack[sp++] = nd.right;
                }
            }
        }

        uint64_t v = 0;
        if (any) v = (uint64_t{1} << 32) | static_cast<uint64_t>(__float_as_uint(best));
        out[idx] = v;
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

GpuTraversalStats gLastStats;

// 排序路径的全部暂存区，上传时一次分配。放在计时区外，避免每轮 cudaMalloc
// 把分配/首次触页的代价算进吞吐。
struct RaySortScratch {
    uint32_t* keysIn = nullptr;
    uint32_t* keysOut = nullptr;
    int* orderIn = nullptr;
    int* orderOut = nullptr;
    Ray* sortedRays = nullptr;
    uint64_t* sortedOut = nullptr;
    unsigned char* temp = nullptr;
    size_t tempBytes = 0;
    AABB originBounds{};
};

AABB computeOriginBounds(const std::vector<Ray>& rays) {
    AABB b;
    b.mn = rays[0].origin;
    b.mx = rays[0].origin;
    for (const Ray& r : rays) {
        b.mn.x = std::min(b.mn.x, r.origin.x);
        b.mn.y = std::min(b.mn.y, r.origin.y);
        b.mn.z = std::min(b.mn.z, r.origin.z);
        b.mx.x = std::max(b.mx.x, r.origin.x);
        b.mx.y = std::max(b.mx.y, r.origin.y);
        b.mx.z = std::max(b.mx.z, r.origin.z);
    }
    return b;
}

void destroySortScratch(void* p) {
    if (!p) return;
    auto* s = static_cast<RaySortScratch*>(p);
    if (s->keysIn) cudaFree(s->keysIn);
    if (s->keysOut) cudaFree(s->keysOut);
    if (s->orderIn) cudaFree(s->orderIn);
    if (s->orderOut) cudaFree(s->orderOut);
    if (s->sortedRays) cudaFree(s->sortedRays);
    if (s->sortedOut) cudaFree(s->sortedOut);
    if (s->temp) cudaFree(s->temp);
    delete s;
}

RaySortScratch* createSortScratch(int count) {
    auto* s = new RaySortScratch{};
    try {
        auto alloc = [](auto*& p, size_t n) {
            cudaOk(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(*p)), "sort scratch");
        };
        alloc(s->keysIn, static_cast<size_t>(count));
        alloc(s->keysOut, static_cast<size_t>(count));
        alloc(s->orderIn, static_cast<size_t>(count));
        alloc(s->orderOut, static_cast<size_t>(count));
        alloc(s->sortedRays, static_cast<size_t>(count));
        alloc(s->sortedOut, static_cast<size_t>(count));
        cub::DeviceRadixSort::SortPairs(nullptr, s->tempBytes, s->keysIn, s->keysOut, s->orderIn,
                                        s->orderOut, count, 0, 30);
        alloc(s->temp, s->tempBytes > 0 ? s->tempBytes : 1);
    } catch (...) {
        destroySortScratch(s);
        throw;
    }
    return s;
}

}  // namespace

GpuRayBuffer uploadRaysGPU(const std::vector<Ray>& rays) {
    GpuRayBuffer buf;
    buf.count = static_cast<int>(rays.size());
    if (buf.count == 0) return buf;

    cudaOk(cudaMalloc(&buf.d_rays, static_cast<size_t>(buf.count) * sizeof(Ray)), "rays");
    cudaOk(cudaMalloc(&buf.d_results, static_cast<size_t>(buf.count) * sizeof(uint64_t)), "results");
    cudaOk(cudaMemcpy(buf.d_rays, rays.data(), static_cast<size_t>(buf.count) * sizeof(Ray),
                      cudaMemcpyHostToDevice),
           "H2D rays");
    buf.d_sortScratch = createSortScratch(buf.count);
    static_cast<RaySortScratch*>(buf.d_sortScratch)->originBounds = computeOriginBounds(rays);
    return buf;
}

void destroyRayBufferGPU(GpuRayBuffer& buf) {
    if (buf.d_rays) cudaFree(buf.d_rays);
    if (buf.d_results) cudaFree(buf.d_results);
    destroySortScratch(buf.d_sortScratch);
    buf = GpuRayBuffer{};
}

double raycastGPU(const GpuBVH& bvh, GpuRayBuffer& buf, bool collectStats, RayOrder order) {
    if (buf.count <= 0 || bvh.nodeCount <= 0) return 0.0;

    gLastStats = GpuTraversalStats{};

    if (collectStats) {
        resetStatsKernel<<<1, 1>>>();
        cudaOk(cudaGetLastError(), "reset stats");
    }

    cudaEvent_t start, mark, mark2, stop;
    cudaOk(cudaEventCreate(&start), "event");
    cudaOk(cudaEventCreate(&mark), "event");
    cudaOk(cudaEventCreate(&mark2), "event");
    cudaOk(cudaEventCreate(&stop), "event");

    const int count = buf.count;
    const int blocks = (count + kBlock - 1) / kBlock;
    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);
    const auto* tris = static_cast<const Tri*>(bvh.d_tris);
    const auto* rays = static_cast<const Ray*>(buf.d_rays);
    auto* out = static_cast<uint64_t*>(buf.d_results);

    cudaOk(cudaEventRecord(start), "record start");

    if (order != RayOrder::Input) {
        auto* s = static_cast<RaySortScratch*>(buf.d_sortScratch);
        if (!s) throw std::runtime_error("ray buffer has no sort scratch");

        const int keyMode = order == RayOrder::MortonDirection ? 1 : 0;
        AABB bounds = s->originBounds;
        if (keyMode == 1) {
            bounds.mn = {-1.0f, -1.0f, -1.0f};
            bounds.mx = {1.0f, 1.0f, 1.0f};
        }

        mortonKeysKernel<<<blocks, kBlock>>>(rays, count, bounds, keyMode, s->keysIn, s->orderIn);
        cudaOk(cudaGetLastError(), "morton rays");

        cub::DeviceRadixSort::SortPairs(s->temp, s->tempBytes, s->keysIn, s->keysOut, s->orderIn,
                                        s->orderOut, count, 0, 30);
        cudaOk(cudaGetLastError(), "sort rays");

        gatherRaysKernel<<<blocks, kBlock>>>(rays, s->orderOut, count, s->sortedRays);
        cudaOk(cudaGetLastError(), "gather rays");

        cudaOk(cudaEventRecord(mark), "record mark");
        if (collectStats) {
            raycastKernel<true><<<blocks, kBlock>>>(nodes, tris, s->sortedRays, count,
                                                    s->sortedOut);
        } else {
            raycastKernel<false><<<blocks, kBlock>>>(nodes, tris, s->sortedRays, count,
                                                     s->sortedOut);
        }
        cudaOk(cudaGetLastError(), "raycast");
        cudaOk(cudaEventRecord(mark2), "record mark2");

        scatterResultsKernel<<<blocks, kBlock>>>(s->sortedOut, s->orderOut, count, out);
        cudaOk(cudaGetLastError(), "scatter results");

        cudaOk(cudaEventRecord(stop), "record stop");
        cudaOk(cudaEventSynchronize(stop), "sync");
        float ms = 0.0f, sortMs = 0.0f, scatterMs = 0.0f;
        cudaOk(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        cudaOk(cudaEventElapsedTime(&sortMs, start, mark), "sort elapsed");
        cudaOk(cudaEventElapsedTime(&scatterMs, mark2, stop), "scatter elapsed");
        gLastStats.sortMs = static_cast<double>(sortMs + scatterMs);
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

    if (collectStats) {
        raycastKernel<true><<<blocks, kBlock>>>(nodes, tris, rays, count, out);
    } else {
        raycastKernel<false><<<blocks, kBlock>>>(nodes, tris, rays, count, out);
    }

    cudaOk(cudaGetLastError(), "raycast");
    cudaOk(cudaEventRecord(stop), "record stop");
    cudaOk(cudaEventSynchronize(stop), "sync");
    float ms = 0.0f;
    cudaOk(cudaEventElapsedTime(&ms, start, stop), "elapsed");
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

RayStats summarizeGPU(const GpuRayBuffer& buf, std::vector<uint64_t>* perRay) {
    RayStats stats;
    if (buf.count <= 0) return stats;

    std::vector<uint64_t> host(static_cast<size_t>(buf.count));
    cudaOk(cudaMemcpy(host.data(), buf.d_results, host.size() * sizeof(uint64_t),
                      cudaMemcpyDeviceToHost),
           "D2H results");

    for (uint64_t v : host) {
        RayHit h;
        h.hit = (v >> 32) != 0;
        if (h.hit) {
            const uint32_t bits = static_cast<uint32_t>(v);
            std::memcpy(&h.t, &bits, sizeof(bits));
        }
        stats.add(h);
    }

    if (perRay) *perRay = std::move(host);
    return stats;
}

GpuTraversalStats lastTraversalStatsGPU() { return gLastStats; }
