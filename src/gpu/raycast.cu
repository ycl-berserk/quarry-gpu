// GPU 上的射线遍历：每个线程一条射线，显式栈做深度优先，最近命中距离 best 当剪枝阈值。
// 关键点是"一个线程一条射线"——一万条射线的延迟同时挂起来，靠并发掩盖显存延迟，
// 这正是 CPU 单线程做不到的（见 docs/baseline.md 的 Embree 32 线程一行）。

#include "gpu/gpu_bvh.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gpu/gpu_node.h"

namespace {

constexpr int kBlock = 512;
constexpr int kStack = 64;  // 每层最多压一个远端子节点，depth<=63 的树不会溢出

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
    return buf;
}

void destroyRayBufferGPU(GpuRayBuffer& buf) {
    if (buf.d_rays) cudaFree(buf.d_rays);
    if (buf.d_results) cudaFree(buf.d_results);
    buf = GpuRayBuffer{};
}

double raycastGPU(const GpuBVH& bvh, GpuRayBuffer& buf, bool collectStats) {
    if (buf.count <= 0 || bvh.nodeCount <= 0) return 0.0;

    if (collectStats) {
        resetStatsKernel<<<1, 1>>>();
        cudaOk(cudaGetLastError(), "reset stats");
    }

    cudaEvent_t start, stop;
    cudaOk(cudaEventCreate(&start), "event");
    cudaOk(cudaEventCreate(&stop), "event");
    cudaOk(cudaEventRecord(start), "record start");

    const int blocks = (buf.count + kBlock - 1) / kBlock;
    const auto* nodes = static_cast<const GpuBVHNode*>(bvh.d_nodes);
    const auto* tris = static_cast<const Tri*>(bvh.d_tris);
    const auto* rays = static_cast<const Ray*>(buf.d_rays);
    auto* out = static_cast<uint64_t*>(buf.d_results);

    if (collectStats) {
        raycastKernel<true><<<blocks, kBlock>>>(nodes, tris, rays, buf.count, out);
    } else {
        raycastKernel<false><<<blocks, kBlock>>>(nodes, tris, rays, buf.count, out);
    }

    cudaOk(cudaGetLastError(), "raycast");
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
