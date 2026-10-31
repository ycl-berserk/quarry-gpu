// 外部用例：只依赖安装后的 quarry 包。
//   CPU：建 BVH，与暴力法逐条对拍。
//   GPU（有 CUDA 时）：建树、射线、最近点，与 CPU 参考对拍，再走一次局部 refit。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <core/scene.h>
#include <cpu/bvh.h>
#include <cpu/brute.h>

#ifdef QUARRY_EXAMPLE_GPU
#include <gpu/gpu_bvh.h>
#endif

namespace {

float bitsToFloat(uint32_t bits) {
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

bool raysMatchBrute(const Scene& scene, const BVH& bvh, const std::vector<Ray>& rays) {
    for (size_t i = 0; i < 2000 && i < rays.size(); ++i) {
        const RayHit a = raycastBVH(bvh, rays[i]);
        const RayHit b = raycastBrute(scene, rays[i]);
        if (a.hit != b.hit || (a.hit && a.t != b.t)) return false;
    }
    return true;
}

}  // namespace

int main() {
    SceneSpec spec;
    spec.grid = 128;
    Scene scene = buildScene(spec);

    const BVH cpuBvh = buildBVH(scene);
    const std::vector<Ray> rays = makeRays(scene.bounds, RayMode::Down, 20000, 1);
    const std::vector<Vec3> points = makeQueryPoints(scene.bounds, 2000, 2);

    std::printf("scene: %zu triangles, %zu bvh nodes\n", scene.tris.size(),
                cpuBvh.nodes.size());
    std::printf("cpu bvh vs brute (2000 rays): %s\n",
                raysMatchBrute(scene, cpuBvh, rays) ? "ok" : "MISMATCH");

#ifdef QUARRY_EXAMPLE_GPU
    std::string device;
    if (!gpuDescribe(&device)) {
        std::printf("no usable GPU: %s\n", device.c_str());
        return 0;
    }
    std::printf("device: %s\n", device.c_str());

    GpuBuildTimings build;
    GpuBVH bvh = buildBVHGPU(scene, &build);
    std::printf("gpu build: %.3f ms, %d nodes, depth %d\n", build.totalMs, bvh.nodeCount,
                bvh.depth);

    GpuRayBuffer rayBuf = uploadRaysGPU(rays);
    warmupQueryGPU();
    const double rayMs = raycastGPU(bvh, rayBuf);
    const RayStats gpuStats = summarizeGPU(rayBuf);

    RayStats cpuStats;
    for (const Ray& r : rays) cpuStats.add(raycastBVH(cpuBvh, r));
    const bool rayOk = gpuStats.hits == cpuStats.hits && gpuStats.hash == cpuStats.hash;
    std::printf("gpu raycast: %.3f ms, hits=%llu hash=0x%016llx (%s)\n", rayMs,
                static_cast<unsigned long long>(gpuStats.hits),
                static_cast<unsigned long long>(gpuStats.hash),
                rayOk ? "matches cpu" : "MISMATCH");

    GpuPointBuffer pointBuf = uploadPointsGPU(points);
    const double closestMs = closestPointGPU(bvh, pointBuf);
    const std::vector<GpuClosestResult> gpuHits = readQueryResultsGPU(pointBuf);
    int mismatches = 0;
    double maxDist2Err = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        const ClosestHit expected = closestPointBVH(cpuBvh, points[i]);
        if (gpuHits[i].triangle != expected.triangle) ++mismatches;
        const double diff =
            std::fabs(static_cast<double>(bitsToFloat(gpuHits[i].dist2Bits)) - expected.dist2);
        maxDist2Err = std::max(maxDist2Err, diff);
    }
    std::printf("gpu closest: %.3f ms, %d/%zu index mismatches, max |d2 err| %.3g\n",
                closestMs, mismatches, points.size(), maxDist2Err);

    std::string err = validateBVHGPU(bvh, scene);
    std::printf("validate: %s\n", err.empty() ? "pass" : err.c_str());

    // 每 100 个三角形沿 y 平移 0.01，走局部 refit（拓扑冻结）。
    std::vector<int> dirty;
    for (int i = 0; i < static_cast<int>(scene.tris.size()); i += 100) {
        Tri& t = scene.tris[static_cast<size_t>(i)];
        t.a.y += 0.01f;
        t.b.y += 0.01f;
        t.c.y += 0.01f;
        dirty.push_back(i);
    }
    scene.updateBounds();

    GpuRefitTimings refit;
    const double refitMs = refitBVHGPU(bvh, scene, dirty, &refit);
    err = validateBVHGPU(bvh, scene);
    std::printf("partial refit: %.3f ms (%d prims, mark %.3f + merge %.3f), validate: %s\n",
                refitMs, refit.updated, refit.markMs, refit.mergeMs,
                err.empty() ? "pass" : err.c_str());

    destroyPointBufferGPU(pointBuf);
    destroyRayBufferGPU(rayBuf);
    destroyBVHGPU(bvh);
#else
    std::printf("built without CUDA: quarry::gpu is unavailable in this install\n");
#endif
    return 0;
}
