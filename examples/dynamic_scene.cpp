// 旗舰用例：动态场景。
//
// 地形是静态的，场景里漂着一批八面体"碎片"。碎片每帧都在动，但只有每 10 帧
// 才有一次生灭（旧碎片到期、新碎片从尾部追加）。于是两条更新路径各就各位：
//
//   - 纯移动帧：refitBVHGPU，只重算 64 个叶子盒和它们的祖先；
//   - 生灭帧：  updateBVHGPU，删旧下标 + 移幸存者 + 尾部追加。
//
// 每帧更新完都在这棵树上跑 2 万条 Morton 序射线与 2000 个最近点；每 10 帧拿
// CPU 参考 BVH 对拍并校验树；最后把最终场景整棵重建，要求增量树的查询结果与
// 全新构建逐位相同。
//
// 这里只用公开 API（安装前缀即可编译），不碰任何内部结构。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <core/scene.h>
#include <cpu/bvh.h>
#include <gpu/gpu_bvh.h>

namespace {

constexpr int kPropTris = 8;     // 一个八面体 = 4 个上表面 + 4 个下表面
constexpr int kFrames = 80;
constexpr int kCheckEvery = 10;
constexpr int kCpuRayChecks = 200;
constexpr int kCpuClosestChecks = 200;
constexpr int kInitialProps = 8;
constexpr int kSpawnEvery = 10;  // 每 10 帧生灭一次
constexpr int kPropTtl = 20;     // 碎片存活 20 帧

struct Prop {
    Vec3 base;
    float radius = 1.0f;
    float phase = 0.0f;
    int ttl = kPropTtl;
    int age = 0;
    int start = 0;  // 在场景里的第一个三角形下标
};

Vec3 propCenter(const Prop& p, int frame, float terrainY) {
    const float x = p.base.x + 0.45f * p.radius * std::sin(0.11f * frame + p.phase);
    const float z = p.base.z + 0.45f * p.radius * std::cos(0.09f * frame + p.phase);
    return {x, terrainY + 1.2f * p.radius + 0.8f, z};
}

void appendOctahedron(std::vector<Tri>& out, const Vec3& c, float r) {
    const Vec3 top{c.x, c.y + r, c.z};
    const Vec3 bottom{c.x, c.y - r, c.z};
    const Vec3 ring[4] = {{c.x + r, c.y, c.z}, {c.x, c.y, c.z + r}, {c.x - r, c.y, c.z},
                          {c.x, c.y, c.z - r}};
    for (int i = 0; i < 4; ++i) {
        const Vec3 a = ring[i];
        const Vec3 b = ring[(i + 1) % 4];
        out.push_back({top, a, b});
        out.push_back({bottom, b, a});
    }
}

// 按当前 props 重新铺一遍场景；顺带把每个 prop 的起始下标写回。
void layoutScene(const std::vector<Tri>& terrain, std::vector<Prop>& props, int frame,
                 Scene& scene) {
    scene.tris = terrain;
    for (Prop& p : props) {
        const float y = terrainHeight(p.base.x, p.base.z);
        p.start = static_cast<int>(scene.tris.size());
        appendOctahedron(scene.tris, propCenter(p, frame, y), p.radius);
    }
    scene.updateBounds();
}

uint32_t fpBits(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

// 与 summarizeGPU 的 perRay 编码一致：(hit << 32) | t 的位模式。
uint64_t encodeRay(const RayHit& h) {
    return (static_cast<uint64_t>(h.hit ? 1u : 0u) << 32) | fpBits(h.t);
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
    return v[mid];
}

}  // namespace

int main() {
    std::string device;
    if (!gpuDescribe(&device)) {
        std::printf("no usable GPU: %s\n", device.c_str());
        return 0;
    }
    std::printf("device: %s\n", device.c_str());

    SceneSpec spec;
    spec.grid = 256;
    const Scene terrainScene = buildScene(spec);
    const std::vector<Tri> terrain = terrainScene.tris;
    const AABB bounds = terrainScene.bounds;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ux(0.08f, 0.92f);
    std::uniform_real_distribution<float> ur(0.7f, 1.6f);
    auto makeProp = [&](int age) {
        Prop p;
        p.base = {bounds.mn.x + (bounds.mx.x - bounds.mn.x) * ux(rng), 0.0f,
                  bounds.mn.z + (bounds.mx.z - bounds.mn.z) * ux(rng)};
        p.radius = ur(rng);
        p.phase = 6.2831853f * ux(rng);
        p.age = age;
        return p;
    };

    std::vector<Prop> props;
    for (int i = 0; i < kInitialProps; ++i) props.push_back(makeProp(0));

    Scene scene;
    layoutScene(terrain, props, 0, scene);
    std::printf("scene: %zu terrain tris + %d props (%d tris), bounds %.1f x %.1f\n",
                terrain.size(), static_cast<int>(props.size()),
                static_cast<int>(props.size()) * kPropTris, bounds.mx.x - bounds.mn.x,
                bounds.mx.z - bounds.mn.z);

    const std::vector<Ray> rays = makeRays(scene.bounds, RayMode::Down, 20000, 1);
    const std::vector<Vec3> points = makeQueryPoints(scene.bounds, 2000, 2);

    warmupBuildGPU();
    GpuBuildTimings buildTimings;
    GpuBVH bvh = buildBVHGPU(scene, &buildTimings);
    GpuRayBuffer rayBuf = uploadRaysGPU(rays);
    GpuPointBuffer pointBuf = uploadPointsGPU(points, 1);
    warmupQueryGPU();

    std::printf("initial build: %.3f ms, %d nodes, depth %d\n", buildTimings.totalMs,
                bvh.nodeCount, bvh.depth);

    std::vector<double> updateMs, refitMs, topoMs, rayMs, closestMs;
    int refitFrames = 0;
    int topoFrames = 0;
    int fullRebuilds = 0;
    int cpuRayMismatch = 0;
    int cpuClosestMismatch = 0;
    int validateFailures = 0;
    std::vector<GpuClosestResult> lastClosest;

    for (int frame = 1; frame <= kFrames; ++frame) {
        // 1. 推进模拟：到期删除、幸存者移动、新碎片追加到尾部。
        const std::vector<Prop> oldProps = props;
        std::vector<int> deleted;
        std::vector<Prop> survivors;
        for (const Prop& old : oldProps) {
            Prop p = old;
            if (++p.age >= p.ttl) {
                for (int k = 0; k < kPropTris; ++k) deleted.push_back(p.start + k);
            } else {
                survivors.push_back(p);
            }
        }
        const int spawned = (frame % kSpawnEvery == 0) ? 1 : 0;
        for (int i = 0; i < spawned; ++i) survivors.push_back(makeProp(0));
        props = std::move(survivors);

        layoutScene(terrain, props, frame, scene);

        // 2. 更新树。所有幸存的碎片都在动（新下标），生灭帧走拓扑手术，
        //    纯移动帧走 refit——后者只碰这些叶子及其祖先。
        const int survivorProps = static_cast<int>(props.size()) - spawned;
        std::vector<int> moved;
        moved.reserve(static_cast<size_t>(survivorProps) * kPropTris);
        for (int i = 0; i < survivorProps; ++i) {
            for (int k = 0; k < kPropTris; ++k) moved.push_back(props[i].start + k);
        }

        const bool structural = !deleted.empty() || spawned > 0;
        bool fullRebuild = false;
        double updateTime = 0.0;
        if (structural) {
            GpuTopoEdit edit;
            edit.deleted = deleted;
            edit.moved = moved;
            edit.appended = spawned * kPropTris;

            GpuTopoUpdateTimings topo;
            updateTime = updateBVHGPU(bvh, scene, edit, &topo);
            fullRebuild = topo.fullRebuild;
            if (fullRebuild) ++fullRebuilds;
            topoMs.push_back(updateTime);
            ++topoFrames;
        } else {
            GpuRefitTimings rt;
            updateTime = refitBVHGPU(bvh, scene, moved, &rt);
            refitMs.push_back(updateTime);
            ++refitFrames;
        }
        updateMs.push_back(updateTime);

        // 3. 同一棵树上的两条查询。
        rayMs.push_back(raycastGPU(bvh, rayBuf, false, RayOrder::Morton));
        closestMs.push_back(closestPointGPU(bvh, pointBuf));

        const bool check = (frame % kCheckEvery == 0) || frame == kFrames;
        if (check) {
            std::vector<uint64_t> perRay;
            summarizeGPU(rayBuf, &perRay);
            lastClosest = readQueryResultsGPU(pointBuf);

            const BVH cpu = buildBVH(scene);
            for (int i = 0; i < kCpuRayChecks; ++i) {
                const RayHit h = raycastBVH(cpu, rays[static_cast<size_t>(i)]);
                if (encodeRay(h) != perRay[static_cast<size_t>(i)]) ++cpuRayMismatch;
            }
            for (int i = 0; i < kCpuClosestChecks; ++i) {
                const ClosestHit h = closestPointBVH(cpu, points[static_cast<size_t>(i)]);
                const GpuClosestResult& g = lastClosest[static_cast<size_t>(i)];
                if (!h.found) {
                    if (g.triangle != -1) ++cpuClosestMismatch;
                } else if (g.triangle != h.triangle || g.dist2Bits != fpBits(h.dist2)) {
                    ++cpuClosestMismatch;
                }
            }
            if (!validateBVHGPU(bvh, scene).empty()) ++validateFailures;

            const char* mode = structural ? (fullRebuild ? "topo+rebuild" : "topo") : "refit";
            std::printf("frame %3d [%-11s]: update %6.3f ms, ray %6.3f ms, "
                        "closest %6.3f ms, %d props, nodes %d, depth %d\n",
                        frame, mode, updateTime, rayMs.back(), closestMs.back(),
                        static_cast<int>(props.size()), bvh.nodeCount, bvh.depth);
        }
    }

    // 4. 增量树 vs 对最终场景的全新构建：查询结果必须逐位相同。
    const RayStats incRays = summarizeGPU(rayBuf, nullptr);
    const std::vector<GpuClosestResult> incClosest = readQueryResultsGPU(pointBuf);

    GpuBuildTimings freshTimings;
    GpuBVH fresh = buildBVHGPU(scene, &freshTimings);
    const double freshRayTime = raycastGPU(fresh, rayBuf, false, RayOrder::Morton);
    closestPointGPU(fresh, pointBuf);
    const RayStats freshRays = summarizeGPU(rayBuf, nullptr);
    const std::vector<GpuClosestResult> freshClosest = readQueryResultsGPU(pointBuf);

    int freshMismatch = 0;
    for (size_t i = 0; i < incClosest.size(); ++i) {
        if (incClosest[i].triangle != freshClosest[i].triangle ||
            incClosest[i].dist2Bits != freshClosest[i].dist2Bits) {
            ++freshMismatch;
        }
    }
    const bool freshRaysOk =
        incRays.hits == freshRays.hits && incRays.hash == freshRays.hash;

    std::printf("\nsummary: %d frames = %d refit + %d topology (%d full rebuilds)\n",
                kFrames, refitFrames, topoFrames, fullRebuilds);
    std::printf("update: median refit %.3f ms, median topology %.3f ms, median all %.3f ms\n",
                median(refitMs), median(topoMs), median(updateMs));
    std::printf("queries: ray %.3f ms, closest %.3f ms (median, 20k / 2k queries)\n",
                median(rayMs), median(closestMs));
    std::printf("tree hash 0x%016llx\n",
                static_cast<unsigned long long>(treeHashGPU(bvh)));
    std::printf("cpu checks: ray mismatch %d/%d, closest mismatch %d/%d, validate failures %d\n",
                cpuRayMismatch, kFrames / kCheckEvery * kCpuRayChecks, cpuClosestMismatch,
                kFrames / kCheckEvery * kCpuClosestChecks, validateFailures);
    std::printf("fresh build: %.3f ms -> refit %.2fx, topology %.2fx\n",
                freshTimings.totalMs, freshTimings.totalMs / median(refitMs),
                freshTimings.totalMs / median(topoMs));
    std::printf("incremental vs fresh: rays %s (hash 0x%016llx), closest mismatches %d/%zu, "
                "fresh ray %.3f ms\n",
                freshRaysOk ? "match" : "MISMATCH",
                static_cast<unsigned long long>(incRays.hash), freshMismatch,
                incClosest.size(), freshRayTime);

    destroyBVHGPU(fresh);
    destroyPointBufferGPU(pointBuf);
    destroyRayBufferGPU(rayBuf);
    destroyBVHGPU(bvh);

    const bool ok = cpuRayMismatch == 0 && cpuClosestMismatch == 0 && validateFailures == 0 &&
                    freshRaysOk && freshMismatch == 0;
    std::printf("%s\n", ok ? "checks ok" : "CHECKS FAILED");
    return ok ? 0 : 1;
}
