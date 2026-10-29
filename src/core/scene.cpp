#include "core/scene.h"

#include <cmath>

#include "core/rng.h"

float terrainHeight(float x, float z) {
    return 3.0f * std::sin(0.13f * x) * std::cos(0.11f * z)
         + 1.2f * std::sin(0.37f * x + 0.7f) * std::sin(0.31f * z + 1.3f)
         + 0.4f * std::sin(0.90f * x) * std::sin(0.80f * z);
}

namespace {

Scene makeTerrain(int grid) {
    Scene scene;
    scene.tris.reserve(static_cast<size_t>(2) * grid * grid);

    const float half = 50.0f;
    const float step = (2.0f * half) / static_cast<float>(grid);

    for (int zi = 0; zi < grid; ++zi) {
        for (int xi = 0; xi < grid; ++xi) {
            const float x0 = -half + step * xi;
            const float z0 = -half + step * zi;
            const float x1 = x0 + step;
            const float z1 = z0 + step;

            const Vec3 p00{x0, terrainHeight(x0, z0), z0};
            const Vec3 p10{x1, terrainHeight(x1, z0), z0};
            const Vec3 p11{x1, terrainHeight(x1, z1), z1};
            const Vec3 p01{x0, terrainHeight(x0, z1), z1};

            scene.tris.push_back({p00, p10, p11});
            scene.tris.push_back({p00, p11, p01});
        }
    }

    scene.updateBounds();
    return scene;
}

Vec3 randomPoint(Pcg32& rng, float half) {
    return {rng.uniform(-half, half), rng.uniform(-half, half), rng.uniform(-half, half)};
}

Scene makeRandomTriangles(int count, uint64_t seed) {
    Scene scene;
    scene.tris.reserve(static_cast<size_t>(count));

    Pcg32 rng(seed, 1);
    while (static_cast<int>(scene.tris.size()) < count) {
        const Vec3 a = randomPoint(rng, 50.0f);
        const Vec3 b = a + Vec3{rng.uniform(-5.0f, 5.0f), rng.uniform(-5.0f, 5.0f),
                                rng.uniform(-5.0f, 5.0f)};
        const Vec3 c = a + Vec3{rng.uniform(-5.0f, 5.0f), rng.uniform(-5.0f, 5.0f),
                                rng.uniform(-5.0f, 5.0f)};

        // 退化三角形面积近零，Möller–Trumbore 会掉进 EPS 分支。与其在求交时特殊处理，
        // 不如在生成输入时就拒绝——输入干净，后面所有引擎的行为才好比较。
        if (length(cross(b - a, c - a)) < 1e-4f) continue;

        scene.tris.push_back({a, b, c});
    }

    scene.updateBounds();
    return scene;
}

}  // namespace

Scene buildScene(const SceneSpec& spec) {
    if (spec.kind == "terrain") return makeTerrain(spec.grid);

    if (spec.kind == "random") return makeRandomTriangles(spec.randomTris, spec.seed);

    if (spec.kind == "mixed") {
        Scene scene = makeTerrain(spec.grid);
        const Scene extra = makeRandomTriangles(spec.randomTris, spec.seed);
        scene.tris.insert(scene.tris.end(), extra.tris.begin(), extra.tris.end());
        scene.updateBounds();
        return scene;
    }

    return Scene{};
}

std::vector<Ray> makeRays(const AABB& bounds, RayMode mode, int count, uint64_t seed) {
    std::vector<Ray> rays;
    rays.reserve(static_cast<size_t>(count));

    Pcg32 rng(seed, 2);
    const Vec3 center = bounds.center();
    const Vec3 ext = bounds.extent();

    if (mode == RayMode::Down) {
        for (int i = 0; i < count; ++i) {
            Ray r;
            r.origin = {rng.uniform(bounds.mn.x, bounds.mx.x),
                        bounds.mx.y + 10.0f,
                        rng.uniform(bounds.mn.z, bounds.mx.z)};
            r.dir = {0.0f, -1.0f, 0.0f};
            r.tMax = ext.y + 20.0f;
            rays.push_back(r);
        }
        return rays;
    }

    const float radius = 1.6f * length(ext);
    const float spread = 0.9f;
    for (int i = 0; i < count; ++i) {
        const float theta = rng.uniform(0.0f, 6.2831853f);
        const float phi = std::acos(rng.uniform(-1.0f, 1.0f));
        const Vec3 origin{center.x + radius * std::sin(phi) * std::cos(theta),
                          center.y + radius * std::sin(phi) * std::sin(theta),
                          center.z + radius * std::cos(phi)};

        // 目标点铺得比包围盒更大，保证一部分射线打空
        const Vec3 target{center.x + rng.uniform(-1.0f, 1.0f) * ext.x * spread,
                          center.y + rng.uniform(-1.0f, 1.0f) * ext.y * spread,
                          center.z + rng.uniform(-1.0f, 1.0f) * ext.z * spread};

        Ray r;
        r.origin = origin;
        r.dir = normalize(target - origin);
        r.tMax = radius * 4.0f;
        rays.push_back(r);
    }

    return rays;
}

const char* rayModeName(RayMode mode) {
    return mode == RayMode::Down ? "down" : "random";
}

std::vector<Vec3> makeQueryPoints(const AABB& bounds, int count, uint64_t seed) {
    std::vector<Vec3> points;
    points.reserve(static_cast<size_t>(count));

    Pcg32 rng(seed, 3);
    const Vec3 center = bounds.center();
    const Vec3 half = bounds.extent() * 0.6f;  // 1.2 倍包围盒

    for (int i = 0; i < count; ++i) {
        points.push_back({center.x + rng.uniform(-half.x, half.x),
                          center.y + rng.uniform(-half.y, half.y),
                          center.z + rng.uniform(-half.z, half.z)});
    }
    return points;
}
