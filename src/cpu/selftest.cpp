#include "cpu/selftest.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/fingerprint.h"
#include "core/scene.h"
#include "cpu/brute.h"
#include "cpu/bvh.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    std::printf("[%s] %s\n", condition ? "PASS" : "FAIL", name);
    if (!condition) ++g_failures;
}

// 手写用例：三个点构成的三角形在 z=0 平面上，法线朝 +z。
void testKnownAnswers() {
    const Tri tri{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};

    bool ok = true;
    float t = 0.0f;

    // 从上方垂直打下来，命中，t = 1
    ok &= intersectTriangle(Ray{{0.25f, 0.25f, 1.0f}, {0.0f, 0.0f, -1.0f}}, tri, t) &&
          std::fabs(t - 1.0f) < 1e-5f;

    // 三角形外侧，不命中
    ok &= !intersectTriangle(Ray{{2.0f, 2.0f, 1.0f}, {0.0f, 0.0f, -1.0f}}, tri, t);

    // 平行于三角形所在平面，不命中
    ok &= !intersectTriangle(Ray{{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}}, tri, t);

    // 交点在射线反方向（t < 0），不命中
    ok &= !intersectTriangle(Ray{{0.25f, 0.25f, 1.0f}, {0.0f, 0.0f, 1.0f}}, tri, t);

    // 从下方打上来，命中，t = 1
    ok &= intersectTriangle(Ray{{0.25f, 0.25f, -1.0f}, {0.0f, 0.0f, 1.0f}}, tri, t) &&
          std::fabs(t - 1.0f) < 1e-5f;

    check(ok, "known-answer triangle tests");
}

// BVH 必须和暴力法给出同一组答案。这是整个项目里最硬的一条约束。
void testBruteEqualsBVH() {
    SceneSpec spec;
    spec.kind = "mixed";
    spec.grid = 32;
    spec.randomTris = 2000;
    spec.seed = 7;

    const Scene scene = buildScene(spec);
    const BVH bvh = buildBVH(scene);

    for (RayMode mode : {RayMode::Down, RayMode::Random}) {
        const std::vector<Ray> rays = makeRays(scene.bounds, mode, 2000, 11);

        RayStats bruteStats;
        RayStats bvhStats;
        float maxDiff = 0.0f;
        bool sameTri = true;

        for (const Ray& ray : rays) {
            const RayHit brute = raycastBrute(scene, ray);
            const RayHit fast = raycastBVH(bvh, ray);

            bruteStats.add(brute);
            bvhStats.add(fast);

            if (brute.hit != fast.hit) maxDiff = 1e30f;
            else if (brute.hit) maxDiff = std::max(maxDiff, std::fabs(brute.t - fast.t));

            if (brute.hit && fast.hit && brute.triangle != fast.triangle) sameTri = false;
        }

        check(bruteStats.hits == bvhStats.hits,
              mode == RayMode::Down ? "brute/BVH hits agree (down)"
                                    : "brute/BVH hits agree (random)");
        check(maxDiff < 1e-3f, mode == RayMode::Down ? "brute/BVH max |dt| < 1e-3 (down)"
                                                     : "brute/BVH max |dt| < 1e-3 (random)");
        check(sameTri, mode == RayMode::Down ? "brute/BVH report the same triangle (down)"
                                             : "brute/BVH report the same triangle (random)");
        check(bruteStats.hash == bvhStats.hash,
              mode == RayMode::Down ? "brute/BVH fingerprint hash identical (down)"
                                    : "brute/BVH fingerprint hash identical (random)");
    }
}

// 手写用例覆盖最近点算法的三个出口：面内投影、顶点区域、边区域。
void testClosestPointKnownAnswers() {
    const Tri tri{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};

    auto at = [](const ClosestHit& h, Vec3 p, float d2) {
        return h.found && std::fabs(h.point.x - p.x) < 1e-6f &&
               std::fabs(h.point.y - p.y) < 1e-6f &&
               std::fabs(h.point.z - p.z) < 1e-6f && std::fabs(h.dist2 - d2) < 1e-6f;
    };

    bool ok = true;
    ok &= at(closestPointOnTriangle({0.25f, 0.25f, 1.0f}, tri), {0.25f, 0.25f, 0.0f}, 1.0f);
    ok &= at(closestPointOnTriangle({-1.0f, -1.0f, 0.0f}, tri), {0.0f, 0.0f, 0.0f}, 2.0f);
    ok &= at(closestPointOnTriangle({2.0f, 0.0f, 0.0f}, tri), {1.0f, 0.0f, 0.0f}, 1.0f);
    ok &= at(closestPointOnTriangle({0.5f, 0.5f, 0.0f}, tri), {0.5f, 0.5f, 0.0f}, 0.0f);
    ok &= at(closestPointOnTriangle({0.5f, -1.0f, 0.0f}, tri), {0.5f, 0.0f, 0.0f}, 1.0f);
    check(ok, "known-answer closest-point tests");

    const AABB box{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    const bool aabbOk =
        std::fabs(distanceSquaredToAABB({0.5f, 0.5f, 0.5f}, box)) < 1e-6f &&
        std::fabs(distanceSquaredToAABB({2.0f, 0.5f, 0.5f}, box) - 1.0f) < 1e-6f &&
        std::fabs(distanceSquaredToAABB({2.0f, 3.0f, 1.0f}, box) - 5.0f) < 1e-6f;
    check(aabbOk, "known-answer AABB distance tests");
}

// 最近点 / kNN / 半径三条路径都要和暴力法逐条对齐：
// 不只是距离，连选中的三角形和并列时的顺序都必须一致。
void testQueryBruteEqualsBVH() {
    SceneSpec spec;
    spec.kind = "mixed";
    spec.grid = 24;
    spec.randomTris = 1500;
    spec.seed = 5;

    const Scene scene = buildScene(spec);
    const BVH bvh = buildBVH(scene);
    const std::vector<Vec3> points = makeQueryPoints(scene.bounds, 200, 13);
    const float radius = 0.05f * length(scene.bounds.extent());

    QueryStats closestBruteStats;
    QueryStats closestBvhStats;
    bool closestSameId = true;
    float closestMaxDiff = 0.0f;

    QueryStats knnBruteStats;
    QueryStats knnBvhStats;
    bool knnSame = true;
    float knnMaxDiff = 0.0f;

    QueryStats radiusBruteStats;
    QueryStats radiusBvhStats;
    bool radiusSame = true;
    float radiusMaxDiff = 0.0f;

    for (const Vec3& p : points) {
        // 最近点
        {
            const ClosestHit b = closestPointBrute(scene, p);
            const ClosestHit f = closestPointBVH(bvh, p);
            closestBruteStats.add(b);
            closestBvhStats.add(f);

            if (b.found != f.found) {
                closestSameId = false;
                closestMaxDiff = 1e30f;
            } else if (b.found) {
                if (b.triangle != f.triangle) closestSameId = false;
                closestMaxDiff = std::max(closestMaxDiff, std::fabs(b.dist2 - f.dist2));
            }
        }

        // kNN
        {
            const auto b = kNearestBrute(scene, p, 8);
            const auto f = kNearestBVH(bvh, p, 8);
            knnBruteStats.add(b);
            knnBvhStats.add(f);

            if (b.size() != f.size()) {
                knnSame = false;
                continue;
            }
            for (size_t i = 0; i < b.size(); ++i) {
                if (b[i].triangle != f[i].triangle) knnSame = false;
                knnMaxDiff = std::max(knnMaxDiff, std::fabs(b[i].dist2 - f[i].dist2));
            }
        }

        // 半径
        {
            const auto b = radiusBrute(scene, p, radius);
            const auto f = radiusBVH(bvh, p, radius);
            radiusBruteStats.add(b);
            radiusBvhStats.add(f);

            if (b.size() != f.size()) {
                radiusSame = false;
                continue;
            }
            for (size_t i = 0; i < b.size(); ++i) {
                if (b[i].triangle != f[i].triangle) radiusSame = false;
                radiusMaxDiff = std::max(radiusMaxDiff, std::fabs(b[i].dist2 - f[i].dist2));
            }
        }
    }

    check(closestSameId, "closest: brute/BVH pick the same triangle");
    check(closestMaxDiff < 1e-6f, "closest: brute/BVH dist2 agrees");
    check(closestBvhStats.hash == closestBruteStats.hash,
          "closest: fingerprint hash identical");

    check(knnSame, "kNN: brute/BVH agree on ids and order");
    check(knnMaxDiff < 1e-6f, "kNN: brute/BVH dist2 agrees");
    check(knnBvhStats.results == 8 * points.size(), "kNN: returns k results per query");
    check(knnBvhStats.hash == knnBruteStats.hash, "kNN: fingerprint hash identical");

    check(radiusSame, "radius: brute/BVH agree on ids and order");
    check(radiusMaxDiff < 1e-6f, "radius: brute/BVH dist2 agrees");
    check(radiusBvhStats.results > 0, "radius: result set is non-empty");
    check(radiusBvhStats.hash == radiusBruteStats.hash,
          "radius: fingerprint hash identical");
}

void testBVHShape() {
    SceneSpec spec;
    spec.kind = "terrain";
    spec.grid = 64;
    const Scene scene = buildScene(spec);

    const BVH bvh = buildBVH(scene);

    bool allLeavesNonEmpty = true;
    bool allInternalHaveRight = true;
    for (const BVHNode& node : bvh.nodes) {
        if (node.count > 0) {
            allLeavesNonEmpty &= node.count > 0 && node.start >= 0;
        } else {
            allInternalHaveRight &= node.right > 0;
        }
    }

    check(bvh.tris.size() == scene.tris.size(), "BVH keeps every triangle");
    check(bvh.nodes[0].box.valid(), "root AABB valid");
    check(allLeavesNonEmpty, "leaves non-empty");
    check(allInternalHaveRight, "internal nodes have right child");
    check(bvhDepth(bvh) < 64, "BVH depth fits traversal stack");
}

void testDeterminism() {
    SceneSpec spec;
    spec.kind = "random";
    spec.randomTris = 1000;
    spec.seed = 42;
    const Scene a = buildScene(spec);
    const Scene b = buildScene(spec);

    bool sameScene = a.tris.size() == b.tris.size();
    for (size_t i = 0; sameScene && i < a.tris.size(); ++i) {
        sameScene &= a.tris[i].a.x == b.tris[i].a.x && a.tris[i].c.z == b.tris[i].c.z;
    }

    const std::vector<Ray> ra = makeRays(a.bounds, RayMode::Random, 500, 3);
    const std::vector<Ray> rb = makeRays(b.bounds, RayMode::Random, 500, 3);

    const BVH bvh = buildBVH(a);
    RayStats sa;
    RayStats sb;
    for (size_t i = 0; i < ra.size(); ++i) {
        sa.add(raycastBVH(bvh, ra[i]));
        sb.add(raycastBVH(bvh, rb[i]));
    }

    check(sameScene, "scene generation is seed-deterministic");
    check(sa.hash == sb.hash && sa.hits == sb.hits, "fingerprint is seed-deterministic");
}

}  // namespace

bool runSelfTest() {
    g_failures = 0;

    testKnownAnswers();
    testClosestPointKnownAnswers();
    testBruteEqualsBVH();
    testQueryBruteEqualsBVH();
    testBVHShape();
    testDeterminism();

    std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "SELFTEST PASS" : "SELFTEST FAIL",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0;
}
