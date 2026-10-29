#include "cpu/selftest.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/fingerprint.h"
#include "core/gaussian.h"
#include "core/point_cloud.h"
#include "core/scene.h"
#include "cpu/brute.h"
#include "cpu/bvh.h"
#include "cpu/gauss_bvh.h"
#include "cpu/point_kd.h"

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

// 点云 kd-tree 与暴力法：距离、索引、并列顺序三者都要一致。
void testPointKdEqualsBrute() {
    PointCloudSpec spec;
    spec.kind = "sheet";
    spec.count = 4096;
    spec.seed = 3;
    const PointCloud cloud = buildPointCloud(spec);
    check(cloud.pts.size() == 4096, "point cloud keeps every point");

    const PointKdTree tree = buildPointKdTree(cloud);
    const std::vector<Vec3> points = makeQueryPoints(cloud.bounds, 100, 17);
    const float radius = 0.08f * length(cloud.bounds.extent());

    bool closestSame = true;
    bool knnSame = true;
    bool radiusSame = true;
    bool distOk = true;

    for (const Vec3& p : points) {
        const ClosestHit b = closestVertexBrute(cloud, p);
        const ClosestHit k = kdClosest(tree, p);
        if (b.found != k.found || b.triangle != k.triangle) closestSame = false;
        if (std::fabs(b.dist2 - k.dist2) > 1e-6f) distOk = false;

        const auto bk = kNearestVertexBrute(cloud, p, 8);
        const auto kk = kdKNearest(tree, p, 8);
        if (bk.size() != kk.size()) {
            knnSame = false;
        } else {
            for (size_t i = 0; i < bk.size(); ++i) {
                if (bk[i].triangle != kk[i].triangle) knnSame = false;
                if (std::fabs(bk[i].dist2 - kk[i].dist2) > 1e-6f) distOk = false;
            }
        }

        const auto br = radiusVertexBrute(cloud, p, radius);
        const auto kr = kdRadius(tree, p, radius);
        if (br.size() != kr.size()) {
            radiusSame = false;
        } else {
            for (size_t i = 0; i < br.size(); ++i) {
                if (br[i].triangle != kr[i].triangle) radiusSame = false;
                if (std::fabs(br[i].dist2 - kr[i].dist2) > 1e-6f) distOk = false;
            }
        }
    }

    check(closestSame, "point kd: closest matches brute force");
    check(knnSame, "point kd: kNN matches brute force");
    check(radiusSame, "point kd: radius matches brute force");
    check(distOk, "point kd: distances agree");
}

// 高斯：先用手算答案钉住"到椭球实体的距离"，再让 BVH 与暴力法逐位对拍。
void testGaussians() {
    // 轴向椭球 a=1, b=2, c=3，位于原点、单位旋转。沿各轴外推，距离就是 |y| - 半轴。
    Gaussian g;
    g.mean = {0.0f, 0.0f, 0.0f};
    g.scale = {1.0f, 2.0f, 3.0f};
    g.rot = {0.0f, 0.0f, 0.0f, 1.0f};

    bool axisOk = true;
    axisOk &= std::fabs(gaussianDist2({5.0f, 0.0f, 0.0f}, g) - 16.0f) < 1e-4f;
    axisOk &= std::fabs(gaussianDist2({0.0f, 4.0f, 0.0f}, g) - 4.0f) < 1e-4f;
    axisOk &= std::fabs(gaussianDist2({0.0f, 0.0f, 7.0f}, g) - 16.0f) < 1e-4f;
    axisOk &= gaussianDist2({0.5f, 0.5f, 0.5f}, g) == 0.0f;

    // 绕 z 轴转 90°：x 半轴变成 2、y 半轴变成 1，长轴跟着转向。
    Quat rotZ90;
    rotZ90.z = std::sqrt(0.5f);
    rotZ90.w = std::sqrt(0.5f);
    Gaussian r = g;
    r.rot = rotZ90;
    axisOk &= std::fabs(gaussianDist2({4.0f, 0.0f, 0.0f}, r) - 4.0f) < 1e-3f;
    axisOk &= std::fabs(gaussianDist2({0.0f, 5.0f, 0.0f}, r) - 16.0f) < 1e-3f;
    check(axisOk, "gaussian: analytic distances along the axes");

    // 球退化成点距：a=b=c 时 d^2 = max(0, |y| - r)^2。
    Gaussian sphere;
    sphere.scale = {2.0f, 2.0f, 2.0f};
    bool sphereOk = true;
    for (const Vec3& p : std::vector<Vec3>{{5.0f, 0.0f, 0.0f}, {0.0f, -3.0f, 0.0f},
                                           {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f}}) {
        const float len = length(p);
        const float want = len > 2.0f ? (len - 2.0f) * (len - 2.0f) : 0.0f;
        sphereOk &= std::fabs(gaussianDist2(p, sphere) - want) < 1e-4f;
    }
    check(sphereOk, "gaussian: sphere case matches point distance");

    // 外接盒必须是下界：盒距离永远不超过椭球距离。
    bool boundsOk = true;
    for (int i = 0; i < 200; ++i) {
        Gaussian q;
        q.mean = {0.5f * i - 50.0f, 0.25f * i, 0.1f * i};
        q.rot = {0.1f * i, 0.2f, -0.1f, 1.0f};
        q.scale = {0.1f + 0.01f * i, 0.05f, 0.4f};
        const Vec3 p{std::sin(0.3f * i) * 60.0f, std::cos(0.7f * i) * 10.0f, 0.5f * i};
        boundsOk &= distanceSquaredToAABB(p, gaussianBounds(q)) <= gaussianDist2(p, q) + 1e-5f;
    }
    check(boundsOk, "gaussian: AABB distance is a lower bound");

    // 独立见证：float64 二分求同一方程的根。float32 牛顿若欠收敛，这里会露馅。
    bool refOk = true;
    float worstRel = 0.0f;
    for (int i = 0; i < 300; ++i) {
        const double angle = 0.37 * i;
        Gaussian q;
        q.mean = {static_cast<float>(std::cos(angle) * 20.0), 1.0f,
                  static_cast<float>(std::sin(angle) * 20.0)};
        q.rot = {static_cast<float>(std::sin(0.11 * i)), 0.3f, static_cast<float>(std::cos(0.7 * i)),
                 1.0f};
        q.scale = {0.2f + 0.02f * i, 0.05f + 0.01f * i, 0.3f};
        const Vec3 p{static_cast<float>(std::cos(0.9 * i) * 45.0),
                     static_cast<float>(std::sin(0.3 * i) * 5.0),
                     static_cast<float>(std::sin(0.5 * i) * 45.0)};

        Vec3 u, v, w;
        gaussianAxes(q.rot, u, v, w);
        const double y0 = (p.x - q.mean.x) * u.x + (p.y - q.mean.y) * u.y + (p.z - q.mean.z) * u.z;
        const double y1 = (p.x - q.mean.x) * v.x + (p.y - q.mean.y) * v.y + (p.z - q.mean.z) * v.z;
        const double y2 = (p.x - q.mean.x) * w.x + (p.y - q.mean.y) * w.y + (p.z - q.mean.z) * w.z;
        const double a = q.scale.x, b = q.scale.y, c = q.scale.z;
        double want = 0.0;
        if ((y0 / a) * (y0 / a) + (y1 / b) * (y1 / b) + (y2 / c) * (y2 / c) > 1.0) {
            auto F = [&](double t) {
                return (a * y0 / (t + a * a)) * (a * y0 / (t + a * a)) +
                       (b * y1 / (t + b * b)) * (b * y1 / (t + b * b)) +
                       (c * y2 / (t + c * c)) * (c * y2 / (t + c * c)) - 1.0;
            };
            double lo = 0.0, hi = 2.0 * std::max({a, b, c}) * std::sqrt(y0 * y0 + y1 * y1 + y2 * y2);
            for (int it = 0; it < 200; ++it) {
                const double mid = 0.5 * (lo + hi);
                if (F(mid) > 0.0) lo = mid;
                else hi = mid;
            }
            const double t = 0.5 * (lo + hi);
            const double x0 = a * a * y0 / (t + a * a);
            const double x1 = b * b * y1 / (t + b * b);
            const double x2 = c * c * y2 / (t + c * c);
            want = (y0 - x0) * (y0 - x0) + (y1 - x1) * (y1 - x1) + (y2 - x2) * (y2 - x2);
        }
        // 用距离（而不是距离平方）比：近表面时平方值本身趋于 0，相对误差没有意义。
        // 分母取 max 轴长做尺度，量纲统一成"相对主轴"的误差。
        const float got = gaussianDist2(p, q);
        const float gotDist = std::sqrt(got);
        const float wantDist = static_cast<float>(std::sqrt(want));
        const float amax = std::max({q.scale.x, q.scale.y, q.scale.z});
        const float rel = std::fabs(gotDist - wantDist) / (wantDist + amax);
        worstRel = std::max(worstRel, rel);
        refOk &= rel < 1e-4f;
    }
    if (!refOk) std::printf("       worst distance rel = %.3e\n", static_cast<double>(worstRel));
    check(refOk, "gaussian: distance matches a float64 reference");

    // BVH 与暴力法：索引、距离、并列顺序全部一致。
    GaussianCloudSpec spec;
    spec.kind = "sheet";
    spec.count = 4096;
    spec.seed = 5;
    const GaussianCloud cloud = buildGaussianCloud(spec);
    check(cloud.gaussians.size() == 4096, "gaussian cloud keeps every primitive");

    const GaussBVH bvh = buildGaussBVH(cloud);
    const std::vector<Vec3> points = makeQueryPoints(cloud.bounds, 100, 19);
    const float radius = 0.08f * length(cloud.bounds.extent());

    bool closestSame = true;
    bool knnSame = true;
    bool radiusSame = true;
    bool distExact = true;

    for (const Vec3& p : points) {
        const ClosestHit b = closestGaussianBrute(cloud, p);
        const ClosestHit f = closestGaussianBVH(bvh, p);
        if (b.found != f.found || b.triangle != f.triangle || b.dist2 != f.dist2) closestSame = false;

        const auto bk = kNearestGaussianBrute(cloud, p, 6);
        const auto fk = kNearestGaussianBVH(bvh, p, 6);
        if (bk.size() != fk.size()) {
            knnSame = false;
        } else {
            for (size_t i = 0; i < bk.size(); ++i) {
                if (bk[i].triangle != fk[i].triangle) knnSame = false;
                if (bk[i].dist2 != fk[i].dist2) distExact = false;
            }
        }

        const auto br = radiusGaussianBrute(cloud, p, radius);
        const auto fr = radiusGaussianBVH(bvh, p, radius);
        if (br.size() != fr.size()) {
            radiusSame = false;
        } else {
            for (size_t i = 0; i < br.size(); ++i) {
                if (br[i].triangle != fr[i].triangle) radiusSame = false;
                if (br[i].dist2 != fr[i].dist2) distExact = false;
            }
        }
    }

    check(closestSame, "gaussian bvh: closest matches brute force bitwise");
    check(knnSame, "gaussian bvh: kNN matches brute force");
    check(radiusSame, "gaussian bvh: radius matches brute force");
    check(distExact, "gaussian bvh: distances are bitwise identical");
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
    testPointKdEqualsBrute();
    testGaussians();
    testDeterminism();

    std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "SELFTEST PASS" : "SELFTEST FAIL",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0;
}
