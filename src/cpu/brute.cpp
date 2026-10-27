#include "cpu/brute.h"

#include <algorithm>

RayHit raycastBrute(const Scene& scene, const Ray& ray) {
    RayHit best;
    float bestT = ray.tMax;

    for (int i = 0; i < static_cast<int>(scene.tris.size()); ++i) {
        Ray sub = ray;
        sub.tMax = bestT;

        float t;
        if (intersectTriangle(sub, scene.tris[i], t)) {
            bestT = t;
            best.hit = true;
            best.t = t;
            best.triangle = i;
        }
    }

    return best;
}

void raycastBruteAll(const Scene& scene, const std::vector<Ray>& rays, std::vector<RayHit>& hits) {
    hits.resize(rays.size());
    for (size_t i = 0; i < rays.size(); ++i) hits[i] = raycastBrute(scene, rays[i]);
}

ClosestHit closestPointBrute(const Scene& scene, const Vec3& p) {
    ClosestHit best;

    for (int i = 0; i < static_cast<int>(scene.tris.size()); ++i) {
        ClosestHit h = closestPointOnTriangle(p, scene.tris[i]);
        h.triangle = i;
        // 严格小于：距离并列时保留先遇到的（场景索引更小）那个。
        if (!best.found || h.dist2 < best.dist2) best = h;
    }

    return best;
}

std::vector<ClosestHit> kNearestBrute(const Scene& scene, const Vec3& p, int k) {
    std::vector<ClosestHit> all;
    all.reserve(scene.tris.size());
    for (int i = 0; i < static_cast<int>(scene.tris.size()); ++i) {
        ClosestHit h = closestPointOnTriangle(p, scene.tris[i]);
        h.triangle = i;
        all.push_back(h);
    }

    const size_t n = std::min(all.size(), static_cast<size_t>(std::max(k, 0)));
    std::partial_sort(all.begin(), all.begin() + n, all.end(), closerHit);
    all.resize(n);
    return all;
}

std::vector<ClosestHit> radiusBrute(const Scene& scene, const Vec3& p, float radius) {
    std::vector<ClosestHit> out;
    const float r2 = radius * radius;

    for (int i = 0; i < static_cast<int>(scene.tris.size()); ++i) {
        ClosestHit h = closestPointOnTriangle(p, scene.tris[i]);
        if (h.dist2 > r2) continue;
        h.triangle = i;
        out.push_back(h);
    }

    std::sort(out.begin(), out.end(), closerHit);
    return out;
}
