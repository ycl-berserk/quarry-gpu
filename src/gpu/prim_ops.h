#pragma once

#include "core/geometry.h"
#include "core/point_cloud.h"

// 叶子图元的两套"算术外壳"：构建只需要 bounds 与 centroid，点查询只需要 dist2。
// kernel 用 Ops 参数化后，三角形与点云共享同一份 Morton 排序、Karras 建树、
// 深度分桶与 refit 代码，差别只剩这里三个函数。
struct TriOps {
    using Item = Tri;

    QUARRY_HD static AABB bounds(const Tri& t) { return t.bounds(); }

    // 与 CPU 完全相同的重心公式（先加后除，三步顺序一致），Morton 码才能两端一致。
    QUARRY_HD static Vec3 centroid(const Tri& t) {
        return {(t.a.x + t.b.x + t.c.x) / 3.0f, (t.a.y + t.b.y + t.c.y) / 3.0f,
                (t.a.z + t.b.z + t.c.z) / 3.0f};
    }

    QUARRY_HD static float dist2(const Vec3& q, const Tri& t) {
        return closestPointOnTriangle(q, t).dist2;
    }
};

struct PointOps {
    using Item = Vec3;

    QUARRY_HD static AABB bounds(const Vec3& p) {
        AABB box;
        box.extend(p);
        return box;
    }

    QUARRY_HD static Vec3 centroid(const Vec3& p) { return p; }

    QUARRY_HD static float dist2(const Vec3& q, const Vec3& p) {
        const Vec3 d = q - p;
        return dot(d, d);
    }
};
