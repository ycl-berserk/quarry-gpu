#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

// 这份头文件同时被 CPU 参考实现和 CUDA kernel 包含：nvcc 下把 inline 函数标成
// __host__ __device__，两端就是同一份求交算术（同样的运算顺序、同样的舍入），
// CPU/GPU 的指纹对拍才有意义。
#if defined(__CUDACC__)
#define QUARRY_HD __host__ __device__
#else
#define QUARRY_HD
#endif

// 本项目所有几何都用 float：GPU 上 float 是默认精度，CPU 参考实现保持一致，
// 否则跨端比较时会混入精度差异。
struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
};

QUARRY_HD inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
QUARRY_HD inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
QUARRY_HD inline Vec3 operator-(const Vec3& a) { return {-a.x, -a.y, -a.z}; }
QUARRY_HD inline Vec3 operator*(const Vec3& a, float s) { return {a.x * s, a.y * s, a.z * s}; }
QUARRY_HD inline Vec3 operator*(float s, const Vec3& a) { return a * s; }
QUARRY_HD inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
QUARRY_HD inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(const Vec3& a) {
    const float l = length(a);
    return l > 0.0f ? a * (1.0f / l) : Vec3{};
}
QUARRY_HD inline float component(const Vec3& v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

struct AABB {
    Vec3 mn{std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()};
    Vec3 mx{-std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()};

    QUARRY_HD void extend(const Vec3& p) {
        mn.x = p.x < mn.x ? p.x : mn.x;
        mn.y = p.y < mn.y ? p.y : mn.y;
        mn.z = p.z < mn.z ? p.z : mn.z;
        mx.x = p.x > mx.x ? p.x : mx.x;
        mx.y = p.y > mx.y ? p.y : mx.y;
        mx.z = p.z > mx.z ? p.z : mx.z;
    }

    QUARRY_HD void extend(const AABB& b) {
        if (!b.valid()) return;
        extend(b.mn);
        extend(b.mx);
    }

    QUARRY_HD bool valid() const { return mn.x <= mx.x && mn.y <= mx.y && mn.z <= mx.z; }
    QUARRY_HD Vec3 center() const { return (mn + mx) * 0.5f; }
    QUARRY_HD Vec3 extent() const { return mx - mn; }

    QUARRY_HD float surfaceArea() const {
        if (!valid()) return 0.0f;
        const Vec3 d = mx - mn;
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};

// dir 不要求归一化；t 的单位随 dir 长度变化。
struct Ray {
    Vec3 origin;
    Vec3 dir;
    float tMin = 1e-6f;
    float tMax = std::numeric_limits<float>::infinity();
};

struct Tri {
    Vec3 a, b, c;

    QUARRY_HD AABB bounds() const {
        AABB box;
        box.extend(a);
        box.extend(b);
        box.extend(c);
        return box;
    }
};

// 射线-三角形求交结果。triangle 是场景内索引，未命中为 -1。
struct RayHit {
    bool hit = false;
    float t = 0.0f;
    int triangle = -1;
};

// Möller–Trumbore。det 很小时视为平行；t 必须落在 [tMin, tMax]。
QUARRY_HD inline bool intersectTriangle(const Ray& ray, const Tri& tri, float& tOut) {
    constexpr float kEps = 1e-7f;

    const Vec3 e1 = tri.b - tri.a;
    const Vec3 e2 = tri.c - tri.a;
    const Vec3 p = cross(ray.dir, e2);
    const float det = dot(e1, p);
    if (fabsf(det) < kEps) return false;

    const float inv = 1.0f / det;
    const Vec3 tv = ray.origin - tri.a;
    const float u = dot(tv, p) * inv;
    if (u < 0.0f || u > 1.0f) return false;

    const Vec3 q = cross(tv, e1);
    const float v = dot(ray.dir, q) * inv;
    if (v < 0.0f || u + v > 1.0f) return false;

    const float t = dot(e2, q) * inv;
    if (t < ray.tMin || t > ray.tMax) return false;

    tOut = t;
    return true;
}

// 点到三角形最近点。triangle 由调用方填场景索引。
// dist2 是平方距离：遍历时只比平方，省一次开方，最后需要真实距离时再 sqrt。
struct ClosestHit {
    bool found = false;
    float dist2 = std::numeric_limits<float>::max();
    int triangle = -1;
    Vec3 point;
    float bary1 = 0.0f;  // point = a + bary1*(b-a) + bary2*(c-a)
    float bary2 = 0.0f;
};

// 查询结果统一按 (平方距离, 场景索引) 升序：距离并列时顺序也必须可复现，
// 否则 CPU 和 GPU 的指纹对不上。
QUARRY_HD inline bool closerHit(const ClosestHit& a, const ClosestHit& b) {
    return a.dist2 < b.dist2 || (a.dist2 == b.dist2 && a.triangle < b.triangle);
}

// Ericson, Real-Time Collision Detection 5.1.5：先把平面外区域划成 7 块，
// 落在哪块就投影到对应的顶点/边/面。三次叉积换来的是一串点乘，
// 没有除法直到最后一步，GPU 上也好移植。
QUARRY_HD inline ClosestHit closestPointOnTriangle(const Vec3& p, const Tri& tri) {
    const Vec3 ab = tri.b - tri.a;
    const Vec3 ac = tri.c - tri.a;
    const Vec3 ap = p - tri.a;

    auto result = [&](const Vec3& q, float b1, float b2) {
        ClosestHit h;
        h.found = true;
        h.point = q;
        h.bary1 = b1;
        h.bary2 = b2;
        const Vec3 d = p - q;
        h.dist2 = dot(d, d);
        return h;
    };

    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return result(tri.a, 0.0f, 0.0f);

    const Vec3 bp = p - tri.b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return result(tri.b, 1.0f, 0.0f);

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float t = d1 / (d1 - d3);  // 边 AB
        return result(tri.a + ab * t, t, 0.0f);
    }

    const Vec3 cp = p - tri.c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return result(tri.c, 0.0f, 1.0f);

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float t = d2 / (d2 - d6);  // 边 AC
        return result(tri.a + ac * t, 0.0f, t);
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float t = (d4 - d3) / ((d4 - d3) + (d5 - d6));  // 边 BC
        return result(tri.b + (tri.c - tri.b) * t, 1.0f - t, t);
    }

    const float denom = va + vb + vc;
    if (!(denom > 1e-20f)) {  // 退化三角形：三个区域都没接住，退回最近顶点
        ClosestHit best = result(tri.a, 0.0f, 0.0f);
        const ClosestHit hb = result(tri.b, 1.0f, 0.0f);
        const ClosestHit hc = result(tri.c, 0.0f, 1.0f);
        if (hb.dist2 < best.dist2) best = hb;
        if (hc.dist2 < best.dist2) best = hc;
        return best;
    }

    const float b1 = vb / denom;
    const float b2 = vc / denom;
    return result(tri.a + ab * b1 + ac * b2, b1, b2);
}

// 点到 AABB 的平方距离。给最近点遍历当优先队列的键，也给 kNN 剪枝当下界。
QUARRY_HD inline float distanceSquaredToAABB(const Vec3& p, const AABB& box) {
    float d2 = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        const float v = component(p, axis);
        const float lo = component(box.mn, axis);
        const float hi = component(box.mx, axis);
        if (v < lo) {
            const float d = lo - v;
            d2 += d * d;
        } else if (v > hi) {
            const float d = v - hi;
            d2 += d * d;
        }
    }
    return d2;
}

// AABB slab 测试。tMax 传入当前最近命中距离，用于剪枝。
QUARRY_HD inline bool intersectAABB(const Ray& ray, const AABB& box, float tMax, float& tEnter) {
    float t0 = ray.tMin;
    float t1 = tMax;

    for (int axis = 0; axis < 3; ++axis) {
        const float o = component(ray.origin, axis);
        const float d = component(ray.dir, axis);
        const float lo = component(box.mn, axis);
        const float hi = component(box.mx, axis);

        if (fabsf(d) < 1e-20f) {
            if (o < lo || o > hi) return false;
            continue;
        }

        const float inv = 1.0f / d;
        float ta = (lo - o) * inv;
        float tb = (hi - o) * inv;
        if (ta > tb) {
            const float tmp = ta;
            ta = tb;
            tb = tmp;
        }

        t0 = ta > t0 ? ta : t0;
        t1 = tb < t1 ? tb : t1;
        if (t0 > t1) return false;
    }

    tEnter = t0;
    return true;
}
