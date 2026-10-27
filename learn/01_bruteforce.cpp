#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

struct Vec3 { float x, y, z; };
static Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static Vec3 operator*(Vec3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
static float dot(Vec3 a, Vec3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
static Vec3  cross(Vec3 a, Vec3 b){ return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }

struct Tri { Vec3 a, b, c; };

// Möller–Trumbore
static bool hit(const Tri& tri, Vec3 o, Vec3 d, float& tOut) {
    const float EPS = 1e-7f;
    Vec3 e1 = tri.b - tri.a;
    Vec3 e2 = tri.c - tri.a;
    Vec3 p  = cross(d, e2);
    float det = dot(e1, p);
    if (fabsf(det) < EPS) return false;      // 平行
    float inv = 1.0f / det;
    Vec3 tv = o - tri.a;
    float u = dot(tv, p) * inv;
    if (u < 0.0f || u > 1.0f) return false;
    Vec3 q = cross(tv, e1);
    float v = dot(d, q) * inv;
    if (v < 0.0f || u + v > 1.0f) return false;
    tOut = dot(e2, q) * inv;
    return tOut > EPS;
}

// 手写三个用例，证明实现正确
static void selfTest() {
    Tri t = {{0,0,0}, {1,0,0}, {0,1,0}};
    float th;
    bool ok = true;
    ok &= hit(t, {0.25f,0.25f,1}, {0,0,-1}, th) && fabsf(th - 1.0f) < 1e-5f;
    ok &= !hit(t, {2.0f, 2.0f, 1}, {0,0,-1}, th);
    ok &= !hit(t, {0,0,1}, {1,0,0}, th);
    printf("selftest %s\n", ok ? "PASS" : "FAIL");
}

int main() {
    selfTest();

    unsigned seed = 1;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / (float)(1u << 24);
    };

    // 1000 个三角形，分布在原点周围 100 单位内
    std::vector<Tri> tris;
    tris.reserve(1000);
    for (int i = 0; i < 1000; i++) {
        Vec3 a = {rnd()*100-50, rnd()*100-50, rnd()*100-50};
        Vec3 b = {a.x + rnd()*10-5, a.y + rnd()*10-5, a.z + rnd()*10-5};
        Vec3 c = {a.x + rnd()*10-5, a.y + rnd()*10-5, a.z + rnd()*10-5};
        tris.push_back({a, b, c});
    }

    // 1 万条射线：起点在球面上，方向指向原点附近
    int hits = 0;
    double sumT = 0.0;
    const int RAYS = 10000;
    for (int i = 0; i < RAYS; i++) {
        float theta = rnd() * 6.2831853f;
        float phi   = rnd() * 3.1415926f;
        Vec3 o = {200.0f * sinf(phi) * cosf(theta),
                  200.0f * sinf(phi) * sinf(theta),
                  200.0f * cosf(phi)};
        Vec3 target = {rnd()*20-10, rnd()*20-10, rnd()*20-10};
        Vec3 d = {target.x - o.x, target.y - o.y, target.z - o.z};
        float len = sqrtf(dot(d, d));
        d = d * (1.0f / len);

        float best = 1e30f;
        for (const Tri& tri : tris) {
            float t;
            if (hit(tri, o, d, t) && t < best) best = t;
        }
        if (best < 1e30f) { hits++; sumT += best; }
    }

    printf("rays=%d hits=%d sum_t=%.3f\n", RAYS, hits, sumT);
    return 0;
}
