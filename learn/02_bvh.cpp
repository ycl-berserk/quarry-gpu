// 02 从暴力到 BVH
// 同一份场景、同一批射线，暴力法 vs 中位数切分 BVH：
// 要求 hits 相同、sum_t 相同、时间明显更短。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

// ---- 与 01 相同的向量与求交，这里只保留定义 ----
struct Vec3 { float x, y, z; };
static Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static Vec3 operator*(Vec3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
static float dot(Vec3 a, Vec3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
static Vec3  cross(Vec3 a, Vec3 b){ return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }

struct Tri { Vec3 a, b, c; };

static bool hit(const Tri& tri, Vec3 o, Vec3 d, float& tOut) {
    const float EPS = 1e-7f;
    Vec3 e1 = tri.b - tri.a;
    Vec3 e2 = tri.c - tri.a;
    Vec3 p  = cross(d, e2);
    float det = dot(e1, p);
    if (fabsf(det) < EPS) return false;
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

// ---- BVH ----
struct AABB {
    Vec3 lo{ 1e30f,  1e30f,  1e30f};
    Vec3 hi{-1e30f, -1e30f, -1e30f};
    void grow(const Vec3& p) {
        lo.x = fminf(lo.x, p.x); lo.y = fminf(lo.y, p.y); lo.z = fminf(lo.z, p.z);
        hi.x = fmaxf(hi.x, p.x); hi.y = fmaxf(hi.y, p.y); hi.z = fmaxf(hi.z, p.z);
    }
    void grow(const AABB& b) { grow(b.lo); grow(b.hi); }
};

struct Node {
    AABB box;
    int  left = -1, right = -1;
    int  start = 0, count = 0;     // count > 0 表示叶子
};

static AABB triBounds(const Tri& t) {
    AABB b; b.grow(t.a); b.grow(t.b); b.grow(t.c); return b;
}

static float centroidAxis(const Tri& t, int axis) {
    float x = (t.a.x + t.b.x + t.c.x) / 3.0f;
    float y = (t.a.y + t.b.y + t.c.y) / 3.0f;
    float z = (t.a.z + t.b.z + t.c.z) / 3.0f;
    return axis == 0 ? x : (axis == 1 ? y : z);
}

// 按最长轴中点分割；返回节点索引
static int buildBVH(const std::vector<Tri>& tris, std::vector<int>& idx,
                    int start, int end, std::vector<Node>& nodes) {
    int self = (int)nodes.size();
    nodes.push_back({});

    AABB box;
    for (int i = start; i < end; i++) box.grow(triBounds(tris[idx[i]]));
    nodes[self].box = box;

    if (end - start <= 2) {                    // 叶子
        nodes[self].start = start;
        nodes[self].count = end - start;
        return self;
    }

    Vec3 ext = {box.hi.x - box.lo.x, box.hi.y - box.lo.y, box.hi.z - box.lo.z};
    int axis = (ext.x > ext.y) ? (ext.x > ext.z ? 0 : 2) : (ext.y > ext.z ? 1 : 2);
    float mid = 0.5f * ((axis == 0 ? box.lo.x : axis == 1 ? box.lo.y : box.lo.z) +
                        (axis == 0 ? box.hi.x : axis == 1 ? box.hi.y : box.hi.z));

    int i = start, j = end;
    while (i < j) {
        if (centroidAxis(tris[idx[i]], axis) < mid) i++;
        else std::swap(idx[i], idx[--j]);
    }
    if (i == start || i == end) i = (start + end) / 2;   // 退化保护

    int l = buildBVH(tris, idx, start, i, nodes);
    int r = buildBVH(tris, idx, i, end, nodes);
    nodes[self].left = l;
    nodes[self].right = r;
    return self;
}

// 射线与 AABB 求交（slab 法）
static bool hitBox(const AABB& b, Vec3 o, const Vec3& invd, float tmax) {
    float t1 = (b.lo.x - o.x) * invd.x, t2 = (b.hi.x - o.x) * invd.x;
    float tmin = fminf(t1, t2), tmx = fmaxf(t1, t2);
    t1 = (b.lo.y - o.y) * invd.y; t2 = (b.hi.y - o.y) * invd.y;
    tmin = fmaxf(tmin, fminf(t1, t2)); tmx = fminf(tmx, fmaxf(t1, t2));
    t1 = (b.lo.z - o.z) * invd.z; t2 = (b.hi.z - o.z) * invd.z;
    tmin = fmaxf(tmin, fminf(t1, t2)); tmx = fminf(tmx, fmaxf(t1, t2));
    return fmaxf(tmin, 0.0f) <= tmx && tmin <= tmax;
}

static bool traverse(const std::vector<Node>& nodes, const std::vector<Tri>& tris,
                     const std::vector<int>& idx, Vec3 o, Vec3 d, float& best,
                     long long& nodeTests, long long& triTests) {
    bool any = false;
    Vec3 invd = {1.0f/d.x, 1.0f/d.y, 1.0f/d.z};
    int stack[128];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const Node& n = nodes[stack[--sp]];
        nodeTests++;
        if (!hitBox(n.box, o, invd, best)) continue;

        if (n.count > 0) {
            for (int i = n.start; i < n.start + n.count; i++) {
                float t;
                triTests++;
                if (hit(tris[idx[i]], o, d, t) && t < best) { best = t; any = true; }
            }
        } else {
            stack[sp++] = n.left;
            stack[sp++] = n.right;
        }
    }
    return any;
}

static double nowMs() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

int main() {
    const int TRIS = 50000;
    const int RAYS = 5000;

    unsigned seed = 1;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / (float)(1u << 24);
    };

    std::vector<Tri> tris;
    tris.reserve(TRIS);
    for (int i = 0; i < TRIS; i++) {
        Vec3 a = {rnd()*100-50, rnd()*100-50, rnd()*100-50};
        Vec3 b = {a.x + rnd()*10-5, a.y + rnd()*10-5, a.z + rnd()*10-5};
        Vec3 c = {a.x + rnd()*10-5, a.y + rnd()*10-5, a.z + rnd()*10-5};
        tris.push_back({a, b, c});
    }

    // 射线提前生成，两种方法吃的是同一批输入
    std::vector<Vec3> origins(RAYS), dirs(RAYS);
    for (int i = 0; i < RAYS; i++) {
        float theta = rnd() * 6.2831853f;
        float phi   = rnd() * 3.1415926f;
        Vec3 o = {200.0f * sinf(phi) * cosf(theta),
                  200.0f * sinf(phi) * sinf(theta),
                  200.0f * cosf(phi)};
        Vec3 target = {rnd()*20-10, rnd()*20-10, rnd()*20-10};
        Vec3 d = {target.x - o.x, target.y - o.y, target.z - o.z};
        d = d * (1.0f / sqrtf(dot(d, d)));
        origins[i] = o;
        dirs[i] = d;
    }

    // 暴力法：每条射线测所有三角形
    int bruteHits = 0;
    double bruteSumT = 0.0;
    long long bruteTriTests = 0;
    double t0 = nowMs();
    for (int r = 0; r < RAYS; r++) {
        float best = 1e30f;
        for (const Tri& tri : tris) {
            float t;
            bruteTriTests++;
            if (hit(tri, origins[r], dirs[r], t) && t < best) best = t;
        }
        if (best < 1e30f) { bruteHits++; bruteSumT += best; }
    }
    const double bruteMs = nowMs() - t0;

    // 建树
    std::vector<int> idx(TRIS);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<Node> nodes;
    nodes.reserve(2 * TRIS);
    t0 = nowMs();
    buildBVH(tris, idx, 0, TRIS, nodes);
    const double buildMs = nowMs() - t0;

    // BVH：3 轮取中位数，同时统计访问量
    long long bvhNodeTests = 0, bvhTriTests = 0;
    std::vector<double> times;
    int bvhHits = 0;
    double bvhSumT = 0.0;
    for (int run = 0; run < 3; run++) {
        bvhNodeTests = 0;
        bvhTriTests = 0;
        bvhHits = 0;
        bvhSumT = 0.0;
        t0 = nowMs();
        for (int r = 0; r < RAYS; r++) {
            float best = 1e30f;
            if (traverse(nodes, tris, idx, origins[r], dirs[r], best, bvhNodeTests, bvhTriTests)) {
                bvhHits++;
                bvhSumT += best;
            }
        }
        times.push_back(nowMs() - t0);
    }
    std::sort(times.begin(), times.end());
    const double bvhMs = times[1];

    const bool ok = bruteHits == bvhHits && fabs(bruteSumT - bvhSumT) < 1e-3;
    printf("tris=%d rays=%d nodes=%zu build_ms=%.1f\n", TRIS, RAYS, nodes.size(), buildMs);
    printf("brute  hits=%d sum_t=%.3f %8.1f ms  tri_tests=%lld\n", bruteHits, bruteSumT,
           bruteMs, bruteTriTests);
    printf("bvh    hits=%d sum_t=%.3f %8.1f ms  node_tests=%lld tri_tests=%lld\n", bvhHits,
           bvhSumT, bvhMs, bvhNodeTests, bvhTriTests);
    printf("speedup=%.1fx  tri_test_reduction=%.0fx  verify=%s\n", bruteMs / bvhMs,
           (double)bruteTriTests / (double)bvhTriTests, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
