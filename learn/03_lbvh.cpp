// 03 Morton 码与 LBVH
// 同一份场景、同一批射线（与 02 完全相同），用"排序 + 每个内部节点独立二分"建树。
// CPU 上串行跑，但每个内部节点的计算互不依赖 —— 这就是它在 GPU 上能全并行的原因。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

// ---- 与 01/02 相同的向量与求交 ----
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

// ---- 包围盒 ----
struct AABB {
    Vec3 lo{ 1e30f,  1e30f,  1e30f};
    Vec3 hi{-1e30f, -1e30f, -1e30f};
    void grow(const Vec3& p) {
        lo.x = fminf(lo.x, p.x); lo.y = fminf(lo.y, p.y); lo.z = fminf(lo.z, p.z);
        hi.x = fmaxf(hi.x, p.x); hi.y = fmaxf(hi.y, p.y); hi.z = fmaxf(hi.z, p.z);
    }
    void grow(const AABB& b) { grow(b.lo); grow(b.hi); }
};

static AABB triBounds(const Tri& t) {
    AABB b; b.grow(t.a); b.grow(t.b); b.grow(t.c); return b;
}

// ---- Morton 码 ----
static uint32_t expand10(uint32_t v) {   // 把 10 bit 散开到偶数位
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

static uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    return expand10(x) | (expand10(y) << 1) | (expand10(z) << 2);
}

static int clz64(uint64_t x) {
    int n = 0;
    if (x == 0) return 64;
    while (!(x & 0x8000000000000000ull)) { x <<= 1; ++n; }
    return n;
}

// ---- LBVH ----
// 节点布局：内部节点 0 .. n-2，叶子 n-1 .. 2n-2（按排序后的位置编号）。
// 没有指针，也没有递归：每个内部节点只靠自己的一段二分搜索确定区间。
struct Node {
    AABB box;
    int left = -1, right = -1;
};

static void buildLBVH(const std::vector<Tri>& tris, std::vector<int>& order,
                      std::vector<Node>& nodes, int& depthOut) {
    const int n = static_cast<int>(tris.size());

    // 1) 重心归一化到 [0,1]^3，再各取 10 bit
    AABB cb;
    std::vector<Vec3> centroids(n);
    for (int i = 0; i < n; ++i) {
        const Tri& t = tris[i];
        centroids[i] = {(t.a.x + t.b.x + t.c.x) / 3.0f,
                        (t.a.y + t.b.y + t.c.y) / 3.0f,
                        (t.a.z + t.b.z + t.c.z) / 3.0f};
        cb.grow(centroids[i]);
    }
    auto quant = [](float v, float lo, float extent) -> uint32_t {
        if (extent <= 0.0f) return 0;
        float q = (v - lo) / extent * 1023.0f + 0.5f;
        int i = static_cast<int>(fminf(1023.0f, fmaxf(0.0f, q)));
        return static_cast<uint32_t>(i);
    };
    const float ex = cb.hi.x - cb.lo.x, ey = cb.hi.y - cb.lo.y, ez = cb.hi.z - cb.lo.z;

    std::vector<uint32_t> codes(n);
    for (int i = 0; i < n; ++i) {
        codes[i] = morton3D(quant(centroids[i].x, cb.lo.x, ex),
                            quant(centroids[i].y, cb.lo.y, ey),
                            quant(centroids[i].z, cb.lo.z, ez));
    }

    // 2) 排序键 = 30 位 Morton 码 + 原始索引（低 32 位）。
    //    索引的作用是让键唯一：两个三角形落在同一个量化格子时公共前缀仍然可比，
    //    树形才唯一。GPU 上用 cub::DeviceRadixSort，这里用 stable_sort 对齐行为。
    std::vector<uint64_t> key(n);
    for (int i = 0; i < n; ++i) key[i] = (uint64_t(codes[i]) << 32) | uint32_t(i);

    order.resize(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });

    std::vector<uint64_t> k(n);
    for (int i = 0; i < n; ++i) k[i] = key[order[i]];

    auto delta = [&](int i, int j) -> int {   // 公共前缀长度；越界记 -1
        return (j < 0 || j >= n) ? -1 : clz64(k[i] ^ k[j]);
    };

    nodes.assign(2 * n - 1, Node{});

    // 3) 每个内部节点独立地二分出自己的区间和切分点。
    //    循环的每一轮在 GPU 上是一个线程，互相不写同一个位置。
    for (int i = 0; i < n - 1; ++i) {
        // 朝公共前缀更长的一侧扩
        const int d = (delta(i, i + 1) > delta(i, i - 1)) ? 1 : -1;
        // 扩到前缀降到"另一侧邻居"的水平为止
        const int sigmaMin = delta(i, i - d);

        int lMax = 2;   // 指数搜索定上界，再在 [0, lMax) 里二分
        while (delta(i, i + lMax * d) > sigmaMin) lMax *= 2;
        int l = 0;
        for (int t = lMax / 2; t > 0; t /= 2) {
            if (delta(i, i + (l + t) * d) > sigmaMin) l += t;
        }

        const int j = i + l * d;
        const int lower = std::min(i, j);
        const int upper = std::max(i, j);

        // 4) 切分点：最大的 gamma，使 delta(lower, gamma) > delta(lower, upper)
        const int deltaNode = delta(i, j);
        int gamma = lower;   // 下面至少会被赋值一次
        int p = lower, q = upper;
        while (p <= q) {
            const int mid = p + (q - p) / 2;
            if (delta(lower, mid) > deltaNode) {
                p = mid + 1;
            } else {
                gamma = mid - 1;
                q = mid - 1;
            }
        }

        // 5) 子节点编号：左子区间的内部节点就是 gamma，右子区间的是 gamma+1；
        //    区间只剩一个叶子时直接用叶子编号（n-1 之后是叶子区）。
        nodes[i].left  = (lower == gamma) ? (n - 1 + lower) : gamma;
        nodes[i].right = (upper == gamma + 1) ? (n - 1 + upper) : gamma + 1;
    }

    // 6) 自底向上算包围盒。GPU 上是按层发 kernel，这里是先序遍历后倒着扫。
    for (int k = 0; k < n; ++k) nodes[n - 1 + k].box = triBounds(tris[order[k]]);

    std::vector<int> preorder;
    preorder.reserve(2 * n - 1);
    std::vector<char> seen(2 * n - 1, 0);
    std::vector<std::pair<int, int>> stack;   // (node, depth)
    stack.push_back({0, 0});
    depthOut = 0;
    while (!stack.empty()) {
        const auto [id, depth] = stack.back();
        stack.pop_back();
        if (id < 0 || id >= 2 * n - 1 || seen[id]) {
            printf("structure error: bad node id %d\n", id);
            depthOut = -1;
            return;
        }
        seen[id] = 1;
        preorder.push_back(id);
        depthOut = std::max(depthOut, depth);
        if (id < n - 1) {
            stack.push_back({nodes[id].left, depth + 1});
            stack.push_back({nodes[id].right, depth + 1});
        }
    }
    if (static_cast<int>(preorder.size()) != 2 * n - 1) {
        printf("structure error: reached %zu of %d nodes\n", preorder.size(), 2 * n - 1);
        depthOut = -1;
        return;
    }
    for (int p = static_cast<int>(preorder.size()) - 1; p >= 0; --p) {
        const int id = preorder[p];
        if (id >= n - 1) continue;
        nodes[id].box = nodes[nodes[id].left].box;
        nodes[id].box.grow(nodes[nodes[id].right].box);
    }
}

// ---- 遍历 ----
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
                     const std::vector<int>& order, int n, Vec3 o, Vec3 d, float& best,
                     long long& nodeTests, long long& triTests, std::vector<int>& stack) {
    bool any = false;
    const Vec3 invd = {1.0f / d.x, 1.0f / d.y, 1.0f / d.z};

    stack.clear();
    stack.push_back(0);
    while (!stack.empty()) {
        const int id = stack.back();
        stack.pop_back();
        const Node& nd = nodes[id];
        nodeTests++;
        if (!hitBox(nd.box, o, invd, best)) continue;

        if (id >= n - 1) {                      // 叶子
            float t;
            triTests++;
            if (hit(tris[order[id - (n - 1)]], o, d, t) && t < best) { best = t; any = true; }
        } else {
            stack.push_back(nd.left);
            stack.push_back(nd.right);
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

    // 暴力法真值
    int bruteHits = 0;
    double bruteSumT = 0.0;
    long long bruteTriTests = 0;
    const double tb0 = nowMs();
    for (int r = 0; r < RAYS; r++) {
        float best = 1e30f;
        for (const Tri& tri : tris) {
            float t;
            bruteTriTests++;
            if (hit(tri, origins[r], dirs[r], t) && t < best) best = t;
        }
        if (best < 1e30f) { bruteHits++; bruteSumT += best; }
    }
    const double bruteMs = nowMs() - tb0;

    // 建 LBVH
    std::vector<int> order;
    std::vector<Node> nodes;
    int depth = 0;
    const double t0 = nowMs();
    buildLBVH(tris, order, nodes, depth);
    if (depth < 0) return 1;   // 结构自检失败，错误信息已在 build 里打印
    const double buildMs = nowMs() - t0;

    // 3 轮取中位数
    std::vector<int> stack;
    stack.reserve(128);
    std::vector<double> times;
    long long nodeTests = 0, triTests = 0;
    int hits_ = 0;
    double sumT = 0.0;
    for (int run = 0; run < 3; run++) {
        nodeTests = 0;
        triTests = 0;
        hits_ = 0;
        sumT = 0.0;
        const double s = nowMs();
        for (int r = 0; r < RAYS; r++) {
            float best = 1e30f;
            if (traverse(nodes, tris, order, TRIS, origins[r], dirs[r], best,
                         nodeTests, triTests, stack)) {
                hits_++;
                sumT += best;
            }
        }
        times.push_back(nowMs() - s);
    }
    std::sort(times.begin(), times.end());
    const double bvhMs = times[1];

    const bool ok = bruteHits == hits_ && fabs(bruteSumT - sumT) < 1e-3;
    printf("tris=%d rays=%d nodes=%zu depth=%d build_ms=%.1f\n", TRIS, RAYS, nodes.size(), depth,
           buildMs);
    printf("brute  hits=%d sum_t=%.3f %8.1f ms  tri_tests=%lld\n", bruteHits, bruteSumT,
           bruteMs, bruteTriTests);
    printf("lbvh   hits=%d sum_t=%.3f %8.1f ms  node_tests=%lld tri_tests=%lld\n", hits_, sumT,
           bvhMs, nodeTests, triTests);
    printf("speedup=%.1fx  tri_test_reduction=%.0fx  verify=%s\n", bruteMs / bvhMs,
           (double)bruteTriTests / (double)triTests, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
