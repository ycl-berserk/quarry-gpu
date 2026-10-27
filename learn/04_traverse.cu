// 04 GPU LBVH 与遍历 —— 一个能跑、能对拍的最小实现
//
// 编译（在 learn\ 下）：
//   build.bat 04_traverse.cu
// 运行：
//   04_traverse.exe --grid 64 --rays 20000 --runs 5
//
// 这份代码是 D:\code\quarry\src\gpu 的教学精简版：
//   - 场景是同一套地形网格（三角形数 = 2 * grid^2）
//   - 建树五步：Morton -> cub 排序 -> Karras 内部节点 -> 深度分桶 -> 自底向上 refit
//   - 遍历：每线程一条射线，64 项显式栈，近子先走
//   - 正确性：每条射线的 t 位模式与 CPU 暴力法逐位比较
//
// 为什么 CPU 和 GPU 能逐位相同：几何与求交用同一份 __host__ __device__ 代码，
// nvcc 关掉 FMA 融合（--fmad=false），两边标量求交的每一步舍入完全一样。

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HD __host__ __device__

// ---------------------------------------------------------------- 几何

struct Vec3 {
    float x, y, z;
};

HD inline Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
HD inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

struct Tri {
    Vec3 a, b, c;
};

struct Ray {
    Vec3 origin;
    Vec3 dir;
    float maxT;
};

// Möller–Trumbore，双面命中。t 必须严格小于 maxT：遍历时把当前最优 t 传进来，
// 就是逐条射线的剪枝。
HD inline bool rayTri(const Ray& r, const Tri& tri, float maxT, float& tOut) {
    const Vec3 e1 = sub(tri.b, tri.a);
    const Vec3 e2 = sub(tri.c, tri.a);
    const Vec3 p = cross(r.dir, e2);
    const float det = dot(e1, p);
    if (fabsf(det) < 1e-12f) return false;
    const float invDet = 1.0f / det;
    const Vec3 s = sub(r.origin, tri.a);
    const float u = dot(s, p) * invDet;
    if (u < 0.0f || u > 1.0f) return false;
    const Vec3 q = cross(s, e1);
    const float v = dot(r.dir, q) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float t = dot(e2, q) * invDet;
    if (t <= 0.0f || t >= maxT) return false;
    tOut = t;
    return true;
}

struct AABB {
    Vec3 mn, mx;
};

HD inline AABB triBounds(const Tri& t) {
    AABB b;
    b.mn.x = fminf(t.a.x, fminf(t.b.x, t.c.x));
    b.mn.y = fminf(t.a.y, fminf(t.b.y, t.c.y));
    b.mn.z = fminf(t.a.z, fminf(t.b.z, t.c.z));
    b.mx.x = fmaxf(t.a.x, fmaxf(t.b.x, t.c.x));
    b.mx.y = fmaxf(t.a.y, fmaxf(t.b.y, t.c.y));
    b.mx.z = fmaxf(t.a.z, fmaxf(t.b.z, t.c.z));
    return b;
}

// 32 字节节点。内部节点 left/right 是子节点编号；叶子的 left = -(槽位+1)，right 不用。
struct Node {
    float mnx, mny, mnz;
    float mxx, mxy, mxz;
    int left, right;
};

// slab test：返回射线是否碰到盒子，以及进入距离（用于决定先走哪个孩子）。
HD inline bool rayBox(const Node& n, const Ray& r, float maxT, float& tNear) {
    float tmin = 0.0f;
    float tmax = maxT;
    const float ox[3] = {r.origin.x, r.origin.y, r.origin.z};
    const float d[3] = {r.dir.x, r.dir.y, r.dir.z};
    const float mn[3] = {n.mnx, n.mny, n.mnz};
    const float mx[3] = {n.mxx, n.mxy, n.mxz};
    for (int k = 0; k < 3; ++k) {
        const float inv = 1.0f / d[k];
        float t0 = (mn[k] - ox[k]) * inv;
        float t1 = (mx[k] - ox[k]) * inv;
        if (t0 > t1) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = tmin > t0 ? tmin : t0;
        tmax = tmax < t1 ? tmax : t1;
        if (tmin > tmax) return false;
    }
    tNear = tmin;
    return true;
}

// ---------------------------------------------------------------- 场景

static float terrainHeight(float x, float z) {
    return 0.5f * sinf(x * 0.31f) + 0.3f * cosf(z * 0.23f) + 0.15f * sinf((x + z) * 0.7f);
}

static void makeTerrain(int grid, std::vector<Tri>& tris, AABB& bounds) {
    const float lo = -50.0f;
    const float step = 100.0f / static_cast<float>(grid);
    tris.clear();
    tris.reserve(static_cast<size_t>(grid) * grid * 2);
    bounds.mn = {1e30f, 1e30f, 1e30f};
    bounds.mx = {-1e30f, -1e30f, -1e30f};
    auto vertex = [&](int ix, int iz) {
        const float x = lo + step * static_cast<float>(ix);
        const float z = lo + step * static_cast<float>(iz);
        return Vec3{x, terrainHeight(x, z), z};
    };
    for (int iz = 0; iz < grid; ++iz) {
        for (int ix = 0; ix < grid; ++ix) {
            const Vec3 p00 = vertex(ix, iz);
            const Vec3 p10 = vertex(ix + 1, iz);
            const Vec3 p01 = vertex(ix, iz + 1);
            const Vec3 p11 = vertex(ix + 1, iz + 1);
            tris.push_back({p00, p10, p11});
            tris.push_back({p00, p11, p01});
        }
    }
    for (const Tri& t : tris) {
        const AABB b = triBounds(t);
        bounds.mn.x = std::min(bounds.mn.x, b.mn.x);
        bounds.mn.y = std::min(bounds.mn.y, b.mn.y);
        bounds.mn.z = std::min(bounds.mn.z, b.mn.z);
        bounds.mx.x = std::max(bounds.mx.x, b.mx.x);
        bounds.mx.y = std::max(bounds.mx.y, b.mx.y);
        bounds.mx.z = std::max(bounds.mx.z, b.mx.z);
    }
}

static uint32_t gRng = 12345u;
static float nextFloat() {
    gRng = gRng * 1664525u + 1013904223u;
    return static_cast<float>(gRng >> 8) * (1.0f / 16777216.0f);
}

static void makeRays(int count, std::vector<Ray>& rays) {
    rays.resize(count);
    for (Ray& r : rays) {
        r.origin = {-45.0f + 90.0f * nextFloat(), 20.0f, -45.0f + 90.0f * nextFloat()};
        r.dir = {0.0f, -1.0f, 0.0f};
        r.maxT = 1e30f;
    }
}

// ---------------------------------------------------------------- GPU 建树

constexpr int kBlock = 256;
constexpr int kMaxDepth = 64;

HD inline int leafIndex(int n, int slot) { return n - 1 + slot; }

HD inline uint32_t expand10(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

HD inline uint32_t morton3(uint32_t x, uint32_t y, uint32_t z) {
    return expand10(x) | (expand10(y) << 1) | (expand10(z) << 2);
}

__device__ inline int delta(const unsigned long long* keys, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    return __clzll(keys[i] ^ keys[j]);
}

// 每图元一线程：算 Morton，同时把 order 填成 0..n-1 作为排序的值数组。
__global__ void mortonKernel(const Tri* tris, int n, Vec3 lo, Vec3 invExtent,
                             unsigned long long* keys, int* order) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Tri& t = tris[i];
    const Vec3 c{(t.a.x + t.b.x + t.c.x) * (1.0f / 3.0f),
                 (t.a.y + t.b.y + t.c.y) * (1.0f / 3.0f),
                 (t.a.z + t.b.z + t.c.z) * (1.0f / 3.0f)};
    int q[3];
    const float p[3] = {c.x - lo.x, c.y - lo.y, c.z - lo.z};
    const float e[3] = {invExtent.x, invExtent.y, invExtent.z};
    for (int k = 0; k < 3; ++k) {
        int v = static_cast<int>(p[k] * e[k] * 1024.0f);
        q[k] = v < 0 ? 0 : (v > 1023 ? 1023 : v);
    }
    const unsigned int code = morton3(static_cast<unsigned>(q[0]), static_cast<unsigned>(q[1]),
                                      static_cast<unsigned>(q[2]));
    keys[i] = (static_cast<unsigned long long>(code) << 32) | static_cast<unsigned>(i);
    order[i] = i;
}

__global__ void scatterTrisKernel(const Tri* tris, const int* sortedIdx, Tri* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = tris[sortedIdx[i]];
}

// Karras 2012：每个内部节点只看排好序的键，独立算出自己管哪段、从哪切。
__global__ void buildInternalKernel(const unsigned long long* keys, int n, Node* nodes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    const int d = delta(keys, n, i, i + 1) > delta(keys, n, i, i - 1) ? 1 : -1;
    const int sigmaMin = delta(keys, n, i, i - d);

    int lMax = 2;
    while (delta(keys, n, i, i + lMax * d) > sigmaMin) lMax *= 2;
    int l = 0;
    for (int t = lMax / 2; t >= 1; t >>= 1) {
        if (delta(keys, n, i, i + (l + t) * d) > sigmaMin) l += t;
    }
    const int j = i + l * d;
    const int lower = i < j ? i : j;
    const int upper = i < j ? j : i;

    const int deltaNode = delta(keys, n, i, j);
    int gamma = lower;
    int p = lower;
    int q = upper;
    while (p <= q) {
        const int mid = (p + q) >> 1;
        if (delta(keys, n, lower, mid) > deltaNode) {
            p = mid + 1;
        } else {
            gamma = mid - 1;
            q = mid - 1;
        }
    }

    nodes[i].left = (lower == gamma) ? leafIndex(n, lower) : gamma;
    nodes[i].right = (upper == gamma + 1) ? leafIndex(n, upper) : gamma + 1;
}

__global__ void leafInitKernel(const Tri* sortedTris, int n, Node* nodes) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const AABB b = triBounds(sortedTris[k]);
    Node& node = nodes[leafIndex(n, k)];
    node.mnx = b.mn.x;
    node.mny = b.mn.y;
    node.mnz = b.mn.z;
    node.mxx = b.mx.x;
    node.mxy = b.mx.y;
    node.mxz = b.mx.z;
    node.left = -(k + 1);
    node.right = 1;
}

__global__ void setParentsKernel(const Node* nodes, int n, int* parent) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;
    if (nodes[i].left >= 0) parent[nodes[i].left] = i;
    if (nodes[i].right >= 0) parent[nodes[i].right] = i;
}

__global__ void depthKernel(const int* parent, int nodeCount, int* depth) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodeCount) return;
    int d = 0;
    for (int p = parent[i]; p >= 0; p = parent[p]) ++d;
    depth[i] = d;
}

__global__ void countDepthsKernel(const int* depth, int nodeCount, int* counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodeCount) return;
    atomicAdd(&counts[depth[i]], 1);
}

__global__ void scatterDepthsKernel(const int* depth, int* fill, int* bucket, int nodeCount) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodeCount) return;
    const int d = depth[i];
    bucket[atomicAdd(&fill[d], 1)] = i;
}

// 按深度分桶后从深到浅 refit：第 d 层只读第 d+1 层，天然无竞争。
// 为什么必须按拓扑序：内部节点编号不保证孩子比父亲小（gamma 可能小于 i），
// 按编号扫会用到旧盒子。按 LCP 降序同样合法——孩子区间的 LCP 严格大于父亲的
// LCP，所以不会和父亲同桶；这里选深度是因为桶数天然有界。
__global__ void refitKernel(Node* nodes, const int* bucket, int count) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int i = bucket[t];
    if (nodes[i].left < 0) return;  // 叶子盒子已经由 leafInit 写好
    const Node& l = nodes[nodes[i].left];
    const Node& r = nodes[nodes[i].right];
    nodes[i].mnx = l.mnx < r.mnx ? l.mnx : r.mnx;
    nodes[i].mny = l.mny < r.mny ? l.mny : r.mny;
    nodes[i].mnz = l.mnz < r.mnz ? l.mnz : r.mnz;
    nodes[i].mxx = l.mxx > r.mxx ? l.mxx : r.mxx;
    nodes[i].mxy = l.mxy > r.mxy ? l.mxy : r.mxy;
    nodes[i].mxz = l.mxz > r.mxz ? l.mxz : r.mxz;
}

// ---------------------------------------------------------------- GPU 遍历

// 每线程一条射线。栈最多 depth+1 项，因为每层只压一个"远端"孩子。
__global__ void traceKernel(const Node* nodes, const Tri* tris, const Ray* rays, float* out,
                            int count, unsigned long long* nodeTests, unsigned long long* triTests) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const Ray r = rays[i];
    float best = r.maxT;
    unsigned long long nTests = 0;
    unsigned long long tTests = 0;

    float rootNear;
    if (!rayBox(nodes[0], r, best, rootNear)) {
        out[i] = best;
        nodeTests[i] = nTests;
        triTests[i] = tTests;
        return;
    }
    ++nTests;

    int stack[kMaxDepth];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const int ni = stack[--sp];
        const Node& node = nodes[ni];
        if (node.left < 0) {
            ++tTests;
            const int ti = -node.left - 1;
            float t;
            if (rayTri(r, tris[ti], best, t)) best = t;
            continue;
        }
        const int c0 = node.left;
        const int c1 = node.right;
        float t0;
        float t1;
        const bool h0 = rayBox(nodes[c0], r, best, t0);
        const bool h1 = rayBox(nodes[c1], r, best, t1);
        nTests += 2;
        if (h0 && h1) {
            // 近的以后弹（LIFO → 先走），远的先压
            if (t0 < t1) {
                stack[sp++] = c1;
                stack[sp++] = c0;
            } else {
                stack[sp++] = c0;
                stack[sp++] = c1;
            }
        } else if (h0) {
            stack[sp++] = c0;
        } else if (h1) {
            stack[sp++] = c1;
        }
    }
    out[i] = best;
    nodeTests[i] = nTests;
    triTests[i] = tTests;
}

// ---------------------------------------------------------------- 主机侧

static void cudaOk(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

static int blocksFor(int n) { return (n + kBlock - 1) / kBlock; }

struct DeviceMesh {
    Node* nodes = nullptr;
    Tri* tris = nullptr;
    unsigned long long* keys = nullptr;
    int* order = nullptr;
    int* parent = nullptr;
    int* depth = nullptr;
    int* counts = nullptr;
    int* fill = nullptr;
    int* bucket = nullptr;
    int nodeCount = 0;
    int maxDepth = 0;
    float mortonMs = 0, sortMs = 0, scatterMs = 0, treeMs = 0, refitMs = 0, totalMs = 0;
};

static void buildLBVH(DeviceMesh& m, const std::vector<Tri>& tris, const AABB& bounds) {
    const int n = static_cast<int>(tris.size());
    m.nodeCount = 2 * n - 1;

    cudaOk(cudaMalloc(&m.nodes, sizeof(Node) * m.nodeCount), "nodes");
    cudaOk(cudaMalloc(&m.tris, sizeof(Tri) * n), "tris");
    cudaOk(cudaMalloc(&m.keys, sizeof(unsigned long long) * n), "keys");
    cudaOk(cudaMalloc(&m.order, sizeof(int) * n), "order");
    cudaOk(cudaMalloc(&m.parent, sizeof(int) * m.nodeCount), "parent");
    cudaOk(cudaMalloc(&m.depth, sizeof(int) * m.nodeCount), "depth");
    cudaOk(cudaMalloc(&m.counts, sizeof(int) * kMaxDepth), "counts");
    cudaOk(cudaMalloc(&m.fill, sizeof(int) * kMaxDepth), "fill");
    cudaOk(cudaMalloc(&m.bucket, sizeof(int) * m.nodeCount), "bucket");

    Tri* dTrisIn = nullptr;
    cudaOk(cudaMalloc(&dTrisIn, sizeof(Tri) * n), "tris in");
    cudaOk(cudaMemcpy(dTrisIn, tris.data(), sizeof(Tri) * n, cudaMemcpyHostToDevice), "H2D tris");

    Vec3 extent{bounds.mx.x - bounds.mn.x, bounds.mx.y - bounds.mn.y, bounds.mx.z - bounds.mn.z};
    Vec3 invExtent{1.0f / extent.x, 1.0f / extent.y, 1.0f / extent.z};

    cudaEvent_t ev[8];
    for (cudaEvent_t& e : ev) cudaEventCreate(&e);
    cudaEventRecord(ev[0]);

    mortonKernel<<<blocksFor(n), kBlock>>>(dTrisIn, n, bounds.mn, invExtent, m.keys, m.order);
    cudaEventRecord(ev[1]);

    // 排序不要自己写：cub 的基数排序比手写快且省心。
    unsigned long long* dKeysSorted = nullptr;
    int* dOrderSorted = nullptr;
    cudaOk(cudaMalloc(&dKeysSorted, sizeof(unsigned long long) * n), "keys sorted");
    cudaOk(cudaMalloc(&dOrderSorted, sizeof(int) * n), "order sorted");
    void* dTemp = nullptr;
    size_t tempBytes = 0;
    cudaOk(cub::DeviceRadixSort::SortPairs(dTemp, tempBytes, m.keys, dKeysSorted, m.order,
                                           dOrderSorted, n, 0, 64),
           "sort probe");
    cudaOk(cudaMalloc(&dTemp, tempBytes), "sort temp");
    cub::DeviceRadixSort::SortPairs(dTemp, tempBytes, m.keys, dKeysSorted, m.order, dOrderSorted, n,
                                    0, 64);
    cudaOk(cudaGetLastError(), "sort");
    cudaEventRecord(ev[2]);

    scatterTrisKernel<<<blocksFor(n), kBlock>>>(dTrisIn, dOrderSorted, m.tris, n);
    cudaEventRecord(ev[3]);

    cudaOk(cudaMemset(m.parent, 0xFF, sizeof(int) * m.nodeCount), "parent -1");
    buildInternalKernel<<<blocksFor(n), kBlock>>>(dKeysSorted, n, m.nodes);
    leafInitKernel<<<blocksFor(n), kBlock>>>(m.tris, n, m.nodes);
    setParentsKernel<<<blocksFor(n), kBlock>>>(m.nodes, n, m.parent);
    cudaEventRecord(ev[4]);

    cudaOk(cudaMemset(m.counts, 0, sizeof(int) * kMaxDepth), "counts");
    depthKernel<<<blocksFor(m.nodeCount), kBlock>>>(m.parent, m.nodeCount, m.depth);
    countDepthsKernel<<<blocksFor(m.nodeCount), kBlock>>>(m.depth, m.nodeCount, m.counts);

    int counts[kMaxDepth] = {};
    cudaOk(cudaMemcpy(counts, m.counts, sizeof(counts), cudaMemcpyDeviceToHost), "counts D2H");
    int offsets[kMaxDepth] = {};
    int fillHost[kMaxDepth] = {};
    int total = 0;
    m.maxDepth = 0;
    for (int d = 0; d < kMaxDepth; ++d) {
        offsets[d] = total;
        fillHost[d] = total;
        total += counts[d];
        if (counts[d]) m.maxDepth = d;
    }
    cudaOk(cudaMemcpy(m.fill, fillHost, sizeof(fillHost), cudaMemcpyHostToDevice), "fill H2D");
    scatterDepthsKernel<<<blocksFor(m.nodeCount), kBlock>>>(m.depth, m.fill, m.bucket,
                                                            m.nodeCount);

    for (int d = m.maxDepth - 1; d >= 0; --d) {
        if (counts[d] == 0) continue;
        refitKernel<<<blocksFor(counts[d]), kBlock>>>(m.nodes, m.bucket + offsets[d], counts[d]);
    }
    cudaEventRecord(ev[5]);
    cudaOk(cudaEventSynchronize(ev[5]), "sync");

    float ms = 0;
    auto span = [&](int a, int b) {
        cudaEventElapsedTime(&ms, ev[a], ev[b]);
        return ms;
    };
    m.mortonMs = span(0, 1);
    m.sortMs = span(1, 2);
    m.scatterMs = span(2, 3);
    m.treeMs = span(3, 4);
    m.refitMs = span(4, 5);
    m.totalMs = span(0, 5);

    for (cudaEvent_t e : ev) cudaEventDestroy(e);
    cudaFree(dTemp);
    cudaFree(dKeysSorted);
    cudaFree(dOrderSorted);
    cudaFree(dTrisIn);
}

static void destroyMesh(DeviceMesh& m) {
    cudaFree(m.nodes);
    cudaFree(m.tris);
    cudaFree(m.keys);
    cudaFree(m.order);
    cudaFree(m.parent);
    cudaFree(m.depth);
    cudaFree(m.counts);
    cudaFree(m.fill);
    cudaFree(m.bucket);
    m = DeviceMesh{};
}

// CPU 暴力法：正确性真值，不做任何加速。
static float bruteForceRay(const std::vector<Tri>& tris, const Ray& r) {
    float best = r.maxT;
    for (const Tri& t : tris) {
        float hit;
        if (rayTri(r, t, best, hit)) best = hit;
    }
    return best;
}

static unsigned long long fingerprint(const std::vector<float>& t) {
    unsigned long long h = 1469598103934665603ull;
    for (float v : t) {
        const bool hit = v < 1e29f;
        uint32_t bits = 0;
        if (hit) std::memcpy(&bits, &v, sizeof(bits));
        h ^= (static_cast<unsigned long long>(hit ? 1 : 0) << 32) | bits;
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char** argv) {
    int grid = 64;
    int rayCount = 20000;
    int runs = 5;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--grid") && i + 1 < argc) {
            grid = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--rays") && i + 1 < argc) {
            rayCount = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "usage: %s [--grid N] [--rays N] [--runs N]\n", argv[0]);
            return 1;
        }
    }
    if (grid < 2 || rayCount < 1 || runs < 1) return 1;

    int device = 0;
    cudaDeviceProp prop{};
    cudaOk(cudaGetDevice(&device), "get device");
    cudaOk(cudaGetDeviceProperties(&prop, device), "props");
    std::printf("device %s sm_%d%d (%d SM)\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);

    std::vector<Tri> tris;
    AABB bounds;
    makeTerrain(grid, tris, bounds);
    std::vector<Ray> rays;
    makeRays(rayCount, rays);
    const int n = static_cast<int>(tris.size());
    std::printf("scene=terrain tris=%d rays=%d\n", n, rayCount);

    DeviceMesh mesh;
    buildLBVH(mesh, tris, bounds);
    std::printf(
        "gpu build morton=%.2f sort=%.2f scatter=%.2f tree=%.2f refit=%.2f total=%.2f ms "
        "nodes=%d depth=%d\n",
        mesh.mortonMs, mesh.sortMs, mesh.scatterMs, mesh.treeMs, mesh.refitMs, mesh.totalMs,
        mesh.nodeCount, mesh.maxDepth);
    if (mesh.maxDepth >= kMaxDepth) {
        std::fprintf(stderr, "tree deeper than %d\n", kMaxDepth);
        return 1;
    }

    Ray* dRays = nullptr;
    float* dOut = nullptr;
    unsigned long long* dNodeTests = nullptr;
    unsigned long long* dTriTests = nullptr;
    cudaOk(cudaMalloc(&dRays, sizeof(Ray) * rayCount), "rays");
    cudaOk(cudaMalloc(&dOut, sizeof(float) * rayCount), "out");
    cudaOk(cudaMalloc(&dNodeTests, sizeof(unsigned long long) * rayCount), "node stats");
    cudaOk(cudaMalloc(&dTriTests, sizeof(unsigned long long) * rayCount), "tri stats");
    cudaOk(cudaMemcpy(dRays, rays.data(), sizeof(Ray) * rayCount, cudaMemcpyHostToDevice), "H2D rays");

    std::vector<double> times;
    for (int r = 0; r < runs; ++r) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
        traceKernel<<<blocksFor(rayCount), kBlock>>>(mesh.nodes, mesh.tris, dRays, dOut, rayCount,
                                                     dNodeTests, dTriTests);
        cudaEventRecord(b);
        cudaOk(cudaEventSynchronize(b), "trace sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        times.push_back(ms);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];

    std::vector<float> hostT(rayCount);
    std::vector<unsigned long long> hostNode(rayCount);
    std::vector<unsigned long long> hostTri(rayCount);
    cudaOk(cudaMemcpy(hostT.data(), dOut, sizeof(float) * rayCount, cudaMemcpyDeviceToHost), "D2H t");
    cudaOk(cudaMemcpy(hostNode.data(), dNodeTests, sizeof(unsigned long long) * rayCount,
                      cudaMemcpyDeviceToHost),
           "D2H node");
    cudaOk(cudaMemcpy(hostTri.data(), dTriTests, sizeof(unsigned long long) * rayCount,
                      cudaMemcpyDeviceToHost),
           "D2H tri");

    unsigned long long sumNode = 0;
    unsigned long long sumTri = 0;
    for (int i = 0; i < rayCount; ++i) {
        sumNode += hostNode[i];
        sumTri += hostTri[i];
    }
    std::printf("gpu runs=%d median_ms=%.3f min_ms=%.3f max_ms=%.3f Mrays_s=%.2f\n", runs, median,
                times.front(), times.back(), rayCount / (median * 1e3));
    std::printf("gpu traversal avg_node_tests=%.1f avg_tri_tests=%.1f\n",
                static_cast<double>(sumNode) / rayCount, static_cast<double>(sumTri) / rayCount);

    // 指纹：(hit<<32) | float 位模式，与 quarry 主项目同一套定义。
    unsigned long long hits = 0;
    double sumT = 0;
    for (int i = 0; i < rayCount; ++i) {
        const bool hit = hostT[i] < 1e29f;
        hits += hit ? 1 : 0;
        sumT += hit ? hostT[i] : 0.0;
    }
    const unsigned long long hash = fingerprint(hostT);
    std::printf("fingerprint gpu rays=%d hits=%llu sum_t=%.6f hash=0x%016llx\n", rayCount, hits,
                sumT, hash);

    // 逐条对拍：CPU 暴力法关掉优化也走同一份 rayTri，因此 t 位模式应当完全一致。
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> cpuT(rayCount);
    for (int i = 0; i < rayCount; ++i) cpuT[i] = bruteForceRay(tris, rays[i]);
    auto t1 = std::chrono::steady_clock::now();
    const double cpuMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    int mismatch = 0;
    for (int i = 0; i < rayCount; ++i) {
        uint32_t a = 0;
        uint32_t b = 0;
        std::memcpy(&a, &hostT[i], sizeof(a));
        std::memcpy(&b, &cpuT[i], sizeof(b));
        if (a != b) ++mismatch;
    }
    std::printf("cpu brute %.1f ms (%.2f Mrays_s) verify rays=%d bit_mismatch=%d %s\n", cpuMs,
                rayCount / (cpuMs * 1e3), rayCount, mismatch, mismatch == 0 ? "PASS" : "FAIL");

    destroyMesh(mesh);
    cudaFree(dRays);
    cudaFree(dOut);
    cudaFree(dNodeTests);
    cudaFree(dTriTests);
    return mismatch == 0 ? 0 : 3;
}
