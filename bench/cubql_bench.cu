// cuBQL 原生对照：同一份 Quarry 场景与查询集，测 cuBQL 的构建与遍历。
//
// 只做计时与校验和，不与 quarry 同进程链接；两边由 head2head_cubql.py 配对运行。
// 查询内核：closest 用 cuBQL 的 CPAT，ray 用 shrinkingRayQuery + 自写 Möller-Trumbore。

#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#define CUBQL_TRIANGLE_CPAT_IMPLEMENTATION 1

#include "cuBQL/bvh.h"
#include "cuBQL/queries/triangleData/Triangle.h"
#include "cuBQL/queries/triangleData/closestPointOnAnyTriangle.h"
#include "cuBQL/traversal/rayQueries.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

using cuBQL::vec3f;
using cuBQL::vec3i;
using cuBQL::box3f;
using cuBQL::bvh3f;
using cuBQL::ray3f;

static void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "cuda error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T>
static T *uploadRaw(const void *src, size_t count, const char *what) {
    T *d = nullptr;
    check(cudaMalloc((void **)&d, count * sizeof(T)), what);
    check(cudaMemcpy(d, src, count * sizeof(T), cudaMemcpyHostToDevice), what);
    return d;
}

struct Scene {
    std::vector<float> verts;
    std::vector<int> idx;
    std::vector<float> origins;
    std::vector<float> dirs;
};

static Scene loadScene(const char *path) {
    FILE *f = std::fopen(path, "rb");
    if (!f) { std::perror(path); std::exit(1); }
    int nv, nt, nq;
    if (std::fread(&nv, 4, 1, f) != 1 ||
        std::fread(&nt, 4, 1, f) != 1 ||
        std::fread(&nq, 4, 1, f) != 1) {
        std::fprintf(stderr, "bad header\n"); std::exit(1);
    }
    Scene s;
    s.verts.resize(static_cast<size_t>(nv) * 3);
    s.idx.resize(static_cast<size_t>(nt) * 3);
    s.origins.resize(static_cast<size_t>(nq) * 3);
    s.dirs.resize(static_cast<size_t>(nq) * 3);
    auto read = [&](auto &v) {
        if (std::fread(v.data(), sizeof(v[0]), v.size(), f) != v.size()) {
            std::fprintf(stderr, "short read\n"); std::exit(1);
        }
    };
    read(s.verts); read(s.idx); read(s.origins); read(s.dirs);
    std::fclose(f);
    return s;
}

__global__ void makeTriangles(cuBQL::Triangle *tris, const vec3f *verts,
                              const vec3i *idx, int nt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nt) return;
    vec3i t = idx[i];
    tris[i] = cuBQL::Triangle(verts[t.x], verts[t.y], verts[t.z]);
}

__global__ void makeBoxes(box3f *boxes, const cuBQL::Triangle *tris, int nt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nt) return;
    boxes[i] = tris[i].bounds();
}

__global__ void closestKernel(bvh3f bvh, const cuBQL::Triangle *tris,
                              const vec3f *pts, float *dist2, int *tri,
                              int nq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nq) return;
    cuBQL::triangles::CPAT cpat;
    cpat.runQuery(tris, bvh, pts[i]);
    dist2[i] = cpat.sqrDist;
    tri[i] = cpat.triangleIdx;
}

// Möller–Trumbore，命中则返回 t，否则返回 inf
__device__ inline float rayTriangle(const vec3f &o, const vec3f &d,
                                    const vec3f &a, const vec3f &b,
                                    const vec3f &c) {
    vec3f e1(b.x - a.x, b.y - a.y, b.z - a.z);
    vec3f e2(c.x - a.x, c.y - a.y, c.z - a.z);
    vec3f p(d.y * e2.z - d.z * e2.y,
            d.z * e2.x - d.x * e2.z,
            d.x * e2.y - d.y * e2.x);
    float det = e1.x * p.x + e1.y * p.y + e1.z * p.z;
    if (det > -1e-12f && det < 1e-12f) return CUBQL_INF;
    float inv = 1.0f / det;
    vec3f s(o.x - a.x, o.y - a.y, o.z - a.z);
    float u = (s.x * p.x + s.y * p.y + s.z * p.z) * inv;
    if (u < 0.0f || u > 1.0f) return CUBQL_INF;
    vec3f q(s.y * e1.z - s.z * e1.y,
            s.z * e1.x - s.x * e1.z,
            s.x * e1.y - s.y * e1.x);
    float v = (d.x * q.x + d.y * q.y + d.z * q.z) * inv;
    if (v < 0.0f || u + v > 1.0f) return CUBQL_INF;
    float t = (e2.x * q.x + e2.y * q.y + e2.z * q.z) * inv;
    return t > 1e-6f ? t : CUBQL_INF;
}

__global__ void rayKernel(bvh3f bvh, const cuBQL::Triangle *tris,
                          const vec3f *origins, const vec3f *dirs,
                          float *t_out, int nq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nq) return;
    ray3f ray(origins[i], dirs[i], 1e-6f, CUBQL_INF);
    auto perPrim = [&](uint32_t prim) -> float {
        const cuBQL::Triangle &tr = tris[prim];
        float t = rayTriangle(ray.origin, ray.direction, tr.a, tr.b, tr.c);
        return t < ray.tMax ? t : ray.tMax;
    };
    cuBQL::shrinkingRayQuery::forEachPrim(perPrim, bvh, ray);
    t_out[i] = ray.tMax;
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s scene.bin closest|ray [--sah] [--reps N]\n",
                     argv[0]);
        return 1;
    }
    const char *path = argv[1];
    std::string mode = argv[2];
    bool useSAH = false;
    int reps = 5;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sah")) useSAH = true;
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    }

    Scene s = loadScene(path);
    const int nv = static_cast<int>(s.verts.size() / 3);
    const int nt = static_cast<int>(s.idx.size() / 3);
    const int nq = static_cast<int>(s.origins.size() / 3);

    vec3f *d_verts = uploadRaw<vec3f>(s.verts.data(), s.verts.size() / 3, "verts");
    vec3i *d_idx = uploadRaw<vec3i>(s.idx.data(), s.idx.size() / 3, "indices");
    vec3f *d_origins = uploadRaw<vec3f>(s.origins.data(), s.origins.size() / 3,
                                        "origins");
    vec3f *d_dirs = uploadRaw<vec3f>(s.dirs.data(), s.dirs.size() / 3, "dirs");
    (void)nv;

    cuBQL::Triangle *d_tris = nullptr;
    check(cudaMalloc((void **)&d_tris, static_cast<size_t>(nt) * sizeof(cuBQL::Triangle)),
          "tris");
    makeTriangles<<<(nt + 255) / 256, 256>>>(d_tris, d_verts, d_idx, nt);
    check(cudaGetLastError(), "makeTriangles");

    cuBQL::BuildConfig cfg;
    if (useSAH) cfg.enableSAH();

    std::vector<float> build_ms, query_ms;
    bvh3f bvh;
    for (int r = 0; r < reps; ++r) {
        box3f *d_boxes = nullptr;
        check(cudaMalloc((void **)&d_boxes, static_cast<size_t>(nt) * sizeof(box3f)),
              "boxes");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0);
        makeBoxes<<<(nt + 255) / 256, 256>>>(d_boxes, d_tris, nt);
        cuBQL::gpuBuilder(bvh, d_boxes, nt, cfg);
        check(cudaGetLastError(), "gpuBuilder");
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        build_ms.push_back(ms);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        check(cudaFree(d_boxes), "free boxes");
        if (r + 1 < reps) cuBQL::cuda::free(bvh);
    }

    float *d_dist2 = nullptr;
    int *d_tri = nullptr;
    float *d_t = nullptr;
    check(cudaMalloc((void **)&d_dist2, static_cast<size_t>(nq) * sizeof(float)), "dist2");
    check(cudaMalloc((void **)&d_tri, static_cast<size_t>(nq) * sizeof(int)), "tri");
    check(cudaMalloc((void **)&d_t, static_cast<size_t>(nq) * sizeof(float)), "t");

    auto launch = [&]() {
        if (mode == "closest") {
            closestKernel<<<(nq + 127) / 128, 128>>>(bvh, d_tris, d_origins,
                                                     d_dist2, d_tri, nq);
        } else {
            rayKernel<<<(nq + 127) / 128, 128>>>(bvh, d_tris, d_origins, d_dirs,
                                                 d_t, nq);
        }
        check(cudaGetLastError(), "query launch");
    };

    launch();  // warm-up
    for (int r = 0; r < reps; ++r) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0);
        launch();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        query_ms.push_back(ms);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }

    std::vector<float> h_val(nq);
    std::vector<int> h_tri(nq);
    if (mode == "closest") {
        check(cudaMemcpy(h_val.data(), d_dist2, static_cast<size_t>(nq) * sizeof(float),
                         cudaMemcpyDeviceToHost), "copy dist2");
        check(cudaMemcpy(h_tri.data(), d_tri, static_cast<size_t>(nq) * sizeof(int),
                         cudaMemcpyDeviceToHost), "copy tri");
    } else {
        check(cudaMemcpy(h_val.data(), d_t, static_cast<size_t>(nq) * sizeof(float),
                         cudaMemcpyDeviceToHost), "copy t");
        for (int i = 0; i < nq; ++i) h_tri[i] = h_val[i] < 1e30f ? 1 : 0;
    }

    double sum = 0.0;
    int hits = 0;
    for (int i = 0; i < nq; ++i) {
        if (mode == "closest" ? h_tri[i] >= 0 : h_tri[i]) {
            sum += mode == "closest" ? std::sqrt(static_cast<double>(h_val[i]))
                                     : static_cast<double>(h_val[i]);
            ++hits;
        }
    }

    std::printf("mode=%s builder=%s tris=%d queries=%d\n", mode.c_str(),
                useSAH ? "sah" : "spatial-median", nt, nq);
    std::printf("cubql_build_ms=%.3f cubql_query_ms=%.3f\n",
                median(build_ms), median(query_ms));
    std::printf("hits=%d sum=%.6f\n", hits, sum);

    cuBQL::cuda::free(bvh);
    check(cudaFree(d_verts), "free verts");
    check(cudaFree(d_idx), "free idx");
    check(cudaFree(d_origins), "free origins");
    check(cudaFree(d_dirs), "free dirs");
    check(cudaFree(d_tris), "free tris");
    check(cudaFree(d_dist2), "free dist2");
    check(cudaFree(d_tri), "free tri");
    check(cudaFree(d_t), "free t");
    return 0;
}
