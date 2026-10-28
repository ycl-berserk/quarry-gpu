// Python 绑定：把 GpuBVH 的生命周期、结果数组和同步语义固定下来。
//
// 所有入口都吃/吐 numpy 数组（float32 / int32），构建与查询期间释放 GIL；
// 返回值在重新拿到 GIL 之后才创建。几何与 CUDA 类型不暴露给 Python。

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/scene.h"
#include "gpu/gpu_bvh.h"

namespace py = pybind11;

namespace {

using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
using I32Array = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

std::vector<Tri> trianglesFromArray(const F32Array& arr) {
    if (arr.ndim() != 3 || arr.shape(1) != 3 || arr.shape(2) != 3) {
        throw std::invalid_argument("triangles must have shape (M, 3, 3)");
    }
    const int m = static_cast<int>(arr.shape(0));
    if (m < 2) throw std::invalid_argument("at least 2 triangles are required");
    std::vector<Tri> tris(static_cast<size_t>(m));
    auto r = arr.unchecked<3>();
    for (int i = 0; i < m; ++i) {
        tris[static_cast<size_t>(i)].a = {r(i, 0, 0), r(i, 0, 1), r(i, 0, 2)};
        tris[static_cast<size_t>(i)].b = {r(i, 1, 0), r(i, 1, 1), r(i, 1, 2)};
        tris[static_cast<size_t>(i)].c = {r(i, 2, 0), r(i, 2, 1), r(i, 2, 2)};
    }
    return tris;
}

std::vector<Tri> trianglesFromMesh(const F32Array& vertices, const I32Array& indices) {
    if (vertices.ndim() != 2 || vertices.shape(1) != 3) {
        throw std::invalid_argument("vertices must have shape (N, 3)");
    }
    if (indices.ndim() != 2 || indices.shape(1) != 3) {
        throw std::invalid_argument("indices must have shape (M, 3)");
    }
    const int n = static_cast<int>(vertices.shape(0));
    const int m = static_cast<int>(indices.shape(0));
    if (m < 2) throw std::invalid_argument("at least 2 triangles are required");

    std::vector<Vec3> verts(static_cast<size_t>(n));
    auto v = vertices.unchecked<2>();
    for (int i = 0; i < n; ++i) verts[static_cast<size_t>(i)] = {v(i, 0), v(i, 1), v(i, 2)};

    std::vector<Tri> tris(static_cast<size_t>(m));
    auto idx = indices.unchecked<2>();
    for (int i = 0; i < m; ++i) {
        const int a = idx(i, 0);
        const int b = idx(i, 1);
        const int c = idx(i, 2);
        if (a < 0 || b < 0 || c < 0 || a >= n || b >= n || c >= n) {
            throw std::invalid_argument("indices out of range");
        }
        tris[static_cast<size_t>(i)] = {verts[static_cast<size_t>(a)], verts[static_cast<size_t>(b)],
                                        verts[static_cast<size_t>(c)]};
    }
    return tris;
}

std::vector<Tri> trianglesFromIndices(const std::vector<Vec3>& verts,
                                      const std::vector<int32_t>& indices) {
    const int n = static_cast<int>(verts.size());
    const size_t m = indices.size() / 3;
    std::vector<Tri> tris(m);
    for (size_t i = 0; i < m; ++i) {
        const int a = indices[i * 3];
        const int b = indices[i * 3 + 1];
        const int c = indices[i * 3 + 2];
        if (a < 0 || b < 0 || c < 0 || a >= n || b >= n || c >= n) {
            throw std::invalid_argument("indices out of range");
        }
        tris[i] = {verts[static_cast<size_t>(a)], verts[static_cast<size_t>(b)],
                   verts[static_cast<size_t>(c)]};
    }
    return tris;
}

std::vector<Vec3> pointsFromArray(const F32Array& arr, const char* what) {
    if (arr.ndim() != 2 || arr.shape(1) != 3) {
        throw std::invalid_argument(std::string(what) + " must have shape (N, 3)");
    }
    const int n = static_cast<int>(arr.shape(0));
    std::vector<Vec3> out(static_cast<size_t>(n));
    auto r = arr.unchecked<2>();
    for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = {r(i, 0), r(i, 1), r(i, 2)};
    return out;
}

py::array_t<float> floatArray(const std::vector<float>& v) {
    py::array_t<float> out(static_cast<py::ssize_t>(v.size()));
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
    return out;
}

py::array_t<int32_t> intArray(const std::vector<int32_t>& v) {
    py::array_t<int32_t> out(static_cast<py::ssize_t>(v.size()));
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(int32_t));
    return out;
}

class PyScene {
public:
    explicit PyScene(const F32Array& triangles) : tris_(trianglesFromArray(triangles)) { build(); }

    PyScene(const F32Array& vertices, const I32Array& indices)
        : tris_(trianglesFromMesh(vertices, indices)) {
        indices_.assign(indices.data(), indices.data() + indices.size());
        build();
    }

    ~PyScene() { destroyBVHGPU(bvh_); }

    PyScene(const PyScene&) = delete;
    PyScene& operator=(const PyScene&) = delete;

    int triangleCount() const { return bvh_.triCount; }
    int nodeCount() const { return bvh_.nodeCount; }
    int depth() const { return bvh_.depth; }
    double buildMs() const { return buildMs_; }
    double lastQueryMs() const { return lastQueryMs_; }
    double refitMs() const { return refitMs_; }

    py::dict refitStats() const {
        py::dict out;
        out["upload_ms"] = refitTimings_.uploadMs;
        out["scatter_ms"] = refitTimings_.gatherMs;
        out["tree_ms"] = refitTimings_.treeMs;
        out["total_ms"] = refitTimings_.totalMs;
        out["updated"] = refitTimings_.updated;
        return out;
    }

    py::dict bounds() const {
        py::dict out;
        out["min"] = py::make_tuple(bounds_.mn.x, bounds_.mn.y, bounds_.mn.z);
        out["max"] = py::make_tuple(bounds_.mx.x, bounds_.mx.y, bounds_.mx.z);
        return out;
    }

    py::dict buildStats() const {
        py::dict out;
        out["morton_ms"] = timings_.mortonMs;
        out["sort_ms"] = timings_.sortMs;
        out["scatter_ms"] = timings_.scatterMs;
        out["tree_ms"] = timings_.treeMs;
        out["refit_ms"] = timings_.refitMs;
        out["total_ms"] = timings_.totalMs;
        return out;
    }

    std::string validate() const { return validateBVHGPU(bvh_, sceneView_); }

    py::tuple raycast(const F32Array& origins, const F32Array& dirs,
                      float tmax = std::numeric_limits<float>::infinity()) {
        const std::vector<Vec3> o = pointsFromArray(origins, "origins");
        const std::vector<Vec3> d = pointsFromArray(dirs, "dirs");
        if (o.size() != d.size()) throw std::invalid_argument("origins and dirs must match");

        std::vector<Ray> rays(o.size());
        for (size_t i = 0; i < o.size(); ++i) rays[i] = {o[i], d[i], 1e-6f, tmax};

        std::vector<uint64_t> perRay;
        double ms = 0.0;
        {
            py::gil_scoped_release release;
            GpuRayBuffer buf = uploadRaysGPU(rays);
            try {
                ms = raycastGPU(bvh_, buf, false);
                summarizeGPU(buf, &perRay);
            } catch (...) {
                destroyRayBufferGPU(buf);
                throw;
            }
            destroyRayBufferGPU(buf);
        }
        lastQueryMs_ = ms;

        py::array_t<bool> hit(static_cast<py::ssize_t>(perRay.size()));
        py::array_t<float> t(static_cast<py::ssize_t>(perRay.size()));
        auto h = hit.mutable_unchecked<1>();
        auto tv = t.mutable_unchecked<1>();
        for (size_t i = 0; i < perRay.size(); ++i) {
            const bool didHit = (perRay[i] >> 32) != 0;
            h(static_cast<py::ssize_t>(i)) = didHit;
            if (didHit) {
                const uint32_t bits = static_cast<uint32_t>(perRay[i]);
                std::memcpy(&tv(static_cast<py::ssize_t>(i)), &bits, sizeof(float));
            } else {
                tv(static_cast<py::ssize_t>(i)) = std::numeric_limits<float>::infinity();
            }
        }
        return py::make_tuple(hit, t);
    }

    py::tuple closest(const F32Array& points) {
        const std::vector<Vec3> pts = pointsFromArray(points, "points");
        std::vector<GpuClosestResult> res;
        double ms = 0.0;
        {
            py::gil_scoped_release release;
            GpuPointBuffer buf = uploadPointsGPU(pts, 1);
            try {
                ms = closestPointGPU(bvh_, buf, false);
                res = readQueryResultsGPU(buf);
            } catch (...) {
                destroyPointBufferGPU(buf);
                throw;
            }
            destroyPointBufferGPU(buf);
        }
        lastQueryMs_ = ms;

        std::vector<float> dist2(res.size());
        std::vector<int32_t> tri(res.size());
        for (size_t i = 0; i < res.size(); ++i) {
            const uint32_t bits = res[i].dist2Bits;
            std::memcpy(&dist2[i], &bits, sizeof(float));
            tri[i] = res[i].triangle;
        }
        return py::make_tuple(floatArray(dist2), intArray(tri));
    }

    py::tuple knn(const F32Array& points, int k) {
        if (k < 1 || k > 32) throw std::invalid_argument("k must be in [1, 32]");
        const std::vector<Vec3> pts = pointsFromArray(points, "points");
        std::vector<GpuClosestResult> res;
        double ms = 0.0;
        {
            py::gil_scoped_release release;
            GpuPointBuffer buf = uploadPointsGPU(pts, k);
            try {
                ms = kNearestGPU(bvh_, buf, k, false);
                res = readQueryResultsGPU(buf);
            } catch (...) {
                destroyPointBufferGPU(buf);
                throw;
            }
            destroyPointBufferGPU(buf);
        }
        lastQueryMs_ = ms;

        const py::ssize_t n = static_cast<py::ssize_t>(pts.size());
        py::array_t<float> dist2({n, static_cast<py::ssize_t>(k)});
        py::array_t<int32_t> tri({n, static_cast<py::ssize_t>(k)});
        auto dv = dist2.mutable_unchecked<2>();
        auto tv = tri.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                const GpuClosestResult& h = res[static_cast<size_t>(i) * k + j];
                uint32_t bits = h.dist2Bits;
                float d = 0.0f;
                std::memcpy(&d, &bits, sizeof(float));
                dv(i, j) = d;
                tv(i, j) = h.triangle;
            }
        }
        return py::make_tuple(dist2, tri);
    }

    py::tuple radius(const F32Array& points, float radius) {
        if (!(radius >= 0.0f)) throw std::invalid_argument("radius must be >= 0");
        const std::vector<Vec3> pts = pointsFromArray(points, "points");
        std::vector<std::vector<ClosestHit>> hits;
        double ms = 0.0;
        {
            py::gil_scoped_release release;
            GpuPointBuffer buf = uploadPointsGPU(pts, 1);
            GpuRadiusBuffer out;
            try {
                ms = radiusQueryGPU(bvh_, buf, radius, out, false);
                hits = readRadiusResultsGPU(out);
            } catch (...) {
                destroyRadiusBufferGPU(out);
                destroyPointBufferGPU(buf);
                throw;
            }
            destroyRadiusBufferGPU(out);
            destroyPointBufferGPU(buf);
        }
        lastQueryMs_ = ms;

        const py::ssize_t n = static_cast<py::ssize_t>(pts.size());
        py::array_t<int64_t> offsets(n + 1);
        auto off = offsets.mutable_unchecked<1>();
        off(0) = 0;
        size_t total = 0;
        for (py::ssize_t i = 0; i < n; ++i) {
            total += hits[static_cast<size_t>(i)].size();
            off(i + 1) = static_cast<int64_t>(total);
        }
        py::array_t<float> dist2(static_cast<py::ssize_t>(total));
        py::array_t<int32_t> tri(static_cast<py::ssize_t>(total));
        auto dv = dist2.mutable_unchecked<1>();
        auto tv = tri.mutable_unchecked<1>();
        size_t at = 0;
        for (py::ssize_t i = 0; i < n; ++i) {
            for (const ClosestHit& h : hits[static_cast<size_t>(i)]) {
                dv(static_cast<py::ssize_t>(at)) = h.dist2;
                tv(static_cast<py::ssize_t>(at)) = h.triangle;
                ++at;
            }
        }
        return py::make_tuple(offsets, dist2, tri);
    }

private:
    void build() {
        sceneView_.tris = tris_;
        sceneView_.updateBounds();
        bounds_ = sceneView_.bounds;
        {
            py::gil_scoped_release release;
            bvh_ = buildBVHGPU(sceneView_, &timings_);
        }
        buildMs_ = timings_.totalMs;
    }

    // dirty 非 None 时只上传这些场景三角形（未列出的三角形在树里保持旧值），
    // 调用方必须保证"没列出来的都没动"——这是增量接口的合同，不会被逐位校验兜住。
    void update(std::vector<Tri> tris, const py::object& dirty) {
        if (static_cast<int>(tris.size()) != bvh_.triCount) {
            throw std::invalid_argument("update must keep the triangle count; rebuild instead");
        }
        std::vector<int> ids;
        const bool partial = !dirty.is_none();
        if (partial) {
            const I32Array arr = dirty.cast<I32Array>();
            ids.assign(arr.data(), arr.data() + arr.size());
        }

        tris_ = std::move(tris);
        sceneView_.tris = tris_;
        sceneView_.updateBounds();
        bounds_ = sceneView_.bounds;

        {
            py::gil_scoped_release release;
            refitMs_ = partial ? refitBVHGPU(bvh_, sceneView_, ids, &refitTimings_)
                               : refitBVHGPU(bvh_, sceneView_, &refitTimings_);
        }
    }

public:
    void updateTriangles(const F32Array& triangles, const py::object& dirty) {
        update(trianglesFromArray(triangles), dirty);
    }

    void updateVertices(const F32Array& vertices, const py::object& dirty) {
        if (indices_.empty()) {
            throw std::invalid_argument("Scene was built from triangles; use update_triangles");
        }
        const std::vector<Vec3> verts = pointsFromArray(vertices, "vertices");
        update(trianglesFromIndices(verts, indices_), dirty);
    }

private:
    std::vector<Tri> tris_;
    std::vector<int32_t> indices_;  // 只有 (vertices, indices) 构造时才有
    ::Scene sceneView_;
    AABB bounds_;
    GpuBVH bvh_;
    GpuBuildTimings timings_{};
    GpuRefitTimings refitTimings_{};
    double buildMs_ = 0.0;
    double lastQueryMs_ = 0.0;
    double refitMs_ = 0.0;
};

py::tuple demoMesh(const std::string& kind, int grid, int randomTris, uint64_t seed) {
    SceneSpec spec;
    spec.kind = kind;
    spec.grid = grid;
    spec.randomTris = randomTris;
    spec.seed = seed;
    const ::Scene scene = buildScene(spec);

    const py::ssize_t m = static_cast<py::ssize_t>(scene.tris.size());
    py::array_t<float> vertices({m * 3, static_cast<py::ssize_t>(3)});
    py::array_t<int32_t> indices({m, static_cast<py::ssize_t>(3)});
    auto v = vertices.mutable_unchecked<2>();
    auto idx = indices.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < m; ++i) {
        const Tri& t = scene.tris[static_cast<size_t>(i)];
        const Vec3 vts[3] = {t.a, t.b, t.c};
        for (int j = 0; j < 3; ++j) {
            v(i * 3 + j, 0) = vts[j].x;
            v(i * 3 + j, 1) = vts[j].y;
            v(i * 3 + j, 2) = vts[j].z;
            idx(i, j) = static_cast<int32_t>(i * 3 + j);
        }
    }
    return py::make_tuple(vertices, indices);
}

}  // namespace

PYBIND11_MODULE(quarry, m) {
    m.doc() = "GPU spatial queries: LBVH build, ray casting, closest / kNN / radius";

    m.def(
        "describe",
        []() {
            std::string why;
            return gpuDescribe(&why) ? why : std::string();
        },
        "CUDA device string, or an empty string when no GPU is available");

    m.def("demo_mesh", &demoMesh, py::arg("kind") = "terrain", py::arg("grid") = 64,
          py::arg("random_tris") = 0, py::arg("seed") = 1,
          "Return (vertices (N,3) float32, indices (M,3) int32) for a demo scene");

    py::class_<PyScene>(m, "Scene", "Static triangle mesh with a GPU LBVH")
        .def(py::init<const F32Array&>(), py::arg("triangles"),
             "Build from triangles with shape (M, 3, 3) float32")
        .def(py::init<const F32Array&, const I32Array&>(), py::arg("vertices"),
             py::arg("indices"), "Build from shared vertices (N,3) and indices (M,3)")
        .def_property_readonly("triangle_count", &PyScene::triangleCount)
        .def_property_readonly("node_count", &PyScene::nodeCount)
        .def_property_readonly("depth", &PyScene::depth)
        .def_property_readonly("build_ms", &PyScene::buildMs,
                               "Wall-clock build time; the first Scene in a process also "
                               "pays CUDA module load")
        .def_property_readonly("last_query_ms", &PyScene::lastQueryMs,
                               "Kernel time of the most recent query (CUDA events, no copies)")
        .def_property_readonly("refit_ms", &PyScene::refitMs,
                               "Time of the most recent update, including the H2D upload")
        .def_property_readonly("refit_stats", &PyScene::refitStats,
                               "Breakdown of the most recent update: upload_ms, scatter_ms, "
                               "tree_ms, total_ms, updated")
        .def_property_readonly("bounds", &PyScene::bounds)
        .def_property_readonly("build_stats", &PyScene::buildStats)
        .def("validate", &PyScene::validate, "Check tree invariants; empty string means pass")
        .def("update_triangles", &PyScene::updateTriangles, py::arg("triangles"),
             py::arg("dirty") = py::none(),
             "Refit in place: triangle i must still describe the same surface element. "
             "dirty is an int32 array of scene indices to upload; omit it to upload everything")
        .def("update_vertices", &PyScene::updateVertices, py::arg("vertices"),
             py::arg("dirty") = py::none(),
             "Same as update_triangles, but rebuilding the triangles from the mesh that was "
             "used to construct this Scene")
        .def("raycast", &PyScene::raycast, py::arg("origins"), py::arg("dirs"),
             py::arg("tmax") = std::numeric_limits<float>::infinity(),
             "Return (hit bool[N], t float32[N]); t is inf on a miss. dirs need not be normalized")
        .def("closest", &PyScene::closest, py::arg("points"),
             "Return (dist2 float32[N], triangle int32[N]); triangle is -1 on a miss")
        .def("knn", &PyScene::knn, py::arg("points"), py::arg("k"),
             "Return (dist2 float32[N,k], triangle int32[N,k]) sorted by (dist2, index); "
             "missing slots are -1")
        .def("radius", &PyScene::radius, py::arg("points"), py::arg("radius"),
             "Return CSR (offsets int64[N+1], dist2 float32[total], triangle int32[total]) "
             "sorted by (dist2, index) inside each segment");
}
