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

#include "core/gaussian.h"
#include "core/point_cloud.h"
#include "core/scene.h"
#include "gpu/gpu_bvh.h"

#ifndef QUARRY_VERSION
#define QUARRY_VERSION "0.0.0"
#endif

namespace py = pybind11;

namespace {

using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
using I32Array = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

// 叶子载荷的种类：三角形、点、高斯椭球。查询骨架与输出格式完全相同，
// 三者的区别只在 gpu/prim_ops.h 里的距离函数。
enum class LeafKind { Triangle, Vertex, Gaussian };

QueryOrder parseQueryOrder(const std::string& s) {
    if (s == "input") return QueryOrder::Input;
    if (s == "morton") return QueryOrder::Morton;
    throw std::invalid_argument("order must be 'input' or 'morton'");
}

bool parseSahBuilder(const std::string& s) {
    if (s == "lbvh") return false;
    if (s == "sah") return true;
    throw std::invalid_argument("builder must be 'lbvh' or 'sah'");
}

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

// 三个数组拼成 Gaussian 列表：均值 (N,3)、四元数 (N,4)（xyzw，可不归一化）、
// 半轴 (N,3)。拆成三个数组而不是一个 (N,10)，是为了让调用方按列自然读写。
std::vector<Gaussian> gaussiansFromArrays(const F32Array& means, const F32Array& rot,
                                          const F32Array& scale) {
    for (const auto* pair : {&means, &scale}) {
        const F32Array& arr = *pair;
        if (arr.ndim() != 2 || arr.shape(1) != 3) {
            throw std::invalid_argument("means and scales must have shape (N, 3)");
        }
    }
    if (rot.ndim() != 2 || rot.shape(1) != 4) {
        throw std::invalid_argument("rotations must have shape (N, 4) as xyzw quaternions");
    }
    const int n = static_cast<int>(means.shape(0));
    if (static_cast<int>(rot.shape(0)) != n || static_cast<int>(scale.shape(0)) != n) {
        throw std::invalid_argument("means, rotations and scales must have the same length");
    }

    std::vector<Gaussian> out(static_cast<size_t>(n));
    auto m = means.unchecked<2>();
    auto q = rot.unchecked<2>();
    auto s = scale.unchecked<2>();
    for (int i = 0; i < n; ++i) {
        Gaussian& g = out[static_cast<size_t>(i)];
        g.mean = {m(i, 0), m(i, 1), m(i, 2)};
        g.rot = {q(i, 0), q(i, 1), q(i, 2), q(i, 3)};
        g.scale = {s(i, 0), s(i, 1), s(i, 2)};
        if (!(g.scale.x > 0.0f) || !(g.scale.y > 0.0f) || !(g.scale.z > 0.0f)) {
            throw std::invalid_argument("semi-axes must be positive");
        }
    }
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

py::dict boundsDict(const AABB& box) {
    py::dict out;
    out["min"] = py::make_tuple(box.mn.x, box.mn.y, box.mn.z);
    out["max"] = py::make_tuple(box.mx.x, box.mx.y, box.mx.z);
    return out;
}

// 三条拓扑更新共用一份编辑脚本：deleted 是旧下标（升序），inserted / moved 是新下标，
// 末尾追加的个数由"新数量 - 旧数量 + 删除数 - 插入数"倒推，调用方不用自己数。
GpuTopoEdit topoEditFromArrays(const py::object& deleted, const py::object& moved,
                               const py::object& inserted, long long oldCount,
                               long long newCount) {
    GpuTopoEdit edit;
    if (!deleted.is_none()) {
        const I32Array arr = deleted.cast<I32Array>();
        edit.deleted.assign(arr.data(), arr.data() + arr.size());
    }
    if (!moved.is_none()) {
        const I32Array arr = moved.cast<I32Array>();
        edit.moved.assign(arr.data(), arr.data() + arr.size());
    }
    if (!inserted.is_none()) {
        const I32Array arr = inserted.cast<I32Array>();
        edit.inserted.assign(arr.data(), arr.data() + arr.size());
    }
    const long long appended = newCount - oldCount + static_cast<long long>(edit.deleted.size()) -
                               static_cast<long long>(edit.inserted.size());
    if (appended < 0 || appended > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "new primitive count must be old count - deleted + inserted + appended");
    }
    edit.appended = static_cast<int>(appended);
    return edit;
}

py::dict topoTimingsDict(const GpuTopoUpdateTimings& t) {
    py::dict out;
    out["upload_ms"] = t.uploadMs;
    out["plan_ms"] = t.planMs;
    out["surgery_ms"] = t.surgeryMs;
    out["relayout_ms"] = t.relayoutMs;
    out["merge_ms"] = t.mergeMs;
    out["wide_ms"] = t.wideMs;
    out["total_ms"] = t.totalMs;
    out["refit_ms"] = t.refitMs;
    out["deleted"] = t.deleted;
    out["moved"] = t.moved;
    out["inserted"] = t.inserted;
    out["appended"] = t.appended;
    out["surgery_roots"] = t.surgeryRoots;
    out["rebuilt_leaves"] = t.rebuiltLeaves;
    out["new_depth"] = t.newDepth;
    out["full_rebuild"] = t.fullRebuild;
    return out;
}

// ---- 三条查询的"跑 + 拷回 + 打包"流水线 ----
// kind 只决定叶子载荷与距离函数；三条路径的输出格式完全一样，
// 所以打包代码只写一份。
py::tuple closestRun(GpuBVH& bvh, const F32Array& points, LeafKind kind, QueryOrder order,
                     double* msOut) {
    const std::vector<Vec3> pts = pointsFromArray(points, "points");
    std::vector<GpuClosestResult> res;
    double ms = 0.0;
    {
        py::gil_scoped_release release;
        GpuPointBuffer buf = uploadPointsGPU(pts, 1);
        try {
            switch (kind) {
                case LeafKind::Triangle: ms = closestPointGPU(bvh, buf, false, order); break;
                case LeafKind::Vertex: ms = closestVertexGPU(bvh, buf, false, order); break;
                case LeafKind::Gaussian: ms = closestGaussianGPU(bvh, buf, false, order); break;
            }
            res = readQueryResultsGPU(buf);
        } catch (...) {
            destroyPointBufferGPU(buf);
            throw;
        }
        destroyPointBufferGPU(buf);
    }
    *msOut = ms;

    std::vector<float> dist2(res.size());
    std::vector<int32_t> idx(res.size());
    for (size_t i = 0; i < res.size(); ++i) {
        const uint32_t bits = res[i].dist2Bits;
        std::memcpy(&dist2[i], &bits, sizeof(float));
        idx[i] = res[i].triangle;
    }
    return py::make_tuple(floatArray(dist2), intArray(idx));
}

py::tuple knnRun(GpuBVH& bvh, const F32Array& points, int k, LeafKind kind, QueryOrder order,
                 double* msOut) {
    if (k < 1 || k > 32) throw std::invalid_argument("k must be in [1, 32]");
    const std::vector<Vec3> pts = pointsFromArray(points, "points");
    std::vector<GpuClosestResult> res;
    double ms = 0.0;
    {
        py::gil_scoped_release release;
        GpuPointBuffer buf = uploadPointsGPU(pts, k);
        try {
            switch (kind) {
                case LeafKind::Triangle: ms = kNearestGPU(bvh, buf, k, false, order); break;
                case LeafKind::Vertex: ms = kNearestVertexGPU(bvh, buf, k, false, order); break;
                case LeafKind::Gaussian: ms = kNearestGaussianGPU(bvh, buf, k, false, order); break;
            }
            res = readQueryResultsGPU(buf);
        } catch (...) {
            destroyPointBufferGPU(buf);
            throw;
        }
        destroyPointBufferGPU(buf);
    }
    *msOut = ms;

    const py::ssize_t n = static_cast<py::ssize_t>(pts.size());
    py::array_t<float> dist2({n, static_cast<py::ssize_t>(k)});
    py::array_t<int32_t> idx({n, static_cast<py::ssize_t>(k)});
    auto dv = dist2.mutable_unchecked<2>();
    auto tv = idx.mutable_unchecked<2>();
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
    return py::make_tuple(dist2, idx);
}

py::tuple radiusRun(GpuBVH& bvh, const F32Array& points, float radius, LeafKind kind,
                    QueryOrder order, double* msOut) {
    if (!(radius >= 0.0f)) throw std::invalid_argument("radius must be >= 0");
    const std::vector<Vec3> pts = pointsFromArray(points, "points");
    std::vector<std::vector<ClosestHit>> hits;
    double ms = 0.0;
    {
        py::gil_scoped_release release;
        GpuPointBuffer buf = uploadPointsGPU(pts, 1);
        GpuRadiusBuffer out;
        try {
            switch (kind) {
                case LeafKind::Triangle:
                    ms = radiusQueryGPU(bvh, buf, radius, out, false, order);
                    break;
                case LeafKind::Vertex:
                    ms = radiusQueryVertexGPU(bvh, buf, radius, out, false, order);
                    break;
                case LeafKind::Gaussian:
                    ms = radiusQueryGaussianGPU(bvh, buf, radius, out, false, order);
                    break;
            }
            hits = readRadiusResultsGPU(out);
        } catch (...) {
            destroyRadiusBufferGPU(out);
            destroyPointBufferGPU(buf);
            throw;
        }
        destroyRadiusBufferGPU(out);
        destroyPointBufferGPU(buf);
    }
    *msOut = ms;

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
    py::array_t<int32_t> idx(static_cast<py::ssize_t>(total));
    auto dv = dist2.mutable_unchecked<1>();
    auto tv = idx.mutable_unchecked<1>();
    size_t at = 0;
    for (py::ssize_t i = 0; i < n; ++i) {
        for (const ClosestHit& h : hits[static_cast<size_t>(i)]) {
            dv(static_cast<py::ssize_t>(at)) = h.dist2;
            tv(static_cast<py::ssize_t>(at)) = h.triangle;
            ++at;
        }
    }
    return py::make_tuple(offsets, dist2, idx);
}

class PyScene {
public:
    explicit PyScene(const F32Array& triangles, const std::string& builder)
        : tris_(trianglesFromArray(triangles)), sah_(parseSahBuilder(builder)) {
        build();
    }

    PyScene(const F32Array& vertices, const I32Array& indices, const std::string& builder)
        : tris_(trianglesFromMesh(vertices, indices)), sah_(parseSahBuilder(builder)) {
        indices_.assign(indices.data(), indices.data() + indices.size());
        build();
    }

    ~PyScene() { destroyBVHGPU(bvh_); }

    PyScene(const PyScene&) = delete;
    PyScene& operator=(const PyScene&) = delete;

    int triangleCount() const { return bvh_.primCount; }
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
        out["mark_ms"] = refitTimings_.markMs;
        out["merge_ms"] = refitTimings_.mergeMs;
        out["total_ms"] = refitTimings_.totalMs;
        out["updated"] = refitTimings_.updated;
        return out;
    }

    py::dict bounds() const { return boundsDict(bounds_); }

    py::dict buildStats() const {
        py::dict out;
        out["morton_ms"] = timings_.mortonMs;
        out["sort_ms"] = timings_.sortMs;
        out["scatter_ms"] = timings_.scatterMs;
        out["tree_ms"] = timings_.treeMs;
        out["refit_ms"] = timings_.refitMs;
        out["total_ms"] = timings_.totalMs;
        out["sah_bin_ms"] = timings_.sahBinMs;
        out["sah_split_ms"] = timings_.sahSplitMs;
        out["sah_partition_ms"] = timings_.sahPartMs;
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

    py::tuple closest(const F32Array& points, const std::string& order) {
        return closestRun(bvh_, points, LeafKind::Triangle, parseQueryOrder(order), &lastQueryMs_);
    }

    py::tuple knn(const F32Array& points, int k, const std::string& order) {
        return knnRun(bvh_, points, k, LeafKind::Triangle, parseQueryOrder(order),
                      &lastQueryMs_);
    }

    py::tuple radius(const F32Array& points, float radius, const std::string& order) {
        return radiusRun(bvh_, points, radius, LeafKind::Triangle, parseQueryOrder(order),
                         &lastQueryMs_);
    }

private:
    void build() {
        sceneView_.tris = tris_;
        sceneView_.updateBounds();
        bounds_ = sceneView_.bounds;
        {
            py::gil_scoped_release release;
            bvh_ = sah_ ? buildBVHSahGPU(sceneView_, &timings_) : buildBVHGPU(sceneView_, &timings_);
        }
        buildMs_ = timings_.totalMs;
    }

    // dirty 非 None 时只上传这些场景三角形（未列出的三角形在树里保持旧值），
    // 调用方必须保证"没列出来的都没动"——这是增量接口的合同，不会被逐位校验兜住。
    void update(std::vector<Tri> tris, const py::object& dirty) {
        if (static_cast<int>(tris.size()) != bvh_.primCount) {
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

    // 拓扑可变的更新：triangles 是新场景 = 幸存者（原顺序）+ 中途插入 + 末尾追加。
    // deleted 用旧下标、升序；inserted / moved 用新下标（inserted 升序），moved 标记
    // 几何变了但拓扑不变的三角形。网格绑定（构造时的 indices）成功之后自动失效。
    py::dict updateTopology(const F32Array& triangles, const py::object& deleted,
                            const py::object& moved, const py::object& inserted) {
        std::vector<Tri> tris = trianglesFromArray(triangles);
        const GpuTopoEdit edit = topoEditFromArrays(deleted, moved, inserted, bvh_.primCount,
                                                    static_cast<long long>(tris.size()));
        ::Scene view;
        view.tris = tris;
        view.updateBounds();
        {
            py::gil_scoped_release release;
            updateBVHGPU(bvh_, view, edit, &topoTimings_);
        }
        tris_ = std::move(tris);
        indices_.clear();
        sceneView_ = std::move(view);
        bounds_ = sceneView_.bounds;
        refitMs_ = topoTimings_.refitMs;
        return topoTimingsDict(topoTimings_);
    }

private:
    std::vector<Tri> tris_;
    std::vector<int32_t> indices_;  // 只有 (vertices, indices) 构造时才有
    bool sah_ = false;
    ::Scene sceneView_;
    AABB bounds_;
    GpuBVH bvh_;
    GpuBuildTimings timings_{};
    GpuRefitTimings refitTimings_{};
    GpuTopoUpdateTimings topoTimings_{};
    double buildMs_ = 0.0;
    double lastQueryMs_ = 0.0;
    double refitMs_ = 0.0;
};

class PyPointCloud {
public:
    explicit PyPointCloud(const F32Array& points, const std::string& builder)
        : sah_(parseSahBuilder(builder)) {
        std::vector<Vec3> pts = pointsFromArray(points, "points");
        if (pts.size() < 2) throw std::invalid_argument("at least 2 points are required");
        cloud_.pts = std::move(pts);
        cloud_.updateBounds();
        bounds_ = cloud_.bounds;
        {
            py::gil_scoped_release release;
            bvh_ = sah_ ? buildPointBVHSahGPU(cloud_, &timings_)
                        : buildPointBVHGPU(cloud_, &timings_);
        }
        buildMs_ = timings_.totalMs;
    }

    ~PyPointCloud() { destroyBVHGPU(bvh_); }

    PyPointCloud(const PyPointCloud&) = delete;
    PyPointCloud& operator=(const PyPointCloud&) = delete;

    int pointCount() const { return bvh_.primCount; }
    int nodeCount() const { return bvh_.nodeCount; }
    int depth() const { return bvh_.depth; }
    double buildMs() const { return buildMs_; }
    double lastQueryMs() const { return lastQueryMs_; }
    double refitMs() const { return refitMs_; }
    py::dict bounds() const { return boundsDict(bounds_); }

    py::dict buildStats() const {
        py::dict out;
        out["morton_ms"] = timings_.mortonMs;
        out["sort_ms"] = timings_.sortMs;
        out["scatter_ms"] = timings_.scatterMs;
        out["tree_ms"] = timings_.treeMs;
        out["refit_ms"] = timings_.refitMs;
        out["total_ms"] = timings_.totalMs;
        out["sah_bin_ms"] = timings_.sahBinMs;
        out["sah_split_ms"] = timings_.sahSplitMs;
        out["sah_partition_ms"] = timings_.sahPartMs;
        return out;
    }

    py::dict refitStats() const {
        py::dict out;
        out["upload_ms"] = refitTimings_.uploadMs;
        out["scatter_ms"] = refitTimings_.gatherMs;
        out["tree_ms"] = refitTimings_.treeMs;
        out["mark_ms"] = refitTimings_.markMs;
        out["merge_ms"] = refitTimings_.mergeMs;
        out["total_ms"] = refitTimings_.totalMs;
        out["updated"] = refitTimings_.updated;
        return out;
    }

    std::string validate() const { return validatePointBVHGPU(bvh_, cloud_); }

    py::tuple closest(const F32Array& points, const std::string& order) {
        return closestRun(bvh_, points, LeafKind::Vertex, parseQueryOrder(order), &lastQueryMs_);
    }

    py::tuple knn(const F32Array& points, int k, const std::string& order) {
        return knnRun(bvh_, points, k, LeafKind::Vertex, parseQueryOrder(order),
                      &lastQueryMs_);
    }

    py::tuple radius(const F32Array& points, float radius, const std::string& order) {
        return radiusRun(bvh_, points, radius, LeafKind::Vertex, parseQueryOrder(order),
                         &lastQueryMs_);
    }

    // 与 Scene.update_triangles 同一份合同：点 i 必须还是建树时那个点，
    // dirty 里的下标才允许改；不传 dirty 表示整份点云都上传。
    void updatePoints(const F32Array& points, const py::object& dirty) {
        std::vector<Vec3> pts = pointsFromArray(points, "points");
        if (static_cast<int>(pts.size()) != bvh_.primCount) {
            throw std::invalid_argument("update must keep the point count; rebuild instead");
        }
        std::vector<int> ids;
        const bool partial = !dirty.is_none();
        if (partial) {
            const I32Array arr = dirty.cast<I32Array>();
            ids.assign(arr.data(), arr.data() + arr.size());
        }

        cloud_.pts = std::move(pts);
        cloud_.updateBounds();
        bounds_ = cloud_.bounds;

        {
            py::gil_scoped_release release;
            refitMs_ = partial ? refitPointBVHGPU(bvh_, cloud_, ids, &refitTimings_)
                               : refitPointBVHGPU(bvh_, cloud_, &refitTimings_);
        }
    }

    // 拓扑可变：points = 幸存者（原顺序）+ 中途插入 + 末尾追加，deleted 用旧下标，
    // inserted / moved 用新下标（点是零体积图元，通常不需要 moved）。
    py::dict updateTopology(const F32Array& points, const py::object& deleted,
                            const py::object& moved, const py::object& inserted) {
        std::vector<Vec3> pts = pointsFromArray(points, "points");
        const GpuTopoEdit edit = topoEditFromArrays(deleted, moved, inserted, bvh_.primCount,
                                                    static_cast<long long>(pts.size()));
        PointCloud view = cloud_;
        view.pts = pts;
        view.updateBounds();
        {
            py::gil_scoped_release release;
            updatePointBVHGPU(bvh_, view, edit, &topoTimings_);
        }
        cloud_ = std::move(view);
        bounds_ = cloud_.bounds;
        refitMs_ = topoTimings_.refitMs;
        return topoTimingsDict(topoTimings_);
    }

private:
    PointCloud cloud_;
    AABB bounds_;
    GpuBVH bvh_;
    GpuBuildTimings timings_{};
    GpuRefitTimings refitTimings_{};
    GpuTopoUpdateTimings topoTimings_{};
    double buildMs_ = 0.0;
    double lastQueryMs_ = 0.0;
    double refitMs_ = 0.0;
    bool sah_ = false;
};

// 高斯云：查询语义是椭球实体（内部距离 0，外部到表面），不是概率密度。
// 因此 closest / knn / radius 的输出与点云完全同构：dist2 + 图元下标。
class PyGaussianCloud {
public:
    PyGaussianCloud(const F32Array& means, const F32Array& rot, const F32Array& scale,
                    const std::string& builder)
        : gaussians_(gaussiansFromArrays(means, rot, scale)), sah_(parseSahBuilder(builder)) {
        if (gaussians_.size() < 2) throw std::invalid_argument("at least 2 gaussians required");
        cloud_.gaussians = gaussians_;
        cloud_.updateBounds();
        bounds_ = cloud_.bounds;
        {
            py::gil_scoped_release release;
            bvh_ = sah_ ? buildGaussianBVHSahGPU(cloud_, &timings_)
                        : buildGaussianBVHGPU(cloud_, &timings_);
        }
        buildMs_ = timings_.totalMs;
    }

    ~PyGaussianCloud() { destroyBVHGPU(bvh_); }

    PyGaussianCloud(const PyGaussianCloud&) = delete;
    PyGaussianCloud& operator=(const PyGaussianCloud&) = delete;

    int gaussianCount() const { return bvh_.primCount; }
    int nodeCount() const { return bvh_.nodeCount; }
    int depth() const { return bvh_.depth; }
    double buildMs() const { return buildMs_; }
    double lastQueryMs() const { return lastQueryMs_; }
    double refitMs() const { return refitMs_; }
    py::dict bounds() const { return boundsDict(bounds_); }

    py::dict buildStats() const {
        py::dict out;
        out["morton_ms"] = timings_.mortonMs;
        out["sort_ms"] = timings_.sortMs;
        out["scatter_ms"] = timings_.scatterMs;
        out["tree_ms"] = timings_.treeMs;
        out["refit_ms"] = timings_.refitMs;
        out["total_ms"] = timings_.totalMs;
        out["sah_bin_ms"] = timings_.sahBinMs;
        out["sah_split_ms"] = timings_.sahSplitMs;
        out["sah_partition_ms"] = timings_.sahPartMs;
        return out;
    }

    py::dict refitStats() const {
        py::dict out;
        out["upload_ms"] = refitTimings_.uploadMs;
        out["scatter_ms"] = refitTimings_.gatherMs;
        out["tree_ms"] = refitTimings_.treeMs;
        out["mark_ms"] = refitTimings_.markMs;
        out["merge_ms"] = refitTimings_.mergeMs;
        out["total_ms"] = refitTimings_.totalMs;
        out["updated"] = refitTimings_.updated;
        return out;
    }

    std::string validate() const { return validateGaussianBVHGPU(bvh_, cloud_); }

    py::tuple closest(const F32Array& points, const std::string& order) {
        return closestRun(bvh_, points, LeafKind::Gaussian, parseQueryOrder(order), &lastQueryMs_);
    }

    py::tuple knn(const F32Array& points, int k, const std::string& order) {
        return knnRun(bvh_, points, k, LeafKind::Gaussian, parseQueryOrder(order),
                      &lastQueryMs_);
    }

    py::tuple radius(const F32Array& points, float radius, const std::string& order) {
        return radiusRun(bvh_, points, radius, LeafKind::Gaussian, parseQueryOrder(order),
                         &lastQueryMs_);
    }

    // 与另外两个 update 同一份合同：第 i 个高斯必须还是建树时那一族（均值可以动，
    // 朝向和半轴也可以动——叶子盒会重算），dirty 里的下标才允许变。
    // 不传 dirty 表示整份上传。
    void update(const F32Array& means, const F32Array& rot, const F32Array& scale,
                const py::object& dirty) {
        std::vector<Gaussian> gs = gaussiansFromArrays(means, rot, scale);
        if (static_cast<int>(gs.size()) != bvh_.primCount) {
            throw std::invalid_argument("update must keep the gaussian count; rebuild instead");
        }
        std::vector<int> ids;
        const bool partial = !dirty.is_none();
        if (partial) {
            const I32Array arr = dirty.cast<I32Array>();
            ids.assign(arr.data(), arr.data() + arr.size());
        }

        gaussians_ = std::move(gs);
        cloud_.gaussians = gaussians_;
        cloud_.updateBounds();
        bounds_ = cloud_.bounds;

        {
            py::gil_scoped_release release;
            refitMs_ = partial ? refitGaussianBVHGPU(bvh_, cloud_, ids, &refitTimings_)
                               : refitGaussianBVHGPU(bvh_, cloud_, &refitTimings_);
        }
    }

    // 拓扑可变：三个数组是幸存者（原顺序）+ 中途插入 + 末尾追加，
    // deleted 用旧下标，inserted / moved 用新下标。
    py::dict updateTopology(const F32Array& means, const F32Array& rot, const F32Array& scale,
                            const py::object& deleted, const py::object& moved,
                            const py::object& inserted) {
        std::vector<Gaussian> gs = gaussiansFromArrays(means, rot, scale);
        const GpuTopoEdit edit = topoEditFromArrays(deleted, moved, inserted, bvh_.primCount,
                                                    static_cast<long long>(gs.size()));
        GaussianCloud view = cloud_;
        view.gaussians = gs;
        view.updateBounds();
        {
            py::gil_scoped_release release;
            updateGaussianBVHGPU(bvh_, view, edit, &topoTimings_);
        }
        gaussians_ = std::move(gs);
        cloud_ = std::move(view);
        bounds_ = cloud_.bounds;
        refitMs_ = topoTimings_.refitMs;
        return topoTimingsDict(topoTimings_);
    }

private:
    std::vector<Gaussian> gaussians_;
    GaussianCloud cloud_;
    AABB bounds_;
    GpuBVH bvh_;
    GpuBuildTimings timings_{};
    GpuRefitTimings refitTimings_{};
    GpuTopoUpdateTimings topoTimings_{};
    double buildMs_ = 0.0;
    double lastQueryMs_ = 0.0;
    double refitMs_ = 0.0;
    bool sah_ = false;
};

py::tuple demoGaussians(const std::string& kind, int count, uint64_t seed) {
    GaussianCloudSpec spec;
    spec.kind = kind;
    spec.count = count;
    spec.seed = seed;
    const GaussianCloud cloud = buildGaussianCloud(spec);

    const py::ssize_t n = static_cast<py::ssize_t>(cloud.gaussians.size());
    py::array_t<float> means({n, static_cast<py::ssize_t>(3)});
    py::array_t<float> rot({n, static_cast<py::ssize_t>(4)});
    py::array_t<float> scale({n, static_cast<py::ssize_t>(3)});
    auto m = means.mutable_unchecked<2>();
    auto q = rot.mutable_unchecked<2>();
    auto s = scale.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        const Gaussian& g = cloud.gaussians[static_cast<size_t>(i)];
        m(i, 0) = g.mean.x;
        m(i, 1) = g.mean.y;
        m(i, 2) = g.mean.z;
        q(i, 0) = g.rot.x;
        q(i, 1) = g.rot.y;
        q(i, 2) = g.rot.z;
        q(i, 3) = g.rot.w;
        s(i, 0) = g.scale.x;
        s(i, 1) = g.scale.y;
        s(i, 2) = g.scale.z;
    }
    return py::make_tuple(means, rot, scale);
}

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
    m.attr("__version__") = QUARRY_VERSION;

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

    m.def("demo_gaussians", &demoGaussians, py::arg("kind") = "sheet", py::arg("count") = 131072,
          py::arg("seed") = 1,
          "Return (means (N,3), rot (N,4) xyzw, scales (N,3)) for a demo gaussian cloud");

    py::class_<PyScene>(m, "Scene", "Static triangle mesh with a GPU LBVH")
        .def(py::init([](const F32Array& triangles, const std::string& builder) {
                 return std::make_unique<PyScene>(triangles, builder);
             }),
             py::arg("triangles"), py::arg("builder") = "lbvh",
             "Build from triangles with shape (M, 3, 3) float32; builder is 'lbvh' (fast, "
             "Morton) or 'sah' (slower build, fewer node tests)")
        .def(py::init([](const F32Array& vertices, const I32Array& indices,
                         const std::string& builder) {
                 return std::make_unique<PyScene>(vertices, indices, builder);
             }),
             py::arg("vertices"), py::arg("indices"), py::arg("builder") = "lbvh",
             "Build from shared vertices (N,3) and indices (M,3)")
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
        .def("update_topology", &PyScene::updateTopology, py::arg("triangles"),
             py::arg("deleted") = py::none(), py::arg("moved") = py::none(),
             py::arg("inserted") = py::none(),
             "Change the primitive count: triangles is the new scene = survivors (original "
             "order minus deleted, old indices ascending) with new triangles inserted at the "
             "new-scene indices listed in inserted, then extra ones appended at the end. moved "
             "lists new-scene indices whose geometry changed but whose topology did not. "
             "Returns a timings dict; query results match a fresh build bit for bit")
        .def("raycast", &PyScene::raycast, py::arg("origins"), py::arg("dirs"),
             py::arg("tmax") = std::numeric_limits<float>::infinity(),
             "Return (hit bool[N], t float32[N]); t is inf on a miss. dirs need not be normalized")
        .def("closest", &PyScene::closest, py::arg("points"),
             py::arg("order") = "input",
             "Return (dist2 float32[N], triangle int32[N]); triangle is -1 on a miss")
        .def("knn", &PyScene::knn, py::arg("points"), py::arg("k"),
             py::arg("order") = "input",
             "Return (dist2 float32[N,k], triangle int32[N,k]) sorted by (dist2, index); "
             "missing slots are -1")
        .def("radius", &PyScene::radius, py::arg("points"), py::arg("radius"),
             py::arg("order") = "input",
             "Return CSR (offsets int64[N+1], dist2 float32[total], triangle int32[total]) "
             "sorted by (dist2, index) inside each segment");

    py::class_<PyPointCloud>(m, "PointCloud", "Static point cloud with a GPU LBVH")
        .def(py::init([](const F32Array& points, const std::string& builder) {
                 return std::make_unique<PyPointCloud>(points, builder);
             }),
             py::arg("points"), py::arg("builder") = "lbvh",
             "Build from points with shape (N, 3) float32; builder is 'lbvh' or 'sah'")
        .def_property_readonly("point_count", &PyPointCloud::pointCount)
        .def_property_readonly("node_count", &PyPointCloud::nodeCount)
        .def_property_readonly("depth", &PyPointCloud::depth)
        .def_property_readonly("build_ms", &PyPointCloud::buildMs)
        .def_property_readonly("last_query_ms", &PyPointCloud::lastQueryMs,
                               "Kernel time of the most recent query (CUDA events, no copies)")
        .def_property_readonly("refit_ms", &PyPointCloud::refitMs)
        .def_property_readonly("refit_stats", &PyPointCloud::refitStats)
        .def_property_readonly("bounds", &PyPointCloud::bounds)
        .def_property_readonly("build_stats", &PyPointCloud::buildStats)
        .def("validate", &PyPointCloud::validate, "Check tree invariants; empty string means pass")
        .def("update_points", &PyPointCloud::updatePoints, py::arg("points"),
             py::arg("dirty") = py::none(),
             "Refit in place: point i must still be the same sample. dirty is an int32 array "
             "of point indices to upload; omit it to upload everything")
        .def("update_topology", &PyPointCloud::updateTopology, py::arg("points"),
             py::arg("deleted") = py::none(), py::arg("moved") = py::none(),
             py::arg("inserted") = py::none(),
             "Change the point count: points is the new scene = survivors (original order minus "
             "deleted, old indices ascending), with new points inserted at the new-scene "
             "indices listed in inserted, then extra ones appended at the end")
        .def("closest", &PyPointCloud::closest, py::arg("points"),
             py::arg("order") = "input",
             "Return (dist2 float32[N], index int32[N]); index is -1 on a miss")
        .def("knn", &PyPointCloud::knn, py::arg("points"), py::arg("k"),
             py::arg("order") = "input",
             "Return (dist2 float32[N,k], index int32[N,k]) sorted by (dist2, index); "
             "missing slots are -1")
        .def("radius", &PyPointCloud::radius, py::arg("points"), py::arg("radius"),
             py::arg("order") = "input",
             "Return CSR (offsets int64[N+1], dist2 float32[total], index int32[total]) "
             "sorted by (dist2, index) inside each segment");

    py::class_<PyGaussianCloud>(
        m, "GaussianCloud",
        "Static gaussian cloud with a GPU LBVH. Query semantics are the solid ellipsoid: "
        "dist2 is 0 inside, otherwise the squared euclidean distance to the surface")
        .def(py::init([](const F32Array& means, const F32Array& rot, const F32Array& scale,
                         const std::string& builder) {
                 return std::make_unique<PyGaussianCloud>(means, rot, scale, builder);
             }),
             py::arg("means"), py::arg("rot"), py::arg("scale"), py::arg("builder") = "lbvh",
             "Build from means (N,3), rotations (N,4) xyzw quaternions, semi-axes (N,3); "
             "builder is 'lbvh' or 'sah'")
        .def_property_readonly("gaussian_count", &PyGaussianCloud::gaussianCount)
        .def_property_readonly("node_count", &PyGaussianCloud::nodeCount)
        .def_property_readonly("depth", &PyGaussianCloud::depth)
        .def_property_readonly("build_ms", &PyGaussianCloud::buildMs)
        .def_property_readonly("last_query_ms", &PyGaussianCloud::lastQueryMs,
                               "Kernel time of the most recent query (CUDA events, no copies)")
        .def_property_readonly("refit_ms", &PyGaussianCloud::refitMs)
        .def_property_readonly("refit_stats", &PyGaussianCloud::refitStats)
        .def_property_readonly("bounds", &PyGaussianCloud::bounds)
        .def_property_readonly("build_stats", &PyGaussianCloud::buildStats)
        .def("validate", &PyGaussianCloud::validate,
             "Check tree invariants; empty string means pass")
        .def("update", &PyGaussianCloud::update, py::arg("means"), py::arg("rot"),
             py::arg("scale"), py::arg("dirty") = py::none(),
             "Refit in place: gaussian i must still be the same primitive. dirty is an int32 "
             "array of indices to upload; omit it to upload everything")
        .def("update_topology", &PyGaussianCloud::updateTopology, py::arg("means"),
             py::arg("rot"), py::arg("scale"), py::arg("deleted") = py::none(),
             py::arg("moved") = py::none(), py::arg("inserted") = py::none(),
             "Change the gaussian count: the arrays are the new scene = survivors (original "
             "order minus deleted, old indices ascending), with new gaussians inserted at the "
             "new-scene indices listed in inserted, then extra ones appended at the end")
        .def("closest", &PyGaussianCloud::closest, py::arg("points"),
             py::arg("order") = "input",
             "Return (dist2 float32[N], index int32[N]); index is -1 on a miss")
        .def("knn", &PyGaussianCloud::knn, py::arg("points"), py::arg("k"),
             py::arg("order") = "input",
             "Return (dist2 float32[N,k], index int32[N,k]) sorted by (dist2, index); "
             "missing slots are -1")
        .def("radius", &PyGaussianCloud::radius, py::arg("points"), py::arg("radius"),
             py::arg("order") = "input",
             "Return CSR (offsets int64[N+1], dist2 float32[total], index int32[total]) "
             "sorted by (dist2, index) inside each segment");
}
