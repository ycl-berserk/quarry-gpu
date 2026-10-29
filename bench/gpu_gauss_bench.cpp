// GPU 高斯基准与对拍：同一份高斯云先走 CPU 椭球 BVH 基线，再走 GPU LBVH。
// 三条查询（最近椭球 / kNN / 半径）逐条比较 (平方距离位模式, 高斯索引)，要求完全一致。
//
// 基线说明：没有任何主流库对"高斯图元"提供空间查询接口——Embree 的 rtcPointQuery
// 对网格找最近点，Warp 同样只有网格；3DGS 的光栅化管线按屏幕空间投影排序，不算空间
// 查询。所以这里的对照是单线程、中位数切分的椭球 BVH，口径在 docs/baseline.md 里写明。
//
// 用法：
//   gpu_gauss_bench --gaussians 131072 --queries 20000
//   gpu_gauss_bench --kind volume --gaussians 1048576 --queries 50000
//   gpu_gauss_bench --query knn --k 8 --gaussians 1048576
//   gpu_gauss_bench --query radius --radius 1.0 --gaussians 1048576
//   gpu_gauss_bench --gaussians 131072 --refit

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/fingerprint.h"
#include "core/gaussian.h"
#include "core/geometry.h"
#include "core/point_cloud.h"
#include "cpu/gauss_bvh.h"
#include "gpu/gpu_bvh.h"

namespace {

double nowMs() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

struct Options {
    std::string kind = "sheet";
    std::string query = "closest";
    int k = 8;
    int gaussians = 131072;
    int queries = 20000;
    int runs = 5;
    float radius = 0.0f;  // 0 = 取包围盒对角线的 1%
    uint64_t seed = 1;
    bool verify = true;
    bool cpu = true;
    bool stats = false;
    bool refit = false;
    bool querySort = false;
    bool wide = false;
    double refitShift = 0.002;
    double refitDirty = 0.01;
    std::string motion = "random";
};

void printUsage() {
    std::printf(
        "gpu_gauss_bench - GPU Gaussian-ellipsoid BVH vs CPU ellipsoid BVH baseline\n"
        "\n"
        "  --kind sheet|volume           高斯分布，default sheet（地形表面 + 抖动）\n"
        "  --gaussians N                 高斯规模，default 131072\n"
        "  --queries N                   查询点数，default 20000\n"
        "  --query closest|knn|radius    default closest\n"
        "  --k N                         kNN 的 k，1..32，default 8\n"
        "  --radius R                    世界空间半径；省略时取包围盒对角线的 1%%\n"
        "  --runs N                      计时轮数，取中位数，default 5\n"
        "  --seed N                      default 1\n"
        "  --stats                       额外跑一轮带访问计数的查询\n"
        "  --verify / --no-verify        是否逐点对拍（default verify）\n"
        "  --no-cpu                      跳过 CPU 椭球 BVH 基线（大场景省时间）\n"
        "  --refit                       额外做增量更新实验（全量 refit → 局部 refit）\n"
        "  --refit-shift F               每次位移幅度（包围盒对角线的比例），default 0.002\n"
        "  --refit-dirty F               局部 refit 改动的高斯比例，default 0.01\n"
        "  --refit-motion random|coherent 位移方式，default random\n"
        "  --query-sort                  查询点按 Morton 序重排后再遍历\n"
        "  --wide                        遍历用 4 宽折叠节点（可与 --query-sort 叠加）\n");
}

bool parseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--kind") {
            const char* v = needValue("--kind");
            if (!v) return false;
            opt.kind = v;
        } else if (arg == "--query") {
            const char* v = needValue("--query");
            if (!v) return false;
            opt.query = v;
        } else if (arg == "--k") {
            const char* v = needValue("--k");
            if (!v) return false;
            opt.k = std::atoi(v);
        } else if (arg == "--radius") {
            const char* v = needValue("--radius");
            if (!v) return false;
            opt.radius = static_cast<float>(std::atof(v));
        } else if (arg == "--gaussians") {
            const char* v = needValue("--gaussians");
            if (!v) return false;
            opt.gaussians = std::atoi(v);
        } else if (arg == "--queries") {
            const char* v = needValue("--queries");
            if (!v) return false;
            opt.queries = std::atoi(v);
        } else if (arg == "--runs") {
            const char* v = needValue("--runs");
            if (!v) return false;
            opt.runs = std::max(1, std::atoi(v));
        } else if (arg == "--seed") {
            const char* v = needValue("--seed");
            if (!v) return false;
            opt.seed = std::strtoull(v, nullptr, 10);
        } else if (arg == "--stats") {
            opt.stats = true;
        } else if (arg == "--verify") {
            opt.verify = true;
        } else if (arg == "--no-verify") {
            opt.verify = false;
        } else if (arg == "--no-cpu") {
            opt.cpu = false;
        } else if (arg == "--refit") {
            opt.refit = true;
        } else if (arg == "--refit-shift") {
            const char* v = needValue("--refit-shift");
            if (!v) return false;
            opt.refitShift = std::atof(v);
            if (opt.refitShift < 0.0) opt.refitShift = 0.0;
        } else if (arg == "--query-sort") {
            opt.querySort = true;
        } else if (arg == "--wide") {
            opt.wide = true;
        } else if (arg == "--refit-dirty") {
            const char* v = needValue("--refit-dirty");
            if (!v) return false;
            opt.refitDirty = std::atof(v);
            if (opt.refitDirty < 0.0 || opt.refitDirty > 1.0) opt.refitDirty = 0.01;
        } else if (arg == "--refit-motion") {
            const char* v = needValue("--refit-motion");
            if (!v) return false;
            opt.motion = v;
            if (opt.motion != "random" && opt.motion != "coherent") {
                std::fprintf(stderr, "unknown --refit-motion: %s\n", v);
                return false;
            }
        } else if (arg == "-h" || arg == "--help") {
            printUsage();
            return false;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

ClosestHit hitFromGpu(const GpuClosestResult& r) {
    ClosestHit h;
    h.found = r.triangle >= 0;
    if (h.found) {
        std::memcpy(&h.dist2, &r.dist2Bits, sizeof(h.dist2));
        h.triangle = r.triangle;
    }
    return h;
}

struct QueryOutput {
    std::vector<std::vector<ClosestHit>> perPoint;
    GpuQueryStats stats;
    double ms = 0.0;
};

QueryOutput queryGpu(GpuBVH& bvh, GpuPointBuffer& buf, const Options& opt, float radius,
                     bool collectStats) {
    QueryOutput out;
    out.perPoint.resize(static_cast<size_t>(buf.count));
    const QueryOrder order = opt.querySort ? QueryOrder::Morton : QueryOrder::Input;
    if (opt.query == "radius") {
        GpuRadiusBuffer rb;
        out.ms = radiusQueryGaussianGPU(bvh, buf, radius, rb, collectStats, order, opt.wide);
        out.perPoint = readRadiusResultsGPU(rb);
        destroyRadiusBufferGPU(rb);
    } else {
        const bool knn = opt.query == "knn";
        out.ms = knn ? kNearestGaussianGPU(bvh, buf, opt.k, collectStats, order, opt.wide)
                     : closestGaussianGPU(bvh, buf, collectStats, order, opt.wide);
        const int stride = knn ? opt.k : 1;
        const std::vector<GpuClosestResult> flat = readQueryResultsGPU(buf);
        for (size_t i = 0; i < out.perPoint.size(); ++i) {
            const GpuClosestResult* row = flat.data() + i * static_cast<size_t>(stride);
            for (int j = 0; j < stride; ++j) {
                const ClosestHit h = hitFromGpu(row[j]);
                if (!h.found) break;
                out.perPoint[i].push_back(h);
            }
        }
    }
    out.stats = lastQueryStatsGPU();
    return out;
}

QueryOutput queryCpu(const GaussBVH& bvh, const std::vector<Vec3>& points, const Options& opt,
                     float radius) {
    QueryOutput out;
    out.perPoint.resize(points.size());
    const double t0 = nowMs();
    for (size_t i = 0; i < points.size(); ++i) {
        if (opt.query == "knn") {
            out.perPoint[i] = kNearestGaussianBVH(bvh, points[i], opt.k);
        } else if (opt.query == "radius") {
            out.perPoint[i] = radiusGaussianBVH(bvh, points[i], radius);
        } else {
            const ClosestHit h = closestGaussianBVH(bvh, points[i]);
            if (h.found) out.perPoint[i].push_back(h);
        }
    }
    out.ms = nowMs() - t0;
    return out;
}

QueryStats fingerprint(const QueryOutput& out, bool listMode) {
    QueryStats s;
    for (const std::vector<ClosestHit>& list : out.perPoint) {
        if (listMode) {
            s.add(list);
        } else {
            for (const ClosestHit& h : list) s.add(h);
        }
    }
    return s;
}

struct Mismatch {
    uint64_t count = 0;
    uint64_t id = 0;
    uint64_t dist = 0;
};

Mismatch compare(const QueryOutput& want, const QueryOutput& got) {
    Mismatch m;
    for (size_t i = 0; i < want.perPoint.size() && i < got.perPoint.size(); ++i) {
        const std::vector<ClosestHit>& a = want.perPoint[i];
        const std::vector<ClosestHit>& b = got.perPoint[i];
        if (a.size() != b.size()) {
            ++m.count;
            continue;
        }
        for (size_t j = 0; j < a.size(); ++j) {
            if (a[j].triangle != b[j].triangle) ++m.id;
            if (std::memcmp(&a[j].dist2, &b[j].dist2, sizeof(float)) != 0) ++m.dist;
        }
    }
    return m;
}

void reportCompare(const char* tag, const Mismatch& m) {
    std::printf("verify %s id_mismatch=%llu dist_mismatch=%llu count_mismatch=%llu\n", tag,
                static_cast<unsigned long long>(m.id), static_cast<unsigned long long>(m.dist),
                static_cast<unsigned long long>(m.count));
}

// 高斯位移：均值单独抖动（random），或按平滑场整体漂移（coherent）。
// 朝向与半轴不动——refit 的语义是"拓扑冻结，只重算盒子"。
void displaceGaussians(GaussianCloud& cloud, float amplitude, const std::vector<int>* which,
                       bool coherent) {
    uint64_t state = 0x9e3779b97f4a7c15ull;
    auto next = [&state]() {
        state ^= state >> 33;
        state *= 0xff51afd7ed558ccdull;
        state ^= state >> 33;
        return static_cast<float>((state >> 40) * (1.0 / 16777216.0));
    };
    auto bump = [&](Gaussian& g) {
        Vec3 d;
        if (coherent) {
            d = {amplitude * std::sin(g.mean.y * 0.5f), amplitude * 0.5f * std::sin(g.mean.x * 0.5f),
                 amplitude * std::cos(g.mean.z * 0.5f)};
        } else {
            d = {(next() * 2.0f - 1.0f) * amplitude, (next() * 2.0f - 1.0f) * amplitude,
                 (next() * 2.0f - 1.0f) * amplitude};
        }
        g.mean = g.mean + d;
    };
    if (which) {
        for (const int id : *which) bump(cloud.gaussians[static_cast<size_t>(id)]);
    } else {
        for (Gaussian& g : cloud.gaussians) bump(g);
    }
    cloud.updateBounds();
}

struct MsStats {
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
};

MsStats summarizeMs(std::vector<double> times) {
    std::sort(times.begin(), times.end());
    return {times[times.size() / 2], times.front(), times.back()};
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        printUsage();
        return 1;
    }

    std::string device;
    if (!gpuDescribe(&device)) {
        std::printf("no CUDA device: %s\n", device.c_str());
        return 2;
    }
    std::printf("device %s\n", device.c_str());

    if (opt.kind != "sheet" && opt.kind != "volume") {
        std::printf("unknown kind: %s\n", opt.kind.c_str());
        return 1;
    }
    if (opt.query != "closest" && opt.query != "knn" && opt.query != "radius") {
        std::printf("unknown query type: %s\n", opt.query.c_str());
        return 1;
    }
    if (opt.query == "knn" && (opt.k < 1 || opt.k > 32)) {
        std::printf("k must be in [1, 32]\n");
        return 1;
    }

    GaussianCloudSpec spec;
    spec.kind = opt.kind;
    spec.count = opt.gaussians;
    spec.seed = opt.seed;
    GaussianCloud cloud = buildGaussianCloud(spec);
    const std::vector<Vec3> queries = makeQueryPoints(cloud.bounds, opt.queries, opt.seed + 1);
    const float radius =
        opt.radius > 0.0f ? opt.radius : 0.01f * length(cloud.bounds.extent());

    std::printf("cloud=%s gaussians=%zu queries=%zu type=%s", opt.kind.c_str(),
                cloud.gaussians.size(), queries.size(), opt.query.c_str());
    if (opt.query == "knn") std::printf(" k=%d", opt.k);
    if (opt.query == "radius") std::printf(" radius=%g", static_cast<double>(radius));
    if (opt.wide) std::printf(" traversal=wide4");
    std::printf(" seed=%llu iterations=%d\n", static_cast<unsigned long long>(opt.seed),
                kGaussianNewtonIters);

    // ---- CPU 椭球 BVH 基线 ----
    GaussBVH tree;
    double cpuMedianMs = 0.0;
    QueryOutput cpuOut;
    if (opt.cpu) {
        const double t0 = nowMs();
        tree = buildGaussBVH(cloud);
        const double buildMs = nowMs() - t0;
        std::printf("cpu bvh build_ms=%.2f nodes=%zu\n", buildMs, tree.nodes.size());

        std::vector<double> times;
        for (int run = 0; run < opt.runs; ++run) {
            QueryOutput o = queryCpu(tree, queries, opt, radius);
            times.push_back(o.ms);
            if (run == 0) cpuOut = std::move(o);
        }
        const MsStats c = summarizeMs(times);
        cpuMedianMs = c.median;
        std::printf("cpu bvh runs=%d median_ms=%.2f min_ms=%.2f max_ms=%.2f Mqueries_s=%.2f\n",
                    opt.runs, c.median, c.min, c.max,
                    c.median > 0.0 ? queries.size() / (c.median * 1e3) : 0.0);
        std::printf("fingerprint cpu %s\n",
                    fingerprint(cpuOut, opt.query != "closest").summary().c_str());
    }

    // ---- GPU 构建 ----
    GpuBuildTimings build{};
    GpuBVH gpu;
    try {
        warmupBuildGPU();
        gpu = buildGaussianBVHGPU(cloud, &build);
    } catch (const std::exception& e) {
        std::printf("gpu build failed: %s\n", e.what());
        return 3;
    }
    std::printf(
        "gpu build morton=%.2f sort=%.2f scatter=%.2f tree=%.2f refit=%.2f total=%.2f ms "
        "nodes=%d depth=%d\n",
        build.mortonMs, build.sortMs, build.scatterMs, build.treeMs, build.refitMs, build.totalMs,
        gpu.nodeCount, gpu.depth);
    const std::string treeError = validateGaussianBVHGPU(gpu, cloud);
    std::printf("gpu tree %s%s\n", treeError.empty() ? "PASS" : "FAIL: ", treeError.c_str());
    if (!treeError.empty()) return 4;

    if (opt.wide) {
        try {
            const double wideMs = buildWideNodesGPU(gpu);
            std::printf("wide nodes=%d build_ms=%.2f node_bytes=128\n", gpu.wideCount, wideMs);
        } catch (const std::exception& e) {
            std::printf("wide build failed: %s\n", e.what());
            return 5;
        }
    }

    // ---- GPU 查询 ----
    const int stride = opt.query == "knn" ? opt.k : 1;
    GpuPointBuffer buf;
    try {
        buf = uploadPointsGPU(queries, stride);
        warmupQueryGPU();
        for (int run = 0; run < std::min(opt.runs, 3); ++run) queryGpu(gpu, buf, opt, radius, false);

        std::vector<double> times;
        QueryOutput gpuOut;
        for (int run = 0; run < opt.runs; ++run) {
            QueryOutput o = queryGpu(gpu, buf, opt, radius, false);
            times.push_back(o.ms);
            if (run == 0) gpuOut = std::move(o);
        }
        if (opt.stats) gpuOut = queryGpu(gpu, buf, opt, radius, true);

        const MsStats g = summarizeMs(times);
        std::printf("gpu runs=%d median_ms=%.2f min_ms=%.2f max_ms=%.2f Mqueries_s=%.2f\n",
                    opt.runs, g.median, g.min, g.max,
                    g.median > 0.0 ? queries.size() / (g.median * 1e3) : 0.0);
        if (opt.querySort) std::printf("gpu sort_ms=%.3f\n", lastQueryStatsGPU().sortMs);
        std::printf("fingerprint gpu %s\n",
                    fingerprint(gpuOut, opt.query != "closest").summary().c_str());
        if (opt.stats) {
            std::printf("gpu stats node_tests=%.1f prim_tests=%.1f per_query overflows=%llu\n",
                        static_cast<double>(gpuOut.stats.nodeTests) / queries.size(),
                        static_cast<double>(gpuOut.stats.triTests) / queries.size(),
                        static_cast<unsigned long long>(gpuOut.stats.stackOverflows));
        }

        if (opt.cpu && opt.verify) {
            reportCompare(opt.query.c_str(), compare(cpuOut, gpuOut));
            if (g.median > 0.0 && cpuMedianMs > 0.0) {
                std::printf("speedup vs cpu bvh (median of %d) = %.2fx\n", opt.runs,
                            cpuMedianMs / g.median);
            }
        }

        // ---- 增量更新实验 ----
        if (opt.refit) {
            const float amp = static_cast<float>(opt.refitShift * length(cloud.bounds.extent()));
            const bool coherent = opt.motion == "coherent";

            std::vector<int> dirty;
            const int dirtyCount =
                std::max(1, static_cast<int>(cloud.gaussians.size() * opt.refitDirty));
            dirty.reserve(static_cast<size_t>(dirtyCount));
            for (int i = 0; i < dirtyCount; ++i) {
                dirty.push_back(static_cast<int>((static_cast<uint64_t>(i) * 2654435761ull) %
                                                 cloud.gaussians.size()));
            }

            GaussianCloud moved = cloud;
            displaceGaussians(moved, amp, nullptr, coherent);
            GpuRefitTimings rt{};
            refitGaussianBVHGPU(gpu, moved, &rt);
            const std::string fullErr = validateGaussianBVHGPU(gpu, moved);
            std::printf(
                "gpu refit full upload=%.2f gather=%.2f tree=%.2f wide=%.2f total=%.2f ms "
                "tree=%s\n",
                rt.uploadMs, rt.gatherMs, rt.treeMs, rt.wideMs, rt.totalMs,
                fullErr.empty() ? "PASS" : "FAIL");
            if (!fullErr.empty()) return 5;

            if (opt.cpu && opt.verify) {
                const GaussBVH movedTree = buildGaussBVH(moved);
                reportCompare("refit-full", compare(queryCpu(movedTree, queries, opt, radius),
                                                    queryGpu(gpu, buf, opt, radius, false)));
            }

            displaceGaussians(moved, amp, &dirty, coherent);
            rt = GpuRefitTimings{};
            refitGaussianBVHGPU(gpu, moved, dirty, &rt);
            const std::string dirtyErr = validateGaussianBVHGPU(gpu, moved);
            std::printf(
                "gpu refit dirty updated=%d upload=%.2f gather=%.2f mark=%.2f merge=%.2f "
                "wide=%.2f total=%.2f ms tree=%s\n",
                rt.updated, rt.uploadMs, rt.gatherMs, rt.markMs, rt.mergeMs, rt.wideMs,
                rt.totalMs, dirtyErr.empty() ? "PASS" : "FAIL");
            if (!dirtyErr.empty()) return 6;

            if (opt.cpu && opt.verify) {
                const GaussBVH movedTree = buildGaussBVH(moved);
                reportCompare("refit-dirty", compare(queryCpu(movedTree, queries, opt, radius),
                                                     queryGpu(gpu, buf, opt, radius, false)));
            }
        }
    } catch (const std::exception& e) {
        std::printf("gpu query failed: %s\n", e.what());
        destroyPointBufferGPU(buf);
        destroyBVHGPU(gpu);
        return 7;
    }

    destroyPointBufferGPU(buf);
    destroyBVHGPU(gpu);
    return 0;
}
