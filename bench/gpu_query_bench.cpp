// GPU 点查询基准与对拍：同一批查询点先走 CPU 参考 BVH，再走 GPU LBVH，
// 要求每个点的结果完全一致（closest 比 (平方距离位模式, 场景三角形索引)，
// kNN 比每条的 (平方距离位模式, 场景三角形索引) 和结果条数）。
//
// 用法：
//   gpu_query_bench --grid 256 --points 20000 --runs 5
//   gpu_query_bench --grid 1024 --points 20000 --runs 5
//   gpu_query_bench --scene random --random-tris 2097152 --points 20000 --no-cpu
//   gpu_query_bench --grid 256 --query knn --k 8 --points 20000
//   gpu_query_bench --grid 256 --query radius --radius 2.0 --points 20000

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/fingerprint.h"
#include "core/scene.h"
#include "cpu/bvh.h"
#include "gpu/gpu_bvh.h"

namespace {

double nowMs() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

struct Options {
    std::string scene = "terrain";
    std::string query = "closest";
    int k = 8;
    float radius = 2.0f;
    int grid = 256;
    int randomTris = 0;
    int points = 20000;
    int runs = 5;
    uint64_t seed = 1;
    bool verify = true;
    bool cpu = true;
    bool stats = false;
    std::string dump;
    int warmupMs = 3000;
    bool refit = false;
    double refitShift = 0.005;
    double refitDirty = 0.01;
    std::string refitMotion = "random";
    bool querySort = false;
    bool wide = false;
};

void printUsage() {
    std::printf(
        "gpu_query_bench - GPU point queries vs CPU reference BVH\n"
        "\n"
        "  --scene terrain|random|mixed   default terrain\n"
        "  --query closest|knn|radius     default closest\n"
        "  --k N                          kNN 的 k，1..32，default 8\n"
        "  --radius R                     radius 查询的世界空间半径，default 2.0\n"
        "  --grid N                       terrain 边长（格数），default 256 → 131072 三角形\n"
        "  --random-tris N                random/mixed 的随机三角形数\n"
        "  --points N                     查询点数，default 20000\n"
        "  --runs N                       计时轮数，取中位数，default 5\n"
        "  --warmup-ms N                  计时前持续负载毫秒数，default 3000\n"
        "  --seed N                       default 1\n"
        "  --stats                        额外跑一轮带节点/三角形计数的查询\n"
        "  --verify / --no-verify         是否逐点对拍（default verify）\n"
        "  --no-cpu                       跳过 CPU 参考查询（大场景省时间）\n"
        "  --dump FILE                    把场景与查询点写成二进制后退出（给外部引擎对照用）\n"
        "  --refit                        额外做一轮增量更新实验（少量改动 → 全量 refit → 重建）\n"
        "  --refit-shift F                每个三角形随机平移的幅度（场景对角线的比例），default 0.005\n"
        "  --refit-dirty F                第一步改动的三角形比例，default 0.01\n"
        "  --refit-motion random|coherent 位移方式：逐三角形独立 / 平滑场，default random\n"
        "  --query-sort                   查询点按 Morton 序重排后再遍历（不能与 --refit 同用）\n");
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

        if (arg == "--scene") {
            const char* v = needValue("--scene");
            if (!v) return false;
            opt.scene = v;
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
        } else if (arg == "--grid") {
            const char* v = needValue("--grid");
            if (!v) return false;
            opt.grid = std::atoi(v);
        } else if (arg == "--random-tris") {
            const char* v = needValue("--random-tris");
            if (!v) return false;
            opt.randomTris = std::atoi(v);
        } else if (arg == "--points") {
            const char* v = needValue("--points");
            if (!v) return false;
            opt.points = std::atoi(v);
        } else if (arg == "--runs") {
            const char* v = needValue("--runs");
            if (!v) return false;
            opt.runs = std::max(1, std::atoi(v));
        } else if (arg == "--warmup-ms") {
            const char* v = needValue("--warmup-ms");
            if (!v) return false;
            opt.warmupMs = std::max(0, std::atoi(v));
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
        } else if (arg == "--dump") {
            const char* v = needValue("--dump");
            if (!v) return false;
            opt.dump = v;
        } else if (arg == "--refit") {
            opt.refit = true;
        } else if (arg == "--refit-shift") {
            const char* v = needValue("--refit-shift");
            if (!v) return false;
            opt.refitShift = std::atof(v);
            if (opt.refitShift < 0.0) opt.refitShift = 0.0;
        } else if (arg == "--refit-dirty") {
            const char* v = needValue("--refit-dirty");
            if (!v) return false;
            opt.refitDirty = std::atof(v);
            if (opt.refitDirty < 0.0 || opt.refitDirty > 1.0) opt.refitDirty = 0.01;
        } else if (arg == "--refit-motion") {
            const char* v = needValue("--refit-motion");
            if (!v) return false;
            opt.refitMotion = v;
            if (opt.refitMotion != "random" && opt.refitMotion != "coherent") {
                std::fprintf(stderr, "unknown --refit-motion: %s\n", v);
                return false;
            }
        } else if (arg == "--wide") {
            opt.wide = true;
        } else if (arg == "--query-sort") {
            opt.querySort = true;
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

// 二进制格式（小端）：magic "QRYD" + version=1 + nTris + nPoints + 三角形顶点 +
// 查询点。外部引擎（Warp 等）读同一份输入，跨语言对照才不是"各自生成一个近似场景"。
bool dumpQueryInput(const std::string& path, const Scene& scene, const std::vector<Vec3>& points) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    const uint32_t magic = 0x44595251u;  // "QRYD"
    const uint32_t version = 1;
    const uint64_t nTris = scene.tris.size();
    const uint64_t nPoints = points.size();
    std::fwrite(&magic, sizeof(magic), 1, f);
    std::fwrite(&version, sizeof(version), 1, f);
    std::fwrite(&nTris, sizeof(nTris), 1, f);
    std::fwrite(&nPoints, sizeof(nPoints), 1, f);
    for (const Tri& t : scene.tris) {
        const float v[9] = {t.a.x, t.a.y, t.a.z, t.b.x, t.b.y, t.b.z, t.c.x, t.c.y, t.c.z};
        std::fwrite(v, sizeof(float), 9, f);
    }
    for (const Vec3& p : points) {
        const float v[3] = {p.x, p.y, p.z};
        std::fwrite(v, sizeof(float), 3, f);
    }
    const bool ok = std::fflush(f) == 0 && std::ferror(f) == 0;
    std::fclose(f);
    return ok;
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

// 把三种查询统一成"每点一个结果列表"：closest 是 0/1 条，kNN / radius 是 k 条或不定长。
// 这样 refit 前后、CPU 与 GPU 之间只需要一套比较逻辑。
struct QueryOutput {
    std::vector<std::vector<ClosestHit>> perPoint;
    GpuQueryStats stats;  // 只有 GPU 路径会填；CPU 侧只用来出结果
    double ms = 0.0;
};

QueryOutput queryGpu(GpuBVH& bvh, GpuPointBuffer& buf, int stride, const std::string& query, int k,
                     float radius, bool collectStats, QueryOrder order, bool wide = false) {
    QueryOutput out;
    out.perPoint.resize(static_cast<size_t>(buf.count));
    if (query == "radius") {
        GpuRadiusBuffer rb;
        out.ms = radiusQueryGPU(bvh, buf, radius, rb, collectStats, order, wide);
        std::vector<std::vector<ClosestHit>> lists = readRadiusResultsGPU(rb);
        destroyRadiusBufferGPU(rb);
        out.perPoint = std::move(lists);
    } else {
        out.ms = (query == "knn") ? kNearestGPU(bvh, buf, k, collectStats, order, wide)
                                  : closestPointGPU(bvh, buf, collectStats, order, wide);
        const std::vector<GpuClosestResult> flat = readQueryResultsGPU(buf);
        for (size_t i = 0; i < out.perPoint.size(); ++i) {
            const GpuClosestResult* row = flat.data() + i * stride;
            for (int j = 0; j < stride; ++j) {
                const ClosestHit g = hitFromGpu(row[j]);
                if (!g.found) break;
                out.perPoint[i].push_back(g);
            }
        }
    }
    out.stats = lastQueryStatsGPU();
    return out;
}

QueryOutput queryCpu(const BVH& bvh, const std::vector<Vec3>& points, const std::string& query,
                     int k, float radius) {
    QueryOutput out;
    out.perPoint.resize(points.size());
    const double t0 = nowMs();
    for (size_t i = 0; i < points.size(); ++i) {
        if (query == "knn") {
            out.perPoint[i] = kNearestBVH(bvh, points[i], k);
        } else if (query == "radius") {
            out.perPoint[i] = radiusBVH(bvh, points[i], radius);
        } else {
            const ClosestHit h = closestPointBVH(bvh, points[i]);
            if (h.found) out.perPoint[i].push_back(h);
        }
    }
    out.ms = nowMs() - t0;
    return out;
}

struct Mismatch {
    uint64_t count = 0;
    uint64_t tri = 0;
    uint64_t dist = 0;
};

Mismatch compareQueryOutput(const QueryOutput& want, const QueryOutput& got) {
    Mismatch m;
    for (size_t i = 0; i < want.perPoint.size() && i < got.perPoint.size(); ++i) {
        const std::vector<ClosestHit>& a = want.perPoint[i];
        const std::vector<ClosestHit>& b = got.perPoint[i];
        if (a.size() != b.size()) {
            ++m.count;
            continue;
        }
        for (size_t j = 0; j < a.size(); ++j) {
            if (a[j].triangle != b[j].triangle) ++m.tri;
            if (std::memcmp(&a[j].dist2, &b[j].dist2, sizeof(float)) != 0) ++m.dist;
        }
    }
    return m;
}

// 每个三角形整体平移：形状不变、顺序不变，refit 的合法输入。random 给每个三角形
// 独立位移（最坏情况的局部性），coherent 用平滑场（位移前后邻居关系基本保留）。
// 幅度按场景对角线取比例，跨规模可比。which 为空表示全动。
void displaceTriangles(Scene& scene, float amplitude, const std::vector<int>* which,
                       bool coherent) {
    uint64_t state = 0x9e3779b97f4a7c15ull;
    auto next = [&state]() {
        state ^= state >> 33;
        state *= 0xff51afd7ed558ccdull;
        state ^= state >> 33;
        return static_cast<float>((state >> 40) * (1.0 / 16777216.0));
    };
    auto bump = [&](Tri& t) {
        Vec3 d;
        if (coherent) {
            const Vec3 c = (t.a + t.b + t.c) * (1.0f / 3.0f);
            d = {amplitude * std::sin(c.y * 0.5f), amplitude * 0.5f * std::sin(c.x * 0.5f),
                 amplitude * std::cos(c.z * 0.5f)};
        } else {
            d = {(next() * 2.0f - 1.0f) * amplitude, (next() * 2.0f - 1.0f) * amplitude,
                 (next() * 2.0f - 1.0f) * amplitude};
        }
        t.a = t.a + d;
        t.b = t.b + d;
        t.c = t.c + d;
    };
    if (which) {
        for (const int id : *which) bump(scene.tris[static_cast<size_t>(id)]);
    } else {
        for (Tri& t : scene.tris) bump(t);
    }
    scene.updateBounds();
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

    const bool knnMode = opt.query == "knn";
    const bool radiusMode = opt.query == "radius";
    if (!knnMode && !radiusMode && opt.query != "closest") {
        std::printf("unknown query type: %s\n", opt.query.c_str());
        return 1;
    }
    if (knnMode && (opt.k < 1 || opt.k > 32)) {
        std::printf("k must be in [1, 32]\n");
        return 1;
    }
    if (radiusMode && opt.radius < 0.0f) {
        std::printf("radius must be >= 0\n");
        return 1;
    }
    const int stride = knnMode ? opt.k : 1;

    if (opt.scene == "random" && opt.randomTris == 0) opt.randomTris = 200000;
    if (opt.scene == "mixed" && opt.randomTris == 0) opt.randomTris = 50000;

    const SceneSpec spec{opt.scene, opt.grid, opt.randomTris, opt.seed};
    const Scene scene = buildScene(spec);
    const std::vector<Vec3> points = makeQueryPoints(scene.bounds, opt.points, opt.seed);
    const QueryOrder order = opt.querySort ? QueryOrder::Morton : QueryOrder::Input;

    std::printf("scene=%s tris=%zu points=%zu type=%s", opt.scene.c_str(), scene.tris.size(),
                points.size(), opt.query.c_str());
    if (knnMode) std::printf(" k=%d", opt.k);
    if (radiusMode) std::printf(" radius=%g", static_cast<double>(opt.radius));
    if (opt.querySort) std::printf(" order=morton");
    if (opt.wide) std::printf(" traversal=wide4");
    std::printf(" seed=%llu\n", static_cast<unsigned long long>(opt.seed));
    if (opt.querySort && opt.refit) {
        std::printf("--query-sort does not support --refit yet\n");
        return 1;
    }
    if (opt.wide && opt.refit) {
        std::printf("--wide does not support --refit yet (collapsed boxes go stale)\n");
        return 1;
    }

    if (!opt.dump.empty()) {
        if (!dumpQueryInput(opt.dump, scene, points)) return 1;
        std::printf("dumped %s (%zu tris, %zu points)\n", opt.dump.c_str(), scene.tris.size(),
                    points.size());
        return 0;
    }

    // ---- CPU 参考 ----
    BVH cpuBVH;
    double cpuBuildMs = 0.0;
    std::vector<ClosestHit> cpuHits;
    std::vector<std::vector<ClosestHit>> cpuLists;
    std::vector<double> cpuTimes;
    QueryStats cpuStats;
    if (opt.cpu) {
        const double t0 = nowMs();
        cpuBVH = buildBVH(scene);
        cpuBuildMs = nowMs() - t0;
        std::printf("cpu build_ms=%.2f nodes=%zu depth=%zu\n", cpuBuildMs, cpuBVH.nodes.size(),
                    bvhDepth(cpuBVH));

        for (int run = 0; run < opt.runs; ++run) {
            const double t = nowMs();
            QueryStats runStats;
            for (const Vec3& p : points) {
                if (knnMode) {
                    std::vector<ClosestHit> list = kNearestBVH(cpuBVH, p, opt.k);
                    runStats.add(list);
                    if (run == 0) cpuLists.push_back(std::move(list));
                } else if (radiusMode) {
                    std::vector<ClosestHit> list = radiusBVH(cpuBVH, p, opt.radius);
                    runStats.add(list);
                    if (run == 0) cpuLists.push_back(std::move(list));
                } else {
                    const ClosestHit h = closestPointBVH(cpuBVH, p);
                    runStats.add(h);
                    if (run == 0) cpuHits.push_back(h);
                }
            }
            cpuTimes.push_back(nowMs() - t);
            if (run == 0) cpuStats = runStats;
        }
        std::sort(cpuTimes.begin(), cpuTimes.end());
        const double cpuMedian = cpuTimes[cpuTimes.size() / 2];
        std::printf("cpu runs=%d median_ms=%.2f min_ms=%.2f max_ms=%.2f Mqueries_s=%.2f\n",
                    opt.runs, cpuMedian, cpuTimes.front(), cpuTimes.back(),
                    cpuMedian > 0.0 ? points.size() / (cpuMedian * 1e3) : 0.0);
        std::printf("fingerprint cpu %s\n", cpuStats.summary().c_str());
    }

    // ---- GPU 构建 ----
    GpuBuildTimings build{};
    GpuBVH gpu;
    try {
        warmupBuildGPU();
        gpu = buildBVHGPU(scene, &build);
    } catch (const std::exception& e) {
        std::printf("gpu build failed: %s\n", e.what());
        return 3;
    }
    std::printf(
        "gpu build morton=%.2f sort=%.2f scatter=%.2f tree=%.2f refit=%.2f total=%.2f ms "
        "nodes=%d depth=%d\n",
        build.mortonMs, build.sortMs, build.scatterMs, build.treeMs, build.refitMs, build.totalMs,
        gpu.nodeCount, gpu.depth);
    const std::string treeError = validateBVHGPU(gpu, scene);
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
    GpuPointBuffer buf;
    try {
        buf = uploadPointsGPU(points, stride);
        warmupQueryGPU();  // 模块懒加载不计入查询时间
    } catch (const std::exception& e) {
        std::printf("gpu upload failed: %s\n", e.what());
        return 3;
    }

    std::vector<double> gpuTimes;
    GpuRadiusBuffer radiusBuf;
    try {
        // 笔记本的 DVFS 不会因为几十毫秒的 kernel 就升到稳定时钟：计时前先持续
        // 跑 warmup_ms（默认 3 s，实测时钟约在 4 s 到顶），否则同一二进制在"冷"
        // 和"热"两种时钟状态下的差可以到 2.5×。
        const double warmStart = nowMs();
        int warmRuns = 0;
        while (nowMs() - warmStart < opt.warmupMs) {
            if (radiusMode) {
                destroyRadiusBufferGPU(radiusBuf);
                radiusQueryGPU(gpu, buf, opt.radius, radiusBuf, false, order, opt.wide);
            } else if (knnMode) {
                kNearestGPU(gpu, buf, opt.k, false, order, opt.wide);
            } else {
                closestPointGPU(gpu, buf, false, order, opt.wide);
            }
            ++warmRuns;
        }
        std::printf("gpu warmup_ms=%.0f runs=%d\n", nowMs() - warmStart, warmRuns);

        for (int run = 0; run < opt.runs; ++run) {
            if (radiusMode) {
                destroyRadiusBufferGPU(radiusBuf);  // 每轮重新分配，分配时间不计入 event
                gpuTimes.push_back(radiusQueryGPU(gpu, buf, opt.radius, radiusBuf, false, order, opt.wide));
            } else {
                gpuTimes.push_back(knnMode ? kNearestGPU(gpu, buf, opt.k, false, order, opt.wide)
                                           : closestPointGPU(gpu, buf, false, order, opt.wide));
            }
        }
    } catch (const std::exception& e) {
        std::printf("gpu query failed: %s\n", e.what());
        return 3;
    }
    std::sort(gpuTimes.begin(), gpuTimes.end());
    const double gpuMedian = gpuTimes[gpuTimes.size() / 2];
    std::printf("gpu runs=%d median_ms=%.3f min_ms=%.3f max_ms=%.3f Mqueries_s=%.2f\n", opt.runs,
                gpuMedian, gpuTimes.front(), gpuTimes.back(),
                gpuMedian > 0.0 ? points.size() / (gpuMedian * 1e3) : 0.0);
    if (opt.querySort) std::printf("gpu sort_ms=%.3f\n", lastQueryStatsGPU().sortMs);

    int exitCode = 0;
    if (opt.verify) {
        QueryStats gpuStats;
        uint64_t countMismatch = 0;
        uint64_t triMismatch = 0;
        uint64_t distMismatch = 0;

        if (knnMode || radiusMode) {
            std::vector<std::vector<ClosestHit>> gpuLists;
            if (radiusMode) {
                gpuLists = readRadiusResultsGPU(radiusBuf);
            } else {
                const std::vector<GpuClosestResult> flat = readQueryResultsGPU(buf);
                gpuLists.resize(points.size());
                for (size_t i = 0; i < points.size(); ++i) {
                    const GpuClosestResult* row = flat.data() + i * stride;
                    for (int j = 0; j < stride; ++j) {
                        const ClosestHit g = hitFromGpu(row[j]);
                        if (!g.found) break;  // 结果表尾部用 triangle=-1 填充
                        gpuLists[i].push_back(g);
                    }
                }
            }

            for (size_t i = 0; i < gpuLists.size(); ++i) {
                gpuStats.add(gpuLists[i]);
                if (!opt.cpu) continue;
                const std::vector<ClosestHit>& c = cpuLists[i];
                if (c.size() != gpuLists[i].size()) {
                    ++countMismatch;
                    continue;
                }
                for (size_t j = 0; j < c.size(); ++j) {
                    if (c[j].triangle != gpuLists[i][j].triangle) ++triMismatch;
                    if (std::memcmp(&c[j].dist2, &gpuLists[i][j].dist2, sizeof(float)) != 0) {
                        ++distMismatch;
                    }
                }
            }
        } else {
            const std::vector<GpuClosestResult> gpuHits = readQueryResultsGPU(buf);
            for (size_t i = 0; i < gpuHits.size(); ++i) {
                const ClosestHit g = hitFromGpu(gpuHits[i]);
                gpuStats.add(g);
                if (!opt.cpu) continue;
                const ClosestHit c = cpuHits[i];
                if (c.found != g.found) {
                    ++triMismatch;
                } else if (g.found) {
                    if (c.triangle != g.triangle) ++triMismatch;
                    if (std::memcmp(&c.dist2, &g.dist2, sizeof(float)) != 0) ++distMismatch;
                }
            }
        }

        std::printf("fingerprint gpu %s\n", gpuStats.summary().c_str());
        if (opt.cpu) {
            if (knnMode || radiusMode) {
                std::printf(
                    "verify points=%zu count_mismatch=%llu tri_mismatch=%llu "
                    "dist2_mismatch=%llu\n",
                    points.size(), static_cast<unsigned long long>(countMismatch),
                    static_cast<unsigned long long>(triMismatch),
                    static_cast<unsigned long long>(distMismatch));
            } else {
                std::printf("verify points=%zu tri_mismatch=%llu dist2_mismatch=%llu\n",
                            points.size(), static_cast<unsigned long long>(triMismatch),
                            static_cast<unsigned long long>(distMismatch));
            }
            if (countMismatch || triMismatch || distMismatch) exitCode = 6;
        }
    }

    if (opt.stats) {
        if (radiusMode) {
            destroyRadiusBufferGPU(radiusBuf);
            radiusQueryGPU(gpu, buf, opt.radius, radiusBuf, true, order, opt.wide);
        } else if (knnMode) {
            kNearestGPU(gpu, buf, opt.k, true, order, opt.wide);
        } else {
            closestPointGPU(gpu, buf, true, order, opt.wide);
        }
        const GpuQueryStats st = lastQueryStatsGPU();
        const double n = static_cast<double>(points.size());
        std::printf(
            "gpu query avg_node_tests=%.1f avg_tri_tests=%.1f stack_overflows=%llu "
            "node_tests=%llu\n",
            st.nodeTests / n, st.triTests / n,
            static_cast<unsigned long long>(st.stackOverflows),
            static_cast<unsigned long long>(st.nodeTests));
    }

    if (opt.refit) {
        // ---- 增量更新三步 ----
        // 1) 只动 1% 的三角形，refit 只上传这些；
        // 2) 整体位移，refit 全量上传；
        // 3) 同一份几何整树重建。
        // 每一步都拿 CPU 参考对拍：refit 不改拓扑，结果必须逐位正确。第 2、3 步
        // 用的是同一份几何，查询快慢才只反映树的质量（第 1 步的几何不同，不参与比较）。
        const double nPoints = static_cast<double>(points.size());
        auto runQuery = [&](GpuBVH& b, GpuPointBuffer& pb, bool st) {
            return queryGpu(b, pb, stride, opt.query, opt.k, opt.radius, st,
                            QueryOrder::Input);
        };
        const QueryOutput base = runQuery(gpu, buf, true);
        std::printf("refit baseline avg_node_tests=%.1f avg_tri_tests=%.1f\n",
                    base.stats.nodeTests / nPoints, base.stats.triTests / nPoints);

        Scene moved = scene;
        const Vec3 ext = moved.bounds.mx - moved.bounds.mn;
        const double diag = std::sqrt(static_cast<double>(dot(ext, ext)));
        const float amplitude = static_cast<float>(opt.refitShift * diag);
        std::printf("refit motion=%s shift=%.4f amplitude=%.4f world tris=%zu\n",
                    opt.refitMotion.c_str(), opt.refitShift, amplitude, moved.tris.size());

        auto cpuFor = [&](const Scene& s) {
            const BVH cpu = buildBVH(s);
            return queryCpu(cpu, points, opt.query, opt.k, opt.radius);
        };
        auto medianQuery = [&](GpuBVH& b) {
            std::vector<double> t;
            for (int i = 0; i < opt.runs; ++i) {
                t.push_back(runQuery(b, buf, false).ms);
            }
            std::sort(t.begin(), t.end());
            return t[t.size() / 2];
        };
        auto verify = [&](const char* label, const QueryOutput& want, const QueryOutput& got) {
            const Mismatch m = compareQueryOutput(want, got);
            std::printf("%s verify points=%zu count_mismatch=%llu tri_mismatch=%llu "
                        "dist2_mismatch=%llu\n",
                        label, points.size(), static_cast<unsigned long long>(m.count),
                        static_cast<unsigned long long>(m.tri),
                        static_cast<unsigned long long>(m.dist));
            if (m.count || m.tri || m.dist) exitCode = 6;
        };

        // 1) 少量改动：只上传 dirty 里的三角形
        std::vector<int> dirty;
        const int step = opt.refitDirty > 0.0
                             ? std::max(1, static_cast<int>(std::lround(1.0 / opt.refitDirty)))
                             : 0;
        for (int i = 0; step > 0 && i < static_cast<int>(moved.tris.size()); i += step) {
            dirty.push_back(i);
        }
        displaceTriangles(moved, amplitude, &dirty, opt.refitMotion == "coherent");

        GpuRefitTimings partial{};
        double partialMs = 0.0;
        try {
            // 同一个几何重复 refit 是幂等的；取中位数，顺手把暂存区的首次分配
            // 和模块状态排除在稳定态之外。
            std::vector<double> times;
            for (int i = 0; i < opt.runs; ++i) {
                times.push_back(refitBVHGPU(gpu, moved, dirty, &partial));
            }
            std::sort(times.begin(), times.end());
            partialMs = times[times.size() / 2];
        } catch (const std::exception& e) {
            std::printf("gpu partial refit failed: %s\n", e.what());
            return 3;
        }
        std::printf("partial updated=%d upload=%.3f scatter=%.3f mark=%.3f merge=%.3f "
                    "total=%.3f ms\n",
                    partial.updated, partial.uploadMs, partial.gatherMs, partial.markMs,
                    partial.mergeMs, partialMs);
        const std::string partialTree = validateBVHGPU(gpu, moved);
        std::printf("partial tree %s%s\n", partialTree.empty() ? "PASS" : "FAIL: ",
                    partialTree.c_str());
        if (!partialTree.empty()) return 4;

        const double partialQuery = medianQuery(gpu);
        const QueryOutput partialRes = runQuery(gpu, buf, true);
        std::printf("partial query median_ms=%.3f avg_node_tests=%.1f avg_tri_tests=%.1f\n",
                    partialQuery, partialRes.stats.nodeTests / nPoints,
                    partialRes.stats.triTests / nPoints);
        if (opt.cpu) verify("partial", cpuFor(moved), partialRes);

        // 2) 整体位移：全量 refit
        displaceTriangles(moved, amplitude, nullptr, opt.refitMotion == "coherent");
        GpuRefitTimings full{};
        double fullMs = 0.0;
        try {
            std::vector<double> times;
            for (int i = 0; i < opt.runs; ++i) {
                times.push_back(refitBVHGPU(gpu, moved, &full));
            }
            std::sort(times.begin(), times.end());
            fullMs = times[times.size() / 2];
        } catch (const std::exception& e) {
            std::printf("gpu refit failed: %s\n", e.what());
            return 3;
        }
        std::printf("refit updated=%d upload=%.3f scatter=%.3f tree=%.3f total=%.3f ms\n",
                    full.updated, full.uploadMs, full.gatherMs, full.treeMs, fullMs);
        const std::string refitTree = validateBVHGPU(gpu, moved);
        std::printf("refit tree %s%s\n", refitTree.empty() ? "PASS" : "FAIL: ", refitTree.c_str());
        if (!refitTree.empty()) return 4;

        const QueryOutput refCpu = opt.cpu ? cpuFor(moved) : QueryOutput{};
        const double refitQuery = medianQuery(gpu);
        const QueryOutput refitRes = runQuery(gpu, buf, true);
        std::printf("refit query median_ms=%.3f Mqueries_s=%.2f avg_node_tests=%.1f "
                    "avg_tri_tests=%.1f\n",
                    refitQuery, refitQuery > 0.0 ? nPoints / (refitQuery * 1e3) : 0.0,
                    refitRes.stats.nodeTests / nPoints, refitRes.stats.triTests / nPoints);
        if (opt.cpu) verify("refit", refCpu, refitRes);

        // 3) 同一份几何整树重建（同样取中位数，和 refit 的计时口径对齐）
        GpuBuildTimings rebuild{};
        GpuBVH rebuilt;
        double rebuildMs = 0.0;
        try {
            std::vector<double> times;
            for (int i = 0; i < opt.runs; ++i) {
                destroyBVHGPU(rebuilt);
                rebuild = GpuBuildTimings{};
                rebuilt = buildBVHGPU(moved, &rebuild);
                times.push_back(rebuild.totalMs);
            }
            std::sort(times.begin(), times.end());
            rebuildMs = times[times.size() / 2];
        } catch (const std::exception& e) {
            std::printf("gpu rebuild failed: %s\n", e.what());
            return 3;
        }
        const double rebuildQuery = medianQuery(rebuilt);
        const QueryOutput rebuiltRes =
            runQuery(rebuilt, buf, true);
        std::printf("rebuild total=%.3f ms query median_ms=%.3f avg_node_tests=%.1f "
                    "avg_tri_tests=%.1f\n",
                    rebuildMs, rebuildQuery, rebuiltRes.stats.nodeTests / nPoints,
                    rebuiltRes.stats.triTests / nPoints);
        if (opt.cpu) verify("rebuild", refCpu, rebuiltRes);

        // 第 1 步与第 2/3 步的几何不同，它的查询时间不参与比较，只比成本。
        std::printf("cost_vs_rebuild partial=%.2fx refit=%.2fx query refit/rebuild=%.2fx\n",
                    rebuildMs / partialMs, rebuildMs / fullMs, refitQuery / rebuildQuery);
        destroyBVHGPU(rebuilt);
    }

    destroyRadiusBufferGPU(radiusBuf);
    destroyPointBufferGPU(buf);
    destroyBVHGPU(gpu);
    return exitCode;
}
