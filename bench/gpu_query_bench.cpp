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
    int leaf = 4;
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
        "  --leaf N                       每叶三角形数，default 4\n"
        "  --seed N                       default 1\n"
        "  --stats                        额外跑一轮带节点/三角形计数的查询\n"
        "  --verify / --no-verify         是否逐点对拍（default verify）\n"
        "  --no-cpu                       跳过 CPU 参考查询（大场景省时间）\n"
        "  --dump FILE                    把场景与查询点写成二进制后退出（给外部引擎对照用）\n");
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
        } else if (arg == "--leaf") {
            const char* v = needValue("--leaf");
            if (!v) return false;
            opt.leaf = std::max(1, std::atoi(v));
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

    std::printf("scene=%s tris=%zu points=%zu type=%s", opt.scene.c_str(), scene.tris.size(),
                points.size(), opt.query.c_str());
    if (knnMode) std::printf(" k=%d", opt.k);
    if (radiusMode) std::printf(" radius=%g", static_cast<double>(opt.radius));
    std::printf(" seed=%llu\n", static_cast<unsigned long long>(opt.seed));

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
        gpu = buildBVHGPU(scene, &build, opt.leaf);
    } catch (const std::exception& e) {
        std::printf("gpu build failed: %s\n", e.what());
        return 3;
    }
    std::printf(
        "gpu build morton=%.2f sort=%.2f scatter=%.2f tree=%.2f refit=%.2f total=%.2f ms "
        "nodes=%d internal=%d leaf=%d depth=%d\n",
        build.mortonMs, build.sortMs, build.scatterMs, build.treeMs, build.refitMs, build.totalMs,
        gpu.nodeCount, gpu.internalCount, gpu.leafSize, gpu.depth);
    const std::string treeError = validateBVHGPU(gpu, scene);
    std::printf("gpu tree %s%s\n", treeError.empty() ? "PASS" : "FAIL: ", treeError.c_str());
    if (!treeError.empty()) return 4;

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
                radiusQueryGPU(gpu, buf, opt.radius, radiusBuf, false);
            } else if (knnMode) {
                kNearestGPU(gpu, buf, opt.k, false);
            } else {
                closestPointGPU(gpu, buf, false);
            }
            ++warmRuns;
        }
        std::printf("gpu warmup_ms=%.0f runs=%d\n", nowMs() - warmStart, warmRuns);

        for (int run = 0; run < opt.runs; ++run) {
            if (radiusMode) {
                destroyRadiusBufferGPU(radiusBuf);  // 每轮重新分配，分配时间不计入 event
                gpuTimes.push_back(radiusQueryGPU(gpu, buf, opt.radius, radiusBuf, false));
            } else {
                gpuTimes.push_back(knnMode ? kNearestGPU(gpu, buf, opt.k, false)
                                           : closestPointGPU(gpu, buf, false));
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
            radiusQueryGPU(gpu, buf, opt.radius, radiusBuf, true);
        } else if (knnMode) {
            kNearestGPU(gpu, buf, opt.k, true);
        } else {
            closestPointGPU(gpu, buf, true);
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

    destroyRadiusBufferGPU(radiusBuf);
    destroyPointBufferGPU(buf);
    destroyBVHGPU(gpu);
    return exitCode;
}
