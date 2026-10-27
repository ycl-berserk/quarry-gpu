#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/fingerprint.h"
#include "core/scene.h"
#include "cpu/brute.h"
#include "cpu/bvh.h"
#include "cpu/selftest.h"

namespace {

struct Options {
    std::string command;
    std::string scene = "terrain";
    std::string engine = "bvh";
    std::string queryType = "closest";  // closest | knn | radius
    int grid = 256;
    int randomTris = 0;
    int rays = -1;    // -1 = 按引擎给默认值
    int points = -1;  // 查询点数，-1 = 按引擎给默认值
    int k = 16;
    float radius = 2.0f;
    int runs = 5;
    uint64_t seed = 1;
    RayMode mode = RayMode::Down;
    bool stats = false;
};

void printUsage() {
    std::printf(
        "quarry - GPU spatial query engine (CPU reference build)\n"
        "\n"
        "  quarry selftest\n"
        "  quarry run [options]\n"
        "  quarry query [options]\n"
        "\n"
        "options:\n"
        "  --scene terrain|random|mixed   default terrain\n"
        "  --grid N                       terrain 边长（格数），default 256 → 131072 三角形\n"
        "  --random-tris N                random/mixed 的随机三角形数\n"
        "  --engine brute|bvh             default bvh\n"
        "  --runs N                       计时重复次数，default 5（取中位数）\n"
        "  --seed N                       default 1\n"
        "  --stats                        额外打印平均遍历节点数/三角形数\n"
        "\n"
        "run 专用:\n"
        "  --rays N                       射线数\n"
        "  --mode down|random             down=全命中, random=远处随机朝向\n"
        "\n"
        "query 专用:\n"
        "  --type closest|knn|radius      default closest\n"
        "  --points N                     查询点数\n"
        "  --k N                          kNN 的 k，default 16\n"
        "  --radius R                     半径查询的半径，default 2.0\n");
}

bool parseArgs(int argc, char** argv, Options& opt) {
    if (argc < 2) return false;
    opt.command = argv[1];

    for (int i = 2; i < argc; ++i) {
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
        } else if (arg == "--engine") {
            const char* v = needValue("--engine");
            if (!v) return false;
            opt.engine = v;
        } else if (arg == "--grid") {
            const char* v = needValue("--grid");
            if (!v) return false;
            opt.grid = std::atoi(v);
        } else if (arg == "--random-tris") {
            const char* v = needValue("--random-tris");
            if (!v) return false;
            opt.randomTris = std::atoi(v);
        } else if (arg == "--rays") {
            const char* v = needValue("--rays");
            if (!v) return false;
            opt.rays = std::atoi(v);
        } else if (arg == "--runs") {
            const char* v = needValue("--runs");
            if (!v) return false;
            opt.runs = std::max(1, std::atoi(v));
        } else if (arg == "--seed") {
            const char* v = needValue("--seed");
            if (!v) return false;
            opt.seed = std::strtoull(v, nullptr, 10);
        } else if (arg == "--mode") {
            const char* v = needValue("--mode");
            if (!v) return false;
            opt.mode = std::strcmp(v, "random") == 0 ? RayMode::Random : RayMode::Down;
        } else if (arg == "--type") {
            const char* v = needValue("--type");
            if (!v) return false;
            opt.queryType = v;
        } else if (arg == "--points") {
            const char* v = needValue("--points");
            if (!v) return false;
            opt.points = std::atoi(v);
        } else if (arg == "--k") {
            const char* v = needValue("--k");
            if (!v) return false;
            opt.k = std::atoi(v);
        } else if (arg == "--radius") {
            const char* v = needValue("--radius");
            if (!v) return false;
            opt.radius = static_cast<float>(std::atof(v));
        } else if (arg == "--stats") {
            opt.stats = true;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            return false;
        }
    }

    if (opt.scene != "terrain" && opt.scene != "random" && opt.scene != "mixed") {
        std::fprintf(stderr, "unknown scene: %s\n", opt.scene.c_str());
        return false;
    }
    if (opt.engine != "brute" && opt.engine != "bvh") {
        std::fprintf(stderr, "unknown engine: %s\n", opt.engine.c_str());
        return false;
    }
    if (opt.queryType != "closest" && opt.queryType != "knn" && opt.queryType != "radius") {
        std::fprintf(stderr, "unknown query type: %s\n", opt.queryType.c_str());
        return false;
    }
    return true;
}

double nowMs() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

int runBenchmark(Options opt) {
    if (opt.rays < 0) opt.rays = opt.engine == "brute" ? 2000 : 200000;
    if (opt.scene == "random" && opt.randomTris == 0) opt.randomTris = 200000;
    if (opt.scene == "mixed" && opt.randomTris == 0) opt.randomTris = 50000;

    const SceneSpec spec{opt.scene, opt.grid, opt.randomTris, opt.seed};

    const double t0 = nowMs();
    const Scene scene = buildScene(spec);
    const double sceneMs = nowMs() - t0;

    const std::vector<Ray> rays = makeRays(scene.bounds, opt.mode, opt.rays, opt.seed);

    BVH bvh;
    double buildMs = 0.0;
    if (opt.engine == "bvh") {
        const double t1 = nowMs();
        bvh = buildBVH(scene);
        buildMs = nowMs() - t1;
    }

    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(opt.runs));

    RayStats stats;
    for (int run = 0; run < opt.runs; ++run) {
        RayStats runStats;
        const double t = nowMs();

        if (opt.engine == "brute") {
            for (const Ray& ray : rays) runStats.add(raycastBrute(scene, ray));
        } else {
            for (const Ray& ray : rays) runStats.add(raycastBVH(bvh, ray));
        }

        timings.push_back(nowMs() - t);
        if (run == 0) stats = runStats;
    }

    std::sort(timings.begin(), timings.end());
    const double median = timings[timings.size() / 2];
    const double best = timings.front();
    const double worst = timings.back();

    std::printf("scene=%s tris=%zu rays=%d mode=%s seed=%llu engine=%s\n", opt.scene.c_str(),
                scene.tris.size(), opt.rays, rayModeName(opt.mode),
                static_cast<unsigned long long>(opt.seed), opt.engine.c_str());
    std::printf("scene_ms=%.2f build_ms=%.2f", sceneMs, buildMs);
    if (opt.engine == "bvh") {
        std::printf(" nodes=%zu depth=%zu", bvh.nodes.size(), bvhDepth(bvh));
    }
    std::printf("\n");
    std::printf("runs=%d median_ms=%.2f min_ms=%.2f max_ms=%.2f Mrays_s=%.2f\n", opt.runs, median,
                best, worst, median > 0.0 ? opt.rays / (median * 1e3) : 0.0);
    std::printf("fingerprint %s\n", stats.summary().c_str());

    if (opt.stats && opt.engine == "bvh") {
        BVHStats total;
        for (const Ray& ray : rays) raycastBVH(bvh, ray, &total);
        const double n = static_cast<double>(rays.size());
        std::printf("traversal avg_node_tests=%.1f avg_tri_tests=%.1f\n",
                    total.nodeTests / n, total.triTests / n);
    }

    // 只有 runs=1 时 fingerprint 才代表这次输出；多次运行时上面打印的是第 0 次。
    return 0;
}

int runQuery(Options opt) {
    if (opt.points < 0) opt.points = opt.engine == "brute" ? 1000 : 20000;
    if (opt.scene == "random" && opt.randomTris == 0) opt.randomTris = 200000;
    if (opt.scene == "mixed" && opt.randomTris == 0) opt.randomTris = 50000;

    const SceneSpec spec{opt.scene, opt.grid, opt.randomTris, opt.seed};

    const double t0 = nowMs();
    const Scene scene = buildScene(spec);
    const double sceneMs = nowMs() - t0;

    const std::vector<Vec3> points = makeQueryPoints(scene.bounds, opt.points, opt.seed);

    BVH bvh;
    double buildMs = 0.0;
    if (opt.engine == "bvh") {
        const double t1 = nowMs();
        bvh = buildBVH(scene);
        buildMs = nowMs() - t1;
    }

    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(opt.runs));

    QueryStats stats;
    for (int run = 0; run < opt.runs; ++run) {
        QueryStats runStats;
        const double t = nowMs();

        for (const Vec3& p : points) {
            if (opt.queryType == "closest") {
                runStats.add(opt.engine == "brute" ? closestPointBrute(scene, p)
                                                   : closestPointBVH(bvh, p));
            } else if (opt.queryType == "knn") {
                runStats.add(opt.engine == "brute" ? kNearestBrute(scene, p, opt.k)
                                                   : kNearestBVH(bvh, p, opt.k));
            } else {
                runStats.add(opt.engine == "brute" ? radiusBrute(scene, p, opt.radius)
                                                   : radiusBVH(bvh, p, opt.radius));
            }
        }

        timings.push_back(nowMs() - t);
        if (run == 0) stats = runStats;
    }

    std::sort(timings.begin(), timings.end());
    const double median = timings[timings.size() / 2];
    const double best = timings.front();
    const double worst = timings.back();

    std::printf("scene=%s tris=%zu points=%d type=%s", opt.scene.c_str(), scene.tris.size(),
                opt.points, opt.queryType.c_str());
    if (opt.queryType == "knn") std::printf(" k=%d", opt.k);
    if (opt.queryType == "radius") std::printf(" radius=%.4f", opt.radius);
    std::printf(" seed=%llu engine=%s\n", static_cast<unsigned long long>(opt.seed),
                opt.engine.c_str());

    std::printf("scene_ms=%.2f build_ms=%.2f", sceneMs, buildMs);
    if (opt.engine == "bvh") {
        std::printf(" nodes=%zu depth=%zu", bvh.nodes.size(), bvhDepth(bvh));
    }
    std::printf("\n");
    std::printf("runs=%d median_ms=%.2f min_ms=%.2f max_ms=%.2f Mqueries_s=%.2f "
                "results_per_query=%.2f\n",
                opt.runs, median, best, worst,
                median > 0.0 ? opt.points / (median * 1e3) : 0.0,
                opt.points > 0 ? static_cast<double>(stats.results) / opt.points : 0.0);
    std::printf("fingerprint %s\n", stats.summary().c_str());

    if (opt.stats && opt.engine == "bvh") {
        BVHStats total;
        for (const Vec3& p : points) {
            if (opt.queryType == "closest") {
                closestPointBVH(bvh, p, &total);
            } else if (opt.queryType == "knn") {
                kNearestBVH(bvh, p, opt.k, &total);
            } else {
                radiusBVH(bvh, p, opt.radius, &total);
            }
        }
        const double n = static_cast<double>(points.size());
        std::printf("traversal avg_node_tests=%.1f avg_tri_tests=%.1f\n",
                    total.nodeTests / n, total.triTests / n);
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        printUsage();
        return 1;
    }

    if (opt.command == "selftest") return runSelfTest() ? 0 : 3;
    if (opt.command == "run") return runBenchmark(opt);
    if (opt.command == "query") return runQuery(opt);

    std::fprintf(stderr, "unknown command: %s\n", opt.command.c_str());
    printUsage();
    return 1;
}
