// GPU LBVH 的基准与对拍：同一份场景、同一批射线，先跑 CPU 参考 BVH，
// 再跑 GPU LBVH，要求逐条射线的命中与 t 值逐位相同。
//
// 用法：
//   gpu_bench --grid 256 --rays 1000000 --runs 5
//   gpu_bench --grid 1024 --rays 1000000 --runs 5
//   gpu_bench --grid 256 --rays 200000 --verify --no-cpu

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
    int grid = 256;
    int randomTris = 0;
    int rays = 1000000;
    RayMode mode = RayMode::Down;
    int runs = 5;
    uint64_t seed = 1;
    bool verify = true;
    bool cpu = true;
    bool stats = false;
};

void printUsage() {
    std::printf(
        "gpu_bench - GPU LBVH build + ray traversal vs CPU reference BVH\n"
        "\n"
        "  --scene terrain|random|mixed   default terrain\n"
        "  --grid N                       terrain 边长（格数），default 256 → 131072 三角形\n"
        "  --random-tris N                random/mixed 的随机三角形数\n"
        "  --rays N                       default 1000000\n"
        "  --mode down|random             default down\n"
        "  --runs N                       计时轮数，取中位数，default 5\n"
        "  --seed N                       default 1\n"
        "  --stats                        额外跑一轮带计数的遍历\n"
        "  --verify / --no-verify         是否逐条射线对拍（default verify）\n"
        "  --no-cpu                       跳过 CPU 参考遍历（大场景省时间）\n");
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
        } else if (arg == "--stats") {
            opt.stats = true;
        } else if (arg == "--verify") {
            opt.verify = true;
        } else if (arg == "--no-verify") {
            opt.verify = false;
        } else if (arg == "--no-cpu") {
            opt.cpu = false;
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            return false;
        }
    }
    if (opt.scene != "terrain" && opt.scene != "random" && opt.scene != "mixed") {
        std::fprintf(stderr, "unknown scene: %s\n", opt.scene.c_str());
        return false;
    }
    return true;
}

uint64_t encodeHit(const RayHit& h) {
    if (!h.hit) return 0;
    uint32_t bits = 0;
    std::memcpy(&bits, &h.t, sizeof(bits));
    return (uint64_t{1} << 32) | static_cast<uint64_t>(bits);
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        printUsage();
        return argc > 1 ? 1 : 0;
    }

    std::string device;
    if (!gpuDescribe(&device)) {
        std::printf("no CUDA device: %s\n", device.c_str());
        return 2;
    }
    std::printf("device %s\n", device.c_str());

    if (opt.scene == "random" && opt.randomTris == 0) opt.randomTris = 200000;
    if (opt.scene == "mixed" && opt.randomTris == 0) opt.randomTris = 50000;

    const SceneSpec spec{opt.scene, opt.grid, opt.randomTris, opt.seed};
    const Scene scene = buildScene(spec);
    const std::vector<Ray> rays = makeRays(scene.bounds, opt.mode, opt.rays, opt.seed);

    std::printf("scene=%s tris=%zu rays=%zu mode=%s seed=%llu\n", opt.scene.c_str(),
                scene.tris.size(), rays.size(), rayModeName(opt.mode),
                static_cast<unsigned long long>(opt.seed));

    // ---- CPU 参考 ----
    BVH cpuBVH;
    double cpuBuildMs = 0.0;
    if (opt.cpu) {
        const double t0 = nowMs();
        cpuBVH = buildBVH(scene);
        cpuBuildMs = nowMs() - t0;
        std::printf("cpu build_ms=%.2f nodes=%zu depth=%zu\n", cpuBuildMs, cpuBVH.nodes.size(),
                    bvhDepth(cpuBVH));
    }

    // ---- GPU 构建 ----
    GpuBuildTimings build{};
    GpuBVH gpu;
    try {
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
    if (gpu.nodeCount != static_cast<int>(2 * scene.tris.size() - 1)) {
        std::printf("gpu tree FAIL: node count %d\n", gpu.nodeCount);
        return 4;
    }

    const std::string treeError = validateBVHGPU(gpu, scene);
    std::printf("gpu tree %s%s\n", treeError.empty() ? "PASS" : "FAIL: ", treeError.c_str());
    if (!treeError.empty()) return 4;

    const double uploadStart = nowMs();
    GpuRayBuffer rayBuf = uploadRaysGPU(rays);
    const double uploadMs = nowMs() - uploadStart;
    std::printf("upload_ms=%.2f\n", uploadMs);

    // 预热：第一次 kernel 会带上上下文初始化
    raycastGPU(gpu, rayBuf);

    std::vector<double> gpuTimes;
    for (int run = 0; run < opt.runs; ++run) gpuTimes.push_back(raycastGPU(gpu, rayBuf));
    const double gpuMedian = median(gpuTimes);
    const double gpuBest = *std::min_element(gpuTimes.begin(), gpuTimes.end());
    const double gpuWorst = *std::max_element(gpuTimes.begin(), gpuTimes.end());

    std::printf("gpu runs=%d median_ms=%.2f min_ms=%.2f max_ms=%.2f Mrays_s=%.2f\n", opt.runs,
                gpuMedian, gpuBest, gpuWorst,
                gpuMedian > 0.0 ? static_cast<double>(rays.size()) / (gpuMedian * 1e3) : 0.0);

    if (opt.stats) {
        raycastGPU(gpu, rayBuf, true);
        const GpuTraversalStats st = lastTraversalStatsGPU();
        const double n = static_cast<double>(rays.size());
        std::printf("gpu traversal avg_node_tests=%.1f avg_tri_tests=%.1f stack_overflows=%llu\n",
                    static_cast<double>(st.nodeTests) / n, static_cast<double>(st.triTests) / n,
                    static_cast<unsigned long long>(st.stackOverflows));
        if (st.stackOverflows != 0) {
            std::printf("gpu traversal FAIL: stack overflow\n");
            return 5;
        }
    }

    // ---- GPU 指纹 ----
    std::vector<uint64_t> gpuPerRay;
    const RayStats gpuStats = summarizeGPU(rayBuf, opt.verify ? &gpuPerRay : nullptr);
    std::printf("fingerprint gpu %s\n", gpuStats.summary().c_str());

    // ---- CPU 参考遍历 ----
    RayStats cpuStats;
    std::vector<double> cpuTimes;
    std::vector<uint64_t> cpuPerRay;
    if (opt.cpu) {
        cpuPerRay.resize(rays.size());
        for (int run = 0; run < opt.runs; ++run) {
            RayStats runStats;
            const double t0 = nowMs();
            const bool collect = run == 0 && opt.verify;
            for (size_t i = 0; i < rays.size(); ++i) {
                const RayHit h = raycastBVH(cpuBVH, rays[i]);
                runStats.add(h);
                if (collect) cpuPerRay[i] = encodeHit(h);
            }
            cpuTimes.push_back(nowMs() - t0);
            if (run == 0) cpuStats = runStats;
        }
        const double cpuMedian = median(cpuTimes);
        std::printf("cpu runs=%d median_ms=%.2f min_ms=%.2f max_ms=%.2f Mrays_s=%.2f\n", opt.runs,
                    cpuMedian, *std::min_element(cpuTimes.begin(), cpuTimes.end()),
                    *std::max_element(cpuTimes.begin(), cpuTimes.end()),
                    cpuMedian > 0.0 ? static_cast<double>(rays.size()) / (cpuMedian * 1e3) : 0.0);
        std::printf("speedup=%.2fx\n", gpuMedian > 0.0 ? cpuMedian / gpuMedian : 0.0);
        std::printf("fingerprint cpu %s\n", cpuStats.summary().c_str());
    }

    // ---- 逐条射线对拍 ----
    if (opt.verify && opt.cpu) {
        int hitMismatch = 0;
        int tMismatch = 0;
        int worstIdx = -1;
        double worstRel = 0.0;
        for (size_t i = 0; i < rays.size(); ++i) {
            if ((gpuPerRay[i] >> 32) != (cpuPerRay[i] >> 32)) {
                ++hitMismatch;
                continue;
            }
            if (gpuPerRay[i] != cpuPerRay[i]) {
                ++tMismatch;
                float tg, tc;
                const uint32_t bg = static_cast<uint32_t>(gpuPerRay[i]);
                const uint32_t bc = static_cast<uint32_t>(cpuPerRay[i]);
                std::memcpy(&tg, &bg, sizeof(tg));
                std::memcpy(&tc, &bc, sizeof(tc));
                const double rel = tc != 0.0f ? std::fabs(static_cast<double>(tg - tc) / tc) : 0.0;
                if (rel >= worstRel) {
                    worstRel = rel;
                    worstIdx = static_cast<int>(i);
                }
            }
        }
        std::printf("verify rays=%zu hit_mismatch=%d t_mismatch=%d", rays.size(), hitMismatch,
                    tMismatch);
        if (worstIdx >= 0) std::printf(" worst_rel=%.3g at ray %d", worstRel, worstIdx);
        std::printf("\n");
        if (hitMismatch || tMismatch) {
            destroyRayBufferGPU(rayBuf);
            destroyBVHGPU(gpu);
            return 6;
        }
    }

    destroyRayBufferGPU(rayBuf);
    destroyBVHGPU(gpu);
    return 0;
}
