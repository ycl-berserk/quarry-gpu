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
    enum class RaySort { None, Host, Gpu, GpuDir };
    RaySort raySort = RaySort::None;
    int diagRay = -1;
    int warmupMs = 3000;
    int leaf = 4;
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
        "  --warmup-ms N                  计时前持续负载毫秒数，default 3000\n"
        "  --leaf N                       每叶三角形数，default 4\n"
        "  --seed N                       default 1\n"
        "  --stats                        额外跑一轮带计数的遍历\n"
        "  --ray-sort                     GPU 内按原点 Morton 序重排射线（端到端计时）\n"
        "  --ray-sort-dir                 GPU 内按方向 Morton 序重排（次生射线用）\n"
        "  --ray-sort-host                主机排序后上传，只测遍历收益（对照实验）\n"
        "  --diag-ray N                   打印第 N 条射线的 CPU/GPU/暴力三方命中\n"
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
        } else if (arg == "--mode") {
            const char* v = needValue("--mode");
            if (!v) return false;
            opt.mode = std::strcmp(v, "random") == 0 ? RayMode::Random : RayMode::Down;
        } else if (arg == "--stats") {
            opt.stats = true;
        } else if (arg == "--ray-sort") {
            opt.raySort = Options::RaySort::Gpu;
        } else if (arg == "--ray-sort-dir") {
            opt.raySort = Options::RaySort::GpuDir;
        } else if (arg == "--ray-sort-host") {
            opt.raySort = Options::RaySort::Host;
        } else if (arg == "--diag-ray") {
            const char* v = needValue("--diag-ray");
            if (!v) return false;
            opt.diagRay = std::atoi(v);
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

// 21 位/轴的 Morton 码：把原点量化到场景包围盒，让相邻线程落在同一块空间里。
uint64_t mortonKey(const Vec3& p, const AABB& b) {
    constexpr int kBits = 21;
    constexpr float kMax = static_cast<float>((1u << kBits) - 1);
    auto quantize = [](float v, float lo, float hi) -> uint32_t {
        if (!(hi > lo)) return 0;
        float f = (v - lo) / (hi - lo);
        f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
        return static_cast<uint32_t>(f * kMax + 0.5f);
    };
    const uint32_t q[3] = {quantize(p.x, b.mn.x, b.mx.x), quantize(p.y, b.mn.y, b.mx.y),
                           quantize(p.z, b.mn.z, b.mx.z)};
    uint64_t key = 0;
    for (int i = kBits - 1; i >= 0; --i) {
        key = (key << 3) | (((q[0] >> i) & 1u) << 2) | (((q[1] >> i) & 1u) << 1) |
              ((q[2] >> i) & 1u);
    }
    return key;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        printUsage();
        return argc > 1 ? 1 : 0;
    }
    int exitCode = 0;

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

    const char* orderName = opt.raySort == Options::RaySort::Gpu    ? "morton-gpu"
                            : opt.raySort == Options::RaySort::GpuDir ? "morton-dir-gpu"
                            : opt.raySort == Options::RaySort::Host ? "morton-host"
                                                                    : "input";
    std::printf("scene=%s tris=%zu rays=%zu mode=%s seed=%llu ray_order=%s\n", opt.scene.c_str(),
                scene.tris.size(), rays.size(), rayModeName(opt.mode),
                static_cast<unsigned long long>(opt.seed), orderName);
    const bool hostSorted = opt.raySort == Options::RaySort::Host;
    const RayOrder gpuOrder = opt.raySort == Options::RaySort::Gpu      ? RayOrder::Morton
                              : opt.raySort == Options::RaySort::GpuDir ? RayOrder::MortonDirection
                                                                        : RayOrder::Input;

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
        warmupBuildGPU();  // 模块懒加载的一次性开销不算进构建
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
    if (gpu.nodeCount != 2 * gpu.internalCount + 1) {
        std::printf("gpu tree FAIL: node count %d != 2*%d+1\n", gpu.nodeCount, gpu.internalCount);
        return 4;
    }

    const std::string treeError = validateBVHGPU(gpu, scene);
    std::printf("gpu tree %s%s\n", treeError.empty() ? "PASS" : "FAIL: ", treeError.c_str());
    if (!treeError.empty()) return 4;

    // ---- 可选：按原点 Morton 序重排射线，只影响遍历顺序，不影响逐条结果 ----
    std::vector<Ray> uploadRays = rays;
    std::vector<int> sortedToOrig;
    if (hostSorted) {
        const double sortStart = nowMs();
        std::vector<uint64_t> keys(rays.size());
        for (size_t i = 0; i < rays.size(); ++i) keys[i] = mortonKey(rays[i].origin, scene.bounds);
        sortedToOrig.resize(rays.size());
        for (size_t i = 0; i < rays.size(); ++i) sortedToOrig[i] = static_cast<int>(i);
        std::sort(sortedToOrig.begin(), sortedToOrig.end(), [&](int a, int b) {
            return keys[a] != keys[b] ? keys[a] < keys[b] : a < b;
        });
        for (size_t i = 0; i < rays.size(); ++i) uploadRays[i] = rays[sortedToOrig[i]];
        std::printf("ray_sort host_ms=%.2f\n", nowMs() - sortStart);
    }

    const double uploadStart = nowMs();
    GpuRayBuffer rayBuf = uploadRaysGPU(uploadRays);
    const double uploadMs = nowMs() - uploadStart;
    std::printf("upload_ms=%.2f\n", uploadMs);

    // 第一次 kernel 会带上上下文初始化；之后持续跑 warmup_ms，让 DVFS 把 SM 时钟
    // 拉到稳定档。笔记本上 32 ms 级的遍历 kernel 单独跑多少次都升不了频，实测
    // 冷态 780 MHz、热态 2025 MHz，同一二进制的吞吐差 2.5× 以上。
    raycastGPU(gpu, rayBuf, false, gpuOrder);
    const double warmStart = nowMs();
    int warmRuns = 0;
    while (nowMs() - warmStart < opt.warmupMs) {
        raycastGPU(gpu, rayBuf, false, gpuOrder);
        ++warmRuns;
    }
    std::printf("gpu warmup_ms=%.0f runs=%d\n", nowMs() - warmStart, warmRuns);

    std::vector<double> gpuTimes;
    for (int run = 0; run < opt.runs; ++run)
        gpuTimes.push_back(raycastGPU(gpu, rayBuf, false, gpuOrder));
    const double gpuMedian = median(gpuTimes);
    const double gpuBest = *std::min_element(gpuTimes.begin(), gpuTimes.end());
    const double gpuWorst = *std::max_element(gpuTimes.begin(), gpuTimes.end());

    std::printf("gpu runs=%d median_ms=%.2f min_ms=%.2f max_ms=%.2f Mrays_s=%.2f\n", opt.runs,
                gpuMedian, gpuBest, gpuWorst,
                gpuMedian > 0.0 ? static_cast<double>(rays.size()) / (gpuMedian * 1e3) : 0.0);

    if (opt.stats) {
        raycastGPU(gpu, rayBuf, true, gpuOrder);
        const GpuTraversalStats st = lastTraversalStatsGPU();
        const double n = static_cast<double>(rays.size());
        std::printf("gpu traversal avg_node_tests=%.1f avg_tri_tests=%.1f stack_overflows=%llu "
                    "ray_sort_ms=%.2f\n",
                    static_cast<double>(st.nodeTests) / n, static_cast<double>(st.triTests) / n,
                    static_cast<unsigned long long>(st.stackOverflows), st.sortMs);
        if (st.stackOverflows != 0) {
            std::printf("gpu traversal FAIL: stack overflow\n");
            return 5;
        }
    }

    // ---- GPU 指纹（重排过就先还原成原始射线顺序再折叠） ----
    std::vector<uint64_t> gpuPerRay;
    RayStats gpuStats = summarizeGPU(rayBuf, (opt.verify || hostSorted) ? &gpuPerRay : nullptr);
    if (hostSorted) {
        std::vector<uint64_t> inOrigOrder(rays.size());
        for (size_t i = 0; i < rays.size(); ++i) inOrigOrder[sortedToOrig[i]] = gpuPerRay[i];
        gpuPerRay = std::move(inOrigOrder);
        gpuStats = RayStats{};
        for (uint64_t v : gpuPerRay) {
            RayHit h;
            h.hit = (v >> 32) != 0;
            if (h.hit) {
                const uint32_t bits = static_cast<uint32_t>(v);
                std::memcpy(&h.t, &bits, sizeof(bits));
            }
            gpuStats.add(h);
        }
    }
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
        if (hitMismatch || tMismatch) exitCode = 6;
    }

    // ---- 单条射线三方对拍：CPU BVH / GPU BVH / 暴力 ----
    if (opt.diagRay >= 0 && opt.diagRay < static_cast<int>(rays.size())) {
        const int i = opt.diagRay;
        RayHit bf;
        bf.t = rays[i].tMax;
        for (size_t k = 0; k < scene.tris.size(); ++k) {
            float t;
            if (intersectTriangle(rays[i], scene.tris[k], t) && t < bf.t) {
                bf.hit = true;
                bf.t = t;
                bf.triangle = static_cast<int>(k);
            }
        }
        auto printHit = [](const char* who, uint64_t v) {
            const bool hit = (v >> 32) != 0;
            uint32_t bits = static_cast<uint32_t>(v);
            float t = 0.0f;
            std::memcpy(&t, &bits, sizeof(t));
            std::printf("diag %s hit=%d t=%.9g bits=0x%08x\n", who, hit ? 1 : 0, t, bits);
        };
        std::printf("diag ray=%d\n", i);
        if (!cpuPerRay.empty()) printHit("cpu", cpuPerRay[i]);
        if (!gpuPerRay.empty()) printHit("gpu", gpuPerRay[i]);
        uint64_t bfv = 0;
        if (bf.hit) {
            uint32_t bits = 0;
            std::memcpy(&bits, &bf.t, sizeof(bits));
            bfv = (uint64_t{1} << 32) | bits;
        }
        printHit("brute", bfv);
        std::printf("diag brute triangle=%d\n", bf.triangle);
    }

    destroyRayBufferGPU(rayBuf);
    destroyBVHGPU(gpu);
    return exitCode;
}
