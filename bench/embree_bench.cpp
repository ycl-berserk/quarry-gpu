// Embree 基准：使用与 quarry 完全相同的场景生成器和射线，保证是同一份输入。
// 这是项目"GPU 比 CPU 快多少"这句话的唯一合法来源——两边喂的东西必须一样。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <embree4/rtcore.h>

#include "core/fingerprint.h"
#include "core/scene.h"
#include "cpu/brute.h"

namespace {

struct Options {
    int grid = 1024;
    int randomTris = 0;
    int buildings = 65536;
    std::string scene = "terrain";
    std::string query = "rays";  // rays | closest
    int rays = 200000;
    int points = 20000;
    int threads = 1;
    int runs = 5;
    bool packets = false;
    int verify = 0;
    uint64_t seed = 1;
    RayMode mode = RayMode::Down;
};

void printUsage() {
    std::printf(
        "embree_bench [options]\n"
        "  --scene terrain|random|mixed|city   default terrain\n"
        "  --grid N                       三角形数 = 2*N^2\n"
        "  --random-tris N\n"
        "  --buildings N                  city 的楼数（每栋 10 个三角形），default 65536\n"
        "  --query rays|closest           default rays\n"
        "  --rays N                       default 200000\n"
        "  --points N                     closest 模式的查询点数，default 20000\n"
        "  --threads N                    default 1\n"
        "  --runs N                       default 5\n"
        "  --mode down|random             default down\n"
        "  --packets                      use rtcIntersect4（只对 rays 有效）\n"
        "  --verify N                     与暴力法交叉核对 N 条射线 / N 个查询点\n");
}

bool parseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--scene") {
            const char* v = value();
            if (!v) return false;
            opt.scene = v;
        } else if (arg == "--grid") {
            const char* v = value();
            if (!v) return false;
            opt.grid = std::atoi(v);
        } else if (arg == "--random-tris") {
            const char* v = value();
            if (!v) return false;
            opt.randomTris = std::atoi(v);
        } else if (arg == "--buildings") {
            const char* v = value();
            if (!v) return false;
            opt.buildings = std::atoi(v);
        } else if (arg == "--rays") {
            const char* v = value();
            if (!v) return false;
            opt.rays = std::atoi(v);
        } else if (arg == "--query") {
            const char* v = value();
            if (!v) return false;
            opt.query = v;
        } else if (arg == "--points") {
            const char* v = value();
            if (!v) return false;
            opt.points = std::atoi(v);
        } else if (arg == "--threads") {
            const char* v = value();
            if (!v) return false;
            opt.threads = std::max(1, std::atoi(v));
        } else if (arg == "--runs") {
            const char* v = value();
            if (!v) return false;
            opt.runs = std::max(1, std::atoi(v));
        } else if (arg == "--mode") {
            const char* v = value();
            if (!v) return false;
            opt.mode = std::strcmp(v, "random") == 0 ? RayMode::Random : RayMode::Down;
        } else if (arg == "--verify") {
            const char* v = value();
            if (!v) return false;
            opt.verify = std::atoi(v);
        } else if (arg == "--packets") {
            opt.packets = true;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            return false;
        }
    }
    if (opt.scene == "random" && opt.randomTris == 0) opt.randomTris = 200000;
    if (opt.scene == "mixed" && opt.randomTris == 0) opt.randomTris = 50000;
    if (opt.query != "rays" && opt.query != "closest") {
        std::fprintf(stderr, "unknown query mode: %s\n", opt.query.c_str());
        return false;
    }
    return true;
}

double nowMs() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

void checkDevice(RTCDevice device) {
    if (RTCError err = rtcGetDeviceError(device); err != RTC_ERROR_NONE) {
        std::fprintf(stderr, "embree device error: %d\n", static_cast<int>(err));
        std::exit(1);
    }
}

RayHit cast1(RTCScene scene, const Ray& ray) {
    RTCRayHit rh;
    std::memset(&rh, 0, sizeof(rh));
    rh.ray.org_x = ray.origin.x;
    rh.ray.org_y = ray.origin.y;
    rh.ray.org_z = ray.origin.z;
    rh.ray.dir_x = ray.dir.x;
    rh.ray.dir_y = ray.dir.y;
    rh.ray.dir_z = ray.dir.z;
    rh.ray.tnear = ray.tMin;
    rh.ray.tfar = ray.tMax;
    rh.ray.mask = 0xFFFFFFFFu;
    rh.ray.flags = 0;
    rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(scene, &rh);

    RayHit out;
    if (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        out.hit = true;
        out.t = rh.ray.tfar;
        out.triangle = static_cast<int>(rh.hit.primID);
    }
    return out;
}

void cast4(RTCScene scene, const Ray* rays, RayHit* out4) {
    RTCRayHit4 rh;
    std::memset(&rh, 0, sizeof(rh));
    int valid[4] = {-1, -1, -1, -1};

    for (int k = 0; k < 4; ++k) {
        rh.ray.org_x[k] = rays[k].origin.x;
        rh.ray.org_y[k] = rays[k].origin.y;
        rh.ray.org_z[k] = rays[k].origin.z;
        rh.ray.dir_x[k] = rays[k].dir.x;
        rh.ray.dir_y[k] = rays[k].dir.y;
        rh.ray.dir_z[k] = rays[k].dir.z;
        rh.ray.tnear[k] = rays[k].tMin;
        rh.ray.tfar[k] = rays[k].tMax;
        rh.ray.mask[k] = 0xFFFFFFFFu;
        rh.ray.flags[k] = 0;
        rh.hit.geomID[k] = RTC_INVALID_GEOMETRY_ID;
        rh.hit.instID[0][k] = RTC_INVALID_GEOMETRY_ID;
    }

    rtcIntersect4(valid, scene, &rh);

    for (int k = 0; k < 4; ++k) {
        out4[k] = RayHit{};
        if (rh.hit.geomID[k] != RTC_INVALID_GEOMETRY_ID) {
            out4[k].hit = true;
            out4[k].t = rh.ray.tfar[k];
            out4[k].triangle = static_cast<int>(rh.hit.primID[k]);
        }
    }
}

struct PointQueryData {
    const std::vector<Tri>* tris = nullptr;
    ClosestHit best;
};

// Embree 只负责"哪些三角形的包围盒落在这个球里"，精确距离还是自己算。
// 返回 true 表示半径变了，Embree 会据此收紧后续遍历；这正是最优优先的等价物。
bool pointQueryFunc(RTCPointQueryFunctionArguments* args) {
    auto* data = static_cast<PointQueryData*>(args->userPtr);
    RTCPointQuery* query = args->query;
    const Vec3 p{query->x, query->y, query->z};

    ClosestHit h = closestPointOnTriangle(p, (*data->tris)[args->primID]);
    h.triangle = static_cast<int>(args->primID);
    if (!data->best.found || closerHit(h, data->best)) data->best = h;

    if (h.dist2 < query->radius * query->radius) {
        query->radius = std::sqrt(h.dist2);
        return true;
    }
    return false;
}

ClosestHit closestPointEmbree(RTCScene scene, const std::vector<Tri>& tris, const Vec3& p) {
    RTCPointQueryContext context;
    rtcInitPointQueryContext(&context);

    RTCPointQuery query{};
    query.x = p.x;
    query.y = p.y;
    query.z = p.z;
    query.radius = std::numeric_limits<float>::infinity();
    query.time = 0.0f;

    PointQueryData data;
    data.tris = &tris;
    rtcPointQuery(scene, &query, &context, pointQueryFunc, &data);
    return data.best;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        printUsage();
        return 1;
    }

    const SceneSpec spec{opt.scene, opt.grid, opt.randomTris, opt.seed, opt.buildings};
    const Scene scene = buildScene(spec);
    const std::vector<Ray> rays =
        opt.query == "rays" ? makeRays(scene.bounds, opt.mode, opt.rays, opt.seed)
                            : std::vector<Ray>{};
    if (scene.tris.empty()) {
        std::fprintf(stderr, "empty scene\n");
        return 1;
    }

    RTCDevice device = rtcNewDevice(nullptr);
    checkDevice(device);

    RTCScene embreeScene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    const size_t numTris = scene.tris.size();
    const size_t numVerts = numTris * 3;
    float* vb = static_cast<float*>(rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), numVerts));
    unsigned* ib = static_cast<unsigned*>(rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), numTris));

    for (size_t i = 0; i < numTris; ++i) {
        const Tri& t = scene.tris[i];
        const Vec3 vs[3] = {t.a, t.b, t.c};
        for (int k = 0; k < 3; ++k) {
            vb[(i * 3 + k) * 3 + 0] = vs[k].x;
            vb[(i * 3 + k) * 3 + 1] = vs[k].y;
            vb[(i * 3 + k) * 3 + 2] = vs[k].z;
            ib[i * 3 + k] = static_cast<unsigned>(i * 3 + k);
        }
    }
    rtcCommitGeometry(geom);
    rtcAttachGeometry(embreeScene, geom);
    rtcReleaseGeometry(geom);

    const double tBuild0 = nowMs();
    rtcCommitScene(embreeScene);
    const double buildMs = nowMs() - tBuild0;
    checkDevice(device);

    if (opt.query == "closest") {
        const std::vector<Vec3> points = makeQueryPoints(scene.bounds, opt.points, opt.seed);
        std::vector<ClosestHit> results(points.size());

        std::vector<double> timingRuns;
        timingRuns.reserve(static_cast<size_t>(opt.runs));
        for (int run = 0; run < opt.runs; ++run) {
            const int chunk =
                (static_cast<int>(points.size()) + opt.threads - 1) / opt.threads;

            const double t0 = nowMs();
            std::vector<std::thread> workers;
            for (int w = 0; w < opt.threads; ++w) {
                const int begin = w * chunk;
                const int end = std::min(static_cast<int>(points.size()), begin + chunk);
                if (begin >= end) break;

                workers.emplace_back([&, begin, end]() {
                    for (int i = begin; i < end; ++i) {
                        results[i] = closestPointEmbree(embreeScene, scene.tris, points[i]);
                    }
                });
            }
            for (auto& w : workers) w.join();
            timingRuns.push_back(nowMs() - t0);
        }

        std::sort(timingRuns.begin(), timingRuns.end());
        const double median = timingRuns[timingRuns.size() / 2];

        QueryStats stats;
        for (const ClosestHit& h : results) stats.add(h);

        std::printf("scene=%s tris=%zu points=%d threads=%d\n", opt.scene.c_str(), numTris,
                    opt.points, opt.threads);
        std::printf("build_ms=%.2f runs=%d median_ms=%.2f min_ms=%.2f max_ms=%.2f Mqueries_s=%.2f\n",
                    buildMs, opt.runs, median, timingRuns.front(), timingRuns.back(),
                    median > 0.0 ? points.size() / (median * 1e3) : 0.0);
        std::printf("embree %s\n", stats.summary().c_str());

        if (opt.verify > 0) {
            const int count = std::min(opt.verify, static_cast<int>(points.size()));
            int idMismatch = 0;
            int tieMismatch = 0;
            int distMismatch = 0;
            float worstRel = 0.0f;
            for (int i = 0; i < count; ++i) {
                const ClosestHit a = closestPointBrute(scene, points[i]);
                const ClosestHit b = closestPointEmbree(embreeScene, scene.tris, points[i]);
                if (a.found != b.found || a.triangle != b.triangle) {
                    ++idMismatch;
                    // 等距不同的三角形是合法的"两个正确答案"，只要距离一致就不算错。
                    if (a.found && b.found && a.dist2 == b.dist2) ++tieMismatch;
                }
                if (a.found != b.found) {
                    ++distMismatch;
                    continue;
                }
                if (!a.found) continue;
                const float rel = std::fabs(a.dist2 - b.dist2) / std::max(1.0f, a.dist2);
                worstRel = std::max(worstRel, rel);
                if (rel > 1e-6f) ++distMismatch;
            }
            std::printf(
                "verify points=%d id_mismatch=%d (ties=%d) dist2_mismatch=%d worst_rel=%.3g\n",
                count, idMismatch, tieMismatch, distMismatch, worstRel);
            if (distMismatch || idMismatch != tieMismatch) {
                rtcReleaseScene(embreeScene);
                rtcReleaseDevice(device);
                return 2;
            }
        }

        rtcReleaseScene(embreeScene);
        rtcReleaseDevice(device);
        return 0;
    }

    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(opt.runs));

    RayStats stable;
    for (int run = 0; run < opt.runs; ++run) {
        // 每个线程独立统计，避免共享 RayStats 产生数据竞争；结束后只合并命中数与距离和。
        // 以"组"为单位切分：标量模式 1 条射线一组，packet 模式 4 条一组，
        // 保证分组边界对齐，不会出现两条线程重复处理同一批射线。
        std::vector<RayStats> partial(static_cast<size_t>(opt.threads));
        const int groupSize = opt.packets ? 4 : 1;
        const int groups = static_cast<int>(rays.size()) / groupSize;
        const int chunk = (groups + opt.threads - 1) / opt.threads;

        const double t0 = nowMs();
        std::vector<std::thread> workers;
        for (int w = 0; w < opt.threads; ++w) {
            const int begin = w * chunk;
            const int end = std::min(groups, begin + chunk);
            if (begin >= end) break;

            workers.emplace_back([&, begin, end, w]() {
                RayStats& s = partial[w];
                if (opt.packets) {
                    RayHit tmp[4];
                    for (int g = begin; g < end; ++g) {
                        cast4(embreeScene, &rays[g * 4], tmp);
                        s.add(tmp[0]);
                        s.add(tmp[1]);
                        s.add(tmp[2]);
                        s.add(tmp[3]);
                    }
                } else {
                    for (int g = begin; g < end; ++g) s.add(cast1(embreeScene, rays[g]));
                }
            });
        }
        for (auto& w : workers) w.join();
        timings.push_back(nowMs() - t0);

        RayStats runStats;
        for (const RayStats& s : partial) {
            runStats.hits += s.hits;
            runStats.rays += s.rays;
            runStats.sumT += s.sumT;
        }
        if (run == 0) stable = runStats;
    }

    std::sort(timings.begin(), timings.end());
    const double median = timings[timings.size() / 2];

    std::printf("scene=%s tris=%zu rays=%d mode=%s threads=%d packets=%d\n", opt.scene.c_str(),
                numTris, static_cast<int>(rays.size()), rayModeName(opt.mode), opt.threads,
                opt.packets ? 1 : 0);
    std::printf("build_ms=%.2f runs=%d median_ms=%.2f min_ms=%.2f max_ms=%.2f Mrays_s=%.2f\n",
                buildMs, opt.runs, median, timings.front(), timings.back(),
                median > 0.0 ? rays.size() / (median * 1e3) : 0.0);
    std::printf("embree rays=%llu hits=%llu sum_t=%.6f\n",
                static_cast<unsigned long long>(stable.rays),
                static_cast<unsigned long long>(stable.hits), stable.sumT);

    if (opt.verify > 0) {
        const int count = std::min(opt.verify, static_cast<int>(rays.size()));
        int hitMismatch = 0;
        int tMismatch = 0;
        int validHits = 0;
        for (int i = 0; i < count; ++i) {
            const RayHit a = raycastBrute(scene, rays[i]);
            const RayHit b = cast1(embreeScene, rays[i]);
            if (a.hit != b.hit) {
                ++hitMismatch;
                continue;
            }
            if (a.hit) {
                ++validHits;
                if (std::fabs(a.t - b.t) > 1e-3f) ++tMismatch;
            }
        }
        std::printf("verify rays=%d hits=%d hit_mismatch=%d t_mismatch=%d\n", count, validHits,
                    hitMismatch, tMismatch);
        if (hitMismatch || tMismatch) return 2;
    }

    rtcReleaseScene(embreeScene);
    rtcReleaseDevice(device);
    return 0;
}
