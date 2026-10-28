#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/fingerprint.h"
#include "core/geometry.h"
#include "core/scene.h"

// GPU 后端的主机侧 API。头文件里不出现任何 CUDA 类型，设备指针一律用 void*：
// 没装 CUDA 的机器（CI、纯 CPU 构建）也能包含它、链接到空实现之外的分支。
//
// 树是 LBVH（Morton + 基数排序 + Karras 建树），节点布局与 learn/03 一致：
//   内部节点 0 .. n-2，叶子 n-1 .. 2n-2。
struct GpuBVH {
    void* d_nodes = nullptr;    // GpuBVHNode[]，共 2n-1 个
    void* d_tris = nullptr;     // Tri[]，按叶子顺序重排
    void* d_primIds = nullptr;  // int[]，tris[i] 对应的场景索引
    int triCount = 0;
    int nodeCount = 0;
    int depth = 0;
};

struct GpuBuildTimings {
    double mortonMs = 0.0;   // 重心 + Morton 码
    double sortMs = 0.0;     // cub::DeviceRadixSort
    double treeMs = 0.0;     // Karras 内部节点二分
    double scatterMs = 0.0;  // 三角形按叶子顺序重排
    double refitMs = 0.0;    // 自底向上合并 AABB
    double totalMs = 0.0;
};

// 射线与结果都常驻显存：计时只包住 kernel，不含 H2D/D2H。
struct GpuRayBuffer {
    void* d_rays = nullptr;         // Ray[]
    void* d_results = nullptr;      // uint64_t[]，每条射线 (hit<<32 | t 的位表示)
    int count = 0;
    void* d_sortScratch = nullptr;  // 内部：Morton 序重排的暂存区，复用避免每轮重新分配
};

// 返回设备名与计算能力；没有可用设备时返回 false 并写 why。
bool gpuDescribe(std::string* why);

// 可能抛 std::runtime_error（CUDA 调用失败时）。
GpuBVH buildBVHGPU(const Scene& scene, GpuBuildTimings* timings = nullptr);

// 在计时构建之前空跑一个小 kernel：CUDA 默认懒加载模块，第一次启动 kernel 才会
// 把整个 cubin 载入设备（~2 ms），不算这一次就把它记到构建头上。
void warmupBuildGPU();

void destroyBVHGPU(GpuBVH& bvh);

GpuRayBuffer uploadRaysGPU(const std::vector<Ray>& rays);
void destroyRayBufferGPU(GpuRayBuffer& buf);

// 遍历前的射线排序。Input 保持上传顺序；Morton 按原点、MortonDirection 按方向
// 做 63 位 Morton 序重排，让同一 warp 的射线命中同一块 BVH——结果原样散回原下标，
// 逐条输出与排序无关。排序的建码、CUB 排序、重排和散回都算在返回值里。
// 原点序适合主射线（相机/竖直射线），方向序适合原点分散的次生射线。
enum class RayOrder { Input, Morton, MortonDirection };

// 跑一遍遍历，返回 kernel 毫秒数（CUDA event 计时，不含拷贝）。
double raycastGPU(const GpuBVH& bvh, GpuRayBuffer& buf, bool collectStats = false,
                  RayOrder order = RayOrder::Input);

// 把显存里的逐射线结果按射线顺序折叠成指纹；perRay 非空时顺便拷回主机。
RayStats summarizeGPU(const GpuRayBuffer& buf, std::vector<uint64_t>* perRay = nullptr);

// 上一次带 collectStats 的遍历的节点/三角形测试总数。
struct GpuTraversalStats {
    uint64_t nodeTests = 0;
    uint64_t triTests = 0;
    uint64_t stackOverflows = 0;
    double sortMs = 0.0;  // 上一次遍历里排序/重排的耗时，Input 顺序为 0
};
GpuTraversalStats lastTraversalStatsGPU();

// ---- 点查询（M1）----
// 结果常驻显存；tri < 0 表示没有命中（半径查询/空场景）。
struct GpuClosestResult {
    uint32_t dist2Bits = 0;  // 平方距离的 32 位浮点表示
    int32_t triangle = -1;   // 场景里的三角形索引
};

struct GpuPointBuffer {
    void* d_points = nullptr;   // Vec3[]
    void* d_results = nullptr;  // GpuClosestResult[]，每个查询点 stride 条
    int count = 0;
    int stride = 1;             // 每个查询点的结果槽数（closest=1，kNN=k）
};

struct GpuQueryStats {
    uint64_t nodeTests = 0;
    uint64_t triTests = 0;
    uint64_t stackOverflows = 0;
};

GpuPointBuffer uploadPointsGPU(const std::vector<Vec3>& points, int resultsPerPoint = 1);
void destroyPointBufferGPU(GpuPointBuffer& buf);

// 单点最近三角形：每线程一个查询点，显式栈 DFS。
// 剪枝与 CPU 参考实现同规则（下界 <= 当前最优才入栈），并列时取场景索引更小者，
// 因此逐点的 (dist2 位模式, triangle) 与 closestPointBVH 完全一致。
double closestPointGPU(const GpuBVH& bvh, GpuPointBuffer& buf, bool collectStats = false);

// k 近邻：每线程维护一个按 (dist2, 场景索引) 升序的定长表（上限 32），
// 满 k 后用第 k 近当剪枝阈值，替换规则与 CPU 的大顶堆一致。
// 输出按查询点顺序每个 stride=k 条，不足 k 条的位置 triangle = -1。
double kNearestGPU(const GpuBVH& bvh, GpuPointBuffer& buf, int k, bool collectStats = false);

// 半径查询的输出是 CSR（压缩稀疏行）：第 i 个查询点的结果是
// [offsets[i], offsets[i+1]) 区间，段内按 (dist2, 场景索引) 升序——和 CPU 的
// radiusBVH 同一把尺子。结果条数不设上限，靠"计数 → 前缀和 → 填充 → 分段排序"
// 两遍遍历撑起来，因此比 closest/kNN 多一遍遍历。
struct GpuRadiusBuffer {
    void* d_offsets = nullptr;  // int[count + 1]
    void* d_keys = nullptr;     // uint64[total]，(dist2 位模式 << 32) | 场景索引，段内升序
    int count = 0;
    int total = 0;
};

void destroyRadiusBufferGPU(GpuRadiusBuffer& buf);

// 一次完整的半径查询流水线，返回 GPU 侧总毫秒数（计数、前缀和、填充、分段排序；
// 不含 H2D/D2H 结果拷贝）。out 每次调用都会重新分配。
double radiusQueryGPU(const GpuBVH& bvh, GpuPointBuffer& buf, float radius, GpuRadiusBuffer& out,
                      bool collectStats = false);

// 按查询点顺序把 CSR 结果拷回主机，还原成 ClosestHit 列表。
std::vector<std::vector<ClosestHit>> readRadiusResultsGPU(const GpuRadiusBuffer& buf);

// 把显存结果拷回主机（按查询点顺序）。
std::vector<GpuClosestResult> readQueryResultsGPU(const GpuPointBuffer& buf);

// 上次 collectStats 的节点/三角形测试总数。
GpuQueryStats lastQueryStatsGPU();
void warmupQueryGPU();

// 检查树的结构不变量：父盒逐位等于两个子盒的并、叶子盒逐位等于三角形包围盒、
// 每个非根节点被恰好引用一次、根盒等于场景包围盒。返回空串表示通过。
std::string validateBVHGPU(const GpuBVH& bvh, const Scene& scene);
