#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/fingerprint.h"
#include "core/gaussian.h"
#include "core/geometry.h"
#include "core/point_cloud.h"
#include "core/scene.h"

// GPU 后端的主机侧 API。头文件里不出现任何 CUDA 类型，设备指针一律用 void*：
// 没装 CUDA 的机器（CI、纯 CPU 构建）也能包含它、链接到空实现之外的分支。
//
// 树是 LBVH（Morton + 基数排序 + Karras 建树），节点布局与 learn/03 一致：
//   内部节点 0 .. n-2，叶子 n-1 .. 2n-2。
struct GpuBVH {
    void* d_nodes = nullptr;    // GpuBVHNode[]，共 2n-1 个
    void* d_tris = nullptr;     // Tri[] 或 Vec3[]（点云），按叶子顺序重排
    void* d_primIds = nullptr;  // int[]，tris[i] 对应的场景索引
    int primCount = 0;
    int nodeCount = 0;
    int depth = 0;

    // refit 用的内部节点重放顺序：按 (深度, 编号) 排好，配合 refitOffsets 逐层发射。
    // 拓扑不变时它也不变，所以建树时算一次、之后每次 refit 直接复用。
    void* d_refitOrder = nullptr;
    std::vector<int> refitOffsets;  // 第 d 层的内部节点在 d_refitOrder 里的 [off[d], off[d+1])
    std::vector<int> primSlot;      // 场景索引 -> 叶子槽，只上传改动三角形时用
    void* d_stage = nullptr;        // 全量 refit 的上传暂存区，首次用到才分配
    int stageCount = 0;
    void* d_dirtySlots = nullptr;   // 局部 refit 的暂存区，按需增长，不每次 cudaMalloc
    void* d_dirtyTris = nullptr;
    int dirtyCapacity = 0;

    // 只合并脏叶子的祖先：沿 parent 链向上标记，按层收集，然后逐层发射。
    // d_dirtyOrder 与 d_refitOrder 用同一套层区间，所以每层不会越界。
    void* d_parent = nullptr;       // int[]，nodeCount，根为 -1
    void* d_depth = nullptr;        // int[]，nodeCount
    void* d_mark = nullptr;         // int[]，nodeCount，记录最后一次标记的 epoch
    void* d_dirtyOrder = nullptr;   // int[]，internalCount
    void* d_levelCounts = nullptr;  // int[]，每层本次要合并的节点数
    void* d_refitOffsets = nullptr; // int[]，refitOffsets 的设备副本（标记 kernel 用）
    int refitEpoch = 0;

    // 可选的 4 宽折叠（buildWideNodesGPU 生成）。d_wideId 把二叉内部节点映射到
    // wide 下标，只对偶数深度的内部节点有效；refit 后重新折叠要复用它。
    void* d_wideNodes = nullptr;    // GpuBVH4Node[]
    void* d_wideId = nullptr;       // int[]，nodeCount
    int wideCount = 0;

    // 建树器。拓扑变更超出增量预算时要按同一套重建，调用方不必再记一遍。
    bool sah = false;

    // 最近一次整树均衡构建（含回退重建）之后的深度。增量更新只允许超出它
    // kTopoDepthSlack 层——连续同向追加会让某条路径逐轮变深，超限就整树重建
    // 重新平衡。0 表示还没记录过（第一次更新时用当时的深度初始化）。
    int topoBaseDepth = 0;
};

struct GpuBuildTimings {
    double mortonMs = 0.0;   // 重心 + Morton 码
    double sortMs = 0.0;     // cub::DeviceRadixSort
    double treeMs = 0.0;     // Karras 内部节点二分（SAH 路径为拓扑总耗时）
    double scatterMs = 0.0;  // 三角形按叶子顺序重排
    double refitMs = 0.0;    // 自底向上合并 AABB（含叶子盒与深度排序）
    double sahBinMs = 0.0;   // SAH：重心域 + 分箱
    double sahSplitMs = 0.0; // SAH：选切分 + 分区扫描
    double sahPartMs = 0.0;  // SAH：稳定分区 + 子节点分配
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

// 自顶向下分箱 SAH（16 箱 × 3 轴，区间 ≤64 图元退回对象中位数 + 串行建子树）。
// 输出与 LBVH 完全同构（内部/叶子编号、refit 缓存、宽节点折叠全部通用），
// 只是树形更好、构建更慢；默认路径仍是 LBVH，基准用 --sah 显式切换。
GpuBVH buildBVHSahGPU(const Scene& scene, GpuBuildTimings* timings = nullptr);
GpuBVH buildPointBVHSahGPU(const PointCloud& cloud, GpuBuildTimings* timings = nullptr);
GpuBVH buildGaussianBVHSahGPU(const GaussianCloud& cloud, GpuBuildTimings* timings = nullptr);

// 点云走同一条 LBVH 骨架：Morton 码按点坐标、叶子盒退化成单点。只有叶子载荷
// （Tri vs Vec3）和求距函数不同，见 gpu/prim_ops.h。
GpuBVH buildPointBVHGPU(const PointCloud& cloud, GpuBuildTimings* timings = nullptr);

// 高斯走同一条骨架：叶子盒是椭球外接盒、重心是均值，只有叶子测试换成
// "到椭球实体的距离"（见 core/gaussian.h）。构建/refit/校验/三种查询全部复用。
GpuBVH buildGaussianBVHGPU(const GaussianCloud& cloud, GpuBuildTimings* timings = nullptr);

// 在计时构建之前空跑一个小 kernel：CUDA 默认懒加载模块，第一次启动 kernel 才会
// 把整个 cubin 载入设备（~2 ms），不算这一次就把它记到构建头上。
void warmupBuildGPU();

// ---- 增量更新（M3 的第一块）----
// 拓扑冻结：只重算叶子盒、再自底向上合并内部节点，树的形状和三角形归属都不变。
// 因此第 i 个三角形必须还是建树时那个面片（数量相同、顺序不变），否则抛异常。
// 盒仍然是精确的并集，所以查询结果照样逐位正确；代价是盒子随位移变松、剪枝变差，
// 位移大到一定程度就该整树重建。
struct GpuRefitTimings {
    double uploadMs = 0.0;  // H2D（全量或只上传改动的三角形）
    double gatherMs = 0.0;  // 按 primIds 重排进叶子顺序 + 重写叶子盒
    double treeMs = 0.0;    // 逐层合并内部节点（增量路径还含祖先标记）
    double markMs = 0.0;    // 增量路径：沿 parent 链标记脏叶子的祖先
    double mergeMs = 0.0;   // 合并；全量路径即 treeMs
    double wideMs = 0.0;    // 有宽节点层时：重发受影响的宽节点
    double totalMs = 0.0;
    int updated = 0;        // 本次上传的三角形数
};

// 全量重传：场景整体动过（或调用方手里只有一份新数组）。
double refitBVHGPU(GpuBVH& bvh, const Scene& scene, GpuRefitTimings* timings = nullptr);

// 只上传 dirty 里的场景三角形索引（scene 仍传完整的数组）。除了省掉传输，树合并
// 也只处理这些叶子的祖先——耗时随改动量 × 树高走，不再与场景大小成正比。
double refitBVHGPU(GpuBVH& bvh, const Scene& scene, const std::vector<int>& dirty,
                   GpuRefitTimings* timings = nullptr);

double refitPointBVHGPU(GpuBVH& bvh, const PointCloud& cloud, GpuRefitTimings* timings = nullptr);
double refitPointBVHGPU(GpuBVH& bvh, const PointCloud& cloud, const std::vector<int>& dirty,
                        GpuRefitTimings* timings = nullptr);

double refitGaussianBVHGPU(GpuBVH& bvh, const GaussianCloud& cloud,
                           GpuRefitTimings* timings = nullptr);
double refitGaussianBVHGPU(GpuBVH& bvh, const GaussianCloud& cloud,
                           const std::vector<int>& dirty, GpuRefitTimings* timings = nullptr);

// ---- 拓扑变更（M3 收尾）----
// 编辑脚本：newScene = 旧场景删掉 deleted 之后的幸存者（保持原有相对顺序）+ 末尾
// 追加 appended 个图元。deleted 是旧场景下标（升序、不含重复），moved 是新场景下标
// （几何变了、拓扑不变）。不变量：newScene.size() == bvh.primCount - deleted + appended。
//
// 只有删除/追加才改变树形；改动落在少数几棵子树里时只重建那几棵，其余子树原样复用，
// 最后整块重编号（叶子恒为 n-1+slot）。改动太散、太深或占比过大时回退成整棵重建，
// timings->fullRebuild 会置 true。查询结果与"对 newScene 全新构建"逐位相同。
//
// 当前实现：切分选择（从删除叶子上溯到 ≤T 叶的 antichain）+ 森林 SAH 只重建受影响
// 的子树 + 全局规范重编号（内部节点按 (深度, 源序)，叶子 = n-1+slot）。新叶子槽是
// 旧槽序的压缩下标，保证子树覆盖的槽区间仍然连续；场景下标由 d_primIds 单独记录，
// 所以更新后 d_primIds 一般不再是 iota。
// 盒子只重算脏节点：重建森林的全部内部节点，加上切分根与移动叶子在新树里的祖先；
// 没被碰过的旧内部节点直接沿用旧盒。追加区间至少吸收与追加块同样多的旧叶子，
// 且深度超过 topoBaseDepth + 6 时整树重建重新平衡（R4 质量漂移）。
// 回退条件：删除/追加占比过大、重建叶数超预算、切分根过散、深度越界、n < 2。
//
// 已知代价（v1）：规划里的前缀和、载荷重排与整块重编号都是 O(n)，2M 三角形下更新
// 约是整棵重建的 1.6×；稳定槽位 / 节点池 / 缓冲复用留给 v1.1。
struct GpuTopoEdit {
    std::vector<int> deleted;
    std::vector<int> moved;
    int appended = 0;
};

struct GpuTopoUpdateTimings {
    double uploadMs = 0.0;
    double planMs = 0.0;      // 切分选择 + 预算判定
    double surgeryMs = 0.0;   // 受影响的子树重建
    double relayoutMs = 0.0;  // 整块重编号 + 叶子重发
    double mergeMs = 0.0;     // 新节点与祖先的盒子合并
    double wideMs = 0.0;      // 宽节点层重建
    double totalMs = 0.0;
    int deleted = 0;
    int moved = 0;            // 实际参与合并的移动图元数
    int appended = 0;
    int surgeryRoots = 0;     // 被重建的子树数
    int rebuiltLeaves = 0;    // 这些子树覆盖的叶子总数
    int newDepth = 0;
    bool fullRebuild = false; // 走了回退（整棵重建）
    double refitMs = 0.0;     // 纯移动编辑走 refit 快路径时的耗时
};

double updateBVHGPU(GpuBVH& bvh, const Scene& scene, const GpuTopoEdit& edit,
                    GpuTopoUpdateTimings* timings = nullptr);
double updatePointBVHGPU(GpuBVH& bvh, const PointCloud& cloud, const GpuTopoEdit& edit,
                         GpuTopoUpdateTimings* timings = nullptr);
double updateGaussianBVHGPU(GpuBVH& bvh, const GaussianCloud& cloud, const GpuTopoEdit& edit,
                            GpuTopoUpdateTimings* timings = nullptr);

void destroyBVHGPU(GpuBVH& bvh);

GpuRayBuffer uploadRaysGPU(const std::vector<Ray>& rays);
void destroyRayBufferGPU(GpuRayBuffer& buf);

// 遍历前的射线排序。Input 保持上传顺序；Morton 按原点、MortonDirection 按方向
// 做 63 位 Morton 序重排，让同一 warp 的射线命中同一块 BVH——结果原样散回原下标，
// 逐条输出与排序无关。排序的建码、CUB 排序、重排和散回都算在返回值里。
// 原点序适合主射线（相机/竖直射线），方向序适合原点分散的次生射线。
enum class RayOrder { Input, Morton, MortonDirection };

// 把二叉 LBVH 折叠成 4 宽节点（偶数深度的内部节点各占一个 128B 节点，孩子 +
// 孙子填满四个槽），返回折叠耗时。结果存进 bvh.d_wideNodes，可随时重新调用
// （refit 之后必须重调，因为折叠节点里存的是二叉节点的盒的拷贝）。
double buildWideNodesGPU(GpuBVH& bvh);

// 跑一遍遍历，返回 kernel 毫秒数（CUDA event 计时，不含拷贝）。
// wide = true 时走 4 宽节点，要求先调用 buildWideNodesGPU。
double raycastGPU(const GpuBVH& bvh, GpuRayBuffer& buf, bool collectStats = false,
                  RayOrder order = RayOrder::Input, bool wide = false);

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
    AABB bounds;                // 查询点包围盒，Morton 序的量化域（上传时在主机算好）
    void* d_sortScratch = nullptr;  // 内部：Morton 序重排的暂存区，首次用到才分配
};

struct GpuQueryStats {
    uint64_t nodeTests = 0;
    uint64_t triTests = 0;  // 叶子测试数；点云路径这里数的是点
    uint64_t stackOverflows = 0;
    double sortMs = 0.0;  // 上一次查询里排序/重排/散回的耗时，Input 顺序为 0
};

// 查询顺序。Input 保持上传顺序；Morton 把查询点量化到 10 位/轴做 30 位 Morton 序
// 重排，让同一 warp 的查询落在同一块 BVH——节点访问从随机读变成 L1/L2 命中。
// 每条查询相互独立，结果按原下标散回，逐点输出与顺序无关。
enum class QueryOrder { Input, Morton };

GpuPointBuffer uploadPointsGPU(const std::vector<Vec3>& points, int resultsPerPoint = 1);
void destroyPointBufferGPU(GpuPointBuffer& buf);

// 单点最近三角形：每线程一个查询点，显式栈 DFS。
// 剪枝与 CPU 参考实现同规则（下界 <= 当前最优才入栈），并列时取场景索引更小者，
// 因此逐点的 (dist2 位模式, triangle) 与 closestPointBVH 完全一致。
// wide = true 时走 4 宽折叠节点（先调用 buildWideNodesGPU）；结果与二叉路径逐位相同。
double closestPointGPU(const GpuBVH& bvh, GpuPointBuffer& buf, bool collectStats = false,
                       QueryOrder order = QueryOrder::Input, bool wide = false);

// k 近邻：每线程维护一个按 (dist2, 场景索引) 升序的定长表（上限 32），
// 满 k 后用第 k 近当剪枝阈值，替换规则与 CPU 的大顶堆一致。
// 输出按查询点顺序每个 stride=k 条，不足 k 条的位置 triangle = -1。
double kNearestGPU(const GpuBVH& bvh, GpuPointBuffer& buf, int k, bool collectStats = false,
                   QueryOrder order = QueryOrder::Input, bool wide = false);

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
                      bool collectStats = false, QueryOrder order = QueryOrder::Input,
                      bool wide = false);

// 按查询点顺序把 CSR 结果拷回主机，还原成 ClosestHit 列表。
std::vector<std::vector<ClosestHit>> readRadiusResultsGPU(const GpuRadiusBuffer& buf);

// ---- 点云查询（M2）----
// 与三角形路径共用遍历骨架和输出格式，只是叶子载荷换成 Vec3、距离换成点距。
// GpuClosestResult.triangle 里装的是点索引，其余读写接口完全复用。
double closestVertexGPU(const GpuBVH& bvh, GpuPointBuffer& buf, bool collectStats = false,
                        QueryOrder order = QueryOrder::Input, bool wide = false);
double kNearestVertexGPU(const GpuBVH& bvh, GpuPointBuffer& buf, int k,
                         bool collectStats = false, QueryOrder order = QueryOrder::Input,
                         bool wide = false);
double radiusQueryVertexGPU(const GpuBVH& bvh, GpuPointBuffer& buf, float radius,
                            GpuRadiusBuffer& out, bool collectStats = false,
                            QueryOrder order = QueryOrder::Input, bool wide = false);

// ---- 高斯查询（M2 后半）----
// 距离是到椭球实体的平方距离（落在椭球内为 0）。GpuClosestResult.triangle 里装
// 的是高斯索引；输出格式与读写接口和点云完全相同。
double closestGaussianGPU(const GpuBVH& bvh, GpuPointBuffer& buf, bool collectStats = false,
                          QueryOrder order = QueryOrder::Input, bool wide = false);
double kNearestGaussianGPU(const GpuBVH& bvh, GpuPointBuffer& buf, int k,
                           bool collectStats = false, QueryOrder order = QueryOrder::Input,
                           bool wide = false);
double radiusQueryGaussianGPU(const GpuBVH& bvh, GpuPointBuffer& buf, float radius,
                              GpuRadiusBuffer& out, bool collectStats = false,
                              QueryOrder order = QueryOrder::Input, bool wide = false);

// 把显存结果拷回主机（按查询点顺序）。
std::vector<GpuClosestResult> readQueryResultsGPU(const GpuPointBuffer& buf);

// 上次 collectStats 的节点/三角形测试总数。
GpuQueryStats lastQueryStatsGPU();
void warmupQueryGPU();

// 检查树的结构不变量：父盒逐位等于两个子盒的并、叶子盒逐位等于三角形包围盒、
// 每个非根节点被恰好引用一次、根盒等于场景包围盒。返回空串表示通过。
std::string validateBVHGPU(const GpuBVH& bvh, const Scene& scene);
std::string validatePointBVHGPU(const GpuBVH& bvh, const PointCloud& cloud);
std::string validateGaussianBVHGPU(const GpuBVH& bvh, const GaussianCloud& cloud);

// 节点数组的 64 位 FNV 指纹（只读）。确定性门禁：同一批脚本跑两次必须同值。
uint64_t treeHashGPU(const GpuBVH& bvh);
