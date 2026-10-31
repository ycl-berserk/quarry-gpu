# API 参考（v1.0）

两个入口：C++ 头文件（`quarry::core` / `quarry::gpu`）与 Python 模块 `quarry`。
本文按"先用哪个"组织：核心类型 → CPU 参考实现 → GPU 主机侧 API → Python 绑定。

所有几何都是 **float32**；距离默认是**平方距离**（省一次开方）；closest / kNN
未命中用 `-1`；`Ray::dir` 不要求归一化（`t` 的单位随 `dir` 长度变化）。GPU 头文件
不出现 CUDA 类型，设备指针一律 `void*`，没装 CUDA 的机器也能包含、编译 CPU 部分。

## 1. 核心类型（`core/geometry.h` 等）

```cpp
#include <core/geometry.h>
#include <core/scene.h>
#include <core/point_cloud.h>
#include <core/gaussian.h>
```

| 类型 | 字段 | 说明 |
|---|---|---|
| `Vec3` | `x, y, z` | 位置 / 方向 / 向量 |
| `AABB` | `mn, mx` | 包围盒；`extend`、`center`、`surfaceArea`、`valid` |
| `Ray` | `origin, dir, tMin=1e-6f, tMax=inf` | 方向不必归一化 |
| `Tri` | `a, b, c` | 三角形；`bounds()` 返回其 AABB |
| `RayHit` | `hit, t, triangle` | 射线结果，`triangle` 为场景索引 |
| `ClosestHit` | `found, dist2, triangle, point, bary1, bary2` | `point = a + bary1*(b-a) + bary2*(c-a)` |
| `Scene` | `tris, bounds` | 三角形场景；`updateBounds()` |
| `PointCloud` | `pts, bounds` | 点云；`updateBounds()` |
| `Gaussian` | `mean, rot, scale` | 椭球：均值、四元数（xyzw）、三个半轴（> 0） |
| `GaussianCloud` | `gaussians, bounds` | 椭球云；`updateBounds()` 用外接盒 |
| `Quat` | `x, y, z, w` | 旋转；使用时归一化，允许传未归一化 |

高斯的查询语义是**椭球实体**：查询点落在椭球内距离为 0，否则是到表面的欧氏距离。
`gaussianDist2(p, g)` 与 GPU 内核走同一串运算（固定 14 轮括号 + 牛顿/二分），
CPU/GPU 结果逐位一致。

结果排序统一按 `(dist2, 场景索引)` 升序（`closerHit`）——距离并列时顺序也唯一，
这是 CPU/GPU 指纹能对上的前提。

`intersectTriangle` 除了 u/v/t 判断，还要求命中参数落在**三角形自包围盒**的 slab
区间内。这不是额外的可见性规则，而是把"Möller–Trumbore 的浮点误差可能把盒外的点
判进"收口：命中点必须在盒内，BVH 的紧叶盒才不会剪掉暴力法会报告的结果。大坐标、
远距离的掠射射线会因此少掉极少数伪命中（实测约 1e-6），换来的是一致的 CPU/GPU
结论。实现见 `hitParamInsideBox`。

### 可复现输入

```cpp
SceneSpec spec;             // kind="terrain" | "random" | "mixed" | "city"
spec.grid = 1024;           // terrain: 三角形数 = 2 * grid^2
spec.buildings = 131072;    // city: 每栋 10 个三角形 + 128x128 地面
Scene scene = buildScene(spec);

auto rays = makeRays(scene.bounds, RayMode::Down, 1'000'000, /*seed=*/1);
auto pts  = makeQueryPoints(scene.bounds, 20'000, /*seed=*/1);

PointCloudSpec cloudSpec;   // kind="sheet" | "volume"
cloudSpec.count = 131072;
PointCloud cloud = buildPointCloud(cloudSpec);

GaussianCloudSpec gaussSpec; // kind="sheet" | "volume"
gaussSpec.count = 131072;
GaussianCloud gs = buildGaussianCloud(gaussSpec);
```

## 2. CPU 参考实现

它们是正确性基线，不追求速度；GPU 路径的每一个结果都可以在这里对拍。

```cpp
#include <cpu/bvh.h>        // buildBVH / raycastBVH / closestPointBVH / kNearestBVH / radiusBVH
#include <cpu/brute.h>      // 暴力法真值（三角形 / 点云 / 高斯）
#include <cpu/point_kd.h>   // 点云的单线程 kd-tree 基线
#include <cpu/gauss_bvh.h>  // 单线程椭球 BVH 基线
```

```cpp
BVH bvh = buildBVH(scene);                       // 隐式二叉树，叶子按叶序重排
RayHit hit = raycastBVH(bvh, ray);
ClosestHit c = closestPointBVH(bvh, p);
std::vector<ClosestHit> knn = kNearestBVH(bvh, p, 8);
std::vector<ClosestHit> ball = radiusBVH(bvh, p, 2.0f);
```

`BVH` 对外一律报**场景索引**（`bvh.primIds`），不是叶序槽位。

## 3. GPU 主机侧 API

```cpp
#include <gpu/gpu_bvh.h>
```

链接 `quarry::gpu`（它公开依赖 `quarry::core`）。可能抛 `std::runtime_error`
（CUDA 调用失败）或 `std::invalid_argument`（入参不合法）。

### 3.1 设备与构建

```cpp
std::string why;
if (!gpuDescribe(&why)) { /* 没有可用设备；why 里是原因 */ }

warmupBuildGPU();  // 计时前空跑一个小 kernel，避免把 CUDA 模块懒加载记到首轮构建

GpuBuildTimings t0;
GpuBVH bvh = buildBVHGPU(scene, &t0);                 // LBVH（Morton + 基数排序 + Karras）
GpuBVH sah = buildBVHSahGPU(scene, &t0);              // 自顶向下分箱 SAH，输出同构

GpuBVH pcloud = buildPointBVHGPU(cloud, &t0);         // 点云
GpuBVH pcloudSah = buildPointBVHSahGPU(cloud, &t0);
GpuBVH gauss = buildGaussianBVHGPU(cloud, &t0);       // 高斯（叶子盒 = 椭球外接盒）
GpuBVH gaussSah = buildGaussianBVHSahGPU(cloud, &t0);
```

`GpuBVH` 里对调用方有意义的只有三个计数：

| 字段 | 含义 |
|---|---|
| `primCount` | 图元数 n |
| `nodeCount` | 节点数 2n-1（内部 0..n-2，叶子 n-1..2n-2） |
| `depth` | 树深 |

其余字段是实现细节（中间缓冲、refit 缓存、宽节点、设备内存池），不要读写或复制
`GpuBVH` 对象；它不提供拷贝语义，生命周期由 `destroyBVHGPU` 收尾。

`GpuBuildTimings` 各段都是 CUDA event 或主机时钟，`totalMs` 是本次构建总计：
`mortonMs`、`sortMs`、`treeMs`、`scatterMs`、`refitMs`，SAH 另外拆出
`sahBinMs`、`sahSplitMs`、`sahPartMs`。

### 3.2 射线查询

```cpp
std::vector<Ray> rays = makeRays(scene.bounds, RayMode::Down, 1'000'000, 1);

GpuRayBuffer buf = uploadRaysGPU(rays);
warmupQueryGPU();

double ms = raycastGPU(bvh, buf, /*collectStats=*/false,
                       RayOrder::Morton, /*wide=*/false);
RayStats stats = summarizeGPU(buf);              // 命中数 + FNV 指纹
std::vector<uint64_t> perRay;
stats = summarizeGPU(buf, &perRay);              // 顺便拷回逐条结果
destroyRayBufferGPU(buf);
destroyBVHGPU(bvh);
```

- 返回的 `ms` **只含 kernel**（CUDA event），不含上传 / 回读。
- `RayOrder`：`Input` 保持上传顺序；`Morton` 按原点、`MortonDirection` 按方向做
  Morton 序重排（结果散回原下标，逐条输出与顺序无关）。排序的耗时包含在返回值里。
- `wide=true` 走 4 宽节点，必须先调用 `buildWideNodesGPU(bvh)`。
- `perRay[i] = (hit << 32) | t 的 32 位浮点表示`。
- `summarizeGPU` 的 `RayStats.hash` 与 CPU 端同一把尺子：同输入必须同值。

```cpp
double wideMs = buildWideNodesGPU(bvh);   // 折叠成 BVH4，结果存在 bvh 内部
```

refit 之后宽节点里的盒是旧的，**必须重新调用** `buildWideNodesGPU`。

`collectStats=true` 时，`lastTraversalStatsGPU()` 给出上一次遍历的
`nodeTests`、`triTests`、`stackOverflows`、`sortMs`。

### 3.3 点查询（三角形 / 点云 / 高斯共用骨架）

```cpp
GpuPointBuffer pts = uploadPointsGPU(queryPoints, /*resultsPerPoint=*/1);
double ms = closestPointGPU(bvh, pts);            // 三角形

GpuPointBuffer knnBuf = uploadPointsGPU(queryPoints, /*resultsPerPoint=*/8);
double ms2 = kNearestGPU(bvh, knnBuf, 8);         // 每点 8 条，stride 必须 >= k
std::vector<GpuClosestResult> res = readQueryResultsGPU(knnBuf);

GpuRadiusBuffer out;
double ms3 = radiusQueryGPU(bvh, pts, 2.0f, out); // 输出 CSR
std::vector<std::vector<ClosestHit>> hits = readRadiusResultsGPU(out);

destroyRadiusBufferGPU(out);
destroyPointBufferGPU(knnBuf);
destroyPointBufferGPU(pts);
```

单查 / kNN / 半径各自用一个 `GpuPointBuffer`：`resultsPerPoint` 在创建时固定，
`closest` 要求 `stride == 1`，kNN 要求 `stride >= k`。

| 查询 | 三角形 | 点云 | 高斯 |
|---|---|---|---|
| closest | `closestPointGPU` | `closestVertexGPU` | `closestGaussianGPU` |
| kNN | `kNearestGPU` | `kNearestVertexGPU` | `kNearestGaussianGPU` |
| radius | `radiusQueryGPU` | `radiusQueryVertexGPU` | `radiusQueryGaussianGPU` |

三条查询都接受 `collectStats`、`QueryOrder`（`Input` / `Morton`）与 `wide`，
语义与射线一致；`k` 上限 32。

- `GpuClosestResult{dist2Bits, triangle}`：`dist2Bits` 是平方距离的 float32 位模式，
  `triangle` 是图元在**场景**里的索引，未命中为 `-1`。
- kNN 输出按查询点连续存放，每个点 `stride = k` 条，不足 k 条的位置为 `-1`。
- radius 输出 `GpuRadiusBuffer{d_offsets, d_keys, count, total}`：第 i 个点的结果在
  `[offsets[i], offsets[i+1])`，段内按 `(dist2, 场景索引)` 升序；结果条数不设上限。
- `uploadPointsGPU` 只上传查询点；`GpuPointBuffer.bounds` 是主机侧算好的包围盒，
  Morton 排序的量化域。
- `collectStats=true` 时，`lastQueryStatsGPU()` 给出上一轮的 `nodeTests`、
  `triTests`（点云 / 高斯路径数的是叶子图元测试）、`stackOverflows`、`sortMs`。

查询是同步的：函数返回时结果已就绪，`read*` 只做 D2H。

### 3.4 增量更新：refit（拓扑冻结）

第 i 个图元必须还是建树时那个图元；数量相同、顺序不变。只重算叶子盒并自底向上
合并内部节点，树的形状和归属不变，结果仍然逐位精确，但盒子会随位移变松。

```cpp
GpuRefitTimings rt;
double ms = refitBVHGPU(bvh, scene, &rt);                 // 全量重传
double ms2 = refitBVHGPU(bvh, scene, dirtyIndices, &rt);  // 只上传 dirty 的场景索引

refitPointBVHGPU(bvh, cloud, &rt);                        // 点云 / 高斯同样的两组重载
refitGaussianBVHGPU(bvh, cloud, &rt);
```

局部路径除省传输外，树合并也只处理这些叶子的祖先，耗时随"改动量 × 树高"走。
`GpuRefitTimings` 拆出 `uploadMs`、`gatherMs`、`treeMs`、`markMs`、`mergeMs`、
`wideMs`、`totalMs` 与 `updated`（本次上传的图元数）。

`dirty` 是**场景索引**，升序不是必须的；未列出的图元必须确实没动——这是接口合同，
不会逐位校验。有宽节点层时，refit 末尾会重发受影响的宽节点。

### 3.5 拓扑更新（增删改）

```cpp
// newScene = 旧场景删掉 deleted 后的幸存者（保持相对顺序）
//          + 在 inserted 这些新下标上插入的新图元
//          + 末尾追加 appended 个图元
GpuTopoEdit edit;
edit.deleted  = {3, 17};        // 旧场景下标，升序、唯一
edit.inserted = {100, 250};     // 新场景下标，升序、唯一
edit.moved    = {42};           // 新场景下标：几何变了、拓扑不变
edit.appended = 10;             // 新场景最后 10 个位置

GpuTopoUpdateTimings tt;
double ms = updateBVHGPU(bvh, newScene, edit, &tt);
```

不变量：

```text
newScene.size() == bvh.primCount - deleted.size() + inserted.size() + appended
```

点云与高斯各有一份（`updatePointBVHGPU` / `updateGaussianBVHGPU`），编辑脚本语义
相同。改动只落在少数几棵子树时只重建那几棵，其余原样复用再整块重编号；改动太散、
太深或占比过大时回退整棵重建，`tt.fullRebuild` 置 `true`。无论走哪条路径，查询
结果都与"对 newScene 全新构建"逐位相同。

`GpuTopoUpdateTimings`：`uploadMs`、`planMs`、`surgeryMs`、`relayoutMs`、`mergeMs`、
`wideMs`、`totalMs`、`refitMs`（纯移动编辑走 refit 快路径时），以及
`deleted / moved / inserted / appended`、`surgeryRoots`、`rebuiltLeaves`、
`newDepth`、`fullRebuild`。

拓扑更新后场景序与槽序不再一致（新图元接在槽尾），但查询结果仍按场景索引报告，
调用方不需要关心槽位。

### 3.6 校验、指纹与生命周期

```cpp
std::string err = validateBVHGPU(bvh, scene);          // 空串 = 通过
std::string err2 = validatePointBVHGPU(bvh, cloud);
std::string err3 = validateGaussianBVHGPU(bvh, cloud);

uint64_t h = treeHashGPU(bvh);                         // 节点数组的 64 位 FNV 指纹

destroyBVHGPU(bvh);                                    // 每个 GpuBVH 都要收尾
```

`validate*` 检查：父盒逐位等于两个子盒的并、叶子盒逐位等于图元包围盒、每个非根
节点恰被引用一次、根盒等于场景包围盒。`treeHashGPU` 用于确定性门禁（同脚本跑两次
必须同值）。

## 4. Python 绑定

```python
import quarry            # 需要先 pip install . 或把 build/python 加进 sys.path
print(quarry.describe()) # 设备字符串；空串表示没有 GPU
```

所有数组是 `float32` / `int32`（`forcecast` 会替你转换）；构建与查询期间释放 GIL，
返回数组在重新拿到 GIL 之后创建。接口是**同步**的：每次调用都上传、内核、回读。

### 4.1 演示数据

```python
vertices, indices = quarry.demo_mesh(kind="terrain", grid=256, random_tris=0, seed=1)
# vertices: (N,3) float32；indices: (M,3) int32
# kind="terrain" | "random" | "mixed" | "city"

vertices, indices = quarry.demo_mesh(kind="city", buildings=131072, seed=1)
# 城市楼群：每栋 10 个三角形（4 面墙 + 屋顶），地面固定 128x128 地形。
# 默认 65,536 栋；131,072 栋约 1.34M 个三角形，跨度约 ±1452。

means, rot, scale = quarry.demo_gaussians(kind="sheet", count=131072, seed=1)
# means: (N,3)，rot: (N,4) xyzw，scale: (N,3) 半轴
```

### 4.2 `quarry.Scene`

```python
scene = quarry.Scene(vertices, indices)                 # 或 Scene(triangles)  # (M,3,3)
scene = quarry.Scene(vertices, indices, builder="sah")  # 默认 "lbvh"
```

| 成员 | 说明 |
|---|---|
| `triangle_count` / `node_count` / `depth` | 规模 |
| `build_ms` | 构建墙钟；进程里第一个 Scene 还付 CUDA 模块加载 |
| `last_query_ms` | 最近一次查询的内核时间（events，不含拷贝） |
| `refit_ms` | 最近一次更新的耗时（含 H2D） |
| `build_stats` / `refit_stats` | 分项计时字典（见 C++ 同名字段） |
| `bounds` | `{"min": (x,y,z), "max": (x,y,z)}` |
| `validate()` | 空串 = 通过 |

```python
dist2, tri = scene.closest(points)                  # (N,) float32, (N,) int32；未命中 -1
dist2, tri = scene.closest(points, order="morton")  # 先按查询点 Morton 序重排

kd2, ktri = scene.knn(points, k=8)                  # (N,8); 不足 8 条为 -1
offsets, rd2, rtri = scene.radius(points, radius=2.0)
# offsets: int64 (N+1)；第 i 个点的结果在 [offsets[i], offsets[i+1])

hit, t = scene.raycast(origins, dirs, tmax=float("inf"))
# hit: bool (N,)；t: float32 (N,)，未命中为 inf；dirs 不必归一化
```

更新（拓扑冻结，数量和顺序不变）：

```python
scene.update_triangles(new_triangles)            # 全量上传
scene.update_triangles(new_triangles, dirty)     # dirty: int32 场景索引数组
scene.update_vertices(new_vertices, dirty)       # 仅 (vertices, indices) 构造的 Scene
```

更新（拓扑可变）：

```python
timings = scene.update_topology(new_triangles, deleted=..., moved=..., inserted=...)
```

`new_triangles` 是**新场景**：幸存者（原顺序去掉 `deleted`）+ 在 `inserted` 指定的
新下标上插入的新三角形 + 末尾追加的三角形。尾部追加量由
`len(new) - 旧数量 + len(deleted) - len(inserted)` 自动倒推，不用显式传。
返回 `dict`：`total_ms`、`upload_ms`、`plan_ms`、`surgery_ms`、`relayout_ms`、
`merge_ms`、`wide_ms`、`refit_ms`、`deleted`、`moved`、`inserted`、`appended`、
`surgery_roots`、`rebuilt_leaves`、`new_depth`、`full_rebuild`。
`update_topology` 之后构造时的网格绑定失效，`update_vertices` 会报错。

### 4.3 `quarry.PointCloud`

```python
cloud = quarry.PointCloud(points, builder="lbvh")    # points: (N,3) float32
cloud.point_count / node_count / depth / bounds / validate()
```

查询与 `Scene` 同形，只是 `triangle` 列换成点索引：

```python
dist2, idx = cloud.closest(points)
kd2, kidx = cloud.knn(points, 8)
offsets, rd2, ridx = cloud.radius(points, 2.0)

cloud.update_points(new_points, dirty=None)
cloud.update_topology(new_points, deleted=None, moved=None, inserted=None)
```

### 4.4 `quarry.GaussianCloud`

```python
gs = quarry.GaussianCloud(means, rot, scale, builder="lbvh")
# means (N,3)、rot (N,4) xyzw（可不归一化）、scale (N,3) 半轴（> 0）
```

查询语义是**椭球实体**（内部距离 0），输出与点云同构：`dist2` + 高斯索引。

```python
dist2, idx = gs.closest(points)
kd2, kidx = gs.knn(points, 8)
offsets, rd2, ridx = gs.radius(points, 2.0)

gs.update(new_means, new_rot, new_scale, dirty=None)
gs.update_topology(new_means, new_rot, new_scale,
                   deleted=None, moved=None, inserted=None)
```

初始化参数不合法（形状、半轴非正、数量 < 2）会抛 `ValueError`；
`k` 不在 `[1, 32]`、`radius < 0`、`order` / `builder` 取值错误、更新时图元数量
改变，同样抛 `ValueError`；CUDA 调用失败抛 `RuntimeError`。

## 5. 最小 C++ 示例

仓库里的 `examples/` 是一个独立 CMake 工程，可直接对安装前缀编译：

```bat
cmake -S examples -B examples\build -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_PREFIX_PATH=C:\quarry
cmake --build examples\build
```

```cpp
#include <cstdio>
#include <vector>

#include <core/scene.h>
#include <gpu/gpu_bvh.h>

int main() {
    SceneSpec spec; spec.grid = 256;
    Scene scene = buildScene(spec);

    GpuBVH bvh = buildBVHGPU(scene);
    std::vector<Ray> rays = makeRays(scene.bounds, RayMode::Down, 1 << 20, 1);
    GpuRayBuffer buf = uploadRaysGPU(rays);
    double ms = raycastGPU(bvh, buf);
    RayStats stats = summarizeGPU(buf);
    std::printf("%.3f ms, %s\n", ms, stats.summary().c_str());

    destroyRayBufferGPU(buf);
    destroyBVHGPU(bvh);
}
```

注意每个 `upload*` 都有配对的 `destroy*`，`GpuBVH`、`GpuRayBuffer`、
`GpuPointBuffer`、`GpuRadiusBuffer` 都是句柄，不要拷贝、不要跨线程共享同一句柄。
多线程可以，但每个线程要有自己的句柄；GPU 侧没有全局锁。

## 6. 版本与兼容

- 当前版本 `1.1.0`，CMake 包按 `SameMajorVersion` 兼容（`find_package(quarry 1.0)`）。
- `GpuBVH` 内部字段（`d_*`、缓存、内存池）不属于公共 API，升级时可能变化；
  公开面是本文的函数签名、`primCount / nodeCount / depth` 与计时 / 编辑结构体。
- Python 的 `__version__` 与包版本一致；wheel 的本地构建方式见 README「安装」。
