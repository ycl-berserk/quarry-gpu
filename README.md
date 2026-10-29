# quarry

[![CI](https://github.com/ycl-berserk/quarry-gpu/actions/workflows/build.yml/badge.svg)](https://github.com/ycl-berserk/quarry-gpu/actions/workflows/build.yml)

GPU 原生空间查询引擎：构建加速结构（BVH / 八叉树 / 哈希网格），支持射线求交、kNN、半径查询、AABB 查询，图元覆盖三角形、点云、高斯，并支持增量更新。

许可证 Apache-2.0（见 `LICENSE`）。CPU 参考实现已完整；GPU 侧（CUDA 12.4，sm_89）已有
并行 LBVH 构建、光线遍历、closest / kNN / radius 三条点查询路径，拓扑冻结的增量更新
（refit），以及**三种图元**：三角形、点云、高斯。三者共用同一棵 LBVH 与同一套查询，
只在叶子里换"测什么"——高斯的叶子测试是精确的**点到椭球实体**距离。
没有 CUDA 工具链时自动降级为纯 CPU 构建，CI 与本文档的 CPU 数字不受影响。

## 现在能跑什么

```bat
build.bat                                  :: CMake + Ninja（用 VS 2022 自带的，无需另装）
build\quarry.exe selftest                  :: 正确性自测（43 项）
build\quarry.exe run --scene terrain --grid 256 --rays 1000000 --engine bvh
build\quarry.exe query --type closest --grid 256 --points 20000 --stats
build\quarry.exe query --type knn --k 16 --grid 256 --points 20000
build\quarry.exe query --type radius --radius 2.0 --grid 256 --points 20000

build\gpu_bench.exe --grid 256 --rays 1000000 --runs 5 --stats   :: GPU 建树 + 遍历，逐条对拍 CPU
build\gpu_query_bench.exe --grid 256 --points 20000 --runs 5 --stats          :: GPU 最近点，逐点对拍 CPU
build\gpu_query_bench.exe --grid 256 --points 20000 --query knn --k 16        :: GPU kNN
build\gpu_query_bench.exe --grid 256 --points 20000 --query radius --radius 2.0  :: GPU 半径查询
build\gpu_query_bench.exe --grid 256 --points 20000 --refit --refit-dirty 0.01  :: 局部 / 全量 refit + 重建对照

build\gpu_cloud_bench.exe --points 131072 --queries 20000                   :: 点云 closest，GPU 对 kd-tree
build\gpu_cloud_bench.exe --points 1048576 --queries 20000 --query knn --k 8
build\gpu_cloud_bench.exe --kind volume --points 1048576 --query radius --radius 1.0
build\gpu_cloud_bench.exe --points 1048576 --refit --refit-dirty 0.01 --refit-motion coherent

build\gpu_gauss_bench.exe --gaussians 131072 --queries 20000                :: 高斯 closest，GPU 对椭球 BVH
build\gpu_gauss_bench.exe --gaussians 1048576 --queries 20000 --query knn --k 8
build\gpu_gauss_bench.exe --kind volume --gaussians 1048576 --query radius --radius 1.0
build\gpu_gauss_bench.exe --gaussians 1048576 --refit --refit-dirty 0.01 --refit-motion coherent
```

输出包含一个**正确性指纹**（每条射线的命中标志 + t 值，按 FNV-1a 折叠）：

```text
scene=terrain tris=131072 rays=1000000 mode=down seed=1 engine=bvh
scene_ms=10.74 build_ms=49.74 nodes=65535 depth=16
runs=5 median_ms=352.37 min_ms=350.14 max_ms=357.38 Mrays_s=2.84
fingerprint rays=1000000 hits=1000000 sum_t=14087276.763081 hash=0x...
```

查询用另一套指纹：平方距离的 32 位表示 + 场景索引，并列按索引定序，
所以暴力法和 BVH 的 hash 可以逐位相同。

```text
scene=terrain tris=131072 points=20000 type=closest seed=1 engine=bvh
runs=3 median_ms=120.17 ... Mqueries_s=0.17 results_per_query=1.00
fingerprint queries=20000 results=20000 sum_dist2=392670.867797 hash=0x254ce454a3bba31f
```

GPU 版本必须给出**同样的 hits 和 hash**。这是项目里最硬的一条约束，比任何吞吐数字都重要。
三个规模（8k / 131k / 2M）上 GPU 与 CPU 的指纹已逐位相同，包括每一条射线的 `t` 位模式。

## 与 Embree 同条件对比

```bat
bench\build_embree.bat
bench\embree_bench.exe --grid 1024 --rays 1000000 --threads 32 --packets
bench\embree_bench.exe --grid 256 --rays 500 --threads 1 --verify 500
```

两边使用**同一份场景生成器和射线生成器**，所以数字可比。

| 三角形 | 实现 | 线程 | 构建 | 吞吐 |
|---|---|---|---|---|
| 13.1 万 | quarry 朴素 BVH | 1 | 49.7 ms | 2.84 Mrays/s |
| 13.1 万 | Embree 标量 | 1 | 6.7 ms | 4.56 Mrays/s |
| 13.1 万 | Embree packet | 32 | 5.5 ms | 96.45 Mrays/s |
| 200 万 | quarry 朴素 BVH | 1 | 1019 ms | 1.32 Mrays/s |
| 200 万 | Embree 标量 | 1 | 68 ms | 1.73 Mrays/s |
| 200 万 | Embree packet | 32 | 68 ms | 19.69 Mrays/s |

i9-14900HX（24 核 32 线程），1M 条射线，垂直向下，命中率 100%，5 轮中位数。

注意 CPU 基线随场景大小塌陷一个数量级：13 万三角形能跑 96 Mrays/s，200 万只有 19.7。前者受缓存保护，后者结构性访存受限。GPU 的倍数同时取决于场景大小和射线相干性，不能只报一个"3×"。

### GPU 后端

RTX 4060 Laptop（24 SM，55 W 功耗墙），同一场景、同一射线、同一 seed，GPU 计时前
空跑 3 s 拉高 DVFS 时钟。`排序端到端` = 建 Morton 码 + 基数排序 + 重排 + 遍历 +
结果散回，全部计入 GPU 时间：

| 三角形 | GPU 输入序 | GPU 排序端到端 | GPU 构建 | Embree 1T | Embree 32T packet | 输入序 / 32T | 排序后 / 32T |
|---|---|---|---|---|---|---|---|
| 8,192 | 215.10 Mrays/s | 463.05 Mrays/s | 1.0–2.2 ms | 16.34 | 136.37 | 1.58× | **3.40×** |
| 131,072 | 143.63 Mrays/s | 343.52 Mrays/s | 1.79 ms | 5.53 | 70.93 | 2.03× | **4.84×** |
| 2,097,152 | 62.98 Mrays/s | 227.05 Mrays/s | 14.63 ms | 2.05 | 26.21 | 2.40× | **8.66×** |

- **主射线按原点 Morton 序重排**后，三个规模分别是 Embree 32T packet 的 3.4× / 4.8× /
  8.7×；排序本身只花 0.5–2.2 ms。非相干次生射线排序无收益（只省 ~2% 遍历），默认不开启。
- **构建和 Embree 打平到领先 3–5×**（8k 持平，131k 3.3×，2M 4.9× 对
  `rtcCommitScene`），LBVH 的并行排序天然适合 GPU。refit 已从"全局 atomic 分桶"
  改成深度稳定基数排序 + 每线程展开 4 个节点：同一时段 A/B 下 2M refit
  6.2 → 2.7 ms、整树构建 11.0 → 7.4 ms。
- **绝对值受笔记本电源状态影响**：同一二进制跨会话可差 1.3–2.5×，引用数字必须带日期
  和配对口径；详见 `docs/baseline.md`。
- 结论来自 `gpu_bench.exe` 的实测，完整分阶段数据与反例记录见 `docs/baseline.md`。

### 点查询（GPU）

`gpu_query_bench.exe`，20,000 个随机查询点，同一份场景与查询点。closest 与同次会话的
Embree `rtcPointQuery`、Warp 1.6.2（`wp.mesh_query_point_no_sign`）对照；kNN / radius
在 Embree 和 Warp 里都没有原生入口，只与 CPU 参考对拍。每一行都是"quarry 跑完立刻
跑对手"，构建时间已扣掉 Warp 进程内第一次建 mesh 的初始化：

| 三角形 | closest | Warp 1.6.2 | Embree 32T | closest / 32T | closest / Warp | kNN k=16 | radius |
|---|---|---|---|---|---|---|---|
| 8,192 | 36.78 Mq/s | 38.58 | 4.81 | **7.6×** | 0.95× | 6.09 Mq/s | 13.54 Mq/s (r=2.0) |
| 131,072 | 13.42 Mq/s | 18.63 | 3.18 | **4.2×** | 0.72× | 3.51 Mq/s | 3.57 Mq/s (r=2.0) |
| 2,097,152 | 2.59 Mq/s | 3.12 | 0.85 | **3.0×** | 0.83× | 1.44 Mq/s | 6.22 Mq/s (r=0.5) |
| 1,000,000 random | 5.04 Mq/s | 4.37 | 0.31 | **16.3×** | 1.15× | — | — |

- 三条路径共用**每线程一个显式栈**的 DFS，剪枝规则与 CPU 参考完全一致（下界等于
  当前最优时仍入栈、并列取场景索引小者），所以 20,000 个点的结果与 CPU 参考逐位相同。
- closest 的 GPU DFS 反而比 CPU 最优优先遍历多测节点（2M 477 vs 135），但几千个查询
  并发把单次 70–100 ns 的冷缓存延迟藏掉了，这正是点查询最该由 GPU 改写的地方。
- **Warp 1.6.2 是更强的外部基线**：地形场景 131k 快 1.4×、2M 快 1.2×，8k 持平；随机
  三角形汤上我们快 1.15×。Warp 是 SAH 建树 + 每叶 4 个图元，我们是 Karras LBVH +
  每叶 1 个（2M → 419 万节点、深度 31）。把 Morton 相邻的 4 个三角形并成叶子实测
  **更慢**（射线 2.2×、kNN 1.8×，构建反而快 1.5×），所以差距在树质量；追它需要
  GPU 自顶向下 SAH 或宽节点。构建则是小场景 Warp 快 1.4–4×、2M 我们快 1.8×。
- kNN 每线程维护按 `(dist2, 索引)` 升序的定长表（k ≤ 32），满 k 后按第 k 近剪枝；
  radius 走 CSR 两遍遍历 + 前缀和，段内排序用每 warp 一个双调网络（长段退回 CUB）。
- radius 的计时含一次流水线中间的同步（CSR 总长度要先算完前缀和才知道）。
  kNN / radius 的 CPU 参考是 0.2–1.8 Mq/s，GPU 快 6–21×。
- **注意精度口径**：quarry 的结果与 CPU 参考逐位相同；Warp 用 float32 重建最近点，
  选中的三角形一致但距离末几位不同（200 个采样点上 `suboptimal=0`）。

## 点云（M2 前半，2026-10-29）

三角形和点共用同一棵 LBVH、同一套 refit 与同一份 DFS，只有叶子里的三个函数不同
（`src/gpu/prim_ops.h` 的 `TriOps` / `PointOps`）。所以下面这张表里的数字和上面的
三角形查询是**同一个内核**，不是另一份实现。

**基线口径**：Embree 4.4 的 `rtcPointQuery` 只对三角形网格求最近点，没有点云图元；
Warp 也只有 `mesh_query_point`。所以点云查询没有 Embree 数字可引用，对照是
**自写的中位数切分 kd-tree**（`src/cpu/point_kd.cpp`，单线程），第二把尺子是
scipy 1.17 的 `cKDTree`。引用倍数时必须写明"对哪个 kd-tree"。

sheet 点云（x/z ∈ [-50, 50]，y 贴地形面 + 2 cm 抖动），20,000 个随机查询点，
3×3×3 矩阵连跑 3 次、每格取中位数：

| 点数 | GPU 构建 | closest | kNN k=8 | radius | 结果数/查询（radius） |
|---|---|---|---|---|---|
| 131,072 | 1.58 ms | 0.64 ms (32.7×) | 1.90 ms (17.8×) | 1.10 ms (14.5×) | 11.2 |
| 1,048,576 | 5.19 ms | 1.81 ms (27.8×) | 3.78 ms (20.3×) | 13.22 ms (8.9×) | 89.8 |
| 10,000,000 | 49.96 ms | 5.81 ms (52.1×) | 8.07 ms (49.3×) | 65.54 ms (27.9×) | 855.5 |

- 括号内是**对同一进程内的单线程 kd-tree 的倍数**。10M 的 GPU 构建在 50.0 ms
  （morton 1.9 / 排 11.4 / 散射 5.2 / 建树 7.1 / refit 24.2），kd 建树是 2.5 s。
- **closest 只碰 110–393 个节点、11–21 个点/查询**，单线程一次 1.0–15.1 µs，几乎
  全是冷缓存行——这是"延迟受限"最纯粹的形态，也是 GPU 倍数随规模上升的原因
  （33× → 52×）。
- **radius 的成本由输出条数主导**：默认半径（对角线 1%）下 10M 每查询吐 855 条，
  GPU 只剩 0.31 Mq/s。按 `r ∝ n^(-1/3)` 把输出拉回约 11 条/查询后，
  1M 是 1.36 ms（13.7×）、10M 是 5.25 ms（3.9×）——10M 的 radius 优势明显收窄，
  报数必须带这个口径。
- **scipy `cKDTree` 的对照要小心**：在 sheet 数据上它会退化，而且随密度加深——
  与自家 kd 比，131k 慢 2.6×、1M 慢 15×、10M 慢 75×（单次查询 1.14 ms）。
  换成体积点云（`--kind volume`）后差距正常：1M 是 32.1 vs 19.4 ms、10M 是 87.7 vs
  35.5 ms（scipy 慢 1.7–2.5×），同会话 GPU 对 kd 的配对倍数是 60.4× / 55.5×
  （热状态不同时会漂到 62.6× / 75.2×）。也就是说自家 kd 比成熟库还快，
  headline 倍数是保守的。`bench/scipy_kd_bench.py` 读 `gpu_cloud_bench --dump`
  写出的同一批浮点数。
- 增量更新：1% 相干位移下，1M 局部 refit 0.86 ms（重建 5.54 ms，6.4×）、
  10M 8.88 ms（重建 50.14 ms，5.6×）。**全量 refit 在这里反而比重建快**
  （10M 38.0 vs 50.1 ms），因为点云重建的 Morton 排序 + 建树比上传 120 MB 更贵；
  这与三角形在 2M 上的结论相反，别混着引用。
- 正确性：9 格 × 3 次重复全部 `id_mismatch=0 dist_mismatch=0 count_mismatch=0`，
  CPU/GPU 指纹逐位相同；refit 后结构校验 PASS；closest 与 radius 还和 scipy 的
  `sum_dist2` + 指纹逐位一致（独立实现的第三个见证），kNN 的 `sum_dist2` 相对差
  在 1e-13 量级（scipy 用 float64 距离选点）。

## 高斯（M2 后半，2026-10-29）

语义先钉死：**椭球实体**。查询点在高斯内部时距离为 0，在外时是到椭球表面的欧氏距离。
`radius` 于是回答"哪些椭球与这个球相交"。中心密度（Mahalanobis）和屏幕空间投影不做，
理由见第 09 章——前者不是欧氏距离、剪枝规则要重写，后者依赖相机、失去静态索引的意义。

**基线口径**：`rtcPointQuery` 只对三角网格找最近点，Warp 只有 `mesh_query_point`，
3DGS 渲染器没有静态索引，3DGRT（NVIDIA 2024）查的是**射线**与高斯的求交。所以点到
椭球的最近/半径查询**没有可引用的外部实现**，下表括号内是**对自写的单线程椭球 BVH**
（`src/cpu/gauss_bvh.cpp`）的倍数，不是对 Embree/Warp。

20,000 个随机查询点，CPU/GPU 同一批浮点数，每格取 5 轮中位数（10M 的 CPU 只跑 3 轮）：

| 分布 | 高斯数 | closest | kNN k=8 | radius | GPU 构建 |
|---|---|---|---|---|---|
| sheet | 131,072 | 2.40 ms (55.2×) | 5.42 ms (44.4×) | 6.62 ms (17.6×) | 1.41 ms |
| sheet | 1,048,576 | 8.01 ms (49.8×) | — | — | 5.81 ms |
| sheet | 10,485,760 | 46.12 ms (65.6×) | — | — | 70.2 ms |
| volume | 131,072 | 0.27 ms (219×) | 0.91 ms (169×) | 1.15 ms (44.9×) | 1.30 ms |
| volume | 1,048,576 | 0.47 ms (168×) | — | — | 5.72 ms |

- **椭球距离不是"贵到不能用"**：一次叶子测试是 14 轮 ×（3 次除法 + 约 12 次乘加），
  同规模点云是 3 次乘加，但总时间只差 1.3–3.5×（sheet 131k closest 0.68 → 2.35 ms；
  volume 1.3×）——节点测试与显存延迟把除法的延迟藏掉了。
- **sheet 分布是剪枝的坏情况**：10M 每查询 1775 次节点、348 次椭球测试
  （sheet 131k 138.3 / 19.0，volume 131k 46.4 / 3.5）。10M sheet 的 GPU 只有
  0.43 Mq/s，比 volume 慢两个数量级——倍数是两侧共同决定的，不能只看吞吐。
- **求根器必须"括号 + 牛顿/二分兜底"**：纯牛顿从 0 起步在远点线性收敛（12 轮仍差 12%），
  球近似初值在扁椭球上落在根上方、第一步冲成负数（误差 30%）。现在给一个符号有保证的
  括号 `[0, 2·a_max·|y|]`，固定 14 轮、不提前退出，对 float64 参考的最差距离相对误差
  **1.7e-7**。
- **CPU/GPU 逐位一致**：所有乘加走 `QUARRY_FMA`（CUDA `fmaf` / MSVC `std::fma`），
  nvcc 加 `--fmad=false`，除法依赖两边的 IEEE 正确舍入，禁止 `--use_fast_math`。
  9 格矩阵 `id/dist/count_mismatch` 全 0，指纹逐位相同（sheet 131k closest
  `0x2ccb47efc9f4d3c7`）。
- 增量更新：1M sheet、1% 相干位移，局部 refit **1.61 ms**（重建 5.73 ms，3.6×）；
  全量 refit 7.42 ms **比重建还慢**，因为 40 MB 上传是硬成本——和点云同一个结论。
  上一会话同一配置测到 0.93 ms，H2D 延迟随电源状态变化，配对倍数是 3.6–6.8×。
- 每个高斯 40 字节（mean 12 + 四元数 16 + 半轴 12），10M 高斯的载荷是 400 MB，
  节点内存 640 MB。这是"每叶 1 个图元"在 8 GB 卡上的实际上限。

## Python 绑定

有 Python 3.8+ 和 `pip install pybind11 numpy` 时，`build.bat` 会顺带把扩展模块建到
`build\python\`，并把 `cudart64_12.dll` 拷到 `.pyd` 旁边（Python 3.8+ 不再用 PATH 找
扩展模块的依赖）：

```python
import numpy as np, quarry

v, f = quarry.demo_mesh(grid=256, seed=1)   # 或别的 (N,3) / (M,3) 网格
scene = quarry.Scene(v, f)                  # 也可 Scene(triangles (M,3,3))
d2, tri = scene.closest(points)             # (N,) float32, (N,) int32
```

```bat
py -3.13 python/demo.py          :: 建场景 + 四种查询 + 开销拆分
py -3.13 python/test_quarry.py   :: 与 numpy 暴力法对拍（72 项检查）
```

| 入口 | 返回 |
|---|---|
| `quarry.describe()` | 显卡字符串；无 CUDA 时为空串 |
| `quarry.demo_mesh(kind, grid, random_tris, seed)` | `(vertices (N,3) f32, indices (M,3) i32)` |
| `quarry.demo_gaussians(kind, count, seed)` | `(means (N,3) f32, rot (N,4) f32 xyzw, scale (N,3) f32 半轴)` |
| `Scene(vertices, indices)` / `Scene(triangles)` | 构建 GPU BVH；`triangle_count` / `node_count` / `depth` / `bounds` / `build_stats` |
| `PointCloud(points)` | 构建点云 LBVH；`point_count` / `node_count` / `depth` / `bounds` / `build_stats` |
| `GaussianCloud(means, rot, scale)` | 构建高斯 LBVH；`gaussian_count` / `node_count` / `depth` / `bounds` / `build_stats` |
| `scene.validate()` / `cloud.validate()` / `gcloud.validate()` | 树结构自检，空串表示通过 |
| `scene.raycast(origins, dirs, tmax=inf)` | `(hit bool[N], t f32[N])`，未命中 `t=inf` |
| `scene.closest(points)` / `cloud.closest(points)` / `gcloud.closest(points)` | `(dist2 f32[N], triangle/point/gaussian i32[N])` |
| `scene.knn(points, k)` / ... | `(dist2 f32[N,k], index i32[N,k])`，按 `(dist2, 索引)` 升序，缺位 -1 |
| `scene.radius(points, r)` / ... | CSR：`(offsets i64[N+1], dist2 f32[total], index i32[total])`，段内有序 |
| `scene.update_vertices(vertices, dirty=None)` | 拓扑冻结的 refit：顶点换位、三角形数不变；`dirty` 是改动三角形索引 |
| `scene.update_triangles(triangles, dirty=None)` | 同上，直接给 `(M,3,3)`；`refit_ms` / `refit_stats` 读上一次的耗时 |
| `cloud.update_points(points, dirty=None)` | 点云 refit：点数不变；`dirty` 是改动的点索引 |
| `gcloud.update(means, rot, scale, dirty=None)` | 高斯 refit：数量不变，中心/朝向/半轴可一起换；半轴必须 > 0 |

- 入参统一按 `float32` / C 连续处理（`forcecast`），传 float64 或非连续数组会先拷贝。
  返回数组在重新拿到 GIL 之后才分配，构建与查询期间 GIL 是释放的。
- 三个类的查询形状完全一致，`closest/knn/radius` 可以互换；区别是结果里的
  `triangle` 字段装的是三角形 / 点 / 高斯索引，且只有 `Scene` 有 `raycast`。
  `GaussianCloud` 的 `closest` 在椭球内部返回 0，并列按索引最小。
- `last_query_ms` 是 CUDA event 计的内核时间，`build_ms` 是构建路径的墙钟；
  端到端还包含上传、回读和建 Python 数组。131k 场景、20k 查询点的 closest：
  墙钟中位数 1.44 ms，内核 1.29 ms，**Python 侧固定开销约 0.14 ms**。
- 进程里**第一个** `Scene` 还要付一次 CUDA 模块懒加载，约 0.5 s 墙钟，而它报告
  的 `build_ms` 只有 2 ms——计时数字对得上内核，对不上你的 `time.perf_counter()`。
- 每个查询调用内部同步一次，返回即可读；还没有异步 API。

## 增量更新（refit）

拓扑冻结：三角形数量与顺序不变，只改顶点位置。叶子盒按新三角形重写，内部盒自底向上
合并成**精确并集**，所以 refit 之后的结果仍与 CPU 参考逐位相同；代价只是盒子变松、
遍历要多测节点。

```bat
build\gpu_query_bench.exe --grid 256 --points 20000 --refit --refit-dirty 0.01 --refit-motion coherent
```

```python
scene.update_vertices(v2)              # 全量 refit
scene.update_triangles(tris2, dirty=[0, 5, 7])   # 只上传改动的三角形
scene.refit_ms, scene.refit_stats       # {upload_ms, scatter_ms, mark_ms, merge_ms, total_ms, updated}
```

局部 refit 只上传 `dirty` 里的三角形，并沿 parent 链**只合并这些叶子的祖先**
（epoch 去重、按层收集，每层合并条数由 kernel 从显存读，不需要 D2H 同步）。

`--refit-dirty 0.01` 表示每 100 个三角形改 1 个。下表是 5 个配置连续跑 3 次的
**每格中位数**（20,000 个随机查询点，同 run 内配对；绝对值在会话间会跳 1.5–3 倍，
只有配对倍数稳定）：

| 三角形 | 运动 | 局部 refit 1% | 全量 refit | 整树重建 | 局部 / 重建 | 查询 refit / 重建 |
|---|---|---|---|---|---|---|
| 8,192 | 随机 | 0.479 ms | 0.433 ms | 1.042 ms | **2.2×** | 1.23× |
| 131,072 | 随机 | 0.489 ms | 0.949 ms | 1.329 ms | **2.7×** | 1.58× |
| 131,072 | 相干 | 0.655 ms | 1.574 ms | 2.118 ms | **2.6×** | 1.09× |
| 2,097,152 | 随机 | 1.646 ms | 15.889 ms | 10.783 ms | **6.7×** | 4.32× |
| 2,097,152 | 相干 | 1.875 ms | 16.570 ms | 12.053 ms | **6.2×** | 1.38× |

- **全量 refit 是上传受限的**：从 CPU 侧场景出发每次要重传整棵三角形数组（2M 是 72 MB，
  H2D 实测 10–22 ms，随电源状态跳），所以 2M 上它比整树重建还贵（0.69–0.73×）。
  它只在小场景划算；真正有用的增量路径是 `dirty` 子集。
- **局部 refit 的耗时随改动量而非场景大小走**：2M 上 1% 改动 1.6–1.9 ms，其中
  主机侧 gather（按场景索引从 72 MB 数组里挑出改动三角形）+ H2D 约 0.9 ms、
  标记祖先 0.17 ms、合并 0.5–0.6 ms。同 run 对照，全量路径的整树合并是 2.0–2.1 ms
  ——祖先合并把树阶段压到 1/4 左右。小场景（8k/131k）里逐层 launch 与主机间隙
  开始主导，0.3–0.7 ms 的绝对时间基本是固定开销，报告时按配对倍数读。
- **refit 后查询会变慢，程度取决于运动是否相干**：随机抖动让盒子迅速变松（平均节点
  测试 131k 139 vs 85、2M 323 vs 99，耗时 1.6× / 3.5–4.4×），整体平移损失小得多
  （131k 1.09×、2M 1.38×）。这是 refit 的固有代价，和构建质量一起读才有意义。
- 正确性门禁：每个规模、两种运动下，refit 树的内部盒逐位等于孩子并集、根盒等于
  场景包围盒；closest / kNN(k=8) / radius(r=2.0) 的结果与 CPU 参考逐位一致
  （`count/tri/dist2_mismatch` 全 0），连续两次局部 refit 共享叶子/祖先时同样成立。
- 语义限制：**不增删三角形、不重排**。改了面数请重建；`update_vertices` 只对
  `Scene(vertices, indices)` 构造的场景有效。

## 我们做什么 / 不做什么

做：

- 加速结构的**构建**与**查询**都在 GPU 上
- 三角形、点云、高斯三种图元用同一套查询接口（均已实现：`TriOps` / `PointOps` / `GaussianOps`）
- 每瓦吞吐与端到端延迟作为一等指标
- 可复现基准：同 seed 同输入，逐条对拍

不做（v1.0）：

- 不做渲染器，不输出图像
- 不做光线追踪的着色/材质，只回答"打到了什么、在哪"
- 不做分布式 / 多卡
- 不追求在极小场景（< 10 万图元）的绝对倍数

## 目录

```text
src/core/     几何、场景 / 点云 / 高斯生成、可复现输入（host/device 共用）
src/cpu/      暴力法真值 + 朴素 BVH 参照 + 点云 kd-tree 基线 + 椭球 BVH 基线
src/gpu/      CUDA：并行 LBVH（lbvh.cu，按 Ops 服务三角形 / 点 / 高斯）、光线遍历、点查询
src/cli/      quarry 命令行
bindings/     pybind11 绑定（quarry_py.cpp）
bench/        Embree 对照、Warp / scipy 对照与 gpu_*_bench（同一份输入，二进制转储互通）
python/       Python 侧的用法示例与 numpy 对拍
tests/        自测入口
learn/        教程配套的单文件练习
```

## 已知限制

- CPU 参考实现只有中位数切分、只有单线程
- GPU 侧已覆盖射线求交与 closest / kNN / radius 三条点查询；kNN 的 k 上限 32，
  radius 的结果条数不设上限（长段退回 CUB 分段排序）
- GPU 射线排序已实现，但只对相干主射线有效；非相干射线需要两遍遍历（先取命中点
  再排序），未做。内联子节点包围盒（64B 布局）实测为负结果，已回退
- 场景类型只有地形网格和随机三角形，缺非均匀模型（Dragon/Lucy）与多层遮挡
- closest 已和 Embree 32 线程（3.0–16.3×）与 Warp 1.6.2（0.72–1.15×）对拍；
  kNN / 半径在两家外部库里都没有原生入口，只有自家 CPU 参考
- 点云查询没有 Embree 对照物（`rtcPointQuery` 只吃三角形）；基线是自写的单线程
  kd-tree，另有 scipy `cKDTree` 作第二把尺子。scipy 在 2D 流形数据上会退化
  （10M 慢 72×），所以 headline 倍数只用自家 kd
- 高斯查询**没有外部基线**：Embree/Warp 没有椭球图元，3DGRT 查的是射线与高斯的求交。
  倍数是相对自写的单线程椭球 BVH；语义是椭球实体，Mahalanobis 密度与屏幕投影未做
- 高斯每个 40 字节，10M 是 400 MB 载荷 + 640 MB 节点；同规模点云约 3.3 倍上传量
- 椭球距离固定 14 轮求根，最差距离相对误差 1.7e-7；没有做自适应提前退出，
  因为 CPU/GPU 逐位一致优先于最后一位精度
- 点云仍是每叶 1 个图元：10M 点 → 2000 万节点、约 640 MB 节点内存，这是 8 GB 卡上
  的实际上限；叶子批量在三角形上是负结果（射线 2.2× 慢），点云没有重试
- Python 绑定是同步的：每次调用都上传、同步、回读，没有异步 / 零拷贝接口
- refit 拓扑冻结：不增删、不重排三角形；局部路径只合并脏叶子的祖先，但逐层 launch
  的数量不变，小场景里启动开销开始主导
- BVH 每叶 1 个三角形，2M 场景 419 万节点；Morton 叶子批量（4–8 个/叶）实测
  遍历更慢（射线 2.2×、kNN 1.8×），已回退。Warp 的叶子 4 个配的是 SAH 建树，
  要追这 1.2–1.4× 需要 GPU 自顶向下 SAH 或 BVH4 宽节点
