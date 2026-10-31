# quarry

[![CI](https://github.com/ycl-berserk/quarry-gpu/actions/workflows/build.yml/badge.svg)](https://github.com/ycl-berserk/quarry-gpu/actions/workflows/build.yml)

GPU 原生空间查询引擎：在 GPU 上并行构建层次包围盒（LBVH，可选分箱 SAH），支持射线求交、
closest、kNN 与半径查询，图元覆盖三角形、点云、高斯，并支持增量更新（refit 与增删改拓扑）。

许可证 Apache-2.0（见 `LICENSE`）。CPU 参考实现已完整；GPU 侧（CUDA 12.4，sm_89）已有
并行 LBVH 构建、光线遍历、closest / kNN / radius 三条点查询路径，拓扑冻结的增量更新
（refit），以及**三种图元**：三角形、点云、高斯。三者共用同一棵 LBVH 与同一套查询，
只在叶子里换"测什么"——高斯的叶子测试是精确的**点到椭球实体**距离。查询批次可以按
Morton 序重排（`--query-sort` / Python 的 `order="morton"`），二叉节点可以折叠成
4 宽节点（`--wide`），建树也可以从并行的 Karras LBVH 换成自顶向下的分箱 SAH
（`--sah` / Python 的 `builder="sah"`）——三者默认关闭，可以互相叠加。
没有 CUDA 工具链时自动降级为纯 CPU 构建，CI 与本文档的 CPU 数字不受影响。

## 安装

两条路径，都从源码构建；发布包尚未上传 PyPI。

### Python

需要 CUDA 12.x（`nvcc` 在 PATH 里，或下面用 `QUARRY_CUDA_COMPILER` 指定）、Python ≥ 3.9、
本机 C++ 编译器。Windows 上请在 **x64 Native Tools Command Prompt for VS 2022** 里执行
（或先 `call vcvars64.bat`），否则 Ninja 找不到 `cl.exe`。

```bat
py -3.13 -m pip install .
py -3.13 -c "import quarry; print(quarry.describe())"
```

`pip` 会从源码编译扩展，并把 `cudart64_*.dll` 放在扩展旁边，第一次约 1–3 分钟。
换架构（例如 sm_86）或显式指定 nvcc：

```bat
py -3.13 -m pip install . --config-settings=cmake.define.QUARRY_CUDA_ARCH=86
py -3.13 -m pip install . --config-settings=cmake.define.QUARRY_CUDA_COMPILER=D:/CUDA/bin/nvcc.exe
```

装完直接看 `python/demo.py` 的用法；不再需要 `sys.path.insert`。

### C++ / CMake

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix C:\quarry
```

消费端：

```cmake
find_package(quarry 1.0 CONFIG REQUIRED)
target_link_libraries(app PRIVATE quarry::gpu)   # 只用 CPU 参考实现就换成 quarry::core
```

```cpp
#include <gpu/gpu_bvh.h>
```

没有 CUDA 时 `quarry::gpu` 不存在，`quarry::core` 仍然可用；`-DQUARRY_BENCH=OFF`
可以跳过 CLI、基准与测试，只编译库。

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
build\gpu_query_bench.exe --grid 1024 --points 100000 --query-sort           :: 查询点按 Morton 序重排后再遍历
build\gpu_bench.exe --grid 1024 --rays 1000000 --runs 5 --wide              :: 4 宽节点折叠后再遍历（--ray-sort 叠加）
build\gpu_query_bench.exe --grid 1024 --points 100000 --wide --query-sort   :: 宽节点 + 查询重排（可叠加）
build\gpu_query_bench.exe --grid 1024 --points 20000 --runs 5 --sah --stats :: 自顶向下分箱 SAH 建树（默认 LBVH），查询逐点对拍 CPU

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
- **Warp 1.6.2 是更强的外部基线**：默认 LBVH 下地形 131k 慢 1.4×、2M 慢 1.2×，8k
  持平；随机三角形汤我们快 1.15×。Warp 是 SAH 建树 + 每叶 4 个图元，我们是 Karras
  LBVH + 每叶 1 个（2M → 419 万节点、深度 31）。把 Morton 相邻的 4 个三角形并成
  叶子实测**更慢**（射线 2.2×、kNN 1.8×，构建反而快 1.5×），所以差距在树质量。
  开 `--sah` 后查询侧就追平了：8k **1.39×**、131k **0.97×**、2M **1.03×**（构建
  仍比 Warp 慢 2.5–5.6×，见下面「SAH 建树」一节）。
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

## 查询调度（M3 前半，2026-10-29）

`--query-sort` 把一批查询点按 Morton 序重排后交给**同一个内核**，结果散回原下标。
三条查询的内核一行没改；排序（30 位/点、CUB `SortPairs`）、gather、scatter 全部计入
返回时间，`sort_ms` 单独报一行。

**基线口径**：这是纵向对照——同一实现、同一场景、同一批查询点、同一会话成对跑，
关/开 `--query-sort` 各 5 轮取中位数。点查询没有可引用的外部 kNN/半径实现
（Embree `rtcPointQuery` 只对三角网格找最近点、Warp 只有 `mesh_query_point`），
横向绝对吞吐仍以 `docs/baseline.md` 里的 Warp 1.6.2 / Embree 数字为准，本章未改内核，
横向关系不变。

地形 2M 三角形（419 万节点 = 134 MB，5× L2），3 s 预热后成对测量：

| 查询 | 输入序 | Morton 序 | 倍数 | 排序开销 |
|---|---|---|---|---|
| closest，20k | 5.226 ms | 5.232 ms | 1.00× | 0.27 ms |
| closest，100k | 19.21 ms | **11.53 ms** | **1.67×** | 0.18 ms |
| closest，200k | 36.14 ms | **19.16 ms** | **1.89×** | 0.42 ms |
| kNN k=8，100k | 30.35 ms | **20.74 ms** | **1.46×** | 0.31 ms |
| radius 2.0，100k | 219.8 ms | 223.7 ms | 0.98× | 0.47 ms |
| 随机汤 2M，closest 100k | 27.62 ms | **15.38 ms** | **1.80×** | 0.22 ms |
| 地形 131k，closest 20k | 0.884 ms | 1.11 ms | 0.80× | 0.25 ms |

- **收益来自带宽，不是少做工作**：两种顺序的节点/图元测试数完全相同
  （closest 474.8 / 123.0，kNN 577.3 / 171.8，radius 1182.9 / 545.1），
  Morton 序只把相邻 warp 的随机 DRAM 读换成共享的上层节点。
- **要 10 万查询起步**：20k 查询只占 40% 占用率，内核是延迟受限的；相干性合并了
  miss，同时在飞的 miss 也变少，收益被抵消，再扣排序钱就是净亏（131k 上 0.80×）。
  这是**未验证的假设**，写在这里是为了标出推论边界，不当结论引用。
- **半径查询不受益**：它遍历两遍，第二遍还要做变长 CSR 写入与段内排序，遍历只占
  一小半，重排改不动大头。
- **高斯**是另一种情况（`gpu_gauss_bench --query-sort`，只有 3 轮热身、±10% 噪声）：
  10M 上 closest 20k 从 40.7 慢到 47.0 ms（0.86×），100k 从 149.3 到 136.9 ms
  （1.09×），kNN 100k 从 252.2 到 220.8 ms（1.14×）。每查询 1775 次节点测试 +
  348 次 14 轮牛顿的椭球距离，存储已不是主成本。
- **逐位不变**：重排不改变任何一条查询，地形 2M 10 万查询两种顺序的指纹都是
  `closest 0x0ecd1029bdb758b4`、`kNN8 0x169e1b986f7ca4b3`；radius 131k 两万查询
  都是 `0xba0f794cf680d9a3`，且 `count/tri/dist2_mismatch` 全 0。Python 新增 5 项
  逐位检查（`Scene.closest/knn/radius`、`PointCloud.closest`、`GaussianCloud.closest`
  传 `order="morton"`），整套 **77 项**全过。
- 库的默认值保持 `QueryOrder::Input`：阈值不是普适常数，代码不替用户猜。
  教程见 `Quarry-学习路径/10-查询调度.md`。

```python
hit, t = scene.closest(points, order="morton")   # 默认 order="input"
```

## 宽节点（M3 续，2026-10-29）

`--wide` 在建树后把相邻两层二叉节点折叠成 **4 宽节点**（128 B = 4 × 32 B 槽）：
宽节点根是偶数深度内部节点，槽位装孩子/孙子的盒子或叶子下标，内部孙节点装
`wideId`（CUB 前缀和生成）。树拓扑、叶子顺序、图元划分完全不动，只是把
"每个节点读 32 B、几乎每次一个新事务"变成"一次读 128 B、四路比较在寄存器里"。
折叠成本 131k 0.17 ms、2M 1.93 ms；宽节点显存与二叉基本持平（2M 上各约 134 MB）。

**基线口径**：纵向对照，同一实现、同一会话、同一批输入，关/开 `--wide` 成对跑，
5 轮中位数，3 s 预热。宽 BVH 本身是 Embree / tinybvh 的标准结构（Aila & Laine 2009
一脉），这里是**结构对齐**，单射线/单查询、Morton 划分，未配 SIMD packet，
绝对吞吐仍按 `docs/baseline.md` 的口径。宽节点与 `--sah` 可叠加（2M closest
二叉 SAH 6.09 → 宽 4.71 ms）。

| 场景 | 工作负载 | 二叉 | 4 宽 | 倍数 |
|---|---|---|---|---|
| 地形 2M | 射线 1M，输入序 | 16.00 ms | **9.31 ms** | **1.72×** |
| 地形 2M | 射线 1M，Morton 序 | 4.27 ms | 4.23 ms | 1.01× |
| 地形 131k | 射线 1M，输入序 | 6.96 ms | **5.21 ms** | **1.29×** |
| 地形 2M | closest，100k | 27.55 ms | **16.33 ms** | **1.69×** |
| 地形 2M | closest + 排序，100k | 16.57 ms | **14.80 ms** | 1.12×（端到端 1.86×） |
| 地形 2M | kNN k=8，100k | 36.49 ms | **25.72 ms** | **1.42×** |
| 点云 1M | closest / kNN8 / radius，50k | 9.34 / 19.59 / 69.46 ms | 6.75 / 14.62 / 68.87 ms | 1.38× / 1.34× / 1.01× |
| 高斯 1M | closest / kNN8 / radius，20k | 18.64 / 36.26 / 114.29 ms | 14.56 / 27.94 / 92.48 ms | 1.28× / 1.30× / 1.24× |

- **省的是取数，不是比较**：2M 射线节点测试 47.2 → 24.4（宽路径按"成功盒子测试"
  计，同单位），图元测试不变。
- **与排序省的不是同一笔钱**：射线排序后二叉已 4.27 ms，宽 4.23 ms（≈1.0×）；
  点查询宽 + 排序叠加到 **1.86×**，是目前 2M closest 的最好数字。
- **点云 radius 持平**（遍历不是瓶颈）；**高斯 radius 1.24×**（椭球距离计算重，
  少读节点仍划算）。随机方向次生射线遍历太浅，无收益。
- **逐位不变**：2M 射线逐条对拍零失配；三角形/点云/高斯三条查询指纹与二叉路径
  完全相同，与 CPU 参考 `id/dist/count_mismatch=0`。
- **支持 `--wide --refit`**：refit 改的是二叉盒子，宽节点是拷贝，所以两种 refit
  结束时都会重发宽节点（拓扑与 `wideId` 不变）。局部 refit 只重发脏祖先所在的
  宽节点，小树或大部分叶子都脏时改为整层重发。2M 上局部 refit 1.26 → 1.78 ms
  （重发 0.40 ms）、全量 15.44 → 17.47 ms（重发 1.37 ms）；refit 后的查询仍是
  3.41 → 2.39 ms、3.83 → 2.53 ms（1.4–1.5×）。100 万点云/高斯局部 refit 多
  0.35 ms，查询 1.28–1.29×。
- 库默认 `wide=false`：收益依赖树规模与访问模式，代码不替用户猜。教程见
  `Quarry-学习路径/11-宽节点.md`。

## SAH 建树（M3，2026-10-30）

`--sah` 把并行的 Karras LBVH 换成**自顶向下分箱 SAH**（Wald 2007，16 箱 × 3 轴）。
节点编号规则、叶子顺序、查询/refit/宽节点全部不动，所以两棵树对同一条查询的结果
**逐位相同**——换的只是树的形状。地形上的收益来自树深和节点测试次数：

| 场景 | 查询 | LBVH | SAH | 倍数 | 节点测试/查询 | 树深 |
|---|---|---|---|---|---|---|
| 地形 8k | closest | 0.557 ms | **0.397 ms** | **1.40×** | 74.9 → 48.8 | 19 → 14 |
| 地形 131k | closest | 1.428 | **1.121** | **1.27×** | 167.4 → 117.6 | 25 → 19 |
| 地形 2M | closest | 7.752 | **5.757** | **1.35×** | 477.3 → 362.6 | 31 → 23 |
| 地形 2M | kNN k=8 | 9.799 | **6.779** | **1.45×** | 580.1 → 396.9 | 31 → 23 |
| 随机 1M 三角形 | closest | **3.777** | 4.158 | 0.91× | 358.8 → 410.0 | 25 → 23 |
| 点云 1M | closest / kNN8 | 4.26 / 8.85 | 4.50 / **7.45** | 0.95× / **1.19×** | — | 32 → 22 |
| 高斯 1M | closest | 18.06 | **12.88** | **1.40×** | 356.4 → 279.1 | 33 → 22 |

20k 查询点，成对测量、5 轮中位数、3 s 预热（`--no-cpu`）。

- **查询侧追平 Warp**：8k **1.39×**、131k 0.97×、2M **1.03×**（同一次会话里先跑
  quarry SAH 再跑 Warp 1.6.2）。默认 LBVH 时这三格是 0.95× / 0.72× / 0.83×。
- **构建是短板**：地形 8k 2.87 ms、131k 6.71 ms、2M 62.2 ms，比 LBVH 慢
  2.5–3.1×、比 Warp 慢 2.5–5.6×——它的叶子装 4 个图元、节点数少一半。10-30 晚
  把选切分的 O(箱²) 扫描换成后缀盒、分区复用左右标记，2M 构建从 74.5 压到
  62.2 ms，树与查询指纹逐位不变；分箱仍占 52%（32.2 ms），是下一个目标。
- **随机三角形汤是负收益（0.91×）**：均匀随机数据上 Morton 序已近似中位切分，
  16 箱量化 + 1/8 平衡守卫反而抬高节点测试。32 箱实验无收益且构建更慢，已回退。
- **可以叠加**：2M closest 二叉 SAH 6.09 → `--wide` 4.71 ms（1.29×）；
  `--sah --refit` 局部 1.03 ms、全量 17.1–19.2 ms，结果与 CPU 参考零失配。
- 库默认仍是 LBVH：SAH 对小场景/均匀数据的收益不足以抵消构建成本，代码不替
  用户猜。教程见 `Quarry-学习路径/12-SAH建树.md`。

```python
scene = quarry.Scene(v, f, builder="sah")   # 默认 builder="lbvh"
```

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
py -3.13 python/test_quarry.py   :: 与 numpy 暴力法对拍（84 项检查）
```

| 入口 | 返回 |
|---|---|
| `quarry.describe()` | 显卡字符串；无 CUDA 时为空串 |
| `quarry.demo_mesh(kind, grid, random_tris, seed)` | `(vertices (N,3) f32, indices (M,3) i32)` |
| `quarry.demo_gaussians(kind, count, seed)` | `(means (N,3) f32, rot (N,4) f32 xyzw, scale (N,3) f32 半轴)` |
| `Scene(vertices, indices, builder="lbvh")` / `Scene(triangles, builder)` | 构建 GPU BVH；`builder` 可为 `"lbvh"`（默认）或 `"sah"`；`triangle_count` / `node_count` / `depth` / `bounds` / `build_stats` |
| `PointCloud(points, builder="lbvh")` | 构建点云 BVH；`point_count` / `node_count` / `depth` / `bounds` / `build_stats` |
| `GaussianCloud(means, rot, scale, builder="lbvh")` | 构建高斯 BVH；`gaussian_count` / `node_count` / `depth` / `bounds` / `build_stats` |
| `scene.validate()` / `cloud.validate()` / `gcloud.validate()` | 树结构自检，空串表示通过 |
| `scene.raycast(origins, dirs, tmax=inf)` | `(hit bool[N], t f32[N])`，未命中 `t=inf` |
| `scene.closest(points)` / `cloud.closest(points)` / `gcloud.closest(points)` | `(dist2 f32[N], triangle/point/gaussian i32[N])` |
| `scene.knn(points, k)` / ... | `(dist2 f32[N,k], index i32[N,k])`，按 `(dist2, 索引)` 升序，缺位 -1 |
| `scene.radius(points, r)` / ... | CSR：`(offsets i64[N+1], dist2 f32[total], index i32[total])`，段内有序 |
| `scene.update_vertices(vertices, dirty=None)` | 拓扑冻结的 refit：顶点换位、三角形数不变；`dirty` 是改动三角形索引 |
| `scene.update_triangles(triangles, dirty=None)` | 同上，直接给 `(M,3,3)`；`refit_ms` / `refit_stats` 读上一次的耗时 |
| `cloud.update_points(points, dirty=None)` | 点云 refit：点数不变；`dirty` 是改动的点索引 |
| `gcloud.update(means, rot, scale, dirty=None)` | 高斯 refit：数量不变，中心/朝向/半轴可一起换；半轴必须 > 0 |
| `scene.update_topology(triangles, deleted=None, moved=None, inserted=None)` / `cloud.update_topology(points)` / `gcloud.update_topology(means, rot, scale, ...)` | 增删图元：数组 = 幸存者（原序去掉 `deleted`）+ `inserted` 指定位插入 + 尾部追加；`moved` 是新场景里几何变了的下标；返回 timings 字典 |

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
  `Scene(vertices, indices)` 构造的场景有效。数量变化走下面的拓扑更新。

## 拓扑更新（增删，M3 收尾，2026-10-30）

`updateBVHGPU` 接受一份编辑脚本：删掉 `deleted` 里的旧图元、几何改了的 `moved`、
在场景中间插入 `inserted` 个新图元、在尾部追加 `appended` 个新图元。它只重建被
这些编辑碰到的子树，其余原样复用，再做一次规范化重编号；查询结果与"对同一份
新场景全新构建"**逐位相同**。

```cpp
// newScene = 旧场景删掉 deleted 后的幸存者（保持原顺序）
//          + 在 inserted 这些位置插入的新图元 + 末尾 appended 个新图元
GpuTopoEdit edit;
edit.deleted  = {12, 40, 41};   // 旧场景下标，升序、不重复
edit.inserted = {6, 7};         // 新场景下标，升序、不重复、必须 < newScene.size() - appended
edit.appended = 2;              // 新图元已经接在 newScene 尾部
edit.moved    = {3, 9};         // 新场景下标：几何变了、拓扑没变
GpuTopoUpdateTimings ts;
double ms = updateBVHGPU(bvh, newScene, edit, &ts);
// ts: planMs / surgeryMs / relayoutMs / mergeMs / wideMs，以及
//     surgeryRoots / rebuiltLeaves / newDepth / fullRebuild / inserted
```

```python
scene.update_topology(tris2, deleted=None, moved=None, inserted=None)
cloud.update_topology(pts2);  gcloud.update_topology(means2, rot2, scale2)
```

只 move（`deleted` 与 `inserted` 为空、`appended == 0`）会直接转给原来的 refit
快路径；`update_vertices` / `update_points` / `update` 仍然拒绝数量变化。

**槽序不跟随场景序**。新图元不论插在场景哪里，都进槽空间的尾部：
`[0, 幸存者数)` 是幸存者按旧槽序压缩后的下标，接着是 `inserted`（按场景序），最后是
`appended`。于是新增块在槽序里始终是一段连续区间，子树覆盖的槽区间仍然连续，
切分与重建不用为中间插入动任何旧槽。场景下标由 `d_primIds` 单独记录——它是唯一
真源，`d_tris` 只是按槽序排好的副本；查询内核只读 `slot = -left-1`，不关心 `n`
与节点 id 布局，所以结果不受影响。幸存者的新场景下标用一张位移表
`shifts[j] = inserted[j] - j`（非降）一次二分求出，插入项越多越省。

地形 131k / 2M，每轮改 1%（cluster：局部簇内；scatter：全场景均匀），
20k 查询点，同会话成对、5 轮中位数（`--no-cpu`）。两列时间分开记：
`update` 是 `updateBVHGPU` 内部的多阶段 CUDA event 计时，`wall` 是调用前后的
steady_clock 端到端；`重建 wall` 是同一份场景 `destroy + build` 的端到端
（含 72 MB 场景上传与全部 cudaMalloc），这也是调用方真正会付的钱。

| 规模 | 建树器 | 编辑 | update | wall | 重建 wall | wall 倍数 | 回退 | 树深 |
|---|---|---|---|---|---|---|---|---|
| 131k | LBVH | cluster 混合 | 2.718 ms | 2.877 ms | 3.521 ms | 0.82× | 0/5 | 25 |
| 131k | LBVH | cluster 只移动 | 0.386 | 0.501 | 3.962 | **0.13×** | 0/5 | 25 |
| 131k | LBVH | cluster 只增 | 4.514 | 4.714 | 4.357 | 1.08× | 0/5 | 25 |
| 131k | LBVH | scatter 混合 | 1.730 | 2.559 | 3.483 | 0.73× | 5/5 | 26 |
| 131k | SAH | cluster 混合 | 4.544 | 4.702 | 17.802 | 0.26× | 0/5 | 24 |
| 131k | SAH | scatter 混合 | 9.488 | 11.786 | 13.404 | 0.88× | 5/5 | 20 |
| 2M | LBVH | cluster 混合 | 11.620 | 11.833 | 44.494 | **0.27×** | 0/5 | 31 |
| 2M | LBVH | cluster + 宽节点 | 14.693 | 14.790 | 54.879 | 0.27× | 0/5 | 31 |
| 2M | SAH | cluster 混合 | 11.612 | 11.807 | 98.092 | **0.12×** | 0/5 | 28 |
| 2M | LBVH | scatter 混合 | 21.398 | 24.245 | 44.023 | 0.55× | 5/5 | 32 |
| 2M | LBVH | scatter + 宽节点 | 23.551 | 26.502 | 55.104 | 0.48× | 5/5 | 32 |
| 2M | SAH | scatter 混合 | 68.206 | 70.461 | 102.426 | 0.69× | 5/5 | 25 |

读法：

- **按端到端算，1/5 的目标在 2M 上过了**（0.12–0.69×）。上一版写的"1.5×/2.6×"
  是内部 event 时间之比，而重建那一侧的内部计时不含 72 MB 上传与 cudaMalloc
  （2M 上这两项合计约 24 ms，比整个增量更新还长），口径本身偏向了重建一侧。
  现在两边的墙钟口径都写在表里，读者可以自己选。
- **v1.1 改了什么**：更新期间所有临时块、以及换树后释放的旧树数组，都进一个
  按 BVH 计的线程局部设备内存池（best-fit 复用、2 GiB 上限、跨轮保留），
  下一轮直接取回；宽节点的 `wideId` / `wideNodes` 按容量增长、跨轮复用；
  连建树的持久数组（节点、图元、parent/depth、refit 缓存）也改成池优先。
  2M 稳态下每轮不再有 cudaMalloc/cudaFree，宽节点那 ~19 ms 的分配器成本直接
  消失（宽层重发本身只有 2.4 ms，整层重发照旧）。
- **回退不再是惩罚**：scatter 1% 仍然回退整树重建，但旧树整棵还进池、重建直接
  用这批同尺寸的块，2M wall 21–24 ms，反而比 `destroy + build`（44 ms）快一倍。
  均匀散布的编辑不必再绕开 `update_topology`。
- **只 move 还是 refit 快路径**：131k 0.50 ms（0.13×），2M 是上传受限的
  1.6 ms 量级，和重建不是一笔账。增删才是本章的成本。
- **剩下的 O(n) 两段**：2M cluster 的 11.6 ms 拆开是 plan 1.7–2.9 / surgery
  5.3–7.3 / relayout 3.6–5.0 / merge 0.6–0.75。plan（两趟前缀和 + 切分选择）
  与 relayout（规范化重编号、整片重发叶子）都是 O(n)，合计约一半；再往下压
  要稳定槽位或稀疏叶子重发。首轮是冷的（2M 37 ms，池还空着），之后才进稳态。
- **编辑种类**：只增最贵——追加区间要吸收同样多的旧叶子，重建块更大；131k 1%
  只增 4.7 ms 略高于直接重建（1.08×），只删和混合都在重建的一半以下。
- **任意位置插入**（`inserted`，v1.2）：中插与尾插成本相同（2M 23.1 vs 24.0 ms），
  因为槽序里新增块一律接在尾部、场景序由 `d_primIds` 单独记录。同会话 5 轮中位数：
  2M LBVH 中插 22.8 / wall 23.1 / 重建 wall 72.0 ms（0.32×），SAH 中插 24.9 / 25.7 /
  130.9 ms（0.20×，1 次回退）；131k 中插 6.2 / 6.4 ms（0.95×）。四合一（删 + 中插 +
  尾插 + 移动）2M LBVH 22.5 / 23.4 ms（0.38×）。这组数字与上表不在同一会话，
  只看行内倍数。
- **插入的查询质量会缓慢漂**：2M 连续 5 轮 1% 中插，更新树节点测试 513→759，同轮
  全新构建 456→439，即 1.1×→1.7×；SAH 触发一次深度回退后回到 1.17×。尾插同形
  （513→742），与插入功能无关，是"旧子树不重切"的固有代价。
- **深度会漂，但有上限**：131k 5% 连续 100 轮，深度 25→26、回退 6/100，节点
  测试在 90–163 之间来回；2M 1% 连续 30 轮回退 1/30、深度 31→36（上限是均衡
  深度 + 6 层）。
- **宽节点**：池化之后 2M cluster 11.6 → 14.7 ms（宽层重发 2.4 ms + 更散的
  合并/更深的遍历），从"不建议开"变成"平时开着也行"。
- **正确性门禁**（每轮都跑）：`validateBVHGPU(newScene)` 空串；closest /
  kNN(k=1,8,16) / radius 全量输出、射线 `t` 位模式与"同一建树器对 `newScene`
  全新构建"的树逐位相同。上表 12 个配置、5 轮共 60 次更新全部
  `mismatch=0/0/0`；同一输入两次更新的节点数组 FNV 哈希相同（`--topo` 行里的
  `tree=`）。
- **显存**：池上限 2 GiB，2M LBVH 稳态约 230 MB 驻留（旧树 + 建树临时块），
  宽节点另占 128 + 16 MB 且跨轮复用；峰值仍是旧树 + 新树 + 暂存同时在世。
  `destroyBVHGPU` 在池不活跃时把所有池块一并 `cudaFree`，回退重建期间则原样
  留着给这一轮的建树用。

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
- 拓扑更新支持**任意位置插入**（新图元落在槽尾，场景序由 `d_primIds` 记录），但
  不做稳定槽位：每次更新仍是 O(n) 的规范化重编号，plan/relayout 两段占掉一半。
  池化之后 2M 上 1% 编辑的端到端耗时是直接重建的 0.12–0.69×（LBVH 0.27×、
  SAH 0.12×），中插 0.20–0.32×；均匀散布的编辑走回退也比重建快（0.48–0.69×）；
  131k 场景因为整树重建只要 4 ms，增量基本持平（0.73–1.10×）。宽节点层每轮
  整层重发，2M 额外约 3 ms；连续多轮插入后查询节点测试会缓慢上升（5 轮约
  1.7× 同场景新鲜树），靠深度上限触发整树重建兜底，"升一级切分"未做
- BVH 每叶 1 个三角形，2M 场景 419 万节点；Morton 叶子批量（4–8 个/叶）实测
  遍历更慢（射线 2.2×、kNN 1.8×），已回退。BVH4 宽节点（`--wide`）与自顶向下
  分箱 SAH（`--sah`）都已落地，查询侧已追平 Warp，剩余差距在构建速度；
  每叶 4 个图元（Warp 的形态）未做
- 宽节点 + refit 已接：局部路径重发脏祖先所在的宽节点，小树/大改动走整层重发；
  代价是每次 refit 多 0.13–2.0 ms（2M 全量约 1.4 ms）。射线已 Morton 排序时
  宽节点无收益（≈1.0×），非相干次生射线遍历太浅，同样无收益
- 宽节点是单射线/单查询路径，没有 SIMD packet；BVH8 未试
