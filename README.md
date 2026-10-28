# quarry

[![CI](https://github.com/ycl-berserk/quarry-gpu/actions/workflows/build.yml/badge.svg)](https://github.com/ycl-berserk/quarry-gpu/actions/workflows/build.yml)

GPU 原生空间查询引擎：构建加速结构（BVH / 八叉树 / 哈希网格），支持射线求交、kNN、半径查询、AABB 查询，图元覆盖三角形、点云、高斯，并支持增量更新。

许可证 Apache-2.0（见 `LICENSE`）。CPU 参考实现已完整；GPU 侧（CUDA 12.4，sm_89）已有
并行 LBVH 构建、光线遍历，以及 closest / kNN / radius 三条点查询路径。没有 CUDA 工具链时
自动降级为纯 CPU 构建，CI 与本文档的 CPU 数字不受影响。

## 现在能跑什么

```bat
build.bat                                  :: CMake + Ninja（用 VS 2022 自带的，无需另装）
build\quarry.exe selftest                  :: 正确性自测（29 项）
build\quarry.exe run --scene terrain --grid 256 --rays 1000000 --engine bvh
build\quarry.exe query --type closest --grid 256 --points 20000 --stats
build\quarry.exe query --type knn --k 16 --grid 256 --points 20000
build\quarry.exe query --type radius --radius 2.0 --grid 256 --points 20000

build\gpu_bench.exe --grid 256 --rays 1000000 --runs 5 --stats   :: GPU 建树 + 遍历，逐条对拍 CPU
build\gpu_query_bench.exe --grid 256 --points 20000 --runs 5 --stats          :: GPU 最近点，逐点对拍 CPU
build\gpu_query_bench.exe --grid 256 --points 20000 --query knn --k 16        :: GPU kNN
build\gpu_query_bench.exe --grid 256 --points 20000 --query radius --radius 2.0  :: GPU 半径查询
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

## 我们做什么 / 不做什么

做：

- 加速结构的**构建**与**查询**都在 GPU 上
- 三角形、点云、高斯三种图元用同一套查询接口
- 每瓦吞吐与端到端延迟作为一等指标
- 可复现基准：同 seed 同输入，逐条对拍

不做（v1.0）：

- 不做渲染器，不输出图像
- 不做光线追踪的着色/材质，只回答"打到了什么、在哪"
- 不做分布式 / 多卡
- 不追求在极小场景（< 10 万图元）的绝对倍数

## 目录

```text
src/core/     几何、场景生成、射线的可复现输入（host/device 共用）
src/cpu/      暴力法真值 + 朴素 BVH 参照
src/gpu/      CUDA：并行 LBVH 构建（lbvh.cu）、光线遍历（raycast.cu）、点查询（query.cu）
src/cli/      quarry 命令行
bench/        Embree 对照、Warp 对照与 gpu_*_bench（同一份输入，二进制转储互通）
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
- BVH 每叶 1 个三角形，2M 场景 419 万节点；Morton 叶子批量（4–8 个/叶）实测
  遍历更慢（射线 2.2×、kNN 1.8×），已回退。Warp 的叶子 4 个配的是 SAH 建树，
  要追这 1.2–1.4× 需要 GPU 自顶向下 SAH 或 BVH4 宽节点
