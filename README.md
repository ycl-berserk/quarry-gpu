# quarry

[![CI](https://github.com/ycl-berserk/quarry-gpu/actions/workflows/build.yml/badge.svg)](https://github.com/ycl-berserk/quarry-gpu/actions/workflows/build.yml)

GPU 原生空间查询引擎：构建加速结构（BVH / 八叉树 / 哈希网格），支持射线求交、kNN、半径查询、AABB 查询，图元覆盖三角形、点云、高斯，并支持增量更新。

许可证 Apache-2.0（见 `LICENSE`）。CPU 参考实现已完整；GPU 侧（CUDA 12.4，sm_89）已有
并行 LBVH 构建与光线遍历，最近点 / kNN / 半径的 GPU 版本在推进中。没有 CUDA 工具链时
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

RTX 4060 Laptop（24 SM，50 W 功耗墙），同一场景、同一射线、同一 seed。`排序端到端`
= 建 Morton 码 + 基数排序 + 重排 + 遍历 + 结果散回，全部计入 GPU 时间：

| 三角形 | GPU 遍历（输入序） | GPU 遍历 + 射线排序 | GPU 构建 | Embree 1T | Embree 32T packet | 排序后 / 32T |
|---|---|---|---|---|---|---|
| 8,192 | 101.41 Mrays/s | 193.57 Mrays/s | 1.61 ms | 8.65 | 60.98 | **3.17×** |
| 131,072 | 71.44 Mrays/s | 191.41 Mrays/s | 2.94 ms | 4.56 | 96.45 | **1.98×** |
| 2,097,152 | 31.41 Mrays/s | 125.41 Mrays/s | 24.38 ms | 1.73 | 19.69 | **6.37×** |

- **主射线按原点 Morton 序重排**后，13 万三角形不再落后 Embree，2M 到 6.4×；排序
  本身只花 0.55–2.15 ms。非相干次生射线排序无收益（只省 ~2% 遍历），默认不开启。
- **构建全面领先**（1.3× / 1.9× / 2.8× Embree `rtcCommitScene`），LBVH 的并行排序
  天然适合 GPU。2M 构建里 refit 占 14.05/24.38 ms，是下一个优化目标。
- 输入序遍历仍然对拍不到 Embree 的 packet 模式（131k 0.74×），对外表述必须带口径。
- 结论来自 `gpu_bench.exe` 的实测，完整分阶段数据与反例记录见 `docs/baseline.md`。

### 最近点查询

`--query closest`，同一批 2 万个随机查询点，Embree 侧走 `rtcPointQuery`：

| 三角形 | quarry BVH（1 线程） | Embree（1 线程） | Embree（32 线程） |
|---|---|---|---|
| 13.1 万 | 0.18 Mq/s | 0.23 Mq/s | 1.93 Mq/s |
| 200 万 | 0.05 Mq/s | 0.06 Mq/s | 0.59 Mq/s |

单线程与 Embree 同档（0.77× / 0.82×）——点查询在两边都是标量随机访存受限；
差距全在多线程。2 万个点的平方距离逐位相同，只有 2 个等距平局选了不同三角形。

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
src/gpu/      CUDA：并行 LBVH 构建（lbvh.cu）、光线遍历（raycast.cu）
src/cli/      quarry 命令行
bench/        Embree 对照程序与 gpu_bench（同一份输入）
tests/        自测入口
learn/        教程配套的单文件练习
```

## 已知限制

- CPU 参考实现只有中位数切分、只有单线程
- GPU 侧目前只有射线求交；closest / kNN / radius 的 GPU 版本未开工
- GPU 射线排序已实现，但只对相干主射线有效；非相干射线需要两遍遍历（先取命中点
  再排序），未做。内联子节点包围盒（64B 布局）实测为负结果，已回退
- 场景类型只有地形网格和随机三角形，缺非均匀模型（Dragon/Lucy）与多层遮挡
- 最近点查询已和 Embree 对拍（单线程同档，多线程差 11×）；kNN / 半径还没有外部对照，
  因为对标实现不提供这两种原生入口；GPU 侧 Warp 对照待 M1
