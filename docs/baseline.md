# Baseline（本机实测，2026-10-27）

所有数字都用**同一份场景生成器 + 同一份射线生成器**产出，因此可以横向比较。

## 机器

| 项 | 值 |
|---|---|
| CPU | Intel i9-14900HX，24 核 / 32 线程 |
| GPU | RTX 4060 Laptop，8188 MiB，**50 W 功耗墙** |
| 内存 | 31.7 GB |
| OS | Windows 11 build 26200 |
| 编译器 | MSVC 19.40（VS 2022 Community），`/O2` |
| Embree | 4.4.1 官方预编译包 |

## 协议

- 输入：地形网格，三角形数 = `2 × grid²`
- 射线：1,000,000 条，垂直向下，命中率 100%
- 预热一次后跑 5 轮，报**中位数**与 min/max
- 构建时间只计结构构建（Embree 为 `rtcCommitScene`，quarry 为 `buildBVH`），不含顶点缓冲区填充
- 单次运行 < 30 s，轮次间不额外冷却（笔记本功耗墙会带来 ±15% 波动）

## 结果

### 射线求交

| 三角形 | 实现 | 线程 | 构建 | 吞吐 |
|---|---|---|---|---|
| 131,072 | quarry 朴素 BVH | 1 | 49.7 ms | 2.84 Mrays/s |
| 131,072 | Embree 标量 | 1 | 6.7 ms | 4.56 Mrays/s |
| 131,072 | Embree packet | 32 | 5.5 ms | 96.45 Mrays/s |
| 2,097,152 | quarry 朴素 BVH | 1 | 1019 ms | 1.32 Mrays/s |
| 2,097,152 | Embree 标量 | 1 | 68 ms | 1.73 Mrays/s |
| 2,097,152 | Embree packet | 32 | 68 ms | 19.69 Mrays/s |

### 最近点 / kNN / 半径（新增，2026-10-27）

查询点均匀撒在包围盒 1.2 倍范围内（`makeQueryPoints`），20,000 个，单线程，
每次都是随机位置——没有相邻查询共享缓存。3 轮中位数。

| 三角形 | 查询 | 时间 | 单次 | 节点测试 | 三角形测试 | 结果数/查询 |
|---|---|---|---|---|---|---|
| 131,072 | closest | 120.2 ms | 6.01 µs | 49.0 | 39.0 | 1.0 |
| 131,072 | kNN k=16 | 159.3 ms | 7.97 µs | 57.2 | 52.1 | 16.0 |
| 131,072 | radius r=2.0 | 90.0 ms | 4.50 µs | 41.3 | 51.0 | 39.5 |
| 2,097,152 | closest | 374.4 ms | 18.7 µs | 135.0 | 131.7 | 1.0 |
| 2,097,152 | kNN k=16 | 391.1 ms | 19.6 µs | 140.8 | 141.4 | 16.0 |
| 2,097,152 | radius r=0.5 | 59.3 ms | 2.97 µs | 14.4 | 12.6 | 9.7 |
| 1,000,000 random | closest | 88.6 ms | 17.7 µs | 162.0 | 131.5 | 1.0 |
| 1,000,000 random | kNN k=16 | 206.1 ms | 41.2 µs | 264.5 | 254.0 | 16.0 |

2M 场景的 radius 用 r=0.5 而不是 2.0：网格加密 16 倍后，同样的世界空间球会套住
约 16 倍三角形（r=2.0 时平均 528 个/查询），跨规模没有可比性。

暴力法对 BVH（131k，500 个查询点，`hash` 逐位相同）：

| 查询 | 暴力 | BVH | 加速 |
|---|---|---|---|
| closest | 2.29 ms/查询 | 5.2 µs | 437× |
| kNN k=16 | 4.04 ms/查询 | 7.0 µs | 576× |
| radius r=2.0 | 2.16 ms/查询 | 5.8 µs | 374× |

### 外部对照：Embree 4.4.1 `rtcPointQuery`（2026-10-27）

同一份场景生成器、同一批查询点（`--seed` 相同），20,000 个点，5 轮中位数。
Embree 走 `rtcPointQuery` + 回调收紧半径（半径从 ∞ 开始，找到第一个三角形后逐步收缩，
等价于 BVH 的最优优先遍历）。

| 三角形 | 实现 | 线程 | 构建 | 时间 | 吞吐 | 对 Embree 单线程 |
|---|---|---|---|---|---|---|
| 131,072 | quarry BVH | 1 | 58.5 ms | 111.6 ms | 0.18 Mq/s | 0.77× |
| 131,072 | Embree | 1 | 14.7 ms | 85.5 ms | 0.23 Mq/s | 1.00× |
| 131,072 | Embree | 32 | 11.0 ms | 10.3 ms | 1.93 Mq/s | 10.8× |
| 2,097,152 | quarry BVH | 1 | 1166 ms | 394.1 ms | 0.05 Mq/s | 0.82× |
| 2,097,152 | Embree | 1 | 123.6 ms | 323.2 ms | 0.06 Mq/s | 1.00× |
| 2,097,152 | Embree | 32 | 92.8 ms | 34.1 ms | 0.59 Mq/s | 11.6× |

读法：

1. **单线程上 quarry 与 Embree 同档（0.77× / 0.82×）。** 点查询在 Embree 里也是标量
   回调路径，没有 SIMD 可吃；两边瓶颈都是随机访存，不是指令数。
2. **差距全在多线程。** Embree 32 线程拉出 11×，quarry 只有单线程路径。这正是 M0 要
   用 GPU 抢的位置——把 10 万条随机查询的延迟同时挂起来。
3. 构建仍差 9.5×（1166 vs 124 ms，2M 场景），中位数切分 vs SAH 的预期结果。

逐条对拍（`--verify 20000`）：

```text
verify points=20000 id_mismatch=2 (ties=2) dist2_mismatch=0 worst_rel=0
```

- 20,000 个点的平方距离**逐位相同**，`sum_dist2` 两边都是 392670.867797（131k）/ 393145.161104（2M）。
- 三角形 id 一致 19,998/20,000。剩下 2 个是地形平面上的**等距平局**：查询点正好落在
  多个三角形共享的位置，两边各选了一个正确答案，距离完全相同。
- 所以两边的指纹 `hash` 不同（hash 里含三角形 id），这不是错误。GPU 版本要求的是
  与 **quarry 自己的** tie-break 规则一致（距离相同取场景序号最小），而非与 Embree 一致。

## 结论

1. **CPU 吞吐随场景大小塌陷一个数量级。** 13 万三角形受缓存保护，32 线程能到 96 Mrays/s；200 万三角形掉到 19.7，属于结构性访存受限，增加线程无用。
2. **"3× Embree" 只在大场景口径下成立。** 小场景的对手是缓存带宽与 kernel 启动延迟，价值主张应换成"CPU 占用为 0"。
3. **朴素 BVH 达到 Embree 单线程的 60–75%**，差距来自 BVH4 布局、SAH 与 SIMD 遍历，不是实现缺陷。
4. 构建时间差 10–15×，符合中位数切分 vs SAH 的预期，M1 用 LBVH 补齐。
5. **最近点 / kNN / 半径是纯延迟受限，和射线求交不是同一种负载。** 每次查询只做
   50–140 次节点测试，但查询点随机分布 → 每次节点访问几乎都是冷缓存行，
   单次测试摊到 70–100 ns。这正好是 GPU 的强项：同时挂 10 万条查询把延迟藏起来，
   而不是靠单线程把指令数压下去。M0 的对照实验必须以这个口径设计。
6. **kNN 的剪枝阈值靠第 k 近距离，随机三角形场景里收敛很慢**（264 次节点测试，
   比 closest 多 63%）。点云和稠密表面会更极端，这里需要一个"提前退出"策略。
7. **单线程 closest 查询已和 Embree 同档（0.77–0.82×），不需要再优化 CPU 路径。**
   CPU 侧欠的是多线程（11×）和构建（9.5×）；这两项在 GPU 上会同时被改写，
   所以 CPU 参考实现停在"正确性基线"就够了。

## 正确性交叉核对

`embree_bench.exe --verify 500`：500 条射线逐条比对
Embree 与 quarry 的 Möller–Trumbore 暴力法

```text
verify rays=500 hits=500 hit_mismatch=0 t_mismatch=0
```

命中数与 t 值零差异。汇总距离和相差 < 1e-3（单精度累加顺序不同）。

查询路径用另一种核对：同一个查询点分别走暴力法和 BVH，要求**选中的三角形、
并列时的顺序、平方距离的 32 位表示**全部一致。131k 场景、500 个点：

```text
closest  hash=0x13d1e7dbca595588  (brute == bvh)
kNN      hash=0xa0beaa4b1c87e0d1  (brute == bvh)
radius   hash=0x19fa5f146ef19453  (brute == bvh)
```

`selftest` 里同样的检查覆盖 closest / kNN / radius 三条路径（含“返回同一个三角形”）。

## 复现

```bat
build.bat
build\quarry.exe run --scene terrain --grid 256  --rays 1000000 --engine bvh --runs 5
build\quarry.exe run --scene terrain --grid 1024 --rays 1000000 --engine bvh --runs 5

build\quarry.exe query --type closest --grid 256  --points 20000 --runs 3 --stats
build\quarry.exe query --type knn     --k 16 --grid 256  --points 20000 --runs 3 --stats
build\quarry.exe query --type radius  --radius 2.0 --grid 256 --points 20000 --runs 3 --stats
build\quarry.exe query --type radius  --radius 0.5 --grid 1024 --points 20000 --runs 3 --stats

build\quarry.exe query --type closest --grid 256 --points 500 --engine brute
build\quarry.exe query --type closest --grid 256 --points 500 --engine bvh

bench\build_embree.bat
bench\embree_bench.exe --grid 256  --rays 1000000 --threads 1  --runs 5
bench\embree_bench.exe --grid 256  --rays 1000000 --threads 32 --packets --runs 5
bench\embree_bench.exe --grid 1024 --rays 1000000 --threads 1  --runs 5
bench\embree_bench.exe --grid 1024 --rays 1000000 --threads 32 --packets --runs 5

bench\embree_bench.exe --query closest --grid 256  --points 20000 --threads 1  --runs 5
bench\embree_bench.exe --query closest --grid 256  --points 20000 --threads 32 --runs 5
bench\embree_bench.exe --query closest --grid 1024 --points 20000 --threads 1  --runs 5
bench\embree_bench.exe --query closest --grid 1024 --points 20000 --threads 32 --runs 5
bench\embree_bench.exe --query closest --grid 256  --points 20000 --threads 1 --verify 20000
```

## 已知缺口

- 只有两种几何（地形网格、随机三角形），缺非均匀模型（Stanford Dragon / Lucy）
- 命中率只有 100%（垂直向下）和约 31%（远处随机朝向），缺多层遮挡
- 只有一个 GPU 型号，每瓦数据要等 M1
- 查询外部对照只有 CPU 侧（Embree `rtcPointQuery`），GPU 侧 Warp `bvh.query_closest` 待 M1 接入
- kNN / radius 还没有外部对照，Embree 没有这两种查询的原生入口
- 第一轮结果（不同场景、20 万条射线）已作废，勿引用
