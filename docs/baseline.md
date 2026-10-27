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

### GPU 后端：LBVH 构建 + 光线遍历（2026-10-27）

`gpu_bench.exe`，与 CPU/Embree 同一场景、同一射线、同一 seed。构建时间只计
GPU 上的建树（不含 H2D 上传与射线缓冲分配），遍历为 5 轮中位数，每次先校对
全部射线的命中位与 `t` 位模式。

| 三角形 | 射线 | 构建 | 遍历 | 吞吐 | 平均节点测试 | 平均三角形测试 |
|---|---|---|---|---|---|---|
| 8,192 | 200,000 | 1.61 ms | 1.97 ms | 101.41 Mrays/s | 29.9 | 1.9 |
| 131,072 | 1,000,000 | 2.94 ms | 14.01 ms | 71.44 Mrays/s | 38.7 | 1.9 |
| 2,097,152 | 1,000,000 | 24.38 ms | 31.84 ms | 31.41 Mrays/s | 47.2 | 1.9 |

对外比较（对手数字取自本文件前面的 Embree 4.4.1 实测，同一场景与射线）：

| 三角形 | GPU | Embree 1T | Embree 32T packet | 对 1T | 对 32T | GPU 构建 / Embree 构建 |
|---|---|---|---|---|---|---|
| 8,192 | 101.41 | 8.65 | 60.98 | 11.7× | **1.66×** | 1.61 / 2.13 ms |
| 131,072 | 71.44 | 4.56 | 96.45 | 15.7× | **0.74×** | 2.94 / 5.5 ms |
| 2,097,152 | 31.41 | 1.73 | 19.69 | 18.2× | **1.60×** | 24.38 / 68 ms |

构建分阶段（ms）：

| 三角形 | Morton | 排序 | 散射 | 建内部节点 | refit | 合计 |
|---|---|---|---|---|---|---|
| 8,192 | 0.84 | 0.36 | 0.02 | 0.08 | 0.32 | 1.61 |
| 131,072 | 0.61 | 0.67 | 0.15 | 0.30 | 1.20 | 2.94 |
| 2,097,152 | 1.63 | 3.38 | 2.25 | 3.07 | 14.05 | 24.38 |

读法：

1. **小场景核启动与队列延迟占主导。** 8k 三角形时 GPU 仍打出 101 Mrays/s，但只比
   Embree 32 线程快 1.7×——射线只有 20 万条，填不满 24 SM。
2. **131k 是 GPU 的劣势区间（0.74×）。** 整个 2 MB 场景塞进 CPU L2/L3，Embree 的
   32 线程 packet 遍历几乎不碰内存；GPU 反而要为每条射线付 HBM 延迟。这不是实现
   缺陷，是负载形状决定的，也是本文件第 1 条结论的另一面。
3. **2M 反超：1.60× Embree 32T、18× 单线程。** 场景超出 CPU 缓存后，Embree 的节点
   访问变成随机 DRAM 读取，而 GPU 有 1M 条射线在飞，延迟被完全藏住。
4. **构建从 10× 落后变成 2.8× 领先。** LBVH 的并行排序比 SAH 的中位数切分更适合
   GPU；refit 占 2M 场景构建的 58%（31 个深度桶各一次 kernel 启动），是下一个
   优化点，但当前已比 Embree `rtcCommitScene` 快。
5. **栈深度不是瓶颈，块大小有一点。** 每线程栈从 64 项压到 32 项，2M 遍历 33.74 ms
   vs 33.75 ms，无差别——kernel 受限于显存带宽/延迟，不是占用率；栈保持 64（2M
   场景树深 31，32 项在一般场景不安全）。块大小 128/256/512/1024 里 512 最好，
   三档遍历比 256 快约 3%（98.44→101.41），已采用。
6. **内联子节点盒子（64B 布局）是负结果。** 试过把两个子盒直接嵌进父节点、省一次
   节点访问：同热状态 A/B 下 8k −15%、131k −15%、2M +14%，整体不值得，已回退。

正确性：三个规模全部通过结构校验（每个内部节点包围盒是子节点的逐位并集、每个
叶子盒逐位等于三角形包围盒、每个非根节点恰好被引用一次、根盒逐位等于场景包围盒），
且 100% 射线的命中位与 `t` 位模式与 CPU BVH 一致：

```text
8k    verify rays=200000  hit_mismatch=0 t_mismatch=0  hash=0xd81f6da710bb9800 (cpu==gpu)
131k  verify rays=1000000 hit_mismatch=0 t_mismatch=0  hash=0xf4a49afdcc3fa8df (cpu==gpu)
2M    verify rays=1000000 hit_mismatch=0 t_mismatch=0  hash=0xa979843d01b01033 (cpu==gpu)
```

逐位一致靠的是：几何与求交代码在 host/device 之间共用同一份头文件（`QUARRY_HD`），
nvcc 关掉 FMA 融合（`--fmad=false`），遍历的节点测试顺序不影响结果（每条射线
维护自己的 bestT，命中取严格更近者）。

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
8. **GPU 的价值随场景大小增长，不是无条件 3×。** 131k 时 0.74×、2M 时 1.60×
   Embree 32T，拐点就是 CPU 缓存装不下场景的位置。对外表述必须带规模口径。
9. **GPU 构建已经全面领先**（1.3×/1.9×/2.8× Embree），因为 LBVH 天然并行；
   下一轮把 refit 从"每个深度一次启动"改成单 kernel 的并行 pull 可以再砍一半。

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

build\gpu_bench.exe --grid 64   --rays 200000  --runs 5 --stats
build\gpu_bench.exe --grid 256  --rays 1000000 --runs 5 --stats
build\gpu_bench.exe --grid 1024 --rays 1000000 --runs 5 --stats
```

## 已知缺口

- 只有两种几何（地形网格、随机三角形），缺非均匀模型（Stanford Dragon / Lucy）
- 命中率只有 100%（垂直向下）和约 31%（远处随机朝向），缺多层遮挡
- 只有一个 GPU 型号，每瓦数据要等 M1
- GPU 侧还只有射线求交：closest/kNN/radius 的 GPU 版本未开工
- GPU 遍历还没做射线排序（Morton 序）与内联子节点包围盒，这是 131k 落后
  Embree 32T 的两个直接原因
- GPU 外部对照只有 CPU 侧 Embree，Warp `bvh.ray_trace` 对照待 M1
- 查询外部对照只有 CPU 侧（Embree `rtcPointQuery`），GPU 侧 Warp `bvh.query_closest` 待 M1 接入
- kNN / radius 还没有外部对照，Embree 没有这两种查询的原生入口
- 第一轮结果（不同场景、20 万条射线）已作废，勿引用
