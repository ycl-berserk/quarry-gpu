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
全部射线的命中位与 `t` 位模式。射线有两种模式，先给出**按输入顺序**遍历的
裸数字（与 Embree 的口径一致）：

| 三角形 | 射线 | 构建 | 遍历（输入序） | 吞吐 | 平均节点测试 | 平均三角形测试 |
|---|---|---|---|---|---|---|
| 8,192 | 200,000 | 1.61 ms | 1.97 ms | 101.41 Mrays/s | 29.9 | 1.9 |
| 131,072 | 1,000,000 | 2.94 ms | 14.01 ms | 71.44 Mrays/s | 38.7 | 1.9 |
| 2,097,152 | 1,000,000 | 24.38 ms | 31.84 ms | 31.41 Mrays/s | 47.2 | 1.9 |

再给出**按原点 Morton 序重排后**的端到端数字（排序 kernel + CUB 基数排序 +
重排 + 遍历 + 结果散回，全在 CUDA event 计时区内）：

| 三角形 | 射线 | 排序 | 端到端 | 吞吐 | 端到端 / 输入序 |
|---|---|---|---|---|---|
| 8,192 | 200,000 | 0.55 ms | 1.03 ms | 193.57 Mrays/s | 1.91× |
| 131,072 | 1,000,000 | 2.15 ms | 5.22 ms | 191.41 Mrays/s | 2.68× |
| 2,097,152 | 1,000,000 | 2.15 ms | 7.97 ms | 125.41 Mrays/s | 3.99× |

对外比较（对手数字取自本文件前面的 Embree 4.4.1 实测，同一场景与射线）：

| 三角形 | GPU 输入序 | GPU 排序端到端 | Embree 1T | Embree 32T packet | 输入序 / 32T | 排序后 / 32T | 排序后 / 1T |
|---|---|---|---|---|---|---|---|
| 8,192 | 101.41 | 193.57 | 8.65 | 60.98 | 1.66× | **3.17×** | 22.4× |
| 131,072 | 71.44 | 191.41 | 4.56 | 96.45 | 0.74× | **1.98×** | 42.0× |
| 2,097,152 | 31.41 | 125.41 | 1.73 | 19.69 | 1.60× | **6.37×** | 72.5× |

GPU 构建 / Embree 构建仍为 1.61/2.13 ms、2.94/5.5 ms、24.38/68 ms（1.3× / 1.9× / 2.8×）。

旧版 refit（全局 atomic 分桶）的构建分阶段（10-27，ms）：

| 三角形 | Morton | 排序 | 散射 | 建内部节点 | refit | 合计 |
|---|---|---|---|---|---|---|
| 8,192 | 0.84 | 0.36 | 0.02 | 0.08 | 0.32 | 1.61 |
| 131,072 | 0.61 | 0.67 | 0.15 | 0.30 | 1.20 | 2.94 |
| 2,097,152 | 1.63 | 3.38 | 2.25 | 3.07 | 14.05 | 24.38 |

**2026-10-28：refit 重写（同场 A/B）。** 上表的 refit 用全局 `atomicAdd` 做深度
计数和落桶：200 万线程抢 128 个计数器，实测"分布 4.81 + 落桶 3.58 ms"就占了
2M 构建的 58%。现在改成一次 7 位稳定基数排序：内部节点编号恰好是 `0..n-2`，
把它们的深度作为键交给 CUB `SortPairs(..., end_bit=7)`；同层内按编号升序，
每层起点由单 block 对有序键做 `lower_bound`，只回传 64 个 int。refit kernel
再从整层改成每线程展开 4 个节点——子节点下标是散的，展开后 8 个 32B 子盒读
能同时挂在显存流水线上（8 路展开会寄存器溢出，实测更慢）。叶子盒在
`leafInitKernel` 已定稿，不再进桶。

同一时段连续跑旧、新两个二进制（`--no-cpu --runs 1`，`CUDA_MODULE_LOADING=EAGER`
消除模块加载差异）：

| 场景 | 版本 | refit | 总构建 |
|---|---|---|---|
| 2M terrain（depth 31） | 旧 → 新 | 6.19 → 2.67 ms | 11.00 → 7.40 ms |
| 2M random（depth 27） | 旧 → 新 | 7.45 → 2.82 ms | 13.54 → 8.72 ms |
| 131k terrain | 旧 → 新 | 0.92 → 0.31 ms | 1.71 → 0.90 ms |

第二轮的抖动范围：2M terrain refit 6.23→3.06、2M random 5.84→2.67、131k 0.60→0.35，
方向一致。refit 本身快了 2.1–2.7×，整树构建快约 1.5×。剩下的 refit 已是
"整层并行 pull + 散读子盒"的显存延迟负载，继续压需要改节点布局（BFS）或单
kernel 协作，改动面与收益不成比例，M0 到此为止。

**两个计时口径要记住。**

- CUDA 默认懒加载模块：进程第一次启动 kernel 才把整个 cubin 载入设备，实测约
  1.5–2.3 ms，之前全被记在 `morton` 阶段头上。`gpu_bench` 现在先调用
  `warmupBuildGPU()` 空跑一个 kernel，把一次性开销移出构建计时。
- 笔记本的电源状态会让绝对吞吐跨会话漂移：同一份二进制在 10-27 与 10-28（电池
  42%、50 W 上限、CPU 后台负载 52%）两次测得的值相差 1.5–2×，Embree 32 线程受
  CPU 负载影响更大。因此**所有 GPU/CPU 对照都必须在同一时段连续测**；上表保留
  10-27 的整组数字，10-28 的 A/B 只用于比较新旧 refit。

读法：

1. **输入序下，小场景受启动延迟限制、131k 受 CPU 缓存限制。** 8k 只有 101 Mrays/s
   （1.66× 32T），射线 20 万条填不满 24 SM；131k 整个 2 MB 场景进 CPU L3，Embree
   32 线程几乎不碰内存，GPU 反而要付 HBM 延迟（0.74×）。
2. **排序把这两档一起改写。** 端到端 1.03 / 5.22 ms 后，8k 和 131k 都到 191–194
   Mrays/s——分别 3.17× 和 1.98× Embree 32T，131k 的劣势区间消失。
3. **2M 排序后 6.37× Embree 32T、72.5× 单线程。** 输入序已经靠"1M 条射线在飞"
   藏住延迟；排序再让一个 warp 的节点访问落进同一片 L1/L2，遍历部分从 31.84 降到
   约 5.8 ms，排序只付 2.15 ms。
4. **构建从 10× 落后变成 2.8× 领先。** LBVH 的并行排序比 SAH 的中位数切分更适合
   GPU；refit 当时占 2M 场景构建的 58%（31 个深度桶各一次 kernel 启动），
   10-28 已按本节的稳定排序方案重写，同一时段 A/B 从 6.2 降到 2.7 ms。
5. **栈深度不是瓶颈，块大小有一点。** 每线程栈从 64 项压到 32 项，2M 遍历 33.74 ms
   vs 33.75 ms，无差别——kernel 受限于显存带宽/延迟，不是占用率；栈保持 64（2M
   场景树深 31，32 项在一般场景不安全）。块大小 128/256/512/1024 里 512 最好，
   三档遍历比 256 快约 3%，已采用。
6. **内联子节点盒子（64B 布局）是负结果。** 试过把两个子盒直接嵌进父节点、省一次
   节点访问：同热状态 A/B 下 8k −15%、131k −15%、2M +14%，整体不值得，已回退。

**射线排序只对相干主射线有效。** 上表的 `down` 射线原点均匀撒在场景地面上方，
输入顺序毫无空间局部性；按原点 Morton 序重排后，同一 warp 的遍历路径高度重合，
排序收益（1.9×–4.0×）远大于自身开销（0.55–2.15 ms）。换成本文件
`--mode random` 的球面原点/随机目标射线，结论反转：

| 场景 | 输入序 | 原点排序 | 方向排序 | 纯遍历增益 |
|---|---|---|---|---|
| 131k | 9.18 ms | 11.09 ms | 11.08 ms | ≈2.6% |
| 2M | 12.40 ms | 14.39 ms | 14.40 ms | ≈1.3% |

原因：这类射线的遍历路径由"打到哪里"决定，而不是"从哪里出发"。排序只把起点
或方向拉近，节点集合仍然发散，省下的访问 ~0.2–0.3 ms 抵不过 2.15 ms 的排序。
要做对得先粗遍历求出命中点、再按命中点做第二遍排序（两遍遍历），不在 M0 范围。
因此 `raycastGPU` 默认仍是 `RayOrder::Input`，排序由调用方按射线类型显式开启。

**顺带修掉一个 CPU 参考实现的剪枝错误。** 2M 随机射线对拍曾出现 1 条 `t` 相差
4 ulp（`0x4336be18` vs `0x4336be14`）。用暴力遍历仲裁：GPU 与暴力逐位一致，
CPU 反而报了一个更远的命中——叶子盒入口 `t` 和三角形 `t` 由不同表达式算出，
盒入口可能因舍入略大于 `bestT`，于是把"三角形其实更近"的命中剪掉。修法是在
公共 `intersectAABB` 的剪枝阈值上留 `1e-6` 的相对+绝对余量；CPU/GPU 同时生效，
修复后三方逐位一致，`down` 场景的指纹不变。

正确性：三个规模全部通过结构校验（每个内部节点包围盒是子节点的逐位并集、每个
叶子盒逐位等于三角形包围盒、每个非根节点恰好被引用一次、根盒逐位等于场景包围盒），
且 100% 射线的命中位与 `t` 位模式与 CPU BVH 一致：

```text
8k    verify rays=200000  hit_mismatch=0 t_mismatch=0  hash=0xd81f6da710bb9800 (cpu==gpu)
131k  verify rays=1000000 hit_mismatch=0 t_mismatch=0  hash=0xf4a49afdcc3fa8df (cpu==gpu)
2M    verify rays=1000000 hit_mismatch=0 t_mismatch=0  hash=0xa979843d01b01033 (cpu==gpu)
2M random verify rays=1000000 hit_mismatch=0 t_mismatch=0 hash=0x3dde33b78beab3a4 (cpu==gpu)
```

Morton 排序只改执行顺序、结果按原下标散回，三个规模的排序路径指纹与上表完全相同
（`0xd81f…` / `0xf4a4…` / `0xa979…`）。另有 `--diag-ray N` 可对单条射线打印
CPU BVH / GPU BVH / 暴力三方结果。

逐位一致靠的是：几何与求交代码在 host/device 之间共用同一份头文件（`QUARRY_HD`），
nvcc 关掉 FMA 融合（`--fmad=false`），遍历顺序不影响结果（每条射线维护自己的
bestT，命中取严格更近者），**且盒测试的剪枝阈值留了浮点余量**——少了最后一条，
`t` 在 ulp 级就会与访问顺序相关（见上面随机射线的 4 ulp 案例）。

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
8. **GPU 的倍数取决于射线相干性和口径，不是无条件的 3×。** 输入序：1.66× /
   0.74× / 1.60× Embree 32T；主射线按 Morton 序重排并计入排序开销后：3.17× /
   1.98× / 6.37×。但同一套排序用在非相干次生射线上反而亏 2 ms，只对主射线开启。
   对外表述必须同时带场景大小、射线类型和是否含排序。
9. **GPU 构建已经全面领先**（1.3×/1.9×/2.8× Embree，10-27 同场口径）；refit 重写
   后同场 A/B 再快 2.1–2.7×（2M 6.2→2.7 ms），整树构建约 1.5×。下一步不是继续
   抠 refit，而是 M1 的增量更新与点/高斯图元。

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
build\gpu_bench.exe --grid 256  --rays 1000000 --runs 5 --stats --ray-sort
build\gpu_bench.exe --grid 1024 --rays 1000000 --runs 5 --stats --ray-sort
build\gpu_bench.exe --grid 1024 --rays 1000000 --runs 5 --mode random --ray-sort
build\gpu_bench.exe --grid 1024 --rays 1000000 --runs 1 --mode random --diag-ray 98645
```

## 已知缺口

- 只有两种几何（地形网格、随机三角形），缺非均匀模型（Stanford Dragon / Lucy）
- 命中率只有 100%（垂直向下）和约 31%（远处随机朝向），缺多层遮挡
- 只有一个 GPU 型号，每瓦数据要等 M1
- GPU 侧还只有射线求交：closest/kNN/radius 的 GPU 版本未开工
- 射线排序已完成（原点/方向 Morton 序），但只对相干主射线有效；非相干射线需要
  "先粗遍历取命中点、再按命中点排序"的两遍方案，未做
- 内联子节点包围盒（64B 布局）实测是负结果，已回退；131k 的差距靠射线排序解决
- GPU 外部对照只有 CPU 侧 Embree，Warp `bvh.ray_trace` 对照待 M1
- 查询外部对照只有 CPU 侧（Embree `rtcPointQuery`），GPU 侧 Warp `bvh.query_closest` 待 M1 接入
- kNN / radius 还没有外部对照，Embree 没有这两种查询的原生入口
- 第一轮结果（不同场景、20 万条射线）已作废，勿引用
