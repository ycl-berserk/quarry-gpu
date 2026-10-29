# Baseline（本机实测，2026-10-28）

所有数字都用**同一份场景生成器 + 同一份射线生成器**产出，因此可以横向比较。

## 机器

| 项 | 值 |
|---|---|
| CPU | Intel i9-14900HX，24 核 / 32 线程 |
| GPU | RTX 4060 Laptop，8188 MiB，**55 W 功耗墙**（上限 140 W，未解锁） |
| 内存 | 31.7 GB |
| OS | Windows 11 build 26200 |
| 编译器 | MSVC 19.40（VS 2022 Community），`/O2` |
| Embree | 4.4.1 官方预编译包 |

## 协议

- 输入：地形网格，三角形数 = `2 × grid²`
- 射线：1,000,000 条，垂直向下，命中率 100%
- 预热一次后跑 5 轮，报**中位数**与 min/max；GPU 计时前额外空跑 3 s 拉高 DVFS 时钟
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

**2026-10-28 复测（同一条命令，另加 3 s DVFS 预热）。** 上面两张表是 10-27 的会话。
今天用同一份二进制重测：输入序 215.10 / 143.63 / 62.98 Mrays/s，排序端到端
463.05 / 343.52 / 227.05，同场 Embree 32T packet 136.37 / 70.93 / 26.21 Mrays/s，
即输入序 1.58× / 2.03× / 2.40×、排序端到端 3.40× / 4.84× / 8.66×。两批数字都是
真的，差别来自笔记本的 DVFS 与散热状态（10-27 那批连预热都没有，却跑在更高的
时钟档）。对外引用必须带日期和配对口径，跨日只比指纹与结构量。

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

### GPU 点查询：closest / kNN / 半径（2026-10-28）

`gpu_query_bench.exe`，20,000 个查询点，位置与 CPU/Embree 的 `makeQueryPoints` 相同。
GPU 时间是 CUDA event 包住的 kernel（结果常驻显存，不含 H2D/D2H）。计时前先持续跑
3 s 把 DVFS 时钟拉起来（原因见本节末尾）；closest / kNN / radius 各 5 轮中位数。
表里的 Embree 与 Warp 都是同一次会话里紧跟着跑出来的。

closest（GPU 每线程一个查询点，显式栈 DFS；剪枝规则和 CPU 相同，并列取场景索引小者）：

| 三角形 | quarry GPU | Warp 1.6.2 | Embree 1T | Embree 32T | / Embree 32T | / Warp |
|---|---|---|---|---|---|---|
| 8,192 | 0.544 ms (36.78 Mq/s) | 38.58 Mq/s | 1.11 | 4.81 | **7.6×** | 0.95× |
| 131,072 | 1.491 ms (13.42 Mq/s) | 18.63 Mq/s | 0.41 | 3.18 | **4.2×** | 0.72× |
| 2,097,152 | 7.731 ms (2.59 Mq/s) | 3.12 Mq/s | 0.08 | 0.85 | **3.0×** | 0.83× |
| 1,000,000 random | 3.971 ms (5.04 Mq/s) | 4.37 Mq/s | 0.03 | 0.31 | **16.3×** | 1.15× |

每一行都是"quarry 跑完立刻跑 Warp"，避免时钟状态在两者之间漂移。

建树（Warp 为 `wp.Mesh` 内部的 BVH 构建，已经扣掉进程内第一次建 mesh 的一次性初始化；
构建本身不受 3 s 预热保护，8k 的 quarry 构建在冷时钟下会在 1.0–2.2 ms 之间跳）：

| 三角形 | quarry LBVH | Warp Mesh | Embree `rtcCommitScene` | quarry / Warp |
|---|---|---|---|---|
| 8,192 | 1.0–2.2 ms | 0.54 ms | 2.00 ms | 1.8–4.0× 慢 |
| 131,072 | 1.79 ms | 1.29 ms | 5.87 ms | 1.4× 慢 |
| 2,097,152 | 14.63 ms | 26.28 ms | 71.62 ms | **1.8× 快** |
| 1,000,000 random | 7.62 ms | 12.33 ms | 32.36 ms | **1.6× 快** |

kNN k=16（每线程维护按 (dist2, 索引) 升序的定长表，上限 32；满 k 后按第 k 近剪枝）：

| 三角形 | GPU | CPU 1T | GPU / CPU 1T | 平均节点 / 三角形测试 |
|---|---|---|---|---|
| 8,192 | 3.286 ms (6.09 Mq/s) | 0.34 | 17.9× | 180.0 / 60.6 |
| 131,072 | 5.694 ms (3.51 Mq/s) | 0.18 | 19.5× | 303.7 / 100.0 |
| 2,097,152 | 13.922 ms (1.44 Mq/s) | 0.07 | 20.6× | 684.1 / 222.7 |

radius（CSR 两遍遍历：计数 → 前缀和 → 填充 → 段内排序；结果条数上不封顶）：

| 三角形 | 半径 | GPU | CPU 1T | GPU / CPU 1T | 结果数/查询 | 平均节点测试 |
|---|---|---|---|---|---|---|
| 8,192 | 2.0 | 1.478 ms (13.54 Mq/s) | 1.79 | 7.6× | 4.3 | 32.4 |
| 131,072 | 2.0 | 5.602 ms (3.57 Mq/s) | 0.29 | 12.3× | 39.5 | 124.2 |
| 2,097,152 | 0.5 | 3.214 ms (6.22 Mq/s) | 1.03 | 6.0× | 9.7 | 35.6 |

读法：

1. **closest 全面超过 Embree 32 线程**（7.6× / 4.2× / 3.0×，随机 1M 到 16.3×）。
   GPU 的 DFS 反而比 CPU 最优优先遍历做了更多节点测试（2M 477 vs 135），但几千个
   查询同时挂起，把单次访问的 70–100 ns 延迟藏掉了——这正是 10-27 的 CPU 结论里
   预期由 GPU 改写的那一项。
2. **但 Warp 1.6.2 在地形场景上仍然更快**（8k 持平，131k 快 1.4×，2M 快 1.2×），
   只在随机三角形汤上被我们反超（1.15×）。差距在树形而不是遍历骨架：Warp 用
   自顶向下 SAH 建树、每叶 4 个图元（`native/bvh.cpp`，`BVH_LEAF_SIZE 4`），
   我们用 Karras LBVH + 每叶 1 个三角形（2M → 419 万节点、深度 31）。
   把 Morton 序相邻的 4 个三角形硬塞进一片叶子实测是**负优化**，见下一小节。
3. **构建的胜负和场景大小反过来**：8k / 131k 我们慢 1.4–4×，2M 和随机 1M 反而
   快 1.6–1.8×。LBVH 的基数排序常数小、完全并行；Warp 的构建在小场景上赢在启动
   路径短，大场景上被它的构建策略拖住。
4. **kNN 的倍数随规模上升**（17.9× → 20.6×）：CPU 的第 k 近阈值收敛慢，kNN 的节点数
   随规模涨得比 closest 快（684 vs 477，2M），但 GPU 上这些额外节点几乎是免费并行。
5. **radius 是三者里最慢的路径**，因为它要遍历两遍、还要一次前缀和与段内排序。
   第一版用 CUB 分段基数排序在两万个短段上花了 13 ms；换成每 warp 一个双调网络
   （段长 ≤ 256，cap 取 2 的幂，尾部补 UINT64_MAX）后排序段降到 ~0.46 ms。
   超过 256 条的长段（例如半径设到包住整个场景）退回 CUB 分段排序，正确但慢。
6. **radius 的计时含一次流水线中间的 D2H 同步**：CSR 总长度只有算完前缀和才知道，
   不同步就没法给填充缓冲定价。closest / kNN 没有这笔开销。
7. **kNN / radius 没有外部基线可对拍**：Embree 的 `rtcPointQuery` 只做最近点，Warp
   也没有 kNN / 半径入口，所以这两行只有自家 CPU 参考。closest 的三方对照是
   唯一的外部口径。

#### 外部 GPU 基线：Warp 1.6.2

两边必须看到**逐字节相同的输入**，否则差值没有意义。`gpu_query_bench --dump FILE`
把场景写成 `QRYD` 二进制（magic、版本、三角形数、点数为两个 u64，之后是连续的
`float` 三角形顶点与查询点），Warp 端用同一份三角形和点建 `wp.Mesh`、跑自己的 BVH。

- 查询用 `wp.mesh_query_point_no_sign(mesh, p, 1e6)`，返回 `face` 与重心坐标 `u, v`；
  Warp 的重心约定是 `u·a + v·b + (1−u−v)·c`（`warp/native/intersect.h`），
  写成 `a + u(b−a) + v(c−a)` 会把 `sum_dist2` 算错 4%。
- Warp 的 BVH 每个叶子是一段图元，且带一条 sliver 剔除：法向长度除以三条边平方和
  小于 `1e-6` 的三角形直接跳过（`mesh.h`）。我们的地形三角形不会触发，随机汤里也
  实测没有影响到结果。
- 计时口径：`runs × 10` 次 launch 取中位数（单次 launch 只有 0.5–6 ms，逐次计时的
  开销占比太大），同样先持续跑 3 s。
- 正确性：200 个随机采样点上 `suboptimal=0`。用相对误差判会误报——查询点几乎落在
  三角形上时 `d² ≈ 1e-8`，float32 重建最近点的末位误差就能让相对误差到 `5e-3`；
  逐点核对过 33 个"超差"样本，Warp 选中的三角形与暴力法**完全相同**，只是距离的
  末几位不同。我们的结果与 CPU 参考逐位相同，Warp 是 float32 精度的同位结果。

**绝对值受电源状态影响明显。** 同一二进制换一个会话，131k 的 kNN 可报 4.7 ms 也可报
11.3 ms、半径 2.5 也可报 5.4 ms，指纹完全不变。本次会话在计时期间采样到的 SM 时钟
是 1.38–1.61 GHz、功耗 27 W（上限 55 W）：几十毫秒的 kernel 无论跑多少轮都不会让
DVFS 升到高频，必须先空跑 3 s（实测时钟约 4 s 到顶 2.0 GHz）。即使如此，本批 GPU
绝对值仍比 10-27 那批低 20–25%，所以**倍数列一律取同一次会话内的配对数字**，
跨会话只比指纹和结构量（节点数、节点测试次数）。

#### 已排除的优化：Morton 叶子批量（2026-10-28）

"Warp 每叶 4 个图元，所以我们也把 4 个 Morton 序相邻的三角形并成一片叶子"——
实现后逐位结果不变，但遍历全面变慢。树的构建反而更快、内存更省：

| 2M 地形 | 每叶 1 个 | 每叶 4 个 | 变化 |
|---|---|---|---|
| 节点数 / 深度 | 4,194,303 / 31 | 1,048,575 / 28 | 节点少 4× |
| 构建 total | 8.28 ms | 5.37 ms | **1.54× 快** |
| 其中 tree + refit | 1.70 + 2.94 ms | 0.41 + 1.47 ms | 2–4× 快 |
| 射线 1M 条 | 9.12 ms | 19.77 ms | **2.17× 慢** |
| 射线平均节点/三角形测试 | 47.2 / 1.9 | 92.0 / 34.4 | 节点 2×、三角形 18× |
| closest 20k 点 | 4.860 ms | 5.032 ms | 3.5% 慢 |

小场景同样单调变差：8k 地形上每叶 1/2/4/8 个的射线吞吐是
320.6 / 202.7 / 154.4 / 116.6 Mrays/s，closest 是 0.315 / 0.400 / 0.502 / 0.651 ms；
131k 的 kNN k=8 从 1.733 ms 变成 3.056 ms（1.76×）。

原因不是实现错误（结果指纹逐位不变、结构校验全过）：Morton 序上相邻的 4 个三角形
并不构成紧凑的包围盒，射线/查询点进入叶子的概率随之上升，而每进一次叶子要固定
测 4 个三角形。我们的三角形测试（closest 是几十次浮点运算 + 一次开方）并不比盒
测试便宜，用"更浅的树"换"多 4 倍三角形测试"在任何场景都不划算。Warp 的叶子 4 个
图元是 **SAH 自顶向下**切出来的，叶子盒和兄弟盒重叠小；叶子粒度必须跟构建质量
一起看。这个实验留在 `exp/leaf-batching` 分支上，主线保持每叶 1 个。

逐位对拍（20,000 点，GPU 对 CPU 参考 BVH）：

```text
8k     closest tri_mismatch=0 dist2_mismatch=0  hash=0xcb166f675fbdaf7f (cpu==gpu)
131k   closest tri_mismatch=0 dist2_mismatch=0  hash=0x254ce454a3bba31f (cpu==gpu)
2M     closest tri_mismatch=0 dist2_mismatch=0  hash=0x9eef1b6e10b68ca2 (cpu==gpu)
8k     kNN k=16 count_mismatch=0 tri_mismatch=0 dist2_mismatch=0
131k   kNN k=16 count_mismatch=0 tri_mismatch=0 dist2_mismatch=0
2M     kNN k=16 count_mismatch=0 tri_mismatch=0 dist2_mismatch=0
8k     radius r=2.0 count_mismatch=0 tri_mismatch=0 dist2_mismatch=0
131k   radius r=2.0 count_mismatch=0 tri_mismatch=0 dist2_mismatch=0
2M     radius r=0.5 count_mismatch=0 tri_mismatch=0 dist2_mismatch=0
```

kNN / radius 的指纹是"先折叠结果条数、再逐条折叠 (dist2 位模式, 索引)", CPU 与 GPU
逐位相同；k=1、k=32 和"半径大到每点几千条"的兜底路径也分别验证过。

#### Python 绑定的固定开销（2026-10-28）

131k 地形、20,000 个随机查询点，`py -3.13 python/demo.py` 同一进程内连跑：

```text
第一个 Scene:  墙钟 571.1 ms   报告 build_ms 2.25 ms
第二个 Scene:  墙钟   8.6 ms   报告 build_ms 1.04 ms
closest 端到端 墙钟中位数 1.436 ms   内核中位数 1.292 ms   内核 15.5 Mq/s
```

差的 0.14 ms 里，上传 20k 点约 0.02 ms、回读 + 建 numpy 数组约 0.10 ms、
pybind11 调用本身约 0.02 ms（按墙钟减内核反推，只有内核是 instrumented 的）。
第一个 `Scene` 的 571 ms 是 CUDA 模块懒加载，发生在第一个 kernel 之前，所以不在
`build_ms` 的计时窗口里——报"建场景耗时"时不能用 `build_ms` 代替用户感知的延迟。

同一进程内重复调用时这项开销稳定在 0.13–0.15 ms，随查询点数线性变化；场景变小后
内核时间下降、占比上升——8k 场景 closest 内核约 0.54 ms（20k 点 36.8 Mq/s），
固定开销就是 20%+。

#### 增量更新：refit vs 重建（2026-10-29）

`gpu_query_bench.exe --refit --refit-dirty 0.01 --refit-motion random|coherent`，
三个规模 × 两种运动，同一条命令里依次测：

1. **局部 refit**：每 100 个三角形改 1 个（8k → 82、131k → 1,311、2M → 20,972 个），
   只上传这些三角形与叶子槽号，并沿 parent 链只合并这些叶子的祖先（epoch 去重，
   每层合并条数留在显存，不做 D2H）；
2. **全量 refit**：整份顶点数组 H2D + 重写所有叶子盒 + 合并全部内部节点；
3. **整树重建**：`buildBVHGPU`，取 `--runs` 中位数。

计时都是 CUDA event（含 H2D），`--refit-shift 0.005`（场景包围盒对角线的 0.5%），
查询是 20,000 个同 seed 的随机点、`--runs` 轮中位数。验证：refit 后跑 `validateBVHGPU`
（内部盒逐位等于孩子并集、叶子盒等于三角形盒、根盒等于场景盒）并与 CPU 参考
逐点对拍。

**口径**：整组矩阵连续跑了 3 次（每次 5 个配置，进程间不重启机器），下表取
**每个单元格 3 次的中位数**。原因是进程之间、甚至同进程内的绝对时间会跳
1.5–3 倍（DVFS + 小 kernel 之间的主机间隙），单次会话的某个数不可复现；
配对倍数（同一 run 内 refit 与重建）才是稳定量。原始三次输出在
`out/refit_matrix.txt`（`out/` 不入库，用上面的命令可复现）。

| 三角形 | 运动 | 局部 1% refit | 全量 refit | 重建 | 局部 / 重建 | 全量 / 重建 | 查询 refit / 重建 |
|---|---|---|---|---|---|---|---|
| 8,192 | 随机 | 0.479 ms | 0.433 ms | 1.042 ms | **2.2×** | 2.9× | 1.23× |
| 131,072 | 随机 | 0.489 ms | 0.949 ms | 1.329 ms | **2.7×** | 1.6× | 1.58× |
| 131,072 | 相干 | 0.655 ms | 1.574 ms | 2.118 ms | **2.6×** | 1.5× | 1.09× |
| 2,097,152 | 随机 | 1.646 ms | 15.889 ms | 10.783 ms | **6.7×** | 0.69× | 4.32× |
| 2,097,152 | 相干 | 1.875 ms | 16.570 ms | 12.053 ms | **6.2×** | 0.73× | 1.38× |

分阶段数字（局部路径取同一组 3 次的中位数；`merge` 只扫脏叶子的祖先）：

```text
8k    随机  partial updated=82     upload=0.038 scatter=0.023 mark=0.045 merge=0.246 total=0.479 ms
131k  相干  partial updated=1311   upload=0.073 scatter=0.019 mark=0.029 merge=0.308 total=0.655 ms
2M    随机  partial updated=20972  upload=0.912 scatter=0.075 mark=0.174 merge=0.484 total=1.646 ms
2M    相干  partial updated=20972  upload=0.846 scatter=0.067 mark=0.174 merge=0.560 total=1.875 ms
2M    随机  refit   upload=12.236 ms / total=15.889 ms（H2D 约 77%，跨会话在 10–22 ms 之间跳）
2M    相干  refit   upload=12.879 ms / total=16.570 ms（同上）
```

同一批 run 里，全量路径的整树合并是 `tree=2.0–2.1 ms`（2M），局部路径的祖先合并
是 `merge=0.48–0.56 ms`——这是祖先合并唯一干净的对照（同 run、同一棵树）。

读法：

1. **全量 refit 在这个工程里是上传受限的。** 从 CPU 侧场景出发，每次都要把整份三角形
   数组推上显存（2M → 72 MB），H2D 在 10–22 ms 之间（页面内存 + 电源状态），比整树
   重建（10.8–12.1 ms）更贵。它的价值只在"数据本来就在显存上"的调用方；我们现在的
   bench 每轮都从 host 场景开始，所以 2M 上 0.69–0.73×。小场景上传小，才是 1.5–2.9×。
2. **局部 refit 才是可用的增量路径。** 只传 1% 的三角形与叶子槽号并只合并祖先，
   2M 上 1.6–1.9 ms 对 10.8–12.1 ms 重建（6.2–6.7×）；131k 2.6–2.7×；8k 只有 2.2×
   ——8k 的 0.4–0.5 ms 已经全是固定开销，任何"更快"的说法都进不了噪声。
3. **局部路径现在的瓶颈不是树合并。** 2M 相干：上传（含主机侧按场景索引从 72 MB
   数组里挑出 21k 个三角形 + 两次小 H2D）0.85 ms、标记祖先 0.17 ms、合并 0.56 ms。
   小场景里每层一次 launch 的固定开销 + 主机间隙开始主导。
4. **祖先合并省下的是合并阶段的规模**：同 run 对照，2M 全量合并 2.0–2.1 ms vs 局部
   0.48–0.56 ms（3.7–4.2×），而且只随脏叶子数与树高变化。实现上不把每层条数回传
   主机（那样每层要一次 30–60 µs 的 D2H），而是按 `min(脏叶子数, 该层节点数)` 发射、
   由 kernel 从显存读实际条数；第一版带 D2H 的方案在 131k 上是 0.421 ms，比全树
   合并的 0.284 ms 还慢。
5. **查询惩罚由运动相干性决定，不由 refit 次数决定。** 随机抖动把叶子盒撑大、
   逐层向上放宽：131k 平均节点测试 139.1（refit 树）vs 84.8（重建树），耗时 1.58×；
   2M 323.3 vs 98.6，耗时 3.5–4.4×，几乎全是无效节点。整体平移（相干）时盒只整体
   移动，排序大体仍有效：131k 1.09×、2M 1.38×。refit 的合理用法是高频、小幅、相干
   运动（布料、蒙皮、地形编辑），不是每帧全场景随机洗牌。
6. 所有 run 的对拍结果 `count_mismatch=0 tri_mismatch=0 dist2_mismatch=0`；
   131k 上另测 kNN(k=8) 与 radius(r=2.0) 在 refit 后逐位一致，连续两次局部 refit
   共享叶子/祖先（epoch 递增、同一批节点重新标记）也验证通过。
7. 注意 `partial` 的 1% 是"每 100 个取 1 个"的均匀采样，不是空间局部的一簇；真实
   编辑（一块地形、一个物体）通常更局部，脏祖先更少、合并更便宜。
8. 绝对时间只当量级看：三次重复里 8k 局部随机出现 0.37 / 0.48 / 2.12 ms（最后一个是
   明显离群），131k 相干重建出现 1.36 / 2.12 / 3.57 ms。表里的中位数掩盖不了这个量级的
   抖动，引用时必须带"同 run 配对 + 3 次中位数"的口径。

### 点云：GPU LBVH vs kd-tree（2026-10-29）

`gpu_cloud_bench.exe`。点云与三角形共用同一棵 LBVH、同一套 refit 和同一份 DFS，
只有叶子里的 `bounds` / `centroid` / `dist2` 三个函数不同（`src/gpu/prim_ops.h` 的
`TriOps` / `PointOps`）。下面这些数字与上面的三角形查询来自**同一个内核**。

**基线口径：Embree 没有点云图元。** Embree 4.4 的 `rtcPointQuery` 只对三角形网格
求最近点，Warp 也只有 `wp.mesh_query_point`，点云查询没有 Embree 数字可引用。
对照用两把 kd-tree：

- **本工程单线程 kd-tree**（`src/cpu/point_kd.cpp`：最长轴中位数切分、叶 8 个点、
  紧凑 AABB、`(dist2, 索引)` 字典序与 GPU 完全一致）——"记录在案"的分母；
- **scipy `cKDTree`**（leafsize 16）——第二个外部实现，读 `--dump` 写出的 QRCD
  二进制，看到的是与 GPU **逐字节相同**的 float32 输入。

引用倍数必须写明"对哪个 kd-tree"。Embree / Warp 在这里没有任何可比数字。

场景：sheet（x/z ∈ [−50, 50]，y = 地形高度 + U(−0.02, 0.02)），20,000 个随机查询点
（与三角形查询同一套 `makeQueryPoints`）。3 个规模 × 3 种查询 × 3 次重复，每格取
**3 次的中位数**；计时前空跑 3 s 拉 DVFS 时钟。原始输出在 `out/cloud_matrix.txt`。

构建：

| 点数 | LBVH 节点 | 深度 | GPU 构建 | kd-tree 建树 | GPU / kd |
|---|---|---|---|---|---|
| 131,072 | 262,143 | 28 | 1.58 ms | 21–22 ms | 14× |
| 1,048,576 | 2,097,151 | 32 | 5.19 ms | 200–210 ms | 39× |
| 10,000,000 | 19,999,999 | 36 | 49.96 ms | 2.4–2.5 s | 49× |

10M 的 GPU 分阶段：morton 1.9 / 排序 11.4 / 散射 5.2 / 建树 7.1 / refit 24.2 ms。
refit 仍占一半——点的叶子盒就是一个点，但内部节点合并的次数与三角形场景相同，
没有可以摊掉的部分。kd 建树是单线程中位数切分 + 一次 `nth_element` 重排，
规模越大差距越大。

查询（GPU 时间为 CUDA event，结果常驻显存，不含 H2D/D2H；括号内是对同进程单线程
kd 的倍数；默认半径 = 包围盒对角线 1% = 1.41656）：

| 点数 | closest | kNN k=8 | radius | 结果数/查询（radius） |
|---|---|---|---|---|
| 131,072 | 0.64 ms (32.7×) | 1.90 ms (17.8×) | 1.10 ms (14.5×) | 11.2 |
| 1,048,576 | 1.81 ms (27.8×) | 3.78 ms (20.3×) | 13.22 ms (8.9×) | 89.8 |
| 10,000,000 | 5.81 ms (52.1×) | 8.07 ms (49.3×) | 65.54 ms (27.9×) | 855.5 |

每查询的平均节点 / 点测试（`--stats`，每格一轮）：

| 点数 | closest | kNN k=8 | radius |
|---|---|---|---|
| 131,072 | 110.0 / 10.7 | 186.7 / 44.4 | 45.3 / 11.2 |
| 1,048,576 | 198.4 / 15.5 | 315.7 / 65.0 | 222.6 / 89.8 |
| 10,000,000 | 392.7 / 21.3 | 616.2 / 100.2 | 1,827.9 / 855.5 |

读法：

1. **closest 是最纯粹的"延迟受限"负载。** 每查询只碰 110–393 个节点、11–21 个点，
   单线程一次 1.0–15.1 µs，几乎全是冷缓存行。树越深，kd 单次下潜越贵；GPU 同时挂
   2 万条查询把延迟藏掉，倍数从 33× 涨到 52×。
2. **kNN k=8 的节点测试是 closest 的 1.6–1.7 倍，GPU 时间只多 1.6–3.0 倍**：第 k 近
   阈值在 GPU 上同样是并行的，剪枝多出来的访问不额外收钱。
3. **radius 的成本由输出条数主导。** 默认半径下 10M 每查询吐 855 条，GPU 只有
   0.31 Mq/s；27.9× 是表面上的最好数字，实际比的是"写 1700 万条结果"。

半径不能跨规模直接比：查询点均匀撒在 3D 包围盒里、点却在 2D 地形面上，输出条数
≈ C·n·r³，所以"同样的结果数"要按 r ∝ n^(−1/3) 缩半径。把输出拉回约 11 条/查询后：

| 点数 | 半径 | 结果/查询 | GPU | kd 1T | 倍数 |
|---|---|---|---|---|---|
| 1,048,576 | 0.708 | 11.4 | 1.36 ms | 18.13 ms | 13.7× |
| 10,000,000 | 0.334 | 11.3 | 5.25 ms | 20.38 ms | 3.9× |

同输出量下 10M 从 27.9× 掉到 3.9×：GPU 时间主要是两遍遍历 + 前缀和 + 段内排序的
固定成本，与输出量几乎无关；kd 的时间随点密度增长。**报 radius 倍数必须带
results/query**。原始输出 `out/cloud_radius_scaled.txt`。

scipy `cKDTree`（同一批浮点数；只计 `tree.query` / `query_ball_point` 本身，float32
距离重算与排序都在计时窗口外）：

| 点数 | 数据 | 查询 | scipy | 本工程 kd | GPU | GPU / scipy | scipy / kd |
|---|---|---|---|---|---|---|---|
| 131,072 | sheet | closest | 53.41 ms | 20.93 ms | 0.64 ms | 84× | 2.6× |
| 131,072 | sheet | kNN k=8 | 62.69 ms | 33.83 ms | 1.90 ms | 33× | 1.9× |
| 1,048,576 | sheet | closest | 747.99 ms | 50.39 ms | 1.81 ms | 413× | 14.8× |
| 1,048,576 | sheet | kNN k=8 | 679.56 ms | 76.61 ms | 3.78 ms | 180× | 8.9× |
| 10,000,000 | sheet | closest | 22,750.12 ms | 302.68 ms | 5.81 ms | 3,916× | 75.2× |
| 10,000,000 | sheet | kNN k=8 | 22,765.99 ms | 397.98 ms | 8.07 ms | 2,821× | 57.2× |
| 1,048,576 | sheet | radius 1.4166 | 334.89 ms | 117.62 ms | 13.22 ms | 25× | 2.8× |
| 1,048,576 | volume | closest | 32.11 ms | 19.35 ms | 0.32 ms | 100× | 1.7× |
| 10,000,000 | volume | closest | 87.72 ms | 35.48 ms | 0.62 ms | 142× | 2.5× |

（sheet 的 kd/GPU 取自 `out/cloud_matrix.txt`；volume 的 kd/GPU 是 3 次配对重复的
中位数，`out/cloud_volume_reps.txt`，该会话 1M/10M 对 kd 是 60.4×/55.5×——跨会话
热状态会把它拉到 62.6×/75.2×，绝对值按同一会话内配对读。scipy 建树 10M 需要
9.1–9.7 s，本工程 kd 是 2.4–2.5 s，两次都比 GPU 的 50 ms 慢两个数量级。）

- **sheet 数据上 scipy 会退化，而且退化随密度加深**：131k 只慢 2.6×，1M 慢 15×，
  10M 慢 75×（单次查询 1.14 ms）。用 numpy 暴力法抽查过随机查询，它选中的点与我们
  **完全相同**——是结果正确、路径变慢（查询点落在 2D 流形附近时，它的剪枝收益
  随树深衰减）。
- **换成体积点云（`--kind volume`）后差距恢复正常**：scipy 只比自家 kd 慢 1.7–2.5×。
  也就是说 headline 用自家 kd 当分母是保守的，scipy 只是第二个旁证；**不要拿
  3–4 个数量级去对外宣传**，那是 scipy 在这份数据上的病态。
- 指纹：closest 在五个格子上 scipy 的 `sum_dist2` 与 hash 与 kd/GPU **逐位相同**
  （sheet `0x31bca88d…` / `0x94d204dd…` / `0x2a8fce89…`；volume `0xb948e15d…` /
  `0xf9f23e87…`）；kNN 的 `sum_dist2` 相对差 6e−13（scipy 用 float64 距离选点，
  近并列的取法不同）；radius 在**精确 float32 半径**下集合、段内顺序、`sum_dist2`
  与 hash 全部相同（`0x27bda28ba8bb7682`）。
- **radius 曾报"hash 不同"，是两个 bug 叠出来的假象**：一是命令行传了四舍五入的
  1.41656（真值 1.4165570735931396），球面正好跨过 9 个点的距离；二是旧版
  `gpu_cloud_bench` 的指纹对列表查询漏了"先混入长度"，与 `fingerprint.h` 的约定
  不一致。两个都修掉后才逐位一致——**并列取法本身没有问题**。
- 原始输出 `out/scipy_cloud.txt`。

增量更新（1% 相干位移，`--refit-dirty 0.01 --refit-motion coherent`，3 次中位数）：

| 点数 | 局部 refit | 全量 refit | 重建 | 局部 / 重建 | 全量 / 重建 |
|---|---|---|---|---|---|
| 1,048,576 | 0.86 ms | 3.27 ms | 5.54 ms | **6.4×** | 1.7× |
| 10,000,000 | 8.88 ms | 38.02 ms | 50.14 ms | **5.6×** | 1.3× |

10M 局部路径的分阶段（已扣除 1% = 10 万个脏点）：上传 5.3–6.7 / 标记祖先 1.2 /
合并 1.6–1.8 ms。**全量 refit 在点云上反而比重建快**，与三角形 2M 上的 0.69–0.73×
相反：点云重建必须重跑 Morton + 基数排序 + 建树（10M 合计约 50 ms），而上传
120 MB 点 + 重写叶子 + 合并只要 38 ms。结论不是"全量 refit 好用"，而是**点云重建的
主成本是排序而不是上传**；数据本来就在显存上时，局部 refit 仍是唯一划算的路径。
原始输出 `out/cloud_refit.txt`。

正确性：9 格 × 3 次重复全部 `id_mismatch=0 dist_mismatch=0 count_mismatch=0`，
CPU kd 与 GPU 的 closest / kNN / radius 指纹逐位相同；refit 后结构校验 PASS
（内部盒逐位等于子节点并集、叶子盒等于点、根盒等于点云盒），且与重建树的 kd
结果逐点对拍一致。

两个只在 10M 暴露的问题（详见 `Quarry-学习路径/08-点云.md`）：

1. **kd 建树在递归中持有 `nodes` 引用会悬空。** `push_back` 扩容后，外层若还用
   引用写回就是 use-after-free——10M 必崩，131k/1M 偶尔不崩。改成递归返回后按
   下标写回。
2. **closest 只比距离时，等距并列由访问顺序决定。** 与暴力法/GPU 的
   `(dist2, 索引)` 字典序不一致，10M 上出现 1 个 id 不一致。改成 `closerHit` 后
   三方逐位一致。

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
8. **GPU 的倍数取决于射线相干性和口径，不是无条件的 3×。** 10-27 输入序：1.66× /
   0.74× / 1.60× Embree 32T；主射线 Morton 序重排并计入排序开销后：3.17× / 1.98× /
   6.37×。10-28 同场复测是 1.58× / 2.03× / 2.40× 与 3.40× / 4.84× / 8.66×。同一套
   排序用在非相干次生射线上反而亏 2 ms，只对主射线开启。对外表述必须同时带场景
   大小、射线类型、是否含排序和测量日期。
9. **GPU 构建已经全面领先**（1.3×/1.9×/2.8× Embree，10-27 同场口径）；refit 重写
   后同场 A/B 再快 2.1–2.7×（2M 6.2→2.7 ms），整树构建约 1.5×。
10. **GPU 点查询把延迟受限的负载改写成吞吐受限**：closest 7.6× / 4.2× / 3.0×
    Embree 32T（随机 1M 16.3×），kNN 与 radius 相对 CPU 参考是 6–21×。三者共用
    "每线程一个显式栈"的骨架，差别只在剪枝阈值；radius 的代价是多一遍遍历和 CSR
    布局，短段的段内排序必须用 warp 双调，CUB 分段基数排序在两万个短段上是负结果。
11. **Warp 1.6.2 是 closest 的第二个外部基线，也是最强的那个**：地形场景 8k 持平、
    131k 快 1.4×、2M 快 1.2×，随机三角形汤我们快 1.15×；构建则是小场景 Warp 快
    1.4–4×、2M 我们快 1.8×。我们每叶 1 个三角形（2M → 419 万节点、深度 31），
    Warp 是 SAH + 每叶 4 个图元。把 Morton 相邻三角形批量成叶子已实测是负优化
    （射线 2.2× 慢、kNN 1.8× 慢），差距来自构建器的树质量，不是叶子粒度本身；
    要继续追，应该上 GPU 自顶向下 SAH（或 BVH4 宽节点），而不是改叶子大小。
    Warp 的结果是 float32 精度（同位但末位不同），我们是与 CPU 参考逐位相同。
12. **点云把"延迟受限"推到极端。** closest 每查询只做 110–393 次节点测试，GPU 对
    同进程单线程 kd 是 32× / 27× / 55×——倍数随规模**上升**，因为 kd 每次下潜更贵
    而 GPU 的并发不变。kNN k=8 是 17.8–49.3×，radius（默认半径、输出 11–855 条）
    是 8.9–27.9×。
13. **radius 的倍数由输出量支配，报数必须带 results/query。** 同输出量（约 11 条/
    查询）下 1M 13.7×、10M 3.9×：GPU 时间几乎只随查询数走，kd 随点密度增长。
    点云查询**没有 Embree 基线**（Embree / Warp 都没有点云图元），分母是自写 kd 与
    scipy `cKDTree`；scipy 在 2D 流形点云上退化，自家 kd 在所有格子上更快，因此
    headline 倍数偏保守。
14. **点云全量 refit 比重建快（1.7× / 1.3×），与三角形 2M 的 0.69–0.73× 相反。**
    点云重建的主成本是 Morton + 基数排序而不是上传；局部 refit 仍是首选
    （5.6–6.4×）。两个 M2 bug（kd 递归中的悬空引用、等距并列的取法）都只在 10M
    暴露——规模本身就是一项测试。


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

build\gpu_query_bench.exe --grid 64   --points 20000 --runs 5 --stats
build\gpu_query_bench.exe --grid 256  --points 20000 --runs 5 --stats
build\gpu_query_bench.exe --grid 1024 --points 20000 --runs 5 --stats
build\gpu_query_bench.exe --grid 256  --points 20000 --query knn --k 16 --runs 3 --stats
build\gpu_query_bench.exe --grid 1024 --points 20000 --query knn --k 16 --runs 3 --stats
build\gpu_query_bench.exe --grid 256  --points 20000 --query radius --radius 2.0 --runs 3 --stats
build\gpu_query_bench.exe --grid 1024 --points 20000 --query radius --radius 0.5 --runs 3 --stats

build\gpu_cloud_bench.exe --points 131072   --queries 20000
build\gpu_cloud_bench.exe --points 1048576  --queries 20000 --query knn --k 8
build\gpu_cloud_bench.exe --points 10000000 --queries 20000 --query radius
build\gpu_cloud_bench.exe --kind volume --points 1048576 --query radius --radius 1.0
build\gpu_cloud_bench.exe --points 1048576 --refit --refit-dirty 0.01 --refit-motion coherent

build\gpu_cloud_bench.exe --points 1048576 --queries 20000 --dump out\cloud_1048576.bin
build\gpu_cloud_bench.exe --points 10000000 --queries 20000 --dump out\cloud_10000000.bin
build\gpu_cloud_bench.exe --kind volume --points 1048576  --queries 20000 --dump out\vol_1048576.bin
build\gpu_cloud_bench.exe --kind volume --points 10000000 --queries 20000 --dump out\vol_10000000.bin
py -3.13 bench\scipy_kd_bench.py out\cloud_1048576.bin --query closest
py -3.13 bench\scipy_kd_bench.py out\cloud_10000000.bin --query knn --k 8
py -3.13 bench\scipy_kd_bench.py out\cloud_1048576.bin --query radius --radius 1.4165570735931396
```

## 已知缺口

- 只有两种几何（地形网格、随机三角形），缺非均匀模型（Stanford Dragon / Lucy）
- 命中率只有 100%（垂直向下）和约 31%（远处随机朝向），缺多层遮挡
- 只有一个 GPU 型号，每瓦数据要等 M1
- GPU 侧已覆盖射线求交与 closest / kNN / radius；kNN 的 k 上限 32（每线程定长表），
  radius 每条查询的结果数不设上限但长段会退回 CUB 分段排序
- 射线排序已完成（原点/方向 Morton 序），但只对相干主射线有效；非相干射线需要
  "先粗遍历取命中点、再按命中点排序"的两遍方案，未做
- 内联子节点包围盒（64B 布局）实测是负结果，已回退；131k 的差距靠射线排序解决
- 外部 GPU 对照：射线侧只有 CPU 的 Embree，Warp `bvh.ray_trace` 对照待 M1
- 点查询外部对照只有 closest（Embree `rtcPointQuery` 与 Warp `mesh_query_point`）；
  kNN / radius 在 Embree 和 Warp 里都没有原生入口
- 点云查询**没有 Embree / Warp 基线**（两者都没有点云图元）。分母是自写中位数切分
  kd-tree 与 scipy `cKDTree`；后者在 2D 流形点云（sheet）上退化，只有体积点云上是
  正常对照
- 点云场景只有 sheet / volume 两种解析分布，缺真实扫描数据（LiDAR、重建网格采样）
- Morton 叶子批量（4–8 图元/叶）实测负结果，已回退到每叶 1 个；追 Warp 需要
  GPU 自顶向下 SAH 或宽节点，不在 M1 范围
- 第一轮结果（不同场景、20 万条射线）已作废，勿引用
