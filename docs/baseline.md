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
```

## 已知缺口

- 只有两种几何（地形网格、随机三角形），缺非均匀模型（Stanford Dragon / Lucy）
- 命中率只有 100%（垂直向下）和约 31%（远处随机朝向），缺多层遮挡
- 只有一个 GPU 型号，每瓦数据要等 M1
- 查询只有内部对照（暴力法），缺外部对照：Embree `rtcPointQuery`（CPU）、Warp `bvh.query_closest`（GPU）
- 第一轮结果（不同场景、20 万条射线）已作废，勿引用
