"""scipy.spatial.cKDTree 对照：读 gpu_cloud_bench --dump 写出的 QRCD 文件，
在**同一批浮点数**上构建中位数切分 kd-tree，跑 closest / kNN / 半径。

为什么需要它：Embree 4.4 没有点云图元，Warp 也只有网格最近点，点云查询没有
"Embree 数字"可直接引用。项目自带的 kd-tree 是参考实现（正确性基线），
scipy 的 cKDTree 是一个被广泛部署的外部基线——两个数一起报，才不算只跟自己比。

用法：
    build\\gpu_cloud_bench.exe --points 1048576 --queries 20000 --dump out\\cloud.bin
    py -3.13 bench/scipy_kd_bench.py out/cloud.bin --query closest
    py -3.13 bench/scipy_kd_bench.py out/cloud.bin --query knn --k 8
    py -3.13 bench/scipy_kd_bench.py out/cloud.bin --query radius --radius 1.41656
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import cKDTree

MAGIC = b"QRCD"


def load_qrcd(path):
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise SystemExit(f"{path}: not a QRCD file")
    version, np_, nq = struct.unpack("<IQQ", raw[4:24])
    if version != 1:
        raise SystemExit(f"{path}: unsupported version {version}")
    body = np.frombuffer(raw, dtype="<f4", offset=24, count=3 * (np_ + nq))
    pts = body[: 3 * np_].reshape(np_, 3)
    qry = body[3 * np_ :].reshape(nq, 3)
    return pts, qry


def dist2_f32(q, p):
    """与 GPU 相同的 float32 计算顺序：((dx*dx + dy*dy) + dz*dz)。"""
    d = q - p
    return (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]


def fp_single(dist2, idx):
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    for d, i in zip(dist2.view(np.uint32), idx):
        for v in ((1 << 32) | int(d), int(i) + 1):
            h = ((h ^ v) * prime) % (1 << 64)
    return h


def fp_list(all_d2, all_idx):
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    for d2, idx in zip(all_d2, all_idx):
        h = ((h ^ len(idx)) * prime) % (1 << 64)
        for d, i in zip(d2.view(np.uint32), idx):
            for v in (int(d), int(i) + 1):
                h = ((h ^ v) * prime) % (1 << 64)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--query", default="closest", choices=["closest", "knn", "radius"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--radius", type=float, default=1.41656)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--leafsize", type=int, default=16)
    args = ap.parse_args()

    pts, qry = load_qrcd(args.file)
    print(f"scipy {scipy.__version__} numpy {np.__version__} cKDTree points={len(pts)} "
          f"queries={len(qry)} type={args.query} leafsize={args.leafsize}")

    t0 = time.perf_counter()
    tree = cKDTree(pts, leafsize=args.leafsize)
    build_ms = (time.perf_counter() - t0) * 1e3
    print(f"scipy kd build_ms={build_ms:.1f} nodes={tree.n}")

    times = []
    out = None
    for _ in range(args.runs):
        # 只给 kd-tree 的查询本身计时；把 float32 平方距离与排序留在窗外，
        # 否则 Python 逐点循环会盖过树的耗时（20k 次 numpy 调用 ~0.1 s）。
        t0 = time.perf_counter()
        if args.query == "closest":
            _, idx = tree.query(qry, k=1)
        elif args.query == "knn":
            _, idx = tree.query(qry, k=args.k)
        else:
            lists = tree.query_ball_point(qry, r=args.radius)
        times.append((time.perf_counter() - t0) * 1e3)

        if args.query == "closest":
            idx = np.asarray(idx, dtype=np.int64)
            d2 = np.array([dist2_f32(qry[i], pts[idx[i]][None])[0] for i in range(len(qry))],
                          dtype=np.float32)
            out = (d2, idx)
        elif args.query == "knn":
            idx = np.asarray(idx, dtype=np.int64)
            d2 = np.empty((len(qry), args.k), dtype=np.float32)
            for i in range(len(qry)):
                d2[i] = dist2_f32(qry[i], pts[idx[i]])
            # 与 C++ 的 closerHit 同序：主键 float32 平方距离，并列按点索引。
            order = np.lexsort((idx, d2))
            rows = np.arange(len(qry))[:, None]
            out = (d2[rows, order], idx[rows, order])
        else:
            d2s, ids = [], []
            for i, lst in enumerate(lists):
                idx = np.asarray(lst, dtype=np.int64)
                d2 = dist2_f32(qry[i], pts[idx])
                order = np.lexsort((idx, d2))
                d2s.append(d2[order])
                ids.append(idx[order])
            out = (d2s, ids)

    times.sort()
    med = times[len(times) // 2]
    print(f"scipy runs={args.runs} median_ms={med:.2f} min_ms={times[0]:.2f} "
          f"max_ms={times[-1]:.2f} Mqueries_s={len(qry) / (med * 1e3):.2f}")

    if args.query == "closest":
        d2, idx = out
        print(f"fingerprint scipy queries={len(d2)} results={len(d2)} "
              f"sum_dist2={float(np.sum(d2, dtype=np.float64)):.6f} hash=0x{fp_single(d2, idx):016x}")
    else:
        d2s, ids = out
        total = sum(len(x) for x in ids)
        sd = sum(float(np.sum(d, dtype=np.float64)) for d in d2s)
        print(f"fingerprint scipy queries={len(ids)} results={total} "
              f"sum_dist2={sd:.6f} hash=0x{fp_list(d2s, ids):016x}")


if __name__ == "__main__":
    sys.exit(main())
