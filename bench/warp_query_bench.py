"""Warp baseline for Quarry's closest-point query.

Reads the binary scene dump written by ``gpu_query_bench --dump`` so both engines
see exactly the same triangles and query points. Warp builds its own BVH on the
device; we time ``mesh_query_point_no_sign`` over the same points and verify a
random subset against a float64 brute force.

Usage:
    py -3.13 bench/warp_query_bench.py --dump build/qry_131k.bin --runs 5
"""

import argparse
import struct
import time

import numpy as np
import warp as wp

MAGIC = 0x44595251  # "QRYD"


def load_dump(path):
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != MAGIC or version != 1:
            raise ValueError(f"{path}: not a quarry scene dump")
        n_tris, n_points = struct.unpack("<QQ", f.read(16))
        tris = np.fromfile(f, dtype=np.float32, count=n_tris * 9).reshape(n_tris, 3, 3)
        points = np.fromfile(f, dtype=np.float32, count=n_points * 3).reshape(n_points, 3)
    return tris, points


def closest_on_triangles(p, tris):
    """Brute-force squared distance and index, float64, vectorized over triangles.

    Segment distances plus an interior (plane) test -- deliberately a different
    formulation from the engine's region-clamped barycentric solution, so the two
    agree only if both are correct.
    """
    a = tris[:, 0].astype(np.float64)
    b = tris[:, 1].astype(np.float64)
    c = tris[:, 2].astype(np.float64)
    q = p.astype(np.float64)

    def seg_dist2(s0, s1):
        e = s1 - s0
        denom = np.einsum("ij,ij->i", e, e)
        denom = np.where(denom > 0.0, denom, 1.0)
        t = np.einsum("ij,ij->i", q - s0, e) / denom
        t = np.clip(t, 0.0, 1.0)
        d = q - (s0 + e * t[:, None])
        return np.einsum("ij,ij->i", d, d)

    out = np.minimum(np.minimum(seg_dist2(a, b), seg_dist2(b, c)), seg_dist2(c, a))

    n = np.cross(b - a, c - a)
    n2 = np.einsum("ij,ij->i", n, n)
    n2 = np.where(n2 > 0.0, n2, 1.0)
    plane2 = np.einsum("ij,ij->i", q - a, n) ** 2 / n2

    inside = (
        (np.einsum("ij,ij->i", np.cross(b - a, q - a), n) >= 0.0)
        & (np.einsum("ij,ij->i", np.cross(c - b, q - b), n) >= 0.0)
        & (np.einsum("ij,ij->i", np.cross(a - c, q - c), n) >= 0.0)
    )
    out = np.where(inside, np.minimum(out, plane2), out)
    best = int(np.argmin(out))
    return out[best], best


@wp.kernel
def closest_kernel(
    mesh: wp.uint64,
    verts: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=wp.vec3),
    dist2: wp.array(dtype=float),
    face_out: wp.array(dtype=int),
    max_dist: float,
):
    i = wp.tid()
    # Warp 1.6 returns face + barycentrics, not a distance. Its (u, v) are the
    # weights of the first two vertices (closest_point_to_triangle), so the
    # closest point is u*a + v*b + (1-u-v)*c.
    q = wp.mesh_query_point_no_sign(mesh, points[i], max_dist)
    if q.result:
        a = verts[3 * q.face]
        b = verts[3 * q.face + 1]
        c = verts[3 * q.face + 2]
        cp = q.u * a + q.v * b + (1.0 - q.u - q.v) * c
        d = points[i] - cp
        dist2[i] = wp.dot(d, d)
        face_out[i] = q.face
    else:
        dist2[i] = -1.0
        face_out[i] = -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--launch-per-sample", type=int, default=10)
    ap.add_argument("--check", type=int, default=200, help="brute-force check count")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-dist", type=float, default=1.0e6)
    ap.add_argument("--warmup-ms", type=float, default=3000.0)
    args = ap.parse_args()

    tris, points = load_dump(args.dump)
    n_tris = len(tris)
    n_points = len(points)
    print(f"dump={args.dump} tris={n_tris} points={n_points}", flush=True)

    wp.init()
    dev = wp.get_device(args.device)

    verts = wp.array(tris.reshape(-1, 3), dtype=wp.vec3, device=dev)
    indices = wp.array(np.arange(n_tris * 3, dtype=np.int32), dtype=int, device=dev)

    # 进程里第一次 wp.Mesh 会带上模块/内核的一次性初始化（实测 8k 网格 14-60 ms，
    # 之后同尺寸只要 0.4 ms）。先用一个三角形热身，再计真正的那次构建。
    tiny_v = wp.array(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                      dtype=wp.vec3, device=dev)
    wp.Mesh(points=tiny_v, indices=wp.array(np.arange(3, dtype=np.int32), dtype=int, device=dev))
    wp.synchronize()

    # Mesh construction includes the device-side BVH build; time it separately.
    t0 = time.perf_counter()
    mesh = wp.Mesh(points=verts, indices=indices)
    wp.synchronize()
    build_ms = (time.perf_counter() - t0) * 1e3
    print(f"warp build_ms={build_ms:.2f}", flush=True)

    d_points = wp.array(points, dtype=wp.vec3, device=dev)
    d_dist2 = wp.zeros(n_points, dtype=float, device=dev)
    d_face = wp.zeros(n_points, dtype=int, device=dev)

    def launch():
        wp.launch(
            closest_kernel,
            dim=n_points,
            inputs=[mesh.id, verts, d_points, d_dist2, d_face, args.max_dist],
            device=dev,
        )

    # DVFS: keep the GPU busy long enough for clocks to settle before timing,
    # same treatment as gpu_query_bench.
    warm_launches = 0
    wp.synchronize()
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < args.warmup_ms:
        launch()
        warm_launches += 1
    wp.synchronize()
    print(f"warp warmup_ms={args.warmup_ms:.0f} launches={warm_launches}", flush=True)

    times = []
    for _ in range(args.runs):
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.launch_per_sample):
            launch()
        wp.synchronize()
        times.append((time.perf_counter() - t0) * 1e3 / args.launch_per_sample)

    times.sort()
    med = times[len(times) // 2]
    print(
        f"warp runs={args.runs}x{args.launch_per_sample} median_ms={med:.3f} "
        f"min_ms={times[0]:.3f} Mqueries_s={n_points / med / 1e3:.2f}",
        flush=True,
    )

    dist2 = d_dist2.numpy().astype(np.float64)
    face = d_face.numpy()
    if np.any(face < 0):
        print(f"warp points without a hit: {int(np.count_nonzero(face < 0))}")
    print(f"sum_dist2={float(np.sum(dist2)):.6f}", flush=True)

    rng = np.random.default_rng(1)
    idx = rng.choice(n_points, size=min(args.check, n_points), replace=False)
    bad = 0
    id_diff = 0
    worst = 0.0
    for n, i in enumerate(idx):
        best2, best_tri = closest_on_triangles(points[i], tris)
        excess = dist2[i] - best2
        worst = max(worst, excess)
        # Warp 用 float32 重新算出最近点，d² 的末位就会有 1e-7 级噪声；查询点几乎
        # 落在三角形上时，这个绝对误差会让相对误差爆到 1e-3。判定只认"更远"：
        # 绝对超出 1e-5（或相对 1e-5，取更宽的那个），同时单独统计选面不同。
        if excess > 1e-5 * max(1.0, best2):
            bad += 1
        if best_tri != int(face[i]):
            id_diff += 1
        if n < 5:
            print(
                f"  [{i}] warp_face={int(face[i])} d2={dist2[i]:.6f} "
                f"brute_face={best_tri} d2={best2:.6f} p={points[i]}"
            )
    print(
        f"check points={len(idx)} suboptimal={bad} id_diff={id_diff} "
        f"worst_abs_excess={worst:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
