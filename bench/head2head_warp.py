"""Quarry 与 NVIDIA Warp 的同场景、同查询集对照。

场景与查询集都由 quarry 生成；Warp 用同一份顶点/索引建 mesh，再跑同一个
内核。两边都只计 kernel 时间（各自预热后 5 轮中位数），构建时间单独列出。
--query closest 比最近点，--query ray 比射线求交（Warp 的 cuBQL 后端只支持射线）。

    py -3.13 -m pip install warp-lang==1.13.0
    py -3.13 bench/head2head_warp.py --grid 256 --points 20000
    py -3.13 bench/head2head_warp.py --grid 1024 --points 20000 --backend sah
    py -3.13 bench/head2head_warp.py --grid 256 --points 20000 --query ray --backend cubql
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build" / "python"))
import quarry  # noqa: E402

import warp as wp  # noqa: E402


@wp.kernel
def closest_kernel(
    mesh_id: wp.uint64,
    pts: wp.array(dtype=wp.vec3),
    dist2: wp.array(dtype=wp.float32),
    face: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh_id, pts[i], 1.0e30)
    if q.result:
        p = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        dist2[i] = wp.length_sq(pts[i] - p)
        face[i] = q.face
    else:
        dist2[i] = 1.0e30
        face[i] = -1


@wp.kernel
def ray_kernel(
    mesh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    dirs: wp.array(dtype=wp.vec3),
    t_out: wp.array(dtype=wp.float32),
    hit: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh_id, origins[i], dirs[i], 1.0e30)
    if q.result:
        t_out[i] = q.t
        hit[i] = 1
    else:
        t_out[i] = 1.0e30
        hit[i] = 0


def timed_warp_query(mesh, kernel, n, inputs, outputs, runs):
    wp.launch(kernel, dim=n, inputs=[mesh.id, *inputs], outputs=outputs)
    wp.synchronize()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        wp.launch(kernel, dim=n, inputs=[mesh.id, *inputs], outputs=outputs)
        wp.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--build-runs", type=int, default=3)
    ap.add_argument("--backend", default="sah",
                    choices=["sah", "median", "lbvh", "cubql"])
    ap.add_argument("--query", default="closest", choices=["closest", "ray"])
    args = ap.parse_args()
    if args.query != "ray" and args.backend == "cubql":
        ap.error("Warp's cuBQL backend only supports ray queries "
                 "(see warp/_src/types.py, bvh_constructor)")

    wp.init()
    cubql_ok = getattr(wp, "is_cubql_available", lambda: "n/a")()
    print(f"warp={wp.__version__} backend={args.backend} query={args.query} "
          f"cubql_available={cubql_ok}")

    vertices, indices = quarry.demo_mesh(grid=args.grid, seed=args.seed)
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    quarry.Scene(vertices, indices)  # 第一次构建付 CUDA 模块加载
    scene = quarry.Scene(vertices, indices)
    flat_indices = indices.reshape(-1)

    mb = scene.bounds
    lo = np.array(mb["min"], dtype=np.float32)
    hi = np.array(mb["max"], dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    origins = (lo + (hi - lo) * rng.random((args.points, 3))).astype(np.float32)
    dirs = rng.normal(size=(args.points, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    # closest 用查询点本身；ray 用 origin + dir
    pts = origins

    if args.query == "closest":
        quarry_run = lambda: scene.closest(pts)  # noqa: E731
    else:
        quarry_run = lambda: scene.raycast(origins, dirs)  # noqa: E731

    quarry_times = []
    for _ in range(args.runs):
        quarry_run()
        quarry_times.append(scene.last_query_ms)
    quarry_med = statistics.median(quarry_times)
    q_res = quarry_run()

    w_verts = wp.array(vertices, dtype=wp.vec3)
    w_tris = wp.array(flat_indices, dtype=wp.int32)
    w_pts = wp.array(pts, dtype=wp.vec3)
    w_origins = wp.array(origins, dtype=wp.vec3)
    w_dirs = wp.array(dirs, dtype=wp.vec3)
    dist2 = wp.zeros(args.points, dtype=wp.float32)
    face = wp.zeros(args.points, dtype=wp.int32)
    t_out = wp.zeros(args.points, dtype=wp.float32)
    hit = wp.zeros(args.points, dtype=wp.int32)

    build_times = []
    mesh = None
    for _ in range(args.build_runs):
        start = time.perf_counter()
        mesh = wp.Mesh(points=w_verts, indices=w_tris,
                       bvh_constructor=args.backend)
        wp.synchronize()
        build_times.append((time.perf_counter() - start) * 1e3)

    if args.query == "closest":
        warp_med = timed_warp_query(mesh, closest_kernel, args.points,
                                    [w_pts], [dist2, face], args.runs)
        q_dist2, _ = q_res
        valid = face.numpy() >= 0
        dq = np.sqrt(q_dist2[valid])
        dw = np.sqrt(dist2.numpy()[valid])
        max_err = float(np.max(np.abs(dq - dw))) if valid.any() else float("nan")
        mismatched = int(np.sum(np.abs(dq - dw) > 1e-3))
        missing = int((~valid).sum())
    else:
        warp_med = timed_warp_query(mesh, ray_kernel, args.points,
                                    [w_origins, w_dirs], [t_out, hit], args.runs)
        q_hit, q_t = q_res
        w_hit = hit.numpy() > 0
        both = q_hit & w_hit
        diff = np.abs(q_t[both] - t_out.numpy()[both])
        max_err = float(diff.max()) if diff.size else float("nan")
        mismatched = int(np.sum(diff > 1e-3))
        missing = int((q_hit != w_hit).sum())
        valid = both

    print(f"tris={scene.triangle_count} queries={args.points} "
          f"depth={scene.depth}")
    print(f"build   quarry={scene.build_ms:.2f} ms  "
          f"warp={statistics.median(build_times):.2f} ms "
          f"({args.build_runs} runs)")
    print(f"query   quarry={quarry_med:.3f} ms  warp={warp_med:.3f} ms  "
          f"quarry/warp={quarry_med / warp_med:.2f}x")
    print(f"check   hit_mismatch={missing}  t_err>1e-3={mismatched}  "
          f"max_err={max_err:.3e}")


if __name__ == "__main__":
    main()
