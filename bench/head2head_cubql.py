"""Quarry 与 cuBQL（NVIDIA 官方仓库）的同场景、同查询集对照。

cuBQL 侧走 out/cubql_bench.exe（原生 cuBQL 构建 + 遍历），quarry 侧用同一份
顶点/三角形与同一批查询。两边都只计 kernel 时间（CUDA events、各自取中位数）。

前置：
    out/build_cubql.bat          # 编译 out/cubql_bench.exe
    py -3.13 bench/head2head_cubql.py --grid 256 1024
"""

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "build" / "python"))

import export_cubql_scene as export  # noqa: E402
import quarry  # noqa: E402


def quarry_timings(scene, origins, dirs, runs):
    c_times, r_times = [], []
    for _ in range(runs):
        scene.closest(origins)
        c_times.append(scene.last_query_ms)
        scene.raycast(origins, dirs)
        r_times.append(scene.last_query_ms)
    d2, tri = scene.closest(origins)
    hit, t = scene.raycast(origins, dirs)
    checks = {
        "closest": (int((tri >= 0).sum()), float(np.sqrt(d2).sum())),
        "ray": (int(hit.sum()), float(t[hit].sum(dtype=np.float64))),
    }
    return statistics.median(c_times), statistics.median(r_times), checks


def run_cubql(binary, scene_path, mode, sah):
    cmd = [str(binary), str(scene_path), mode] + (["--sah"] if sah else [])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    build_ms = query_ms = None
    hits = sum_ = None
    for line in out.splitlines():
        if line.startswith("cubql_build_ms="):
            parts = dict(p.split("=") for p in line.split())
            build_ms = float(parts["cubql_build_ms"])
            query_ms = float(parts["cubql_query_ms"])
        elif line.startswith("hits="):
            parts = dict(p.split("=") for p in line.split())
            hits, sum_ = int(parts["hits"]), float(parts["sum"])
    return build_ms, query_ms, (hits, sum_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--binary", default=str(HERE / "cubql_bench.exe"))
    ap.add_argument("--scene-file", default=str(ROOT / "out" / "cubql_scene.bin"))
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        sys.exit(f"missing {binary}; run out/build_cubql.bat first")

    for grid in args.grid:
        vertices, indices, _, origins, dirs = export.build(
            grid, args.seed, args.points)
        export.write(vertices, indices, origins, dirs, args.scene_file)
        scene = quarry.Scene(vertices, indices)  # 第二次构建，不含模块加载
        q_c, q_r, q_check = quarry_timings(scene, origins, dirs, args.runs)

        print(f"\n== grid={grid} tris={scene.triangle_count} "
              f"queries={args.points} depth={scene.depth} ==")
        b_med = run_cubql(binary, args.scene_file, "ray", False)[0]
        b_sah = run_cubql(binary, args.scene_file, "ray", True)[0]
        print(f"build   quarry={scene.build_ms:.2f} ms (wall, warm)  "
              f"cubql_median={b_med:.2f} ms ({scene.build_ms / b_med:.2f}x)  "
              f"cubql_sah={b_sah:.2f} ms ({scene.build_ms / b_sah:.2f}x)")
        print(f"{'query':8s} {'builder':16s} {'quarry':>9s} {'cubql':>9s} "
              f"{'q/c':>7s}  checks")
        for mode, q_ms, q_sum, label in (
            ("closest", q_c, q_check["closest"], "closest"),
            ("ray", q_r, q_check["ray"], "ray"),
        ):
            for sah in (False, True):
                _, c_ms, c_check = run_cubql(
                    binary, args.scene_file, mode, sah)
                name = "sah" if sah else "spatial-median"
                agree = (q_sum[0] == c_check[0] and
                         abs(q_sum[1] - c_check[1]) < 1e-2)
                print(f"{label:8s} {name:16s} {q_ms:9.3f} {c_ms:9.3f} "
                      f"{q_ms / c_ms:6.2f}x  "
                      f"{'ok' if agree else 'MISMATCH'}")


if __name__ == "__main__":
    main()
