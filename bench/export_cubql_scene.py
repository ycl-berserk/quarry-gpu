"""把 quarry 的 demo 场景与查询集导出为 cubql_bench 读取的二进制。

格式（小端）：
    int32 nv, nt, nq
    float32 verts[nv*3]
    int32   indices[nt*3]
    float32 origins[nq*3]
    float32 dirs[nq*3]
"""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build" / "python"))
import quarry  # noqa: E402


def build(grid, seed, points):
    """返回 (vertices, indices, scene, origins, dirs)。"""
    vertices, indices = quarry.demo_mesh(grid=grid, seed=seed)
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    scene = quarry.Scene(vertices, indices)

    mb = scene.bounds
    lo = np.array(mb["min"], dtype=np.float32)
    hi = np.array(mb["max"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    origins = (lo + (hi - lo) * rng.random((points, 3))).astype(np.float32)
    dirs = rng.normal(size=(points, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs.astype(np.float32)
    return vertices, indices, scene, origins, dirs


def write(vertices, indices, origins, dirs, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = np.array([vertices.shape[0], indices.shape[0], origins.shape[0]],
                      dtype=np.int32)
    with out_path.open("wb") as f:
        for arr in (header, vertices, indices, origins, dirs):
            f.write(np.ascontiguousarray(arr).tobytes())
    return header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--out", default="out/cubql_scene.bin")
    args = ap.parse_args()

    v, i, _, o, d = build(args.grid, args.seed, args.points)
    header = write(v, i, o, d, args.out)
    print(f"wrote {args.out} nv={header[0]} nt={header[1]} nq={header[2]}")


if __name__ == "__main__":
    main()
