"""最短的 Python 用法：建场景 → 查询 → 读结果。

    py -3.13 python/demo.py

扩展模块在 build/python/ 下；build.bat 会连同 CUDA 运行时一起放好。
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build" / "python"))
import quarry  # noqa: E402


def main():
    print("device:", quarry.describe() or "(no GPU)")

    vertices, indices = quarry.demo_mesh(grid=256, seed=1)
    scene = quarry.Scene(vertices, indices)
    print(f"tris={scene.triangle_count} nodes={scene.node_count} depth={scene.depth} "
          f"build={scene.build_ms:.2f} ms")

    rng = np.random.default_rng(7)
    mb = scene.bounds
    lo = np.array(mb["min"], dtype=np.float32)
    hi = np.array(mb["max"], dtype=np.float32)
    points = (lo + (hi - lo) * rng.random((20000, 3))).astype(np.float32)

    dist2, tri = scene.closest(points)
    nearest = np.sqrt(dist2)
    print(f"closest: kernel {scene.last_query_ms:.3f} ms, "
          f"nearest mean {nearest.mean():.3f}, misses {int((tri < 0).sum())}")

    kdist2, ktri = scene.knn(points[:1000], 8)
    print(f"knn(8) on 1000 points: kernel {scene.last_query_ms:.3f} ms, "
          f"row0 (dist2, tri) = {list(zip(kdist2[0], ktri[0]))}")

    offsets, rdist2, rtri = scene.radius(points, 0.5)
    counts = np.diff(offsets)
    print(f"radius(0.5): kernel {scene.last_query_ms:.3f} ms, "
          f"{offsets[-1]} hits, mean {counts.mean():.2f} max {counts.max()} per point")

    origins = points[:1000]
    dirs = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (1000, 1))
    hit, t = scene.raycast(origins, dirs)
    print(f"raycast: {int(hit.sum())}/1000 hit, mean t {t[hit].mean():.3f}")

    # Python 侧固定开销：一次调用 = 上传 + 内核 + 回读 + 建返回数组
    walls = []
    for _ in range(5):
        start = time.perf_counter()
        scene.closest(points)
        walls.append((time.perf_counter() - start) * 1e3)
    print(f"closest end-to-end {np.median(walls):.3f} ms "
          f"(kernel {scene.last_query_ms:.3f} ms)")


if __name__ == "__main__":
    main()
