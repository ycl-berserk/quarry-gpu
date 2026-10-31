# quarry

[![CI](https://github.com/ycl-berserk/quarry-gpu/actions/workflows/build.yml/badge.svg)](https://github.com/ycl-berserk/quarry-gpu/actions/workflows/build.yml)

**中文**：[README.md](README.md)

A GPU-native spatial query engine. It builds bounding volume hierarchies (Karras LBVH,
with an optional top-down binned SAH builder) entirely on the GPU, and answers ray
queries, closest-point, k-nearest-neighbor and radius queries over **triangles, point
clouds and Gaussians** through one shared tree and one shared set of kernels. Scenes can
be updated incrementally: `refit` for pure motion, and full topology surgery (delete /
move / insert) without rebuilding the tree.

Every result is verified against a CPU reference. Apache-2.0.

## Why it exists

- **Builds are the fast part.** A 2M-triangle LBVH builds in ~7–15 ms on an RTX 4060
  Laptop. Against Embree 4.4 `rtcCommitScene` that is parity at 8k triangles and
  3–5× at 131k–2M in our sessions; against Warp's builders we measured 14–80×.
- **One API, three primitives.** Triangles, points and Gaussians share the same LBVH,
  the same traversal skeleton and the same incremental-update path. The leaf test is the
  only thing that changes (Möller–Trumbore, point distance, exact point-to-ellipsoid).
- **Incremental updates are first-class.** A pure-motion frame refits the tree in
  ~0.2 ms (7.6× faster than a fresh build on a 131k-triangle scene); topology edits are
  applied and validated bit-for-bit against a fresh rebuild.
- **Results are deterministic across CPU and GPU.** Same arithmetic on both sides, same
  ordering rule, no fast-math. It is the hardest constraint in the project and it is
  enforced by every benchmark.

## Correctness contract

`src/core/geometry.h` and the primitive ops are compiled for both host and device
(`QUARRY_HD`). Multiply-adds go through `QUARRY_FMA` (CUDA `fmaf`, MSVC `std::fma`);
nvcc is built with `--fmad=false` and fast-math is forbidden. Hits are ordered by
`(distance², scene index)` so ties are reproducible. A ray result is encoded as
`(hit << 32) | bitcast(t)`.

Every run of `gpu_bench` / `gpu_query_bench` replays the scene on the CPU and compares
**per ray / per point**: `hit_mismatch=0 t_mismatch=0` or
`tri_mismatch=0 dist2_mismatch=0 count_mismatch=0`. The stress scenes (terrain, random,
mixed, city) pass with zero mismatches at 1M rays and 20k queries. Details, including a
float-robustness fix found by the city scene, are in
[docs/baseline.md](docs/baseline.md).

## Quick start

### Python

A Windows wheel (CPython 3.13, sm_89) is on the
[Releases](https://github.com/ycl-berserk/quarry-gpu/releases/tag/v1.1.0) page. Other
platforms and GPU architectures build from source (CUDA 12.x, Python ≥ 3.9, a C++
compiler; on Windows run from an **x64 Native Tools Command Prompt** or call
`vcvars64.bat` first).

```bat
py -3.13 -m pip install .
py -3.13 python/demo.py
```

```python
import quarry

print(quarry.describe())
vertices, indices = quarry.demo_mesh(kind="city", buildings=131072)
scene = quarry.Scene(vertices, indices)          # builder="sah" for SAH

dist2, tri = scene.closest(points)               # (20000,)
kdist2, ktri = scene.knn(points, 8)              # (20000, 8)
offsets, rdist2, rtri = scene.radius(points, 2.0)
hit, t = scene.raycast(origins, dirs)
```

`demo_mesh` supports `kind="terrain" | "random" | "mixed" | "city"`.

### C++ / CMake

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix C:\quarry
```

```cmake
find_package(quarry 1.0 CONFIG REQUIRED)
target_link_libraries(app PRIVATE quarry::gpu)   # quarry::core for the CPU reference
```

```cpp
#include <gpu/gpu_bvh.h>
```

Without CUDA, `quarry::gpu` is not defined and `quarry::core` still works. The
[examples/](examples/) directory is a standalone CMake project against the install
prefix; it builds `quarry_example` (CPU vs GPU cross-checks) and `dynamic_scene`
(80-frame refit + topology-editing demo with per-frame CPU verification). API details:
[docs/api.md](docs/api.md).

## Benchmarks

RTX 4060 Laptop (24 SM, 55 W), single device. Every reported run is verified against the
CPU reference; timings are 5-run medians after a 3 s DVFS warm-up. **Absolute numbers
move 1.3–2.5× between sessions on this laptop**, so quote ratios from the same session
and carry the date.

### Rays, terrain (2026-10-28 session)

| Triangles | Input order | Morton-sorted (end-to-end) | GPU build | Embree 1T | Embree 32T packet | Input / 32T | Sorted / 32T |
|---|---|---|---|---|---|---|---|
| 8,192 | 215.10 Mrays/s | 463.05 | 1.0–2.2 ms | 16.34 | 136.37 | 1.58× | **3.40×** |
| 131,072 | 143.63 | 343.52 | 1.79 ms | 5.53 | 70.93 | 2.03× | **4.84×** |
| 2,097,152 | 62.98 | 227.05 | 14.63 ms | 2.05 | 26.21 | 2.40× | **8.66×** |

Sorting 1M rays by origin takes 0.5–2.2 ms and is included in the sorted column. It pays
off for coherent primary rays; for incoherent secondary rays it is disabled by default.

### Rays, city (2026-10-31 session)

1,343,488 triangles: a rolling 128×128 ground plus 131,072 buildings (4 walls + roof
each), heights 4–60, footprints 1.5–5, spacing 8. This scene has non-uniform scale and
deep occlusion, which is where tree quality matters most.

| Configuration | Traversal | Throughput | Build |
|---|---|---|---|
| LBVH, down | 29.36 ms | 34.06 Mrays/s | **7.6–8.2 ms** |
| LBVH, random | 19.10 ms | 52.35 Mrays/s | 7.6–8.2 ms |
| LBVH, random + ray-sort | 16.73 ms | 59.78 Mrays/s | 7.6–8.2 ms |
| **SAH, down** | **4.25 ms** | **235.26 Mrays/s** | 24.4–26.5 ms |
| SAH, random | 5.22 ms | 191.72 Mrays/s | 24.4–26.5 ms |
| Embree 32T packet, down / random | 14.54 / 11.76 ms | 68.79 / 85.03 Mrays/s | 30–47 ms |

- The SAH tree has the **same fingerprint** as the LBVH tree; it cuts average node tests
  from 173.3 to 28.4 and traversal by 6.9×. SAH is 3.42× / 2.25× faster than Embree 32T
  packet (down / random) here; LBVH wins only on build time.
- A robustness fix found by this scene: one ray in a million grazed a wall edge, and
  Möller–Trumbore's float cancellation accepted a point 0.0126 world units outside the
  triangle while the tight leaf box rejected it. `intersectTriangle` now also requires
  the hit parameter to lie inside the triangle's own AABB slab, so CPU BVH, GPU and
  brute force reject that false positive consistently. All scenes are back to `0/1M`.

### Point queries (2026-10-28 session)

20,000 query points; closest is compared against Embree `rtcPointQuery` and Warp 1.6.2.

| Triangles | closest | Warp | Embree 32T | closest / 32T | kNN k=16 | radius |
|---|---|---|---|---|---|---|
| 8,192 | 36.78 Mq/s | 38.58 | 4.81 | **7.6×** | 6.09 Mq/s | 13.54 Mq/s (r=2.0) |
| 131,072 | 13.42 Mq/s | 18.63 | 3.18 | **4.2×** | 3.51 Mq/s | 3.57 Mq/s (r=2.0) |
| 2,097,152 | 2.59 Mq/s | 3.12 | 0.85 | **3.0×** | 1.44 Mq/s | 6.22 Mq/s (r=0.5) |

On the 1.34M-triangle city (2026-10-31): closest 0.996 ms (20.08 Mq/s, 4.8× Embree
32T, 54× the CPU reference), kNN k=8 1.97 ms, radius r=2.0 0.916 ms. All bitwise
identical to the CPU reference.

### External baselines

Measured against Embree 4.4.1, Warp 1.13.0 and NVIDIA's native cuBQL, with the same
scene generators and ray/query sets:

- **Build**: parity at 8k and 3–5× faster than Embree at 131k–2M; 2× faster than
  cuBQL (median split) and 14× faster than cuBQL's SAH in the 2M case; 14–80× faster
  than Warp's builders.
- **closest**: on par with cuBQL (0.85–1.05×), 1.4–1.6× behind Warp (SAH + 4 prims per
  leaf).
- **ray**: 3.5–4.4× behind native cuBQL (Ingo Wald's home turf), 1.2–1.4× ahead of Warp
  at 2M. We do not claim a ray-traversal lead over cuBQL.

The honest positioning is **fast builds + one unified API + exact incremental updates**,
not "fastest pixel pusher".

## Repository layout

| Path | Contents |
|---|---|
| `src/core` | Geometry, scenes, fingerprints; shared by CPU and CUDA |
| `src/cpu` | Brute force, median-split BVH, kd-tree, ellipsoid BVH (reference implementations) |
| `src/gpu` | LBVH/SAH builders, ray traversal, point/kNN/radius kernels, refit/topology, wide nodes |
| `src/cli` | `quarry.exe` (selftest, run, query) |
| `bench` | GPU benchmarks, Embree/cuBQL/Warp head-to-head harnesses |
| `bindings/python` | pybind11 module |
| `examples` | Standalone consumer project (CPU/GPU checks, dynamic scene) |
| `docs` | API, design rationale, measurement log (Chinese) |

## Limitations

- CUDA only, one GPU per process. No HIP/SYCL backend yet.
- The prebuilt wheel targets CPython 3.13 + sm_89; everything else compiles from source.
- Gaussian queries use the **solid ellipsoid** semantics only (distance 0 inside);
  Mahalanobis density / screen-space projection are out of scope.
- kNN `k ≤ 32` (fixed-size per-thread table). radius outputs are unbounded but long
  runs fall back to a CUB segmented sort.
- No real scanned/reconstructed meshes in the test suite yet (procedural terrain,
  random triangles and the city scene only).
- Documentation is currently in Chinese; this file and the code comments are the
  English entry points.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

If you use quarry in academic work, see [CITATION.cff](CITATION.cff).
