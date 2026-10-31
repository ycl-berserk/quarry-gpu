# Changelog

All notable changes to quarry are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions use
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **City scene**: a rolling 128×128 ground plus tens of thousands of jittered buildings
  (4 walls + roof each), covering non-uniform scale and deep occlusion. Wired into the
  CLI, all GPU benchmarks, the Embree comparison harness and `quarry.demo_mesh`
  (`kind="city", buildings=N`). See [docs/baseline.md](docs/baseline.md).
- `dynamic_scene` example: an 80-frame incrementally updated scene (72 refit frames +
  8 topology-surgery frames) with per-frame CPU cross-checks and a final bitwise
  comparison against a fresh build.
- English README, changelog, contributing guide, citation metadata and issue/PR
  templates.

### Fixed

- **Ray robustness at large coordinates.** Möller–Trumbore could accept a hit whose
  computed point lies outside the triangle (float cancellation), while the tight
  per-triangle AABB used by the GPU traversal rejected it, so CPU, GPU and brute force
  disagreed on roughly one grazing ray per million in the city scene. `intersectTriangle`
  now also requires the hit parameter to lie inside the triangle's own AABB slab, and the
  AABB interval test keeps a small relative margin. All scenes are back to zero
  mismatches per 1M rays; the cost is one extra slab test per accepted triangle.

## [1.0.0] - 2026-10-31

First public release.

### Added

- Karras LBVH build on the GPU, with an optional top-down binned SAH builder.
- Ray traversal, closest-point, k-nearest-neighbor and radius queries.
- Three primitives sharing one tree and one query skeleton: triangles, point clouds and
  Gaussian ellipsoids.
- Incremental updates: `refit` for pure motion, topology surgery (delete / move /
  insert) with validation against a fresh build; optional 4-wide node folding.
- CPU reference implementations and a bitwise CPU/GPU correctness contract.
- CLI (`quarry.exe selftest|run|query`), GPU benchmarks with per-ray/per-point
  verification, Embree/cuBQL/Warp head-to-head harnesses.
- Python bindings, a `quarry-gpu` wheel (CPython 3.13, sm_89), CMake package export
  (`quarry::core` / `quarry::gpu`) and a standalone consumer example.
- CI on Windows (CPU + CUDA) and Linux (CPU-only).

[Unreleased]: https://github.com/ycl-berserk/quarry-gpu/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ycl-berserk/quarry-gpu/releases/tag/v1.0.0
