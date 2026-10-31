# Contributing to quarry

Thanks for taking a look. This project is a small, verification-heavy GPU engine, and
the rules below exist mostly to keep the correctness contract intact.

## Ground rules

1. **Never weaken verification to make a test pass.** CPU/GPU results are compared
   bit-for-bit (per ray, per point). If an optimization changes results, it is a bug,
   even if the new result "looks fine". The only accepted changes are ones that make the
   CPU reference and the GPU agree on the same, documented predicate.
2. **Keep host and device arithmetic identical.** Shared geometry lives in
   `src/core/geometry.h` behind `QUARRY_HD`; fused multiply-adds must go through
   `QUARRY_FMA`. Do not enable `--use_fast_math` or compact float flags.
3. **Document measurements with a date, paired baseline and scene/ray type.** Absolute
   throughput on a laptop moves 1.3–2.5× between sessions. A number without a date and
   a same-session denominator will be rejected.

## Building

Windows, with the CMake/Ninja bundled in Visual Studio 2022:

```bat
build.bat
build\quarry.exe selftest
```

or directly:

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Requirements: CMake ≥ 3.20, a C++17 compiler, and optionally CUDA 12.x (set
`QUARRY_GPU=OFF` to build the CPU reference only). Python bindings need
`scikit-build-core` and pybind11 (`py -3.13 -m pip install .`).

## Before opening a PR

Run at least:

```bat
build\quarry.exe selftest
build\gpu_bench.exe --grid 256 --rays 1000000 --runs 3
build\gpu_bench.exe --scene city --buildings 4096 --rays 200000 --runs 3
build\gpu_query_bench.exe --grid 256 --points 20000 --runs 3
```

and confirm every run ends with `mismatch=0` (and `SELFTEST PASS`). If you touched
sphere/ellipsoid code, run `gpu_cloud_bench` / `gpu_gauss_bench` as well. If you changed
documented numbers, update `docs/baseline.md` with the command, date and hardware.

## Style

- C++17, 4-space indent, 100-column soft limit. Comments explain *why*, not *what*.
- Keep comments in Chinese unless the file is English-facing; the codebase is
  bilingual but internally consistent per file.
- Prefer deleting code over keeping compatibility shims; there is no external ABI to
  preserve inside the project.

## Reporting bugs

Use the issue templates. For anything that could be a security issue, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.

## License

By contributing you agree that your contribution is licensed under Apache-2.0.
