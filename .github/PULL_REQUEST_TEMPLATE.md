## Summary

<!-- What changed and why. Link the issue if there is one. -->

## Verification

- [ ] `build\quarry.exe selftest` passes
- [ ] `gpu_bench` reports `hit_mismatch=0 t_mismatch=0` on the affected scenes
- [ ] `gpu_query_bench` reports `tri_mismatch=0 dist2_mismatch=0 count_mismatch=0`
- [ ] Documented numbers in `docs/baseline.md` / `README*.md` are updated (with date,
      hardware and a same-session denominator), or unchanged
- [ ] No fast-math / relaxed-float flags added; shared host/device math still goes
      through `QUARRY_FMA`

## Notes

<!-- Benchmark deltas, trade-offs, follow-ups. -->
