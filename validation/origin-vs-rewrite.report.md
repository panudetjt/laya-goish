# origin-vs-rewrite: goish alpha.14 vs the std-Rust rewrite (2026-09-30)

Both binaries fresh-built on this machine, baseline x86-64 codegen, both
self-tests PASS, outputs bit-identical (CI 42-combo contract) — so the
latency delta is free of output-difference concerns.

- goish: origin/main @ 6dd3afc, worktree build (the released alpha.14 recipe)
- rewrite: feat/benchmark @ 50d8ad1, engine/run.sh build
- protocol: daemon mode, 1 warmup + 8 measured, engine usage.latency_ms

| model | threads | goish ms | rewrite ms | speedup |
|---|---:|---:|---:|---:|
| laya_f16 | 1 | 1170.0 | 574.6 | 2.04x |
| laya_f16 | 4 | 419.4 | 210.1 | 2.00x |
| laya_f16 | 8 | 323.8 | 188.2 | 1.72x |
| laya_f16 | 16 | 252.2 | 179.4 | 1.41x |
| ot_f16 | 1 | 5412.0 | 2882.5 | 1.88x |
| ot_f16 | 4 | 2129.5 | 1049.6 | 2.03x |
| ot_f16 | 8 | 1570.6 | 895.1 | 1.75x |
| ot_f16 | 16 | 1388.0 | 811.1 | 1.71x |
| ot_q8_0 | 1 | 5708.4 | 2792.4 | 2.04x |
| ot_q8_0 | 4 | 2171.0 | 933.8 | 2.32x |
| ot_q8_0 | 8 | 1632.0 | 847.8 | 1.92x |
| ot_q8_0 | 16 | 1555.2 | 787.6 | 1.97x |

Speedup range: **1.41x – 2.32x** (mean 1.90x),
consistent across models and thread counts. The rewrite ships the same
outputs ~2x faster than the engine it replaces.
