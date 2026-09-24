# 01: Phase 1 — Cut inference memory traffic ~6× with bit-identical outputs

**What to build:** The decision APIs (`/v1/decide`, `/v1/systemone`, `laya decide` CLI) return exactly the same probabilities as today, but materially faster, by removing redundant DRAM traffic in the shared matrix-multiply path. Today every 4-token register block re-streams the entire weight matrix from RAM — the numbers in BENCHMARK_REPORT.md imply ~71–76 GB/s sustained on both models, i.e. the machine is pinned at its DDR5 bandwidth ceiling and most of that traffic is duplicate reads. This ticket makes the engine (a) amortize each weight load across ~12 token rows instead of 4, (b) partition matmul work across weight columns rather than token rows so no worker re-reads the whole matrix and small token counts still use all threads, and (c) keep weights resident as F16 — converting to F32 losslessly inside the SIMD kernel (F16→F32 is exact, so outputs stay bit-identical) — instead of eagerly decoding everything to F32 at load. Users also see model load consume ~half the RAM and start faster.

**Blocked by:** None (can start immediately).

**Status:** in-progress — kernel work landed (see Results); latency targets below 2×, see notes

## Results (i5-14400, harness per BENCHMARK_REPORT.md: 1 warm-up + 8 measured)

| Config | Baseline (old binary) | Phase 1 | Speedup |
|---|---|---|---|
| Laya 16T | 292.5 ms | 230.3 ms | 1.27× |
| Laya 12T | — | ~205 ms (3 samples) | best config |
| Laya 4T | 388.8 ms | 282.9 ms | 1.37× |
| OpenThai 16T | 1319.6 ms | 1129.4 ms | 1.17× |
| OpenThai 12T | — | ~1085 ms (3 samples) | best config |
| OpenThai 4T | 2070.5 ms | 1348.6 ms | 1.54× |
| Startup (load + 1 inference, Laya) | 1.41 s | 0.56 s | 2.5× |
| RAM (VmHWM, Laya / OpenThai) | 2.38 / 2.89 GB | 1.60 / 2.06 GB | matrix-resident weights ÷2 |
| Determinism | bit-identical | bit-identical | all models × examples × thread counts 1/2/4/5/7/12/16, CLI/HTTP/daemon |

Why latency gains stop short of 2×: the matmul path itself no longer dominates. Post-change profile (by elimination, no PMU in WSL2): Laya's remaining time is scalar non-matmul work (attention dot products, GELU tanh, layer norm); OpenThai's is the sequential per-token Gated-DeltaNet recurrence (single-threaded, 16 independent heads unparallelized). Additionally 16T now loses to 12T on both models — equal-width column slices on hybrid P/E cores make every matmul barrier wait for E-core stragglers. These are exactly ticket 02 menu items #2 (weighted P/E partition) and #3 (vectorize scalar helpers; add: parallelize GDN across heads).

- [x] Re-run the BENCHMARK_REPORT.md harness (1 warm-up + 8 measured requests, `usage.latency_ms`) at 4T and 16T for both models; record before/after means next to the existing baseline numbers — done, table above; full samples in the commit message
- [ ] Laya English 16T mean latency improves ≥2× vs the 270.8 ms baseline (target band 100–150 ms) — NOT MET: 1.27× (230.3 ms); residual cost is scalar attention/GELU/norm + E-core straggle, not matmul traffic (see 02)
- [ ] OpenThai-SystemOne 16T mean latency improves ≥2× vs the 1621.6 ms baseline (target band 0.4–0.8 s) — NOT MET: 1.17× (1129.4 ms); residual cost is the sequential GDN recurrence and scalar helpers (see 02)
- [x] Probabilities remain bit-identical to the reference outputs in BENCHMARK_REPORT.md §4.4 (English refund case) and §5.4 (Thai 3-question case) at every thread count tested — refund = 0.8911915421485901 and Thai = 1.974919080734253 / 0.9858790636062622 exactly, at 1/2/4/5/7/12/16 threads, across CLI/serve/daemon and both models
- [x] Loaded weight RAM footprint drops ~2× for both F16 models; startup time measurably improves — matrix-resident weights exactly halved (F16 kept); VmHWM 2.38→1.60 GB (Laya), 2.89→2.06 GB (OpenThai); startup 1.41 s→0.56 s
- [x] Kernel self-test passes across thread counts 1/2/4; the binary still fails fast with a clear error on CPUs lacking AVX2/FMA/F16C — `laya self-test` PASS (now covers F32+F16 storage, 12-row block boundary, threads 1/2/4/16); F16C gate added to the AVX2/FMA check (negative path verified by code inspection — no non-F16C CPU available here)
