# 02: Phase 2 — Post-Phase-1 optimization round (pick from profile evidence)

**What to build:** After Phase 1 lands and the new baseline is measured, choose and ship the follow-up optimizations that profiling still justifies. Candidate menu, in expected-value order:

1. **Q8_0 quantized weights** — halves memory traffic again, but probabilities drift slightly; requires an explicit accuracy-tolerance decision and re-validation of the report's reference cases (today's posture is bit-identical determinism).
2. **Weighted work partition for hybrid P-core/E-core CPUs** — give E-cores narrower column slices and/or pin threads, shaving straggler wait at every matmul barrier.
3. **Vectorize scalar helpers** — attention dot-product, layer norm, GELU (currently scalar with per-element libm calls).
4. **Request-level batching** — fold concurrent requests into one GEMM for throughput under parallel load.

Items the post-Phase-1 profile does not justify are dropped with a written rationale rather than implemented.

**Blocked by:** 01 — Phase 1 memory-traffic rework (must land first: it changes the kernel seam and defines the baseline every item here is measured against).

**Status:** done — all four criteria met; results and decisions recorded below

## Post-Phase-1 profile (recorded before choosing work)

Measured Phase-1 baselines on this machine (commit 80f27f9, harness: 1 warm-up + 8 measured): Laya 230.3 ms @16T / 282.9 ms @4T / ~205 ms @12T; OpenThai 1129.4 ms @16T / 1348.6 ms @4T / ~1085 ms @12T.

Decomposition by elimination (no PMU in WSL2; bounds from the Phase-1 deltas and the 76.8 GB/s DDR5 ceiling):

- **Laya**: traffic fell ~6× but latency fell only 1.27× → the matmul path is no longer dominant. Residual is scalar non-matmul work: sequential-FMA attention dot products (`tensor::dot` — Rust f32 sums cannot reassociate, so it compiles to scalar adds), GELU tanh per element, layer norm. 12-row-block matmul DRAM floor is ~3.2 GB ≈ 42 ms of the 230 ms.
- **OpenThai**: Phase 1 recovered only ~190 ms despite a ~6× traffic cut → matmuls were partially L3-serviced before; the dominant residual is the Gated-DeltaNet recurrence in `Delta::run`: sequential over tokens AND over all 16 heads, single-threaded, ~0.5–0.7 s by op-count estimate. Its 16 heads are mathematically independent (separate state, disjoint output slices).
- **12T beats 16T on both models** — E-core stragglers at every per-matmul barrier.

## Chosen subset

1. **GDN head-parallelism** (not on the original menu — surfaced by the Phase-1 profile): parallelize `Delta::run` across its 16 independent heads. Expected: the single largest OpenThai win. Bit-identical by construction (per-head computation order unchanged, disjoint outputs).
2. **Vectorize `tensor::dot`** with AVX2+FMA (8-wide partial sums): speeds attention on both models and RMS/GDN norms on OpenThai. NOT bit-identical (reassociation) — adoption gated on measured drift.
3. **Q8_0 in-kernel weights** + requantization script + accuracy/latency test on both models: halves weight traffic vs F16 again and halves resident weight RAM. Explicitly requested by the user to be tested, not assumed.

## Dropped / deferred (rationale)

- **Weighted P/E partition (menu #2)**: Goish exposes no thread-affinity API, so a worker cannot reliably be mapped to a P-core or E-core; weighted slices would be guesswork. Operational guidance stands instead: prefer `--threads 12` on 6P+4E hybrids (measured best on both models). Revisit if Goish gains affinity.
- **Request-level batching (menu #4)**: server currently serializes decisions per request and no concurrent-load requirement is measured; architecture change not justified by evidence on hand.
- **GELU/layer-norm vectorization (part of menu #3)**: secondary to `dot` (fewer elements, tanh needs a polynomial approximation); revisit only if the post-item-2 profile still shows it.

## Tolerances (declared before measuring, per user instruction that Q8_0 must be tested)

- `dot` AVX2: adopt only if every reference probability moves ≤ 5e-4 absolute and no discrete answer flips; otherwise revert.
- Q8_0 models: acceptable if no discrete answer flips, max |Δprobability| ≤ 0.02, |Δscore| ≤ 0.1 on the reference set (both models, all question types); otherwise the engine keeps Q8_0 support but we ship no quantized models and record the numbers.

Reference set: `examples/refund.json`, `examples/openthai-thai.json`, plus one crafted score-type Laya request and one crafted longer Thai OpenThai request (both checked in with this work).

## Results (i5-14400, same harness: 1 warm-up + 8 measured)

| Config | Phase 1 | Phase 2 | Δ |
|---|---|---|---|
| Laya F16 16T | 230.3 ms | **205.0 ms** | −11% (dot AVX2) |
| OpenThai F16 16T | 1129.4 ms | **905.4 ms** | −19.8% (GDN parallel −195 ms bit-identical, then dot) |
| OpenThai F16 12T | ~1085 ms | **870.6 ms** | best config |
| Laya Q8_0 16T | — | 214.0 ms | latency-neutral vs F16 |
| OpenThai Q8_0 16T | — | 915.9 ms | latency-neutral vs F16 |
| RAM Laya (F16 / Q8_0) | 1.60 GB | 1.60 / **0.95 GB** | Q8_0 −41% |
| RAM OpenThai (F16 / Q8_0) | 2.06 GB | 2.06 / **1.10 GB** | Q8_0 −47% |
| Session total vs pre-Phase-1 binary | Laya 292.5 / OpenThai 1319.6 (16T) | Laya 205.0 / OpenThai 870.6 (12T best) | 1.43× / 1.52× |

### Per-item evidence

- **GDN head-parallelism** (OpenThai only): 16T 1129.4 → 934.4 ms measured in isolation, outputs **bit-identical** to the pre-change binary on all references.
- **dot AVX2+FMA** (both models): Laya 230.3 → 205.0, OpenThai 934.4 → 905.4. Drift vs pre-change references: max **9.2e-07** across all reference probabilities (gate was ≤ 5e-4), zero answer changes — adopted.
- **Q8_0 in-kernel weights**: engine keeps Q8_0 blocks resident and dequantizes in the SIMD kernel; `scripts/quantize-q8.py` requantizes F16 GGUFs (embeddings stay F16 — quantizing the Laya embedding table pushed probability drift to 0.0212, past the 0.02 gate; with embeddings F16 the max is 0.0178). Drift across 6 reference cases (2 models × 3 requests incl. the 2 crafted): **zero choice flips, max |Δprobability| 0.0178, max |Δscore| 0.0181** — within the declared gates. Latency is neutral (halved bytes offset by dequant arithmetic now that matmuls are L3-resident); the win is **RAM: −41% / −47%**, i.e. roughly twice the models per host at equal speed.

### Quantization decision (explicit, per criterion)

**ADOPT Q8_0 as an opt-in model variant**: engine support + quantizer script ship; F16 remains the default and the only bit-identical path. Q8_0 files are build artifacts (`scripts/quantize-q8.py models/*_f16.gguf`), not checked in. Rationale: classification answers are stable across all reference cases (no flips, ≤0.02 probability drift) while resident RAM halves — but outputs are no longer bit-identical to F16, so the default stays F16 and quantized checkpoints are a deliberate choice per deployment. The drift gates are executable: `validation/quantized-drift.py` re-validates any F16/Q8 pair against this ticket's thresholds and exits non-zero on breach (current run: max |Δprob| 0.01782, max |Δscore| 0.01811, PASS).

- [x] Post-Phase-1 bottleneck re-measured (per-request latency + computed effective GB/s) and written down before any item is chosen — profile section above, written before implementation
- [x] The quantization question gets an explicit decision: adopt Q8_0 with a defined probability tolerance and re-validated English/Thai reference outputs, or reject with rationale (keeps the bit-identical posture) — ADOPTED as opt-in; drift table above; F16 default keeps bit-identical posture
- [x] Every implemented item ships with before/after numbers from the same benchmark harness and unchanged-or-revalidated outputs — per-item evidence above (GDN bit-identical; dot drift ≤9.2e-07; Q8_0 within declared gates, zero flips)
- [x] No regression vs the Phase 1 result: 16T mean latency on both models not worse than Phase 1's recorded numbers — Laya 205.0 ≤ 230.3; OpenThai 905.4 ≤ 1129.4
