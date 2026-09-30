# aarch64 NEON tier — M1 acceptance package (wayfinder tickets 17 + 18)

The engine's aarch64 NEON kernel tier is implemented in `engine/src/tensor.rs`
(behind `#[cfg(target_arch = "aarch64")]`, gemm-0.19-lifted grid + hardware
FCVT f16→f32 dequant fused in the k-loop; scalar reference kernels remain the
selectable fallback). Ticket 18 adds three runtime-selectable variants —
`LAYA_AARCH64_KERNEL={scalar,fcvt,fmlal,sdot}`:

- **fcvt** (default) — the ticket-17 portable ARMv8-A path (FCVT dequant +
  `vfmaq_f32` grid, now on 16-row blocks).
- **fmlal** (FEAT_FHM, ARMv8.2 — M1 has it) — F16 matmuls via the fused
  half-precision FMA-long (`fmlal`/`fmlal2`, katgpt-rs Issue-201 technique,
  lifted per the T2 rule) against per-row-block f16-narrowed activations
  (hardware FCVTN). Different summation arithmetic than fcvt: A/B candidate,
  adoption gated on this run's drift + latency tables.
- **sdot** (FEAT_DOTPROD, ARMv8.2) — Q8_0 matmuls via `sdot` i8×i8→i32 dots
  against per-32-block quantized activations (llama.cpp-style arithmetic).
  Same A/B status.

This package is everything the owner's MacBook M1 needs to produce the
acceptance numbers. Cross-ISA outputs are NEVER bit-identical (map decision):
everything here is judged by the tolerance gates.

## What was already proven off-M1 (i5-14400 dev machine, 2026-09-29 refresh)

- `cargo check --target aarch64-unknown-linux-gnu` and the release static
  `aarch64-unknown-linux-musl` build: PASS (CI `cross` + `qemu` build tier).
  (blake3 — katgpt-types' transitive dep — is pinned to `no_neon` in
  `engine/Cargo.toml`: the engine stays C-compiler-free on every target.)
- Full engine `self-test` PASS running the aarch64-musl binary under
  docker arm64 emulation (binfmt/qemu): F32/F16/Q8_0 NEON kernels vs scalar
  reference across thread partitions, exhaustive 65536-pattern FCVT
  bit-exactness (vs `gguf::half`; sNaN quieted by hardware, compared as
  NaN), FCVTN f32→f16 RNE probe + 1M-value sweep, per-variant arms
  (fmlal/sdot — the emulated CPU exposes asimdfhm + asimddp), and the
  scalar-fallback A/B. PASS with each of the five
  `LAYA_AARCH64_KERNEL` settings.
- Drift gates for all 6 reference cases under the same emulation, per
  variant (`bash engine/aarch64/emu-drift.sh`): results in
  `results-emu/`. Emulated timings are meaningless and were never taken
  (map rule).

The M1 run re-proves all of it on real hardware and adds the latency table.

## Run on the MacBook M1 (one command)

Prereqs: Xcode Command Line Tools, rustup (stable, aarch64), python3,
~4 GB free RAM, the repo with `models/` populated (both F16 + Q8_0).

```sh
cd <repo>
bash engine/aarch64/run-m1.sh
```

Stages: (1) native `aarch64-apple-darwin` release build
(`TARGET_CPU`, default generic), (2) self-test, (3) drift gates vs the
shipped x86-64 reference dumps in `reference/` — per kernel variant,
(4) Laya + OpenThai latency per variant, 1 warmup + 8 measured decides per
thread count (1/4/8/10 — M1 is 4P+4E), (5) optional candle-aarch64
reference bar (`CANDLE=1 bash engine/aarch64/run-m1.sh`), (6) a paste-back
summary table grouped by variant.

Model paths, thread counts, and the variant set are overridable:

```sh
THREADS="1 4 8" VARIANTS="fcvt fmlal" LAYA_F16=/path/to/laya.gguf ... \
    bash engine/aarch64/run-m1.sh
```

Every stage prints PASS/FAIL; the script exits non-zero on any FAIL.
`results/` receives the raw outputs (drift `.got.json`, bench `.jsonl`,
summary files) for spot-checking.

## Reference dumps (`reference/`)

Refreshed (2026-09-29, ticket 18) by the x86-64 gnu engine binary — the
previous set predated the katgpt softmax swap (commit bf63886); the refresh
keeps the gate purely cross-ISA. Regenerate with
`decide MODEL --request CASE --raw --threads 4`:

| file | model | case |
|---|---|---|
| `laya_f16_refund.json` | laya_english_f16 | examples/refund.json |
| `laya_f16_urgent_score.json` | laya_english_f16 | examples/urgent-score.json |
| `laya_q8_refund.json` | laya_english_q8_0 | examples/refund.json |
| `ot_f16_thai.json` | openthai_systemone_v03_f16 | examples/openthai-thai.json |
| `ot_f16_mixed.json` | openthai_systemone_v03_f16 | examples/openthai-mixed.json |
| `ot_q8_thai.json` | openthai_systemone_v03_q8_0 | examples/openthai-thai.json |

Like-for-like (same model + case on both ISAs). The drift gate
(`drift.py`): max |Δlogit| ≤ 0.05 and max |Δact_logit| ≤ 0.05,
max |Δp| ≤ 0.02 (probabilities, scores, confidence), 0 choice flips.

## Acceptance checklist (filled in from the M1 run)

- [ ] `run-m1.sh` stage 1 build PASS (rustc version: ______)
- [ ] stage 2 self-test PASS on hardware
- [ ] stage 3: 6/6 drift gates PASS (worst max|Δlogit| = ______,
      worst max|Δp| = ______, flips = 0)
- [ ] stage 4 latency table (paste the summary markdown):
- [ ] optional `CANDLE=1`: candle-aarch64 mean per thread count
      (Laya convert mode): ______
- [ ] recorded by the lead into wayfinder (closes the acceptance half)

Notes for interpretation: threads > 8 should show the E-core ceiling;
`-Ctarget-cpu=generic` is the B-stage build (ticket 18 owns per-chip
tuning, e.g. `-Ctarget-cpu=apple-m1` A/B). The candle bar mirrors how the
wasm candle references were taken (convert mode, 1 warmup + 8 repeats).
