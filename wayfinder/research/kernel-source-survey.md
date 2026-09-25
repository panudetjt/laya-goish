---
wayfinder: research
status: resolved
ticket: wayfinder/tickets/15-kernel-source-survey.md
date: 2026-09-25
verification: load-bearing citations (license fields, microkernel grids, dispatch lines, wasm f16 scalar fallback) independently re-checked by the lead against raw sources
---

# Kernel source survey: gemm NEON/SIMD128 + cubecl (the B lift-plan)

Sources read at pinned revisions: `sarah-quinones/gemm` @ `86102c5b712737978371ac9ef7a11982f686d7bc`
(= tag `gemm-v0.19.0`; the crates' own `repository` field says `github.com/sarah-ek/gemm` — same repo),
`sarah-quinones/pulp` @ `10b6867c9a11e5002ebe79fb356d8951fc62b611` (pulp 0.22.0, gemm 0.19's SIMD layer),
`tracel-ai/cubecl` @ `04676cd906552cb9a2b38e869d4f3c4064d73dec` (main). Every upstream claim below
carries the file + line read. "not pinned" = read at file granularity only.

## Executive verdict

| Target | B is liftable | What we lift | Effort | License carry |
|---|---|---|---|---|
| aarch64 (Apple Silicon) | Yes | NEON f32 microkernel bodies + tiling contract (gemm-f32/src/microkernel.rs:358-448), f16→f32 widen pattern (gemm-common/src/simd.rs:2288) | **M** | MIT ("Copyright (c) 2021 sarah") per copied file |
| wasm32 SIMD128 | Yes, easiest | simd128 f32 microkernels (gemm-f32/src/microkernel.rs:501-562) + our own f16→f32 bit-expand in the k-loop | **S–M** | same |
| cubecl | **No** | Nothing liftable: CPU leg is an LLVM-JIT runtime (native-only, no wasm), kernels are CubeCL IR, not static NEON/SIMD128 source | — | n/a |

The B strategy: keep our `Matrix::apply` thread partitioning and 12-row outer blocking, lift gemm's
*microkernel bodies* (the `microkernel!` macro + the NEON / SIMD128 op sets), and replace gemm's
pack-once pipeline with our fused dequant-on-load k-loop — exactly what our AVX2 `block::<NR>` does
today (src/tensor.rs:198-259). Do not lift gemm's packing/cache/ptr machinery; do not take the
native-fp16 (f16-accumulate) paths.

## Our kernel's requirements (the frame)

- `Matrix::apply` (src/tensor.rs:92-130): threads own disjoint output-feature slices `[c0,c1)`;
  weights stream once per row block regardless of thread count.
- `kernel` (src/tensor.rs:151-194): `#[target_feature(enable = "avx2,fma,f16c")]`, outer row blocks
  of up to 12 token rows (`nr = (m - r).min(12)`), one weight column `c` at a time.
- `block::<NR>` (src/tensor.rs:198-259): const-generic NR accumulators in YMM; per 8-lane k step:
  kind 1 = F16 weights widened in-kernel via `_mm256_cvtph_ps` (line 235); kind 8 = Q8_0 block
  dequant (lines 213-231); kind 0 = plain f32; `_mm256_fmadd_ps`; horizontal lane sum + scalar k-tail
  (lines 245-257) preserves per-row sequential summation order (the x86-64 bit-exactness mandate).
- Shapes: thin m = 45 (Laya) / 161 (OpenThai), k,n ∈ 1024–6144; F16-resident weights, f32 accumulate.
- Off x86-64 there is **no** bit-exactness mandate: NEON/SIMD128 tiers pass the tolerance gates
  (0 flips, |Δp| ≤ 0.02) measured in proto-wasm-gate.md — a different lane-parallel summation order
  is acceptable there.

## Q1. gemm 0.19 on aarch64: NEON microkernels, blocking, dispatch, F16

**f32 microkernels** — `gemm-f32/src/microkernel.rs`, `pub mod neon` at line 358 (gated
`#[cfg(target_arch = "aarch64")]`):
- Ops (lines 380-415): `vmulq_f32`, `vaddq_f32`, `mul_add` = `vfmaq_f32` (line 390),
  `mul_add_lane::<LANE>` = `vfmaq_laneq_f32` (line 395), scalar fallback `neon_fmaf` (413-415).
- Multi-row loads: `vld1q_f32`, `_x2`, `_x3`, `_x4` for MR 4/8/12/16 (lines 420-423).
- Kernel grid (lines 428-446): `microkernel!(["neon"], 4, x1x1, 1, 1)` … `x4x4, 4, 4` —
  **N=4 lanes, unroll 4, MR ∈ {4,8,12,16}, NR ≤ 4**. The x4 variants (`x1x4`…`x4x4`, lines
  431-446) take the lane path, broadcasting one rhs lane and running `mul_add_lane` per token row.
- So aarch64 f32 tiling is **MR≤16 × NR≤4**, k-unroll 4 — a bigger row tile than our 12, same NR≈4
  shape. An Apple AMX module exists behind the non-default `experimental-apple-amx` feature
  (gemm-f32/src/microkernel.rs:462; selected only if the feature is on AND `HasAmx::get()`,
  gemm-f16/src/gemm.rs:817-819) — not our path.

**f16 on aarch64** — two distinct paths:
- `Neon` (no fp16 feature): `MixedSimd<f16, f16, f16, f32> for Neon` at gemm-common/src/simd.rs:2288 —
  F16 inputs are **widened to f32 during packing**; the k-loop runs the f32 NEON microkernels above.
  This is dequant-on-pack, one step removed from our dequant-on-load.
- `NeonFp16` (fp16 feature detected): native f16 SIMD in gemm-f16/src/microkernel.rs:16-96 —
  `vfmaq_f16` (line 34), `vfmaq_laneq_f16` (53-54), grid `microkernel!(["neon,fp16"], 4, …)`
  MR≤16 × NR≤8 (73-96). Dispatch picks it whenever `feature_detected!("fp16")`
  (gemm-f16/src/gemm.rs:820-821). Two accumulate variants exist: `MixedSimd<f16,f16,f16,f32>`
  (gemm-common/src/simd.rs:2078, f32 accumulator — gate-compatible) and
  `MixedSimd<f16,f16,f16,f16>` (:2183, f16 accumulator — **must not be taken**; f16 accumulation
  would blow the |Δp| ≤ 0.02 gates, cf. proto-wasm-gate.md's f16-gemm drift).
- **bf16: none.** The 0.19 workspace has no bf16 crate (workspace Cargo.toml members: gemm,
  gemm-common, gemm-f16, gemm-f32, gemm-f64, gemm-c32, gemm-c64) — irrelevant for us (weights are F16).

**Dispatch** — two layers:
1. pulp defines `Neon` via `__impl!(Neon, "neon")` (pulp/src/core_arch/aarch64/mod.rs:44) whose
   `try_new()` is `feature_detected!($feature)` (lines 21-22) — NEON is baseline on aarch64, so this
   always succeeds; optional features (`fp16`, `aes`, `dotprod`, …) get their own structs (lines 89+).
2. gemm caches the choice once: `init_gemm_fn()` checks `feature_detected!("neon")` →
   (`experimental-apple-amx`?) → `"fp16"` → `neonfp16::gemm_basic` else `neon::gemm_basic`, then
   stores the fn pointer in `GEMM_PTR: AtomicPtr<()>` and every call goes through `get_gemm_fn()`
   (gemm-f16/src/gemm.rs:799-849). Same one-time-dispatch pattern as our CPU-gate + `#[target_feature]`
   kernel (src/tensor.rs:151, 426-432). The generated microkernel functions themselves carry
   `#[target_feature(enable = $target)]` from the macro (gemm-common/src/microkernel.rs:400).

**Thin-m characteristics**: nothing measured is published in the repo (benches/bench.rs is
large-m shaped; no thin-m data found — open question). Structural facts: for m=45, MR selection
from {4,8,12,16} gives 16+16+12+1 or similar — weights re-stream once per MR-row band per n-panel,
the same disease proto-candle-laya.md measured on x86 ("at m=45 every m-tile re-streams raw
weights" below ~512 rows). gemm's count (⌈45/16⌉ = 3 bands + remainder) is slightly better than our
12-row blocking (⌈45/12⌉ = 4), so the microkernel is not the thin-m problem; the x86 evidence blames
packing + rayon work-split, which our lift-plan never takes (we keep our own `apply()` partitioning).
**No local aarch64 measurement exists — bench gemm on Apple Silicon before committing either option.**

## Q2. gemm 0.19 on wasm32: SIMD128 kernels and the runtime-dispatch pattern

**f32 SIMD128 microkernels** — `gemm-f32/src/microkernel.rs`, `pub mod simd128` at line 501
(gated `#[cfg(target_arch = "wasm32")]`):
- Ops (508-535): `f32x4_mul` (508-510), `f32x4_add` (513-515), and **`mul_add` is NOT fused — it is
  `add(c, mul(a, b))` (518-520)**; wasm SIMD128 has no FMA. Scalar tail via `a*b + c` (533-535).
- Kernel grid (537-550): `microkernel!(["simd128"], 2, x1x1, 1, 1)` … `x3x4, 3, 4` —
  **N=4 lanes, unroll 2, MR ∈ {4,8,12}, NR ≤ 4**. Note MR max is 12 — exactly our engine's 12-row
  block; `x3x4` (549-550) is the same effective tile as our `block::<12>`.
- The `["simd128"]` string lands in the macro's `$(#[target_feature(enable = $target)])?`
  (gemm-common/src/microkernel.rs:400) — **this is the `#[target_feature]` runtime-dispatch pattern
  our wasm gate measured at 38.6 GFLOPS with zero compile flags** (proto-wasm-gate.md, "SIMD128:
  already on by default"). It also explains why forcing `-Ctarget-feature=+simd128` is unnecessary —
  and why it broke candle-core's f16 microkernel (`CurrentCpuF16` alias only exists for x86/aarch64;
  proto-wasm-gate.md blockers #4).

**Selection**: `gemm-common/src/gemm.rs:1038-1041` — under `#[cfg(target_arch = "wasm32")]`,
`if $crate::feature_detected!("simd128") { simd128::gemm_basic }`, falling back to scalar
(1051 region). pulp's wasm side (pulp/src/wasm.rs) defines `impl Simd for Simd128` (line 125) and
`Arch::new()` at ~2440-2451 probing RelaxedSimd → Simd128 → Scalar. The same shape holds for f16 —
**except there is no SIMD128 f16 kernel at all**:
- gemm-f16/src/microkernel.rs (112 lines total) contains only `v128_common::f16` (scalar-op helpers,
  lines 2-14), `neonfp16` (16-96), `amx` (99+) — **no wasm32 module**.
- The f16 dispatch has a final `#[cfg(not(any(x86, x86_64, aarch64)))] scalar::gemm_basic` arm
  (gemm-f16/src/gemm.rs:832) — wasm32 lands there. `scalar::gemm_basic` (855-903) uses
  `gemm_f32::microkernel::scalar::f32::*` (N=1) with `MixedSimd<f16,f16,f16,f32> for Scalar`
  (gemm-common/src/simd.rs:29) doing per-lane f16→f32 widening in scalar code.
- **This is the code-level proof of our measured wasm f16 pessimization**: f16-gemm ran scalar at
  ~4.4× the f32 SIMD128 kernels (60.8 vs 13.9 ms on 256×1024×1024; proto-wasm-gate.md).

**Same-shape adaptation for us** (m=45/161, k,n 1024–6144, F16-resident):
- Keep the 12-row outer loop (src/tensor.rs:173-192); for each weight column run the lifted `x3x4`
  SIMD128 kernel body — MR 12 × 4 lanes, unroll 2 — and the x1/x2 variants for remainders (45 →
  12+12+12+8+1; 161 → 13×12 + 5).
- Replace gemm's f32 rhs pack with our fused dequant: per k step, load 4× u16 F16 lanes and expand to
  f32x4 **with bit manipulation** (wasm has no f16 instructions — exponent/mantissa shift, same math
  as `crate::gguf::half`), then `f32x4_mul` + `f32x4_add` per token row. This is the one piece gemm
  cannot give us on wasm (its x86 f16 paths dequantize during pack via `V3Half`/`V3`,
  gemm-common/src/simd.rs:657/752; nothing equivalent exists for SIMD128).
- Scalar k-tail exactly like our AVX2 kernel's (src/tensor.rs:249-253), or gemm's `k_leftover`
  loop (gemm-common/src/microkernel.rs, generic execute path).
- Q8_0: same shape as our AVX2 path (src/tensor.rs:213-231) — per-32 block scale + i8 widen; wasm
  i16x8_extend_s + f32x4_convert covers the i8→i32→f32 chain.
- Expected ceiling: the measured 38.6 GFLOPS f32 SIMD128 (proto-wasm-gate.md) bounds the B tier;
  our convert-mode forwards already sustain ≈28.6 GFLOPS effective through gemm's full pipeline, so
  the fused lift plausibly lands at or above that (drops the pack pass) — verify with a microbench.

## Q3. cubecl: honest verdict — GPU-shaped, nothing liftable

cubecl @ `04676cd` is a multi-backend compute framework (CUDA/HIP/Metal/SPIR-V/WGSL per its README).
Its CPU leg, `crates/cubecl-cpu`, is **not a kernel library but an LLVM-JIT runtime**: per its README
at that commit, it "executes kernels in-process on a worker thread pool, using SIMD where the host
CPU supports it. Compilation is delegated to `cubecl-llvm`, which JIT-compiles each kernel through
LLVM". Consequences for us:

1. **No static kernel source to lift.** Kernels are authored in CubeCL IR and compiled at runtime;
   there are no readable NEON/SIMD128 microkernel .rs files to copy — the opposite of gemm's layout.
2. **No wasm path.** LLVM JIT produces host-native code only; `cubecl-cpu` cannot target
   wasm32-unknown-unknown. Our second tier is unreachable through it by construction.
3. **Wrong trade for aarch64.** Adopting it would mean shipping a CubeCL-IR + LLVM-JIT stack to
   obtain a GEMM we could instead lift 300 lines of NEON to get. Its SIMD scheduling might inform a
   future C strategy conceptually, but that is an architecture reference, not reusable code.

Verdict: **cubecl contributes nothing to the B lift**; the ticket's hypothesis ("CPU/wasm kernel
generation as source #2") is refuted at the source level. Its GEMM kernels are GPU-tiling-shaped and
JIT-delivered. (cubecl's exact license at this commit: not pinned — moot given the verdict; if ever
revisited, check `Cargo.toml` `license` fields first.)

## Q4. The B lift-plan

### What to lift (and what not)

Lift the **microkernel layer only**:
- `gemm-f32/src/microkernel.rs` — the `neon` op set + grid (lines 358-448) for aarch64; the
  `simd128` op set + grid (501-562) for wasm32. These are small, self-contained, and depend only on
  `core::arch` intrinsics + the `microkernel!` macro.
- `gemm-common/src/microkernel.rs` — the `microkernel!` macro's loop skeleton (lines 397-709):
  accumulator storage `[[splat(0.0); $mr_div_n]; $nr]` (line 422), generic `execute()` k-loop
  (441-466), the NEON lane-broadcast path `execute_neon()` with `mul_add_lane` (468-494), k_unroll /
  k_leftover split, and the epilogue's full-tile fast path vs scalar remainder (632-705). For our
  use the epilogue simplifies drastically: we write one output element per (row, c) with bias add —
  our existing epilogue (src/tensor.rs:245-258) already does this and preserves the order we want.
- The **f16→f32 widen pattern**, reimplemented in our k-loop (dequant-on-load, not gemm's
  dequant-on-pack): NEON `vcvt_f32_f16` on an 8-lane u16 load (the `_mm256_cvtph_ps` analog);
  SIMD128 manual bit-expand as described in Q2. gemm's `MixedSimd<f16,f16,f16,f32>` impls
  (gemm-common/src/simd.rs:29, 657, 752, 2078, 2288) are the reference for edge cases.

**Not lifted**: gemm's packing pipeline (`gemm-common/src/pack_operands.rs`, `cache.rs`, the
`gemm_basic!` machinery, `dyn-stack` temp allocs, rayon work-splitting) — our `Matrix::apply`
(src/tensor.rs:92-130) already partitions by output-feature slices and streams weights without a
pack pass, which is precisely the overhead that made stock gemm lose to the engine at thin m on x86
(proto-candle-laya.md). Also not lifted: `neonfp16`/AMX/AVX512 paths, and every
`MixedSimd<…,f16>` f16-accumulate variant.

### License headers to carry

gemm 0.19 is **plain MIT** — verified, not assumed: `gemm-f32/Cargo.toml:9` `license = "MIT"`
(gemm-f16's identical), LICENSE file (all workspace members share the same file,
sha `84e3f44`) reading `MIT License / Copyright (c) 2021 sarah`. There is no Apache-2.0 text in the
repo despite the ticket's "MIT/Apache" assumption; do not add one. pulp is MIT likewise (LICENSE at
`10b6867c`). Carry per copied file:

```text
MIT License

Copyright (c) 2021 sarah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Plus a NOTICE line: "NEON/SIMD128 microkernel structure adapted from gemm 0.19.0 (sarah),
MIT license." Only needed for files that are actually lifted; an independently written kernel
informed by reading needs attribution courtesy, not the license block.

### Per-target plan

**aarch64 (Apple Silicon) — effort M (1–2 wk)**
1. Port `block::<NR>` (src/tensor.rs:198-259) to NEON keeping the exact control flow: NR ∈ {1..12}
   const-generic accumulators as `[float32x4_t; 4]` groups; adopt gemm's lane-broadcast trick
   (`vfmaq_laneq_f32`, gemm-f32/src/microkernel.rs:395) for the rhs — one F16 dequant serves 4
   accumulator rows per lane broadcast.
2. F16 path: `vld1q_u16` ×8 lanes → `vcvt_f32_f16` (lower) + `vcvt_high_f32_f16` → fma into
   accumulators; Q8_0 via `vld1_s8` → `vmovl_s16`/`vmovl_s32` chain → `vcvtq_f32_s32` × block scale.
   Scalar k-tail identical to today's (src/tensor.rs:249-253).
3. Gate with `is_aarch64_feature_detected!("neon")` (baseline) in the CPU check replacing
   src/tensor.rs:426-432 on non-x86; `#[target_feature(enable = "neon")]` on the kernel fn — pulp's
   runtime-probe pattern (pulp/src/core_arch/aarch64/mod.rs:21-22, 44) is the reference.
4. Validation: tolerance gates (0 flips, |Δp| ≤ 0.02) against the x86-64 reference outputs — NOT
   bit-exactness (lane-parallel fma order differs by design). Bench m=45/161 single-thread first;
   Apple P-core count will differ from the i5-14400 numbers.
5. Explicitly skip `fp16` detection: never take `neonfp16` paths (gemm-f16/src/gemm.rs:820-821)
   since the f16-accumulate variant fails our gates and the f32-accumulate fp16 variant buys nothing
   over the plain Neon widen for our k-loop-fused design.

**wasm32 (SIMD128) — effort S–M**
1. Same skeleton, ops swapped for `f32x4_*` (no fused FMA — gemm's `add(c, mul(a,b))`,
   gemm-f32/src/microkernel.rs:518-520, is already the shape V8 gets); MR 12 tiles → `x3x4`
   equivalent.
2. The real work item is the **f16→f32 bit-expand** (4 lanes: u16 → mask/sign/exponent/mantissa →
   f32x4) fused into the k-loop, plus the Q8_0 widen chain; verify subnormal handling against
   `crate::gguf::half` (weights are F16-resident, so dequant must match the engine's decode).
3. `#[target_feature(enable = "simd128")]` + runtime probe matching gemm's
   `feature_detected!("simd128")` (gemm-common/src/gemm.rs:1038-1041) — V8 executes it
   unconditionally (measured, proto-wasm-gate.md).
4. Validation: reuse the proto-wasm-gate harness gates (6.7e-6 like-for-like on convert mode today;
   the lifted kernel must hold ≤1e-2-class logit drift and 0 flips).
5. Keep the scalar fallback compiled in (getrandom-free) for engines without simd128.

### Where stage C (full hand-tune) diverges from B

- B reuses gemm's tiling (MR≤16/NEON, MR≤12/SIMD128, NR≤4) and gemm's k-unroll (4/2). C would
  re-tile per target: Apple NEON has 4× 128-bit pipes — a 4-accumulator × 4-column schedule with
  explicit load pipelining (gemm's `execute_neon` lane path is the starting hint,
  gemm-common/src/microkernel.rs:468-494) can go wider than B; V8 SIMD128 wants maximal unroll of
  the non-fused mul+add pairs to hide latency.
- C adds m-remainder specialization (m=45: 16+16+12+1 → dedicated epilogues; m=161: 13×12+5 on
  SIMD128), k-loop peeling to 32/64 so F16/Q8_0 block decode is branch-free, and possibly
  n-panel blocking across weight columns to cut store traffic (gemm's NR≤4 vs our 1-column loop —
  C could take gemm's full MR×NR tile and batch 4 weight columns per pass, which B deliberately
  does not).
- C owns its own validation burden; B inherits gemm's battle-tested loop shape and only fuses dequant.

## Q5. Fallback note — option A: just call gemm 0.19 for non-x86 targets

**Cost of option A.** Integration is already proven end-to-end by the prototypes: candle-core 0.11
(gemm 0.19 + pulp + rayon) compiles for wasm32 with exactly one cfg
(`RUSTFLAGS='--cfg getrandom_backend="custom"'` + the shared `__getrandom_v03_custom` stub;
proto-wasm-gate.md, MUST-DO 1) and runs both models correctly. As a bare dependency (gemm = "0.19",
features default) the tree is pulp, dyn-stack, raw-cpuid, rayon, half, once_cell, seq-macro, paste,
num-traits, num-complex (gemm workspace Cargo.toml @ `86102c5b`) — heavier than our zero-dep engine
but finite. License: nothing to carry (crate boundary; NOTICE credit suffices). Maintenance: zero —
upstream owns dispatch, tail handling, and arch coverage; we keep our F16/Q8_0 residency and do
per-matmul dequant exactly as the candle prototypes' `convert` mode already does
(proto-wasm-gate.md). Threads: gemm takes a `Parallelism` arg (rayon-backed).

**Performance option A leaves on the table** (all measured, same i5-14400 machine/day):
- m=45 (Laya): gemm-stack best is 240.9 ms (candle upcast, 4T) vs engine 215.8 @4T / **163.6 @12T**
  — 0.90× at candle's own optimum, 0.36× at 12T where gemm/rayon work-split regresses past 4 threads;
  every candle mode loses to the engine at every thread count (proto-candle-laya.md latency table).
  Microbench cause: gemm f32 ≈55–60 GFLOPS at m=45 vs **350 GFLOPS at m=360 and 431 at m=1024**
  (proto-candle-laya.md, "gemm targets training-shaped (large-m) GEMMs"); f16 is worse than f32 on
  AVX2 (73–105 vs 111–155 GFLOPS) because packing-side conversion never fuses like our kernel.
- m=161 (OpenThai): gemm-stack reaches rough parity — candle convert 719.8 ms @4T vs engine
  765.0 ms @12T (0.94×–1.36× same-thread up to 4T, losing at 12/16T; proto-openthai-gdn.md). Thin-m
  pain shrinks as m grows.
- wasm32: option A is **already the measured baseline** — Laya convert 1551.5 ms, OpenThai Q8_0
  5942.8 ms, f16-gemm 18843.7 ms single-thread, with f32 SIMD128 at 38.6 GFLOPS
  (proto-wasm-gate.md). B's realistic wasm win is the pack-pass elimination and fused dequant —
  plausibly tens of percent, not multiples; the 4.4× f16 cliff is already avoided by convert mode.
- aarch64: **no data.** gemm's dispatch/tiling structure (Q1) suggests its generic path is
  serviceable but nobody has measured m=45/161 on Apple Silicon through it. This is the one
  bench that must precede the A-vs-B commitment for the native-Apple tier.

**Recommendation.** Option A is the right default for bring-up, CI, and any deploy target where
latency is not the product (async/batch wasm already ships measured and correct). B earns its cost
exactly where the engine's x86-64 edge matters: sustained interactive serving on Apple Silicon at
m≈45 (where option A starts at a 0.90×/4T-optimum ceiling with a 12T regression baked in) and the
wasm pack-pass elimination. Given effort M vs the zero-cost fallback, sequence it: ship A behind the
kernel-enum seam, bench gemm on target aarch64 hardware, and only then spend the M on the NEON lift
if the numbers say the thin-m deficit matters there. The seam is cheap either way: our `run_kernel`
dispatch (src/tensor.rs:135-149) already hides the kernel behind one call site.

## Open questions

- gemm publishes no thin-m benchmarks for aarch64 or wasm32; the m=45/161 numbers on Apple
  Silicon/NEON must come from our own microbench before A-vs-B is decided (both options are
  benchable in a day through the candle prototype harness).
- Exact `RelaxedSimd` availability in shipping wasm engines (pulp prefers it over Simd128,
  pulp/src/wasm.rs Arch::new ~2440-2451) — V8 ships relaxed-simd behind a flag, so Simd128 is the
  de-facto floor; confirm no engine we target dispatches RelaxedSimd with different semantics.
- pulp `Neon` gates only on the baseline `"neon"` feature (core_arch/aarch64/mod.rs:44); whether
  `vfmaq_f32` needs explicit feature enabling under older Rust targets — verify at implementation
  time (file read at file granularity only: not pinned).
- cubecl license at the pinned commit (not pinned; moot for B).
- Whether stage C's 4-column (NR=4) weight batching pays for m=45 given our L2-resident weights —
  measure before building.
