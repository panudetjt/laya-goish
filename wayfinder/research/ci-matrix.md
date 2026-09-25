---
wayfinder: research
ticket: wayfinder/tickets/14-release-ci-matrix.md
artifact: engine/ci/ (run.sh, smoke.py, reduced-drift.py), .github/workflows/engine-ci.yml, engine/src/lib.rs, engine/src/tensor.rs (cfg seams), engine/Cargo.toml ([lib])
date: 2026-09-25
machine: i5-14400 (6P+4E, 16 threads), WSL2, rust 1.96 (mise)
status: resolved
---

# Release/CI matrix v0 — 5 targets from day one (ticket 14)

One command now proves all five deploy targets of the std engine still
build and the runnable ones pass their gates:

```bash
./engine/ci/run.sh            # quick tier (default): build + smoke + reduced drift
./engine/ci/run.sh full       # everything, incl. the two 42-combo drift gates
./engine/ci/run.sh list       # all tiers
```

Every tier prints PASS/FAIL (or SKIP with a reason when a precondition is
absent — models, reference binary, qemu) and the run exits non-zero on any
FAIL. `.github/workflows/engine-ci.yml` calls the same runner, so CI and
local execute identical tiers.

## Verdict at a glance

| Tier | What it runs | Runtime (measured) | Status |
|---|---|---:|---|
| `build` | gnu release via `engine/run.sh` | 13.5 s cold (deps incl.), ~0 s warm | **PASS** |
| `smoke` | CLI surface gate (`engine/ci/smoke.py`: self-test, 10 presets, invalid CLI/GGUF, daemon recovery, repeated requests, 1/4-thread parity, numeric-state parity) | 3 s | **PASS** |
| `drift` | frozen `validation/scalar-tail-drift.py`, 42 combos (6 cases × threads 1/2/4/5/7/12/16), `./laya` vs engine gnu | 164 s | **PASS — 42/42 bit-identical** |
| `drift-quick` | reduced drift (`engine/ci/reduced-drift.py`, 2 cases × 2 threads, default laya:refund + openthai:openthai-thai @ 1/4 t) | 13 s | **PASS** |
| `quant` | frozen `validation/quantized-drift.py` Q8_0 gates | 17 s | **PASS** (maxima = §8.3 reference: 0.01782 / 0.01811) |
| `musl` | musl release build + `file` static-pie assert + 42-combo drift on the musl binary | 179 s (build 9.1 s + drift ~170 s) | **PASS — static-pie linked, 42/42** |
| `wasm` | `cargo check --target wasm32-unknown-unknown` (lib + bin) with the getrandom custom-backend cfg recipe from prototypes/wasm-gate | 2.2 s | **PASS (compile-check)** |
| `cross` | `cargo check` for x86_64-pc-windows-msvc, x86_64-apple-darwin, aarch64-apple-darwin, aarch64-unknown-linux-gnu | ~2.2 s/triple | **PASS ×4 (compile-check)** |
| `lint` | `cargo fmt --check` + `cargo clippy -D warnings` on engine/ ONLY | ~1 s warm | **PASS** |
| `qemu` (optional) | aarch64-musl static build (`rust-lld`), then self-test + one decide under `qemu-aarch64-static` | build 10.5 s | **SKIP** (no qemu-user-static on this machine; build path proven) |
| `quick` (default) | build + smoke + drift-quick | **15 s warm / ~30 s cold** (budget < 3 min) | **PASS** |
| `full` | all of the above at full depth | **6 m 03 s** | **PASS** |

The 42-combo gate ran exactly twice (gnu + musl, the runtime budget):
both PASS after the engine changes below, so bit-identity to `./laya` is
re-proven at ticket-14 close, not inherited from ticket 12's word.

## Engine changes required to make five targets compile

The engine was bin-only and hard-wired x86_64 (`use core::arch::x86_64::*`
plus unconditional `#[target_feature(enable = "avx2,fma,f16c")]`). Three
minimal changes, all inside `engine/`:

1. **`engine/src/lib.rs` (new) + `[lib] laya_engine` in Cargo.toml.** The
   lib mirrors main.rs's module list 1:1 (plus a duplicated `format!`
   macro — `#[macro_export]` lives per crate root); the bin is untouched
   and still compiles its own copies, so the shipped CLI cannot drift. The
   bin's `#[global_allocator]` (heap.rs, compiled in both crates) resolves
   by the standard root-crate-wins rule — outputs re-proven identical.
   Keep the lib.rs mod list in sync with main.rs.
2. **`tensor.rs` cfg seams** at exactly the dispatch seam the map pinned
   (`run_kernel`, plus `norm_chunk`/`dot_kernel`): the AVX2/F16C kernels
   are now `#[cfg(target_arch = "x86_64")]` (x86 codegen unchanged), and
   non-x86 targets get scalar reference kernels — same per-output
   arithmetic as `self_test`'s reference. `norm`'s scalar form is
   bit-identical to the textbook form by construction; `dot`'s sequential
   sum differs from the x86 8-partial reassociation, which is fine:
   cross-ISA outputs are judged by the tolerance gate (map decision), and
   tickets 17–19 replace these stubs with NEON/simd128 kernels at the
   same seam. `require_simd` returns Ok on non-x86 until those land.
3. **one-time `cargo fmt`** over engine/src (12 hunks in 6 files,
   whitespace-only). Ticket 12 kept goish's verbatim formatting, which is
   not rustfmt-clean; a fmt gate needs a normalized baseline. Bit-identity
   after the reformat is re-proven by both 42-combo runs above.

The wasm tier sets `RUSTFLAGS='--cfg getrandom_backend="custom"'` per the
prototypes/wasm-gate recipe. The engine's dep tree currently pulls **no
getrandom at all** (engine/Cargo.lock), so the cfg is inert today; it is
kept in the tier so the gate cannot rot when ticket 19's wasm runtime work
adds a dep that needs it (the `__getrandom_v03_custom` stub then belongs
in the lib, per the prototype's src/lib.rs).

## Lint policy (engine only; goish `src/` is a frozen reference, never linted)

- `cargo fmt --check`: hard gate since the normalization above.
- `cargo clippy --all-targets -- -D warnings` with a documented allow-list
  (see `engine/ci/run.sh lint`): `dead_code` (registry/`validate` are
  ported-but-unused until ticket 13) and 9 clippy style lints
  (`manual_clamp`, `excessive_precision`, `manual_is_multiple_of`,
  `manual_range_contains`, `needless_range_loop`, `len_zero`,
  `too_many_arguments`, `unnecessary_to_owned`, `redundant_closure`) —
  every one fires on verbatim-port code (kernel signatures, loop shapes,
  byte-copied constants) that must not be modernized. Anything not listed
  still hard-fails, so tickets 17–19 kernels get full clippy scrutiny.

## GitHub Actions (`.github/workflows/engine-ci.yml`)

Manual (`workflow_dispatch`, with a `fetch_models` checkbox) — push/PR
triggers are commented in the file; the owner flips them on to enable.
Jobs, all ubuntu-latest, each calling the local runner:

| Job | Runs |
|---|---|
| `lint` | `run.sh lint` |
| `x86-64` | `run.sh build smoke drift quant` |
| `musl` | `run.sh build` then `REF_BIN=<engine-gnu> run.sh musl` |
| `wasm32` | `run.sh wasm` |
| `cross` (matrix ×4) | `run.sh cross:<triple>` |

**Honest limitations, documented in the workflow header:**

- Models are external artifacts (not in git). Without them the runtime
  tiers print SKIP and CI proves the build/static-pie/compile surface only
  (verified: the SKIP path exits 0 — simulated with all model paths
  pointed at /nope). `fetch_models=true` downloads the Laya F16/Q8_0
  weights via `scripts/download-models.sh`; the OpenThai GGUFs are
  conversion artifacts (docs/openthai.md) and unpublished, so the 42-combo
  drift tiers stay **local-only** (`run.sh full` on this machine) until
  they are published.
- The drift reference `./laya` is not in git. In CI the musl job therefore
  drifts **engine-gnu vs engine-musl** (`REF_BIN` override) — a real
  target-codegen drift gate, weaker than the reference gate. A commented
  step shows how to build the frozen goish reference from `src/` and run
  the full `./laya`-vs-engine drift in CI instead.
- Compile-only jobs never need linkers or SDKs: `cargo check` for
  windows-msvc and both apple-darwin triples runs on ubuntu (no linking
  step), which is the point of the check-only tier.

## QEMU aarch64 tier (optional, currently SKIP)

`run.sh qemu` builds `aarch64-unknown-linux-musl` with
`RUSTFLAGS="-Ctarget-cpu=generic -Clinker=rust-lld"` (self-contained musl
+ bundled lld; **proven on this machine**: static ARM aarch64 ELF, 10.5 s)
and, when `qemu-aarch64-static` exists, runs `self-test` + one refund
decide under emulation (correctness only, never perf — map decision).
qemu-user-static is not installed here, so the tier self-skips with a
note; the emulation half is written but unexercised on this machine.

## How this is the safety net for 17/18/19

- Every kernel PR: `./engine/ci/run.sh quick` locally (~15 s), `full`
  before landing (6 min, two 42-combo gates = the bit-identity contract on
  x86-64).
- NEON work (17/18) develops against `cross:aarch64-unknown-linux-gnu`
  plus the qemu tier once qemu is installed; acceptance stays on the
  owner's M1 (ticket 16), where the same runner works unmodified.
- wasm runtime (19): the `wasm` compile gate already carries the
  getrandom recipe; the runtime gate plugs in as a new tier beside it.

## Open items

1. Publish OpenThai GGUFs (or a conversion step in CI) → CI drift tiers
   activate without code changes (runner already resolves model paths).
2. Uncomment push/PR triggers when the owner wants mandatory gating.
3. Install qemu-user-static (one apt package) to light up the qemu smoke
   half of the aarch64 tier.
4. Ticket 13 (serve) will un-`allow` `dead_code` for registry/validate.
