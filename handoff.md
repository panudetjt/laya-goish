# handoff.md — laya-goish project state

Updated: 2026-09-30 · branch `feat/benchmark` · git `bf63886` + `464aa2c`
Reader: this is the "continue from here" document. Facts below are backed by
runs on this machine (WSL2, rustc 1.96) unless marked unverified.

## What this project is

laya-goish = a std-Rust, zero-runtime-dep rewrite (RIIR) of the goish engine:
GGUF decision-classification inference for two models —

- **Laya** (`models/laya_english_f16.gguf`) — English email triage (5 choices + noul heads)
- **OpenThai** (`models/openthai_systemone_v03_f16.gguf`) — Thai classifier with 18 Gated-DeltaNet (GDN) layers

Architecture map, decisions, and the full ledger live in `wayfinder/`
(tickets 01–24 + research notes). Ticket 18's doc is
`wayfinder/research/neon-kernel-c.md`; the katgpt integration contract is
`wayfinder/research/katgpt-rs-capability-index.md` (now with an adoption
ledger).

## Where the project stands

### Done and verified

- **Ticket 17 CLOSED** — M1 (aarch64) acceptance: all 6 drift gates PASS.
  Latency @ 4 threads: Laya F16 261.5 ms, OT F16 1404.8 ms, OT Q8_0 1578.1 ms.
  `wayfinder/research/m1-acceptance-run1.md`.
- **katgpt-rs integration** — `katgpt-types` path dep (T0 leaf, wasm-clean);
  softmax uses `simd_exp_sum_inplace` **on aarch64/wasm32 only** (see the
  rule-3 incident below). blake3 (transitive) pinned `pure` at our dep edge —
  the engine stays C-compiler-free on every target.
- **Ticket 18: implementation complete + M1 run #1 done (2026-09-30)**.
  aarch64 tier v2 = 16-row MR (default) + `fmlal`/`sdot` A/B variants
  (`LAYA_AARCH64_KERNEL={scalar,fcvt,fmlal,sdot}`). M1 run #1 paste-back:
  fcvt drift **6/6 PASS on hardware** (identical to the emu table — TCG is
  an exact oracle), full bench table in ticket 18. **fmlal REJECTED**
  (slower everywhere: Laya F16 t1 +25%, OT F16 t1 +50% — the FCVT was
  already amortized by the 16-row grid; katgpt's Issue-201 premise doesn't
  transfer). **sdot = 1.75× on OT Q8_0** (976.6 vs 1711.1 ms @ t8; now
  beats OT F16) but stays non-default: Laya Q8_0 breaches the absolute
  action-head dact bar under block-quantized activations (emu table) —
  owner decision pending between (i) relative-dact criterion, (ii)
  OpenThai-scoped sdot, (iii) research-only. **Remaining work**: re-run
  `VARIANTS="fmlal sdot" bash engine/aarch64/run-m1.sh` on the M1 with the
  FIXED script (run #1's drift stage lost every non-fcvt decide to an
  `env -e` typo — BSD env rejects `-e`; bench numbers were valid) to get
  the hardware drift rows, then the owner's scope decision closes 18.
  run-m1.sh now checks decide's exit code, captures per-case stderr, and
  sets pipefail.

### The rule-3 incident (read this before touching kernels)

The 2026-09-28 katgpt softmax swap was committed (bf63886) with
tolerance-gate evidence only and **breached the x86-64-vs-goish
bit-identity contract** (CI drift/musl tiers: 42-combo BREACHes — the full
matrix had not been run between the swap and 2026-09-30). Fixed by cfg-
splitting softmax: x86-64 keeps the goish libm form verbatim (bit-identity
re-proven, drift+musl PASS), aarch64/wasm32 keep the katgpt kernel.
**Standing rule: a kernel swap is verified only when `engine/ci/run.sh full`
passes** — tolerance drift is not a substitute. Recorded in the capability
index's adoption ledger.

### Known pitfalls (don't rediscover these)

- Running the engine binary built via bare `cargo build` from the repo root
  **segfaults at startup** — the repo-root `.cargo/config.toml` injects
  no_std link flags. ALWAYS build/run through `engine/run.sh`
  (`CARGO_ENCODED_RUSTFLAGS="-Ctarget-cpu=x86-64"`).
- `cargo test` in `engine/` SIGSEGVs (custom `#[global_allocator]` under the
  test harness). The real test surface: `./run.sh self-test` + CI gates +
  `engine/wasm/run.sh smoke`, NOT `cargo test`.
- **docker on this machine**: the WSL distro's `docker` shim is broken; use
  `docker.exe` (Docker Desktop's Windows binary, on PATH) and launch Docker
  Desktop from `/mnt/c` first. The emulated arm64 CPU (`--platform
  linux/arm64`) exposes `asimdfhm` + `asimddp`, so fmlal/sdot are fully
  exercisable off-M1 via `engine/aarch64/emu-drift.sh`. Emu timings are
  meaningless (map rule) — correctness only.
- ARMv8.2 asm (`fmlal`/`sdot`/`fcvtn`) only assembles inside a fn with the
  matching `#[target_feature]` — LLVM rejects it at `-Ctarget-cpu=generic`
  even in `asm!` blocks (that's why the variant loops live in
  `nblock_fmlal`/`nblock_sdot`).
- **Per-block scratch buffers must be block-relative** — the first panel
  version indexed with absolute rows and segfaulted on m > 16 while every
  single-block self-test arm passed; the 19-row multi-block arms now guard
  this class. Lesson recorded in `neon-kernel-c.md`.
- CI needs rustup targets: x86_64-{gnu,musl}, aarch64-{gnu,musl},
  wasm32, and the three cross-check triples (windows-msvc + 2×apple-darwin)
  — `rustup target add` them on a fresh machine or `ci/run.sh full`
  fails in `cross`.

## Open tickets (order to work)

1. **#18 (C: NEON tuning)** — implementation done; **owner M1 A/B run +
   paste-back closes it**. Optional inherited item: candle bar.
2. **#20 (Ship v0.2)** — blocked by #18's close only. `engine/ci/package.sh`
   produces the artifact sets; note the Cargo.toml now carries the katgpt
   path dep + blake3 pin (packaging on machines without
   `~/work/my-project/katgpt-rs` needs the git rev-pin switch — capability
   index rule 5).
3. **#19, #23** — wasm kernel BC + windows drain follow-ups.

## Uncommitted working tree (as of this doc)

Clean on `feat/benchmark` (ticket-18 work committed as `464aa2c`). Only
`handoff.md` + `.zcodeignore` are untracked, and `engine/aarch64/results-emu/`
holds transient emu outputs (untracked, like the M1 `results/`).

## Quick commands

```bash
cd engine
./run.sh build                       # build release the CORRECT way (see pitfall)
./run.sh self-test                   # engine self-test gate
./ci/run.sh quick | full             # CI tiers (full = the bit-identity gate)
RUSTFLAGS="-Ctarget-cpu=generic -Clinker=rust-lld" \
  cargo build --release --target aarch64-unknown-linux-musl
bash aarch64/emu-drift.sh            # emu correctness: 6 cases × variants
cd wasm && ./run.sh build && ./run.sh smoke             # wasm32 gate
# on the M1 (owner):
bash engine/aarch64/run-m1.sh        # drift + bench per variant, A/B decision
```

## Key cross-repo facts

- `~/work/my-project/katgpt-rs` — 35-crate workspace, edition 2024, pinned
  rustc 1.98.1 via `rust-toolchain.toml`. `BOUNDARY.md` = upstream contract
  (modelless-only; dependency direction: laya-goish → katgpt-rs, never
  reverse). Capability index + adoption ledger:
  `wayfinder/research/katgpt-rs-capability-index.md`.
- laya-goish compiles clean on rustc 1.98.1; edition 2021 stays.
