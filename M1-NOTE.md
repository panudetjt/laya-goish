# laya-goish — M1 acceptance transfer (ticket 18 A/B) — ESSENTIALS

Extract the zip anywhere: it creates TWO side-by-side folders —

    <wherever>/laya-goish/    the engine tree (run everything from here)
    <wherever>/katgpt-rs/     vendored katgpt-types path dependency

The engine's Cargo.toml resolves the dependency as `../../katgpt-rs/...`
(from engine/ two levels up), so the side-by-side layout is REQUIRED — do
not flatten or move either folder. Needs crates.io on the first build
(cargo fetch); nothing else external.

## Prereqs
- rustup stable with **rustc >= 1.85** (`rustup update stable`) — the vendored
  katgpt-types crate is edition 2024. (Ticket 17 ran on 1.83; too old now.)
  If you must not touch the machine's default stable, drop this next to the
  engine instead — rustup downloads it on first use, tree-scoped only:
  `laya-goish/rust-toolchain.toml` containing
  `[toolchain]` / `channel = "1.98.1"`.
- python3 (drift gates + bench summarization), Xcode Command Line Tools
- ~4 GB free (models + build target dir)

## MODELS ARE NOT IN THIS ZIP (~3.6 GB, unchanged since ticket 17)

Place the four .gguf files at the paths the script defaults to (relative to
laya-goish/):

    models/laya_english_f16.gguf
    models/laya_english_q8_0.gguf
    models/openthai_systemone_v03_f16.gguf
    models/openthai_systemone_v03_q8_0.gguf

If the MacBook still has them from the ticket-17 run (laya-models-only.zip —
checksums recorded in wayfinder/research/m1-acceptance-run1.md), reuse those
copies in place, or point the four env vars (LAYA_F16 LAYA_Q8 OPENTHAI_F16
OT_Q8) at them.

## Run (one command, from laya-goish/)

    cd laya-goish
    bash engine/aarch64/run-m1.sh

Stages: build (aarch64-apple-darwin) -> self-test -> drift gates per kernel
variant (fcvt / fmlal / sdot) vs the shipped x86-64 references in
engine/aarch64/reference/ -> latency bench per variant -> summary table to
paste back into wayfinder/tickets/18-c-neon-tuning.md.

Knobs (env):

    VARIANTS="fcvt fmlal sdot"     # the A/B set (default)
    TARGET_CPU=apple-m1            # chip-tuned codegen A/B (default generic)
    THREADS="1 4 8 10"             # 1 warmup + 8 measured per count
    CANDLE=1                       # optional candle-aarch64 reference bar
    LAYA_F16 LAYA_Q8 OPENTHAI_F16 OT_Q8   # model path overrides

## Layout notes

- katgpt-rs/crates/katgpt-types/ is vendored at the exact relative path the
  engine's Cargo.toml path-dep expects — see the layout above. The crate
  manifest is self-contained (no workspace inheritance).
- Do NOT run bare `cargo build` inside engine/ expecting the CLI: use
  run-m1.sh (it sets the target + rustflags). There is intentionally no
  .cargo/config.toml in this transfer (the dev machine's one is x86-64
  scoped and would only confuse an M1).
- Ticket 18 question + decision rule: also summarized in
  engine/aarch64/README.md (shipped here). Expected per the off-M1 emulation
  table: fcvt passes all six gates; fmlal and sdot pass every gate except
  Laya's action-head dact (arithmetic, not noise — it will reproduce on the
  M1). The latency table is the new information this run adds; adoption is
  decided against it per the ticket's decision rule.
