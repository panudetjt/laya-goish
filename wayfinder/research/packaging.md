# Release packaging — artifacts + checksums (ticket 22)

One command — `engine/ci/package.sh [--skip-verify] [target …]` — produces
every artifact set this machine can produce under `engine/dist/` (the frozen
repo-root `dist/` is untouched). Default target list: `linux-musl wasm32
windows macos`. Verified end-to-end 2026-09-25, rustc 1.96.0, git
`3b743260e1fb` (branch `feat/benchmark`, read from `.git/HEAD` without git
commands), sources as of the run (concurrent heap/http/server edits in
flight — see Windows).

## What was produced (RESULT: PASS, ~40 s warm)

### `engine/dist/linux-x86_64-musl/` — PASS

- `laya-engine-0.1.0-linux-x86_64-musl.tar.gz` — **555,371 bytes** (outer
  SHA256SUMS next to it)
- staged tree (also kept unpacked for inspection): `laya-engine`
  (**1,159,968 bytes**, ELF static-pie, stripped), `presets.json`
  (10,586 B — reference copy; the binary embeds an identical one via
  `include_str!`), `USAGE.md` (quickstart: self-test / decide --preset /
  daemon / serve), `MANIFEST.md` (sizes, sha256s, rustc, git rev, recipe,
  verification record), inner `SHA256SUMS`
- recipe reused verbatim: `engine/run.sh build musl`
  (`CARGO_ENCODED_RUSTFLAGS=-Ctarget-cpu=x86-64`, `--target
  x86_64-unknown-linux-musl`), i.e. the CI `musl` tier's build; the
  42-combo drift gates were NOT rerun (ticket 14 owns them — same binary
  recipe)
- verification: `engine/ci/smoke.py` (the `quick` tier's smoke gate) run
  against the **staged artifact bytes**: PASS — self-tests, 10 presets,
  invalid CLI/GGUF rejection, daemon recovery, repeated requests,
  1/4-thread parity, numeric-state parity. Independently re-verified after
  the run: extracted the tar.gz to a temp dir, `sha256sum -c SHA256SUMS`
  OK, self-test PASS, one real `decide --preset email` produced a correct
  decision JSON.

### `engine/dist/wasm32/` — PASS

- `laya-engine-0.1.0-wasm32.zip` — **126,444 bytes** (deterministic zip via
  python3 `zipfile` — no `zip(1)` on this machine; fixed 1980 timestamps,
  sorted entries, deflate-9)
- contents: `laya_engine_wasm.wasm` (**342,385 bytes** cdylib) at
  `target/wasm32-unknown-unknown/release/…` — deliberately the harness's
  default `--wasm` path so `RUN.md` commands run verbatim after extraction;
  `harness.mjs` (ticket 19's Node harness, unmodified), `RUN.md`
  (self-test + drift-checked forward + bench pointers), `MANIFEST.md`,
  inner `SHA256SUMS`
- recipe reused verbatim: `engine/wasm/run.sh build` (`--cfg
  getrandom_backend="custom"`, wasm32-unknown-unknown)
- verification: `engine/wasm/run.sh smoke` PASS **before** packaging
  (self-test inside wasm + one Laya forward, drift gate max|dlogit| ≤ 0.05 /
  max|dp| ≤ 0.02); post-run: extracted the zip, `sha256sum -c` OK,
  `node harness.mjs --self-test` PASS on the extracted wasm
- note: wasm drift/bench flows need a native dump + GGUF; RUN.md documents
  generating it with the (linux artifact's) engine binary

### `engine/dist/windows-x86_64-msvc/` — link NOT possible here (recorded)

The ticket's open question is answered empirically, in two stages:

1. **Controlled attempt (stable sources)**: `CARGO_ENCODED_RUSTFLAGS=
   '-Ctarget-cpu=x86-64 -Clinker=rust-lld' cargo build --release --target
   x86_64-pc-windows-msvc` — **compile PASS** (all crates incl. the pure-Rust
   engine + memmap2), **link FAIL**:
   `rust-lld: error: could not open 'kernel32.lib': No such file or directory`
   (likewise ntdllib/userenv/ws2_32/dbghelp). rust-lld itself drives the
   MSVC-flavor link line correctly; rustup's rustlib for this target ships
   only Rust rlibs — **no `self-contained/` dir, no Windows import libs, no
   CRT**. Without a Windows SDK import-lib set the link cannot succeed on
   this machine. (Also: plain `RUSTFLAGS=` per the ticket sketch would
   concatenate the repo-root no_std link flags —
   `CARGO_ENCODED_RUSTFLAGS` is required, the `engine/run.sh` pattern.)
2. **During packaging runs**: the concurrent heap.rs agent's Windows
   allocator work lands mid-edit, so the windows target transiently fails
   at COMPILE (3 × E0599 on `System::alloc*`). The script distinguishes the
   two from the log and records which happened; the link verdict above is
   from stage 1 and is not affected.

The set therefore ships `MANIFEST.md` + `build-link.log` (exact failure) and
no `.exe`: **Windows artifacts require a Windows machine** — tickets 23
(runtime/drain wiring) and 20 (ship) own that; one plain `cargo build
--release` on Windows (rustup provides the MSVC libs there) or a Windows
SDK import-lib set + lld-link here closes it.

### `engine/dist/macos/` — note only

`MANIFEST.md` points at the two producer flows: aarch64 →
`bash engine/aarch64/run-m1.sh` on the M1 (ticket 17's script builds
natively and drops the binary in `engine/target/aarch64-apple-darwin/
release/laya-engine`); x86_64 → native `cargo build --release --target
x86_64-apple-darwin` on an Intel Mac. Both `cargo check`s are already
proven (CI `cross` tier) — only the Apple SDK is missing here, not code.
`package.sh` can be rerun ON a Mac to repackage those builds.

## No candle, no goish

No artifact set contains anything from `prototypes/candle-laya` or the
goish binary — engine builds only (`laya-engine` + wasm cdylib + docs).

## Script contract

- `engine/ci/package.sh [--skip-verify] [linux-musl wasm32 windows macos]`,
  default all four; every step prints `--- LABEL: PASS/FAIL`; exit non-zero
  if a requested set fails to build/verify/stage/checksum. The windows
  link-failure is the one *recorded-not-fatal* outcome (its artifact set is
  the note itself); a windows COMPILE failure is likewise recorded as
  transient-when-concurrent-edits-are-in-flight.
- Gates run against the **staged bytes**, not the build dir, so the smoke
  proves exactly what ships. Staging is asserted before any archive is
  built (no vacuous checksum passes).
- `--skip-verify` skips only the runtime smokes; builds, static-pie check,
  zip CRC, inner+outer checksum verification always run.
- Deterministic layout `engine/dist/<target>/…`; reproducible-ish archives
  (tar: `--sort=name`, uid/gid 0, mtime epoch 0, `gzip -n`; zip: fixed
  1980 timestamps). Byte-reproducibility is NOT claimed across runs: each
  MANIFEST embeds the build date (drop it from MANIFEST if bit-identical
  reruns are ever needed).
- Engine version comes from `engine/Cargo.toml` (currently 0.1.0 — bump to
  0.2.0 in ticket 20's release step and rerun; names embed the version).

## Ticket 20 handoff notes

1. **Linux musl + wasm32 are done** — re-run `package.sh` after the version
   bump (and after ticket 18 lands, since the aarch64 numbers go in the
   same release notes). Artifacts are gate-proven as of the run.
2. **Windows**: build on a Windows machine (plain `cargo build --release`),
   or supply SDK import libs + lld-link here; then `package.sh windows`
   packages the `.exe` set automatically (USAGE/MANIFEST/checksum path is
   already wired for the success branch). Ticket 23 owns runtime wiring.
3. **macOS**: run `run-m1.sh` on the M1 (also produces ticket 18's tuning
   baseline); Intel Mac native build for x86_64; repackage on the Mac.
4. The 42-combo drift gates were deliberately NOT run here (ticket 14
   owns them; the binaries are the same recipes those gates already
   proved). If ticket 20 wants a final gate pass before shipping, run
   `engine/ci/run.sh full` once on the release commit.
5. `engine/dist/` is generated output, not gitignored — decide in ticket
   20 whether to commit a build or add it to `.gitignore` (the frozen
   root `dist/` pattern `!/dist/.gitkeep` is the precedent).
