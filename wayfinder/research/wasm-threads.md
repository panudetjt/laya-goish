---
wayfinder: research
status: closed
ticket: wayfinder/tickets/24-wasm-threads.md
artifact: engine/src/team.rs (wasm_threads arm), engine/src/heap.rs (SpinLock + serialized small path), engine/wasm/ (patch-shared.mjs, harness.mjs --threads, run.sh build-threads/smoke-threads/bench-threads), engine/wasm/results/*_t{2,4,8}.json
date: 2026-09-25
machine: i5-14400 (6P+4E), WSL2, rust 1.96.0 (ac68faa20 2026-05-25), Node v22.22.3 (V8)
reference: wayfinder/research/wasm-kernel-bc.md (ticket 19 single-thread numbers + candle bars)
policy: wayfinder/tickets/11-wasm-format-policy.md (Q8_0 judged vs F16 native)
---

# wasm threads (SAB): the OpenThai bar CLOSES — GO

**Verdict up front: GO.** Host-driven wasm threads work on wasm32-unknown-unknown
+ atomics under Node 22 without wasm-bindgen or emscripten, and at K≥2 workers
the engine's OpenThai latency beats the candle reference on both formats with
bit-identical drift to the single-thread engine:

| Configuration | 1-thread (t19) | K=2 | K=4 | K=8 | candle bar | verdict |
|---|---:|---:|---:|---:|---:|---|
| OpenThai F16 (161 tok, 151 GFLOP) | 8232 ms | **4690 ms** | **2761 ms** (2nd session 2849) | 2241 ms | ≤ 6194.5 ms | **PASS from K=2 (−24%)** |
| OpenThai Q8_0 (161 tok, 159 GFLOP) | 7317 ms | **4209 ms** | **2500 ms** | — | ≤ 5942.8 ms | **PASS from K=2 (−29%)** |
| Laya F16 (45 tok, 36.3 GFLOP) | 1536.5 ms | **930 ms** | 980 ms | — | ≤ 1551.5 ms | no regression (bar already met) |

Protocol unchanged from ticket 19: 1 warmup + 5 measured (K=8 and repro
sessions: 3 measured), inside-wasm timer. Ambient 1-min load 1.3–4.5 during
the runs — the same regime as ticket 19's honest caveat, and the margins
(−24% at worst) are far outside that noise. Scaling: 1.76×/2.98×/3.67× at
K=2/4/8 on OpenThai F16 — K=4 is the sweet spot on this 6P+4E part (K=8
still gains 22% over K=4 but at 46% marginal efficiency, and Laya's small
fan-outs don't pay for K=4: 930→980 ms, noise-adjacent).

## What the platform actually provides (all verified empirically)

Five hard findings define the design space. Each is a one-liner to reproduce:

1. **`std::thread` does not function on wasm32-unknown-unknown, even with
   atomics.** Rust 1.96.0: `std::thread::Builder::spawn` compiles but returns
   `Err(io::Error)` — probe returned `operation not supported on this platform`
   (`std::sys::pal::unsupported`). Not a panic; a clean unsupported error.
   There is no spawn mechanism in the module: the host must create agents.
2. **`-Ctarget-feature=+atomics` compiles on stable but does NOT publish
   `cfg!(target_feature = "atomics")`** (unstable feature; rustc warns
   "unstable feature specified for `-Ctarget-feature`" and the cfg reads 0).
   Code gating needs its own `--cfg wasm_threads` — the same RUSTFLAGS-cfg
   pattern run.sh already uses for getrandom. Atomic RMWs/loads/stores ARE
   emitted (cross-agent coherence proven below).
3. **wasm-ld only marks an imported memory shared under `--shared`, and
   `--shared` demands PIC-built std** ("recompile with -fPIC" against the
   distributed rlibs — i.e. nightly `-Zbuild-std`; this is exactly the wall
   that makes wasm-bindgen-rayon want a post-processing step). Workaround
   shipped here: build with `--import-memory --max-memory=4294967296`, then
   byte-patch the import's limits flags `0x01 → 0x03` (shared+max) —
   length-preserving, `engine/wasm/patch-shared.mjs`.
4. **std's wasm32 dlmalloc is not thread-safe** (no surprise: std for this
   target assumes one agent). 3-instance concurrent alloc/free of 16 KiB
   pattern-verified blocks corrupts the arena: `RuntimeError: memory access
   out of bounds at dlmalloc::unlink_chunk` / panic in `__rdl_dealloc`;
   single-agent identical loop passes 4000/4000.
5. **std's `std::sync::Mutex` on this target is the `no_threads` fake** —
   the distributed std rlib is built without atomics, so contending agents
   hit `panic: assertion left == right failed: cannot recursively acquire
   mutex ... sys/sync/mutex/no_threads.rs:19`. (A microsecond-scale
   3-agent mutex probe "passed" by never actually overlapping — the
   engine's first threaded forward exposed it within one model.) The
   atomics build therefore uses a real `compare_exchange` spinlock
   (heap.rs `SpinLock`) for both allocator paths.

Two more constraints shaped the bootstrap (probe-verified, then inherited):

- **`core::arch::wasm32::memory_atomic_wait32/notify` are unstable**
  (`stdarch_wasm_atomic_wait`) on stable 1.96 — Rust-side futex parking is
  unavailable. Parking lives in JS instead: workers `Atomics.wait` on the
  GEN word of the shared linear memory; the module imports
  `env.futex_wake` (JS `Atomics.notify`). This is the emscripten pthread
  shape, ~60 lines of host JS instead of a framework.
- **Active data segments re-apply per instantiation**, so a worker
  instantiating after the coordinator mutated statics wipes them (probe:
  3-agent counter lost exactly one agent's increments to a late
  instantiation). Discipline: the harness instantiates every worker and
  gates on `wasm_team_ready()` before the first wasm call. Consequence:
  workers cannot be respawned after a crash without resetting engine
  state — acceptable for a batch harness, noted for any future serving
  work.

What DOES work, and carries the design: one shared
`WebAssembly.Memory({shared:true, maximum:65536})`, the module instantiated
once per agent (statics live in linear memory ⇒ common to all instances),
per-instance `__stack_pointer` reassigned by the host into private 2 MiB
carve-outs (exported via `--export=__stack_pointer`), and `core::sync`
atomics executing as real shared-memory atomics in V8. A minimal probe
team (2 workers + caller, GEN/DONE protocol with JS futex wake) ran the
fan-out protocol correctly across repeated generations before any engine
wiring was attempted.

## What was built (all under the existing change surface)

- `engine/src/team.rs` — `wasm_threads` arm: the native team's job-cell
  GEN/ACTIVE/DONE protocol re-expressed for host-driven workers. `fanout`
  publishes the job, bumps GEN (SeqCst), calls the imported
  `env.futex_wake`, runs range 0, and joins on DONE; falls back to the
  existing sequential loop when fewer than count−1 workers registered
  (single-instance use of the threads artifact stays correct). New ABI:
  `wasm_worker_hello` (register + baseline), `wasm_worker_poll` (one wake
  iteration), `wasm_team_gen_addr` (host parks on it), `wasm_team_ready`.
  The no-atomics arm and the x86 arm are untouched (cfg-split).
- `engine/src/heap.rs` — wasm Router: `BIG_LOCK` and the new small-block
  `SMALL_LOCK` become a real `SpinLock` in the `wasm_threads` build (std
  Mutex panics under contention there); the no-atomics build keeps std
  Mutex, byte-for-byte the old behavior. Lock order is one-way
  (BIG → SMALL, never reverse), no recursion.
- `engine/wasm/src/lib.rs` — `wasm_set_threads(k)` (threads artifact only)
  + `THREADS` feeding `Model::new(g, threads)` in `wasm_load`; default
  artifact never writes it, so its models stay 1-threaded.
- `engine/wasm/patch-shared.mjs` — the shared-memory import patch (see
  finding 3).
- `engine/wasm/harness.mjs` — `--threads K`: creates the shared memory,
  instantiates, spawns K−1 Node workers (inline worker source: instantiate
  same bytes + memory, re-point `__stack_pointer`, hello, then the
  Atomics.wait/poll loop), gates on `wasm_team_ready`, then runs the
  existing measurement path unchanged (same drift machinery, same
  inside-wasm timer; `linear()` follows the shared memory).
- `engine/wasm/run.sh` — `build-threads` (atomics + import-memory +
  wasm_threads cfg + patch), `selftest-threads [K]`, `smoke-threads [K]`
  (self-test under workers + one Laya forward behind the drift gate),
  `bench-threads [K]` (OT F16 + OT Q8_0 + Laya, 5 repeats each, results to
  `results/*_tK.json`). Default `build`/`smoke`/`bench` untouched.
- `engine/Cargo.toml` + `engine/wasm/Cargo.toml` — check-cfg registration
  for `wasm_threads` so the lint tier stays green.

Zero new Rust dependencies (the zero-dep constraint holds); the host side
is Node stdlib only (`worker_threads`, `Atomics`).

## Thread-safety audit (what the workers actually touch)

- Allocator: both paths serialized (SpinLock) — the only shared mutable
  engine state the fan-out closures allocate through. Probe + 5×161-token
  forwards at K=4 clean.
- `tensor.rs` `WASM_SIMD128`/`F16_HAS_INF`: AtomicBool written by the
  coordinator before workers exist, read by workers after the GEN
  acquire/release edge — ordered, and atomic anyway.
- `lib.rs` statics (models, OUT/OUT2, stats): coordinator-only, exactly as
  the native team serializes inference through one decide path.
- Worker stacks: disjoint 2 MiB regions grown before load; the allocator's
  `memory.grow` appends beyond them, never into them.

## Gates

Drift vs the same fresh native dumps as ticket 19 (threading is
range-disjoint, so outputs are element-for-element identical to the
sequential engine — the numbers confirm):

| Wasm run (threads) | Native ref | max \|Δlogit\| | max \|Δp\| | choice flips |
|---|---|---:|---:|---:|
| OT F16 K=2 / K=4 | ot F16 | 1.0e-5 | 7.9e-8 | 0 |
| OT Q8_0 K=2 / K=4 | ot F16 (t11 gate) | 6.7e-2 | 4.7e-4 | 0 |
| Laya F16 K=2 / K=4 | laya F16 | 2.4e-7 | 0 | 0 |

Identical to ticket 19's single-thread values to the digit — 0 flips
everywhere, |Δp| ≤ 4.7e-4 vs the 0.02 gate. Memory: peak linear 2.399 /
2.403 / 2.412 GiB (OT F16 at K=2/4/8) vs 2.382 single-thread — the +17–30
MiB is worker stacks + shared granularity; every configuration stays far
under the 4 GiB ceiling and under candle's 2.861 GiB. `wasm threads smoke`
and `wasm threads self-test` PASS at K=2.

## CI / regression status

- `engine/ci/run.sh quick` — PASS (build + smoke + drift-quick; x86 paths
  untouched by the cfg split).
- `engine/ci/run.sh wasm` — PASS (default no-atomics compile check + cdylib
  + self-test + Laya forward drift gate).
- `engine/ci/run.sh lint` — PASS (fmt + clippy; `wasm_threads` registered
  via check-cfg). Clippy under the threads cfg adds no source warnings
  over the default wasm set (the only delta is rustc's own "unstable
  target-feature" notice inherent to any atomics build).
- Default artifact imports/exports unchanged (`env.now_ms` only; no worker
  symbols) — the CI tier's artifact is not altered by this ticket.

## Recommendation for ticket 20 (ship-note wording)

The ticket-19 decision "candle STAYS for OpenThai-class wasm" was
conditional on single-thread. With K≥2 workers the engine now beats candle
on every measured axis — latency (both formats, from K=2), memory (always),
drift (bit-identical to the shipped engine), dependencies (zero vs
candle+wasm-bindgen's tree). Suggested wording:

> wasm serving of the OpenThai class uses the engine's threads build when
> the host can provide a shared-memory worker pool (Node 22 today; browsers
> need COOP/COEP for the SharedArrayBuffer). K=2 already clears the candle
> reference on both formats; K=4 is the default recommendation (2.8 s F16 /
> 2.5 s Q8_0 on the reference part, 3.0× single-thread). Single-threaded
> wasm hosts keep ticket 19's status quo: engine for Laya-class, candle
> stopgap for OpenThai-class. The threads path adds no dependencies; its
> costs are host-side (worker bootstrap) and the spawn-once discipline
> (workers must instantiate before the first load).

## Explicitly untested / out of scope

- Browsers (needs crossOriginIsolated; the harness bootstrap is Node-shaped
  but mechanism-identical), Safari/JSC, Firefox SpiderMonkey.
- Dual-model (laya+OT resident) under threads; serving/soak under threads;
  worker-crash behavior (no respawn without state reset — see segment
  re-init above); K>8; atomics + simd128-off combination.
- The threads artifact is NOT wired into CI as a gate (build-threads is a
  manual tier; wiring it in is ticket-20-or-later platform work).
- wasm32v1-memory64, emscripten/wasm-bindgen comparison (unnecessary — the
  hand-rolled layer works; recorded because the ticket asked the framework
  question be answered honestly: nothing beyond Node stdlib is required).

## Reproduce

```bash
cd engine/wasm
./run.sh build-threads          # atomics artifact + shared-memory patch
./run.sh selftest-threads 4     # engine self-test inside wasm, 3 workers
./run.sh smoke-threads 2        # + one Laya forward, drift-gated
./run.sh bench-threads 2        # OT F16/Q8 + Laya at K=2 (5 repeats)
./run.sh bench-threads 4        # ... at K=4
# negative findings, standalone:
#   std::thread: rustc -O --crate-type=cdylib --target wasm32-unknown-unknown \
#     -Ctarget-feature=+atomics ... (spawn -> Err "operation not supported")
#   dlmalloc corruption / no_threads Mutex panic: 3 agents hammering
#     System alloc or std Mutex over one shared memory (see findings 4-5)
# tiers: ../ci/run.sh quick && ../ci/run.sh wasm && ../ci/run.sh lint
```

Raw numbers: `engine/wasm/results/{ot_f16,ot_q80,laya_f16}_t{2,4}.json`,
`ot_f16_t8.json`, `ot_f16_t4b.json`, `smoke_laya_t2.json`.
