---
wayfinder: research
ticket: wayfinder/tickets/21-heap-return.md
artifact: engine/src/heap.rs
date: 2026-09-25
machine: i5-14400 (6P+4E, 16 threads), WSL2, rust 1.96 (mise)
status: resolved-in-scope (allocator retention fixed and bounded; dominant
soak term root-caused to engine/src/http.rs thread-stack retention — outside
this ticket's file scope, handed forward with evidence)
---

# Heap page-return — findings (ticket 21)

## Verdict

`heap.rs` now returns released pages to the OS and its retention is bounded
by construction (≤ 31.75 MiB worst case, ~2 MiB measured steady state in the
soak's quiet rounds). Latency parity and bit-identity hold.

But the headline +310 MiB was never primarily the allocator. Per-mapping
`smaps` evidence shows **~+300 MiB of it is connection-thread stacks in
`engine/src/http.rs`**: every mixed-round request opens a fresh connection,
the server spawns a 2 MiB-stack thread per connection, and finished threads
are never joined — `serve()`'s `kids: Vec<JoinHandle<()>>` only ever pushes
and counts (`live_conns` filters `is_finished()`; nothing joins or drops
finished handles), so every finished thread's 2 MiB pthread stack stays
mapped until process exit. That is outside this ticket's edit scope
(`heap.rs` only — http.rs/server.rs are owned by the concurrent
windows-drain agent), so acceptance #1 (≤ ~32 MiB total soak delta) is not
reachable from `heap.rs` alone. The fix is characterized below; the owner
decides where it lands.

## What changed in heap.rs

Two mechanisms, both cfg-gated to the non-wasm32 section (the wasm32 router
from tickets 19/24 is untouched):

1. **Class blocks are raw anonymous page mappings on Linux** (`mmap`
   PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS via a small `extern "C"`
   block — no new dependency). Freed blocks past the retention cap `munmap`
   straight back to the kernel instead of returning to glibc's ptmalloc.
   Two reasons this matters even though it turned out not to be the main
   soak term: (a) freeing a large glibc mmap'd chunk *raises* glibc's
   dynamic mmap threshold, after which same-sized allocations come from the
   brk arena and are effectively never returned — the classic silent
   arena-growth path; (b) fresh mappings arrive zeroed from the kernel, so
   the `alloc_zeroed` refill path pays page faults (what glibc's own mmap
   path always paid) rather than a memset, and the ticket-12 win — reusing
   *warm* cached blocks — is preserved by the cache, not by glibc.
   Non-Linux non-wasm targets (Windows/macOS compile checks) keep the
   system allocator with identical semantics behind the same `sys` module
   interface.
2. **Per-class retention cap**: `slots_for(c) = min(SLOTS, CLASS_RETAIN >>
   class_log2)` with `CLASS_RETAIN = 8 MiB` gives the slot table
   `4,4,4,4,4,4,2,1,0` for the 64 KiB..16 MiB classes — worst-case cache
   4×(64+128+256+512+1024+2048) KiB + 2×4 MiB + 1×8 MiB = **31.75 MiB**
   (was: up to 128 MiB if every class filled its 4 slots). Classes above
   the cap (16 MiB) are never cached; their blocks munmap on free. The
   inference latency path is unaffected by the cap: `Matrix::apply` and the
   GeGLU/GDN/conv fan-out sites allocate whole buffers *before* fanning
   out and free them after the join, so in-flight demand per class is 1–2
   blocks — well under 4 slots for the ≤2 MiB classes that dominate
   per-apply outputs (100 KB–4 MB).

The hot paths are unchanged in shape: cache hit = pop + optional re-zero
(identical to before); cache miss = one `mmap` (same syscall glibc makes
for large sizes); free-under-cap = push (identical); free-over-cap =
`munmap` (was `System.dealloc`).

## The real soak retention, measured

Fresh soak runs (`engine/ci/soak.py`, laya f16, 20,271 requests each; counts
identical to the frozen goish baseline in every run: 10191×200, 3360×404,
3280×400, 3280×422, 155×429, 160×413; server exit 0):

| Run | heap | initial→final RSS | delta | round-end steps (MiB) |
|---|---|---|---|---|
| ticket 13 | old | 1635.4 → 1952.9 | **+317.5** | +105, +5, +97, +4, +98 |
| fresh baseline (this session) | old | 1674.7 → 2003.6 | **+328.9** | +105, +7, +100, +4, +99 |
| after heap change | new | 1673.5 → 1987.7 | **+314.2** | +108, +2, +98, +2, +100 |
| after, final binary (latest run) | new | 1634.4 → 1941.3 | **+306.9** | +105, +2, +96, +2, +96 |

Idle checkpoints flat in all runs; anon tracks RSS (file ~810 MiB constant).

The staircase is *identical in shape before and after the allocator change*
— proof the steps are not allocator retention. Decomposition:

- **Rounds 1/3/5 (+~100 MiB each)**: the mixed phase uses **non-persistent
  connections** on odd rounds (4000 fresh connections per round). A
  one-round mini-repro (manual server; warmup → 4000 fresh small requests
  → 32 oversized; `/proc/PID/smaps` snapshots each phase) grew +104 MiB,
  and the idle snapshot showed **4028 new anonymous mappings of exactly
  2048 KiB** — Rust's default thread stacks — totaling ~106 MiB RSS
  (~26 KiB touched per stack), all still mapped *after* all requests
  finished and the threads exited. 3 rounds × 4000 connections ≈
  12,000 × 26 KiB ≈ +300 MiB. Threads exit (Threads count stays 5–7);
  their stacks are retained because their JoinHandles are never joined or
  dropped (`http.rs` accept loop).
- **Rounds 2/4 (+~2 MiB each)**: persistent connections — no new threads,
  no stacks. This is the allocator's actual steady-state retention with
  the new cap (body-buffer doubling classes filling their slots).
- **Residual**: total minus stacks ≈ +7..14 MiB (heap cache fill + glibc
  small-alloc churn), inside the 31.75 MiB bound.

The frozen goish baseline is flat (2.39→2.40 GiB) because Go's net/http
multiplexes connections onto a few runtime threads — no thread-per-conn,
no stacks to retain.

### The http.rs fix (handed forward — owned by the windows-drain agent)

In `serve()`'s accept loop, sweep finished handles instead of only counting
them — e.g. replace the `live_conns` pass with a retain-join sweep:

```rust
// join finished connections so their 2 MiB stacks are released
let mut live = 0;
kids.retain(|k| {
    if k.is_finished() { let _ = k.join(); false } else { live += 1; true }
});
```

then gate on `live` (same MAX_CONNS=32 semantics). Expected effect on this
soak: the three ~+100 MiB odd-round steps collapse to the ~+2 MiB
allocator floor — comfortably inside the ≤ ~32 MiB acceptance target. (Not
applied here: http.rs is under active edit by the concurrent agent, and
this ticket's mandate confines edits to heap.rs + research.)

## Latency parity (ticket-12 protocol: CLI decide, interleaved A/B vs
`./laya`, `usage.latency_ms`, 1 warmup each)

| Config | goish | engine (new heap) | delta |
|---|---:|---:|---:|
| Laya F16 12T (refund), n=20 | 151.0 ms | 151.9 ms | **+0.60%** (medians 148.5/149.5) |
| OpenThai F16 12T (thai), n=16 | 720.3 ms | 722.7 ms | **+0.33%** |

A first Laya round at n=16 read +1.97% on means with *identical medians*
(152.5/152.5) — two ~180 ms outliers in the engine leg, the WSL2
per-process outliers ticket 12 already documented; the n=20 rerun on a
quiet machine is +0.60%. Both models sit inside ticket 12's ≤ +1% gate
(their session band was −3.6..+0.6%). No allocation-cost regression is
visible: the warm-block reuse the ticket-12 win depends on is intact, and
the fresh-path `mmap` is the same syscall class glibc used.

## Gates

| Gate | Result |
|---|---|
| `engine/ci/run.sh quick` | **PASS** (build + smoke + reduced drift bit-identical) |
| `engine/ci/run.sh soak` | **PASS** — counts identical, server exit 0 (RSS above) |
| `engine/ci/run.sh full` | **PASS** — drift 42/42 gnu, quant exact, http 14 checks, musl static-pie + 42/42, wasm, cross ×4, lint (fmt + clippy `-D warnings`); qemu SKIP (binary not installed, pre-existing) |
| Latency A/B 12T | Laya +0.60% (n=20), OpenThai +0.33% (n=16) — in band |

## Incident: two unreproduced SIGSEGVs during bring-up

The first two processes run after the initial rebuild (both Laya 12T
decides) died with SIGSEGV, dmesg-confirmed `segfault at 0 ip 0` (call
through NULL — consistent with a stack smashed by pages the kernel reused
after an early-return bug). They never reproduced: 109 subsequent decide
runs clean across models/threads, including a deliberately hostile config
(cache fully disabled — every class free munmaps immediately, maximal
use-after-free sensitivity) with a temporary in-process SIGSEGV+
backtrace hook installed (removed before ship), plus the full gate matrix
and two complete soaks. The two crashes occurred in the minute this
session's builds were contending on cargo's build-directory lock with the
packaging agent's concurrent builds in the same `target/`; a definitive
attribution wasn't possible. If a `laya-engine` SIGSEGV recurs, the triage
recipe is: rebuild with `CARGO_PROFILE_RELEASE_DEBUG=2
CARGO_PROFILE_RELEASE_STRIP=none`, temporarily re-add the SIGSEGV handler
(the diff is in this ticket's history), and symbolicate with addr2line.

## Trade-off notes

- `munmap` on eviction costs a TLB shootdown where glibc's arena-stash
  cost nothing but memory. Steady-state decide traffic never evicts
  (in-flight ≤ 2 blocks/class < slots), so the cost lands only on
  genuinely cold classes — the right trade.
- The 16 MiB class is uncached by design (8 MiB cap → 0 slots): nothing in
  either model's forward path allocates 8–16 MiB per apply (per-apply
  outputs are 100 KB–4 MB), so a fresh `mmap` there is rare and no more
  expensive than glibc's first allocation of that size. If a future
  workload churns 8–16 MiB buffers per request, raise `CLASS_RETAIN` (16
  MiB → slots 4,4,4,4,4,4,4,2,1, worst-case ~63.7 MiB) and re-run this
  ticket's gates.
- Non-Linux targets keep the old System-delegating behavior behind
  `sys`; they are compile-check targets today, and porting the page-return
  there means VirtualAlloc/VirtualFree (Windows) or mmap with BSD flag
  values (macOS) when one of them becomes a shipped target.
