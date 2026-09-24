# GPU sidecar proof of concept — measured report (ticket 03)

Date: 2026-09-24 · Machine: WSL2 on the i5-14400 host (BENCHMARK_REPORT.md §1),
RTX 5060 Ti 16 GB (compute cap 12.0, driver 610.57 / CUDA 13.3 UMD, WSL
passthrough), Intel UHD Graphics 730 on the Windows side.
Ticket: `.scratch/acceleration/issues/03-gpu-sidecar-poc.md`.

## What was built

`sidecar/` — a separate standard-OS Rust process (the Goish CPU binary is
untouched; no file under `src/` changed):

- `src/cuda.rs` — minimal CUDA **driver API** FFI linked against the WSL
  `libcuda.so` shim (no CUDA toolkit on the Rust side). WSL findings baked
  in: use `cuCtxCreate` (primary-context APIs half-fail on the shim),
  avoid `cuMemGetInfo` (garbage after allocations — we account bytes
  ourselves), observed allocation budget ≈ 3.4 GiB per process despite the
  16 GiB card (WDDM commit view — 2.3 GiB of models fits comfortably).
- `src/gguf.rs` — std port of the CPU engine's GGUF v3 reader; F16→F32 is
  the bit-exact `gguf::half` port.
- `src/laya.rs` — walks `ggmlc.graph_spec` **exactly as the CPU engine**
  (`src/models/laya/model.rs`): norms by opcode-201 node name, linears by
  opcode-29 first input, folded biases from third inputs plus opcode-200
  fused consumers (summed in the engine's order), full-vs-sliding
  attention from mask identity (full = every 3rd layer starting at 0).
  The whole GGUF tensor data section uploads verbatim into one VRAM
  region; kernels address weights at file offsets.
- `kernels/laya.cu` — CUDA C kernels (built by `kernels/build.sh` with the
  pip-wheel nvcc `nvidia-cuda-nvcc` 13.4 to `sm_120`; no system toolkit):
  F16→F32-exact `matmul` (16×16 tiles, F32 accumulate, `scorer_fc2` handled
  as the one F32-weighted linear), sequential-order `layernorm`,
  warp-per-(head,query) `attention` with RoPE applied on q/k loads only and
  CPU-order serial softmax, `geglu`/`relu`/`gelu`/`add`/`embed_gather`/
  `gather_rows`/`add_typeemb`.
- `src/forward.rs` — the CPU `Model::forward` op sequence over
  pre-allocated max-len activation buffers; the k≤16 softmax/top-stats tail
  runs on the host in the engine's exact op order to build the pooled
  act-head input.
- `src/main.rs` — `serve` (Unix-domain socket, JSON lines, one round trip
  per request: `{ids, markers, qtype}` → `{logits, act, gpu_ms, h2d/d2h
  bytes}`), `bench`, `info`, `stage-check` (debug oracle).
- `src/oracle.rs` — scalar transcription of the CPU forward used by
  `stage-check`; validated by reproducing the committed
  `validation/refund.reference.json` logits ([-2.8763168, 1.2947116] vs
  committed [-2.87634, 1.29493]).
- `src/igpu_probe.rs` — Vulkan device enumeration via `ash` (runtime-loaded
  loader; no SDK needed to build).

Toolchain note: the root `.cargo/config.toml` carries the no_std link flags
for the Goish CPU binary; cargo **appends** rustflags down the directory
tree, so `sidecar/build.sh` builds with `RUSTFLAGS` set (the documented
override), which is required for a std binary to link a normal crt0.

## Acceptance criteria — measured

### 1. Round trip ≤ 60 ms (target band 30–60 ms) — PASS

`examples/refund.json` token IDs (45 tokens, 1 noul question), protocol 1
warm-up + 8 measured (same as BENCHMARK_REPORT.md §3), full client round
trip over the sidecar socket:

| Metric | Value |
|---|---|
| GPU round trip incl IPC | **33.1 ms mean** (min 32.9, p50 33.2, max 33.4) |
| Sidecar-internal forward | 33.0 ms mean |
| IPC + JSON share | ~0.2 ms |
| CPU engine today (16T `usage.latency_ms`) | 169.2 ms mean (min 156, p50 167) |
| Reference CPU means | §7: 205.0 ms, §8: 172.3 ms @16T |
| Speedup vs CPU (round trip vs engine latency) | **5.11×** |

Evidence: `validation/gpu-sidecar.bench.json` (regenerate:
`.venv/bin/python3 validation/gpu-bench.py` with the sidecar serving).
Model load (~2.4 s warm) is excluded, per the ticket.

### 2. Drift gates on the 6-case reference set — PASS (gates declared before measuring)

Reference set (all on `models/laya_english_f16.gguf`, declared in
`validation/gpu-drift.py` before any measurement): `examples/refund.json`
(noul), `examples/urgent-score.json` (score+choice),
`examples/openthai-thai.json` (choice+score+noul, Thai),
`examples/openthai-mixed.json`, `validation/types.request.json` (typed
state), `validation/unicode.request.json` (unicode edges) — 13 questions,
all three kinds.

CPU reference: `./laya decide <f16> --request X --threads 16 --raw` (inputs
replayed token-for-token through the sidecar; decision layer recomputed
from GPU logits with the engine's exact temperature selection).

| Gate | Declared | Measured |
|---|---|---|
| Discrete choice flips | 0 | **0** |
| max \|Δprobability\| | ≤ 0.01 | **< 5e-06** |
| max \|Δscore\| | ≤ 0.05 | **< 5e-06** |
| max \|Δlogit\| (info) | — | 2.7e-05 |
| max \|Δact\| (info) | — | 6.4e-03 (on ±4000-magnitude values) |

Harness: `validation/gpu-drift.py` (exit non-zero on breach).
Stage-by-stage agreement vs the scalar oracle: 270/270 stages within
tolerance (`stage-check`).

### 3. VRAM residency, no per-request weight traffic — PASS

- Sidecar holds **both F16 models resident: 804 MiB (Laya) + 1435 MiB
  (OpenThai) = 2240 MiB**, plus 32 MiB activation buffers, uploaded once at
  load. OpenThai is resident-only in this PoC (forward not ported — ticket
  04 scope).
- Per-request PCIe: **≤ 4300 B host→device, ≤ 4112 B device→host** (token
  ids, markers, pooled vector, logits, act) — no weight bytes cross PCIe
  per request, by construction (kernels address the load-time region).
- `nvidia-smi memory.used`: 6605 MiB with the sidecar loaded vs 1740 MiB
  after stop (WDDM commit pads the 2272 MiB of `cuMemAlloc`s).

### 4. iGPU (UHD 730) Vulkan probe — NO-GO, measured

- Vulkan enumeration from this OS (`sidecar/src/igpu_probe.rs`, ash): the
  only visible device is **llvmpipe** (CPU rasterizer, api 1.4.318). No
  Intel UHD 730, no discrete-GPU Vulkan.
- The UHD 730 exists and is healthy on the Windows host (PowerShell
  `Win32_VideoController`: "Intel(R) UHD Graphics 730", 2 GiB, OK) — it is
  **not paravirtualized into WSL2**: `/dev/dxg` exposes the NVIDIA adapter
  for CUDA only; Mesa's Vulkan ICD set here (`anv/hasvk/radv/nouveau/
  lvp/...`) targets bare-metal DRM devices WSL2 lacks, and Ubuntu noble
  ships no `dzn`/dozen (Vulkan-on-D3D12) driver that could reach it.
- Even granting a zero-overhead port, the numbers don't support it: the
  forward is ~33 GFLOP; 24 EU × 8 ALU × 2 × ~1.9 GHz ≈ **0.7 TFLOPS F32**
  → ~45 ms compute floor, sharing the CPU's DDR5 ceiling (76.8 GB/s,
  unified memory, contended) vs the measured 33.1 ms CUDA round trip that
  already exists. The ticket's honest prior ("beating 16 AVX2 threads is
  unlikely") is confirmed in the stronger form: there is nothing to win
  even if the device were reachable.

**Verdict: no-go for the iGPU path.** Revisit only if a bare-metal Linux
deployment (real `/dev/dri` + ANV, or a distro shipping dzn) makes the
UHD 730 reachable AND the product needs iGPU-class hardware; then re-run
`igpu-probe` first.

### 5. Decision

**Proceed to ticket 04 (standalone GPU decision engine), CUDA runtime.**

- The ≤60 ms gate is met with 45 % margin including IPC; drift is ~3 orders
  of magnitude inside the declared gates; residency and PCIe behavior are
  proven. CUDA is reachable, fast, and now has a working, gate-tested
  forward implementation to reuse.
- Vulkan-first (per ticket 04's text) is **not** justified by this machine:
  no Vulkan path exists to either GPU from the deployment OS (measured
  above). CUDA-first with the kernel layer kept runtime-agnostic is the
  evidence-backed choice.
- Inputs for 04: kernels + drift harness + loader in `sidecar/` are the
  reuse base. OpenThai's Gated-DeltaNet/GQA forward remains unported (its
  1.4 GB residency is already proven). Known headroom if 04's ≥3× Laya
  target tightens: ~10 µs/launch × ~300 launches ≈ 3 ms of the 33 ms is
  launch overhead (CUDA graphs), and the matmul path is scalar-FMA F32
  (tensor cores untouched); the 3.4 GiB WSL CUDA budget (measured) bounds
  per-process model residency on this host.

## Reproduce

```bash
cd sidecar && ./build.sh                       # kernels (pip nvcc) + binary
./build.sh >/dev/null # optional rebuild of cubin only: sh kernels/build.sh
# terminal 1
sidecar/target/x86_64-unknown-linux-gnu/release/laya-gpu-sidecar serve \
  --laya models/laya_english_f16.gguf \
  --resident models/openthai_systemone_v03_f16.gguf \
  --cubin sidecar/kernels/laya.cubin
# terminal 2 (from repo root)
.venv/bin/python3 validation/gpu-drift.py      # 6-case gates vs CPU F16
.venv/bin/python3 validation/gpu-bench.py      # latency + residency + PCIe
sidecar/target/x86_64-unknown-linux-gnu/release/igpu-probe  # iGPU probe
# correctness oracle (slow, debug)
sidecar/target/x86_64-unknown-linux-gnu/release/laya-gpu-sidecar stage-check \
  --laya models/laya_english_f16.gguf --cubin sidecar/kernels/laya.cubin
```
