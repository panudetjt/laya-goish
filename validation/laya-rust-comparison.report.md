# Cross-engine comparison: laya-rust (candle) — measured report

Date: 2026-09-24 · Machine: WSL2 on the i5-14400 host (BENCHMARK_REPORT.md §1),
RTX 5060 Ti 16 GB (driver 13030), 16 logical CPUs — same host and day as
`gpu-sidecar.report.md`. Context: ticket 04 runtime choice; this report closes
the "use candle via laya-rust instead of the raw CUDA driver path" alternative
raised against `gpu-sidecar.report.md` §5.

## What was evaluated

[`laya-rust`](https://github.com/aovestdipaperino/laya-rust) @ `5104bf2`
(2026-09-20, "Vendor the ModernBERT encoder so an f16 backbone actually runs")
— pure-Rust inference for the original HuggingFace Laya checkpoints on
candle-core/candle-nn 0.9 + `tokenizers` 0.21, Apache-2.0, 8 stars, single
maintainer.

- Checkpoint: `convaiinnovations/laya` (base, English) — `model.safetensors`
  804 MB (f16 on disk), downloaded from the Hub.
- CPU build: `cargo build --release` — 46.2 s, 140 crates, clean.
- CUDA build: **passes, and runs** — but not out of the box. The stock
  invocation fails in `cudarc v0.19.9` (build-script exit 101: `nvcc
  --version` fails — no toolkit on this host). It can be made to work entirely
  from pip wheels + a symlink farm; see "CUDA build recipe" below.
- candle CPU is f32-only (no f16 CPU kernels — their README), so weights
  double in memory at compute time.

## Why the comparison is apples-to-apples on weights

`models/laya_english_f16.gguf` (807 MB) and the HF checkpoint are the same
weights in different containers, confirmed by output agreement on the same
45-token refund prompt: laya-rust `noul = 0.8911` vs our engine
`0.8911915421` (= committed upstream `examples/refund.output.json`
`0.8911915421485901`). Token counts match (45) when the state string is
byte-identical; a trailing newline in `--state-file` yields 46 tokens and
`noul = 0.8902` — their CLI counts it, our request JSON does not carry it.

## Warm latency — measured

Warm = in-process loop after load and 3 warm-up iterations (laya-rust:
`examples/bench.rs` added to the throwaway clone, `Agent::system_one` timed,
includes tokenization; ours: `usage.latency_ms` from `./laya decide --raw`
@16T, includes tokenization; GPU: sidecar socket round trip, token IDs in —
tokenization is ~ms and excluded, per `gpu-sidecar.report.md` protocol).

### refund (`examples/refund.json`, 1 noul question, 45 tokens)

| Engine | mean | p50 | min | max | n |
|---|---:|---:|---:|---:|---:|
| laya-rust CPU (candle f32, 16 cores) | 467.5 ms | 461.4 | 435.5 | 543.8 | 30 |
| laya-rust CUDA (candle f32, cuBLAS) | **18.7 ms** | 17.9 | 16.7 | 27.1 | 30 |
| **our CPU engine** (F16→F32, AVX2, 16T) | **158.3 ms** | 157 | 152 | 166 | 3 |
| **our GPU sidecar** (CUDA, socket RT) | **33.4 ms** | 33.3 | 32.8 | 34.4 | 8 |

Speed-up vs laya-rust CPU: 2.95× (our CPU), 14.0× (GPU sidecar). laya-rust
CUDA is **1.8× faster than our sidecar** on this case — cuBLAS GEMM vs our
scalar-FMA 16×16-tile matmul.

### urgent (`examples/urgent-score.json`, score + choice, 2 questions)

Prompt strategies differ by design: laya-rust renders **all questions into one
batched 124-token forward pass**; our engine asks **per-question prompts**
(68 + 81 tokens = two round trips). Both rows answer the full question set.

| Engine | mean | p50 | n | notes |
|---|---:|---:|---:|---|
| laya-rust CPU | 810.8 ms | 788.6 | 30 | 1 batched pass, 124 tok |
| laya-rust CUDA | **27.6 ms** | 27.0 | 30 | 1 batched pass, 124 tok |
| **our CPU engine** | **548.3 ms** | 544 | 3 | 2 passes, 68 + 81 tok |
| **our GPU sidecar** | **110.3 ms** | 111.7 | 10 | urgency 50.2 + route 60.1 |

Speed-up vs laya-rust CPU: 1.48× (our CPU), 7.3× (GPU sidecar). laya-rust CUDA
answers both questions in 27.6 ms — **4× faster than our sidecar** here, since
one batched 124-token GEMM beats two sequential 68/81-token passes. The CPU
gap narrows for the same reason. A batched multi-question mode is therefore
real headroom for ticket 04 (our engine currently asks per-question).

CUDA run verified on-device: `nvidia-smi` shows 4,473 MiB in use during the
bench (one f32 model ≈ 1.7 GB VRAM); n=200 re-run stable at 18.8/27.8 ms.
laya-rust CUDA answers equal its own CPU answers exactly (0.8911 refund).

### Load and memory

| | laya-rust CPU | laya-rust CUDA | our CPU engine | our GPU sidecar |
|---|---|---|---|---|
| Model load (warm cache) | 2,911 ms | 814 ms | ~1.5 s (f16 GGUF) | 20–35 s (both models, one shot) |
| Resident | 2,499,256 KB RSS (f32) | 4,473 MiB VRAM observed (one f32 model ≈ 1.7 GB + workspace) | 807 MB (F16 weights) | 2,240 MiB VRAM (**both** models, F16) |
| Cold CLI (load + 1 request) | 14.68 s | n/a (warm bench) | ~2 s | n/a (daemon) |

## Behavioral finding: same weights, different decisions

On the urgent case laya-rust answered `route = billing` (billing 0.6837,
technical 0.2488, security 0.0675); our engine answers `route = technical`
(billing 0.2653, technical 0.7047, security 0.0300) — a **choice flip from the
same weights**. Cause: prompt composition, not arithmetic. The batched
renderer packs both questions' instructions and options around all five
[MASK] markers in one sequence, so the encoder sees different context per
marker than our per-question prompts. `urgency` stays close (1.4071 vs
1.4062; confidence 0.2906 vs 0.2907).

This is the drift-contract argument made concrete: an interchangeable
"same model" engine can flip decisions through prompt discipline alone. Our
GPU sidecar reproduces the CPU engine's per-question prompts and its drift
gates bound |Δp| at 5e-6 (`gpu-sidecar.report.md` §2); laya-rust's outputs
are not comparable under those gates and were never intended to be.

## CUDA build recipe (works on this host, no system toolkit)

The chain of blockers, each fixed in order — all components from pip + the
WSL driver shim:

1. **cudarc build.rs runs bare `nvcc --version`** and panics when absent or
   when the version is not in its supported list (13.4 is not; 13.3 is).
   Fix: `CUDARC_CUDA_VERSION=13030` short-circuits detection (env beats
   build-system detection), and nvcc comes from pip.
2. **bindgen_cuda (candle-kernels) finds CUDA via `CUDA_PATH`/`CUDA_ROOT`
   only** — `CUDA_HOME` is not in its list (unlike cudarc). Export all three.
3. **pip-wheel nvcc emits PTX ISA `.version 9.4`** (toolchain-driven, not
   arch-driven — `CUDA_COMPUTE_CAP` does not change it), which the 13.3-era
   WSL driver (13030) rejects at load: `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`,
   and mixed 13.4 backend pieces also break the in-toolchain ptxas.
   Fix: pin **every** CUDA wheel to 13.3.x so nvcc/nvvm/crt/cccl agree and
   PTX comes out `.version 9.3 / .target sm_120`.
4. **Link-time needs dev-named libs** candle links: cuda, cublas, cublasLt,
   curand, nvrtc, cudart. Assemble a `CUDA_HOME` symlink farm: `bin/`,
   `include/` → pip `nvidia/cu13/`; `lib64/` → unversioned `.so` symlinks to
   the wheel libs plus `libcuda.so → /usr/lib/wsl/lib/libcuda.so.1`.
5. **Runtime**: `LD_LIBRARY_PATH=$CUDA_HOME/lib64` (resolves libcublas.so.13
   etc.; libcuda.so.1 resolves via the system).

```bash
python3 -m venv /tmp/cuda-venv
/tmp/cuda-venv/bin/pip install \
  nvidia-cuda-nvcc==13.3.73 nvidia-cuda-nvrtc==13.3.33 \
  nvidia-cuda-runtime==13.3.29 nvidia-cuda-cccl==13.3.3.4.1 \
  nvidia-cublas==13.3.0.5 nvidia-curand==10.4.4.72 \
  nvidia-nvvm==13.3.73 nvidia-cuda-crt==13.3.73
V=/tmp/cuda-venv/lib/python3*/site-packages/nvidia/cu13
mkdir -p $HOME/cuda-home/lib64
ln -sfn $V/bin $HOME/cuda-home/bin; ln -sfn $V/include $HOME/cuda-home/include
cd $HOME/cuda-home/lib64
for l in libcublas libcublasLt libcurand libnvrtc libcudart libnvvm; do
  ln -sfn $V/lib/$l.so* .; done
for l in libcublas libcublasLt libcurand libnvrtc libcudart libnvvm; do
  ln -sfn $l.so.* $l.so; done   # unversioned dev names
ln -sfn /usr/lib/wsl/lib/libcuda.so.1 libcuda.so
# build (env vars are load-bearing; cargo re-runs cudarc's build script)
export PATH=$HOME/cuda-home/bin:$PATH CUDA_HOME=$HOME/cuda-home \
       CUDA_PATH=$HOME/cuda-home CUDARC_CUDA_VERSION=13030
cargo build --release --features cuda
# run
export LD_LIBRARY_PATH=$HOME/cuda-home/lib64
./target/release/laya --model ... --device cuda --dtype f32 ...
```

Caveats: `--dtype f16` on CUDA hits a NaN and panics in laya-rust's own
decision code (`partial_cmp` on NaN logits, src/model.rs:232) — f16 is
unusable there, so CUDA means f32 (≈1.7 GB VRAM per Laya model). Two f32
models resident ≈ 3.4 GB, right at the measured WSL CUDA allocation budget —
our sidecar's F16 residency (844 MB/model, 2,240 MiB both) still fits where
candle's would not.

## Decision

**Not adopted as the engine for ticket 04; adopt its matmul path.** The
earlier belief that candle cannot build here is corrected by the recipe above.
What the measurement now says:

1. **Speed evidence for ticket 04:** laya-rust CUDA (cuBLAS GEMM, f32) beats
   our sidecar's hand-rolled scalar-FMA matmul **18.7 vs 33.4 ms** (refund)
   and answers a 2-question state in one batched pass at **27.6 vs 110.3 ms**
   (urgent). The sidecar should link the pip-wheel `libcublas.so.13` (proven
   viable on this host) or otherwise move matmul onto tensor cores — and
   ticket 04 should consider batched multi-question prompts, which is also
   how laya-rust gets its urgent-case win.
2. Still not a drop-in engine: no GGUF `graph_spec` walk (HF checkpoints
   only), no OpenThai-SystemOne, no per-question prompt discipline, and no
   drift contract against the shipping engine — its batched prompts flip
   `route` on the urgent case (below), and f16-on-CUDA is broken in its own
   code. F32 residency (1.7 GB/model) does not fit the two-model sidecar
   budget that our F16 path fits.
3. Where laya-rust is the right tool: a standalone Rust app targeting the
   HF checkpoints on macOS (Metal) or a CUDA box, or as an independent
   cross-check implementation — the refund agreement above did exactly that.

## Reproduce

```bash
git clone --depth 1 https://github.com/aovestdipaperino/laya-rust.git /tmp/laya-rust
mkdir -p /tmp/laya-models/laya-base/{encoder,tokenizer}
B=https://huggingface.co/convaiinnovations/laya/resolve/main
for f in model.safetensors encoder/config.json tokenizer/tokenizer.json \
         tokenizer/tokenizer_config.json rl_agent_config.json; do
  curl -sL -o /tmp/laya-models/laya-base/$f "$B/$f"
done
cd /tmp/laya-rust && cargo build --release            # CPU: OK out of the box
# CUDA: follow the "CUDA build recipe" section above (pip wheels 13.3.x +
# CUDA_HOME symlink farm + CUDARC_CUDA_VERSION=13030), then:
#   warm bench (examples/bench.rs source preserved in the bench JSON appendix;
#   Options::default() auto-picks CUDA in a --features cuda build):
LD_LIBRARY_PATH=$HOME/cuda-home/lib64 \
  cargo run --release --features cuda --example bench -- /tmp/laya-models/laya-base 30
# ours, same inputs:
./laya decide models/laya_english_f16.gguf --request examples/refund.json --raw --threads 16
./laya decide models/laya_english_f16.gguf --request examples/urgent-score.json --raw --threads 16
# GPU rows: see gpu-sidecar.report.md "Reproduce" (sidecar + gpu-bench.py;
# urgent per-question timing script preserved in bench JSON appendix)
```

Raw numbers: `validation/laya-rust-comparison.bench.json`.

## Caveats

- Single host, single day; n = 30 (laya-rust CPU/CUDA), 3 (our CPU), 8/10
  (GPU sidecar); CUDA re-checked at n=200 (18.8/27.8 ms, stable). Our CPU and
  GPU rows use the engine's in-band latency, not wall clock.
- laya-rust numbers include their tokenizer; GPU rows receive token IDs
  (tokenization ≈ 1 ms at this scale, inside measurement noise).
- candle CPU f32 vs our F16-weights/F32-accumulate is a structural choice of
  candle (no f16 CPU kernels), not a tuning difference — it is the main
  reason the CPU gap exists, alongside framework overhead. On CUDA,
  laya-rust is also f32 (f16 NaNs in its own decision code); our sidecar
  keeps F16 weights with F32 accumulate, trading some GEMM throughput for
  half the VRAM.
- The throwaway clone (`/tmp/opencode/laya-rust`) may be gone; the report
  and bench JSON carry everything needed to rebuild it.
