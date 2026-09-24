# Reality-check: burn & candle backends across platforms

**Ticket**: `wayfinder/tickets/01-backend-reality.md` · **Research date**: 2026-09-24 · **Status**: findings
**Scope**: burn & candle inference backends — CPU, GPU (wgpu/CUDA/Metal), wasm32, threading, GGUF — as of Sep 2026, against primary sources (crates.io metadata, repo manifests, source files, GitHub issues/releases, official docs). All claims carry a source tag `[S#]` resolved in [Sources](#sources); "accessed" = date I read the page; other dates are the source's own publish date.

---

## 0. Version snapshot (primary source: crates.io API)

| Crate | Stable | Pre-release | Notes |
|---|---|---|---|
| `burn` | **0.21.0** (pub 2026-05-07) | **0.22.0-pre.4** (pub 2026-09-22) | MSRV 1.92, edition 2024 [S1][S6] |
| `candle-core` | **0.11.0** (pub 2026-06-26) | — | workspace candle 0.11.0, edition 2021 [S2][S18] |
| `cubecl` (burn's compute layer) | 0.11.0-pre (pinned rev by burn) | — | **self-declared "alpha"**, API breaking between minors [S35] |

**Churn warning (relevant to go/no-go)**: between 0.21 and 0.22-pre, burn **removed** `burn-candle` (#5343, in 0.22.0-pre.3, 2026-08-25), **deprecated** `burn-ndarray` (#5344) and **deprecated** `burn-tch` (#5525, in 0.22.0-pre.4, 2026-09-22) [S3][S4]. The 0.22 line replaces the `Tensor<B>` backend-generic API with a runtime `Device` + dispatch model [S14]. Backend churn this fast is itself a migration risk.

---

## 1. Backend × platform reality table

Legend: ✅ GO · ⚠️ conditional · ❌ no-go. "For us" = the Laya Goish use case (encoder + GDN/GQA hybrid, 0.8–1.5 GiB GGUF weights, CPU-first, wasm required, RTX 5060 Ti available).

### burn (0.21.0 stable / 0.22.0-pre.4)

| Backend | Linux x86_64 | Linux aarch64 | Windows | macOS (AS) | wasm32-unknown-unknown | Docker | Verdict per cell |
|---|---|---|---|---|---|---|---|
| **flex** (CPU, new default) | ✅ GO | ✅ GO | ✅ GO | ✅ GO | ✅ GO — **only burn CPU path that runs in browser** (SIMD128) | ✅ (DIY image) | Best portable CPU backend; macerator SIMD AVX2/AVX-512/NEON/SIMD128, gemm matmul, rayon; opt-in `x86-v4` AVX-512 [S8][S9] |
| **ndarray** (CPU, old default) | ⚠️ works, deprecated | ⚠️ works, deprecated | ⚠️ works, deprecated | ⚠️ works, deprecated | ⚠️ builds (historically the wasm path) | ✅ | Deprecated in 0.22-pre.3 #5344 — do not adopt new code on it [S4][S7] |
| **cpu** (CubeCL CPU) | ⚠️ early | ⚠️ early | ⚠️ early | ⚠️ early | ❌ no evidence | ✅ | MLIR/CubeCL CPU backend; GEMM-on-CPU just landed (#5538); README says "Example coming soon" — watch, don't bet [S3][S43] |
| **wgpu** (GPU: Vulkan/DX12/Metal/WebGPU) | ✅ GO (Vulkan; DX12 via wgpu) | ✅ GO (Vulkan) | ✅ GO (DX12/Vulkan) | ✅ GO (Metal) | ⚠️ GO only in WebGPU-capable browsers | ✅ | Mature-ish but CubeCL is "alpha"; no special WebGPU features required (adapts to adapter limits); tensor cores not yet on WebGPU path; known shader/adapter issues open [S10][S11][S34][S35][S36] |
| **cuda** (native CUDA via CubeCL+cudarc) | ✅ GO (CUDA 12.x) | ⚠️ | ✅ GO (CUDA 12.x) | ❌ | ❌ | ⚠️ needs CUDA runtime in image | Shipped & actively maintained; requires CUDA 12.x toolkit/driver; 5060 Ti (Blackwell) needs a cu12.8+ build — verify sm_120 in a spike [S12] |
| **rocm** (AMD) | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ⚠️ | Exists, less exercised than cuda/wgpu [S6] |
| **tch** (LibTorch) | ✅ works | ✅ works | ✅ works | ✅ works (MPS too) | ❌ **No** (own platform table) | ⚠️ huge LibTorch binary | **Deprecated as of 0.22.0**, removal planned; LibTorch 2.9.0; don't migrate onto it [S3][S13] |
| **candle** backend | — | — | — | — | — | — | **Removed** from workspace in 0.22-pre.3 (#5343); last published crate 0.22.0-pre.2, description "[Deprecated]" [S4][S39][S40] |

### candle (0.11.0)

| Backend | Linux x86_64 | Linux aarch64 | Windows | macOS (AS) | wasm32-unknown-unknown | Docker | Verdict per cell |
|---|---|---|---|---|---|---|---|
| **CPU** (`gemm`, opt MKL/Accelerate) | ✅ GO | ✅ GO | ✅ GO | ✅ GO | ✅ GO — **proven: quantized GGUF Qwen3 runs in browser** | ✅ (DIY) | Pure-Rust `gemm` core (SIMD128 enabled on wasm); `mkl`/`accelerate` features optional; quant vec_dot dispatches AVX2+FMA / NEON / simd128 at runtime [S17][S18][S20][S22][S25] |
| **CUDA** (cudarc + candle-kernels) | ✅ GO | ⚠️ | ✅ GO | ❌ | ❌ | ⚠️ cudarc dynamic-linking | Battle-tested for LLM/transformer inference; cudarc 0.19.10 (cublas/cublaslt/curand/nvrtc/f16/f8), cudnn/nccl features, cutile; quantized CUDA kernels included [S17][S18][S21] |
| **Metal** (objc2-metal + candle-metal-kernels) | ❌ | ❌ | ❌ | ⚠️ works, thinner than CUDA | ❌ | — | Real kernel set exists (metal.rs quant kernels, candle-metal-kernels crate) but is the least-exercised GPU path [S17][S21] |
| **wgpu/WebGPU** | — | — | — | — | — | — | **Does not exist**; open feature request #344 (updated 2026-04-17) [S30] |
| **wasm threading** | — | — | — | — | ⚠️ single-thread only | — | rayon falls back to 1-thread on wasm; no wasm-bindgen-rayon integration shipped [S28] |

---

## 2. Q1 — CPU kernels / BLAS / SIMD coverage

**burn**
- `burn-flex` (recommended CPU backend since 0.21, added #4761 [S5]): matmul via **`gemm` 0.19** (native f16; optional `x86-v4` = AVX-512 kernels for Sapphire Rapids/Zen 4/5; experimental `apple-amx`); elementwise SIMD via **`macerator`** — aarch64 NEON, x86_64 AVX2/AVX-512/SSE, **wasm32 SIMD128**, scalar fallback; optional rayon; no_std capable [S8][S9][S37][S38].
- `burn-ndarray` (0.21 stable default; deprecated in 0.22-pre): `matrixmultiply` for gemm (with `matrixmultiply/threading`), f32 elementwise SIMD via macerator since PR #2851; optional BLAS features: `blas-accelerate` (macOS), `blas-netlib`, `blas-openblas`, `blas-openblas-system`; `multi-threads` = rayon [S7].
- SIMD coverage gap: AVX-512 requires the opt-in `x86-v4` feature in both gemm-based stacks; default builds target AVX2-class code paths. Neither framework ships Intel AMX by default (burn-flex has it behind an experimental flag) [S8][S9][S37].

**candle**
- CPU matmul always via **`gemm` 0.19.0** with `wasm-simd128-enable` [S18]. Optional features: `mkl` (`intel-mkl-src` 0.8.1, `mkl-static-lp64-iomp`) and `accelerate` (macOS) [S17]. README: "Optimized CPU backend with optional MKL support for x86 and Accelerate for macs" [S22]. Known linker-symbol pitfalls for mkl/accelerate documented in README [S22].
- **Quantized ops have their own SIMD kernels** in `candle-core/src/quantized/`: `avx.rs` (dispatched only when runtime-detected **AVX2+FMA**), `neon.rs` (aarch64, incl. repacked Q4Kx8), `simd128.rs` (wasm32) — dispatch per dtype in `k_quants.rs` [S20][S21].
- Runtime detection example from candle's own logs: `avx: false, neon: true, simd128: false` [Context7 mirror of candle examples].
- Coverage gap: **no AVX-512 quant kernels** (avx.rs is AVX2-based), no IQxx/TQ/MXFP4 (see Q6).

---

## 3. Q2 — burn wgpu backend maturity

- Stack: `burn-wgpu` → `cubecl-wgpu` → wgpu. Features: `metal`, `vulkan`, `webgpu` [S10]. Adapter coverage per the burn-wgpu README platform table: Vulkan (Linux/Win/macOS-portable/Android/iOS), **Dx11/Dx12 (Windows)**, OpenGL (native), Metal (macOS), WebGPU (wasm) [S11]. burn 0.21 pinned **wgpu v28** [S5].
- **Required WebGPU features/limits: essentially none** — cubecl requests `adapter.features()` minus `MAPPABLE_PRIMARY_BUFFERS` and `required_limits: adapter.limits()` (i.e., the adapter's own maxima), then adapts kernel launch config (workgroup storage size, bindings = `max_storage_buffers_per_shader_stage - 1` to co-exist with a wgpu-internal Metal buffer, etc.) [S36]. Compilation targets: SPIR-V (Vulkan, with CMMA tensor cores when available), Metal C++/MSL, WGSL fallback (covers WebGPU and non-Vulkan/non-Metal adapters) [S36]. **Tensor cores are not yet available on the WebGPU/WGSL path** (CubeCL README states this) [S35].
- **Known open issues** (state: open, as of 2026-09-24): #597 *f16 with WGPU* (15 comments); #3583 *WGPU shader parsing error: Incompatible operands: LogicalAnd* (updated 2025-08-22); #3779 *vulkan backend — no possible adapter (Alpine)* (2025-09-25); #4503 *ci: add wasm32 build check to prevent getrandom/no-std regressions in burn-wgpu* (2026-02-13) [S33][S34]. Also relevant: wgpu fused-reduce broadcast bug fixed in 0.22-pre.2 (#5268) [S5-series releases].
- Maturity read: it is burn's default GPU path and powers the public web demo (`burn.dev/demo`) [S16], but the substrate (CubeCL) is **alpha**, and 0.22-pre releases still carry GPU fixes — expect to pin exact versions and re-test per driver/OS.

---

## 4. Q3 — candle-cuda status; burn-tch status

**candle-cuda**: `cuda` feature = `cudarc` 0.19.10 (cublas, cublaslt, curand, driver, nvrtc, f16, f8, `dynamic-linking`) + `candle-kernels` (CUDA kernels compiled at build); extra features `cudnn`, `nccl`, `cutile` (0.3.1) [S17][S18]. Quantized ops have CUDA kernels too (`quantized/cuda.rs`, ~41 KB) [S21]. It is the standard GPU path for candle's transformer zoo. Custom user kernels are an officially advertised extension point (flash-attention v2 embedding example in README) [S22].

**burn-tch**: **deprecated as of 0.22.0** (PR #5525, released 2026-09-22), "will be removed in a future release" [S3][S13]. Binds LibTorch **2.9.0** via tch-rs 0.22; supports CPU (multithreaded), CUDA (TORCH_CUDA_VERSION=cu126/cu128/cu130 dists), MPS; its own platform table marks **WASM: No** for all rows [S13]. Notable: SafeTensors/PyTorch weight reading was decoupled from it into the `pytorch-reader` crate in 0.22-pre.4 (#5656), so dropping tch does not remove .pt import [S3].

**burn-cuda** (the non-tch native path): CubeCL + cudarc, **requires CUDA 12.x on PATH** [S12]. For our RTX 5060 Ti (Blackwell): LibTorch/tch only documents cu126/cu128/cu130 builds [S13]; for burn-cuda verify sm_120 codegen in a spike before committing.

---

## 5. Q4 — wasm32: unknown-unknown vs wasi

**burn** — `wasm32-unknown-unknown` is the documented web target: the book's WebAssembly chapter says burn supports WASM via the **Flex and WebGpu backends** and lists two example apps [S15]. The `mnist-inference-web` example builds with either `flex` or `wgpu` (WebGPU browsers only for the latter), explicitly targets `wasm32-unknown-unknown` **without WASI**, runs inference with `#![no_std]` flex, ships model weights as burnpack `.bpk`, and is live at burn.dev/demo [S16]. `getrandom` needs explicit configuration on wasm with WebGpu [S15]; a CI issue exists specifically to stop wasm32/no_std regressions in burn-wgpu (#4503) [S33]. **wasi: no first-party examples or docs found** (neither repo ships a wasi example) — treat as unproven.

**candle** — `wasm32-unknown-unknown` is first-class for CPU inference: `candle-wasm-tests` run under `wasm-pack test --chrome` [S27]; **11 wasm examples** ship in-tree (bert, whisper, t5, phi, moondream, segment-anything, yolo, llama2-c, blip, chat-template, **quant-qwen3**) [S24]. `quant-qwen3` **parses a GGUF file in the browser** (`candle_core::quantized::gguf_file::Content::read` → `QuantizedQwen3::from_gguf`), Q8_0 and Q4_K(Q4_K_M) models [S25][S26]; its Cargo.toml shows the wasm plumbing (getrandom `wasm_js`, tokenizers `unstable_wasm`, wasm-bindgen/web-sys) [S25]. tokenizers is the only candle-core dep gated off `wasm32` [S17]. No WebGPU/GPU on wasm — CPU only [S30].

---

## 6. Q5 — Threading & wasm memory

**Native**: burn (flex/ndarray) parallelizes with **rayon** (default features) [S7][S9]; candle-core depends on rayon unconditionally for CPU op parallelism [S17]. CubeCL's CPU runtime is documented as sequential cube scheduling (plane = 1) — the CubeCL CPU backend is not (yet) the parallel story; burn-flex is [S35].

**wasm threads**: rayon ≥1.13 documents an explicit **single-threaded fallback** on `wasm32-unknown-unknown`/`wasm32-wasi` ("Rayon configures a fallback mode… as if RAYON_NUM_THREADS=1"; `join` runs sequentially), plus a `web_spin_lock` feature for browser main-thread use; CI builds wasm32-unknown-unknown and tests wasm32-wasip1 under wasmtime [S28][S29]. Consequence: **both frameworks run single-threaded in the browser out of the box**; real wasm threads would require integrating `wasm-bindgen-rayon` + SharedArrayBuffer + COOP/COEP headers yourself — neither framework ships that integration (no trace of wasm-bindgen-rayon in either dependency tree; burn-flex/candle simply forward to rayon) [S9][S17][S28].

**wasm memory (4 GiB / wasm64)**: neither framework handles memory64. Candle has open issue #1034 *"Memory64 Support"* (filed 2023-10-09, still open 2026-09-24) [S31]; burn has no memory64 issue or code found (search 2026-09-24). Practical ceiling is the 32-bit 4-GiB linear memory; our 0.8–1.5 GiB weights fit, but headroom for activations/runtime in-browser needs a spike. No `--memory64`/wasm64 flags anywhere in either build system.

---

## 7. Q6 — GGUF

**candle — native GGUF support (read + write)**
- Reader: `candle_core::quantized::gguf_file` (GGUF v2; "gguf v2 support" and "Api to write GGUF files" in CHANGELOG) [S23].
- Supported dtypes (`GgmlDType` enum, source as of 2026-09-24): **F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K** [S19]. Files quantized as `Q4_K_M`/`Q4_K_S` etc. read as `Q4K` (the file format dtype is Q4_K; the `_M/_S` suffix is a quantization-time flavor) — the in-browser quant-qwen3 demo explicitly lists Q8_0 and Q4_K_M as usable [S26]. **Not supported: llama.cpp IQxx (i-quants), TQ1_0/TQ2_0, MXFP4** — absent from the enum [S19].
- Tooling: `tensor-tools` quantizes/converts tensors into any of those GgmlDTypes (incl. F16), i.e., candle can both consume GGUF and produce GGUF-quantized tensors [S41].

**burn — no GGUF**
- No GGUF reader anywhere; open issue **#1187 "Support importing GGUF files"** (24 comments; last updated 2025-08-08, still open 2026-09-24) [S32].
- Weight formats burn does support: native **burnpack `.bpk`** (zero-copy, mmap, F16 adapter), **SafeTensors**, **PyTorch `.pt`/`.pth` (read-only, via burn-store / pytorch-reader)**, plus ONNX import via the separate `burn-onnx` repo [S3][S5][S14].
- **Conversion path for us**: GGUF → (de)quantize/export once at build time, e.g. via candle `tensor-tools` or llama.cpp → `.safetensors` → `burn-store` `SafetensorsStore` (+ `PyTorchToBurnAdapter`/key remapping) [S14][S41]. Note burn's own quantization (int8/f16, per-tensor & per-block symmetric, two-level scales in 0.22-pre) is **not** GGUF-k-quants compatible — you either run fp16/fp32 or requantize into burn's scheme [S8][S3].

---

## 8. Docker

Neither repo contains a Dockerfile or publishes an official image (repo-root listings checked 2026-09-24) [S42]. Both are libraries compiled into your own image. Consequences: burn-cuda/candle-cuda images must carry a CUDA 12.x runtime (candle defaults to cudarc `dynamic-linking`) [S12][S18]; CPU/wasm builds are trivial static Rust binaries.

---

## 9. Implications for the Laya Goish migration (opinion, grounded in the above)

Baseline to beat: 163.6 ms (Laya, 12 CPU threads), 788.9 ms (OpenThai, 12 CPU threads) — hand-written AVX2 kernels.

1. **CPU (primary target today)**: candle keeps our GGUF assets untouched (Q4_K/Q8_0 read natively, AVX2+FMA quant kernels, MKL/Accelerate optional) — lowest-risk port for parity, but its per-op CPU perf is only "tied at memory bandwidth" vs burn-flex's self-reported wins (batched matmul 8–11×, fused layernorm/softmax 1.4–3.4× — vendor benchmarks on M3 Max, verify on AVX2 x86) [S8]. burn requires a **one-time GGUF→safetensors conversion** and re-quantization decisions [S14][S32][S41].
2. **wasm (hard requirement)**: both are GO on `wasm32-unknown-unknown` with SIMD128 [S15][S16][S24][S25]. candle is the only one that can load our **existing GGUF** in-browser [S25]. Both are **single-threaded on wasm** [S28]; 4-GiB ceiling fits 0.8–1.5 GiB weights but budget activations carefully; memory64 unsupported in both [S31]. WASI: unproven in both.
3. **GPU (RTX 5060 Ti)**: burn gives one API across CUDA/Vulkan/DX12/Metal/WebGPU with autotune+fusion — but on an alpha substrate with live shader/adapter issues [S10][S34][S35]. candle-cuda is the boring, proven option for NVIDIA-only; candle has no wgpu path at all (#344 open) [S30].
4. **Model-arch fit (Gated-DeltaNet + GQA hybrid)**: neither ships GDN out of the box; budget a custom op: candle = plain Rust CPU op + optional CUDA kernel (extension point advertised) [S22]; burn = CubeCL kernel or flex Rust path (examples `custom-cubecl-kernel`, `custom-wgpu-kernel` exist) [S43][S14].
5. **Team-level go/no-go**: candle = GO for CPU+wasm now (GGUF-native, stable-ish 0.11, but no GPU portability beyond CUDA/Metal). burn = conditional GO for GPU-first deployments and if we accept the 0.21→0.22 API churn, weight-conversion pipeline, and alpha CubeCL; not GO for "keep GGUF" workflows.

---

## Sources

All accessed 2026-09-24 unless a publication date is given.

- [S1] crates.io API — burn versions: https://crates.io/api/v1/crates/burn (0.21.0 pub 2026-05-07; 0.22.0-pre.4 pub 2026-09-22)
- [S2] crates.io API — candle-core: https://crates.io/api/v1/crates/candle-core (0.11.0 pub 2026-06-26)
- [S3] burn v0.22.0-pre.4 release notes (2026-09-22; deprecate burn-tch #5525; CubeCL GEMM on CPU #5538; pytorch-reader split #5656): https://github.com/tracel-ai/burn/releases/tag/v0.22.0-pre.4
- [S4] burn v0.22.0-pre.3 release notes (2026-08-25; remove burn-candle #5343; deprecate burn-ndarray #5344): https://github.com/tracel-ai/burn/releases/tag/v0.22.0-pre.3
- [S5] burn v0.21.0 release notes (2026-05-07; add burn-flex #4761; deprecate burn-candle #4416; global Dispatch #4508; wgpu v28 #4244; ONNX → burn-onnx #4393): https://github.com/tracel-ai/burn/releases/tag/v0.21.0 (post: https://burn.dev/blog/release-0.21.0)
- [S6] burn workspace Cargo.toml @ main (no burn-candle member; gemm 0.19, macerator 0.3.4, matrixmultiply 0.3.10, ndarray 0.17.2, rayon 1.10.0, tch 0.22.0): https://raw.githubusercontent.com/tracel-ai/burn/main/Cargo.toml
- [S7] burn-ndarray Cargo.toml @ main (matrixmultiply, blas-* features, macerator simd, rayon): https://github.com/tracel-ai/burn/blob/main/crates/burn-ndarray/Cargo.toml
- [S8] burn-flex README @ main (SIMD coverage, gemm, benchmarks vs ndarray & candle, wasm32/embedded builds): https://github.com/tracel-ai/burn/blob/main/crates/burn-flex/README.md
- [S9] burn-flex Cargo.toml @ main (default std+simd+rayon; x86-v4; apple-amx; wasm-simd128-enable): https://github.com/tracel-ai/burn/blob/main/crates/burn-flex/Cargo.toml
- [S10] burn-wgpu Cargo.toml @ main (features metal/vulkan/webgpu): https://github.com/tracel-ai/burn/blob/main/crates/burn-wgpu/Cargo.toml
- [S11] burn-wgpu README platform table (Metal/Vulkan/OpenGL/WebGPU/Dx11-Dx12 × OS): https://github.com/tracel-ai/burn/blob/main/crates/burn-wgpu/README.md
- [S12] burn-cuda README ("Requires CUDA 12.x"): https://github.com/tracel-ai/burn/blob/main/crates/burn-cuda/README.md
- [S13] burn-tch README (deprecated 0.22.0; LibTorch 2.9.0; WASM: No): https://github.com/tracel-ai/burn/blob/main/crates/burn-tch/README.md
- [S14] burn book — backend & device (runtime Device API; ndarray/libtorch marked deprecated) and saving-and-loading (burnpack/safetensors/pytorch): https://github.com/tracel-ai/burn/blob/main/burn-book/src/building-blocks/backend.md · https://github.com/tracel-ai/burn/blob/main/burn-book/src/saving-and-loading.md
- [S15] burn book — WebAssembly (Flex & WebGpu backends; getrandom note): https://github.com/tracel-ai/burn/blob/main/burn-book/src/advanced/web-assembly.md
- [S16] burn mnist-inference-web example (wasm32-unknown-unknown, no_std flex, flex|wgpu build, live demo): https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web
- [S17] candle-core Cargo.toml @ main (gemm always; mkl/accelerate/cuda/metal/cutile features; rayon; tokenizers wasm32-gated): https://raw.githubusercontent.com/huggingface/candle/main/candle-core/Cargo.toml
- [S18] candle workspace Cargo.toml @ 0.11.0 (gemm 0.19.0 + wasm-simd128-enable; cudarc 0.19.10; cutile 0.3.1; intel-mkl-src 0.8.1; rayon 1.7): https://raw.githubusercontent.com/huggingface/candle/main/Cargo.toml
- [S19] candle GgmlDType enum @ main: https://github.com/huggingface/candle/blob/main/candle-core/src/quantized/mod.rs
- [S20] candle k_quants.rs (AVX2+FMA runtime check; NEON; simd128 vec_dot dispatch): https://github.com/huggingface/candle/blob/main/candle-core/src/quantized/k_quants.rs
- [S21] candle-core/src/quantized/ file listing (avx.rs, neon.rs, simd128.rs, cuda.rs, metal.rs, repack*.rs, gguf_file.rs): https://api.github.com/repos/huggingface/candle/contents/candle-core/src/quantized
- [S22] candle README (MKL/Accelerate/CUDA/NCCL/WASM claims; user-defined kernels; mkl linker pitfalls): https://github.com/huggingface/candle/blob/main/README.md
- [S23] candle CHANGELOG (gguf v2 support; GGUF write API; simd128 quantized vecdots): https://github.com/huggingface/candle/blob/main/CHANGELOG.md
- [S24] candle-wasm-examples tree (11 examples incl. quant-qwen3, bert, whisper): https://github.com/huggingface/candle/tree/main/candle-wasm-examples
- [S25] quant-qwen3 wasm example (browser GGUF parse: m.rs lines 76–88; Cargo.toml wasm plumbing): https://github.com/huggingface/candle/blob/main/candle-wasm-examples/quant-qwen3/src/m.rs · https://github.com/huggingface/candle/blob/main/candle-wasm-examples/quant-qwen3/Cargo.toml
- [S26] quant-qwen3 README (GGUF Q8_0 / Q4_K_M): https://github.com/huggingface/candle/blob/main/candle-wasm-examples/quant-qwen3/README.md
- [S27] candle-wasm-tests README (wasm-pack test --chrome): https://github.com/huggingface/candle/blob/main/candle-wasm-tests/README.md
- [S28] rayon-core lib.rs docs — "Global fallback when threading is unsupported" (wasm32 single-threaded fallback; web_spin_lock): https://github.com/rayon-rs/rayon/blob/main/rayon-core/src/lib.rs
- [S29] rayon CI (wasm32-unknown-unknown builds; wasm32-wasip1 tested in wasmtime): https://github.com/rayon-rs/rayon/blob/main/.github/workflows/ci.yaml
- [S30] candle issue #344 "WebGPU support" (open; updated 2026-04-17): https://github.com/huggingface/candle/issues/344
- [S31] candle issue #1034 "Memory64 Support" (open since 2023-10-09): https://github.com/huggingface/candle/issues/1034
- [S32] burn issue #1187 "Support importing GGUF files" (open; 24 comments; updated 2025-08-08): https://github.com/tracel-ai/burn/issues/1187
- [S33] burn issue #4503 (wasm32 CI check for burn-wgpu getrandom/no-std): https://github.com/tracel-ai/burn/issues/4503
- [S34] burn open wgpu issues (search 2026-09-24): #597 f16 with WGPU — https://github.com/tracel-ai/burn/issues/597 · #3583 WGSL LogicalAnd parse error — https://github.com/tracel-ai/burn/issues/3583 · #3779 Alpine Vulkan no adapter — https://github.com/tracel-ai/burn/issues/3779
- [S35] CubeCL README (supported platforms incl. CPU SIMD; "CubeCL is currently in alpha"; tensor cores not yet on WebGPU; CPU cubes sequential): https://github.com/tracel-ai/cubecl/blob/main/README.md
- [S36] cubecl-wgpu request_device (features = adapter minus MAPPABLE_PRIMARY_BUFFERS; limits = adapter.limits(); Vulkan→Metal→WGSL fallback; macOS vulkan-portability): https://github.com/tracel-ai/cubecl/blob/main/crates/cubecl-wgpu/src/backend/wgsl.rs · .../backend/base.rs · .../Cargo.toml
- [S37] gemm 0.19.0 feature flags (docs.rs; default f16+rayon+std; x86-v4; experimental-apple-amx; wasm-simd128-enable): https://docs.rs/crate/gemm/0.19.0/features
- [S38] crates.io — gemm 0.19.0 (pub 2025-11-14; sarah-ek/gemm): https://crates.io/api/v1/crates/gemm
- [S39] crates.io — burn-candle (max 0.22.0-pre.2 pub 2026-08-10; description "[Deprecated]"): https://crates.io/api/v1/crates/burn-candle
- [S40] burn-candle Cargo.toml @ v0.21.0 (deprecated description; cuda/metal/accelerate passthrough features): https://github.com/tracel-ai/burn/blob/v0.21.0/crates/burn-candle/Cargo.toml
- [S41] candle tensor-tools (quantize/convert to GgmlDType Q4_0…Q8K, F16): https://github.com/huggingface/candle/blob/main/tensor-tools/src/main.rs
- [S42] Repo-root listings (GitHub contents API): tracel-ai/burn and huggingface/candle — no Dockerfile/official image found (checked 2026-09-24)
- [S43] burn-cpu README ("MLIR based CPU backend… Example coming soon") + examples tree (custom-cubecl-kernel, custom-wgpu-kernel): https://github.com/tracel-ai/burn/blob/main/crates/burn-cpu/README.md · https://github.com/tracel-ai/burn/tree/main/examples
