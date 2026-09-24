# Rust ML framework model coverage: Candle vs Burn

**Access date:** 2026-09-24  
**Decision context:** This research feeds a go/no-go decision on migrating a GGUF inference server from a custom `no_std` Rust engine with hand-written AVX2 kernels to Candle or Burn. The target models are **Laya** (28-layer ModernBERT-style encoder with GeGLU) and **OpenThai-SystemOne** (24-layer hybrid: 18 GDN layers + 6 GQA layers, depthwise `conv1d`+SiLU in GDN, F16 weights, Qwen3-Next-style linear-attention hybrid).

## Executive answer

| Requirement | Candle 0.11 / upstream at access date | Burn upstream / 0.21 docs at access date |
|---|---|---|
| ModernBERT | **Yes.** First-party `modernbert.rs` and runnable example. Its MLP explicitly implements GeGLU. | **No first-party ModernBERT model found.** Generic Tensor/NN APIs and ONNX/model-weight import can be used to write one. |
| GDN / Qwen3-Next-style hybrid | **No merged Qwen3-Next/Qwen3.5/GDN model in upstream at the checked commit.** A Qwen3.5 GDN implementation exists in PR #3396, but the referenced files are absent from `main` on 2026-09-24. | **No first-party GDN model found.** Burn has a documented custom-backend/custom-kernel route; third-party `burn-gdn2` is evidence that a Burn GDN implementation is possible, not first-party coverage of this target. |
| `conv1d`, SiLU/GELU, split/chunk | Core Tensor has `conv1d`, `silu`, `gelu`, `gelu_erf`, and `chunk`; GeGLU is composition, not a named core op. | Burn NN/Tensor docs expose convolution/activation modules and Tensor `chunk`/`split`; GeGLU is composition. Exact depthwise causal-conv/state semantics still need model code. |
| GDN pure tensor recurrence | **Feasible.** Candle's own Qwen3.5 proposal describes an optimized pure-tensor sequential recurrence. CPU and CUDA use normal ops, but prefill remains sequential without a fused scan. | **Feasible in principle** by threading state through Rust loops over Burn Tensor operations; autodiff is supported by the Tensor API. It is not a first-party GDN implementation. |
| Custom kernel | Public `CustomOp1/2/3` traits support CPU and CUDA/Metal implementations; first-party custom-op example covers CPU+CUDA. | **Documented and supported extension path.** `backend_extension` plus CubeCL/WGPU examples; custom backward must be supplied for training. More invasive than composing existing ops, but does not require forking Burn. |
| GGUF | **Direct reader.** `gguf_file::Content` reads metadata/tensor descriptors and `QTensor` supports F16/BF16/F32 and GGML quantized types including Q8_0. | **No GGUF reader/import path found.** Use a conversion step to SafeTensors/Burn Store or write a GGUF adapter. |
| F16 SafeTensors | First-party `VarBuilder::from_mmaped_safetensors` path. | First-party `burn-store::SafetensorsStore`, including PyTorch key adapter and F16 storage. |

**Bottom line:** Candle is the lower-risk migration for Laya and is the only evaluated framework with a near-complete, first-party ModernBERT path. For OpenThai-SystemOne, neither framework is a turnkey answer. Candle is the better starting point because its tensor API already covers the primitives and its upstream discussion has a pure-tensor GDN design; however, upstream does not yet ship the needed GDN model at the checked commit. Burn is viable if the team wants Burn's backend portability and is willing to own the complete encoder/hybrid model plus a backend extension/custom kernel. A realistic plan is to prototype GDN as pure tensor ops first, then fuse only after profiling.

## 1. Candle

### 1.1 ModernBERT and Laya

**Claim:** Candle-transformers ships a ModernBERT implementation and example as of the checked upstream commit.  
**Evidence:** `candle-transformers/src/models/mod.rs` registers `pub mod modernbert`; the source module documents ModernBERT and exposes `ModernBert`, masked-LM, and sequence-classification types. ([source, accessed 2026-09-24](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5c/candle-transformers/src/models/mod.rs#L1-L2), [implementation](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5c/candle-transformers/src/models/modernbert.rs#L1-L504))

**Claim:** The implementation matches the relevant Laya structure closely enough to be a useful starting point: it has configurable layer count, attention, local/global attention masks, and a GeGLU MLP.  
**Evidence:** The ModernBERT layer loads attention and MLP submodules; the MLP projects to `2 * intermediate_size`, chunks the last dimension, applies `gelu_erf` to one half, multiplies by the other, then projects back. ([source, accessed 2026-09-24](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5c/candle-transformers/src/models/modernbert.rs#L130-L190))

**Claim:** Candle also has a runnable ModernBERT example that downloads `answerdotai/ModernBERT-base` or `answerdotai/ModernBERT-large`, loads SafeTensors or PyTorch weights, and runs masked-LM inference.  
**Evidence:** ([example, accessed 2026-09-24](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5c/candle-examples/examples/modernbert/main.rs#L1-L176))

**Migration implication:** Laya still needs checkpoint-key/config validation, tokenizer/task-head adaptation, and correctness tests against the existing server. It is not a blank architecture port.

### 1.2 GDN / Qwen3-Next-style hybrid

**Claim:** Upstream Candle at commit `66a8cf...` does not register `qwen3_5`, `quantized_qwen3_5`, `qwen3_5_linear_attn_scan`, `qwen3_next`, or `deltanet` modules in the checked model list.  
**Evidence:** The complete checked `models/mod.rs` list contains `qwen2`, `qwen3`, `qwen3_moe`, etc., and `modernbert`, but no Qwen3.5/Next or DeltaNet module. ([model registry, accessed 2026-09-24](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5c/candle-transformers/src/models/mod.rs#L1-L2)) The repository search for `deltanet` and `qwen3_5` returned no results in `huggingface/candle` on 2026-09-24. The direct candidate paths were also absent from `main`: [qwen3_5.rs](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5c/candle-transformers/src/models/qwen3_5.rs), [quantized_qwen3_5.rs](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5c/candle-transformers/src/models/quantized_qwen3_5.rs), [linear scan](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5c/candle-transformers/src/models/qwen3_5_linear_attn_scan.rs).

**Important qualification:** A first-party Candle PR exists for Qwen3.5 dense and GGUF-quantized support, including GDN and a pure tensor path, but it was not merged into the checked upstream commit. ([PR #3396, accessed 2026-09-24](https://github.com/huggingface/candle/pull/3396)) The PR says the optimized tensor path uses matrix multiplications in a sequential loop and that production-grade prefill would benefit from a fused parallel-scan kernel. This is valuable implementation evidence, not current upstream support.

**Claim:** Candle does not have a Qwen3-Next example in the checked examples tree; it has Qwen/Qwen3 examples, but no `qwen3-next`/GDN example.  
**Evidence:** The checked examples listing has `qwen`, `quantized-qwen3`, and `modernbert`; no Qwen3-Next or GDN example. ([examples tree, accessed 2026-09-24](https://github.com/huggingface/candle/tree/66a8cf184a5a519671454066b1b9efd446ec9f5/candle-examples/examples))

### 1.3 Native operations

**Claim:** Candle core has native `conv1d`, `silu`, `gelu`, `gelu_erf`, `erf`, and `chunk`.  
**Evidence:** The current Candle 0.11 API lists `Tensor::conv1d`; it also lists the activation methods and `Tensor::chunk`. ([docs.rs, accessed 2026-09-24](https://docs.rs/candle-core/latest/candle_core/struct.Tensor.html)) Candle's NN ops define `silu` and a named `swiglu` composition (`chunk`, SiLU, multiply). ([source, accessed 2026-09-24](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5/candle-nn/src/ops.rs#L1-L20))

**GeGLU:** There is no need for a special GeGLU primitive: `x.chunk(2, last_dim)`, `gelu_erf` (or the model's specified GELU), and elementwise multiplication are sufficient. Candle's ModernBERT source does exactly this. ([source, accessed 2026-09-24](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5/candle-transformers/src/models/modernbert.rs#L130-L145))

**GDN convolution:** `conv1d` and SiLU exist, but OpenThai's causal/depthwise short-convolution semantics, cache/ring state, layout, padding, and F16 accumulation policy must be implemented and verified by the model port.

### 1.4 Custom-op path and maturity

**Pure tensor recurrence:** Candle supports matmul, elementwise arithmetic, reductions, reshape/slice/chunk, and autograd graph construction. A Rust loop that carries a `[heads, key_dim, value_dim]` state and applies the DeltaNet update one token at a time is therefore feasible. The strongest framework-owned evidence is the unmerged Candle Qwen3.5 PR, which explicitly chose an optimized pure-tensor sequential recurrence before proposing a fused scan. ([PR #3396, accessed 2026-09-24](https://github.com/huggingface/candle/pull/3396))

**Custom kernel:** Candle exposes `CustomOp1`, `CustomOp2`, and `CustomOp3`; the public Tensor API has `apply_op1/2/3` and no-backward variants. ([docs.rs, accessed 2026-09-24](https://docs.rs/candle-core/latest/candle_core/struct.Tensor.html)) The first-party custom-ops example implements forward/backward for CPU and CUDA for RMSNorm. ([README, accessed 2026-09-24](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5c/candle-examples/examples/custom-ops/README.md#L1-L13))

**Assessment:** CPU custom ops are a documented trait implementation route, not a framework fork. CUDA/Metal implementations require backend-specific code and kernel registration/build integration; the example proves the route works, but it is a low-level extension API rather than a high-level stable “register one kernel” plugin system. For this project, a CPU scalar/vector loop can remain a model-local/custom op; a CUDA fused scan is feasible but should be treated as a separate engineering track. There is no equivalent Candle custom-op route for a generic `no_std` AVX2 kernel through the Tensor API: retain a model-specific CPU implementation or integrate below Candle.

## 2. Burn

### 2.1 ModernBERT and BERT-family import

**Claim:** No first-party ModernBERT implementation or ModernBERT example was found in the checked `tracel-ai/burn` repository. The official examples tree contains generic text classification/generation and custom-kernel examples, not a BERT/ModernBERT model.  
**Evidence:** ([Burn examples tree, accessed 2026-09-24](https://github.com/tracel-ai/burn/tree/0b6f7cbf04128e4e6e739a15d281096499941c6e/examples)) Searches performed on 2026-09-24 for `ModernBERT` in `tracel-ai/burn` and for `BERT`/`ModernBERT` in the official examples returned no matches.

**Claim:** Burn can still be used to write/import a BERT-family encoder.  
**Evidence:** Burn documents ONNX import into native Burn Rust APIs, with the explicit limitation that ONNX operator coverage is limited. ([official README, accessed 2026-09-24](https://github.com/tracel-ai/burn/blob/0b6f7cbf04128e4e6e739a15d281096499941c6e/README.md#L1-L1)) Burn's current model-storage documentation supports loading PyTorch and SafeTensors weights into a user-defined Burn model. ([burn-store README, accessed 2026-09-24](https://github.com/tracel-ai/burn/blob/0b6f7cbf04128e4e6e739a15d281096499941c6e/crates/burn-store/README.md#L1-L93))

**Assessment:** ONNX is a possible import path only if the exported ModernBERT graph uses operators supported by the chosen importer. The safer path for Laya is to define the encoder in Burn and load/remap SafeTensors with `burn-store`; neither path is a ready-made ModernBERT equivalent.

### 2.2 GDN / hybrid model coverage

**Claim:** No first-party Burn GDN/Qwen3-Next implementation was found.  
**Evidence:** Searches performed on 2026-09-24 for `GatedDeltaNet`, `gated_delta`, `deltanet`, `qwen3_next`, and `ModernBERT` in `tracel-ai/burn` returned no matches. The official Burn repository structure contains backend, tensor, NN, store, and example crates, but no GDN model crate. ([repository, accessed 2026-09-24](https://github.com/tracel-ai/burn/tree/0b6f7cbf04128e4e6e739a15d281096499941c6e))

**Qualification:** Rust GDN ports do exist outside these two frameworks, so “no support” means no support in upstream Candle/Burn, not “no Rust implementation exists.” Examples found by the exact searches include `sonos/tract`'s GDN recurrent op ([CPU op, accessed 2026-09-24](https://github.com/sonos/tract/blob/main/transformers/src/ops/gdn_recurrent.rs)), `EricLBuehler/mistral.rs`'s GDN layer ([source, accessed 2026-09-24](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-core/src/gdn/layer.rs)), and the third-party Burn crate `burn-gdn2` ([crates.io, accessed 2026-09-24](https://crates.io/crates/burn-gdn2)). These are useful algorithm/kernel references; they do not provide turnkey OpenThai-SystemOne support in upstream Burn.

### 2.3 Native operations

**Claim:** Burn's documented Tensor API has the composition primitives needed for GeGLU and recurrence: `chunk`, `split`, `matmul`, `exp`, `silu`/activation support through the NN layer, slicing/assignment, and autodiff.  
**Evidence:** Burn Tensor 0.21 documents `chunk`, `split`, `matmul`, `exp`, slicing, `slice_assign`, and autodiff methods. ([docs.rs, accessed 2026-09-24](https://docs.rs/burn-tensor/0.21.0/burn_tensor/struct.Tensor.html)) Burn's official `Gelu` module provides exact and approximate GELU and a backend-generic `forward`. ([docs.rs, accessed 2026-09-24](https://docs.rs/burn-nn/0.21.0/burn_nn/activation/struct.Gelu.html))

**Conv1d/SiLU qualification:** Burn NN has convolution modules and activation infrastructure, but the checked official docs/source did not expose a single stable URL that proves the exact current `Conv1d` and `Silu` symbol names for the 0.22-dev repository. Treat these as API items to pin and verify in the selected Burn release before implementation. Even with those modules, depthwise causal convolution plus recurrent cache is model code, not automatic support.

**GeGLU:** As in Candle, implement it as split/chunk + GELU + multiply (or a custom activation module). Burn's official Transformer example uses `nn::Gelu` as a normal module, demonstrating the intended composition style. ([README example, accessed 2026-09-24](https://github.com/tracel-ai/burn/blob/0b6f7cbf04128e4e6e739a15d281096499941c6e/README.md#L1-L1))

### 2.4 Custom-op path and maturity

**Pure tensor recurrence:** Burn's Tensor is backend-generic and supports autodiff; a Rust loop can carry the recurrent state and produce each step with tensor ops. The state should be represented explicitly and passed between calls; this is feasible but is not a built-in scan primitive. For inference-only use, this is a straightforward model implementation; for training, the loop creates a long autodiff graph and should be profiled.

**Documented custom extension:** Burn's backend-extension guide explicitly says to define a backend trait for a new primitive, implement it for supported backends, and expose a backend-independent Tensor function through Dispatch. It shows a default differentiable composition and backend overrides. ([official guide, accessed 2026-09-24](https://github.com/tracel-ai/burn/blob/0b6f7cbf04128e4e6e739a15d281096499941c6e/burn-book/src/advanced/backend-extension/README.md#L1-L80))

**GPU kernels:** Burn documents CubeCL as the recommended portable custom-kernel route and provides both a raw WGSL WGPU example and a CubeCL example. ([backend extension guide, accessed 2026-09-24](https://github.com/tracel-ai/burn/blob/0b6f7cbf04128e4e6e739a15d281096499941c6e/burn-book/src/advanced/backend-extension/README.md#L1-L80), [WGSL example, accessed 2026-09-24](https://github.com/tracel-ai/burn/blob/0b6f7cbf04128e4e6e739a15d281096499941c6e/examples/custom-wgpu-kernel/src/forward.rs#L1-L119), [CubeCL example, accessed 2026-09-24](https://github.com/tracel-ai/burn/blob/0b6f7cbf04128e4e6e739a15d281096499941c6e/examples/custom-cubecl-kernel/src/forward.rs#L1-L72))

**Maturity/effort assessment:** This is a supported, documented extension mechanism, not a required fork. However, a GDN operation with persistent state and a useful training path needs a custom primitive, backend implementations, Dispatch registration, and an explicit `Autodiff` backward implementation (or a default composition). The WGPU example is raw WGSL and runtime-specific; CubeCL is the better portability route for CUDA/WGPU/ROCm/CPU-compatible backends. Burn's extension API is more structured than Candle's low-level custom-op trait, but the implementation surface for a stateful GDN kernel is larger.

## 3. GGUF and weight loading

### 3.1 Candle

**Claim:** Candle can read GGUF tensor metadata and tensor payloads directly.  
**Evidence:** `gguf_file::Content` stores metadata, tensor descriptors, and data offset; `Content::read` parses the file and `Content::tensor` reads a named tensor. `TensorInfo::read` validates block alignment and loads the raw bytes into a `QTensor`. ([source, accessed 2026-09-24](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5/candle-core/src/quantized/gguf_file.rs#L1-L638), [docs.rs module, accessed 2026-09-24](https://docs.rs/candle-core/latest/candle_core/quantized/gguf_file/index.html))

**Claim:** Candle's GGML dtype mapping includes F16/BF16/F32 and common quantized types including Q8_0, K-quants, and Q4/Q5/Q6 variants.  
**Evidence:** ([docs.rs `GgmlDType`, accessed 2026-09-24](https://docs.rs/candle-core/latest/candle_core/quantized/enum.GgmlDType.html))

**Limitation:** GGUF is not a generic model importer. It provides named tensor/metadata access and quantized storage; the model implementation must know the tensor names, shapes, metadata keys, and how to dequantize/use each tensor. The Qwen3 quantized example demonstrates the intended flow: parse `Content`, create a model-specific `from_gguf`, and use quantized matmul weights. ([example, accessed 2026-09-24](https://github.com/huggingface/candle/blob/39355c6c/candle-examples/examples/quantized-qwen3/main.rs#L90-L145), [model, accessed 2026-09-24](https://github.com/huggingface/candle/blob/39355c6c/candle-transformers/src/models/quantized_qwen3.rs#L1-L120))

**SafeTensors:** Candle's `safetensors` module supports direct loading, memory-mapped loading, in-memory loading, and saving. ([docs.rs, accessed 2026-09-24](https://docs.rs/candle-core/latest/candle_core/safetensors/index.html)) The ModernBERT example uses `VarBuilder::from_mmaped_safetensors`. ([example, accessed 2026-09-24](https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5c/candle-examples/examples/modernbert/main.rs#L35-L75))

**Realistic Candle path:**
1. Keep the existing GGUF if it contains the needed F16/Q formats and map its names through a hand-written `OpenThaiSystemOne::from_gguf`.
2. For Laya, use the HF SafeTensors checkpoint and `VarBuilder`; convert only if the checkpoint naming/configuration differs.
3. For OpenThai F16 GGUF, Candle can load tensor payloads directly, but the complete hybrid model and GDN state/update remain hand-written.

### 3.2 Burn

**Claim:** Burn's current official storage path covers SafeTensors and PyTorch, not GGUF.  
**Evidence:** `burn-store` documents SafeTensors, PyTorch, Burnpack, filtering/remapping, F32/F16 conversion, and no-std availability; its quick start uses `SafetensorsStore` and `PytorchStore`. GGUF is not listed as a supported format. ([official README, accessed 2026-09-24](https://github.com/tracel-ai/burn/blob/0b6f7cbf04128e4e6e739a15d281096499941c6e/crates/burn-store/README.md#L1-L93), [migration guide, accessed 2026-09-24](https://github.com/tracel-ai/burn/blob/0b6f7cbf04128e4e6e739a15d281096499941c6e/crates/burn-store/MIGRATION.md#L1-L120))

**Import route:** Burn's ONNX importer is a graph/model importer, not a GGUF reader; official documentation warns that ONNX operator coverage is limited. ([README, accessed 2026-09-24](https://github.com/tracel-ai/burn/blob/0b6f7cbf04128e4e6e739a15d281096499941c6e/README.md#L1-L1))

**Realistic Burn path for F16 weights:**
1. Prefer original HF SafeTensors. Define the Burn model, then load `SafetensorsStore` with `PyTorchToBurnAdapter` and key remapping as needed.
2. If only GGUF is available, use an external conversion step to dequantize/export tensors to SafeTensors (or write a small GGUF reader/adapter that materializes Burn tensors). There is no first-party direct GGUF path in the checked Burn sources.
3. Load F16 storage where supported, or use Burn's documented half-precision adapter/Burnpack path; verify backend dtype behavior because a backend may compute in a different internal precision.

## 4. Effort and go/no-go recommendation

### Laya / ModernBERT

- **Candle: GO for a migration spike.** Start from first-party ModernBERT; validate Laya's 28-layer config, tensor names, tokenizer, pooling/task head, and GeGLU/GELU variant. Expected work shape: integration and parity testing, not an architecture rewrite.
- **Burn: CONDITIONAL GO.** Use only if Burn's backend portability is a primary requirement. Expect a new encoder module plus SafeTensors import/key mapping or ONNX export/import validation.

### OpenThai-SystemOne / GDN hybrid

- **Candle: CONDITIONAL GO, not turnkey.** Implement the 24-layer schedule, GQA layers, depthwise causal conv+SiLU cache, GDN projections/gates, per-head state, and F16 weight mapping. A pure tensor CPU/GPU prototype is the shortest path; the unmerged Candle PR suggests this is viable. A fused scan/custom kernel is an optimization, not a prerequisite for correctness.
- **Burn: CONDITIONAL GO with higher framework-integration risk.** Pure tensor implementation is possible, but no first-party model exists. A portable custom GDN kernel is supported through Burn backend extension/CubeCL, yet the stateful primitive plus autodiff/backends is a larger owned surface.

### Time-boxed planning guidance

The evidence does not justify a precise calendar estimate without hidden dimensions, sequence lengths, target devices, and throughput targets. A defensible shape is:

- **Candle CPU pure-tensor GDN:** approximately **several engineer-days to 1–2 weeks** for a first correct model-local recurrence, including GGUF name mapping and parity tests; performance may be poor for long prefill because the recurrence is sequential.
- **Candle fused CPU/CUDA path:** add a separate optimization phase measured in **weeks, not days**, because it requires layout/state design, backend kernel code, and numerical parity. The custom-op example proves the extension mechanism, not production GDN performance.
- **Burn pure tensor GDN:** similar algorithmic effort plus model/backend generic integration; budget **1–2 weeks for a correctness prototype** and more for backend-specific performance.
- **Burn custom CubeCL/WGPU/CUDA GDN:** budget **weeks** for the primitive, dispatch/extension, state management, and (if training) backward implementation. Do not make this the first milestone.

**Recommendation:** Do not approve a wholesale replacement based on “framework supports both models.” Approve a two-stage spike: (1) Candle ModernBERT/Laya parity, and (2) Candle pure-tensor OpenThai GDN recurrence using the existing GGUF reader. If stage (2) meets correctness and acceptable CPU throughput, decide whether a fused Candle kernel is warranted. Evaluate Burn in parallel only if its backend portability or training/autodiff story outweighs the extra model and import work.

## Appendix A — searches performed and negative results

All searches below were performed/accessed on **2026-09-24**.

| Exact search/check | Where | Result |
|---|---|---|
| List `candle-transformers/src/models/` | `huggingface/candle` GitHub tree and `models/mod.rs` | `modernbert.rs` exists; no `qwen3_next`, `qwen3next`, `qwen3_5`, `deltanet`, or `gated_delta` module in checked upstream. |
| List `candle-examples/examples/` | `huggingface/candle` GitHub tree | `modernbert` exists; no Qwen3-Next/GDN example. Qwen3 examples are for Qwen2/Qwen3 families. |
| Search `deltanet` | GitHub repository code search, repo `huggingface/candle`, Rust | **No results.** |
| Search `qwen3_5` | GitHub repository code search, repo `huggingface/candle`, Rust | **No results.** |
| Search `GatedDeltaNet` | GitHub repository code search, repo `tracel-ai/burn`, Rust/Markdown | **No results.** |
| Search `gated_delta` | GitHub repository code search, repo `tracel-ai/burn`, Rust | **No results.** |
| Search `ModernBERT` | GitHub repository code search, repo `tracel-ai/burn`, Rust/Markdown | **No results.** |
| Search `GatedDeltaNet` | GitHub-wide code search, Rust | Results included `sonos/tract`, `mistral.rs`, and other third-party ports; therefore Rust GDN exists, just not upstream Candle/Burn. |
| Search `gated deltanet rust` | Web search | Results included third-party Rust crates/projects such as `burn-gdn2`, `inferencelayer`, `mistral.rs`, and `tract`. |
| Search `delta net rust` | Web search | Results included third-party GDN implementations; no first-party Candle/Burn result establishing upstream support. |
| Search crates.io for `gated deltanet` / `delta net` | Web/crates search | `burn-gdn2` exists; it is third-party Burn GDN2, not an upstream Burn model implementation. |
| Direct candidate paths | GitHub `main` | `candle-transformers/src/models/qwen3_5.rs`, `quantized_qwen3_5.rs`, `qwen3_5_linear_attn_scan.rs`, and the corresponding `quantized-qwen3_5` example were absent at the checked upstream commit. |

## Appendix B — primary-source index

- Candle HEAD used for source permalinks: [`66a8cf184a5a519671454066b1b9efd446ec9f5c`](https://github.com/huggingface/candle/commit/66a8cf184a5a519671454066b1b9efd446ec9f5), accessed 2026-09-24.
- Burn HEAD used for source permalinks: [`0b6f7cbf04128e4e6e739a15d281096499941c6e`](https://github.com/tracel-ai/burn/commit/0b6f7cbf04128e4e6e739a15d281096499941c6e), accessed 2026-09-24.
- Official Candle docs: [Candle documentation](https://huggingface.github.io/candle/), accessed 2026-09-24.
- Official Burn docs: [Burn Book](https://burn.dev/books/burn/), accessed 2026-09-24.
- Resolved official Context7 libraries: [`/huggingface/candle`](https://context7.com/huggingface/candle) and [`/tracel-ai/burn`](https://context7.com/tracel-ai/burn), resolved/query accessed 2026-09-24.
