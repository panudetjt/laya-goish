# Laya-Goish Benchmark & Architecture Report

**Platform:** WSL2 on Windows 11 (Linux kernel 5.15.167.4-microsoft-standard-WSL2)  
**Date:** September 2026  
**Tested Revision:** `6dd3afc246a2076a0f78a94cacf5b6682fc304d5` (Goish alpha.14) vs v0.1.0 (Goish alpha.13)

---

## 1. System Specifications

| Component | Specification |
|---|---|
| **CPU** | Intel(R) Core(TM) i5-14400 (14th Gen) |
| **Cores / Threads** | 10 Cores (6 Performance-Cores + 4 Efficient-Cores), 16 Threads |
| **Vector Extensions** | AVX2, FMA, BMI1, BMI2, SSE4.2 (Hardware Native) |
| **Memory** | 26 GB RAM assigned to WSL2 |
| **Dedicated GPU** | NVIDIA GeForce RTX 5060 Ti (CUDA 13.3, Driver 610.57) |
| **Integrated GPU** | Intel(R) UHD Graphics 730 |
| **Toolchain** | Managed via `mise.toml`: Rust 1.96.0 (stable-x86_64), Python 3.12 |

---

## 2. GPU Feasibility & Architectural Finding

### Verdict: **Pure CPU Only — No GPU Path Exists**

A thorough source-code and upstream repository audit confirmed:

1. **Official Position:**  
   `docs/getting-started.md:121` states:
   > *"There is no GPU backend in this release."*
2. **Freestanding Engine Design:**  
   Laya-Goish is implemented in `#![no_std]` and `#![no_main]` Rust. The core matrix multiplication (`Matrix::apply` in `src/tensor.rs`) is written in direct x86-64 SIMD assembly intrinsics (`core::arch::x86_64::_mm256_fmadd_ps`) dispatched via Goish goroutines (`go!`).
3. **No GPU Kernels in Dependency:**  
   The underlying runtime `goish` (tested at both pinned revision `e72d157` and published `1.0.0-alpha.14`) contains no CUDA, Vulkan, OpenCL, or WGPU compute code. It is a freestanding implementation of the Go runtime and standard library in no_std Rust.
4. **Optimization Path:**  
   Because GPU acceleration is unavailable without rewriting the engine into a different framework, maximum performance on this machine is achieved through **CPU vectorization, thread saturation (matching the 16 hardware threads), and Goish scheduler updates**.

---

## 3. Benchmark Methodology

- **Model:** `models/laya_english_f16.gguf` (807 MiB, 28 transformer layers, F16 weights)
- **Endpoint:** `POST http://127.0.0.1:8080/v1/decide`
- **Request:** `examples/refund.json` (45 input tokens, 1 boolean decision question)
- **Protocol:** 1 warm-up request (for page-in and weight mapping) followed by 8 measured sequential requests.
- **Metric:** `usage.latency_ms` reported directly by the engine's internal timer.

---

## 4. Benchmark Results

### 4.1 Thread Scaling on i5-14400 (v0.1.0 Shipped Binary)

Tests performed across thread configurations to observe CPU scaling behavior:

| Threads | Measured Latencies (ms) | Mean Latency | Speedup vs Baseline (4T) |
|:---:|---|:---:|:---:|
| **2** | 640, 687, 747, 615, 659, 681, 606, 671 | **663.2 ms** | 0.64x (Slower) |
| **4** (Default) | 426, 433, 419, 411, 427, 426, 411, 454 | **425.9 ms** | 1.00x (Baseline) |
| **8** | 365, 370, 370, 363, 374, 367, 375, 373 | **369.6 ms** | 1.15x |
| **16** (Max vCPU) | 285, 302, 285, 294, 290, 316, 304, 290 | **295.7 ms** | **1.44x faster** |

*Official Reference baseline in README: ~1,955 ms.*  
*16 threads on i5-14400 achieves a **6.6x speedup** over the upstream reference.*

---

### 4.2 Upstream Update Impact: Goish alpha.13 vs alpha.14

Commit `6dd3afc246a2076a0f78a94cacf5b6682fc304d5` updated the Goish dependency to `1.0.0-alpha.14`.  
**Key Architectural Change:** Alpha.14 uses `XSAVE`/`XRSTOR` to preserve YMM vector register state during asynchronous preemption. This eliminated the need to acquire and hold the OS thread lock (`acquirem()` / `releasem()`) inside the inner SIMD kernel wrapper in `src/tensor.rs`.

Comparison measured with identical input at 4 and 16 threads:

| Configuration | Goish alpha.13 Mean | Goish alpha.14 Mean | Improvement |
|---|:---:|:---:|:---:|
| **4 Threads** | 425.9 ms | **402.5 ms** | **+5.5% faster** |
| **16 Threads** | 295.7 ms | **270.8 ms** | **+8.4% faster** |

#### Alpha.14 Sample Measurements (16 Threads):
`274, 274, 279, 259, 282, 264, 284, 264 ms` (Min: **259 ms**, Mean: **270.8 ms**)

---

### 4.3 Shipped Binary vs Local Native Build (`target-cpu=native`)

| Build Variant | Compiler Flags | 16T Mean Latency | Note |
|---|---|:---:|---|
| **Shipped Binary (alpha.13)** | Release static | 295.7 ms | Pre-compiled AVX2 assembly |
| **Native Build (alpha.13)** | `-C target-cpu=native` + static | 313.4 ms | Hand-tuned AVX2 kernel already optimal |
| **Release Build (alpha.14)** | Official `build-release.py` | **270.8 ms** | Fastened by runtime scheduler improvement |

---

### 4.4 Result Correctness & Thai Language Inference (Laya English model)

In all test iterations across every binary and thread count, the returned decision probability remained strictly deterministic:
```json
{
  "model": "laya",
  "answers": {
    "refund": {
      "type": "noul",
      "noul": 0.8911915421485901,
      "confidence": 0.8911915421485901,
      "probabilities": {
        "false": 0.1088084876537323,
        "true": 0.8911915421485901
      },
      "action": { "act_probability": 1 }
    }
  }
}
```

#### Thai Request Inference Test:
```json
// Input: "ลูกค้าขอคืนเงินสินค้าที่ได้รับชำรุด" (Customer asking for refund for damaged item)
// Tokens: 73 tokens
// Latency: 597 ms (16 threads, alpha.14)
{
  "refund": {
    "type": "noul",
    "noul": 0.41935548186302185,
    "confidence": 0.5806444883346558,
    "probabilities": {
      "false": 0.5806445479393005,
      "true": 0.41935548186302185
    }
  }
}
```

---

## 5. OpenThai-SystemOne Benchmark

### 5.1 Model Conversion & Setup

The OpenThai-SystemOne v0.3 checkpoint (`iapp/OpenThai-SystemOne` @ `f3709948b5e3cc9606a57e74ba62b7a639d17dd3`) was converted to GGUF on this machine:

| Property | Value |
|---|---|
| **File** | `models/openthai_systemone_v03_f16.gguf` (1.5 GiB) |
| **Architecture** | `openthai_systemone` — 24 layers (18 Gated-DeltaNet + 6 grouped-query attention, Qwen3.5 text stack) |
| **Tensors** | 323 (F16 matrices, F32 vectors) |
| **Max conversion error** | 2.98e-08 (FP16 precision floor) |
| **SHA-256** | `2441d5b503d6c3d74374bf2e0a85adf371d154eef28a72e906cd33e4cf653bb5` |
| **Context length** | 4,096 tokens |
| **Conversion command** | `.venv/bin/python scripts/convert-openthai.py --download` |

Conversion checksum matches the upstream reference `validation/openthai-conversion.report.json` exactly.

### 5.2 Methodology

- **Endpoint:** `POST http://127.0.0.1:8081/v1/systemone`
- **Request:** `examples/openthai-thai.json` (161 input tokens, 3 questions: `choice` + `score` + `noul`)
- **Protocol:** 1 warm-up request, then 8 measured sequential requests per thread configuration
- **Metric:** `usage.latency_ms` from the engine's internal timer
- **Binary:** `./dist/laya` (alpha.14 release build)

### 5.3 Thread Scaling Results

| Threads | Measured Latencies (ms) | Mean Latency | Speedup vs 2T |
|:---:|---|:---:|:---:|
| **2** | 3686, 3651, 3592, 3517, 3398, 3231, 3409, 3549 | **3441.6 ms** | 1.00x |
| **4** | 2311, 2300, 2453, 2387, 2318, 2276, 2208, 2131 | **2298.0 ms** | 1.50x |
| **8** | 1821, 1844, 1856, 1738, 1778, 1858, 1888, 2472 | **1906.9 ms** | 1.80x |
| **16** (Max vCPU) | 1879, 1547, 1437, 1515, 1520, 1580, 1766, 1729 | **1621.6 ms** | **2.12x** |

### 5.4 Inference Result (Deterministic Across All Runs)

Input state: *"ลูกค้าแจ้งว่าโดนหักเงินซ้ำสองครั้ง ขอเงินคืนด่วน โทรมาสามรอบแล้ว"* (customer reports duplicate charge, demands urgent refund, called 3 times)

| Question | Type | Result | Confidence |
|---|---|---|---|
| `department` | choice | **billing** (97.37%) vs technical 0.49% / sales 2.14% | 0.878, abstain 0.87% |
| `frustration` | score (0-2) | **1.975 / 2.0** → "โกรธมาก" (very angry, 97.86%) | 0.897 |
| `refund_requested` | noul | **true (0.9859)** | — |

CLI (`laya decide`) and HTTP (`/v1/systemone`) return bit-identical probabilities at every thread count — thread count affects only latency, never output.

### 5.5 Laya vs OpenThai Comparison (alpha.14, identical protocol)

| Metric | Laya English F16 | OpenThai-SystemOne v0.3 |
|---|:---:|:---:|
| Model file | 807 MiB | 1.5 GiB |
| Layers | 28 (ModernBERT-style) | 24 (Gated-DeltaNet + GQA) |
| Request size | 45 tokens, 1 question | 161 tokens, 3 questions |
| **4 threads** | **402.5 ms** | **2298.0 ms** (5.7x slower) |
| **16 threads** | **270.8 ms** | **1621.6 ms** (6.0x slower) |
| Scaling efficiency (4T→16T) | 1.49x faster | 1.42x faster |
| Output determinism | bit-identical | bit-identical |

The ~6x gap is architectural, not a defect: OpenThai runs 24 sequential transformer layers over a 3.5x longer input with 3 separate answer positions read from one causal forward pass, and options with >10 choices would additionally average 8 cyclic permutation orders (this request used 3 options, so `permutations: 1`). Laya's 28-layer encoder benefits from shorter inputs and a single-question-per-pass design.

### 5.6 Recommended OpenThai Configuration

```bash
# Fastest measured configuration on i5-14400 (~1.62 s per request)
./dist/laya serve models/openthai_systemone_v03_f16.gguf --host 127.0.0.1 --port 8081 --threads 16

# Dual-model server (route by request "model" field)
./dist/laya serve models/laya_english_f16.gguf \
  --model models/openthai_systemone_v03_f16.gguf \
  --default-model laya --port 8080 --threads 16
```

---

## 6. Summary & Recommended Operations

1. **Recommended Server Launch Command:**
   ```bash
   ./dist/laya serve models/laya_english_f16.gguf --host 127.0.0.1 --port 8080 --threads 16
   ```
2. **Key Takeaway:**
   - Setting `--threads 16` yields the highest throughput on the Intel i5-14400 (utilizing all 10 cores / 16 threads).
   - Goish `1.0.0-alpha.14` provides a measurable ~8.4% latency drop over alpha.13 thanks to non-blocking SIMD preemption.
   - GPU utilization is unsupported by design, but CPU latency of ~260–275 ms is fully viable for real-time classification workflows.
   - OpenThai-SystemOne runs correctly on the same binary; expect ~1.6 s per Thai decision at 16 threads (see section 5).

> **Note:** sections 1–6 record the v0.1.0 / alpha.14 baseline as shipped. Section 7 supersedes the operational recommendations with the optimized engine.

---

## 7. Optimization Results (2026-09-24, commits `80f27f9` + `8fdf3bc`)

Same machine, same harness as sections 3–5 (1 warm-up + 8 measured requests, `usage.latency_ms`).

### 7.1 What changed

**`80f27f9` — matmul memory-traffic rework (outputs bit-identical):**

1. **F16-resident weights**: matrices keep their on-disk F16 layout in RAM and the SIMD kernel converts halves to floats via F16C on the fly. F16→F32 is exact, so results are unchanged while streamed weight bytes and resident weight RAM halve.
2. **12-row register blocking**: each streamed weight load is amortized across up to 12 token rows (const-generic `block::<NR>` keeps accumulators in YMM registers) instead of 4, cutting passes over each weight matrix ~3×.
3. **Output-feature partitioning**: workers own disjoint weight-row slices instead of token rows, so the weight matrix streams once per row block regardless of worker count, and small token counts still engage every thread.

**`8fdf3bc` — Phase 2 round:**

1. **GDN head-parallelism**: the Gated-DeltaNet recurrence runs its 16 independent heads across workers (per-head order unchanged, disjoint outputs) — bit-identical, OpenThai 16T 1129.4 → 934.4 ms measured in isolation.
2. **AVX2+FMA `dot`**: 8-wide partial sums replace the sequential scalar reduction. Max probability drift vs the previous binary: **9.2e-07** (adoption gate was 5e-4), zero answer changes.
3. **Q8_0 residency (opt-in)**: matrices can stay resident as Q8_0 blocks (34 B / 32 values) dequantized in-kernel; `scripts/quantize-q8.py` requantizes F16 GGUFs.

Dropped with rationale: weighted P/E-core partitions (Goish exposes no thread-affinity API; prefer `--threads 12` on 6P+4E hybrids where 12T wins), request batching (no measured concurrent load), GELU/norm vectorization (secondary to `dot`).

### 7.2 Latency & memory

| Config | Baseline (1d94f0b) | After `80f27f9` | Final (`8fdf3bc`) | Total |
|---|:---:|:---:|:---:|:---:|
| Laya F16 16T | 292.5 ms | 230.3 ms | **205.0 ms** | **1.43×** |
| Laya F16 12T | — | ~205 ms | 220.9 ms | |
| OpenThai F16 16T | 1319.6 ms | 1129.4 ms | **905.4 ms** | **1.46×** |
| OpenThai F16 12T | — | ~1085 ms | **870.6 ms** | **1.52×** (best) |
| Laya Q8_0 16T / 12T | — | — | 214.0 / 221.5 ms | latency-neutral |
| OpenThai Q8_0 16T / 12T | — | — | 915.9 / 934.5 ms | latency-neutral |
| Startup (load + 1 inference, Laya) | 1.41 s | 0.56 s | 0.56 s | 2.5× |

Resident RAM (VmHWM): Laya 2.38 → 1.59 GiB (F16) → **0.95 GiB** (Q8_0); OpenThai 2.89 → 1.96 GiB (F16) → **1.10 GiB** (Q8_0).

Thread guidance flipped with the optimizations: Laya is now fastest at `--threads 16` (scalar share shrank, so E-core straggle matters less), while OpenThai remains fastest at `--threads 12`.

### 7.3 Determinism

- **F16 path stays bit-identical** through every change above, verified against the baseline binary across both models, all reference requests, thread counts 1/2/4/5/7/12/16, and CLI/serve/daemon paths.
- Q8_0 carries quantization error by design; see 7.4.

### 7.4 Q8_0 decision (measured, not assumed)

Drift measured across 6 reference cases (2 models × 3 requests, including crafted `examples/urgent-score.json` and `examples/openthai-mixed.json`):

| Metric | Gate | Measured |
|---|---|---|
| Discrete choice flips | 0 | **0** |
| max \|Δprobability\| | ≤ 0.02 | **0.0178** |
| max \|Δscore\| | ≤ 0.10 | **0.0181** |

**Decision: adopted as an opt-in model variant.** F16 remains the default and the only bit-identical path; Q8_0 trades bit-exactness for ~half the resident RAM at equal latency (≈2× models per host). One rule is baked into the quantizer: **embedding tables stay F16** — quantizing the Laya embedding table pushed probability drift to 0.0212, past the gate.

The gates are executable and re-runnable:

```bash
scripts/quantize-q8.py models/laya_english_f16.gguf models/laya_english_q8_0.gguf
scripts/quantize-q8.py models/openthai_systemone_v03_f16.gguf models/openthai_systemone_v03_q8_0.gguf
.venv/bin/python3 validation/quantized-drift.py ./laya \
  models/laya_english_f16.gguf models/laya_english_q8_0.gguf \
  models/openthai_systemone_v03_f16.gguf models/openthai_systemone_v03_q8_0.gguf examples
```

### 7.5 Recommended operations (updated)

```bash
# Fastest Laya English configuration on i5-14400 (~205 ms per request)
./laya serve models/laya_english_f16.gguf --host 127.0.0.1 --port 8080 --threads 16

# Fastest OpenThai configuration (~0.87 s per Thai decision)
./laya serve models/openthai_systemone_v03_f16.gguf --host 127.0.0.1 --port 8081 --threads 12

# Memory-constrained deployments: Q8_0 variants, equal speed, ~half the RAM
./laya serve models/laya_english_q8_0.gguf --port 8080 --threads 16
./laya serve models/openthai_systemone_v03_q8_0.gguf --port 8081 --threads 12
```

Requires AVX2/FMA/F16C (checked at startup with a clear error on unsupported CPUs).
