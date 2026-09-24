# iGPU compute tier

The GPU engine's kernel layer, ported to integrated-GPU runtimes, for hosts
with no discrete GPU. This is the deployment story for that tier (ticket 06);
the kernel work itself is gated on the probe sequence below.

## Where it runs, where it does not

| Host | Status |
| --- | --- |
| Bare-metal Linux with a real `/dev/dri` (i915/xe) | **Target.** ANV or Level Zero/OpenCL reach the device. |
| Windows native | **Target** (via Intel Compute Runtime). |
| WSL2 | **Out of scope.** Measured (ticket 03 §4): Mesa enumerates llvmpipe only; `/dev/dxg` exposes the NVIDIA adapter for CUDA only; the only DRM node is virtual (`vgem`). No Vulkan-on-D3D12 ICD ships for this stack. |

## The gate sequence (never reorder it)

1. **Probe first, on the candidate host.** Build and run the probe:

   ```sh
   cd sidecar
   RUSTFLAGS="-Cforce-frame-pointers=yes" cargo build --release --bin igpu-probe
   ./target/x86_64-unknown-linux-gnu/release/igpu-probe > probe.json
   ```

   The gate: `recommended_tier == "igpu"` **and** the device shows up as a
   real GPU through both the Vulkan and the Level Zero/OpenCL sections
   (`is_real_device: true`, no llvmpipe/lavapipe, no `vgem`). A probe that
   lists only llvmpipe or only `vgem` disqualifies the host — do not write
   kernels for it. Record the JSON in the ticket.

2. **Pick the runtime by measurement on that device.** Run an F16-storage
   matmul microbench through Vulkan and through Level Zero/OpenCL on the same
   GPU; the winner becomes the tier's runtime. The choice is never made by
   preference — same discipline as ticket 03's CUDA-vs-Vulkan call.

3. **Only then port kernels**, reusing ticket 03's graph-spec walk, F16
   weight residency, and declared drift gates unchanged. fp16 arithmetic
   (`shaderFloat16` / `cl_khr_fp16`) is allowed where cheap, but only behind
   the drift gates; the default port mirrors the CUDA path: F16 storage,
   F32 accumulate, CPU-order reductions.

## Performance envelope (sized by the analytic bound)

UHD 730 ≈ 24 EU × 8-wide FMA × 2 × ~1.9 GHz ≈ **0.7 TFLOPS F32** → a ~45 ms
compute floor for the ~33 GFLOP Laya forward, while sharing the CPU's DDR5
ceiling. The win condition is **beating the CPU engine where no dGPU exists**
(CPU 16T reference 152–205 ms → 3–4× available), not matching the CUDA tier
(33 ms on the RTX 5060 Ti). The probe reports the same bound per Level Zero
device (`analytic_f32_tflops`).

## Driver packages (Ubuntu 24.04 bare-metal)

| Component | Package | Provides |
| --- | --- | --- |
| Vulkan (ANV) | `mesa-vulkan-intel` | `libvulkan_intel.so`, `intel_icd.json` |
| Level Zero loader | `libze_loader1` (oneAPI) or `level-zero` | `libze_loader.so.1` |
| Level Zero GPU driver | `intel-level-zero-gpu` | L0 driver for Intel Compute Runtime |
| OpenCL | `intel-opencl-icd` | `libOpenCL.so` vendor ICD, `cl_khr_fp16` |
| Kernel driver | `i915` or `xe` (in-tree) + `linux-firmware` | DRM card/render nodes |

No NVIDIA driver, no CUDA toolkit, no dGPU. Verify with the probe,
`vulkaninfo --summary`, and `clinfo` — all three should show the same Intel
device before step 2 runs.

## Automatic tier selection

`igpu-probe` encodes the selection rule the engine uses (ticket 06's
deployment gate, consumed by ticket 04's kernel-layer seam — no engine code
consumes it yet, so until then it is a documented rule plus a recorded
verdict):

- Any NVIDIA GPU visible → **CUDA tier** (ticket 03's measured choice). The
  probe checks for this directly (dlopens `libcuda.so.1`, `cuInit`) and not
  just through the Vulkan/Level Zero/OpenCL sections, so a host with a
  working CUDA stack but no Mesa-visible NVIDIA still reports `cuda`.
- Else a real GPU through Vulkan and/or Level Zero/OpenCL → **iGPU tier**,
  with the runtime picked by the step-2 microbench.
- Else (llvmpipe-only, vgem-only, nothing) → **CPU tier**; the host is not a
  candidate for iGPU kernel work.

## Acceptance gates (from ticket 06)

- Probe recorded in the ticket before any kernel work.
- Drift gates identical to ticket 03's harness on the reference set through
  the chosen runtime: zero discrete flips, |Δp| ≤ 0.01, |Δscore| ≤ 0.05
  (Laya; OpenThai if ticket 04 ships it).
- Batch-1 Laya latency ≥ 2× the CPU engine **on the same iGPU host**, CPU
  baseline measured there with the BENCHMARK_REPORT §3 protocol.
- Both models resident in GPU-accessible shared memory, no per-request
  weight traffic, verified through the runtime's residency query API.
- This deployment story (this document) stays accurate.
