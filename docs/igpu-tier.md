# iGPU compute tier

The GPU engine's kernel layer, ported to integrated-GPU runtimes, for hosts
with no discrete GPU. This is the deployment story for that tier (ticket 06);
the kernel work itself is gated on the probe sequence below.

## Where it runs, where it does not

| Host | Status |
| --- | --- |
| Bare-metal Linux with a real `/dev/dri` (i915/xe) | **Target.** ANV or Level Zero/OpenCL reach the device. |
| Windows native | **Target** (via Intel Compute Runtime). |
| WSL2 (Ubuntu 24.04) | **Works for the OpenCL runtime, measured.** After `sudo apt-get install intel-opencl-icd libigdgmm12`, compute-runtime's WSL/dxg backend enumerates the UHD 730 inside WSL2 (probe: device 0x4682, 24 CU, 1550 MHz, fp16, NEO 23.43). Vulkan stays llvmpipe-only from stock packages — a self-built Mesa `dzn` (Vulkan-on-D3D12, `-Dvulkan-drivers=microsoft-experimental`) is the escape hatch, at a measured ~30% tax. Level Zero loader (`libze1`) installs but the GPU driver is not in Ubuntu repos, so `zeInit` stays `ERROR_UNINITIALIZED`. |

Known NEO-on-WSL quirks (keep these in mind when reading probe JSONs): the
driver reports `CL_DEVICE_TYPE_CPU` for the iGPU and rejects
`CL_DEVICE_AVAILABLE` with `CL_INVALID_VALUE`, so the probe judges
real-devices by name, and its OpenCL `available` field can be `null`.

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

## Performance envelope (measured 2026-09-24 on UHD 730 [0x4682], WSL2/OpenCL)

- Analytic peak: 24 EU × 8-wide FMA × 2 × 1.55 GHz ≈ **0.595 TFLOPS F32**
  (matches `igpu-probe`'s `analytic_f32_tflops`).
- Measured, tiled F16-storage / F32-accumulate matmul on the engine's real
  shapes: **0.068–0.079 TFLOPS sustained** (11–13% of peak) after kernel
  tuning; correctness vs host reference 2.0e-6 max rel err.
- Projected batch-1 Laya forward at s=40 (33.7 GFLOP): ~433 ms vs the
  CPU engine's 169.2 ms on this host — **0.39×, NO-GO** against the
  ≥ 2× win condition. Meeting the gate requires 71% of theoretical peak
  sustained; `validation/igpu-bench.wsl2.json` has the full record.
- Ticket 06 is therefore parked. Revisit triggers: bare-metal measurement
  ≥ 0.45 TF on this silicon with a tuned GEMM sustaining > 75% of it —
  or a product shift toward hosts with far slower CPU baselines.

## Driver packages (Ubuntu 24.04 bare-metal)

| Component | Package | Provides |
| --- | --- | --- |
| Vulkan (ANV) | `mesa-vulkan-intel` | `libvulkan_intel.so`, `intel_icd.json` |
| OpenCL (works in WSL2 too) | `intel-opencl-icd` + `libigdgmm12` | `libigdrcl.so` vendor ICD, `cl_khr_fp16` |
| Level Zero loader | `libze1` (Ubuntu 24.04 name for `level-zero`) | `libze_loader.so.1` |
| Level Zero GPU driver | `intel-level-zero-gpu` (not in Ubuntu repos — Intel's GPU APT repo) | L0 driver for Intel Compute Runtime |
| Kernel driver (bare metal) | `i915` or `xe` (in-tree) + `linux-firmware` | DRM card/render nodes |

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
