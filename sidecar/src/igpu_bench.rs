//! iGPU F16-storage matmul microbench (ticket 06 gate step 2): measure the
//! UHD 730's real matmul throughput through OpenCL on the engine's exact
//! tensor shapes, verify F32-accumulate correctness against a host
//! reference, and project batch-1 forward latency against the CPU engine's
//! measured baseline.
//!
//! Discipline mirrored from the CUDA path: F16 weight storage, F32
//! accumulate, sequential-k order. This tool decides the runtime gate —
//! it does not port kernels.
use serde_json::{json, Value};
use std::ffi::c_void;
use std::ffi::CStr;
use std::time::Instant;

// ---------------------------------------------------------------------------
// dlopen plumbing (same pattern as igpu_probe.rs — kept self-contained so the
// two diagnostic bins stay independent).
// ---------------------------------------------------------------------------

const RTLD_NOW: i32 = 2;

struct Lib(*mut c_void);

impl Lib {
    fn open(names: &[&CStr]) -> Result<Lib, String> {
        let mut last_err = String::new();
        for name in names {
            unsafe {
                dlerror();
                let h = dlopen(name.as_ptr(), RTLD_NOW);
                if !h.is_null() {
                    return Ok(Lib(h));
                }
                let e = dlerror();
                last_err = if e.is_null() {
                    format!("{:?}", name.to_string_lossy())
                } else {
                    CStr::from_ptr(e).to_string_lossy().into_owned()
                };
            }
        }
        Err(format!("not_installed: {last_err}"))
    }

    unsafe fn sym<T: Copy>(&self, name: &CStr) -> Result<T, String> {
        let p = dlsym(self.0, name.as_ptr());
        if p.is_null() {
            return Err(format!("symbol {} missing", name.to_string_lossy()));
        }
        debug_assert_eq!(
            std::mem::size_of::<T>(),
            std::mem::size_of::<*mut c_void>(),
            "sym<T> only converts function pointers"
        );
        Ok(std::mem::transmute_copy(&p))
    }
}

impl Drop for Lib {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { dlclose(self.0) };
        }
    }
}

extern "C" {
    fn dlopen(filename: *const i8, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
    fn dlerror() -> *const i8;
}

// ---------------------------------------------------------------------------
// Minimal OpenCL 1.2 surface (values from CL/cl.h).
// ---------------------------------------------------------------------------

type ClInt = i32;
type ClUint = u32;
type ClBool = u32;
type ClDeviceId = *mut c_void;
type ClContext = *mut c_void;
type ClQueue = *mut c_void;
type ClMem = *mut c_void;
type ClProgram = *mut c_void;
type ClKernel = *mut c_void;
type ClEvent = *mut c_void;

const CL_DEVICE_TYPE_GPU: cl_bitfield = 1 << 1;
#[allow(non_camel_case_types)]
type cl_bitfield = u64;

const CL_DEVICE_NAME: ClUint = 0x102B;
const CL_DEVICE_MAX_COMPUTE_UNITS: ClUint = 0x1002;
const CL_DEVICE_MAX_CLOCK_FREQUENCY: ClUint = 0x100C;
const CL_DEVICE_MAX_MEM_ALLOC_SIZE: ClUint = 0x1010;
const CL_DEVICE_HALF_FP_CONFIG: ClUint = 0x1033;

const CL_MEM_READ_ONLY: ClUint = 4;
const CL_MEM_WRITE_ONLY: ClUint = 2;
const CL_MEM_COPY_HOST_PTR: ClUint = 32;

const CL_QUEUE_PROFILING_ENABLE: cl_bitfield = 1 << 1;
const CL_PROFILING_COMMAND_START: ClUint = 0x1280;
const CL_PROFILING_COMMAND_END: ClUint = 0x1281;
const CL_PROGRAM_BUILD_LOG: ClUint = 0x1183;
const CL_TRUE: ClBool = 1;

extern "C" fn on_context_destroyed(_: *const i8, _: *mut c_void, _: usize, _: *mut c_void) {}

// ---------------------------------------------------------------------------
// f32 <-> f16 bit conversion (round-to-nearest-even on encode; data is
// normal-range so subnormal branches exist for completeness only).
// ---------------------------------------------------------------------------

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_field = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp_field == 0xFF {
        // inf / nan
        return if mant == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7E00
        };
    }
    let exp = exp_field - 127 + 15;
    if exp >= 0x1F {
        return sign | 0x7C00; // overflow -> inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflow -> zero
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let mut h = (m >> shift) as u16;
        // round-to-nearest-even
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && h & 1 == 1) {
            h += 1;
        }
        return sign | h;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x03FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut m = man;
            let mut e = 113i32;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// The kernel: tiled F16-weight matmul, F32 accumulate, sequential k.
// Workgroup 16x16; local tiles 16x32 for both x (F32) and w (F32 after
// convert). One work-item computes one out element.
// ---------------------------------------------------------------------------

const KERNEL_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define LX 16
#define LY 16
#define KC 64
__kernel void matmul_f16(__global float *out,
                         __global const float *x,
                         __global const half *w,
                         __global const float *bias,
                         const int M, const int N, const int K) {
    __local float xs[LY][KC];
    __local float ws[LX][KC];
    const int n0 = (int)get_group_id(0) * LX;
    const int m0 = (int)get_group_id(1) * LY;
    const int ln = (int)get_local_id(0);
    const int lm = (int)get_local_id(1);
    const int n = n0 + ln;
    const int m = m0 + lm;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += KC) {
        // Aligned cooperative loads: work-item (ln, lm) owns the 4-wide
        // slice [4*ln, 4*ln+3] of its xs row and [4*lm, 4*lm+3] of its ws
        // row. vload4/vstore4 stay 16B/8B aligned; after the barrier every
        // work-item reads both full 64-wide rows from local memory.
        const int xm = m0 + lm;
        if (xm < M) {
            vstore4(vload4(0, x + (size_t)xm * K + k0 + 4 * ln), 0, &xs[lm][4 * ln]);
        } else {
            xs[lm][4 * ln] = 0.0f; xs[lm][4 * ln + 1] = 0.0f;
            xs[lm][4 * ln + 2] = 0.0f; xs[lm][4 * ln + 3] = 0.0f;
        }
        const int wn = n0 + ln;
        if (wn < N) {
            vstore4(convert_float4(vload4(0, w + (size_t)wn * K + k0 + 4 * lm)), 0, &ws[ln][4 * lm]);
        } else {
            ws[ln][4 * lm] = 0.0f; ws[ln][4 * lm + 1] = 0.0f;
            ws[ln][4 * lm + 2] = 0.0f; ws[ln][4 * lm + 3] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (n < N && m < M) {
            float4 a = (float4)(0.0f);
            for (int kk = 0; kk < KC; kk += 4) {
                a += vload4(0, &xs[lm][kk]) * vload4(0, &ws[ln][kk]);
            }
            acc += a.x + a.y + a.z + a.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (n < N && m < M) {
        out[(size_t)m * N + n] = acc + bias[n];
    }
}
"#;

const LX: usize = 16;
const LY: usize = 16;

/// The engine's real matmul shapes and their per-forward counts, from a
/// tensor census of models/laya_english_f16.gguf (2D weights >= 8x8;
/// embeddings/rope tables excluded). (N, K, count).
const ENGINE_SHAPES: [(usize, usize, usize); 6] = [
    (3072, 1024, 30), // qkv
    (1024, 1024, 31), // wo + scorer/aux linears
    (5248, 1024, 28), // mlp wi (gate+up)
    (1024, 2624, 28), // mlp down (geglu-halved)
    (4096, 1024, 2),  // head fc1
    (1024, 4096, 2),  // head fc2
];

/// CPU engine baseline on this host (validation/gpu-sidecar.report.md).
const CPU_BASELINE_MS: f64 = 169.2;
/// Win condition (ticket 06): batch-1 >= 2x the CPU engine.
const WIN_RATIO: f64 = 2.0;

fn lcg(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

fn fill_random(v: &mut [f32], state: &mut u64) {
    for x in v.iter_mut() {
        *x = (lcg(state) >> 9) as f32 / 8.388_608e6 - 1.0;
    }
}

fn main() {
    let mut report = json!({
        "bench_version": 1,
        "kernel": "tiled 16x16 local, KC=64, float4/half4 aligned cooperative tiles, F16 storage / F32 accumulate, sequential k",
    });

    let lib = match Lib::open(&[c"libOpenCL.so.1", c"libOpenCL.so"]) {
        Ok(l) => l,
        Err(e) => {
            report["error"] = json!(e);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            return;
        }
    };

    unsafe {
        let get_platform_ids: unsafe extern "C" fn(ClUint, *mut ClDeviceId, *mut ClUint) -> ClInt =
            match lib.sym(c"clGetPlatformIDs") {
                Ok(f) => f,
                Err(e) => {
                    report["error"] = json!(e);
                    println!("{}", serde_json::to_string_pretty(&report).unwrap());
                    return;
                }
            };
        let get_device_ids: unsafe extern "C" fn(
            ClDeviceId,
            cl_bitfield,
            ClUint,
            *mut ClDeviceId,
            *mut ClUint,
        ) -> ClInt = match lib.sym(c"clGetDeviceIDs") {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(e);
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let get_device_info: unsafe extern "C" fn(
            ClDeviceId,
            ClUint,
            usize,
            *mut c_void,
            *mut usize,
        ) -> ClInt = match lib.sym(c"clGetDeviceInfo") {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(e);
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let create_context: unsafe extern "C" fn(
            *const c_void,
            ClUint,
            *mut ClDeviceId,
            unsafe extern "C" fn(*const i8, *mut c_void, usize, *mut c_void),
            *mut c_void,
            *mut ClInt,
        ) -> ClContext = match lib.sym(c"clCreateContext") {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(e);
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let create_queue: unsafe extern "C" fn(
            ClContext,
            ClDeviceId,
            cl_bitfield,
            *mut ClInt,
        ) -> ClQueue = match lib.sym(c"clCreateCommandQueue") {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(format!("{e} (clCreateCommandQueue missing)"));
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let create_buffer: unsafe extern "C" fn(
            ClContext,
            ClUint,
            usize,
            *const c_void,
            *mut ClInt,
        ) -> ClMem = match lib.sym(c"clCreateBuffer") {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(e);
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let create_program: unsafe extern "C" fn(
            ClContext,
            ClUint,
            *mut *const i8,
            *const usize,
            *mut ClInt,
        ) -> ClProgram = match lib.sym(c"clCreateProgramWithSource") {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(e);
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let build_program: unsafe extern "C" fn(
            ClProgram,
            ClUint,
            *mut ClDeviceId,
            *const i8,
            *mut c_void,
            *mut c_void,
        ) -> ClInt = match lib.sym(c"clBuildProgram") {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(e);
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let get_build_log: unsafe extern "C" fn(
            ClProgram,
            ClDeviceId,
            ClUint,
            usize,
            *mut c_void,
            *mut usize,
        ) -> ClInt = match lib.sym(c"clGetProgramBuildInfo") {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(e);
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let create_kernel: unsafe extern "C" fn(ClProgram, *const i8, *mut ClInt) -> ClKernel =
            match lib.sym(c"clCreateKernel") {
                Ok(f) => f,
                Err(e) => {
                    report["error"] = json!(e);
                    println!("{}", serde_json::to_string_pretty(&report).unwrap());
                    return;
                }
            };
        let set_arg: unsafe extern "C" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt =
            match lib.sym(c"clSetKernelArg") {
                Ok(f) => f,
                Err(e) => {
                    report["error"] = json!(e);
                    println!("{}", serde_json::to_string_pretty(&report).unwrap());
                    return;
                }
            };
        let enqueue_kernel: unsafe extern "C" fn(
            ClQueue,
            ClKernel,
            ClUint,
            *const usize,
            *const usize,
            *const usize,
            ClUint,
            *const ClEvent,
            *mut ClEvent,
        ) -> ClInt = match lib.sym(c"clEnqueueNDRangeKernel") {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(e);
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let profiling: unsafe extern "C" fn(ClEvent, ClUint, usize, *mut c_void, *mut usize) -> ClInt =
            match lib.sym(c"clGetEventProfilingInfo") {
                Ok(f) => f,
                Err(e) => {
                    report["error"] = json!(e);
                    println!("{}", serde_json::to_string_pretty(&report).unwrap());
                    return;
                }
            };
        let wait_events: unsafe extern "C" fn(ClUint, *const ClEvent) -> ClInt =
            match lib.sym(c"clWaitForEvents") {
                Ok(f) => f,
                Err(e) => {
                    report["error"] = json!(e);
                    println!("{}", serde_json::to_string_pretty(&report).unwrap());
                    return;
                }
            };
        let finish: unsafe extern "C" fn(ClQueue) -> ClInt = match lib.sym(c"clFinish") {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(e);
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let read_buffer: unsafe extern "C" fn(
            ClQueue,
            ClMem,
            ClBool,
            usize,
            usize,
            *mut c_void,
            ClUint,
            *const ClEvent,
            *mut ClEvent,
        ) -> ClInt = match lib.sym(c"clEnqueueReadBuffer") {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(e);
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let release: unsafe extern "C" fn(*mut c_void) -> ClInt = match lib.sym(c"clReleaseMemObject")
        {
            Ok(f) => f,
            Err(e) => {
                report["error"] = json!(e);
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        };
        let release_event: unsafe extern "C" fn(*mut c_void) -> ClInt = release;
        let release_kernel: unsafe extern "C" fn(*mut c_void) -> ClInt = release;
        let release_program: unsafe extern "C" fn(*mut c_void) -> ClInt = release;
        let release_queue: unsafe extern "C" fn(*mut c_void) -> ClInt = release;
        let release_context: unsafe extern "C" fn(*mut c_void) -> ClInt = release;

        // Device + context + queue ------------------------------------------
        let mut platform: ClDeviceId = std::ptr::null_mut();
        let mut n_platforms: ClUint = 0;
        if get_platform_ids(1, &mut platform, &mut n_platforms) != 0 || n_platforms == 0 {
            report["error"] = json!("no OpenCL platform");
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            return;
        }
        let mut devices = [std::ptr::null_mut::<c_void>(); 8];
        let mut n_devices: ClUint = 0;
        // GPU type first; fall back to whatever exists.
        let _ = get_device_ids(platform, CL_DEVICE_TYPE_GPU, 8, devices.as_mut_ptr(), &mut n_devices);
        if n_devices == 0 {
            let _ = get_device_ids(platform, u32::MAX as cl_bitfield, 8, devices.as_mut_ptr(), &mut n_devices);
        }
        if n_devices == 0 {
            report["error"] = json!("no OpenCL device");
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            return;
        }
        let dev = devices[0];

        let mut name_buf = [0u8; 256];
        let mut got: usize = 0;
        get_device_info(dev, CL_DEVICE_NAME, 256, name_buf.as_mut_ptr() as *mut c_void, &mut got);
        let dev_name = cstr_lossy(&name_buf, got);
        let mut cus: ClUint = 0;
        get_device_info(dev, CL_DEVICE_MAX_COMPUTE_UNITS, 4, &mut cus as *mut ClUint as *mut c_void, &mut got);
        let mut clock: ClUint = 0;
        get_device_info(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, 4, &mut clock as *mut ClUint as *mut c_void, &mut got);
        let mut fp16: u64 = 0;
        get_device_info(dev, CL_DEVICE_HALF_FP_CONFIG, 8, &mut fp16 as *mut u64 as *mut c_void, &mut got);
        let mut max_alloc: u64 = 0;
        get_device_info(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 8, &mut max_alloc as *mut u64 as *mut c_void, &mut got);

        report["device"] = json!({
            "name": dev_name,
            "compute_units": cus,
            "max_clock_mhz": clock,
            "fp16": fp16 != 0,
            "max_mem_alloc_mb": max_alloc as f64 / 1e6,
            "analytic_f32_tflops": cus as f64 * 8.0 * 2.0 * (clock as f64 * 1e6) / 1e12,
        });
        if fp16 == 0 {
            report["error"] = json!("device lacks cl_khr_fp16");
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            return;
        }

        let mut ctx_err: ClInt = 0;
        eprintln!("stage: context");
        let ctx = create_context(
            std::ptr::null(),
            1,
            &mut devices[..1][0] as *mut ClDeviceId,
            on_context_destroyed,
            std::ptr::null_mut(),
            &mut ctx_err,
        );
        if ctx_err != 0 {
            report["error"] = json!(format!("clCreateContext: {ctx_err}"));
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            return;
        }
        let mut queue_err: ClInt = 0;
        eprintln!("stage: queue");
        let queue = create_queue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &mut queue_err);
        if queue_err != 0 {
            report["error"] = json!(format!("clCreateCommandQueue: {queue_err}"));
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            return;
        }

        // Program ------------------------------------------------------------
        let src = KERNEL_SRC.as_bytes();
        let mut src_ptr = src.as_ptr() as *const i8;
        let src_len = src.len();
        let mut prog_err: ClInt = 0;
        eprintln!("stage: program");
        let prog = create_program(ctx, 1, &mut src_ptr, &src_len, &mut prog_err);
        if prog_err != 0 {
            report["error"] = json!(format!("clCreateProgramWithSource: {prog_err}"));
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            return;
        }
        eprintln!("stage: build");
        let build_err = build_program(prog, 1, &mut devices[..1][0] as *mut ClDeviceId, std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut());
        if build_err != 0 {
            let mut log_size: usize = 0;
            get_build_log(prog, dev, CL_PROGRAM_BUILD_LOG, 0, std::ptr::null_mut(), &mut log_size);
            let mut log = vec![0u8; log_size.max(1)];
            get_build_log(prog, dev, CL_PROGRAM_BUILD_LOG, log.len(), log.as_mut_ptr() as *mut c_void, &mut got);
            report["error"] = json!({
                "build_failed": build_err,
                "log": String::from_utf8_lossy(&log).into_owned(),
            });
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            return;
        }
        let kernel_name = c"matmul_f16";
        let mut kern_err: ClInt = 0;
        eprintln!("stage: kernel");
        let kernel = create_kernel(prog, kernel_name.as_ptr(), &mut kern_err);
        if kern_err != 0 {
            report["error"] = json!(format!("clCreateKernel: {kern_err}"));
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            return;
        }

        // Bench ---------------------------------------------------------------
        let mut rng: u64 = 0x9E3779B97F4A7C15;
        let mut results: Vec<Value> = Vec::new();
        let max_m = 256;
        let max_n = ENGINE_SHAPES.iter().map(|s| s.0).max().unwrap_or(0);
        let max_k = ENGINE_SHAPES.iter().map(|s| s.1).max().unwrap_or(0);

        let mut x = vec![0f32; max_m * max_k];
        fill_random(&mut x, &mut rng);
        let mut w_f32 = vec![0f32; max_n * max_k];
        fill_random(&mut w_f32, &mut rng);
        let w_half: Vec<u16> = w_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let bias = vec![0f32; max_n];

        eprintln!("stage: buffers");
        let x_buf = create_buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.len() * 4, x.as_ptr() as *const c_void, &mut queue_err);
        let w_buf = create_buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_half.len() * 2, w_half.as_ptr() as *const c_void, &mut queue_err);
        let bias_buf = create_buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bias.len() * 4, bias.as_ptr() as *const c_void, &mut queue_err);
        let out_buf = create_buffer(ctx, CL_MEM_WRITE_ONLY, max_m * max_n * 4, std::ptr::null_mut(), &mut queue_err);
        for (label, b) in [("x", x_buf), ("w", w_buf), ("bias", bias_buf), ("out", out_buf)] {
            if b.is_null() {
                report["error"] = json!(format!("clCreateBuffer {label} failed"));
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
        }

        eprintln!("stage: correctness");
        // correctness first: one small full check against the host reference
        {
            let (n, k) = (3072usize, 1024usize);
            let m = 8usize;
            let mut m_i = m as ClInt;
            let mut n_i = n as ClInt;
            let mut k_i = k as ClInt;
            set_arg(kernel, 0, std::mem::size_of::<ClMem>(), &out_buf as *const ClMem as *const c_void);
            set_arg(kernel, 1, std::mem::size_of::<ClMem>(), &x_buf as *const ClMem as *const c_void);
            set_arg(kernel, 2, std::mem::size_of::<ClMem>(), &w_buf as *const ClMem as *const c_void);
            set_arg(kernel, 3, std::mem::size_of::<ClMem>(), &bias_buf as *const ClMem as *const c_void);
            set_arg(kernel, 4, std::mem::size_of::<ClInt>(), &mut m_i as *mut ClInt as *const c_void);
            set_arg(kernel, 5, std::mem::size_of::<ClInt>(), &mut n_i as *mut ClInt as *const c_void);
            set_arg(kernel, 6, std::mem::size_of::<ClInt>(), &mut k_i as *mut ClInt as *const c_void);
            let g = [n.div_ceil(LX) * LX, m.div_ceil(LY) * LY];
            let mut ev: ClEvent = std::ptr::null_mut();
            let r = enqueue_kernel(queue, kernel, 2, std::ptr::null(), g.as_ptr(), [LX, LY].as_ptr(), 0, std::ptr::null(), &mut ev);
            finish(queue);
            if r != 0 {
                report["error"] = json!(format!("enqueue failed: {r}"));
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
                return;
            }
            let mut out = vec![0f32; m * n];
            read_buffer(queue, out_buf, CL_TRUE, 0, out.len() * 4, out.as_mut_ptr() as *mut c_void, 0, std::ptr::null(), std::ptr::null_mut());
            let mut max_abs = 0f64;
            let mut max_rel = 0f64;
            for mm in 0..m {
                for nn in 0..n {
                    let mut refv = 0f32;
                    for kk in 0..k {
                        refv += x[mm * k + kk] * f16_bits_to_f32(w_half[nn * k + kk]);
                    }
                    let d = ((out[mm * n + nn] - refv) as f64).abs();
                    max_abs = max_abs.max(d);
                    max_rel = max_rel.max(d / (refv.abs() as f64).max(1e-6));
                }
            }
            report["correctness"] = json!({
                "shape": {"m": m, "n": n, "k": k},
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
                "pass": max_rel < 1e-3,
            });
        }

        eprintln!("stage: timed");
        // timed shapes
        for m in [40usize, 64, 128, 256] {
            let mut per_shape = Vec::new();
            let mut matmul_ms = 0f64;
            let mut wbytes_forward = 0f64;
            for (n, k, count) in ENGINE_SHAPES {
                let mut m_i = m as ClInt;
                let mut n_i = n as ClInt;
                let mut k_i = k as ClInt;
                set_arg(kernel, 0, std::mem::size_of::<ClMem>(), &out_buf as *const ClMem as *const c_void);
                set_arg(kernel, 1, std::mem::size_of::<ClMem>(), &x_buf as *const ClMem as *const c_void);
                set_arg(kernel, 2, std::mem::size_of::<ClMem>(), &w_buf as *const ClMem as *const c_void);
                set_arg(kernel, 3, std::mem::size_of::<ClMem>(), &bias_buf as *const ClMem as *const c_void);
                set_arg(kernel, 4, std::mem::size_of::<ClInt>(), &mut m_i as *mut ClInt as *const c_void);
                set_arg(kernel, 5, std::mem::size_of::<ClInt>(), &mut n_i as *mut ClInt as *const c_void);
                set_arg(kernel, 6, std::mem::size_of::<ClInt>(), &mut k_i as *mut ClInt as *const c_void);
                let g = [n.div_ceil(LX) * LX, m.div_ceil(LY) * LY];
                // Clean wall-clock loop: no per-launch sync or event queries
                // (wait_events/profiling round-trip through dxg ioctls and
                // would pollute the measurement), one finish at the end.
                let wall_start = Instant::now();
                for _ in 0..34 {
                    if enqueue_kernel(queue, kernel, 2, std::ptr::null(), g.as_ptr(), [LX, LY].as_ptr(), 0, std::ptr::null(), std::ptr::null_mut()) != 0 {
                        break;
                    }
                }
                finish(queue);
                let clean_wall_us = wall_start.elapsed().as_secs_f64() * 1e6 / 34.0;
                // Informational event-timing loop (timestamps mis-scale under
                // WSL2/dxg — kept only to expose the discrepancy).
                let mut times_ns: Vec<f64> = Vec::with_capacity(10);
                for _ in 0..10 {
                    let mut ev: ClEvent = std::ptr::null_mut();
                    if enqueue_kernel(queue, kernel, 2, std::ptr::null(), g.as_ptr(), [LX, LY].as_ptr(), 0, std::ptr::null(), &mut ev) != 0 {
                        break;
                    }
                    wait_events(1, &ev);
                    let mut t0: u64 = 0;
                    let mut t1: u64 = 0;
                    profiling(ev, CL_PROFILING_COMMAND_START, 8, &mut t0 as *mut u64 as *mut c_void, &mut got);
                    profiling(ev, CL_PROFILING_COMMAND_END, 8, &mut t1 as *mut u64 as *mut c_void, &mut got);
                    release_event(ev);
                    times_ns.push((t1.saturating_sub(t0)) as f64);
                }
                times_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let event_min_us = times_ns.first().copied().unwrap_or(0.0) / 1e3;
                let event_p50_us = times_ns[times_ns.len() / 2] / 1e3;
                let flops = 2.0 * (m * n * k) as f64;
                per_shape.push(json!({
                    "n": n, "k": k, "count": count,
                    "clean_wall_us": clean_wall_us,
                    "tflops_clean_wall": flops / (clean_wall_us * 1e3) / 1e3,
                    "event_min_us": event_min_us,
                    "event_p50_us": event_p50_us,
                }));
                matmul_ms += clean_wall_us / 1e3 * count as f64;
                wbytes_forward += (n * k * 2) as f64 * count as f64;
            }
            let total_flops = 2.0 * (m as f64) * (ENGINE_SHAPES.iter().map(|(n, k, c)| (n * k * c) as f64).sum::<f64>());
            let projected = matmul_ms + 5.0; // + attention/norm/elementwise/launch estimate
            let ratio = CPU_BASELINE_MS / projected;
            results.push(json!({
                "m_tokens": m,
                "forward_gflop": total_flops / 1e9,
                "weight_bytes_mb": wbytes_forward / 1e6,
                "matmul_ms_clean_wall": matmul_ms,
                "projected_forward_ms": projected,
                "cpu_baseline_ms": CPU_BASELINE_MS,
                "speedup_vs_cpu": ratio,
                "meets_2x": ratio >= WIN_RATIO,
                "shapes": per_shape,
            }));
        }
        report["results"] = json!(results);
        report["notes"] = json!([
            "projection adds a flat 5 ms for attention/norms/elementwise/launch overhead (CUDA sidecar measured ~3 ms at ~300 launches; OpenCL per-launch cost is comparable or higher)",
            "event profiling timestamps mis-scale under WSL2/dxg — clean wall-clock loops are the authoritative numbers; event_* fields are informational only",
            "win condition: projected_forward_ms <= 84.6 ms (2x the 169.2 ms 16-thread CPU baseline on this host)",
            "iterating the same matmul weights back-to-back as measured here is a best case for DDR3-style caches; a real forward streams 842 MB of weights between layers, so treat results as an upper bound on throughput",
        ]);

        release(out_buf);
        release(bias_buf);
        release(w_buf);
        release(x_buf);
        release_kernel(kernel);
        release_program(prog);
        release_queue(queue);
        release_context(ctx);
    }

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn cstr_lossy(buf: &[u8], got: usize) -> String {
    let n = got.min(buf.len());
    let end = buf[..n].iter().position(|&b| b == 0).unwrap_or(n);
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_constants() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(-0.25), 0xB400);
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7C00);
    }

    #[test]
    fn f16_roundtrip_identity() {
        // normal-range f32 values survive f32 -> f16 -> f32 exactly to half
        // precision: f32(f16(x)) must re-encode to the same f16.
        let mut state = 0x12345678u64;
        for _ in 0..10_000 {
            let mut v = vec![0f32; 1];
            fill_random(&mut v, &mut state);
            let h = f32_to_f16_bits(v[0]);
            let back = f16_bits_to_f32(h);
            assert_eq!(f32_to_f16_bits(back), h);
        }
    }

    #[test]
    fn f16_decode_values() {
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert_eq!(f16_bits_to_f32(0x8000), -0.0);
        // 0x1580: exp 5 -> 2^-10, mantissa 1.375
        assert_eq!(f16_bits_to_f32(0x1580), 1.375 * 2f32.powi(-10));
    }
}
