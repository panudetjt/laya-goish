//! iGPU probe (tickets 03/06): enumerate every compute device reachable from
//! this OS across all three candidate runtimes — Vulkan, Level Zero, OpenCL —
//! plus the DRM node state underneath them, and emit one machine-readable
//! verdict: is there a real integrated GPU here, through which runtime, and
//! which kernel tier should the engine select?
//!
//! Ticket 06's checklist requires BOTH `igpu-probe` (Vulkan) and a
//! oneAPI/Level Zero enumeration to list the real device (no llvmpipe/CPU
//! fallback) before any kernel work; the runtime choice itself is then made
//! by a measured F16-storage matmul microbench on that device — never here.
//! This tool answers "is a candidate host real?", not "which iGPU runtime
//! wins?".
//!
//! Loader-only, like the Vulkan path: every runtime library (libvulkan.so.1,
//! libze_loader.so.1, libOpenCL.so.1) is dlopened at run time and reported as
//! `not_installed` when absent. No Vulkan/oneAPI SDK is needed to build.
use ash::vk;
use serde_json::{json, Map, Value};
use std::ffi::{c_void, CStr};

// ---------------------------------------------------------------------------
// Pure logic (unit-tested below): fallback detection, DRM classification,
// ICD parsing, tier verdict.
// ---------------------------------------------------------------------------

/// Software-rasterizer / CPU-fallback device names seen in the wild.
const CPU_FALLBACK_MARKERS: [&str; 4] = ["llvmpipe", "lavapipe", "swiftshader", "softpipe"];

fn is_cpu_fallback(name: &str, dev_type: &str) -> bool {
    if dev_type == "CPU" || dev_type == "OTHER" {
        return true;
    }
    let lower = name.to_ascii_lowercase();
    CPU_FALLBACK_MARKERS.iter().any(|m| lower.contains(m))
}

fn osrelease_is_wsl2(osrelease: &str) -> bool {
    osrelease.to_ascii_lowercase().contains("microsoft")
}

/// Classify one /sys/class/drm/cardN/device node from its raw sysfs text.
/// Returns (bus, driver, virtual). PCI devices expose a `vendor` file;
/// platform devices (vgem) only a MODALIAS in uevent.
fn classify_drm_device(
    vendor_file: Option<&str>,
    uevent: &str,
    driver_link: Option<&str>,
) -> (&'static str, Option<String>, bool) {
    if let Some(v) = vendor_file {
        // vgem has no PCI vendor file; any parseable PCI vendor is real hardware.
        if u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).is_ok() {
            let driver = driver_link.map(str::to_string);
            return ("pci", driver, false);
        }
    }
    let lower = uevent.to_ascii_lowercase();
    let virtual_dev = lower.contains("vgem") || lower.contains("dxg");
    let driver = driver_link
        .map(str::to_string)
        .or_else(|| lower.split("platform:").nth(1).map(|s| s.trim().to_string()));
    ("platform", driver, virtual_dev)
}

fn pci_vendor_name(vendor: u32) -> &'static str {
    match vendor {
        0x8086 => "Intel",
        0x10DE => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        _ => "unknown",
    }
}

/// Parse a Vulkan ICD manifest (subset of JSON we care about).
fn parse_icd_json(text: &str) -> Option<(String, String)> {
    let v: Value = serde_json::from_str(text).ok()?;
    let lib = v
        .pointer("/ICD/library_path")?
        .as_str()?
        .to_string();
    let api = v
        .pointer("/ICD/api_version")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_string();
    Some((lib, api))
}

/// Parse an OpenCL vendor ICD file: one .so path per line, # comments.
fn parse_icd_text(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// The kernel-layer selection rule (ticket 06, mirrors 03/04 decisions):
/// NVIDIA visible anywhere -> CUDA tier (measured winner, ticket 03);
/// else a real GPU through any runtime -> iGPU tier, listing every runtime
/// that sees the device (Vulkan-vs-oneAPI pick is the on-device microbench's
/// job); else CPU tier.
fn recommend(nvidia_seen: bool, igpu_runtimes: &[&'static str]) -> (&'static str, Vec<String>) {
    let mut notes = Vec::new();
    if nvidia_seen {
        notes.push(
            "NVIDIA GPU visible: select the CUDA tier (ticket 03 measured choice). \
             iGPU tier is not the primary path on this host."
                .to_string(),
        );
        return ("cuda", notes);
    }
    if igpu_runtimes.is_empty() {
        notes.push(
            "No real GPU device through any runtime: CPU engine remains the only tier. \
             This host is not a candidate for iGPU kernel work."
                .to_string(),
        );
        return ("cpu", notes);
    }
    notes.push(format!(
        "Real GPU visible via: {}. Pick Vulkan vs oneAPI by the measured F16-storage \
         matmul microbench on this device before writing kernels (ticket 06).",
        igpu_runtimes.join(", ")
    ));
    ("igpu", notes)
}

/// Per-runtime verdict inputs, computed at enumeration time so the verdict
/// never navigates JSON. `saw_nvidia` / `saw_real_device` / `saw_cpu_fallback`
/// are per device seen through that runtime.
#[derive(Default, Clone)]
struct RuntimeSummary {
    available: bool,
    saw_nvidia: bool,
    saw_real_device: bool,
    saw_cpu_fallback: bool,
}

/// OpenCL vendor strings are free-form ("NVIDIA Corporation", "Mesa/X.Y");
/// containment, not equality, is the correct NVIDIA test.
fn opencl_vendor_is_nvidia(vendor: &str) -> bool {
    vendor.to_ascii_lowercase().contains("nvidia")
}

// ---------------------------------------------------------------------------
// dlopen plumbing shared by the Level Zero and OpenCL backends.
// ---------------------------------------------------------------------------

const RTLD_NOW: i32 = 2;

struct Lib(*mut c_void);

impl Lib {
    fn open(names: &[&CStr]) -> Result<Lib, String> {
        let mut last_err = String::new();
        for name in names {
            unsafe {
                // Clear any stale error state first.
                dlerror();
                let h = dlopen(name.as_ptr(), RTLD_NOW);
                if !h.is_null() {
                    return Ok(Lib(h));
                }
                let e = dlerror();
                last_err = if e.is_null() {
                    format!("{:?}: dlopen failed", name.to_string_lossy())
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
// Level Zero backend (oneAPI / Intel Compute Runtime).
// ABI per ze_api.h v1.x: stype u32, pad, pNext ptr, then fields in order.
// ---------------------------------------------------------------------------

const ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES: u32 = 0x1;
const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x3;
const ZE_DEVICE_TYPE_GPU: i32 = 1;
const ZE_DEVICE_PROPERTY_FLAG_INTEGRATED: u32 = 1 << 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct ZeDriverProperties {
    stype: u32,
    p_next: *mut c_void,
    uuid: [u8; 16],
    driver_version: u32,
}

impl ZeDriverProperties {
    fn zeroed() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ZeDeviceProperties {
    stype: u32,
    p_next: *mut c_void,
    dtype: i32,
    vendor_id: u32,
    device_id: u32,
    flags: u32,
    subdevice_id: u32,
    core_clock_rate: u32,
    max_mem_alloc_size: u64,
    max_hardware_contexts: u32,
    max_command_queue_priority: u32,
    num_threads_per_eu: u32,
    physical_eu_simd_width: u32,
    num_eus_per_subslice: u32,
    num_subslices_per_slice: u32,
    num_slices: u32,
    timer_resolution: u64,
    timestamp_valid_bits: u32,
    kernel_timestamp_valid_bits: u32,
    uuid: [u8; 16],
    name: [u8; 256],
}

impl ZeDeviceProperties {
    fn zeroed() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Analytic F32 peak, the same bound ticket 06 sizes this tier with
/// (EU x SIMD-width x 2 FLOP x clock — for Xe-LP "8 ALU per EU" is the EU's
/// 8-wide FMA pipe). Threads per EU hide latency; they do not add FLOPS.
/// None when the device hides its topology.
fn analytic_f32_tflops(p: &ZeDeviceProperties) -> Option<f64> {
    let eus = p.num_slices as u64 * p.num_subslices_per_slice as u64 * p.num_eus_per_subslice as u64;
    if eus == 0 || p.physical_eu_simd_width == 0 || p.core_clock_rate == 0 {
        return None;
    }
    Some(eus as f64 * p.physical_eu_simd_width as f64 * 2.0 * (p.core_clock_rate as f64 * 1e6) / 1e12)
}

fn ze_name(name: &[u8; 256]) -> String {
    let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
    String::from_utf8_lossy(&name[..end]).into_owned()
}

fn probe_level_zero() -> (Value, RuntimeSummary) {
    let mut summary = RuntimeSummary::default();
    let lib = match Lib::open(&[c"libze_loader.so.1", c"libze_loader.so"]) {
        Ok(l) => l,
        Err(e) => {
            return (
                json!({"available": false, "loader": "not_installed", "error": e}),
                summary,
            )
        }
    };
    unsafe {
        let sym_err = |e: String| (json!({"available": false, "error": e}), summary.clone());
        let ze_init: unsafe extern "C" fn(u32) -> i32 = match lib.sym(c"zeInit") {
            Ok(f) => f,
            Err(e) => return sym_err(e),
        };
        let ze_driver_get: unsafe extern "C" fn(*mut u32, *mut *mut c_void) -> i32 =
            match lib.sym(c"zeDriverGet") {
                Ok(f) => f,
                Err(e) => return sym_err(e),
            };
        let ze_device_get: unsafe extern "C" fn(*mut c_void, *mut u32, *mut *mut c_void) -> i32 =
            match lib.sym(c"zeDeviceGet") {
                Ok(f) => f,
                Err(e) => return sym_err(e),
            };
        let ze_driver_props: unsafe extern "C" fn(*mut c_void, *mut ZeDriverProperties) -> i32 =
            match lib.sym(c"zeDriverGetProperties") {
                Ok(f) => f,
                Err(e) => return sym_err(e),
            };
        let ze_device_props: unsafe extern "C" fn(*mut c_void, *mut ZeDeviceProperties) -> i32 =
            match lib.sym(c"zeDeviceGetProperties") {
                Ok(f) => f,
                Err(e) => return sym_err(e),
            };

        let r = ze_init(0);
        if r != 0 {
            return (
                json!({"available": true, "loader": "loaded", "error": format!("zeInit failed: 0x{r:x}")}),
                summary,
            );
        }
        summary.available = true;

        let mut n_drivers: u32 = 0;
        if ze_driver_get(&mut n_drivers, std::ptr::null_mut()) != 0 {
            return (
                json!({"available": true, "loader": "loaded", "error": "zeDriverGet(count) failed"}),
                summary,
            );
        }
        let mut drivers = vec![std::ptr::null_mut::<c_void>(); n_drivers as usize];
        if n_drivers > 0 && ze_driver_get(&mut n_drivers, drivers.as_mut_ptr()) != 0 {
            return (
                json!({"available": true, "loader": "loaded", "error": "zeDriverGet failed"}),
                summary,
            );
        }

        let mut devices_out = Vec::new();
        for d in drivers.iter().take(n_drivers as usize) {
            let mut dp = ZeDriverProperties::zeroed();
            dp.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
            let mut driver_version = json!(null);
            if ze_driver_props(*d, &mut dp) == 0 {
                driver_version = json!(dp.driver_version);
            }
            let mut n_dev: u32 = 0;
            if ze_device_get(*d, &mut n_dev, std::ptr::null_mut()) != 0 {
                continue;
            }
            let mut devs = vec![std::ptr::null_mut::<c_void>(); n_dev as usize];
            if n_dev > 0 && ze_device_get(*d, &mut n_dev, devs.as_mut_ptr()) != 0 {
                continue;
            }
            for dev in devs.iter().take(n_dev as usize) {
                let mut p = ZeDeviceProperties::zeroed();
                p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                if ze_device_props(*dev, &mut p) != 0 {
                    continue;
                }
                let dtype = match p.dtype {
                    ZE_DEVICE_TYPE_GPU => "GPU",
                    2 => "CPU",
                    3 => "FPGA",
                    4 => "MCA",
                    5 => "VPU",
                    _ => "OTHER",
                };
                let real = p.dtype == ZE_DEVICE_TYPE_GPU;
                if real {
                    summary.saw_real_device = true;
                    if p.vendor_id == 0x10DE {
                        summary.saw_nvidia = true;
                    }
                } else if p.dtype == 2 {
                    summary.saw_cpu_fallback = true;
                }
                devices_out.push(json!({
                    "name": ze_name(&p.name),
                    "type": dtype,
                    "vendor_id": format!("0x{:04x}", p.vendor_id),
                    "vendor": pci_vendor_name(p.vendor_id),
                    "device_id": format!("0x{:04x}", p.device_id),
                    "integrated": p.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED != 0,
                    "core_clock_mhz": p.core_clock_rate,
                    "eu_count": p.num_slices as u64 * p.num_subslices_per_slice as u64 * p.num_eus_per_subslice as u64,
                    "threads_per_eu": p.num_threads_per_eu,
                    "eu_simd_width": p.physical_eu_simd_width,
                    "analytic_f32_tflops": analytic_f32_tflops(&p),
                    "driver_version": driver_version.clone(),
                    "is_real_device": real,
                }));
            }
        }
        (
            json!({
                "available": true,
                "loader": "loaded",
                "drivers": n_drivers,
                "devices": devices_out,
            }),
            summary,
        )
    }
}

// ---------------------------------------------------------------------------
// OpenCL backend (cl_khr_fp16 via Intel Compute Runtime or others).
// ---------------------------------------------------------------------------

const CL_PLATFORM_NAME: u32 = 0x0902;
const CL_PLATFORM_VERSION: u32 = 0x0903;
const CL_DEVICE_TYPE: u32 = 0x1000;
const CL_DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
const CL_DEVICE_MAX_CLOCK_FREQUENCY: u32 = 0x100C;
const CL_DEVICE_AVAILABLE: u32 = 0x1027;
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_DEVICE_VENDOR: u32 = 0x102C;
const CL_DRIVER_VERSION: u32 = 0x102D;
const CL_DEVICE_VERSION: u32 = 0x102F;
const CL_DEVICE_HALF_FP_CONFIG: u32 = 0x1033;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 1;
const CL_DEVICE_TYPE_CPU: u64 = 1 << 2;
const CL_DEVICE_TYPE_ALL: u64 = u32::MAX as u64;

unsafe fn read_cl_str(
    f: unsafe extern "C" fn(*mut c_void, u32, usize, *mut c_void, *mut usize) -> i32,
    obj: *mut c_void,
    param: u32,
) -> String {
    let mut buf = [0u8; 512];
    let mut got: usize = 0;
    if f(obj, param, buf.len(), buf.as_mut_ptr() as *mut c_void, &mut got) == 0 && got > 0 {
        let end = got.min(buf.len() - 1);
        buf[end] = 0;
        String::from_utf8_lossy(&buf[..end]).into_owned()
    } else {
        String::new()
    }
}

unsafe fn read_cl_scalar(
    f: unsafe extern "C" fn(*mut c_void, u32, usize, *mut c_void, *mut usize) -> i32,
    obj: *mut c_void,
    param: u32,
    bytes: usize,
) -> u64 {
    let mut v: u64 = 0;
    let mut got: usize = 0;
    if f(obj, param, bytes, &mut v as *mut u64 as *mut c_void, &mut got) != 0 {
        return 0;
    }
    v
}

fn probe_opencl() -> (Value, RuntimeSummary) {
    let mut summary = RuntimeSummary::default();
    let lib = match Lib::open(&[c"libOpenCL.so.1", c"libOpenCL.so"]) {
        Ok(l) => l,
        Err(e) => {
            return (
                json!({"available": false, "loader": "not_installed", "error": e}),
                summary,
            )
        }
    };
    summary.available = true;
    unsafe {
        let sym_err = |e: String| (json!({"available": false, "error": e}), summary.clone());
        let get_platform_ids: unsafe extern "C" fn(u32, *mut *mut c_void, *mut u32) -> i32 =
            match lib.sym(c"clGetPlatformIDs") {
                Ok(f) => f,
                Err(e) => return sym_err(e),
            };
        let get_info: unsafe extern "C" fn(*mut c_void, u32, usize, *mut c_void, *mut usize) -> i32 =
            match lib.sym(c"clGetDeviceInfo") {
                Ok(f) => f,
                Err(e) => return sym_err(e),
            };
        let get_platform_info: unsafe extern "C" fn(
            *mut c_void,
            u32,
            usize,
            *mut c_void,
            *mut usize,
        ) -> i32 = match lib.sym(c"clGetPlatformInfo") {
            Ok(f) => f,
            Err(e) => return sym_err(e),
        };
        let get_device_ids: unsafe extern "C" fn(
            *mut c_void,
            u64,
            u32,
            *mut *mut c_void,
            *mut u32,
        ) -> i32 = match lib.sym(c"clGetDeviceIDs") {
            Ok(f) => f,
            Err(e) => return sym_err(e),
        };

        let mut n_platforms: u32 = 0;
        let r = get_platform_ids(0, std::ptr::null_mut(), &mut n_platforms);
        if r != 0 || n_platforms == 0 {
            return (
                json!({"available": true, "loader": "loaded", "platforms": [],
                      "error": if r != 0 { format!("clGetPlatformIDs: 0x{r:x}") } else { "no platforms".to_string() }}),
                summary,
            );
        }
        let mut platforms = vec![std::ptr::null_mut::<c_void>(); n_platforms as usize];
        if get_platform_ids(n_platforms, platforms.as_mut_ptr(), std::ptr::null_mut()) != 0 {
            return (
                json!({"available": true, "loader": "loaded", "error": "clGetPlatformIDs failed"}),
                summary,
            );
        }

        let mut platforms_out = Vec::new();
        for p in platforms {
            let name = read_cl_str(get_platform_info, p, CL_PLATFORM_NAME);
            let version = read_cl_str(get_platform_info, p, CL_PLATFORM_VERSION);
            let mut devices_out = Vec::new();
            let mut n_dev: u32 = 0;
            if get_device_ids(p, CL_DEVICE_TYPE_ALL, 0, std::ptr::null_mut(), &mut n_dev) == 0
                && n_dev > 0
            {
                let mut devs = vec![std::ptr::null_mut::<c_void>(); n_dev as usize];
                if get_device_ids(p, CL_DEVICE_TYPE_ALL, n_dev, devs.as_mut_ptr(), std::ptr::null_mut()) == 0
                {
                    for d in devs {
                        let type_bits = read_cl_scalar(get_info, d, CL_DEVICE_TYPE, 8);
                        let dtype = if type_bits & CL_DEVICE_TYPE_GPU != 0 {
                            "GPU"
                        } else if type_bits & CL_DEVICE_TYPE_CPU != 0 {
                            "CPU"
                        } else {
                            "OTHER"
                        };
                        let real = dtype == "GPU";
                        let vendor = read_cl_str(get_info, d, CL_DEVICE_VENDOR);
                        if real {
                            summary.saw_real_device = true;
                            if opencl_vendor_is_nvidia(&vendor) {
                                summary.saw_nvidia = true;
                            }
                        } else if dtype == "CPU" {
                            summary.saw_cpu_fallback = true;
                        }
                        devices_out.push(json!({
                            "name": read_cl_str(get_info, d, CL_DEVICE_NAME),
                            "vendor": vendor,
                            "type": dtype,
                            "available": read_cl_scalar(get_info, d, CL_DEVICE_AVAILABLE, 1) != 0,
                            "driver_version": read_cl_str(get_info, d, CL_DRIVER_VERSION),
                            "device_version": read_cl_str(get_info, d, CL_DEVICE_VERSION),
                            "compute_units": read_cl_scalar(get_info, d, CL_DEVICE_MAX_COMPUTE_UNITS, 4),
                            "max_clock_mhz": read_cl_scalar(get_info, d, CL_DEVICE_MAX_CLOCK_FREQUENCY, 4),
                            "fp16": read_cl_scalar(get_info, d, CL_DEVICE_HALF_FP_CONFIG, 8) != 0,
                            "is_real_device": real,
                        }));
                    }
                }
            }
            platforms_out.push(json!({
                "name": name,
                "version": version,
                "devices": devices_out,
            }));
        }
        (
            json!({"available": true, "loader": "loaded", "platforms": platforms_out}),
            summary,
        )
    }
}

// ---------------------------------------------------------------------------
// CUDA detection: cuDeviceGet only ever enumerates NVIDIA devices, so a
// successful cuInit + count >= 1 is the definitive `saw_nvidia` signal —
// the verdict must not conflate "no iGPU" with "no GPU" (ticket 06's rule
// picks the iGPU tier when no NVIDIA device EXISTS, not when Vulkan is blind).
// ---------------------------------------------------------------------------

const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: i32 = 13;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

fn probe_cuda() -> (Value, RuntimeSummary) {
    let mut summary = RuntimeSummary::default();
    let lib = match Lib::open(&[
        c"/usr/lib/wsl/lib/libcuda.so.1",
        c"libcuda.so.1",
        c"libcuda.so",
    ]) {
        Ok(l) => l,
        Err(e) => {
            return (
                json!({"available": false, "loader": "not_installed", "error": e}),
                summary,
            )
        }
    };
    summary.available = true;
    unsafe {
        let sym_err = |e: String| (json!({"available": false, "error": e}), summary.clone());
        let cu_init: unsafe extern "C" fn(u32) -> i32 = match lib.sym(c"cuInit") {
            Ok(f) => f,
            Err(e) => return sym_err(e),
        };
        let cu_device_get_count: unsafe extern "C" fn(*mut i32) -> i32 =
            match lib.sym(c"cuDeviceGetCount") {
                Ok(f) => f,
                Err(e) => return sym_err(e),
            };
        let cu_device_get_name: unsafe extern "C" fn(*mut i8, i32, i32) -> i32 =
            match lib.sym(c"cuDeviceGetName") {
                Ok(f) => f,
                Err(e) => return sym_err(e),
            };
        let cu_device_get_attr: unsafe extern "C" fn(*mut i32, i32, i32) -> i32 =
            match lib.sym(c"cuDeviceGetAttribute") {
                Ok(f) => f,
                Err(e) => return sym_err(e),
            };

        let r = cu_init(0);
        if r != 0 {
            return (
                json!({"available": true, "loader": "loaded", "error": format!("cuInit failed: 0x{r:x}")}),
                summary,
            );
        }
        let mut n: i32 = 0;
        if cu_device_get_count(&mut n) != 0 {
            return (
                json!({"available": true, "loader": "loaded", "error": "cuDeviceGetCount failed"}),
                summary,
            );
        }
        let mut devices_out = Vec::new();
        for i in 0..n {
            let mut namebuf = [0i8; 256];
            if cu_device_get_name(namebuf.as_mut_ptr(), 256, i) != 0 {
                continue;
            }
            let bytes: Vec<u8> = namebuf
                .iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as u8)
                .collect();
            let mut sm = 0;
            let mut khz = 0;
            let mut major = 0;
            let mut minor = 0;
            cu_device_get_attr(&mut sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, i);
            cu_device_get_attr(&mut khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, i);
            cu_device_get_attr(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i);
            cu_device_get_attr(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i);
            summary.saw_nvidia = true;
            summary.saw_real_device = true;
            devices_out.push(json!({
                "name": String::from_utf8_lossy(&bytes),
                "compute_capability": format!("{major}.{minor}"),
                "multiprocessors": sm,
                "clock_mhz": khz / 1000,
                "is_real_device": true,
            }));
        }
        (
            json!({"available": true, "loader": "loaded", "devices": devices_out}),
            summary,
        )
    }
}

// ---------------------------------------------------------------------------
// DRM node inspection: the ground truth of what hardware the kernel sees.
// ---------------------------------------------------------------------------

fn read_small(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn drm_nodes() -> Vec<Value> {
    let dir = match std::fs::read_dir("/dev/dri") {
        Ok(d) => d,
        Err(e) => return vec![json!({"error": format!("cannot read /dev/dri: {e}")})],
    };
    let mut out = Vec::new();
    for entry in dir.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_card = name.starts_with("card");
        let is_render = name.starts_with("renderD");
        if !is_card && !is_render {
            continue;
        }
        let sys = std::path::Path::new("/sys/class/drm").join(&name).join("device");
        let vendor = read_small(&sys.join("vendor"));
        let device_id = read_small(&sys.join("device"));
        let uevent = read_small(&sys.join("uevent")).unwrap_or_default();
        let driver_link = std::fs::read_link(sys.join("driver"))
            .ok()
            .map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        let (bus, driver, virtual_dev) = classify_drm_device(
            vendor.as_deref(),
            &uevent,
            driver_link.as_deref(),
        );
        let mut e = json!({
            "node": name,
            "kind": if is_card { "card" } else { "render" },
            "bus": bus,
            "driver": driver,
            "virtual": virtual_dev,
        });
        let obj = e.as_object_mut().unwrap();
        let vendor_id = vendor
            .as_deref()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
        if let Some(vn) = vendor_id {
            obj.insert("pci_vendor".into(), json!(format!("0x{vn:04x}")));
            obj.insert("pci_vendor_name".into(), json!(pci_vendor_name(vn)));
            if let Some(dv) = device_id {
                obj.insert("pci_device".into(), json!(dv));
            }
        }
        out.push(e);
    }
    out
}

// ---------------------------------------------------------------------------
// Vulkan backend (ash, loader-only — carried over from ticket 03).
// ---------------------------------------------------------------------------

fn probe_vulkan() -> (Value, RuntimeSummary) {
    let mut summary = RuntimeSummary::default();
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(e) => {
            return (
                json!({"available": false, "loader": "not_installed", "error": e.to_string()}),
                summary,
            )
        }
    };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"laya-igpu-probe")
        .api_version(vk::API_VERSION_1_1);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(i) => i,
        Err(e) => {
            return (
                json!({"available": true, "loader": "loaded", "error": format!("vkCreateInstance failed: {e}")}),
                summary,
            )
        }
    };
    let devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(d) => d,
        Err(e) => {
            return (
                json!({"available": true, "loader": "loaded", "error": format!("vkEnumeratePhysicalDevices failed: {e}")}),
                summary,
            )
        }
    };
    let mut out = Vec::new();
    for dev in devices.iter().copied() {
        let props = unsafe { instance.get_physical_device_properties(dev) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy();
        let dtype = format!("{:?}", props.device_type);
        let real = !is_cpu_fallback(&name, &dtype)
            && matches!(
                props.device_type,
                vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::DISCRETE_GPU
            );
        summary.available = true;
        if real {
            summary.saw_real_device = true;
            if props.vendor_id == 0x10DE {
                summary.saw_nvidia = true;
            }
        } else {
            summary.saw_cpu_fallback = true;
        }
        out.push(json!({
            "name": name,
            "type": dtype,
            "api_version": format!(
                "{}.{}.{}",
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version)
            ),
            "vendor_id": format!("0x{:04x}", props.vendor_id),
            "vendor": pci_vendor_name(props.vendor_id),
            "device_id": format!("0x{:04x}", props.device_id),
            "compute_queue_families": unsafe { instance.get_physical_device_queue_family_properties(dev) }
                .iter()
                .filter(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .count(),
            "is_real_device": real,
        }));
    }
    let _ = unsafe { instance.destroy_instance(None) };
    (
        json!({"available": true, "loader": "loaded", "devices": out}),
        summary,
    )
}

/// Installed Vulkan ICD manifests (driver-stack documentation for the ticket).
fn vulkan_icds() -> Vec<Value> {
    let dir = match std::fs::read_dir("/usr/share/vulkan/icd.d") {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for entry in dir.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".json") {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(entry.path()) {
            if let Some((lib, api)) = parse_icd_json(&text) {
                out.push(json!({"manifest": name, "library_path": lib, "api_version": api}));
            }
        }
    }
    out
}

/// Installed OpenCL vendor ICDs.
fn opencl_icds() -> Vec<String> {
    let dir = match std::fs::read_dir("/etc/OpenCL/vendors") {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for entry in dir.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".icd") {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(entry.path()) {
            out.extend(parse_icd_text(&text));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Main: compose the report and the tier verdict.
// ---------------------------------------------------------------------------

fn main() {
    let osrelease = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let wsl2 = osrelease_is_wsl2(&osrelease);

    let (vulkan, vk_s) = probe_vulkan();
    let (level_zero, l0_s) = probe_level_zero();
    let (opencl, ocl_s) = probe_opencl();
    let (cuda, cuda_s) = probe_cuda();
    let drm = drm_nodes();

    let nvidia_seen =
        cuda_s.saw_nvidia || vk_s.saw_nvidia || l0_s.saw_nvidia || ocl_s.saw_nvidia;
    let mut igpu_runtimes: Vec<&'static str> = Vec::new();
    if vk_s.saw_real_device {
        igpu_runtimes.push("vulkan");
    }
    if l0_s.saw_real_device {
        igpu_runtimes.push("level-zero");
    }
    if ocl_s.saw_real_device {
        igpu_runtimes.push("opencl");
    }
    let any_cpu_fallback =
        vk_s.saw_cpu_fallback || l0_s.saw_cpu_fallback || ocl_s.saw_cpu_fallback;

    let (tier, notes) = recommend(nvidia_seen, &igpu_runtimes);
    let mut all_notes: Vec<Value> = notes.into_iter().map(Value::String).collect();
    if wsl2 {
        all_notes.push(json!(
            "WSL2 host: per ticket 06 constraint 1 and the ticket 03 measured probe, \
             the iGPU is not paravirtualized into WSL2 (/dev/dxg is NVIDIA-CUDA only; \
             Mesa ICDs target bare-metal DRM). A llvmpipe/vgem-only result here is \
             expected and is NOT a candidate host — re-run on bare-metal Linux or \
             Windows native."
        ));
    }

    fn merge(base: Value, extra: &[(&str, Value)]) -> Value {
        let mut obj = base.as_object().cloned().unwrap_or_else(Map::new);
        for (k, v) in extra {
            obj.insert((*k).to_string(), v.clone());
        }
        Value::Object(obj)
    }

    let report = json!({
        "probe_version": 2,
        "host": {
            "osrelease": osrelease,
            "wsl2": wsl2,
        },
        "vulkan": merge(vulkan, &[("icds", json!(vulkan_icds()))]),
        "level_zero": level_zero,
        "opencl": merge(opencl, &[("icds", json!(opencl_icds()))]),
        "cuda": cuda,
        "drm_nodes": drm,
        "cpu_fallback_seen": any_cpu_fallback,
        "recommended_tier": tier,
        "igpu_runtime_candidates": igpu_runtimes,
        "notes": all_notes,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

// ---------------------------------------------------------------------------
// Tests for the pure logic (no hardware required).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_fallback_names_detected() {
        assert!(is_cpu_fallback(
            "llvmpipe (LLVM 18.1.3, 256 bits)",
            "CPU"
        ));
        assert!(is_cpu_fallback("lavapipe (LLVM 18.1.3)", "INTEGRATED_GPU"));
        assert!(is_cpu_fallback("SwiftShader device", "CPU"));
        assert!(!is_cpu_fallback("Intel(R) UHD Graphics 730", "INTEGRATED_GPU"));
        assert!(!is_cpu_fallback("NVIDIA GeForce RTX 5060 Ti", "DISCRETE_GPU"));
        // Type alone trips it even with a plausible name.
        assert!(is_cpu_fallback("weird-name", "CPU"));
    }

    #[test]
    fn wsl2_detection() {
        assert!(osrelease_is_wsl2("5.15.167.4-microsoft-standard-WSL2"));
        assert!(!osrelease_is_wsl2("6.8.0-40-generic"));
    }

    #[test]
    fn vgem_classified_virtual() {
        let (bus, driver, virtual_dev) =
            classify_drm_device(None, "MODALIAS=platform:vgem\n", Some("vgem"));
        assert_eq!(bus, "platform");
        assert_eq!(driver.as_deref(), Some("vgem"));
        assert!(virtual_dev);
    }

    #[test]
    fn pci_intel_classified_real() {
        let (bus, driver, virtual_dev) =
            classify_drm_device(Some("0x8086"), "", Some("xe"));
        assert_eq!(bus, "pci");
        assert_eq!(driver.as_deref(), Some("xe"));
        assert!(!virtual_dev);
    }

    #[test]
    fn dxg_paravirt_flagged() {
        let (_, _, virtual_dev) = classify_drm_device(None, "MODALIAS=platform:dxg\n", Some("dxg"));
        assert!(virtual_dev);
    }

    #[test]
    fn vendor_names() {
        assert_eq!(pci_vendor_name(0x8086), "Intel");
        assert_eq!(pci_vendor_name(0x10DE), "NVIDIA");
        assert_eq!(pci_vendor_name(0x1002), "AMD");
        assert_eq!(pci_vendor_name(0x1234), "unknown");
    }

    #[test]
    fn icd_json_parsed() {
        let text = r#"{"ICD":{"library_path":"libGLX_mesa.so.0","api_version":"1.4.318"}}"#;
        let (lib, api) = parse_icd_json(text).unwrap();
        assert_eq!(lib, "libGLX_mesa.so.0");
        assert_eq!(api, "1.4.318");
        assert!(parse_icd_json("not json").is_none());
    }

    #[test]
    fn icd_text_parsed() {
        let text = "# comment\nlibintel_opencl.so\n\n  libpocl.so  \n";
        assert_eq!(parse_icd_text(text), vec!["libintel_opencl.so", "libpocl.so"]);
    }

    #[test]
    fn verdict_nvidia_beats_everything() {
        let (tier, _) = recommend(true, &["vulkan"]);
        assert_eq!(tier, "cuda");
    }

    #[test]
    fn verdict_igpu_lists_runtimes() {
        let (tier, notes) = recommend(false, &["vulkan", "level-zero"]);
        assert_eq!(tier, "igpu");
        assert!(notes[0].contains("vulkan, level-zero"));
        assert!(notes[0].contains("microbench"));
    }

    #[test]
    fn verdict_cpu_when_only_fallback() {
        let (tier, notes) = recommend(false, &[]);
        assert_eq!(tier, "cpu");
        assert!(notes[0].contains("not a candidate"));
    }

    #[test]
    fn analytic_tflops_matches_ticket_bound() {
        // UHD 730 shape: 24 EU x 8 threads x 8 lanes x 2 x 1.9 GHz ≈ 0.7 TF.
        let mut p = ZeDeviceProperties::zeroed();
        p.num_slices = 6;
        p.num_subslices_per_slice = 4;
        p.num_eus_per_subslice = 1;
        p.num_threads_per_eu = 8;
        p.physical_eu_simd_width = 8;
        p.core_clock_rate = 1900;
        let tf = analytic_f32_tflops(&p).unwrap();
        assert!((tf - 0.7296).abs() < 1e-6, "got {tf}");
        let mut empty = p;
        empty.num_slices = 0;
        assert!(analytic_f32_tflops(&empty).is_none());
    }

    #[test]
    fn ze_name_reads_cstring() {
        let mut buf = [0u8; 256];
        buf[..7].copy_from_slice(b"UHD 730");
        assert_eq!(ze_name(&buf), "UHD 730");
    }

    #[test]
    fn opencl_nvidia_vendor_matching() {
        assert!(opencl_vendor_is_nvidia("NVIDIA Corporation"));
        assert!(opencl_vendor_is_nvidia("nvidia corp"));
        assert!(!opencl_vendor_is_nvidia("Mesa/X.org"));
        assert!(!opencl_vendor_is_nvidia("Intel Corporation"));
        assert!(!opencl_vendor_is_nvidia(""));
    }

    #[test]
    fn ze_struct_sizes_match_abi() {
        // Offsets per ze_api.h v1.x on x86-64; guards against silent layout drift.
        assert_eq!(std::mem::size_of::<ZeDriverProperties>(), 40);
        assert_eq!(std::mem::size_of::<ZeDeviceProperties>(), 368);
    }
}
