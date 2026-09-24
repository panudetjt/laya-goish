//! Minimal CUDA driver API wrapper (no toolkit dependency at build time on
//! the Rust side; the only external requirement is libcuda.so from the
//! driver — on WSL2 that is /usr/lib/wsl/lib/libcuda.so, wired up by build.rs).
//!
//! WSL2 notes (measured on driver 610.57 / CUDA 13.3 UMD, RTX 5060 Ti):
//! - Use cuCtxCreate. The primary-context retain path half-works
//!   (cuDevicePrimaryCtxRetain succeeds but cuMemGetInfo on it returns
//!   CUDA_ERROR_INVALID_CONTEXT).
//! - cuMemGetInfo reports garbage after allocations; track bytes ourselves.
//! - The practical allocation budget observed is ~3.4 GiB even though the
//!   card has 16 GiB (WDDM commit view through /dev/dxg). 2.3 GiB of model
//!   weights plus ~40 MiB of activations fits comfortably.
use std::ffi::{c_char, c_int, c_void, CStr, CString};

pub type DevPtr = u64;

extern "C" {
    fn cuInit(flags: u32) -> c_int;
    fn cuDriverGetVersion(version: *mut c_int) -> c_int;
    fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> c_int;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> c_int;
    fn cuDeviceTotalMem(bytes: *mut usize, device: c_int) -> c_int;
    fn cuCtxCreate(ctx: *mut *mut c_void, flags: u32, device: c_int) -> c_int;
    fn cuCtxSetCurrent(ctx: *mut c_void) -> c_int;
    fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> c_int;
    fn cuModuleGetFunction(f: *mut *mut c_void, module: *mut c_void, name: *const c_char) -> c_int;
    fn cuMemAlloc(ptr: *mut DevPtr, bytes: usize) -> c_int;
    fn cuMemcpyHtoD(dst: DevPtr, src: *const c_void, bytes: usize) -> c_int;
    fn cuMemcpyDtoH(dst: *mut c_void, src: DevPtr, bytes: usize) -> c_int;
    fn cuCtxSynchronize() -> c_int;
    fn cuLaunchKernel(
        f: *mut c_void,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared: u32,
        stream: *mut c_void,
        params: *mut *mut c_void,
        extra: *mut c_void,
    ) -> c_int;
    fn cuGetErrorString(err: c_int, s: *mut *const c_char) -> c_int;
}

fn ck(rv: c_int, what: &str) -> Result<(), String> {
    if rv == 0 {
        return Ok(());
    }
    unsafe {
        let mut p: *const c_char = std::ptr::null();
        if cuGetErrorString(rv, &mut p) == 0 && !p.is_null() {
            let name = CStr::from_ptr(p).to_string_lossy().into_owned();
            return Err(format!("{what}: CUDA error {rv} ({name})"));
        }
    }
    Err(format!("{what}: CUDA error {rv}"))
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct Cuda {
    pub device_name: String,
    pub driver_version: i32,
    pub device_total: usize,
    /// Bytes currently held in cuMemAlloc allocations (cuMemGetInfo is
    /// unreliable on the WSL shim; we track our own budget).
    pub allocated: std::cell::Cell<usize>,
}

pub struct Module {
    module: *mut c_void,
}

pub struct Function {
    f: *mut c_void,
}

pub struct DevMem {
    pub ptr: DevPtr,
    pub bytes: usize,
}

impl Cuda {
    pub fn init() -> Result<Cuda, String> {
        ck(unsafe { cuInit(0) }, "cuInit")?;
        let mut version: c_int = 0;
        ck(unsafe { cuDriverGetVersion(&mut version) }, "cuDriverGetVersion")?;
        let mut device: c_int = -1;
        ck(unsafe { cuDeviceGet(&mut device, 0) }, "cuDeviceGet")?;
        let mut name = [0i8; 256];
        ck(
            unsafe { cuDeviceGetName(name.as_mut_ptr(), name.len() as c_int, device) },
            "cuDeviceGetName",
        )?;
        let device_name: String = unsafe {
            let p = name.as_ptr() as *const u8;
            let mut end = 0usize;
            while end < 255 && *p.add(end) != 0 {
                end += 1;
            }
            std::str::from_utf8(std::slice::from_raw_parts(p, end))
                .unwrap_or("?")
                .to_string()
        };
        let mut total: usize = 0;
        ck(
            unsafe { cuDeviceTotalMem(&mut total, device) },
            "cuDeviceTotalMem",
        )?;
        let mut ctx: *mut c_void = std::ptr::null_mut();
        ck(
            unsafe { cuCtxCreate(&mut ctx, 0, device) },
            "cuCtxCreate",
        )?;
        ck(unsafe { cuCtxSetCurrent(ctx) }, "cuCtxSetCurrent")?;
        // Leak the context handle deliberately: the sidecar keeps the CUDA
        // context for the process lifetime (Box::leak equivalent for the
        // raw handle).
        Ok(Cuda {
            device_name,
            driver_version: version,
            device_total: total,
            allocated: std::cell::Cell::new(0),
        })
    }

    pub fn alloc(&self, bytes: usize) -> Result<DevMem, String> {
        let mut ptr: DevPtr = 0;
        ck(unsafe { cuMemAlloc(&mut ptr, bytes) }, "cuMemAlloc")?;
        self.allocated.set(self.allocated.get() + bytes);
        Ok(DevMem { ptr, bytes })
    }

    /// Copy host bytes into a device allocation in <=256 MiB chunks.
    pub fn upload(&self, dst: &DevMem, host: &[u8]) -> Result<(), String> {
        assert_eq!(dst.bytes, host.len(), "upload size mismatch");
        const CHUNK: usize = 256 << 20;
        let mut off = 0usize;
        while off < host.len() {
            let n = CHUNK.min(host.len() - off);
            ck(
                unsafe {
                    cuMemcpyHtoD(
                        dst.ptr + off as DevPtr,
                        host[off..off + n].as_ptr() as *const c_void,
                        n,
                    )
                },
                "cuMemcpyHtoD",
            )?;
            off += n;
        }
        Ok(())
    }

    pub fn download(&self, src: &DevMem, host: &mut [u8]) -> Result<(), String> {
        assert_eq!(src.bytes, host.len(), "download size mismatch");
        ck(
            unsafe { cuMemcpyDtoH(host.as_mut_ptr() as *mut c_void, src.ptr, src.bytes) },
            "cuMemcpyDtoH",
        )
    }

    pub fn load_module(&self, image: &[u8]) -> Result<Module, String> {
        let mut module: *mut c_void = std::ptr::null_mut();
        ck(
            unsafe { cuModuleLoadData(&mut module, image.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        Ok(Module { module })
    }

    pub fn synchronize(&self) -> Result<(), String> {
        ck(unsafe { cuCtxSynchronize() }, "cuCtxSynchronize")
    }

    /// Copy host bytes to an arbitrary device address (synchronous; the
    /// legacy NULL stream keeps ordering with our NULL-stream launches).
    pub fn h2d(&self, dst: DevPtr, host: &[u8]) -> Result<(), String> {
        ck(
            unsafe { cuMemcpyHtoD(dst, host.as_ptr() as *const c_void, host.len()) },
            "cuMemcpyHtoD",
        )
    }

    /// Copy device bytes to host (synchronous, NULL stream).
    pub fn d2h(&self, src: DevPtr, host: &mut [u8]) -> Result<(), String> {
        ck(
            unsafe { cuMemcpyDtoH(host.as_mut_ptr() as *mut c_void, src, host.len()) },
            "cuMemcpyDtoH",
        )
    }

    /// Launch a kernel on the legacy NULL stream.
    pub fn launch(
        &self,
        f: &Function,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), String> {
        ck(
            unsafe {
                cuLaunchKernel(
                    f.f,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

impl Module {
    pub fn function(&self, name: &str) -> Result<Function, String> {
        let c = CString::new(name).map_err(|e| e.to_string())?;
        let mut f: *mut c_void = std::ptr::null_mut();
        ck(
            unsafe { cuModuleGetFunction(&mut f, self.module, c.as_ptr()) },
            &format!("cuModuleGetFunction({name})"),
        )?;
        Ok(Function { f })
    }
}
