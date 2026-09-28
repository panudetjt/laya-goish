// wasm32 extern "C" ABI over the engine core (wayfinder ticket 19). The Node
// harness (harness.mjs) copies GGUF bytes in via wasm_alloc_u8, calls
// wasm_load / wasm_forward, and reads logits out through wasm_out_ptr
// (f32-aligned). ABI shape proven in prototypes/wasm-gate; this crate adapts
// it to the engine's Model API. No wasm-bindgen.
//
// wasm32 has no clock: the host injects performance.now() as env.now_ms and
// all load/forward timings are deltas taken inside wasm around the real work.
use core::arch::wasm32::{i16x8_eq, i16x8_splat, v128_and, v128_any_true, v128_load};
use laya_engine::{
    json,
    models::{laya::model::Model as LayaModel, openthai::model::Model as OtModel},
    tensor,
};
use std::alloc::Layout;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

static LOAD_MS: AtomicU64 = AtomicU64::new(0);
static PARSE_MS: AtomicU64 = AtomicU64::new(0);
static BUILD_MS: AtomicU64 = AtomicU64::new(0);
static FORWARD_MS: AtomicU64 = AtomicU64::new(0);
static THREADS: AtomicU32 = AtomicU32::new(1);

// The threads artifact's harness sets this before wasm_load; the default
// (no-atomics) artifact never writes it, so its models stay 1-threaded.
fn threads() -> usize {
    THREADS.load(Ordering::Relaxed).max(1) as usize
}

// Fan-out width for the +atomics build's host-driven workers
// (engine/src/team.rs wasm_threads). No-op in the default artifact.
#[no_mangle]
#[cfg(wasm_threads)]
pub extern "C" fn wasm_set_threads(k: u32) {
    THREADS.store(k, Ordering::Relaxed);
}

static mut LAYA: Option<Box<LayaModel>> = None;
static mut OPENTHAI: Option<Box<OtModel>> = None;
static mut OUT: Vec<f32> = Vec::new();
static mut OUT2: Vec<f32> = Vec::new();
static mut LAST_ERROR: Option<String> = None;

#[link(wasm_import_module = "env")]
extern "C" {
    fn now_ms() -> f64;
}

fn clock() -> f64 {
    unsafe { now_ms() }
}

// Panic payloads have no stderr on wasm32-unknown-unknown; route them into
// LAST_ERROR so the harness can read the actual message after the trap.
fn init_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        unsafe {
            LAST_ERROR = Some(format!(
                "panic: {msg}{}",
                info.location()
                    .map(|l| format!(" at {l}"))
                    .unwrap_or_default()
            ));
        }
    }));
}

unsafe fn set_err(e: impl Into<String>) {
    LAST_ERROR = Some(e.into());
}

#[no_mangle]
pub extern "C" fn wasm_alloc_u8(n: usize) -> *mut u8 {
    let layout = Layout::from_size_align(n.max(1), 1).unwrap();
    unsafe { std::alloc::alloc(layout) }
}

#[no_mangle]
pub extern "C" fn wasm_alloc_u32(n: usize) -> *mut u32 {
    let layout = Layout::from_size_align(n.max(1) * 4, 4).unwrap();
    unsafe { std::alloc::alloc(layout) as *mut u32 }
}

#[no_mangle]
pub unsafe extern "C" fn wasm_free_u32(ptr: *mut u32, n: usize) {
    if !ptr.is_null() {
        let layout = Layout::from_size_align(n.max(1) * 4, 4).unwrap();
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

// Parses + builds the model from GGUF bytes previously copied to `ptr`
// (wasm_alloc_u8). Takes ownership of the buffer: capacity must equal the
// wasm_alloc_u8 size or the Vec's later dealloc presents a wrong layout to
// the allocator (the wasm-gate prototype's one-line lesson).
//
// model: 0 = laya, 1 = openthai. Weight residency is per file kind (F16 or
// Q8_0 blocks stay resident; the kernel dequantizes on load). Laya reads
// only metadata after build, so its mapping is released to halve live bytes;
// OpenThai gathers embedding rows from the mapping every forward.
// Returns 0 on success, 1 on error (see wasm_last_error).
#[no_mangle]
pub unsafe extern "C" fn wasm_load(model: u32, ptr: *mut u8, len: usize) -> u32 {
    init_panic_hook();
    let data = Vec::from_raw_parts(ptr, len, len);
    let t0 = clock();
    let g = match laya_engine::gguf::Gguf::from_bytes(data) {
        Ok(g) => g,
        Err(e) => {
            set_err(format!("gguf: {e}"));
            return 1;
        }
    };
    let parse_ms = clock() - t0;
    let t1 = clock();
    // One-time scan: any inf/nan half in any F16 tensor switches the kernel
    // to the full widen (tensor::set_wasm_f16_has_inf). Trained checkpoints
    // scan clean; the scan is SIMD and adds ~0.2-0.6 s to model load
    // (counted in the build stat).
    for t in g.tensors.values() {
        if t.kind != 1 {
            continue;
        }
        let bytes = &g.map.bytes()[t.offset..t.offset + t.len];
        let mut p = bytes.as_ptr();
        let end = unsafe { p.add(bytes.len() & !15) };
        let pat = unsafe { i16x8_splat(0x7C00) };
        while p != end {
            let h = unsafe { v128_load(p as *const _) };
            if unsafe { v128_any_true(i16x8_eq(v128_and(h, pat), pat)) } {
                unsafe { tensor::set_wasm_f16_has_inf() };
                break;
            }
            unsafe { p = p.add(16) };
        }
    }
    let t1 = clock();
    let result = match model {
        0 => LayaModel::new(g, threads()).map(|mut m| {
            m.g.map.release();
            LAYA = Some(Box::new(m));
        }),
        1 => OtModel::new(g, threads()).map(|m| OPENTHAI = Some(Box::new(m))),
        _ => Err("bad model id (0=laya 1=openthai)".to_string()),
    };
    let build_ms = clock() - t1;
    match result {
        Ok(()) => {
            LOAD_MS.store(((parse_ms + build_ms) * 1000.0) as u64, Ordering::Relaxed);
            PARSE_MS.store((parse_ms * 1000.0) as u64, Ordering::Relaxed);
            BUILD_MS.store((build_ms * 1000.0) as u64, Ordering::Relaxed);
            0
        }
        Err(e) => {
            set_err(e);
            1
        }
    }
}

// Runs one forward. ids: u32 token ids. aux: markers (laya — positions in
// the stacked-batch sense) or answer positions (openthai). qtype: 0 choice /
// 1 score / 2 noul (laya only). Logits land in OUT (wasm_out_ptr), laya act
// logits in OUT2 (wasm_out2_ptr); counts packed in the return (low = OUT,
// high = OUT2). Returns u64::MAX on error.
#[no_mangle]
pub unsafe extern "C" fn wasm_forward(
    model: u32,
    ids_ptr: *const u32,
    ids_len: usize,
    aux_ptr: *const u32,
    aux_len: usize,
    qtype: u32,
) -> u64 {
    init_panic_hook();
    let ids: Vec<usize> = std::slice::from_raw_parts(ids_ptr, ids_len)
        .iter()
        .map(|&v| v as usize)
        .collect();
    let aux: Vec<usize> = std::slice::from_raw_parts(aux_ptr, aux_len)
        .iter()
        .map(|&v| v as usize)
        .collect();
    let t0 = clock();
    let result = match model {
        0 => LAYA
            .as_ref()
            .ok_or("laya not loaded".to_string())
            .and_then(|m| {
                m.forward(&ids, &aux, qtype as usize)
                    .map(|(l, a)| (l, Some(a)))
            }),
        1 => OPENTHAI
            .as_ref()
            .ok_or("openthai not loaded".to_string())
            .and_then(|m| m.forward(&ids, &aux, &|| false).map(|l| (l, None))),
        _ => Err("bad model id".to_string()),
    };
    let ms = clock() - t0;
    match result {
        Ok((logits, act)) => {
            OUT = logits;
            OUT2 = act.unwrap_or_default();
            FORWARD_MS.store((ms * 1000.0) as u64, Ordering::Relaxed);
            ((OUT2.len() as u64) << 32) | OUT.len() as u64
        }
        Err(e) => {
            set_err(e);
            u64::MAX
        }
    }
}

#[no_mangle]
pub extern "C" fn wasm_out_ptr() -> *const f32 {
    unsafe { OUT.as_ptr() }
}

#[no_mangle]
pub extern "C" fn wasm_out2_ptr() -> *const f32 {
    unsafe { OUT2.as_ptr() }
}

// NUL-terminated; valid until the next error.
#[no_mangle]
pub unsafe extern "C" fn wasm_last_error() -> *const u8 {
    match LAST_ERROR.as_ref() {
        Some(e) => e.as_ptr(),
        None => b"\0".as_ptr(),
    }
}

// Drops resident models (OUT buffers kept for capacity reuse).
#[no_mangle]
pub unsafe extern "C" fn wasm_free_models(laya: u32, openthai: u32) {
    if laya != 0 {
        LAYA = None;
    }
    if openthai != 0 {
        OPENTHAI = None;
    }
}

// (which: 0=load total, 1=parse, 2=build, 3=forward) -> microseconds
#[no_mangle]
pub extern "C" fn wasm_stat_us(which: u32) -> u64 {
    match which {
        0 => LOAD_MS.load(Ordering::Relaxed),
        1 => PARSE_MS.load(Ordering::Relaxed),
        2 => BUILD_MS.load(Ordering::Relaxed),
        _ => FORWARD_MS.load(Ordering::Relaxed),
    }
}

#[no_mangle]
pub extern "C" fn wasm_stat_peak_alloc() -> usize {
    laya_engine::heap::peak_bytes()
}

#[no_mangle]
pub extern "C" fn wasm_stat_live_alloc() -> usize {
    laya_engine::heap::live_bytes()
}

#[no_mangle]
pub extern "C" fn wasm_stat_alloc_calls() -> usize {
    laya_engine::heap::alloc_calls()
}

// simd128 on/off (default on) — the A/B switch for the scalar fallback.
#[no_mangle]
pub extern "C" fn wasm_set_simd128(on: u32) {
    tensor::set_wasm_simd128(on != 0);
}

// decision::decide_with_cancel's temperature rule (bucket key + floor),
// evaluated against the resident model's metadata so the JS harness can
// rebuild the engine's temperatured probabilities.
#[no_mangle]
pub unsafe extern "C" fn wasm_laya_temp(kind: u32, k: u32) -> f32 {
    let Some(m) = LAYA.as_ref() else {
        return 1.0;
    };
    let temp = json::parse(m.g.get("laya.temperature").str()).unwrap_or(json::J::Null);
    let buckets =
        json::parse(m.g.get("laya.temperature_by_options").str()).unwrap_or(json::J::Null);
    let bucket = format!(
        "{}:{}",
        ["choice", "score", "noul"][kind as usize],
        if k <= 2 {
            "2"
        } else if k <= 5 {
            "3-5"
        } else if k <= 10 {
            "6-10"
        } else {
            "11+"
        }
    );
    let tv = buckets.get(&bucket);
    let t = if *tv != json::J::Null {
        tv.num() as f32
    } else {
        temp.arr()
            .get(kind as usize)
            .map(|v| v.num() as f32)
            .unwrap_or(1.)
    };
    t.max(1e-3)
}

// exp(log_temperature) per kind (choice/score/noul) — lets the JS harness
// compute the engine's temperatured probabilities for OpenThai.
#[no_mangle]
pub unsafe extern "C" fn wasm_openthai_temps(out: *mut f32) {
    if let Some(m) = OPENTHAI.as_ref() {
        for i in 0..3 {
            *out.add(i) = m.temperature[i];
        }
    }
}

// Engine self-test inside wasm: kernel sanity (F32/F16/Q8_0 vs reference,
// simd128 + scalar fallback, bit-exact widen over all 65536 half patterns)
// plus the decision-head self-test. Returns 0 on pass, 1 on error.
#[no_mangle]
pub extern "C" fn wasm_self_test() -> u32 {
    init_panic_hook();
    let r = tensor::self_test().and_then(|_| laya_engine::models::laya::decision::self_test());
    match r {
        Ok(()) => 0,
        Err(e) => {
            unsafe { set_err(e) };
            1
        }
    }
}
