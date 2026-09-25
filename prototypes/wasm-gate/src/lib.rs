// wasm32 gate harness (wayfinder ticket 08): Laya + OpenThai forwards inside
// wasm32 linear memory over a plain extern "C" ABI. The Node harness
// (harness.mjs) copies GGUF bytes in via wasm_alloc_u8, calls wasm_load /
// wasm_forward, and reads logits out through wasm_out_ptr (f32-aligned).
//
// wasm32 has no clock: std::time::Instant panics on this target, so the host
// injects performance.now() as env.now_ms and all load/forward timings below
// are deltas taken inside wasm around the real work.
mod gguf;
mod laya;
mod openthai;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

// ---------------------------------------------------------------- allocator

// std's wasm32 allocator is dlmalloc over memory.grow. Its Rust wrapper
// asserts chunk-size == requested-size on every free/realloc (validate_size),
// and that assert fires on this workload's multi-hundred-MB weight tensors
// once total memory crosses ~2.1 GiB (panic: "psize <= size + max_overhead",
// see report). Big blocks therefore bypass dlmalloc: >= BIG_THRESHOLD goes to
// a page-granular grow-allocator with a coalescing free list (usize math,
// correct across the full 4 GiB wasm32 span); small blocks stay on dlmalloc.
// Live/peak requested-bytes are counted on top. Linear-memory high-water is
// read from JS as memory.buffer.byteLength (monotonic; wasm never shrinks).
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOC: AtomicUsize = AtomicUsize::new(0);
static GROWS: AtomicUsize = AtomicUsize::new(0);

const PAGE: usize = 65536;
const BIG_THRESHOLD: usize = 1 << 20;

struct Big {
    free: Vec<(usize, usize)>,
}

static mut BIG: Big = Big { free: Vec::new() };

impl Big {
    unsafe fn alloc(&mut self, size: usize, zeroed: bool) -> *mut u8 {
        let want = size.div_ceil(PAGE);
        let mut idx = None;
        let mut best = usize::MAX;
        for (i, &(_, pages)) in self.free.iter().enumerate() {
            if pages >= want && pages < best {
                best = pages;
                idx = Some(i);
                if pages == want {
                    break;
                }
            }
        }
        let (ptr, from_free) = match idx {
            Some(i) => {
                let (ptr, pages) = self.free.swap_remove(i);
                if pages > want {
                    self.free.push((ptr + want * PAGE, pages - want));
                }
                (ptr, true)
            }
            None => {
                let old = core::arch::wasm32::memory_grow(0, want);
                if old == usize::MAX {
                    return std::ptr::null_mut();
                }
                (old * PAGE, false)
            }
        };
        if zeroed && from_free {
            std::ptr::write_bytes(ptr as *mut u8, 0, want * PAGE);
        }
        ptr as *mut u8
    }

    unsafe fn dealloc(&mut self, ptr: *mut u8, size: usize) {
        let p = ptr as usize;
        let pages = size.div_ceil(PAGE);
        // coalesce with adjacent free blocks to survive the load transient
        let mut merged_start = p;
        let mut merged_pages = pages;
        let mut i = 0;
        while i < self.free.len() {
            let (fp, fpg) = self.free[i];
            if fp + fpg * PAGE == merged_start {
                merged_start = fp;
                merged_pages += fpg;
                self.free.swap_remove(i);
            } else if merged_start + merged_pages * PAGE == fp {
                merged_pages += fpg;
                self.free.swap_remove(i);
            } else {
                i += 1;
            }
        }
        self.free.push((merged_start, merged_pages));
    }
}

static mut DIAG: Option<String> = None;

// dlmalloc chunk head sits at mem-4 ({prev_foot, head} at mem-8); size is
// head with the 3 flag bits masked off. If a small-block layout disagrees
// with its chunk by more than page slack, record and leak instead of
// tripping dlmalloc's abort assert mid-gate.
unsafe fn dl_chunk_size(p: *mut u8) -> usize {
    if (p as usize) < 16 {
        return 0;
    }
    ((p as *const usize).sub(1)).read_volatile() & !7
}

unsafe fn small_ok(p: *mut u8, size: usize) -> bool {
    let c = dl_chunk_size(p);
    let ok = c + PAGE >= size + 8 && size + 8 + PAGE >= c;
    if !ok {
        DIAG = Some(format!(
            "dlmalloc mismatch: chunk at {:x} csize {} layout {}",
            p as usize, c, size
        ));
    }
    ok
}

struct Router;

unsafe impl GlobalAlloc for Router {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = if l.size() >= BIG_THRESHOLD {
            BIG.alloc(l.size(), false)
        } else {
            System.alloc(l)
        };
        if !p.is_null() {
            bump(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        ALLOCATED.fetch_sub(l.size(), Ordering::Relaxed);
        if l.size() >= BIG_THRESHOLD {
            BIG.dealloc(p, l.size());
        } else if small_ok(p, l.size()) {
            System.dealloc(p, l);
        }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, ns: usize) -> *mut u8 {
        if ns >= BIG_THRESHOLD {
            let old_pages = l.size().div_ceil(PAGE);
            if ns.div_ceil(PAGE) <= old_pages {
                // fits in the same pages; the block may now be smaller than
                // its rounded page span, which later deallocs handle fine
                if ns > l.size() {
                    bump(ns - l.size());
                } else {
                    ALLOCATED.fetch_sub(l.size() - ns, Ordering::Relaxed);
                }
                return p;
            }
            let q = BIG.alloc(ns, false);
            if q.is_null() {
                return q;
            }
            std::ptr::copy_nonoverlapping(p, q, l.size().min(ns));
            ALLOCATED.fetch_sub(l.size(), Ordering::Relaxed);
            if l.size() >= BIG_THRESHOLD {
                BIG.dealloc(p, l.size());
            } else {
                System.dealloc(p, l);
            }
            bump(ns);
            return q;
        }
        // shrinking big -> small must also leave dlmalloc territory: the old
        // block is a raw page region, not a dlmalloc chunk
        if l.size() >= BIG_THRESHOLD {
            let q = BIG.alloc(ns.max(1), false);
            if q.is_null() {
                return q;
            }
            std::ptr::copy_nonoverlapping(p, q, l.size().min(ns));
            ALLOCATED.fetch_sub(l.size(), Ordering::Relaxed);
            BIG.dealloc(p, l.size());
            bump(ns);
            return q;
        }
        if !small_ok(p, l.size()) {
            return std::ptr::null_mut();
        }
        let q = System.realloc(p, l, ns);
        if !q.is_null() {
            if ns >= l.size() {
                bump(ns - l.size());
            } else {
                ALLOCATED.fetch_sub(l.size() - ns, Ordering::Relaxed);
            }
        }
        q
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = if l.size() >= BIG_THRESHOLD {
            BIG.alloc(l.size(), true)
        } else {
            System.alloc_zeroed(l)
        };
        if !p.is_null() {
            bump(l.size());
        }
        p
    }
}

fn bump(delta: usize) {
    let n = ALLOCATED.fetch_add(delta, Ordering::Relaxed) + delta;
    PEAK_ALLOC.fetch_max(n, Ordering::Relaxed);
    GROWS.fetch_add(1, Ordering::Relaxed);
}

#[global_allocator]
static A: Router = Router;

// getrandom 0.3/0.4 custom backend, activated by RUSTFLAGS
// --cfg getrandom_backend="custom" (wasm32-unknown-unknown has no OS
// entropy). Deterministic fill: the forward path never draws randomness.
#[no_mangle]
pub extern "Rust" fn __getrandom_v03_custom(dest: *mut u8, len: usize) -> Result<(), getrandom::Error> {
    let mut state = 0x9E3779B97F4A7C15u64;
    for i in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        unsafe { *dest.add(i) = state as u8 }
    }
    Ok(())
}

// -------------------------------------------------------------------- clock

#[link(wasm_import_module = "env")]
extern "C" {
    fn now_ms() -> f64;
}

fn clock() -> f64 {
    unsafe { now_ms() }
}

// ------------------------------------------------------------------- state

static LOAD_MS: AtomicU64 = AtomicU64::new(0);
static FORWARD_MS: AtomicU64 = AtomicU64::new(0);

static mut LAYA: Option<Box<laya::Model>> = None;
static mut OPENTHAI: Option<Box<openthai::Model>> = None;
static mut OUT: Vec<f32> = Vec::new();
static mut OUT2: Vec<f32> = Vec::new();
static mut LAST_ERROR: Option<String> = None;

// Panic payloads have no stderr on wasm32-unknown-unknown; route them into
// LAST_ERROR so the harness can read the actual message after the trap.
pub(crate) fn init_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        unsafe {
            LAST_ERROR = Some(format!("panic: {msg}{}", info.location().map(|l| format!(" at {l}")).unwrap_or_default()));
        }
    }));
}

unsafe fn set_err(e: impl Into<String>) {
    LAST_ERROR = Some(e.into());
}

fn mode_from_u32(m: u32) -> Option<laya::WeightMode> {
    match m {
        0 => Some(laya::WeightMode::F16),
        1 => Some(laya::WeightMode::Convert),
        _ => None,
    }
}

fn ot_mode_from_u32(m: u32) -> Option<openthai::WeightMode> {
    match m {
        0 => Some(openthai::WeightMode::F16),
        1 => Some(openthai::WeightMode::Convert),
        2 => Some(openthai::WeightMode::Q80),
        _ => None,
    }
}

// ------------------------------------------------------------------ exports

#[no_mangle]
pub extern "C" fn wasm_alloc_u8(n: usize) -> *mut u8 {
    let layout = Layout::from_size_align(n.max(1), 1).unwrap();
    unsafe { A.alloc(layout) }
}

#[no_mangle]
pub extern "C" fn wasm_alloc_u32(n: usize) -> *mut u32 {
    let layout = Layout::from_size_align(n.max(1) * 4, 4).unwrap();
    unsafe { A.alloc(layout) as *mut u32 }
}

#[no_mangle]
pub unsafe extern "C" fn wasm_free_u32(ptr: *mut u32, n: usize) {
    if !ptr.is_null() {
        let layout = Layout::from_size_align(n * 4, 4).unwrap();
        A.dealloc(ptr as *mut u8, layout);
    }
}

// Parses + builds the model from GGUF bytes previously copied to `ptr`
// (wasm_alloc_u8). Takes ownership of the buffer: the byte vec is freed when
// the Gguf drops, so linear memory stays committed (never shrinks) but the
// allocator live-bytes counter drops by the file size.
//
// model: 0 = laya, 1 = openthai. mode: 0 = f16, 1 = convert, 2 = q8_0
// (openthai only). Returns 0 on success, 1 on error (see wasm_last_error).
#[no_mangle]
pub unsafe extern "C" fn wasm_load(model: u32, mode: u32, ptr: *mut u8, len: usize) -> u32 {
    init_panic_hook();
    // capacity must equal the wasm_alloc_u8 size, or the Vec's later dealloc
    // presents a wrong layout to the allocator
    let data = Vec::from_raw_parts(ptr, len, len);
    let t0 = clock();
    let g = match gguf::Gguf::from_bytes(data) {
        Ok(g) => g,
        Err(e) => {
            set_err(format!("gguf: {e}"));
            return 1;
        }
    };
    let parse_ms = clock() - t0;
    let t1 = clock();
    let result = match model {
        0 => {
            let m = mode_from_u32(mode)
                .ok_or_else(|| "bad mode (laya: 0=f16 1=convert)".to_string());
            let temps = serde_json::from_str::<Vec<f32>>(g.meta_str("laya.temperature"))
                .map(|v| [v.first().copied().unwrap_or(1.0), v.get(1).copied().unwrap_or(1.0), v.get(2).copied().unwrap_or(1.0)])
                .unwrap_or([1.0, 1.0, 1.0]);
            let buckets = serde_json::from_str::<std::collections::HashMap<String, f32>>(
                g.meta_str("laya.temperature_by_options"),
            )
            .unwrap_or_default();
            m.and_then(|m| laya::Model::load(&g, m))
                .map(|m| {
                    LAYA = Some(Box::new(m));
                    LAYA_TEMPS = Some((temps, buckets));
                })
        }
        1 => {
            let m = ot_mode_from_u32(mode)
                .ok_or_else(|| "bad mode (openthai: 0=f16 1=convert 2=q80)".to_string());
            m.and_then(|m| openthai::Model::load(&g, m))
                .map(|m| OPENTHAI = Some(Box::new(m)))
        }
        _ => Err("bad model id (0=laya 1=openthai)".to_string()),
    };
    let build_ms = clock() - t1;
    drop(g);
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

static PARSE_MS: AtomicU64 = AtomicU64::new(0);
static BUILD_MS: AtomicU64 = AtomicU64::new(0);
static mut LAYA_TEMPS: Option<([f32; 3], std::collections::HashMap<String, f32>)> = None;

// decision::Temps::temp port (bucket key + floor), evaluated against the
// metadata stashed at load time so the JS harness can rebuild the engine's
// temperatured probabilities.
fn laya_temp(kind: usize, k: usize) -> f32 {
    unsafe {
        let (base, buckets) = match LAYA_TEMPS.as_ref() {
            Some(t) => (&t.0, &t.1),
            None => return 1.0,
        };
        let bucket = format!(
            "{}:{}",
            ["choice", "score", "noul"][kind],
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
        buckets
            .get(&bucket)
            .copied()
            .unwrap_or(base[kind])
            .max(1e-3)
    }
}

#[no_mangle]
pub extern "C" fn wasm_laya_temp(kind: u32, k: u32) -> f32 {
    laya_temp(kind as usize, k as usize)
}

// Runs one forward. ids: u32 token ids. aux: markers (laya, positions in the
// stacked batch sense of the native prototype) or answer positions (openthai).
// qtype: 0 choice / 1 score / 2 noul (laya only). Logits land in OUT (f32,
// wasm_out_ptr / count in low 32 bits of the return), laya act logits land in
// OUT2 (count in the high 32 bits). Returns u64::MAX on error.
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
    let ids = std::slice::from_raw_parts(ids_ptr, ids_len);
    let aux = std::slice::from_raw_parts(aux_ptr, aux_len);
    let t0 = clock();
    let result = match model {
        0 => LAYA.as_ref().ok_or("laya not loaded".to_string()).and_then(|m| {
            let markers: Vec<usize> = aux.iter().map(|&v| v as usize).collect();
            m.forward(ids, &markers, qtype as usize)
                .map(|(logits, act)| (logits, Some(act)))
        }),
        1 => OPENTHAI
            .as_ref()
            .ok_or("openthai not loaded".to_string())
            .and_then(|m| {
                let positions: Vec<usize> = aux.iter().map(|&v| v as usize).collect();
                m.forward(ids, &positions).map(|l| (l, None))
            }),
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
pub unsafe extern "C" fn wasm_out_ptr() -> *const f32 {
    OUT.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn wasm_out2_ptr() -> *const f32 {
    OUT2.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn wasm_diag() -> *const u8 {
    match DIAG.as_ref() {
        Some(e) => e.as_ptr(),
        None => b"\0".as_ptr(),
    }
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
    PEAK_ALLOC.load(Ordering::Relaxed)
}

#[no_mangle]
pub extern "C" fn wasm_stat_live_alloc() -> usize {
    ALLOCATED.load(Ordering::Relaxed)
}

#[no_mangle]
pub extern "C" fn wasm_stat_alloc_calls() -> usize {
    GROWS.load(Ordering::Relaxed)
}

// exp(log_temperature) per kind (choice/score/noul) — lets the JS harness
// compute the engine's temperatured probabilities without a decision port.
#[no_mangle]
pub unsafe extern "C" fn wasm_openthai_temps(out: *mut f32) {
    if let Some(m) = OPENTHAI.as_ref() {
        for i in 0..3 {
            *out.add(i) = m.temperature[i];
        }
    }
}
