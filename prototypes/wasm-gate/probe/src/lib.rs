// wasm32 probe: candle-core 0.11 on wasm32-unknown-unknown.
//
// Exercises: tensor alloc (global allocator -> memory.grow), f32 matmul
// (gemm), f16 matmul (gemm f16 path), rayon par_iter (fallback check),
// to_dtype/index/softmax-ish elementwise, and a host-injected clock
// (std::time::Instant does not exist on this target).
use rayon::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};

// getrandom 0.3/0.4 custom backend (RUSTFLAGS: --cfg getrandom_backend="custom").
// The forward path never draws randomness; a deterministic fill is honest for
// this gate and avoids wasm-bindgen entirely.
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

static ALLOCATED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PEAK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let n = ALLOCATED.fetch_add(l.size(), std::sync::atomic::Ordering::Relaxed)
                + l.size();
            PEAK.fetch_max(n, std::sync::atomic::Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        ALLOCATED.fetch_sub(l.size(), std::sync::atomic::Ordering::Relaxed);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, ns: usize) -> *mut u8 {
        let old = l.size();
        let q = System.realloc(p, l, ns);
        if !q.is_null() {
            let d = ns as isize - old as isize;
            let n = (ALLOCATED.fetch_add(d as usize, std::sync::atomic::Ordering::Relaxed) as isize
                + d) as usize;
            PEAK.fetch_max(n, std::sync::atomic::Ordering::Relaxed);
        }
        q
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if !p.is_null() {
            let n = ALLOCATED.fetch_add(l.size(), std::sync::atomic::Ordering::Relaxed)
                + l.size();
            PEAK.fetch_max(n, std::sync::atomic::Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static A: Counting = Counting;

#[link(wasm_import_module = "env")]
extern "C" {
    fn now_ms() -> f64;
}

fn t0() -> f64 {
    unsafe { now_ms() }
}

#[no_mangle]
pub extern "C" fn probe_gemm(m: usize, k: usize, n: usize, reps: usize) -> f64 {
    // f32 matmul timing probe; returns ms of the last rep.
    let a = candle_core::Tensor::from_vec(
        (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect(),
        (m, k),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    let b = candle_core::Tensor::from_vec(
        (0..k * n).map(|i| (i % 89) as f32 * 0.01).collect(),
        (k, n),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    let mut ms = 0.0f64;
    for _ in 0..reps {
        let t = t0();
        let c = a.matmul(&b).unwrap();
        let _ = c.to_vec1::<f32>();
        ms = t0() - t;
    }
    ms
}

#[no_mangle]
pub extern "C" fn probe_gemm_f16(m: usize, k: usize, n: usize, reps: usize) -> f64 {
    // f16-resident matmul (cast activations to f16): candle's gemm f16 path.
    let a = candle_core::Tensor::from_vec(
        (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect(),
        (m, k),
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .to_dtype(candle_core::DType::F16)
    .unwrap();
    let b = candle_core::Tensor::from_vec(
        (0..k * n).map(|i| (i % 89) as f32 * 0.01).collect(),
        (k, n),
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .to_dtype(candle_core::DType::F16)
    .unwrap();
    let mut ms = 0.0f64;
    for _ in 0..reps {
        let t = t0();
        let c = a.matmul(&b).unwrap().to_dtype(candle_core::DType::F32).unwrap();
        let _ = c.to_vec1::<f32>();
        ms = t0() - t;
    }
    ms
}

#[no_mangle]
pub extern "C" fn probe_rayon(n: usize) -> u64 {
    // rayon par_iter fallback check: must not panic on wasm32-unknown-unknown.
    let v: Vec<u64> = (0..n as u64).collect();
    v.par_iter().map(|x| x.wrapping_mul(2654435761)).sum()
}

#[no_mangle]
pub extern "C" fn probe_stats() -> usize {
    PEAK.load(std::sync::atomic::Ordering::Relaxed)
}

// Replay the laya-load allocation shape: one big buffer (the GGUF bytes),
// then per-layer {f32 weight temp -> f16 copy kept, f32 freed}, with a
// doubling-Vec realloc chain interleaved (mirrors serde/tensor churn).
// Returns total allocated MB on success; traps if dlmalloc asserts.
#[no_mangle]
pub extern "C" fn probe_alloc_shape(gguf_mb: usize, layer_mb: usize, steps: usize, grow_mb: usize) -> usize {
    let big: Vec<u8> = vec![0u8; gguf_mb << 20];
    let mut kept: Vec<Vec<u8>> = Vec::new();
    for _ in 0..steps {
        let f32w = vec![1u8; layer_mb << 20];
        let mut f16w = Vec::with_capacity(layer_mb >> 1 << 20);
        f16w.extend_from_slice(&f32w[..layer_mb >> 1 << 20]);
        kept.push(f16w);
        drop(f32w);
        let mut v: Vec<u32> = Vec::new();
        while v.capacity() * 4 < grow_mb << 20 {
            v.reserve_exact(v.capacity().max(1024) * 2);
            v.resize(v.capacity(), 1);
        }
        drop(v);
    }
    drop(kept);
    drop(big);
    PEAK.load(std::sync::atomic::Ordering::Relaxed) >> 20
}
