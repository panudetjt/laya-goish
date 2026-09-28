//! Global allocator for the inference path.
//!
//! The goish engine's allocator is a Go-style size-class heap: large
//! recurring buffers (per-apply outputs, attention activations) come back
//! page-aligned and are recycled whole. glibc's ptmalloc instead
//! splits/merges 16-byte-aligned heap chunks (or mmap/munmaps them), which
//! measurably slowed the streaming AVX2 kernels. This allocator keeps a
//! small fixed-slot free list per power-of-two size class (64 KiB..16 MiB,
//! page-aligned, at most 4 cached blocks per class and at most
//! `CLASS_RETAIN` bytes per class) and delegates everything else to the
//! system allocator. `alloc_zeroed` re-zeroes cached blocks — required
//! because e.g. the GeGLU and GDN outputs accumulate with `+=` into the
//! fresh `vec![0.]` buffers.
//!
//! Page return (ticket 21): class blocks are anonymous page mappings, not
//! glibc chunks. Blocks freed past the retention cap munmap straight back
//! to the OS instead of being stashed in ptmalloc's arena — freeing a
//! large mmap'd chunk raises glibc's dynamic mmap threshold, after which
//! same-sized allocations come from the brk arena and are never returned;
//! over the ticket-13 soak that arena dance retained +310 MiB at the
//! concurrent peak. Fresh mappings arrive zeroed, so the refill path pays
//! page faults (what glibc's own mmap path paid too), not memset.
#[cfg(not(target_arch = "wasm32"))]
use std::alloc::GlobalAlloc;
#[cfg(not(target_arch = "wasm32"))]
use std::alloc::Layout;
#[cfg(not(target_arch = "wasm32"))]
use std::alloc::System;
#[cfg(not(target_arch = "wasm32"))]
use std::ptr;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

#[cfg(not(target_arch = "wasm32"))]
const MIN_CLASS: usize = 1 << 16;
#[cfg(not(target_arch = "wasm32"))]
const MAX_CLASS: usize = 1 << 24;
#[cfg(not(target_arch = "wasm32"))]
const CLASSES: usize = MAX_CLASS.ilog2() as usize - MIN_CLASS.ilog2() as usize + 1;
#[cfg(not(target_arch = "wasm32"))]
const SLOTS: usize = 4;
#[cfg(not(target_arch = "wasm32"))]
const PAGE: usize = 4096;
// Per-class retention cap: cached bytes per size class stay at or below
// this (SLOTS still bounds the block count), so total retention across the
// nine classes is bounded (~32 MiB worst case) instead of the 128 MiB a
// full 4-slot cache of every class could hold. Classes larger than the
// cap are never cached — their blocks return to the OS on free.
#[cfg(not(target_arch = "wasm32"))]
const CLASS_RETAIN: usize = 8 << 20;

#[cfg(not(target_arch = "wasm32"))]
struct Cache {
    blocks: [[*mut u8; SLOTS]; CLASSES],
    used: [usize; CLASSES],
}
// Blocks move between threads via the Mutex; each cached block is owned by
// whichever thread the allocator hands it to, so sharing the free list is
// sound.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for Cache {}
#[cfg(not(target_arch = "wasm32"))]
impl Cache {
    const fn new() -> Self {
        Self {
            blocks: [[ptr::null_mut(); SLOTS]; CLASSES],
            used: [0; CLASSES],
        }
    }
}
#[cfg(not(target_arch = "wasm32"))]
pub struct EngineAlloc {
    large: Mutex<Cache>,
}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl GlobalAlloc for EngineAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.take(layout, false)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.take(layout, true)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        match class_of(layout) {
            Some(c) => {
                let mut cache = self.large.lock().unwrap();
                let i = cache.used[c];
                if i < slots_for(c) {
                    cache.blocks[c][i] = ptr;
                    cache.used[c] = i + 1;
                    return;
                }
                drop(cache);
                sys::free(ptr, class_size(c));
            }
            None => System.dealloc(ptr, layout),
        }
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl EngineAlloc {
    unsafe fn take(&self, layout: Layout, zeroed: bool) -> *mut u8 {
        let c = match class_of(layout) {
            Some(c) => c,
            None => {
                return if zeroed {
                    System.alloc_zeroed(layout)
                } else {
                    System.alloc(layout)
                };
            }
        };
        let mut cache = self.large.lock().unwrap();
        if cache.used[c] > 0 {
            cache.used[c] -= 1;
            let p = cache.blocks[c][cache.used[c]];
            drop(cache);
            if zeroed {
                ptr::write_bytes(p, 0, layout.size());
            }
            return p;
        }
        drop(cache);
        sys::alloc(class_size(c), zeroed)
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn class_of(layout: Layout) -> Option<usize> {
    let size = layout.size();
    if size < MIN_CLASS || size > MAX_CLASS || layout.align() > PAGE {
        return None;
    }
    let class = size.max(layout.align()).next_power_of_two();
    Some(class.ilog2() as usize - MIN_CLASS.ilog2() as usize)
}
#[cfg(not(target_arch = "wasm32"))]
fn class_size(c: usize) -> usize {
    1usize << (c + MIN_CLASS.ilog2() as usize)
}
#[cfg(not(target_arch = "wasm32"))]
fn slots_for(c: usize) -> usize {
    SLOTS.min(CLASS_RETAIN >> (MIN_CLASS.ilog2() as usize + c))
}

// Backing store for class blocks. On Linux they are raw anonymous page
// mappings: freed-overflow blocks munmap straight back to the kernel (no
// ptmalloc arena stash, no dynamic mmap-threshold growth), and fresh
// mappings arrive zeroed so the zeroed-alloc path pays nothing extra.
// Other non-wasm targets (Windows/macOS compile checks) keep the system
// allocator with identical semantics.
#[cfg(target_os = "linux")]
mod sys {
    use std::ptr;

    const PROT_READ_WRITE: i32 = 0x1 | 0x2;
    const MAP_PRIVATE_ANONYMOUS: i32 = 0x02 | 0x20;

    extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
        fn munmap(addr: *mut u8, len: usize) -> i32;
    }

    // Safety: `size` is a class size (a power of two >= 64 KiB), so the
    // mapping length and page alignment always match the dealloc side.
    pub unsafe fn alloc(size: usize, _zeroed: bool) -> *mut u8 {
        let p = mmap(
            ptr::null_mut(),
            size,
            PROT_READ_WRITE,
            MAP_PRIVATE_ANONYMOUS,
            -1,
            0,
        );
        if p as usize == usize::MAX {
            ptr::null_mut()
        } else {
            p
        }
    }

    // Safety: `ptr` came from `alloc(size)` with the same class size.
    pub unsafe fn free(ptr: *mut u8, size: usize) {
        let _ = munmap(ptr, size);
    }
}
#[cfg(all(not(target_os = "linux"), not(target_arch = "wasm32")))]
mod sys {
    use std::alloc::{GlobalAlloc, Layout, System};

    fn layout(size: usize) -> Layout {
        Layout::from_size_align(size, super::PAGE).unwrap()
    }

    pub unsafe fn alloc(size: usize, zeroed: bool) -> *mut u8 {
        if zeroed {
            System.alloc_zeroed(layout(size))
        } else {
            System.alloc(layout(size))
        }
    }
    pub unsafe fn free(ptr: *mut u8, size: usize) {
        System.dealloc(ptr, layout(size));
    }
}
#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static ALLOC: EngineAlloc = EngineAlloc {
    large: Mutex::new(Cache::new()),
};

// ---------------------------------------------------------------------------
// wasm32 allocator (ticket 19): page-granular router proven in
// prototypes/wasm-gate. std's wasm32 allocator is dlmalloc over memory.grow;
// its Rust wrapper asserts chunk-size == requested-size on every
// free/realloc, and that assert fires on this workload's multi-hundred-MB
// weight tensors once total memory crosses ~2.1 GiB. Blocks >= 1 MiB bypass
// dlmalloc into page-granular memory.grow regions with a coalescing free
// list (usize math, correct across the full 4 GiB wasm32 span); small
// blocks stay on dlmalloc. Live/peak requested bytes are counted on top so
// the Node harness can read them through the wasm ABI.
#[cfg(target_arch = "wasm32")]
mod wasm {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::alloc::{GlobalAlloc, Layout, System};
    #[cfg(not(wasm_threads))]
    use std::sync::Mutex;

    const PAGE: usize = 65536;
    const BIG_THRESHOLD: usize = 1 << 20;

    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
    static PEAK_ALLOC: AtomicUsize = AtomicUsize::new(0);
    static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);

    struct Big {
        free: Vec<(usize, usize)>,
    }

    // std's wasm32-unknown-unknown locks are the no_threads fakes: the
    // distributed std rlib is built without atomics, so std Mutex panics
    // ("cannot recursively acquire mutex") the moment two agents contend
    // (observed on rustc 1.96 under the threads artifact). The atomics
    // build therefore serializes both allocator paths with this real
    // compare_exchange spinlock; the no-atomics build keeps the std Mutex,
    // which is exact for its single-agent regime.
    #[cfg(wasm_threads)]
    use core::sync::atomic::AtomicBool;

    #[cfg(wasm_threads)]
    struct SpinLock(AtomicBool);

    #[cfg(wasm_threads)]
    impl SpinLock {
        const fn new() -> Self {
            Self(AtomicBool::new(false))
        }
        fn lock(&self) -> SpinGuard<'_> {
            while self
                .0
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                core::hint::spin_loop();
            }
            SpinGuard(self)
        }
    }

    #[cfg(wasm_threads)]
    struct SpinGuard<'a>(&'a SpinLock);

    #[cfg(wasm_threads)]
    impl Drop for SpinGuard<'_> {
        fn drop(&mut self) {
            self.0 .0.store(false, Ordering::Release);
        }
    }

    #[cfg(not(wasm_threads))]
    static BIG_LOCK: Mutex<()> = Mutex::new(());
    #[cfg(wasm_threads)]
    static BIG_LOCK: SpinLock = SpinLock::new();

    static mut BIG: Big = Big { free: Vec::new() };

    // std's wasm32 dlmalloc arena is not safe for concurrent callers
    // (cross-agent alloc/free corrupts its free list, verified on rustc
    // 1.96), so the small-block delegation is serialized whenever the
    // atomics build lets workers allocate in parallel. BIG_LOCK already
    // serializes the large path.
    #[cfg(not(wasm_threads))]
    static SMALL_LOCK: Mutex<()> = Mutex::new(());
    #[cfg(wasm_threads)]
    static SMALL_LOCK: SpinLock = SpinLock::new();

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

    // dlmalloc chunk head sits at mem-4 ({prev_foot, head} at mem-8); size is
    // the head with the 3 flag bits masked off. If a small-block layout
    // disagrees with its chunk by more than page slack, leak instead of
    // tripping dlmalloc's abort assert mid-run (the prototype recorded zero
    // genuine mismatches across every configuration).
    unsafe fn small_ok(p: *mut u8, size: usize) -> bool {
        if (p as usize) < 16 {
            return false;
        }
        let c = ((p as *const usize).sub(1)).read_volatile() & !7;
        c + PAGE >= size + 8 && size + 8 + PAGE >= c
    }

    pub struct Router;

    unsafe impl GlobalAlloc for Router {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let p = if l.size() >= BIG_THRESHOLD {
                big(|b| b.alloc(l.size(), false))
            } else {
                sys_alloc(l)
            };
            if !p.is_null() {
                bump(l.size());
            }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            ALLOCATED.fetch_sub(l.size(), Ordering::Relaxed);
            if l.size() >= BIG_THRESHOLD {
                big(|b| b.dealloc(p, l.size()));
            } else if small_ok(p, l.size()) {
                sys_dealloc(p, l);
            }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, ns: usize) -> *mut u8 {
            if ns >= BIG_THRESHOLD {
                let old_pages = l.size().div_ceil(PAGE);
                if ns.div_ceil(PAGE) <= old_pages {
                    // fits in the same pages; the block may now be smaller
                    // than its rounded page span, which later deallocs of the
                    // ORIGINAL size handle fine (the engine's large buffers
                    // realloc with the same layout contract)
                    if ns > l.size() {
                        bump(ns - l.size());
                    } else {
                        ALLOCATED.fetch_sub(l.size() - ns, Ordering::Relaxed);
                    }
                    return p;
                }
                let q = big(|b| b.alloc(ns, false));
                if q.is_null() {
                    return q;
                }
                std::ptr::copy_nonoverlapping(p, q, l.size().min(ns));
                ALLOCATED.fetch_sub(l.size(), Ordering::Relaxed);
                if l.size() >= BIG_THRESHOLD {
                    big(|b| b.dealloc(p, l.size()));
                } else if small_ok(p, l.size()) {
                    sys_dealloc(p, l);
                }
                bump(ns);
                return q;
            }
            // shrinking big -> small must also leave dlmalloc territory: the
            // old block is a raw page region, not a dlmalloc chunk
            if l.size() >= BIG_THRESHOLD {
                let q = big(|b| b.alloc(ns.max(1), false));
                if q.is_null() {
                    return q;
                }
                std::ptr::copy_nonoverlapping(p, q, l.size().min(ns));
                ALLOCATED.fetch_sub(l.size(), Ordering::Relaxed);
                big(|b| b.dealloc(p, l.size()));
                bump(ns);
                return q;
            }
            if !small_ok(p, l.size()) {
                return std::ptr::null_mut();
            }
            let q = sys_realloc(p, l, ns);
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
                big(|b| b.alloc(l.size(), true))
            } else {
                sys_alloc_zeroed(l)
            };
            if !p.is_null() {
                bump(l.size());
            }
            p
        }
    }

    unsafe fn big<R>(f: impl FnOnce(&mut Big) -> R) -> R {
        #[cfg(not(wasm_threads))]
        let _guard = BIG_LOCK.lock().unwrap();
        #[cfg(wasm_threads)]
        let _guard = BIG_LOCK.lock();
        f(&mut *core::ptr::addr_of_mut!(BIG))
    }

    unsafe fn sys_alloc(l: Layout) -> *mut u8 {
        #[cfg(not(wasm_threads))]
        let _g = SMALL_LOCK.lock().unwrap();
        #[cfg(wasm_threads)]
        let _g = SMALL_LOCK.lock();
        System.alloc(l)
    }

    unsafe fn sys_alloc_zeroed(l: Layout) -> *mut u8 {
        #[cfg(not(wasm_threads))]
        let _g = SMALL_LOCK.lock().unwrap();
        #[cfg(wasm_threads)]
        let _g = SMALL_LOCK.lock();
        System.alloc_zeroed(l)
    }

    unsafe fn sys_dealloc(p: *mut u8, l: Layout) {
        #[cfg(not(wasm_threads))]
        let _g = SMALL_LOCK.lock().unwrap();
        #[cfg(wasm_threads)]
        let _g = SMALL_LOCK.lock();
        System.dealloc(p, l)
    }

    unsafe fn sys_realloc(p: *mut u8, l: Layout, ns: usize) -> *mut u8 {
        #[cfg(not(wasm_threads))]
        let _g = SMALL_LOCK.lock().unwrap();
        #[cfg(wasm_threads)]
        let _g = SMALL_LOCK.lock();
        System.realloc(p, l, ns)
    }

    fn bump(delta: usize) {
        let n = ALLOCATED.fetch_add(delta, Ordering::Relaxed) + delta;
        PEAK_ALLOC.fetch_max(n, Ordering::Relaxed);
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    #[global_allocator]
    static A: Router = Router;

    pub fn live_bytes() -> usize {
        ALLOCATED.load(Ordering::Relaxed)
    }
    pub fn peak_bytes() -> usize {
        PEAK_ALLOC.load(Ordering::Relaxed)
    }
    pub fn alloc_calls() -> usize {
        ALLOC_CALLS.load(Ordering::Relaxed)
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::{alloc_calls, live_bytes, peak_bytes};
