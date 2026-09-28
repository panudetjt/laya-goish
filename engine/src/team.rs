//! go!/WaitGroup replacement for the four engine fan-out sites.
//!
//! The goish engine dispatches each fan-out as count-1 goroutines while the
//! caller runs range 0; its scheduler wakes idle Ms with per-thread futex
//! calls, so a fan-out costs ~1µs. rayon's pool (scope-spawn and
//! parallel-iterator forms were both measured) parks workers between the
//! engine's many small fan-outs (~220 per Laya decide) and pays
//! split/steal/condvar costs that added 20-40% at 12/16 threads — outside
//! the ticket's 1% latency gate. This module reproduces the goish
//! mechanism directly: persistent worker threads woken per fan-out with
//! `Thread::unpark` tokens (no shared lock, no missed wakes — an unpark
//! that lands before park simply makes park return immediately), and
//! workers park immediately when idle so they cost no CPU between
//! fan-outs.
//!
//! Protocol (single coordinator thread — the engine's decide path, and the
//! goish server it replaces, serialize inference): the caller publishes the
//! job descriptor and `active` count, resets `done`, then bumps `gen`
//! (SeqCst). Workers that see a new `gen` (Acquire) read the job and, if
//! their index t is below `active`, run f(t) and `done.fetch_add(1)`. The
//! caller runs f(0) and returns once done reaches active-1. Ranges are
//! computed by the callers with the goish engine's exact per/count
//! arithmetic, preserving the disjoint-output bit-exactness invariant.
#[cfg(not(target_arch = "wasm32"))]
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::cell::UnsafeCell;
#[cfg(not(target_arch = "wasm32"))]
use std::hint::spin_loop;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Mutex, Once, OnceLock};

// wasm32 without atomics cannot spawn threads. Every range still runs,
// sequentially on the caller, with the same disjoint-range arithmetic, so
// outputs match the threaded engine element-for-element; the wasm bars are
// single-threaded by design (wayfinder ticket 19).
#[cfg(all(target_arch = "wasm32", not(wasm_threads)))]
pub fn fanout<F: Fn(usize) + Sync>(count: usize, f: &F) {
    for t in 0..count {
        f(t);
    }
}

// wasm32 +atomics worker fan-out (wayfinder ticket 24). std::thread is not
// implemented for wasm32-unknown-unknown — Builder::spawn returns
// Err("operation not supported on this platform") on rustc 1.96 — so the
// host drives the workers: it instantiates this module once per worker
// over ONE shared WebAssembly.Memory, points each instance's exported
// __stack_pointer at a private stack slot, and parks the worker in JS on
// the GEN word of the shared linear memory (std's wasm futex intrinsics
// are unstable, so Atomics.wait/notify live host-side). The caller
// publishes the job, bumps GEN, and wakes the workers through the
// imported env.futex_wake; workers run wasm_worker_poll per wake. All
// statics live in shared linear memory, so the protocol state — and the
// sync edges the job cell relies on — are common to every instance.
// Workers spawn before any state exists (late instantiation re-applies
// active data segments and would reset statics), so the harness gates on
// wasm_team_ready before the first load.
#[cfg(all(target_arch = "wasm32", wasm_threads))]
mod wasm_threads {
    use core::cell::UnsafeCell;
    use core::sync::atomic::{AtomicUsize, Ordering};

    type Run = fn(*const u8, usize);
    struct JobDesc {
        run: Run,
        ctx: *const u8,
    }
    struct JobCell(UnsafeCell<JobDesc>);
    // SAFETY: a published JobDesc is only invoked, never moved, and is not
    // mutated while visible to workers (the gen protocol serializes
    // publish vs. read exactly as the native team does).
    unsafe impl Sync for JobCell {}
    static JOB: JobCell = JobCell(UnsafeCell::new(JobDesc {
        run: |_ctx, _t| {},
        ctx: core::ptr::null(),
    }));

    static GEN: AtomicUsize = AtomicUsize::new(0);
    static DONE: AtomicUsize = AtomicUsize::new(0);
    static ACTIVE: AtomicUsize = AtomicUsize::new(0);
    static READY: AtomicUsize = AtomicUsize::new(0);

    #[link(wasm_import_module = "env")]
    extern "C" {
        fn futex_wake(addr: u32, count: u32);
    }

    /// Same contract as the native fanout: f(t) for t in 0..count, caller
    /// runs t=0, returns after every range completed. Falls back to the
    /// sequential loop when the host spawned fewer workers than count-1
    /// (or none — the artifact stays usable single-instance).
    pub fn fanout<F: Fn(usize) + Sync>(count: usize, f: &F) {
        if count <= 1 {
            f(0);
            return;
        }
        if READY.load(Ordering::Acquire) < count - 1 {
            for t in 0..count {
                f(t);
            }
            return;
        }
        // SAFETY: workers only read JOB after observing the gen store
        // below (Release/Acquire pair) and only invoke it while this
        // function is still on the stack (it returns once done reaches
        // count-1), so f outlives every use.
        unsafe {
            (*JOB.0.get()) = JobDesc {
                run: |ctx, t| (*(ctx as *const F))(t),
                ctx: f as *const F as *const u8,
            };
        }
        ACTIVE.store(count, Ordering::Release);
        DONE.store(0, Ordering::Relaxed);
        // SeqCst store + futex wake: every worker either loads the new gen
        // or is inside its host-side wait, whose expected value was read
        // before — Atomics.wait re-checks the value atomically, so a wake
        // can never be missed.
        GEN.store(GEN.load(Ordering::Relaxed) + 1, Ordering::SeqCst);
        unsafe {
            futex_wake(&GEN as *const AtomicUsize as usize as u32, count as u32 - 1);
        }
        f(0);
        // The caller is the critical path; at K<=4 workers the join tail
        // is short enough that spinning beats a host wait round-trip.
        while DONE.load(Ordering::Acquire) < count - 1 {
            core::hint::spin_loop();
        }
    }

    /// Worker registration: bumps the ready count and returns the GEN
    /// baseline the host should park on.
    #[no_mangle]
    pub extern "C" fn wasm_worker_hello() -> u32 {
        READY.fetch_add(1, Ordering::SeqCst);
        GEN.load(Ordering::Acquire) as u32
    }

    /// One wake iteration for worker t: runs the job if GEN moved past
    /// `seen` and t is below ACTIVE, then returns the observed gen (the
    /// host re-arms its wait with it).
    #[no_mangle]
    pub extern "C" fn wasm_worker_poll(t: u32, seen: u32) -> u32 {
        let g = GEN.load(Ordering::Acquire);
        if g == seen as usize {
            return seen;
        }
        let active = ACTIVE.load(Ordering::Acquire);
        if (t as usize) < active {
            // SAFETY: gen was published after the coordinator wrote JOB
            // (Release/Acquire pair), and the coordinator cannot return
            // from fanout until this worker's fetch_add lands.
            unsafe {
                let desc = &*JOB.0.get();
                (desc.run)(desc.ctx, t as usize);
            }
            DONE.fetch_add(1, Ordering::Release);
        }
        g as u32
    }

    /// GEN's address in linear memory (the host parks on it).
    #[no_mangle]
    pub extern "C" fn wasm_team_gen_addr() -> u32 {
        &GEN as *const AtomicUsize as usize as u32
    }

    /// Workers registered so far (host gates its first load on this).
    #[no_mangle]
    pub extern "C" fn wasm_team_ready() -> u32 {
        READY.load(Ordering::Acquire) as u32
    }
}

#[cfg(all(target_arch = "wasm32", wasm_threads))]
pub use wasm_threads::fanout;

#[cfg(not(target_arch = "wasm32"))]
type Run = fn(*const u8, usize);
#[cfg(not(target_arch = "wasm32"))]
struct JobDesc {
    run: Run,
    ctx: *const u8,
}
#[cfg(not(target_arch = "wasm32"))]
struct JobCell(UnsafeCell<JobDesc>);
// SAFETY: a published JobDesc is only invoked, never moved, and is not
// mutated while visible to workers (see fanout's gen protocol).
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for JobCell {}
#[cfg(not(target_arch = "wasm32"))]
static JOB: JobCell = JobCell(UnsafeCell::new(JobDesc {
    run: |_ctx, _t| {},
    ctx: std::ptr::null(),
}));

#[cfg(not(target_arch = "wasm32"))]
static GEN: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(target_arch = "wasm32"))]
static DONE: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(target_arch = "wasm32"))]
static ACTIVE: AtomicUsize = AtomicUsize::new(1);
// Bit t-1 set while worker t is parked.
#[cfg(not(target_arch = "wasm32"))]
static PARKED: AtomicU64 = AtomicU64::new(0);

#[cfg(not(target_arch = "wasm32"))]
static SPAWNED: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(target_arch = "wasm32"))]
static SPAWN_LOCK: Mutex<()> = Mutex::new(());
#[cfg(not(target_arch = "wasm32"))]
static HANDLES: OnceLock<Mutex<Vec<std::thread::Thread>>> = OnceLock::new();
#[cfg(not(target_arch = "wasm32"))]
static INIT: Once = Once::new();

// Idle policy: workers park immediately (goish's idle Ms futex-park too).
// Idle spinning burns scheduler slots that the running kernels need on
// this SMT machine — measured 2x total CPU burn and +20-30% latency when
// workers spin between the engine's many small fan-outs. Wake cost is one
// unpark per parked worker per fan-out, the same shape as goish's wakep.
#[cfg(not(target_arch = "wasm32"))]
const PARK_TIMEOUT: std::time::Duration = std::time::Duration::from_micros(60);

/// Run f(t) for t in 0..count on the caller (t=0) and persistent workers
/// (t=1..count-1), returning after every range completed. Mirrors the
/// goish go!/WaitGroup sites: same range arithmetic, caller included.
#[cfg(not(target_arch = "wasm32"))]
pub fn fanout<F: Fn(usize) + Sync>(count: usize, f: &F) {
    if count <= 1 {
        f(0);
        return;
    }
    ensure_workers(count - 1);
    // SAFETY: workers only read JOB after observing the gen store below
    // (Release/Acquire pair) and only invoke it while this function is
    // still on the stack (it returns once done reaches count-1), so f
    // outlives every use.
    unsafe {
        (*JOB.0.get()) = JobDesc {
            run: |ctx, t| (*(ctx as *const F))(t),
            ctx: f as *const F as *const u8,
        };
    }
    ACTIVE.store(count, Ordering::Release);
    DONE.store(0, Ordering::Relaxed);
    // SeqCst: the gen store must be globally visible before the PARKED scan
    // below. With only Release, x86's store buffer lets this thread read
    // PARKED while the gen store is still buffered — a worker could then
    // set its bit, load the stale gen, and park after we decided not to
    // wake it (deadlock, observed at 16 threads).
    GEN.store(GEN.load(Ordering::Relaxed) + 1, Ordering::SeqCst);
    let parked = PARKED.load(Ordering::SeqCst);
    if parked != 0 {
        if let Some(h) = HANDLES.get() {
            let handles = h.lock().unwrap();
            let mut m = parked;
            while m != 0 {
                let t = m.trailing_zeros() as usize + 1;
                m &= m - 1;
                if let Some(th) = handles.get(t - 1) {
                    th.unpark();
                }
            }
        }
    }
    f(0);
    // Brief spin then timed sleep: spinning through the whole join pins a
    // hardware slot that a straggler worker needs (CFS only hands it over
    // at preemption granularity — measured 100µs+ join tails at 12/16T).
    // goish's WaitGroup parks the caller the same way.
    let mut spun = 0u32;
    while DONE.load(Ordering::Acquire) < count - 1 {
        if spun < 2_000 {
            spin_loop();
            spun += 1;
        } else {
            std::thread::park_timeout(PARK_TIMEOUT);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_workers(needed: usize) {
    INIT.call_once(|| {
        HANDLES.get_or_init(|| Mutex::new(Vec::new()));
    });
    let _spawn = SPAWN_LOCK.lock().unwrap();
    let handles = HANDLES.get().expect("team handles");
    while SPAWNED.load(Ordering::Relaxed) < needed {
        let t = SPAWNED.load(Ordering::Relaxed) + 1;
        // Baseline = gen at spawn time, captured before the caller can
        // publish its fan-out (ensure_workers returns first), so the new
        // worker is guaranteed to observe the next gen bump.
        let baseline = GEN.load(Ordering::Acquire);
        let jh = std::thread::Builder::new()
            .name(format!("laya-worker-{t}"))
            .spawn(move || worker(t, baseline))
            .expect("spawn engine worker");
        let thread = jh.thread().clone();
        let mut v = handles.lock().unwrap();
        while v.len() < t {
            v.push(thread.clone());
        }
        SPAWNED.store(t, Ordering::Relaxed);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn worker(t: usize, mut seen: usize) {
    loop {
        let g = GEN.load(Ordering::Acquire);
        if g != seen {
            seen = g;
            let active = ACTIVE.load(Ordering::Acquire);
            if t < active {
                // SAFETY: gen was published after the coordinator wrote JOB
                // (Release/Acquire pair), and the coordinator cannot return
                // from fanout until this worker's fetch_add lands.
                unsafe {
                    let desc = &*JOB.0.get();
                    (desc.run)(desc.ctx, t);
                }
                DONE.fetch_add(1, Ordering::Release);
            }
            continue;
        }
        // SeqCst RMW orders this before the gen recheck: the coordinator's
        // SeqCst gen store / parked scan pair then guarantees any worker
        // that still sees the old gen here is included in that scan's
        // unpark set.
        PARKED.fetch_or(1u64 << (t - 1), Ordering::SeqCst);
        let g = GEN.load(Ordering::Acquire);
        if g == seen {
            // unpark tokens make this race-free: an unpark that landed
            // between the recheck above and the sleep returns immediately.
            std::thread::park();
        }
        PARKED.fetch_and(!(1u64 << (t - 1)), Ordering::SeqCst);
    }
}
