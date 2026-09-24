---
wayfinder: research
ticket: 03-goish-portability
date: 2026-09-24
---

# goish portability audit (today, not folklore)

**Scope.** This is a go/no-go audit of the incumbent stack — this codebase plus `goish = "=1.0.0-alpha.14"` (`Cargo.toml:8`) — against the owner's deploy targets: **wasm32, Linux, Windows, macOS, Docker**. Evidence comes from (a) the local codebase, read file-by-file, and (b) upstream goish primary sources: the published crate `1.0.0-alpha.14` (the exact bytes in the local cargo registry cache, `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/goish-1.0.0-alpha.14/`), the upstream repo [cogentica-ai/goish](https://github.com/cogentica-ai/goish) at tag [`v1.0.0-alpha.14`](https://github.com/cogentica-ai/goish/releases/tag/v1.0.0-alpha.14) (commit `28e1f88`), its CI workflows, docs.rs build, and issue tracker. All sources accessed 2026-09-24. No toolchains were installed; no builds were run (probe constraint honored — see §7).

**Note on provenance:** the repo's `upstream/` directory holds only OpenThai-SystemOne model artifacts (`config.json`, `model.safetensors`, `tokenizer.json`) — it does **not** vendor goish. The goish source cited locally is the registry cache copy of the pinned version, which is what actually compiles into the release binary.

---

## 1. Verdict matrix

| Target | Verdict | One-line reason |
| --- | --- | --- |
| Linux x86-64 | **Works today** (shipped, v0.1.0) | The only designed, CI'd, documented target — codebase and goish are built for it. |
| Docker `linux/amd64` on AVX2-capable host | **Works today** | Static binary, `debian:bookworm-slim` base, no runtime deps (`Dockerfile:8-19`). |
| Docker on arm64 hosts (Apple Silicon, emulated amd64) | **Conditional, fragile — not a ship path** | No native arm64 build exists; emulation must expose AVX2/FMA/F16C, which the repo's own docs flag as unverified in practice (`docs/getting-started.md:39-42`). |
| Windows (MSVC or GNU) | **No path short of an upstream runtime port** | goish alpha.14 is raw-Linux-syscall + x86-64 naked asm, zero cfg fallbacks; repo link flags are GNU-ld-only. |
| macOS (x86_64 or aarch64) | **No path on the pinned alpha.14** | goish has no Darwin support in this release; the arm64/Darwin port (upstream PR #31) merged *after* alpha.14 and is still gap-ridden (#33, #35). |
| wasm32 (`unknown-unknown` or WASI) | **No path short of a rewrite** | No threads/signals/mmap/EPOLL surface on wasm for goish's scheduler; local x86_64 SIMD kernels don't even compile there. |

**Bottom line for the go/no-go:** staying on goish satisfies **2 of the owner's 5 deploy targets** (Linux, Docker-amd64) today. The incumbent's portability must not be assumed because it does not exist: upstream goish declares itself single-target in its README, and this repo's kernels, link flags, and I/O layer are all hard-coupled to that target.

---

## 2. Local hard gates (file:line evidence)

### 2.1 AVX2/FMA/F16C startup check

`src/tensor.rs:423-435` — `require_simd()` gates all inference on x86_64 CPUID + XGETBV:

```rust
pub fn require_simd() -> Result<(), String> {
    let f = core::arch::x86_64::__cpuid(1);
    let f7 = core::arch::x86_64::__cpuid_count(7, 0);
    if f.ecx & (1 << 12) == 0        // FMA
        || f.ecx & (1 << 27) == 0    // OSXSAVE
        || f.ecx & (1 << 29) == 0    // F16C
        || f7.ebx & (1 << 5) == 0    // AVX2
        || unsafe { core::arch::x86_64::_xgetbv(0) } & 6 != 6 {
        return Err("CPU requires AVX2/FMA/F16C and OSXSAVE".into());
    }
```

- Enforced at **startup before weights load**: `src/engine.rs:18` (`Engine::new`), `src/models/laya/model.rs:165` (`Model::new`), and `src/tensor.rs:437` (`self_test`).
- The kernels themselves are unconditional x86_64 intrinsics: `use core::arch::x86_64::*;` (`src/tensor.rs:2`); `#[target_feature(enable = "avx2,fma,f16c")]` matmul kernel with `_mm256_fmadd_ps` / `_mm256_cvtph_ps` (F16) / Q8 expand (`src/tensor.rs:152,210-247`); `#[target_feature(enable = "avx2,fma")]` norm kernel (`src/tensor.rs:291,310-320`) and dot kernel (`src/tensor.rs:405-422`).
- **No fallback path exists**: `dot()` calls `dot_kernel()` unconditionally (`src/tensor.rs:399-402`); the only scalar code is the k-remainder tail *inside* the AVX kernel (`src/tensor.rs:249-253`). A grep for `cfg(` across `src/` returns only the three `require_simd` call sites — there are **zero `cfg(target_arch)` gates** in this codebase. On any non-x86_64 target the crate does not even *compile* (`core::arch::x86_64` is target-gated by rustc), so the startup check is a runtime guard on x86_64-only hardware, not a portability shim.
- This design is deliberate, per `docs/implementation.md:41-43`: "The target has no global AVX/native-CPU compilation flag: AVX2/FMA is enabled only on the kernel function. The application verifies CPU and OS AVX support before loading inference weights."

### 2.2 no_std + no_main freestanding setup

- `src/main.rs:1-3` — `#![no_std]`, `#![no_main]`, `extern crate alloc;`. No `std::` usage anywhere in `src/` (grep: 0 matches). No `#[panic_handler]` in `src/` — goish provides it (see §3).
- `Cargo.toml:5` — package description: "Laya GGUF decision inference in no_std Goish Rust".
- `Cargo.toml:8` — `goish = "=1.0.0-alpha.14"` (exact pin; also exact in `Cargo.lock`). All other deps are `default-features = false` alloc-only: `libm` (:9), `serde` (:10), `serde_json` (:11), `unicode-normalization` (:13), `unicode-general-category` (:14).
- `Cargo.toml:16-19` — `panic = "abort"` in **both** dev and release profiles (removes the `eh_personality` sysroot requirement); release adds `lto="thin"`, `codegen-units=1`, `strip="symbols"` (:20-22).
- `.cargo/config.toml:2` — default target hard-pinned: `target = "x86_64-unknown-linux-gnu"`.
- `.cargo/config.toml:4` — freestanding static link via **GNU-ld-only** flags: `-nostartfiles -nodefaultlibs -static -Wl,--no-dynamic-linker -C relocation-model=static -C force-frame-pointers=yes`. These args are ELF/GNU-binutils shaped; MSVC `link.exe` and Apple `ld64` reject the `-Wl,--...` forms outright.
- No root `build.rs`, no `build-std`, no custom sysroot config — freestanding-ness comes entirely from goish's runtime (`_start` + `#[panic_handler]`) plus panic=abort.
- Toolchain pin: `mise.toml:5-6` — `rust = "1.96.0"` ("Pin the Rust stable toolchain used to build the release binary"); goish itself declares `rust-version = "1.86"`.

### 2.3 goish API usage inventory (what the codebase actually leans on)

| Surface | Call sites | Runtime dependency |
| --- | --- | --- |
| `#[goish::main]`, `os::Exit`, `runtime::GOMAXPROCS` | `src/main.rs:24,28,94` | goish `_start` asm entry, M:N scheduler sizing |
| `go!` + `sync::WaitGroup` goroutine fan-out | `src/tensor.rs:3,112-128` (go! at :120); `src/models/laya/model.rs:12,371-385` (go! at :377); `src/models/openthai/model.rs:9,394,419` | stackful goroutines multiplexed over OS worker threads, futex parking, SIGURG async preemption |
| Channels `goish::make!(chan bool,1)` + `Recv`/`Send` | `src/server.rs:374,378,385,394` | scheduler + parking |
| `goish::net` + `net::http` server | `src/server.rs:15-19,362-369` (`net::Listen("tcp", addr)`, `JoinHostPort`), `:355-359` (timeouts), `:392` (`errors::Is(..., http::ErrServerClosed)`) | goish TCP stack over raw Linux socket syscalls + **EPOLL** netpoller |
| `os::signal::NotifyContext(SIGINT, SIGTERM)` | `src/server.rs:370-373` | **Unix signals** (`goish::syscall::SIGINT/SIGTERM`) |
| `goish::time::Now/Since/Second` | `src/models/laya/decision.rs:183,294`; `src/models/openthai/decision.rs:250,378`; `src/server.rs:355-357` | clock syscalls, runtime timers |
| `goish::io::ReadAll` | `src/server.rs:204` | blocking read via netpoller |
| Raw `goish::syscall` | `src/io.rs:5-13` (`Write`), `:39` (`Read` on stdin) | raw fd syscalls |
| `goish::os::{ReadFile, Args}` | `src/io.rs:25,32` | file + argv syscalls |
| `goish::strconv::FormatFloat` | `src/json.rs:84` | pure |
| **mmap model loading** | `src/io.rs:56-97` — `syscall::Open` (:71), `syscall::Fstat` (:76), `syscall::Mmap(null, n, 1, 2, fd, 0)` = PROT_READ/MAP_PRIVATE (:82), `syscall::Munmap` (:95); errno range check `-4095..-1` (:84) is the **Linux kernel errno convention**; consumed by `Gguf::open` (`src/gguf.rs`) | **mmap(2)** — no std::fs, no buffered loader |

### 2.4 Platform-specific assumptions in the codebase

- Pointer round-trips through `usize` inside `go!` closures: `src/tensor.rs:104-111`, `src/models/laya/model.rs:367-368`, `src/models/openthai/model.rs:385-410` — width-neutral in principle (usize == pointer width per target) but deep unsafe coupling to the goish ABI.
- Atomics: `AtomicBool`/`AtomicU64` for server state (`src/server.rs:14,28-30`) — fine on all 64-bit targets, needs wasm-threads + SharedArrayBuffer on wasm32.
- No inline `asm!`, no `transmute` in `src/` — all machine coupling lives in `tensor.rs` intrinsics and inside goish itself.
- `sidecar/` is a separate std-based PoC crate (GPU probe), Linux-only by construction: `std::os::unix::net::UnixListener` (`sidecar/src/main.rs:15,146`), links `libcuda`/`-ldl` with a WSL2 path (`sidecar/build.rs:4-14`). Not part of the main binary; irrelevant to the main portability verdict.

### 2.5 What the repo's own docs claim

- `README.md` — "A single Linux executable runs inference on your CPU." / "Requires Linux x86-64 with AVX2/FMA."
- `docs/getting-started.md:5-6` — "The v0.1.0 executable requires Linux x86-64 with AVX2 and FMA. It is statically linked…".
- `docs/getting-started.md:39-42` — "this release has no native arm64 build. On an arm64 host, the emulation layer must expose these CPU features to the container. Setting `platform: linux/amd64` alone does not verify that requirement."
- `docs/implementation.md:31-34` — "Goish `1.0.0-alpha.14` preserves OS-enabled AVX and AVX-512 state with `xsave64` / `xrstor64` during asynchronous preemption." (x86_64-only instructions, inside the *scheduler*.)
- `docs/openthai.md:35` — "The existing Goish dependency and Linux x86-64 AVX2/FMA requirements apply"; `docs/validation.md:3` — "Validated on Linux x86-64…". No doc claims wasm, Windows, or macOS for the main binary; `docs/igpu-tier.md:11-13` mentions Windows only for the experimental GPU sidecar.
- Packaging pins the same single platform: `Dockerfile:3-4,8` ("Packages the statically linked Linux x86-64 release executable. AVX2/FMA must be available to the container, including under emulation."; `FROM --platform=linux/amd64 debian:bookworm-slim`); `docker-compose.yml` `platform: linux/amd64`; `scripts/docker/fetch-release.sh` fetches only the `linux-x86_64` tarball.

**Docs vs code: consistent.** Every claim checks out against the code; nothing in the docs overpromises portability — the folklore risk was in the planning assumptions, not the docs.

---

## 3. Upstream goish `1.0.0-alpha.14` — what it claims and what CI exercises

Upstream repo: [cogentica-ai/goish](https://github.com/cogentica-ai/goish), release tag [`v1.0.0-alpha.14`](https://github.com/cogentica-ai/goish/releases/tag/v1.0.0-alpha.14) (commit `28e1f88`). Verified against both the GitHub sources at that tag and the byte-local registry copy of the published crate.

1. **The README declares single-target Linux x86_64, verbatim** (registry copy `README.md:92,113`; same lines upstream at [README.md#L90-L114](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/README.md#L90-L114)):
   > "**Linux `x86_64` only.** Other targets are out of scope for now."
   > "Goish is single-target: `x86_64-unknown-linux-gnu`."
2. **CI has no target matrix.** `e2e.yml` ([#L23-L41](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/.github/workflows/e2e.yml#L23-L41)), `e2e-race.yml` ([#L34-L55](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/.github/workflows/e2e-race.yml#L34-L55)) and `publish.yml` ([#L35-L46](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/.github/workflows/publish.yml#L35-L46)) all run one `ubuntu-latest` job and install only `x86_64-unknown-linux-gnu`. No Windows, macOS, AArch64, or wasm job exists at this tag.
3. **docs.rs builds only `x86_64-unknown-linux-gnu`** for this release ([docs.rs crate page](https://docs.rs/crate/goish/1.0.0-alpha.14), [builds](https://docs.rs/crate/goish/1.0.0-alpha.14/builds)).
4. **Manifest** ([Cargo.toml](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/Cargo.toml); registry copy lines verified): `description` = "Goish v1 — Go-style stdlib for Rust, no_std and no glibc, built on raw Linux syscalls." (`:32`); `rust-version = "1.86"` (`:14`); categories include `os::linux-apis` (`:43`); features are only `default = []` and `ffi-system-tls` (`:52-54`) — there is **no feature, cfg gate, or backend switch for any other OS/arch**. A grep for `target_arch|target_os` across the crate's `src/` returns **zero matches**: every platform-specific construct is unconditional.
5. **The goroutine runtime is Linux-x86_64 machinery end to end:**
   - `go!` creates a stackful `G` (saved register context `Gobuf`, allocated stack, boxed closure) — it does **not** spawn `std::thread`s ([`builtin_macros.rs#L547-L595`](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/src/builtin_macros.rs#L547-L595), [`sched/g.rs#L32-L60`](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/src/runtime/sched/g.rs#L32-L60)).
   - Worker OS threads are created with **`clone(2)`** ([`sched/scheduler.rs#L1766-L1865`](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/src/runtime/sched/scheduler.rs#L1766-L1865)); run queues/work-stealing at [`scheduler.rs#L141-L180`](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/src/runtime/sched/scheduler.rs#L141-L180).
   - Goroutine context switches are **naked x86-64 assembly** ([`sched/gobuf.rs#L95-L150`](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/src/runtime/sched/gobuf.rs#L95-L150); registry copy confirms `naked_asm!` at `gobuf.rs:117,214,285`).
   - **Async preemption uses SIGURG + Linux `ucontext_t` + x86-64 register indices + CPUID/XGETBV** ([`runtime/preempt.rs#L1-L152`](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/src/runtime/preempt.rs#L1-L59)) — this is the mechanism `docs/implementation.md:31-34` describes from the consumer side (`xsave64`/`xrstor64`).
   - The syscall layer is "raw Linux x86-64 syscalls via inline assembly": `mmap`/`mprotect`/`madvise`, `clone`, `futex`, `rt_sigaction`/`sigaltstack`, sockets + **`epoll`** + `eventfd`, `fork`/`execve`/`wait4`, `arch_prctl` ([`syscall/mod.rs#L1-L100`](https://github.com/cogentica-ai/goish/blob/28e1f88eb68e77eceeb60728ad17166c5c9cf326/src/syscall/mod.rs#L1-L100); registry copy: `use core::arch::asm` at `:37`, `naked_asm!` syscall stubs at `:1669,1794,1852`).
6. **Issue tracker** (searched open + closed, [via GitHub API](https://api.github.com/search/issues?q=repo%3Acogentica-ai%2Fgoish)):
   - [#31 "Port goish to darwin/arm64 (full e2e suite) and linux/arm64 (boot)"](https://github.com/cogentica-ai/goish/pull/31) — **merged after alpha.14**; it is the only multi-platform effort and it postdates the pinned release.
   - [#33 "Track runtime and platform parity gaps in the ARM64/Darwin port"](https://github.com/cogentica-ai/goish/issues/33) — **open**; scheduler, signal-context, and preemption gaps remain.
   - [#35 darwin/arm64 stack-overflow reporting/fixes](https://github.com/cogentica-ai/goish/pull/35) — **open** follow-up.
   - [#27 "lib test target fails with duplicate panic_impl from std and Goish"](https://github.com/cogentica-ai/goish/issues/27) — **open**; even the std-test boundary is unfinished.
   - [#8](https://github.com/cogentica-ai/goish/issues/8) (closed) confirms the net stack at this era was TCP/IPv4-only, no Unix sockets.
   - **Zero wasm-related issues; zero Windows-port issues** in the tracker (title+body search returned nothing relevant).

---

## 4. What no_std + freestanding implies per target (for THIS codebase)

The app itself is clean `core` + `alloc` (`#![no_std]`, `extern crate alloc`, all deps `default-features = false`, `src/main.rs:1-3`, `Cargo.toml:9-14`) with `panic = "abort"` (`Cargo.toml:16-19`), so the *language-level* portability story is better than average. The blockers are concrete and stacked:

- **Entry + panic + allocator**: no `#[panic_handler]`, `_start`, or global allocator exists in `src/` — all three are supplied by goish (`goish::main` macro emits the `_start` asm; `#[panic_handler]` and the mmap-based page allocator live in `runtime/mod.rs`). Any non-Linux target means replacing the entire substrate that `main`, panics, and `alloc` run on.
- **No custom sysroot / build-std needed on Linux** — and none configured. That convenience does not transfer: on wasm/WASI the standard `core`/`alloc` ship with the target, but goish's allocator (mmap of Linux memory) has no counterpart; on Windows-MSVC a libc-free link is possible but needs MSVC-flavored flags, and this repo's `.cargo/config.toml:4` uses GNU-ld-only syntax (`-Wl,--no-dynamic-linker`); on macOS, Apple platforms have no stable raw-syscall ABI (everything goes through libSystem) and Mach-O/ld64 rejects the ELF-specific link args.
- **panic=abort** travels fine everywhere (no unwinding runtime needed) — one of the few per-target non-issues.
- **The I/O and concurrency layers are the real couplings**: model loading assumes `Open/Fstat/Mmap/Munmap` with Linux errno conventions (`src/io.rs:71-95`); the server assumes Unix signals (`SIGINT`/`SIGTERM`, `src/server.rs:370-373`) and an EPOLL-backed TCP stack (`src/server.rs:362-369`); inference parallelism assumes stackful goroutines over `clone(2)` threads with SIGURG preemption (§2.3). wasm32 has none of: threads (without the wasm-threads + SharedArrayBuffer stack), signals, mmap, raw syscalls, or a `clone` equivalent; Windows would need IOCP + Win32 threads; macOS would need kqueue + Mach threads + libSystem syscalls. Each is an operating-system port of goish, not a flag flip.

---

## 5. Probe results (allowed single probe only)

- `rustup target list --installed` → **`x86_64-unknown-linux-gnu` only.** No wasm32 target of any flavor is installed.
- Therefore the optional `cargo check --target <wasm-target>` was **skipped** — running it would have required `rustup target add`, which the ticket forbids. No toolchains were installed, no builds were run.
- Given §2.1 (x86_64 intrinsics compile-unconditionally in `src/tensor.rs`) and §3.4 (zero `cfg` gates in goish), a wasm `cargo check` was never going to get past codegen — its result was not load-bearing for the verdict.

---

## 6. Per-platform verdicts

### Linux x86-64 — **works today**
This is the designed target end to end: repo link config (`.cargo/config.toml:2-4`), goish's only supported triple (§3.1-3.4), shipped v0.1.0 release artifacts, and the validation suite (`docs/validation.md:3`). The AVX2/FMA/F16C gate fails cleanly with an actionable error on older CPUs (`src/tensor.rs:432`) rather than crashing.

### Docker — **works today for `linux/amd64` on AVX2-capable hosts; conditional/fragile otherwise**
- amd64: static freestanding binary on `debian:bookworm-slim` (`Dockerfile:8-19`) — no runtime library coupling; the only environmental requirement is that the *host CPU* (and any emulation layer) exposes AVX2/FMA/F16C + OSXSAVE.
- arm64 hosts (Apple Silicon): **no native arm64 build exists** (`docs/getting-started.md:39`), so this path is amd64 emulation, and Rosetta/QEMU AVX support is exactly what the repo docs refuse to promise (`:40-42`). Treat as unsupported until an arm64 build exists — which, per §3.5-3.6, requires an upstream goish port that postdates alpha.14 and is still gap-ridden.
- Docker-wasm: presupposes the wasm build that §6.5 shows has no path.

### Windows (MSVC or GNU) — **no path short of an upstream runtime port**
Everything is stacked against it with zero escape hatches: goish alpha.14 has no Windows code path at all (no cfg gates, no features, no CI, no tracker activity — §3.4-3.6), and its essentials (`clone(2)` threads, SIGURG preemption, EPOLL, Linux raw syscalls via `syscall` instruction, x86-64 `ucontext_t`) have no Windows equivalents — they'd each be written from scratch. Locally, `.cargo/config.toml:4` link flags are GNU-ld-only and would need a parallel MSVC configuration even if the runtime existed. This is an OS-port project inside goish, not bounded work on this repo. Notably, upstream's post-alpha.14 multi-platform energy went to Darwin/arm64 — no Windows effort is visible anywhere.

### macOS (x86_64, aarch64) — **no path on the pinned alpha.14; a possible-but-unproven bounded path only via a future goish upgrade**
With `=1.0.0-alpha.14`: upstream explicitly claims x86_64-Linux-only (§3.1), has no Darwin CI or code (§3.2), and raw Linux syscalls/naked asm/SIGURG don't exist on Darwin (§3.5). Verdict today: no path. Forward-looking (for the map, not this release): upstream PR #31 (merged *after* alpha.14) ports darwin/arm64 + linux/arm64, and issues #33/#35 track the remaining parity and stack-overflow gaps. Even once a post-#31 goish release is usable, this repo still needs: an `aarch64` rewrite of the AVX2/FMA/F16C kernels with a startup fallback (`src/tensor.rs` has none), ld64-compatible link flags replacing `.cargo/config.toml:4`, and full revalidation. So macOS is "needs bounded-ish work, contingent on upstream we don't control" — not a today target, and x86_64-macOS has no upstream effort at all.

### wasm32 (`unknown-unknown` / WASI) — **no path short of a rewrite**
- **Upstream**: goish's README declares other targets out of scope (§3.1); there is no wasm cfg, feature, target, CI job, docs.rs target, or issue (§3.2-3.6). The runtime's foundations — stackful goroutines context-switched by x86-64 naked asm, `clone(2)` worker threads, SIGURG-based async preemption, EPOLL netpoller, futex parking, mmap allocator — have **no wasm32-unknown-unknown equivalent at all** (no threads without wasm-threads + SharedArrayBuffer/COOP-COEP, no signals, no raw syscalls, no mmap) and only partial WASI analogues (wasi-threads, no signals, no usable file-backed mmap). Porting goish to wasm means re-implementing its scheduler, allocator, and net stack against a different concurrency model — a rewrite of goish, not a port.
- **Local**: `src/tensor.rs:2` imports `core::arch::x86_64::*` unconditionally with zero `cfg` gates — the crate fails to *compile* for wasm32 before any runtime question arises; the 256-bit AVX2/FMA/F16C kernels have no SIMD128/scalar fallback path (`dot()` calls the AVX kernel unconditionally, `src/tensor.rs:399-402`). Model loading via `Mmap` (`src/io.rs:82`) has no wasm counterpart and would need a streaming loader. 
- **Consistent with the map's own structure**: the wasm tickets (04, 05-07, 08) evaluate candle/burn/GDN prototype stacks as replacements — i.e., the map already treats wasm as "different framework," and this audit confirms the incumbent can't be carried there.
- Cross-reference: the memory-side constraints for any future wasm attempt are covered separately in [`wasm-constraints.md`](wasm-constraints.md) (ticket 04).

---

## 7. Caveats

1. **Pin is load-bearing.** `Cargo.toml:8` pins `=1.0.0-alpha.14` exactly. Upstream `main` contains post-release platform work (PR #31 etc.); none of it is in the shipped dependency, and issue #33 shows that work is not parity-complete. If the map later re-opens "stay on goish," re-audit against the then-current alpha — this document is a snapshot of alpha.14, not of the project.
2. **Evidence provenance.** Upstream claims were verified twice: once at the GitHub tag (URLs in §3) and once against the byte-local registry copy of the published crate (§2-§3 quotes like README lines 92/113, `rust-version`, zero-cfg-gate grep). The repo's `upstream/` dir is model weights only, not goish source.
3. **The probe was deliberately not run to completion** (no wasm target installed; installing is forbidden). The compile-time blockers cited (§2.1, §3.4) are deterministic from source and do not depend on the probe.
4. **Not audited here:** runtime *correctness* of goish's scheduler under load (race/preemption behavior), binary size, or performance — all out of scope for portability. The repo's own repeatability regression (`docs/implementation.md:45-48`) is the guard for the alpha.13→alpha.14 preemption change on the supported target.
