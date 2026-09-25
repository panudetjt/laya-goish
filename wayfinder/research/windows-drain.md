# Windows drain wiring — findings (ticket 23)

Status: **DONE (compile-gated)** — 2026-09-25, i5-14400 / WSL2 (Linux), rust 1.96.
The Windows tier of the std engine now captures Ctrl+C / console-close
through the same drain state machine as unix. Zero new dependencies
(extern declarations against kernel32, which std already links on
windows-msvc and windows-gnu); the unix path is byte-identical to before
(diff confined to the former `#[cfg(windows)]` stub region of
`engine/src/http.rs`). Runtime verification on real Windows was **not
possible on this machine** — the honest bar per the ticket: cross-compile
gates green + the manual checklist below for the first Windows machine.

## The wiring

One seam, three cfg bodies. `http::serve` builds the `on_drain` closure
once — `shared.draining.store(true, Release)` then the drain log line
("laya HTTP: draining requests (30 second deadline)") — and hands it to
`install_signal_drain`, which has one body per OS family:

| Target | `install_signal_drain` body |
|---|---|
| unix | self-pipe: raw `signal()` handler does one `write(2)`; a `laya-drain` thread blocked in `read(2)` runs `on_drain` (unchanged since ticket 13) |
| **windows (new)** | `SetConsoleCtrlHandler(Some(on_ctrl_event), 1)`: the OS-run console-ctrl handler does one atomic store to `static CTRL_SEEN: AtomicBool`; a `laya-drain` monitor thread polling `CTRL_SEEN` at `ACCEPT_POLL` (5 ms) runs the identical `on_drain` closure |
| other (wasm32) | no-op stub (unchanged; serve path is native-only per ticket 19) |

The new code (engine/src/http.rs, ~35 lines, all `#[cfg(windows)]`):

```rust
#[cfg(windows)]
static CTRL_SEEN: AtomicBool = AtomicBool::new(false);

#[cfg(windows)]
extern "system" fn on_ctrl_event(_ctrl_type: u32) -> i32 {
    CTRL_SEEN.store(true, Ordering::Release);
    1 // TRUE: handled — the default ^C/^Break termination is suppressed.
}

#[cfg(windows)]
fn install_signal_drain<F: Fn() + Send + Sync + 'static>(on_drain: F) {
    extern "system" {
        fn SetConsoleCtrlHandler(handler: Option<extern "system" fn(u32) -> i32>, add: i32) -> i32;
    }
    if unsafe { SetConsoleCtrlHandler(Some(on_ctrl_event), 1) } == 0 {
        return; // failure means no console is attached: no ctrl events
    } // will ever arrive
    std::thread::Builder::new()
        .name("laya-drain".into())
        .spawn(move || loop {
            if CTRL_SEEN.load(Ordering::Acquire) {
                on_drain();
                return;
            }
            std::thread::sleep(ACCEPT_POLL);
        })
        .ok();
}
```

No `#[link]` attribute is needed: std links kernel32 on both Windows
targets, so the symbol resolves exactly like the unix arm's bare
`extern "C" { fn pipe, signal, write, read }` resolve against libc. No
winapi/windows-rs crate, per the ticket constraint.

## Why the design is safe

**The handler does one atomic store — nothing else.** MSDN runs console
ctrl handlers on a thread the OS creates for the event, so this is not a
Unix-style async-signal interrupt context; we keep it to a single
`store(Release)` anyway. No locks, no allocation, no I/O, no console
calls — nothing that can deadlock against a thread mid-`println!` or
holding the loader lock. It is strictly more minimal than the unix raw
handler (which does a load plus a `write` syscall). The static fn has no
captures by construction (Win32 `PHANDLER_ROUTINE` = plain
`extern "system" fn(DWORD) -> BOOL`).

**The 5 ms poll is the wake mechanism — Windows needs no EINTR
machinery.** This is the property ticket 13 built and the reason the
Windows arm is small: the accept loop is a non-blocking
`TcpListener::accept()` that sleeps `ACCEPT_POLL` (5 ms) between
`WouldBlock`s and re-checks `shared.draining` at the top of every
iteration and in both error arms; `read_more` polls connections with a
5 ms read timeout and checks `draining` on every expiry; idle keep-alive
connections return on the next 5 ms tick. So once `CTRL_SEEN` flips and
the monitor runs `on_drain` (setting `shared.draining`), every part of
the server observes it within one 5 ms granule — ^C-to-503 is ≤ ~10 ms
typical (5 ms monitor poll + 5 ms accept/read poll), the same latency
class as the unix self-pipe (instant reader wake + 5 ms accept poll).
Unix needs the self-pipe because the raw handler runs on the kernel
interrupted context and must not touch `Shared`/logging; Windows'
OS-thread delivery would technically allow calling `on_drain` from the
handler, but that would run `io::log` (allocation + stderr I/O) inside
the console-ctrl thread — exactly what MSDN warns handlers must not do
(lengthy operations risk the "process not responding" path and console
lock contention). The monitor thread keeps handler hygiene identical in
spirit to unix: raw context minimal, real work on our own thread.

**Failure modes.** If `SetConsoleCtrlHandler` returns 0 there is no
console (detached/service context) — no ctrl events exist, so doing
nothing is correct. If the monitor `spawn` fails, ^C is swallowed but
drain never starts and the process keeps serving — the identical
tradeoff the unix arm already makes with its own `.ok()` (self-pipe
write succeeds, nobody reads it). Escape hatch in both cases: hard kill
(`taskkill /F` on Windows, `kill -9` on unix). A second ^C is also
swallowed (handler returns TRUE again, monitor already exited) — same as
unix's second SIGINT.

## The shared-state-machine claim (not a fork)

Everything downstream of `on_drain` is cfg-neutral, single-source code —
the Windows arm adds **zero** drain logic:

1. `on_drain` (built once in `http::serve`, http.rs:936-940): sets
   `Shared::draining`, logs the drain line — identical closure object on
   both OSes.
2. In-flight protection: `Shared::active` enter/leave around each
   request; `serve` drops the listener first, then waits
   `DRAIN_TIMEOUT` (30 s) for `active == 0`, `Err` past the deadline.
3. 503s: `/health` reports `"status":"draining"` / 503 (server.rs:131),
   new non-public requests get 503 `unavailable_error` (server.rs:173),
   keep-alive connections close after their current response
   (http.rs:843, :907), idle connections return at the next 5 ms poll
   (http.rs:872).
4. Exit: drain completes → `serve` → `Ok` → `main` → **exit 0**;
   deadline exceeded → `Err("HTTP shutdown deadline exceeded")` → **exit 1**.

`git diff` confirms only the stub region of http.rs changed (+37/−8);
the unix self-pipe arm, the accept loop, `read_more`, and the drain
deadline are byte-identical. The wasm stub is untouched.

## Ctrl-type mapping (what the handler sees)

| Event | Value | Delivery | Default if unhandled | Our mapping |
|---|---|---|---|---|
| `CTRL_C_EVENT` | 0 | new OS thread | process terminated | **graceful drain** — handler returns TRUE, default suppressed, full 30 s drain machine, exit 0 |
| `CTRL_BREAK_EVENT` | 1 | new OS thread | process terminated | **graceful drain** (same as C) |
| `CTRL_CLOSE_EVENT` | 2 | console-close thread | process terminated after handler returns (~5 s force window) | **best-effort, unavoidable termination** — flag is set, but the handler returns in nanoseconds so the OS ends the process almost immediately; in-flight requests should not be expected to complete |
| `CTRL_LOGOFF_EVENT` | 5 | new OS thread (not delivered if user32/gdi32 loaded — our binary loads neither) | session ends | **best-effort, unavoidable** — session teardown governs, not our deadline |
| `CTRL_SHUTDOWN_EVENT` | 4 | new OS thread (same user32 caveat) | system shuts down | **best-effort, unavoidable** |

Design note on CLOSE: blocking inside the handler to buy drain time was
rejected — it violates the minimal-handler discipline (the whole safety
argument above), hangs the console close dialog, and the OS force-kills
at ~5 s regardless. C/BREAK — the signals operators actually use to
stop a foreground server — get the full graceful path.

## Compile evidence (this machine: Linux/WSL2 — no runtime Windows claims)

| Gate | Command | Result |
|---|---|---|
| CI cross tier (Windows) | `cd engine && cargo check --target x86_64-pc-windows-msvc` | **PASS** — 0 errors, 1 warning (pre-existing, see below) |
| Apple silicon | `cd engine && cargo check --target aarch64-apple-darwin` | **PASS** |
| unix path unchanged | `./engine/ci/run.sh quick` (build + smoke + reduced drift) | **PASS** — drift bit-identical (4 combos) |
| fmt + clippy | `./engine/ci/run.sh lint` | **PASS** (`-D warnings` with the ticket-13 allow-list) |
| wasm compile | `RUSTFLAGS='--cfg getrandom_backend="custom"' cargo check --target wasm32-unknown-unknown` | **PASS** — 0 errors (pre-existing wasm-target warnings in tensor.rs only) |

Pre-existing, out-of-scope observations (not introduced by this ticket):
- `cargo check --target x86_64-pc-windows-msvc` warns `unused_variables: s`
  in `Cancel::canceled` (http.rs:711) — the `#[cfg(not(unix))]` arm of the
  client-disconnect probe returns `false` without using the binding. Warning
  only; existed since ticket 13 on this target.
- `Cancel::canceled` is a no-op on Windows (the `poll(2)`-based peer-close
  probe is unix-only), so client-disconnect-cancels-inference does not fire
  there. That is the adjacent "serving hardening on Windows beyond drain"
  item in the map's not-yet-specified list — deliberately not smuggled into
  this ticket.

## Manual test checklist — first Windows machine

Build there first: install the MSVC toolchain (rustup default stable
x86_64-pc-windows-msvc + VS Build Tools), then from `engine/`:
`cargo build --release` — produces `target/release/laya-engine.exe`.
Download the f16 model (gguf) to e.g. `models\laya_english_f16.gguf`.
Run each check from a separate terminal (PowerShell or cmd with
`curl.exe`); server command used throughout:

```
laya-engine.exe serve models\laya_english_f16.gguf --port 8080 --threads 4
```

1. **Serve + baseline**: server starts, logs
   `laya HTTP listening at http://127.0.0.1:8080 (bearer auth: off)`;
   `curl.exe http://127.0.0.1:8080/health` → 200 `"status":"ok"`.
2. **Ctrl+C starts graceful drain with in-flight request**: fire a
   decide (`curl.exe -s http://127.0.0.1:8080/v1/decide -H
   "Content-Type: application/json" -d @refund.json`), press Ctrl+C in
   the server console mid-inference. Expect: stderr line
   `laya HTTP: draining requests (30 second deadline)` within ~10 ms;
   the in-flight request still completes 200 (with X-Request-Id); server
   process then exits with **code 0** (`echo %errorlevel%` / `$LASTEXITCODE`).
3. **503 during drain**: same setup, but make the in-flight request a
   large batch (long enough to poll health while draining). During
   drain: `curl.exe http://127.0.0.1:8080/health` → 503
   `"status":"draining"`; a new `POST /v1/decide` → 503
   `unavailable_error "Server is shutting down"`. Both only while the
   in-flight request runs; after exit, connections are refused
   (listener already closed at drain start — check 2's curl also
   confirms nothing new is accepted).
4. **Ctrl+Break**: repeat check 2 with Ctrl+Break — identical graceful
   drain (both map to the full machine).
5. **Second Ctrl+C swallowed**: press Ctrl+C twice during a long drain —
   process is NOT terminated by the second event; it still exits 0 when
   the in-flight request finishes (or 1 at the 30 s deadline).
6. **Close window (expectation-setting, not a drain test)**: start a
   request, close the console window. Expect the process to die
   essentially immediately (CTRL_CLOSE grace window, not our deadline);
   in-flight request does not complete. This is documented-unavoidable;
   operators should stop the server with Ctrl+C.
7. **No console**: run via `start /b` or Task Scheduler (no console) —
   server must still serve normally (no ctrl events exist; drain via
   `taskkill /PID <pid>` is a hard kill, exit not 0).
8. **Exit code on deadline**: optional stress case — start a
   multi-minute batch, Ctrl+C, and if inference outlives 30 s expect
   exit code 1 with the deadline error path (unix behavior parity).

Record results here (append a dated section) and flip the ticket's
runtime-verification caveat when checks 1–5 pass.

## Files changed

| File | Change |
|---|---|
| `engine/src/http.rs` | `#[cfg(windows)]` stub → real arm: `CTRL_SEEN` static, `on_ctrl_event` handler, `SetConsoleCtrlHandler` extern + install, `laya-drain` monitor (+37/−8; unix arm untouched) |
| `wayfinder/research/windows-drain.md` | this report |
