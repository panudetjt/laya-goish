//! Library target for the engine core (wayfinder ticket 14).
//!
//! The shipped artifact is the CLI bin (src/main.rs — unchanged); this lib
//! mirrors its module list 1:1 so the cross-compile gates in engine/ci/run.sh
//! (`wasm`, `cross` tiers) can compile the same core for wasm32 and non-x86
//! targets without touching the bin. Keep this mod list in sync with main.rs.
//!
//! wasm32 also needs the getrandom custom-backend recipe proven in
//! prototypes/wasm-gate (RUSTFLAGS='--cfg getrandom_backend="custom"'; the
//! engine's dep tree does not currently pull getrandom, but the cfg is set
//! by the CI tier so the gate cannot rot when a future dep does).
extern crate alloc;
#[macro_export]
macro_rules! format {
 ($($arg:tt)*) => {{
  let mut out = alloc::string::String::new();
  core::fmt::Write::write_fmt(&mut out, core::format_args!($($arg)*)).expect("string formatting");
  out
 }};
}
pub mod engine;
pub mod gguf;
pub mod gofmt;
pub mod heap;
pub mod http;
pub mod io;
pub mod json;
pub mod models;
pub mod registry;
pub mod request;
pub mod server;
pub mod team;
pub mod tensor;
pub mod tokenizer;
