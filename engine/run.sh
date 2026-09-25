#!/bin/bash
# Build/run wrapper for the std engine crate.
#
# The repo root .cargo/config.toml carries the no_std engine's link flags
# (-nostartfiles -nodefaultlibs ...). Cargo concatenates target rustflags
# across config levels, so a local config cannot remove them; a std crate
# built with them has entry point 0x0 and segfaults at startup. Setting
# CARGO_ENCODED_RUSTFLAGS replaces ALL config rustflags (same pattern as
# prototypes/candle-laya/run.sh).
#
# Usage:
#   ./run.sh build [gnu|musl]   build release (default gnu)
#   ./run.sh BIN [args...]      build if needed, then exec BIN with args
set -e
CALLER_PWD="$PWD"
cd "$(dirname "$0")"
MODE="${1:-run}"
TRIPLE="x86_64-unknown-linux-gnu"
if [ "$MODE" = "build" ] && [ "${2:-gnu}" = "musl" ]; then
    TRIPLE="x86_64-unknown-linux-musl"
fi
# CARGO_ENCODED_RUSTFLAGS replaces ALL config rustflags (same pattern as
# prototypes/candle-laya/run.sh). NOTE: keep -Ctarget-cpu=x86-64 (the crate
# baseline, matching how ./laya is built): -Ctarget-cpu=native costs ~10%
# on this workload even for the goish crate itself (measured, ticket 12) —
# the AVX2/FMA/F16C kernels are target_feature-gated and do not need it.
export CARGO_ENCODED_RUSTFLAGS="-Ctarget-cpu=x86-64"
CARGO=/home/jame/.cargo/bin/cargo
BIN="$PWD/target/$TRIPLE/release/laya-engine"
if [ "$MODE" = "build" ]; then
    exec "$CARGO" build --release --target "$TRIPLE"
fi
if [ ! -x "$BIN" ]; then
    "$CARGO" build --release --target "$TRIPLE"
fi
cd "$CALLER_PWD"
exec "$BIN" "$@"
