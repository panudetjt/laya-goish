#!/bin/bash
# Build/run wrapper for the candle-laya prototype.
#
# The repo root .cargo/config.toml carries the no_std engine's link flags
# (-nostartfiles -nodefaultlibs ...). Cargo concatenates target rustflags
# across config levels, so a local config cannot remove them; a prototype
# built with them has entry point 0x0 and segfaults at startup. Setting
# CARGO_ENCODED_RUSTFLAGS replaces ALL config rustflags.
set -e
cd "$(dirname "$0")"
export CARGO_ENCODED_RUSTFLAGS="-Ctarget-cpu=native"
BIN=target/x86_64-unknown-linux-gnu/release/candle-laya
if [ "$1" = "build" ]; then
    exec /home/jame/.cargo/bin/cargo build --release
fi
if [ ! -x "$BIN" ]; then
    /home/jame/.cargo/bin/cargo build --release
fi
exec "$BIN" "$@"
