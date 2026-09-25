#!/bin/bash
# Build/run wrapper for the wasm32 gate prototype (ticket 08).
#
# RUSTFLAGS must carry the getrandom custom-backend cfg for
# wasm32-unknown-unknown (see src/lib.rs); env rustflags REPLACE the repo
# root .cargo/config.toml target rustflags, which are x86_64-only here and
# would otherwise not apply to this target anyway.
set -e
cd "$(dirname "$0")"
WASM=target/wasm32-unknown-unknown/release/wasm_gate.wasm
case "${1:-}" in
build)
    RUSTFLAGS='--cfg getrandom_backend="custom"' /home/jame/.cargo/bin/cargo build --release --target wasm32-unknown-unknown
    ;;
probe)
    (cd probe && RUSTFLAGS='--cfg getrandom_backend="custom"' /home/jame/.cargo/bin/cargo build --release --target wasm32-unknown-unknown)
    node probe/probe.mjs
    ;;
*)
    if [ ! -f "$WASM" ]; then
        RUSTFLAGS='--cfg getrandom_backend="custom"' /home/jame/.cargo/bin/cargo build --release --target wasm32-unknown-unknown
    fi
    shift 2>/dev/null || true
    exec node harness.mjs "$@"
    ;;
esac
