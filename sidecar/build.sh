#!/bin/sh
# Build the GPU sidecar. The root .cargo/config.toml carries no_std link
# flags (-nostartfiles/-nodefaultlibs/-static/--no-dynamic-linker) for the
# Goish CPU binary; cargo APPENDS rustflags across the config hierarchy, so
# a child config cannot neutralize them. RUSTFLAGS (env) is the documented
# override that makes cargo ignore all configured rustflags — required for
# this std binary to link a normal crt0.
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
sh "$DIR/kernels/build.sh"
cd "$DIR"
RUSTFLAGS="-Cforce-frame-pointers=yes" cargo build --release "$@"
