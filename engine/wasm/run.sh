#!/bin/bash
# Build/bench wrapper for the engine's wasm32 SIMD128 tier (ticket 19).
#
# RUSTFLAGS carries the getrandom custom-backend cfg (the CI wasm tier's
# recipe; currently inert — the engine dep tree has no getrandom — kept so
# the gate cannot rot when a future dep does).
#
# Usage:
#   ./run.sh build                 wasm32-unknown-unknown release build
#   ./run.sh selftest              engine self-test inside wasm (fast)
#   ./run.sh native                regenerate native engine reference dumps
#                                  (decide --raw, thread-invariant outputs)
#   ./run.sh bench                 full benchmark: Laya F16, OpenThai F16 +
#                                  Q8_0 (both drift gates), dual, scalar A/B
#   ./run.sh smoke                 selftest + one Laya forward vs native dump
set -e
cd "$(dirname "$0")"
CARGO=/home/jame/.cargo/bin/cargo
WASM=target/wasm32-unknown-unknown/release/laya_engine_wasm.wasm
ENGINE=../target/x86_64-unknown-linux-gnu/release/laya-engine
MODELS=../../models
RUSTFLAGS='--cfg getrandom_backend="custom"'

case "${1:-bench}" in
build)
    RUSTFLAGS="$RUSTFLAGS" "$CARGO" build --release --target wasm32-unknown-unknown
    ;;
selftest)
    node harness.mjs --self-test
    ;;
native)
    mkdir -p native results
    [ -x "$ENGINE" ] || (cd .. && ./run.sh build)
    "$ENGINE" decide "$MODELS/laya_english_f16.gguf" \
        --request ../../examples/refund.json --raw > native/laya_refund_f16.json
    "$ENGINE" decide "$MODELS/openthai_systemone_v03_f16.gguf" \
        --request ../../examples/openthai-thai.json --raw > native/openthai_thai_f16.json
    "$ENGINE" decide "$MODELS/openthai_systemone_v03_q8_0.gguf" \
        --request ../../examples/openthai-thai.json --raw > native/openthai_thai_q80.json
    echo "native dumps: $(ls native/*.json | xargs -n1 basename | tr '\n' ' ')"
    ;;
smoke)
    # CI-sized check: self-test + ONE Laya forward, drift vs the native dump
    node harness.mjs --self-test
    [ -f native/laya_refund_f16.json ] || ./run.sh native
    node harness.mjs --model laya --gguf "$MODELS/laya_english_f16.gguf" \
        --native native/laya_refund_f16.json --warmup 0 --repeats 1 \
        --out results/smoke_laya.json >/dev/null
    node -e '
        const r = require("./results/smoke_laya.json");
        const run = r.runs[0];
        if (run.forwards.trapped) { console.error("trapped:", run.forwards.trapped); process.exit(1); }
        const d = run.drift_vs_native;
        console.log(`smoke: forward ${run.forwards.inside_ms[0]} ms, max|dlogit| ${d.max_dlogit.toExponential(2)}, dp ${d.dp_after_temp.toExponential(2)}`);
        if (d.max_dlogit > 0.05 || d.dp_after_temp > 0.02) { console.error("FAIL: drift gate breach"); process.exit(1); }
        console.log("wasm smoke: PASS");
    '
    ;;
bench)
    ./run.sh build
    ./run.sh native
    node harness.mjs --model laya --gguf "$MODELS/laya_english_f16.gguf" \
        --native native/laya_refund_f16.json --repeats 5 --out results/laya_f16.json
    node harness.mjs --model openthai --gguf "$MODELS/openthai_systemone_v03_f16.gguf" \
        --native native/openthai_thai_f16.json --repeats 5 --out results/ot_f16.json
    node harness.mjs --model openthai --gguf "$MODELS/openthai_systemone_v03_q8_0.gguf" \
        --native native/openthai_thai_f16.json --native2 native/openthai_thai_q80.json \
        --repeats 5 --out results/ot_q80.json
    node harness.mjs --model dual --gguf "$MODELS/laya_english_f16.gguf" \
        --gguf2 "$MODELS/openthai_systemone_v03_q8_0.gguf" \
        --native native/laya_refund_f16.json --native2 native/openthai_thai_f16.json \
        --repeats 3 --out results/dual_f16_q80.json
    # scalar-fallback A/B datapoint (simd128 off) for the report
    node harness.mjs --model laya --gguf "$MODELS/laya_english_f16.gguf" \
        --native native/laya_refund_f16.json --simd 0 --warmup 0 --repeats 2 \
        --out results/laya_f16_scalar.json
    ;;
*)
    echo "usage: $0 build|selftest|native|smoke|bench" >&2
    exit 2
    ;;
esac
