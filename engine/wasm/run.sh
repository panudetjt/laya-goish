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
WASM_THREADS=target-threads/wasm32-unknown-unknown/release/laya_engine_wasm.shared.wasm
ENGINE=../target/x86_64-unknown-linux-gnu/release/laya-engine
MODELS=../../models
RUSTFLAGS='--cfg getrandom_backend="custom"'
# Threads artifact (ticket 24): atomics codegen + imported memory + the
# wasm_threads cfg gating the host-driven worker fan-out (team.rs) and the
# serialized small-block allocator path (heap.rs). patch-shared.mjs then
# marks the memory import shared (lld only does that under --shared, which
# needs PIC std / nightly).
RUSTFLAGS_THREADS="$RUSTFLAGS --cfg wasm_threads -Ctarget-feature=+atomics,+bulk-memory,+mutable-globals -Clink-arg=--import-memory -Clink-arg=--max-memory=4294967296 -Clink-arg=--export=__stack_pointer"

case "${1:-bench}" in
build)
    RUSTFLAGS="$RUSTFLAGS" "$CARGO" build --release --target wasm32-unknown-unknown
    ;;
build-threads)
    RUSTFLAGS="$RUSTFLAGS_THREADS" CARGO_TARGET_DIR=target-threads \
        "$CARGO" build --release --target wasm32-unknown-unknown
    node patch-shared.mjs \
        target-threads/wasm32-unknown-unknown/release/laya_engine_wasm.wasm \
        "$WASM_THREADS"
    ;;
selftest)
    node harness.mjs --self-test
    ;;
selftest-threads)
    node harness.mjs --self-test --threads "${2:-4}" --wasm "$WASM_THREADS"
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
smoke-threads)
    # threads artifact: self-test under K workers + one Laya forward with
    # the drift gate (validation before any threads bench number is read)
    K="${2:-2}"
    node harness.mjs --self-test --threads "$K" --wasm "$WASM_THREADS"
    [ -f native/laya_refund_f16.json ] || ./run.sh native
    node harness.mjs --model laya --threads "$K" --wasm "$WASM_THREADS" \
        --gguf "$MODELS/laya_english_f16.gguf" \
        --native native/laya_refund_f16.json --warmup 0 --repeats 1 \
        --out results/smoke_laya_t${K}.json >/dev/null
    node -e '
        const r = require(`./results/smoke_laya_t${process.argv[1]}.json`);
        const run = r.runs[0];
        if (run.forwards.trapped) { console.error("trapped:", run.forwards.trapped); process.exit(1); }
        const d = run.drift_vs_native;
        console.log("smoke K=" + process.argv[1] + ": forward " + run.forwards.inside_ms[0] +
            " ms, max|dlogit| " + d.max_dlogit.toExponential(2) + ", dp " + d.dp_after_temp.toExponential(2));
        if (d.max_dlogit > 0.05 || d.dp_after_temp > 0.02) { console.error("FAIL: drift gate breach"); process.exit(1); }
        console.log("wasm threads smoke: PASS");
    ' "$K"
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
bench-threads)
    # K-thread OpenThai/Laya benchmarks on the threads artifact (ticket 24).
    # Usage: ./run.sh bench-threads [K]   (default 4)
    K="${2:-4}"
    ./run.sh build-threads
    ./run.sh native
    node harness.mjs --model openthai --threads "$K" --wasm "$WASM_THREADS" \
        --gguf "$MODELS/openthai_systemone_v03_f16.gguf" \
        --native native/openthai_thai_f16.json --repeats 5 --out results/ot_f16_t${K}.json
    node harness.mjs --model openthai --threads "$K" --wasm "$WASM_THREADS" \
        --gguf "$MODELS/openthai_systemone_v03_q8_0.gguf" \
        --native native/openthai_thai_f16.json --native2 native/openthai_thai_q80.json \
        --repeats 5 --out results/ot_q80_t${K}.json
    node harness.mjs --model laya --threads "$K" --wasm "$WASM_THREADS" \
        --gguf "$MODELS/laya_english_f16.gguf" \
        --native native/laya_refund_f16.json --repeats 5 --out results/laya_f16_t${K}.json
    ;;
*)
    echo "usage: $0 build|build-threads|selftest|selftest-threads|native|smoke|smoke-threads|bench|bench-threads" >&2
    exit 2
    ;;
esac
