#!/bin/bash
# engine/aarch64/run-m1.sh — one-script M1 acceptance run for the NEON
# kernel tier (wayfinder ticket 17) + the ticket-18 variant A/B. Run ON the
# MacBook M1, from anywhere:
#
#   bash engine/aarch64/run-m1.sh
#
# Stages (each prints PASS/FAIL/SKIP; any FAIL is recorded and reported):
#   1. build   — native aarch64-apple-darwin release build
#   2. self    — engine self-test (kernels vs scalar reference, exhaustive
#                65536-pattern FCVT f16->f32 check, FCVTN RNE probe + sweep,
#                per-variant arms, scalar-fallback A/B)
#   3. drift   — decide --raw for both models (F16 + Q8_0) vs the shipped
#                x86-64 reference dumps, tolerance gates (cross-ISA is
#                NEVER bit-identical): |dlogit|<=0.05, |dp|<=0.02, 0 flips —
#                run per kernel variant (fcvt = shipping default; fmlal /
#                sdot are the ticket-18 ARMv8.2 A/B candidates)
#   4. bench   — Laya + OpenThai latency per variant, 1 warmup + 8 measured
#                decides per thread count (daemon mode: model loads once,
#                timing is the engine's own usage.latency_ms), threads 1 4 8 10
#   5. candle  — OPTIONAL reference bar (CANDLE=1): builds
#                prototypes/candle-laya (candle-core 0.11 CPU) natively and
#                benches the same case, same 1+8 protocol
#   6. summary — a paste-back table with every number above, grouped by variant
#
# Env overrides: LAYA_F16 OPENTHAI_F16 LAYA_Q8 OT_Q8 (model paths),
# THREADS (default "1 4 8 10"), VARIANTS (default "fcvt fmlal sdot" — the
# A/B set; "fcvt" reproduces the ticket-17 run), TARGET_CPU (default
# generic = ARMv8-A baseline+NEON; "apple-m1" turns on the chip's fp16/
# dotprod codegen — our ARMv8.2 asm is runtime-gated either way), CANDLE=1.
set -u
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ENGINE="$REPO/engine"
CARGO="${CARGO:-$(command -v cargo || true)}"
[ -n "$CARGO" ] || { echo "FAIL: cargo not found (install rustup: https://rustup.rs)"; exit 1; }
TRIPLE=aarch64-apple-darwin
TARGET_CPU="${TARGET_CPU:-generic}"
LAYA_F16="${LAYA_F16:-$REPO/models/laya_english_f16.gguf}"
OT_F16="${OPENTHAI_F16:-$REPO/models/openthai_systemone_v03_f16.gguf}"
LAYA_Q8="${LAYA_Q8:-$REPO/models/laya_english_q8_0.gguf}"
OT_Q8="${OT_Q8:-$REPO/models/openthai_systemone_v03_q8_0.gguf}"
THREADS="${THREADS:-1 4 8 10}"
VARIANTS="${VARIANTS:-fcvt fmlal sdot}"
WARMUP=1
MEASURED=8
BIN="$ENGINE/target/$TRIPLE/release/laya-engine"
OUT="$HERE/results"
FAILED=0
mkdir -p "$OUT"
: > "$OUT/drift_summary.txt"
: > "$OUT/bench_summary.txt"

skip_missing() { # skip_missing PATH LABEL -> 0 present / 1 skip
    [ -f "$1" ] && return 0
    echo "SKIP: missing $2: $1"
    return 1
}

# ------------------------------------------------------------------ 1 build
echo "=== stage 1: build ($TRIPLE release, -Ctarget-cpu=$TARGET_CPU) ==="
# --target overrides the repo root's [build] target (x86_64); the root
# rustflags are x86_64-scoped and do not apply here. TARGET_CPU=generic =
# ARMv8-A baseline + NEON; the ticket-18 variant asm (fmlal/sdot) is
# runtime-hwcaps-gated so it works at either setting.
if ! (cd "$ENGINE" && CARGO_ENCODED_RUSTFLAGS="-Ctarget-cpu=$TARGET_CPU" \
        "$CARGO" build --release --target "$TRIPLE"); then
    echo "FAIL: build"
    exit 1
fi
echo "binary: $BIN"
echo "--- build: PASS"

# -------------------------------------------------------------- 2 self-test
echo "=== stage 2: self-test ==="
if "$BIN" self-test; then
    echo "--- self-test: PASS"
else
    echo "FAIL: self-test"; FAILED=1
fi

# ------------------------------------------------------------- 3 drift gates
echo "=== stage 3: drift gates vs x86-64 reference dumps (per variant) ==="
DPASS=0; DFAIL=0
drift_case() { # drift_case MODEL REFNAME REQFILE LABEL VARIANT
    # env takes NAME=value directly — no -e flag (BSD/GNU env reject it;
    # the M1 run #1 lost every non-fcvt decide to `env: illegal option`).
    local envs=()
    [ "$5" != "fcvt" ] && envs=("LAYA_AARCH64_KERNEL=$5")
    if ! env ${envs[@]+"${envs[@]}"} "$BIN" decide "$1" --request "$3" --raw --threads 4 \
            > "$OUT/$5.$2.got.json" 2> "$OUT/$5.$2.got.err"; then
        echo "FAIL: decide exited non-zero for $4 [$5] — stderr tail:"
        tail -3 "$OUT/$5.$2.got.err" | sed 's/^/    /' | tee -a "$OUT/drift_summary.txt"
        DFAIL=$((DFAIL+1)); FAILED=1
        return
    fi
    if python3 "$HERE/drift.py" "$OUT/$5.$2.got.json" "$HERE/reference/$2.json" \
        --label "$4 [$5]" | tee -a "$OUT/drift_summary.txt"; then
        DPASS=$((DPASS+1))
    else
        DFAIL=$((DFAIL+1)); FAILED=1
    fi
}
for v in $VARIANTS; do
    for spec in \
        "laya_f16_refund|$LAYA_F16|$REPO/examples/refund.json|Laya F16 refund" \
        "laya_f16_urgent_score|$LAYA_F16|$REPO/examples/urgent-score.json|Laya F16 urgent-score" \
        "laya_q8_refund|$LAYA_Q8|$REPO/examples/refund.json|Laya Q8_0 refund" \
        "ot_f16_thai|$OT_F16|$REPO/examples/openthai-thai.json|OpenThai F16 thai" \
        "ot_f16_mixed|$OT_F16|$REPO/examples/openthai-mixed.json|OpenThai F16 mixed" \
        "ot_q8_thai|$OT_Q8|$REPO/examples/openthai-thai.json|OpenThai Q8_0 thai" \
    ; do
        ref="${spec%%|*}"; rest="${spec#*|}"
        model="${rest%%|*}"; rest="${rest#*|}"
        req="${rest%%|*}"; label="${rest#*|}"
        if skip_missing "$model" "model ($ref)"; then
            drift_case "$model" "$ref" "$req" "$label" "$v"
        fi
    done
done
echo "--- drift: $DPASS PASS, $DFAIL FAIL"

# ------------------------------------------------------------------ 4 bench
echo "=== stage 4: bench per variant (1 warmup + $MEASURED measured, daemon mode) ==="
bench_model() { # bench_model MODEL LABEL REQFILE VARIANT -> BENCH|...|FAIL lines
    local req t
    # one compact JSON per stdin line (request files are pretty-printed)
    req="$(python3 -c 'import json,sys
print(json.dumps(json.load(open(sys.argv[1])), separators=(",", ":")))' "$3")"
    for t in $THREADS; do
        {
            for _ in $(seq $((WARMUP + MEASURED))); do printf '%s\n' "$req"; done
        } | env ${4:+LAYA_AARCH64_KERNEL=$4} "$BIN" daemon "$1" --threads "$t" --raw \
            > "$OUT/bench_$2_$4_t$t.jsonl" 2>/dev/null
        python3 - "$OUT/bench_$2_$4_t$t.jsonl" "$WARMUP" "$2" "$t" "$4" <<'EOF'
import json, sys
lines = [l for l in open(sys.argv[1]) if l.strip()]
warm = int(sys.argv[2]); label = sys.argv[3]; t = sys.argv[4]; v = sys.argv[5]
ms = [float(json.loads(l)["usage"]["latency_ms"]) for l in lines if "usage" in json.loads(l)]
if len(ms) < warm + 1:
    print(f"FAIL|bench {label} {v} t{t}|only {len(ms)} decide responses")
    sys.exit(0)
ms = ms[warm:]
mean = sum(ms) / len(ms)
print(f"BENCH|{label}|{v}|{t}|{mean:.1f}|{min(ms):.1f}|{len(ms)}")
EOF
    done
}
for v in $VARIANTS; do
    for job in \
        "laya_f16|$LAYA_F16|$REPO/examples/refund.json" \
        "ot_f16|$OT_F16|$REPO/examples/openthai-thai.json" \
        "ot_q8|$OT_Q8|$REPO/examples/openthai-thai.json" \
    ; do
        label="${job%%|*}"; rest="${job#*|}"
        model="${rest%%|*}"; req="${rest#*|}"
        if skip_missing "$model" "model ($label bench)"; then
            r="$(bench_model "$model" "$label" "$req" "$v")"
            echo "$r"
            echo "$r" | grep "^BENCH|" >> "$OUT/bench_summary.txt"
            if echo "$r" | grep "^FAIL|" >/dev/null; then FAILED=1; fi
        fi
    done
done

# -------------------------------------------------- 5 optional candle bar
if [ "${CANDLE:-0}" = "1" ]; then
    echo "=== stage 5: candle-aarch64 reference bar ==="
    CAN="$REPO/prototypes/candle-laya/target/$TRIPLE/release/candle-laya"
    if [ ! -d "$REPO/prototypes/candle-laya" ]; then
        echo "SKIP: prototypes/candle-laya not present"
    elif skip_missing "$LAYA_F16" "laya f16 model" && \
         (cd "$REPO/prototypes/candle-laya" && "$CARGO" build --release --target "$TRIPLE"); then
        for t in $THREADS; do
            echo "[candle threads=$t]"
            "$CAN" bench "$LAYA_F16" --case "$REPO/examples/refund.json" \
                --mode convert --warmup $WARMUP --repeats $MEASURED \
                --threads "$t" 2>/dev/null || { echo "candle bench failed at threads=$t"; break; }
        done | tee "$OUT/candle_summary.txt"
    else
        echo "SKIP: candle-laya build failed"
    fi
else
    echo "=== stage 5: candle bar skipped (run with CANDLE=1 for the reference numbers) ==="
fi

# ---------------------------------------------------------------- 6 summary
echo ""
echo "==== M1 acceptance summary (paste back into the ticket) ===="
{
    uname -m
    sysctl -n machdep.cpu.brand_string 2>/dev/null
    sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GiB RAM\n", $1/1073741824}'
} | sed 's/^/machine: /'
echo "rustc: $("$CARGO" --version 2>/dev/null || echo unknown)"
echo ""
echo "self-test: $([ "$FAILED" = 0 ] || echo "see stages above; ")PASS"
echo ""
echo "Drift gates (vs shipped x86-64 dumps; bars: |dlogit|<=0.05, |dp|<=0.02, flips=0):"
sed 's/^/  /' "$OUT/drift_summary.txt"
echo ""
echo "Latency per kernel variant (1 warmup + $MEASURED measured decides, engine usage.latency_ms):"
echo ""
echo "| config | variant | threads | mean ms | min ms | n |"
echo "|---|---|---:|---:|---:|---:|"
sed 's/^BENCH|/|/; s/$/|/' "$OUT/bench_summary.txt"
if [ -f "$OUT/candle_summary.txt" ]; then
    echo ""
    echo "candle-aarch64 reference bar (convert mode, same 1+$MEASURED protocol):"
    sed 's/^/  /' "$OUT/candle_summary.txt"
fi
echo ""
[ "$FAILED" = 0 ] && echo "RESULT: PASS" || { echo "RESULT: FAIL (see stages above)"; exit 1; }
