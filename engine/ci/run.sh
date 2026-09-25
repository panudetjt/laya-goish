#!/bin/bash
# engine/ci/run.sh — one-command release/CI matrix for the std engine
# (wayfinder ticket 14). Mirrors .github/workflows/engine-ci.yml: the
# workflow calls this script, so local and CI run the same tiers.
#
# Tiers:
#   quick (default)  build + smoke + reduced drift (2 cases x 2 threads)
#   full             all tiers below at full depth (exactly two 42-combo
#                    drift runs: gnu and musl)
#   build            gnu release via engine/run.sh
#   smoke            CLI surface gate (engine/ci/smoke.py; parameterized
#                    port of validation/smoke.py, which is frozen and
#                    hardcodes the goish binary path)
#   http             HTTP integration gate (engine/ci/http_gate.py; the
#                    parameterized replica of the frozen
#                    validation/test_http.py + test_openthai_http.py)
#   parity           byte-parity gate goish vs engine over raw-socket
#                    probes incl. auth/busy/drain (engine/ci/parity.py;
#                    needs REF_BIN and both f16 models)
#   soak             bounded memory soak (engine/ci/soak.py; the
#                    parameterized replica of frozen soak_http.py)
#   drift            full 42-combo bit-exactness vs REF_BIN
#                    (validation/scalar-tail-drift.py — frozen, unmodified)
#   drift-quick      reduced drift gate (engine/ci/reduced-drift.py)
#   quant            Q8_0 drift gates (validation/quantized-drift.py)
#   musl             musl build + static-pie check + 42-combo drift on the
#                    musl binary
#   wasm             wasm32-unknown-unknown compile check (lib+bin; the
#                    getrandom custom-backend cfg recipe from
#                    prototypes/wasm-gate — currently inert since the dep
#                    tree has no getrandom, kept so the gate cannot rot)
#   cross            cargo check: windows-msvc + both apple-darwin +
#                    aarch64-linux-gnu
#   cross:TRIPLE     single-triple check
#   lint             cargo fmt --check + cargo clippy (engine/ ONLY; goish
#                    src/ is a frozen reference and is not linted)
#   qemu             aarch64-musl static build (+ self-test/decide under
#                    qemu-aarch64-static when installed; else SKIP)
#
# Usage: engine/ci/run.sh [tier ...]        (default: quick)
#
# Env overrides:
#   CARGO, PYTHON     tool locators (default: PATH, then /home/jame/.cargo)
#   REF_BIN           drift reference binary (default: ./laya at repo root)
#   LAYA_F16, OPENTHAI_F16, LAYA_Q8, OPENTHAI_Q8   model paths
#
# Etiquette: tiers run sequentially; each prints PASS/FAIL (or SKIP with a
# reason when a precondition — models, reference binary, qemu — is absent)
# and the run exits non-zero if any tier FAILed. SKIP never fails the run:
# gate breaches always do.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE="$ROOT/engine"
CI="$ENGINE/ci"
RUNSH="$ENGINE/run.sh"
CARGO="${CARGO:-$(command -v cargo || true)}"
CARGO="${CARGO:-/home/jame/.cargo/bin/cargo}"
PY="${PYTHON:-$ROOT/.venv/bin/python3}"
[ -x "$PY" ] || PY=python3
REF_BIN="${REF_BIN:-$ROOT/laya}"
BIN_GNU="$ENGINE/target/x86_64-unknown-linux-gnu/release/laya-engine"
BIN_MUSL="$ENGINE/target/x86_64-unknown-linux-musl/release/laya-engine"
BIN_AARCH64="$ENGINE/target/aarch64-unknown-linux-musl/release/laya-engine"
LAYA_F16="${LAYA_F16:-$ROOT/models/laya_english_f16.gguf}"
OPENTHAI_F16="${OPENTHAI_F16:-$ROOT/models/openthai_systemone_v03_f16.gguf}"
LAYA_Q8="${LAYA_Q8:-$ROOT/models/laya_english_q8_0.gguf}"
OPENTHAI_Q8="${OPENTHAI_Q8:-$ROOT/models/openthai_systemone_v03_q8_0.gguf}"
CROSS_TRIPLES="x86_64-pc-windows-msvc x86_64-apple-darwin aarch64-apple-darwin aarch64-unknown-linux-gnu"
SKIP_RC=77

# ---------------------------------------------------------------- helpers
need() { # need PATH LABEL -> rc 77 (skip) when PATH is missing
    if [ ! -f "$1" ]; then
        echo "SKIP: missing $2: $1"
        return "$SKIP_RC"
    fi
    return 0
}
ensure_target() {
    "$CARGO" --version >/dev/null 2>&1 || return 1
    rustup target list --installed 2>/dev/null | grep -qx "$1" && return 0
    echo "rustup target add $1"
    rustup target add "$1" || return 1
}

# ------------------------------------------------------------------ tiers
do_build() {
    "$RUNSH" build || return 1
    echo "binary: $BIN_GNU"
}
do_smoke() {
    need "$LAYA_F16" "laya f16 model" || return
    "$PY" "$CI/smoke.py" "$BIN_GNU" "$LAYA_F16" "$ROOT" || return 1
}
do_http() {
    need "$LAYA_F16" "laya f16 model" || return
    need "$OPENTHAI_F16" "openthai f16 model" || return
    "$PY" "$CI/http_gate.py" "$BIN_GNU" "$LAYA_F16" "$OPENTHAI_F16" "$ROOT" || return 1
}
do_parity() {
    need "$REF_BIN" "reference binary (REF_BIN)" || return
    need "$LAYA_F16" "laya f16 model" || return
    need "$OPENTHAI_F16" "openthai f16 model" || return
    "$PY" "$CI/parity.py" "$REF_BIN" "$BIN_GNU" \
        "$LAYA_F16" "$OPENTHAI_F16" "$ROOT" || return 1
}
do_soak() {
    need "$LAYA_F16" "laya f16 model" || return
    "$PY" "$CI/soak.py" "$BIN_GNU" "$LAYA_F16" || return 1
}
do_drift() {
    need "$REF_BIN" "drift reference binary (REF_BIN)" || return
    need "$LAYA_F16" "laya f16 model" || return
    need "$OPENTHAI_F16" "openthai f16 model" || return
    "$PY" "$ROOT/validation/scalar-tail-drift.py" "$REF_BIN" "$BIN_GNU" \
        "$LAYA_F16" "$OPENTHAI_F16" "$ROOT/examples" || return 1
}
do_drift_quick() {
    need "$REF_BIN" "drift reference binary (REF_BIN)" || return
    need "$LAYA_F16" "laya f16 model" || return
    need "$OPENTHAI_F16" "openthai f16 model" || return
    "$PY" "$CI/reduced-drift.py" "$REF_BIN" "$BIN_GNU" \
        "$LAYA_F16" "$OPENTHAI_F16" "$ROOT/examples" || return 1
}
do_quant() {
    need "$LAYA_F16" "laya f16 model" || return
    need "$LAYA_Q8" "laya q8_0 model" || return
    need "$OPENTHAI_F16" "openthai f16 model" || return
    need "$OPENTHAI_Q8" "openthai q8_0 model" || return
    "$PY" "$ROOT/validation/quantized-drift.py" "$BIN_GNU" \
        "$LAYA_F16" "$LAYA_Q8" "$OPENTHAI_F16" "$OPENTHAI_Q8" \
        "$ROOT/examples" || return 1
}
do_musl() {
    "$RUNSH" build musl || return 1
    if ! file "$BIN_MUSL" | grep -q "static-pie linked"; then
        file "$BIN_MUSL"
        echo "FAIL: musl binary is not static-pie"
        return 1
    fi
    echo "static-pie: $(file "$BIN_MUSL" | grep -o 'static-pie linked')"
    need "$REF_BIN" "drift reference binary (REF_BIN)" || return
    need "$LAYA_F16" "laya f16 model" || return
    need "$OPENTHAI_F16" "openthai f16 model" || return
    "$PY" "$ROOT/validation/scalar-tail-drift.py" "$REF_BIN" "$BIN_MUSL" \
        "$LAYA_F16" "$OPENTHAI_F16" "$ROOT/examples" || return 1
}
do_wasm() {
    ensure_target wasm32-unknown-unknown || return 1
    (cd "$ENGINE" && RUSTFLAGS='--cfg getrandom_backend="custom"' \
        "$CARGO" check --target wasm32-unknown-unknown) || return 1
    echo "wasm32-unknown-unknown: compile-check PASS (lib + bin)"
    # Runtime smoke (ticket 19): real inference inside wasm via the
    # engine/wasm cdylib — self-test (kernels + bit-exact widen), then one
    # Laya forward drift-gated against a fresh native dump. Skips without
    # Node or the model; the cdylib build is cached after first use.
    command -v node >/dev/null 2>&1 || { echo "SKIP: node not found (wasm runtime smoke)"; return; }
    need "$LAYA_F16" "laya f16 model" || return
    need "$BIN_GNU" "gnu engine binary (build tier)" || return
    (cd "$ENGINE/wasm" && RUSTFLAGS='--cfg getrandom_backend="custom"' \
        "$CARGO" build --release --target wasm32-unknown-unknown) || return 1
    local dump="$ENGINE/wasm/results/ci_laya_ref.json"
    mkdir -p "$ENGINE/wasm/results"
    "$BIN_GNU" decide "$LAYA_F16" --request "$ROOT/examples/refund.json" \
        --raw > "$dump" || return 1
    (cd "$ENGINE/wasm" && node harness.mjs --self-test) || return 1
    (cd "$ENGINE/wasm" && node harness.mjs --model laya --gguf "$LAYA_F16" \
        --native "$dump" --warmup 0 --repeats 1 \
        --out results/ci_smoke.json >/dev/null) || return 1
    "$PY" - "$ENGINE/wasm/results/ci_smoke.json" <<'EOF' || return 1
import json, sys
r = json.load(open(sys.argv[1]))
run = r["runs"][0]
if run["forwards"].get("trapped"):
    print("FAIL: wasm forward trapped:", run["forwards"]["trapped"]); sys.exit(1)
d = run["drift_vs_native"]
print(f"wasm smoke: {run['forwards']['inside_ms'][0]} ms, max|dlogit| {d['max_dlogit']:.2e}, dp {d['dp_after_temp']:.2e}")
if d["max_dlogit"] > 0.05 or d["dp_after_temp"] > 0.02:
    print("FAIL: drift gate breach"); sys.exit(1)
EOF
    echo "wasm32 runtime smoke: PASS (self-test + Laya forward + drift gate)"
}
do_cross() {
    local triples="${1:-$CROSS_TRIPLES}"
    for t in $triples; do
        ensure_target "$t" || return 1
        (cd "$ENGINE" && "$CARGO" check --target "$t") || return 1
        echo "cargo check $t: PASS"
    done
}
do_lint() {
    (cd "$ENGINE" && "$CARGO" fmt --check) || return 1
    echo "fmt --check: PASS"
    # The allow-list documents verbatim-port exceptions (kernels/loops are
    # byte-copies of the goish sources) — anything NOT listed still
    # hard-fails, so new code (tickets 17-19 kernels) gets full clippy
    # scrutiny. registry/Engine::validate were un-allowed by ticket 13:
    # the serve path now wires both.
    (cd "$ENGINE" && CARGO_ENCODED_RUSTFLAGS="-Ctarget-cpu=x86-64" \
        "$CARGO" clippy --release --target x86_64-unknown-linux-gnu \
        --all-targets -- -D warnings \
        -A clippy::manual_clamp \
        -A clippy::excessive_precision \
        -A clippy::manual_is_multiple_of \
        -A clippy::manual_range_contains \
        -A clippy::needless_range_loop \
        -A clippy::len_zero \
        -A clippy::too_many_arguments \
        -A clippy::unnecessary_to_owned \
        -A clippy::redundant_closure) || return 1
    echo "clippy: PASS"
}
do_qemu() {
    ensure_target aarch64-unknown-linux-musl || return 1
    (cd "$ENGINE" && RUSTFLAGS="-Ctarget-cpu=generic -Clinker=rust-lld" \
        "$CARGO" build --release --target aarch64-unknown-linux-musl) || return 1
    file "$BIN_AARCH64" | grep -q "statically linked" || {
        file "$BIN_AARCH64"; return 1; }
    echo "aarch64-musl static build: PASS"
    if ! command -v qemu-aarch64-static >/dev/null 2>&1; then
        echo "SKIP: qemu-aarch64-static not installed (apt: qemu-user-static);"
        echo "      with it, this tier also runs self-test + one decide under QEMU"
        return "$SKIP_RC"
    fi
    qemu-aarch64-static "$BIN_AARCH64" self-test || return 1
    need "$LAYA_F16" "laya f16 model" || return
    qemu-aarch64-static "$BIN_AARCH64" decide "$LAYA_F16" \
        --request "$ROOT/examples/refund.json" --threads 4 \
        > /dev/null || return 1
    echo "qemu-aarch64 self-test + decide: PASS (correctness only, never perf)"
}

# ------------------------------------------------------------------ driver
FAILED=0
SUMMARY=()
tier() { # tier NAME FN [ARGS...]
    local name="$1" fn="$2"
    shift 2
    local t0=$SECONDS rc=0
    echo ""
    echo "=== tier: $name ==="
    "$fn" "$@" || rc=$?
    local dt=$((SECONDS - t0))
    if [ "$rc" = 0 ]; then
        SUMMARY+=("PASS  $name (${dt}s)")
        echo "--- $name: PASS (${dt}s)"
    elif [ "$rc" = "$SKIP_RC" ]; then
        SUMMARY+=("SKIP  $name (${dt}s)")
        echo "--- $name: SKIP (${dt}s)"
    else
        FAILED=1
        SUMMARY+=("FAIL  $name (${dt}s, rc=$rc)")
        echo "--- $name: FAIL (${dt}s, rc=$rc)"
    fi
}

ARGS=("$@")
[ ${#ARGS[@]} = 0 ] && ARGS=(quick)
for arg in "${ARGS[@]}"; do
    case "$arg" in
        quick) tier build do_build; tier smoke do_smoke; tier drift-quick do_drift_quick ;;
        full)
            tier build do_build
            tier smoke do_smoke
            tier drift do_drift
            tier quant do_quant
            tier http do_http
            tier musl do_musl
            tier wasm do_wasm
            tier cross do_cross
            tier lint do_lint
            tier qemu do_qemu
            ;;
        build | smoke | drift | drift-quick | quant | http | parity | soak | musl | wasm | lint | qemu)
            tier "$arg" "do_${arg/-/_}"
            ;;
        cross:*)
            tier "$arg" do_cross "${arg#cross:}"
            ;;
        cross) tier cross do_cross "" ;;
        list)
            echo "tiers: quick full build smoke drift drift-quick quant http parity soak musl wasm cross cross:TRIPLE lint qemu"
            ;;
        *)
            echo "unknown tier: $arg (run: $0 list)" >&2
            exit 2
            ;;
    esac
done

echo ""
echo "==== engine CI matrix summary ===="
for s in "${SUMMARY[@]}"; do echo "$s"; done
if [ "$FAILED" = 1 ]; then
    echo "RESULT: FAIL"
    exit 1
fi
echo "RESULT: PASS"
