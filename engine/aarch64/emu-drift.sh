#!/bin/bash
# engine/aarch64/emu-drift.sh — ticket-18 dev-loop correctness: run the six
# shipped drift cases under docker's arm64 emulation (binfmt/qemu — the
# sudo-less substitute for qemu-user-static, recorded in ticket 17) for each
# kernel variant, gating each against the shipped x86-64 reference dumps.
#
# Emulation timing is MEANINGLESS (map rule, ticket 16) — this script is
# correctness only. Perf numbers come from run-m1.sh on the MacBook M1.
#
# Usage: bash engine/aarch64/emu-drift.sh [VARIANT...]   (default: fcvt fmlal sdot)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BIN="$REPO/engine/target/aarch64-unknown-linux-musl/release/laya-engine"
[ -x "$BIN" ] || { echo "build first: RUSTFLAGS='-Ctarget-cpu=generic -Clinker=rust-lld' cargo build --release --target aarch64-unknown-linux-musl"; exit 1; }
# docker.exe (Docker Desktop's Windows binary) first — the WSL2 distro shim
# can be broken while the Windows side works; a plain Linux box just uses docker.
DOCKER="${DOCKER:-$(command -v docker.exe || command -v docker)}"
# WSL2: docker lives on the Windows side; translate the paths for -v mounts.
case "$BIN" in
    /mnt/*|*) BIN_W="$(wslpath -w "$BIN" 2>/dev/null || echo "$BIN")" ;;
esac
MODELS_W="$(wslpath -w "$REPO/models" 2>/dev/null || echo "$REPO/models")"
EXAMPLES_W="$(wslpath -w "$REPO/examples" 2>/dev/null || echo "$REPO/examples")"
OUT="$HERE/results-emu"
mkdir -p "$OUT"
VARIANTS="${*:-fcvt fmlal sdot}"
FAILED=0

for v in $VARIANTS; do
    echo "=== variant: $v ==="
    DPASS=0; DFAIL=0
    for spec in \
        "laya_f16_refund|laya_english_f16.gguf|refund.json|Laya F16 refund" \
        "laya_f16_urgent_score|laya_english_f16.gguf|urgent-score.json|Laya F16 urgent-score" \
        "laya_q8_refund|laya_english_q8_0.gguf|refund.json|Laya Q8_0 refund" \
        "ot_f16_thai|openthai_systemone_v03_f16.gguf|openthai-thai.json|OpenThai F16 thai" \
        "ot_f16_mixed|openthai_systemone_v03_f16.gguf|openthai-mixed.json|OpenThai F16 mixed" \
        "ot_q8_thai|openthai_systemone_v03_q8_0.gguf|openthai-thai.json|OpenThai Q8_0 thai" \
    ; do
        ref="${spec%%|*}"; rest="${spec#*|}"
        model="${rest%%|*}"; rest="${rest#*|}"
        req="${rest%%|*}"; label="${rest#*|}"
        envs=()
        [ "$v" != "fcvt" ] && envs=(-e "LAYA_AARCH64_KERNEL=$v")
        if ! "$DOCKER" run --rm --platform linux/arm64 "${envs[@]}" \
                -v "$BIN_W:/work/e:ro" -v "$MODELS_W:/models:ro" -v "$EXAMPLES_W:/examples:ro" \
                alpine /work/e decide "/models/$model" --request "/examples/$req" --raw --threads 4 \
                > "$OUT/$v.$ref.json" 2>"$OUT/$v.$ref.err"; then
            echo "FAIL (decide errored): $label [$v] — see $OUT/$v.$ref.err"
            DFAIL=$((DFAIL+1)); FAILED=1; continue
        fi
        if python3 "$HERE/drift.py" "$OUT/$v.$ref.json" "$HERE/reference/$ref.json" \
            --label "$label [$v]"; then
            DPASS=$((DPASS+1))
        else
            DFAIL=$((DFAIL+1)); FAILED=1
        fi
    done
    echo "--- variant $v: $DPASS PASS, $DFAIL FAIL"
done
[ "$FAILED" = 0 ] && echo "EMU DRIFT: ALL PASS" || { echo "EMU DRIFT: FAIL"; exit 1; }
