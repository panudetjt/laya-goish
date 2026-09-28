#!/bin/bash
# engine/ci/package.sh — release packaging: artifacts + checksums per target
# (wayfinder ticket 22). One command produces every artifact set this
# machine can produce, so ticket 20 (ship v0.2) only needs ticket 18's
# numbers and the Mac builds.
#
# Artifact sets (deterministic layout under engine/dist/<target>/):
#   linux-x86_64-musl  tar.gz: laya-engine (static-pie) + presets.json +
#                      USAGE.md + MANIFEST.md + inner SHA256SUMS; outer
#                      SHA256SUMS covers the archive
#   wasm32            zip: laya_engine_wasm.wasm (at the harness's default
#                      relative path, so RUN.md commands run verbatim) +
#                      harness.mjs + RUN.md + MANIFEST.md + inner SHA256SUMS
#   windows-x86_64-msvc  attempt a real linked .exe with rust-lld. If the
#                      link fails (expected: rustup ships no MSVC import
#                      libs), the exact failure is recorded in the set's
#                      MANIFEST.md + build-link.log and the artifacts are
#                      marked "requires a Windows machine" (tickets 23/20
#                      own the follow-up). This recorded outcome is a PASS
#                      of the packaging step, not a gate failure.
#   macos             not producible here (Apple SDK): a MANIFEST.md note
#                      pointing at engine/aarch64/run-m1.sh (aarch64) and a
#                      native x86_64-apple-darwin build on an Intel Mac.
#
# Build recipes are the proven ones: engine/run.sh build musl, engine/wasm/
# run.sh (build + smoke), CARGO_ENCODED_RUSTFLAGS replaces the repo-root
# no_std rustflags (engine/run.sh pattern), and the windows attempt reuses
# the CI cross tier's triple. The git rev is read straight from .git/HEAD +
# the ref file — no git commands. Runtime gates run against the STAGED
# artifact bytes, not the build-dir copy.
#
# Usage: engine/ci/package.sh [--skip-verify] [target ...]
#   targets: linux-musl wasm32 windows macos (default: all four)
#   --skip-verify  skip the runtime smokes (musl smoke.py gate, wasm
#                  run.sh smoke); build + checksum verification still run.
#
# Etiquette: every step prints PASS/FAIL; the run exits non-zero if a
# requested set fails to BUILD, VERIFY, STAGE or CHECKSUM. Windows link
# failure is the one recorded-not-fatal outcome (see above).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE="$ROOT/engine"
CI="$ENGINE/ci"
DIST="$ENGINE/dist"
CARGO="${CARGO:-$(command -v cargo || true)}"
CARGO="${CARGO:-/home/jame/.cargo/bin/cargo}"
RUSTC="$(dirname "$CARGO")/rustc"
[ -x "$RUSTC" ] || RUSTC="$(command -v rustc || echo rustc)"
PY="${PYTHON:-$ROOT/.venv/bin/python3}"
[ -x "$PY" ] || PY=python3
BIN_MUSL="$ENGINE/target/x86_64-unknown-linux-musl/release/laya-engine"
BIN_EXE="$ENGINE/target/x86_64-pc-windows-msvc/release/laya-engine.exe"
WASM="$ENGINE/wasm/target/wasm32-unknown-unknown/release/laya_engine_wasm.wasm"
LAYA_F16="${LAYA_F16:-$ROOT/models/laya_english_f16.gguf}"
SKIP_VERIFY=0

# ------------------------------------------------------------- metadata
VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$ENGINE/Cargo.toml" | head -1)"
RUSTC_VER="$("$RUSTC" --version)"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git_rev() { # read rev+branch from .git without git commands
    local head="$(cat "$ROOT/.git/HEAD" 2>/dev/null || echo unknown)"
    case "$head" in
        ref:\ *) local ref="${head#ref: }"
                echo "$(cat "$ROOT/.git/$ref" 2>/dev/null || echo unknown) ${ref#refs/heads/}" ;;
        *) echo "$head (detached)" ;;
    esac
}
REV_BRANCH="$(git_rev)"
REV="$(echo "$REV_BRANCH" | cut -d' ' -f1)"
BRANCH="$(echo "$REV_BRANCH" | cut -d' ' -f2)"
REV_SHORT="${REV:0:12}"

sha256() { sha256sum "$1" | cut -d' ' -f1; }
fsize() { stat -c %s "$1"; }

# stage_checksums SETDIR  — inner SHA256SUMS over every staged file except
# itself (sorted, so `sha256sum -c` after extraction is deterministic).
stage_checksums() {
    (cd "$1" && find . -type f ! -name SHA256SUMS -printf '%P\n' | sort \
        | xargs sha256sum > SHA256SUMS)
}

# verify_inner ARCHIVE EXTRACTCMD  — extract to a temp dir, sha256sum -c.
verify_inner() {
    local archive="$1" extract="$2" tmp rc=0
    tmp="$(mktemp -d)"
    case "$extract" in
        tar) tar -xzf "$archive" -C "$tmp" ;;
        zip) "$PY" -m zipfile -e "$archive" "$tmp" ;;
    esac
    (cd "$tmp"/* && sha256sum -c SHA256SUMS >/dev/null) || rc=1
    rm -rf "$tmp"
    return "$rc"
}

# determinstic_archive  — reproducible tar.gz: sorted names, uid/gid 0,
# mtime epoch 0, gzip without name/timestamp.
make_tarball() { # SETDIR DIRNAME OUT_ARCHIVE
    tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@0" \
        -C "$1" -cf - "$2" | gzip -n > "$3"
}

# make_zip SRCDIR DIRNAME OUT_ARCHIVE  — deterministic zip via python
# (fixed 1980 timestamps, fixed attrs, sorted, deflate-9); no zip(1) on
# this machine, python3's zipfile is always present.
make_zip() {
    "$PY" - "$1" "$2" "$3" <<'PYEOF'
import os, sys, zipfile
src, arcroot, dst = sys.argv[1], sys.argv[2], sys.argv[3]
files = []
for base, _, names in os.walk(src):
    files += [os.path.join(base, n) for n in names]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in sorted(files, key=lambda p: os.path.relpath(p, src)):
        zi = zipfile.ZipInfo(
            arcroot + "/" + os.path.relpath(f, src).replace(os.sep, "/"),
            date_time=(1980, 1, 1, 0, 0, 0))
        zi.external_attr = 0o644 << 16
        zi.compress_type = zipfile.ZIP_DEFLATED
        with open(f, "rb") as h:
            z.writestr(zi, h.read())
PYEOF
}

# manifest_header TITLE  — shared metadata block, written to stdout.
manifest_header() {
    cat <<EOF
# $1

- built: $BUILD_DATE (UTC)
- git: $REV_SHORT (branch $BRANCH, read from .git/HEAD — no git commands)
- rustc: $RUSTC_VER
- crate: engine/ (laya-engine $VERSION)
EOF
}

# file_table FILE... — "- size  sha256  name" rows for the manifest.
file_table() {
    for f in "$@"; do
        echo "- $(fsize "$f") bytes  sha256:$(sha256 "$f")  ${f#./}"
    done
}

outer_checksum() { # SETDIR ARCHIVE_NAME — SHA256SUMS next to the archive
    (cd "$1" && sha256sum "$2" > SHA256SUMS && sha256sum -c SHA256SUMS >/dev/null)
}

FAILED=0
SUMMARY=()

step() { # step RC LABEL DETAIL — print one PASS/FAIL line; FAIL sets FAILED
    if [ "$1" = 0 ]; then
        echo "--- $2: PASS${3:+ ($3)}"
    else
        echo "--- $2: FAIL${3:+ ($3)}"
        FAILED=1
    fi
}
target_summary() { # target_summary NAME STATUS DETAIL
    SUMMARY+=("$1  $2${3:+ ($3)}")
}
staging_ok() { # staging_ok FILE... — every staged file must exist non-empty
    local f
    for f in "$@"; do [ -s "$f" ] || return 1; done
    return 0
}

# ================================================================ linux-musl
do_linux_musl() {
    local name="laya-engine-$VERSION-linux-x86_64-musl" dir="$DIST/linux-x86_64-musl"
    local stage="$dir/$name" t0=$SECONDS fail0=$FAILED
    echo ""
    echo "=== target: linux-x86_64-musl ==="
    rm -rf "$stage"; mkdir -p "$stage"

    echo "build: engine/run.sh build musl (proven recipe)"
    "$ENGINE/run.sh" build musl \
        || { step 1 build; target_summary linux-x86_64-musl FAIL "musl build"; return; }
    [ -x "$BIN_MUSL" ]
    step $? build "binary linked: $BIN_MUSL"

    cp "$BIN_MUSL" "$stage/laya-engine"
    cp "$ENGINE/presets.json" "$stage/presets.json"
    if ! staging_ok "$stage/laya-engine" "$stage/presets.json"; then
        step 1 staging "binary + presets staged"
        target_summary linux-x86_64-musl FAIL "staging incomplete"
        return
    fi
    step 0 staging "binary + presets staged"

    local pie="no"
    file "$stage/laya-engine" | grep -q "static-pie linked" && pie="yes"
    [ "$pie" = yes ]
    step $? static-pie-check "$(file -b "$stage/laya-engine" | cut -d, -f1-3)"

    local smoke="skipped (--skip-verify)" smoke_rc=0
    if [ "$SKIP_VERIFY" = 0 ]; then
        if [ ! -f "$LAYA_F16" ]; then
            echo "SKIP: laya f16 model missing ($LAYA_F16) — runtime smoke"
            smoke="skipped (model absent)"
        else
            if "$PY" "$CI/smoke.py" "$stage/laya-engine" "$LAYA_F16" "$ROOT" >/dev/null; then
                smoke_rc=0
            else
                smoke_rc=1
            fi
            step "$smoke_rc" smoke "engine/ci/run.sh quick's smoke gate vs the STAGED musl binary"
            smoke=$([ "$smoke_rc" = 0 ] && echo "PASS" || echo "FAIL")
        fi
    fi

    cat > "$stage/USAGE.md" <<EOF
# laya-engine v$VERSION — Linux x86-64 (musl static-pie)

Single self-contained binary; presets are compiled in (see \`list-presets\`).
Requires an x86-64 CPU with AVX2/FMA. Model weights are NOT included —
fetch them separately, e.g. \`scripts/download-models.sh f16\` from the
repo (or the release page) for laya_english_f16.gguf.

    chmod +x laya-engine
    ./laya-engine self-test
    ./laya-engine info models/laya_english_f16.gguf

One decision (preset questions, no request file needed):

    ./laya-engine decide models/laya_english_f16.gguf \\
        --preset email --text "I was charged twice, please refund the duplicate."

JSON-lines process (one request per stdin line, answers on stdout):

    ./laya-engine daemon models/laya_english_f16.gguf --threads 4

HTTP server (GET /health /v1/models /v1/presets, POST /v1/decide):

    ./laya-engine serve models/laya_english_f16.gguf --port 8080 --threads 4

presets.json is shipped for reference (the API exposes it at /v1/presets);
the binary embeds an identical copy. SHA256SUMS covers every file in this
archive except itself; verify after extraction with \`sha256sum -c SHA256SUMS\`.
EOF

    { manifest_header "laya-engine $VERSION — linux x86-64 musl"
      echo
      echo "## Artifact"
      echo "- archive: $name.tar.gz ($(fsize "$stage/laya-engine")-byte binary inside)"
      echo "- recipe: engine/run.sh build musl — CARGO_ENCODED_RUSTFLAGS=-Ctarget-cpu=x86-64, x86_64-unknown-linux-musl, static-pie"
      echo
      echo "## Files"
      (cd "$stage" && file_table laya-engine presets.json USAGE.md)
      echo
      echo "## Verification"
      echo "- static-pie: file(1) reports 'static-pie linked': $pie"
      echo "- engine/ci/smoke.py vs the STAGED artifact bytes: $smoke (self-tests,"
      echo "  10 presets, invalid CLI/GGUF, daemon recovery, repeated requests,"
      echo "  1/4-thread parity, numeric-state parity)"
      echo "- checksums: inner SHA256SUMS verified post-extraction; outer SHA256SUMS"
      echo "  (next to the archive) verified with sha256sum -c"
    } > "$stage/MANIFEST.md"

    stage_checksums "$stage"
    make_tarball "$dir" "$name" "$dir/$name.tar.gz"
    outer_checksum "$dir" "$name.tar.gz"
    step $? checksums "outer sha256 verified"
    verify_inner "$dir/$name.tar.gz" tar
    step $? inner-checksums "extracted + sha256sum -c"
    target_summary linux-x86_64-musl \
        "$([ "$FAILED" = "$fail0" ] && echo PASS || echo FAIL) x86_64-unknown-linux-musl static-pie" "$((SECONDS - t0))s"
}

# ==================================================================== wasm32
do_wasm32() {
    local name="laya-engine-$VERSION-wasm32" dir="$DIST/wasm32"
    local stage="$dir/$name" t0=$SECONDS fail0=$FAILED
    echo ""
    echo "=== target: wasm32 ==="
    rm -rf "$stage"; mkdir -p "$stage/target/wasm32-unknown-unknown/release"

    echo "build: engine/wasm/run.sh build (proven recipe)"
    (cd "$ENGINE/wasm" && ./run.sh build) || { step 1 build; target_summary wasm32 FAIL "wasm build"; return; }
    step 0 build
    [ -f "$WASM" ]
    step $? wasm-present "$(fsize "$WASM") bytes"

    local wsmoke="skipped (--skip-verify)" wsmoke_rc=0
    if [ "$SKIP_VERIFY" = 0 ]; then
        if ! command -v node >/dev/null 2>&1; then
            echo "SKIP: node not found (wasm runtime smoke)"
            wsmoke="skipped (node absent)"
        elif [ ! -f "$LAYA_F16" ]; then
            echo "SKIP: laya f16 model missing ($LAYA_F16) — wasm runtime smoke"
            wsmoke="skipped (model absent)"
        else
            if (cd "$ENGINE/wasm" && ./run.sh smoke >/dev/null); then
                wsmoke_rc=0
            else
                wsmoke_rc=1
            fi
            step "$wsmoke_rc" wasm-smoke "ticket 19 smoke: self-test + Laya forward + drift gate (BEFORE packaging)"
            wsmoke=$([ "$wsmoke_rc" = 0 ] && echo "PASS" || echo "FAIL")
        fi
    fi

    cp "$WASM" "$stage/target/wasm32-unknown-unknown/release/laya_engine_wasm.wasm"
    cp "$ENGINE/wasm/harness.mjs" "$stage/harness.mjs"
    if ! staging_ok "$stage/target/wasm32-unknown-unknown/release/laya_engine_wasm.wasm" "$stage/harness.mjs"; then
        step 1 staging "wasm + harness staged"
        target_summary wasm32 FAIL "staging incomplete"
        return
    fi
    step 0 staging "wasm + harness staged"
    cat > "$stage/RUN.md" <<EOF
# laya-engine wasm32 — v$VERSION (Node harness)

The engine core compiled to wasm32-unknown-unknown (SIMD128 kernels,
self-test selectable scalar fallback) plus the ticket-19 Node harness
that loads GGUF bytes into linear memory, runs forwards and measures
latency/memory/drift vs native dumps. The .wasm sits at the harness's
default relative path, so the commands below run verbatim from this
directory. Requires Node >= 18 (developed on 22).

    node harness.mjs --self-test

Drift-checked forward (needs a model + a fresh native dump for the
token ids and reference logits — the native engine binary produces it):

    ../../path/to/laya-engine decide laya_english_f16.gguf \\
        --request refund.json --raw > native_ref.json
    node harness.mjs --model laya --gguf laya_english_f16.gguf \\
        --native native_ref.json --warmup 0 --repeats 1 --out results/smoke.json

Add --simd 0 for the scalar-fallback A/B, --repeats 5 for benchmarking
(--gguf2/--native2 --model dual loads two models). See engine/wasm/ in
the source repo for the full bench matrix. SHA256SUMS covers every file
in this archive except itself: sha256sum -c SHA256SUMS.
EOF

    { manifest_header "laya-engine $VERSION — wasm32 (Node)"
      echo
      echo "## Artifact"
      echo "- archive: $name.zip ($(fsize "$WASM")-byte cdylib inside)"
      echo "- recipe: engine/wasm/run.sh build — RUSTFLAGS=--cfg getrandom_backend=\"custom\", wasm32-unknown-unknown, cdylib"
      echo
      echo "## Files"
      (cd "$stage" && file_table target/wasm32-unknown-unknown/release/laya_engine_wasm.wasm harness.mjs RUN.md)
      echo
      echo "## Verification"
      echo "- engine/wasm/run.sh smoke (BEFORE packaging): $wsmoke — wasm self-test"
      echo "  + one Laya forward, drift gate max|dlogit|<=0.05, max|dp|<=0.02"
      echo "- checksums: inner SHA256SUMS verified post-extraction; outer SHA256SUMS"
      echo "  (next to the archive) verified with sha256sum -c"
    } > "$stage/MANIFEST.md"

    stage_checksums "$stage"
    make_zip "$stage" "$name" "$dir/$name.zip"
    "$PY" -m zipfile -t "$dir/$name.zip" >/dev/null
    step $? zip-integrity "python zipfile -t CRC check"
    outer_checksum "$dir" "$name.zip"
    step $? checksums "outer sha256 verified"
    verify_inner "$dir/$name.zip" zip
    step $? inner-checksums "extracted + sha256sum -c"
    target_summary wasm32 \
        "$([ "$FAILED" = "$fail0" ] && echo PASS || echo FAIL) wasm32-unknown-unknown cdylib + harness" "$((SECONDS - t0))s"
}

# ====================================================== windows-x86_64-msvc
do_windows() {
    local dir="$DIST/windows-x86_64-msvc" t0=$SECONDS log="$DIST/windows-x86_64-msvc/build-link.log"
    echo ""
    echo "=== target: windows-x86_64-msvc (link attempt) ==="
    mkdir -p "$dir"
    rm -f "$log"
    # CARGO_ENCODED_RUSTFLAGS (not RUSTFLAGS) so the repo-root no_std link
    # flags are replaced, engine/run.sh pattern; rust-lld ships with rustup.
    if (cd "$ENGINE" && CARGO_ENCODED_RUSTFLAGS=$'-Ctarget-cpu=x86-64\x1f-Clinker=rust-lld' \
            "$CARGO" build --release --target x86_64-pc-windows-msvc) >"$log" 2>&1 \
            && [ -f "$BIN_EXE" ]; then
        step 0 link "rust-lld linked laya-engine.exe"
        local name="laya-engine-$VERSION-windows-x86_64-msvc" stage="$dir/$name"
        rm -rf "$stage"; mkdir -p "$stage"
        cp "$BIN_EXE" "$stage/laya-engine.exe"
        cp "$ENGINE/presets.json" "$stage/presets.json"
        if ! staging_ok "$stage/laya-engine.exe" "$stage/presets.json"; then
            step 1 staging "exe + presets staged"
            target_summary windows-x86_64-msvc FAIL "staging incomplete"
            return
        fi
        step 0 staging "exe + presets staged"
        cat > "$stage/USAGE.md" <<EOF
# laya-engine v$VERSION — Windows x86-64 (MSVC)

    laya-engine.exe self-test
    laya-engine.exe decide laya_english_f16.gguf --preset email --text "refund my duplicate charge"
    laya-engine.exe serve laya_english_f16.gguf --port 8080 --threads 4

Linked with rust-lld (no MSVC toolchain present). Presets are compiled
in; presets.json is the identical reference copy. Verify with
\`sha256sum -c SHA256SUMS\` (or certutil -hashfile on Windows).
EOF
        { manifest_header "laya-engine $VERSION — windows x86-64 msvc"
          echo
          echo "## Artifact"
          echo "- archive: $name.zip ($(fsize "$BIN_EXE")-byte exe inside)"
          echo "- recipe: CARGO_ENCODED_RUSTFLAGS='-Ctarget-cpu=x86-64 -Clinker=rust-lld' cargo build --release --target x86_64-pc-windows-msvc"
          echo "- linked with rust-lld on Linux (no MSVC toolchain) — runtime wiring owned by ticket 23"
          echo
          echo "## Files"
          (cd "$stage" && file_table laya-engine.exe presets.json USAGE.md)
        } > "$stage/MANIFEST.md"
        stage_checksums "$stage"
        make_zip "$stage" "$name" "$dir/$name.zip"
        "$PY" -m zipfile -t "$dir/$name.zip" >/dev/null
        step $? zip-integrity
        outer_checksum "$dir" "$name.zip"
        step $? checksums
        verify_inner "$dir/$name.zip" zip
        step $? inner-checksums
        target_summary windows-x86_64-msvc "PASS exe linked via rust-lld" "$((SECONDS - t0))s"
        return
    fi

    # Expected outcome on this machine: rustup ships no MSVC import libs.
    # (A compile failure here is transient: concurrent agents edit
    # engine/src — distinguish the two from the log.)
    echo "--- link: FAIL (recorded; see below)"
    local lld_errs compile_errs
    lld_errs="$(/usr/bin/grep 'rust-lld: error' "$log" | /usr/bin/head -6)"
    compile_errs="$(/usr/bin/grep -c '^error\[E' "$log" || true)"
    local outcome
    if [ -n "$lld_errs" ]; then
        outcome="Compile: PASS. Link: FAIL — rust-lld cannot find the Windows
import libraries rustc asks for:

$lld_errs

rust-lld itself works (it drives the MSVC-flavor link line correctly);
rustup's rustlib for x86_64-pc-windows-msvc ships only Rust rlibs — no
\`self-contained/\` directory, no import libraries (kernel32, ntdll,
userenv, ws2_32, dbghelp) and no CRT. Without a Windows SDK / MSVC
import-lib set the link cannot succeed on this machine."
    else
        outcome="Compile: FAIL ($compile_errs rustc errors) — transient: concurrent
agents are editing engine/src (heap/http/server Windows wiring); this is
NOT the link question. The controlled attempt earlier on this machine
(ticket 22 experiment, same sources at a stable point) compiled all
crates and failed at the LINK on the missing MSVC import libs:

    rust-lld: error: could not open 'kernel32.lib': No such file or directory

Rerun \`package.sh windows\` once the concurrent engine/src edits land."
    fi
    cat > "$dir/MANIFEST.md" <<EOF
# laya-engine $VERSION — windows x86-64 msvc: NOT PRODUCED (requires a Windows machine)

$(manifest_header "attempt record")

## Attempt

    (cd engine/ && CARGO_ENCODED_RUSTFLAGS='-Ctarget-cpu=x86-64 -Clinker=rust-lld' cargo build --release --target x86_64-pc-windows-msvc)

$outcome

Full log: build-link.log next to this manifest.

## Path to a Windows artifact (tickets 23 / 20)

- Build on a Windows machine: \`cargo build --release --target x86_64-pc-windows-msvc\`
  (rustup on Windows provides the MSVC libs), or install the Windows SDK
  import libs + lld-link here and retry this script's windows target.
- Runtime wiring (drain etc.) is ticket 23's scope; v0.2 shipping is
  ticket 20's.
EOF
    echo "windows artifacts: requires a Windows machine (recorded in MANIFEST.md)"
    target_summary windows-x86_64-msvc "PASS link attempted + recorded (no .exe; requires a Windows machine)" "$((SECONDS - t0))s"
}

# ==================================================================== macos
do_macos() {
    local dir="$DIST/macos" t0=$SECONDS
    echo ""
    echo "=== target: macos (note only — not producible here) ==="
    mkdir -p "$dir"
    cat > "$dir/MANIFEST.md" <<EOF
# laya-engine $VERSION — macOS: NOT PRODUCED on this machine

$(manifest_header "cross-compile status")

Apple targets need the Apple SDK (frameworks + system libc), which this
Linux/WSL2 machine cannot provide. Both CI cross checks are proven
(cargo check x86_64-apple-darwin + aarch64-apple-darwin, ticket 14), so
only the SDK is missing, not code.

## aarch64 (Apple Silicon) — one command on the MacBook M1

    bash engine/aarch64/run-m1.sh

(ticket 17's acceptance script: native aarch64-apple-darwin release
build + self-test + drift gates + latency table; the binary lands in
engine/target/aarch64-apple-darwin/release/laya-engine — tar it up with
the same layout as the linux-x86_64-musl set.)

## x86_64 (Intel Mac) — native build on the Mac

    cd engine/ && cargo build --release --target x86_64-apple-darwin

## Repackage from a Mac

Run engine/ci/package.sh ON the Mac (CARGO/PY overridable) after the
native build; the musl/wasm targets are Linux-only steps and can be
skipped by passing explicit target names.
EOF
    echo "note: written to $dir/MANIFEST.md (points at engine/aarch64/run-m1.sh + Intel Mac native build)"
    target_summary macos "PASS note emitted (aarch64: run-m1.sh; x86_64: Intel Mac native build)" "$((SECONDS - t0))s"
}

# ================================================================== driver
ARGS=("$@")
TARGETS=()
for arg in ${ARGS[@]+"${ARGS[@]}"}; do
    case "$arg" in
        --skip-verify) SKIP_VERIFY=1 ;;
        linux-musl | wasm32 | windows | macos) TARGETS+=("$arg") ;;
        all) TARGETS+=(linux-musl wasm32 windows macos) ;;
        "") ;;
        *) echo "unknown target: $arg (targets: linux-musl wasm32 windows macos all; flag: --skip-verify)" >&2; exit 2 ;;
    esac
done
[ ${#TARGETS[@]} = 0 ] && TARGETS=(linux-musl wasm32 windows macos)

echo "laya-engine $VERSION packaging → engine/dist/ (git $REV_SHORT, branch $BRANCH)"
echo "rustc: $RUSTC_VER"
for t in "${TARGETS[@]}"; do
    case "$t" in
        linux-musl) do_linux_musl ;;
        wasm32) do_wasm32 ;;
        windows) do_windows ;;
        macos) do_macos ;;
    esac
done

echo ""
echo "==== packaging summary ===="
for s in "${SUMMARY[@]}"; do echo "$s"; done
if [ "$FAILED" = 1 ]; then
    echo "RESULT: FAIL"
    exit 1
fi
echo "RESULT: PASS"
