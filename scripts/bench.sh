#!/usr/bin/env bash
# Benchmark laya serve /v1/decide latency.
# Usage: bench.sh <binary-path> <threads> <runs>
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="$1"; THREADS="$2"; RUNS="${3:-10}"
PORT="${PORT:-8080}"

"$BIN" serve models/laya_english_f16.gguf --host 127.0.0.1 --port "$PORT" --threads "$THREADS" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Wait for readiness
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 0.5
done

# Warm-up (model load + page-in)
curl -sf "http://127.0.0.1:$PORT/v1/decide" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/refund.json >/dev/null

echo "== binary=$BIN threads=$THREADS runs=$RUNS =="
for i in $(seq 1 "$RUNS"); do
  curl -sf "http://127.0.0.1:$PORT/v1/decide" \
    -H 'Content-Type: application/json' \
    --data-binary @examples/refund.json \
    | jq -c '{latency_ms: .usage.latency_ms, refund: .answers.refund.noul}'
done
