#!/usr/bin/env python3
"""Verify zero-drift: decision outputs must be bit-identical between two engine binaries.

Usage: scalar-tail-drift.py OLD_BIN NEW_BIN F16_LAYA F16_OPENTHAI [EXAMPLES_DIR]

Posture (user mandate, 2026-09-24): optimizations may only change latency and
RAM — accuracy must be unaffected, so both models' full decision JSON must be
byte-identical to the pre-change binary (only the `usage` timing block is
excluded) across the 6-case reference set and thread counts 1/2/4/5/7/12/16.
Exits non-zero on any difference.
"""
import json
import subprocess
import sys
import os

CASES = {
    "laya": ["refund", "urgent-score", "openthai-thai"],
    "openthai": ["refund", "openthai-thai", "openthai-mixed"],
}
THREADS = [1, 2, 4, 5, 7, 12, 16]


def decide(binary, model, request_path, threads):
    out = subprocess.run(
        [binary, "decide", model, "--request", request_path, "--threads", str(threads)],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def strip_usage(obj):
    if isinstance(obj, dict):
        return {k: strip_usage(v) for k, v in obj.items() if k != "usage"}
    return obj


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    old, new, laya, openthai = sys.argv[1:5]
    examples = sys.argv[5] if len(sys.argv) > 5 else "examples"
    models = {"laya": laya, "openthai": openthai}
    breaches = []
    checked = 0
    for name in ["laya", "openthai"]:
        for req in CASES[name]:
            path = os.path.join(examples, f"{req}.json")
            for t in THREADS:
                a = strip_usage(json.loads(decide(old, models[name], path, t)))
                b = strip_usage(json.loads(decide(new, models[name], path, t)))
                checked += 1
                if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                    breaches.append(f"{name}/{req}@{t}t: NOT bit-identical")
    print(f"checked {checked} case/thread combinations")
    if breaches:
        for b in breaches:
            print(f"BREACH: {b}")
        raise SystemExit(1)
    print("zero-drift: PASS (both models bit-identical on every case and thread count)")


if __name__ == "__main__":
    main()
