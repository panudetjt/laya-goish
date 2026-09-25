#!/usr/bin/env python3
"""Reduced zero-drift gate for the quick CI tier (wayfinder ticket 14).

Same comparison as validation/scalar-tail-drift.py (full decision JSON minus
the `usage` timing block, bit-identical), parameterized to a subset of
cases x thread counts so engine/ci/run.sh quick stays fast. The frozen
42-combo script stays the authority (tier `drift`); this lives in engine/ci/
because validation/ is untouchable.

Usage: reduced-drift.py OLD_BIN NEW_BIN F16_LAYA F16_OPENTHAI [EXAMPLES_DIR]
                          [CASES] [THREADS]
       CASES   comma list of model:case (default laya:refund,openthai:openthai-thai)
       THREADS comma list (default 1,4)
"""
import json
import os
import subprocess
import sys

if len(sys.argv) < 5:
    raise SystemExit(__doc__)
old, new, laya, openthai = sys.argv[1:5]
examples = sys.argv[5] if len(sys.argv) > 5 else "examples"
cases = [c.split(":", 1) for c in (sys.argv[6] if len(sys.argv) > 6 else "laya:refund,openthai:openthai-thai").split(",")]
threads = [int(t) for t in (sys.argv[7] if len(sys.argv) > 7 else "1,4").split(",")]
models = {"laya": laya, "openthai": openthai}


def decide(binary, model, request_path, t):
    out = subprocess.run(
        [binary, "decide", model, "--request", request_path, "--threads", str(t)],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def strip_usage(obj):
    if isinstance(obj, dict):
        return {k: strip_usage(v) for k, v in obj.items() if k != "usage"}
    return obj


breaches = []
checked = 0
for name, req in cases:
    path = os.path.join(examples, f"{req}.json")
    for t in threads:
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
print(f"reduced drift ({checked} combos): PASS (bit-identical)")
