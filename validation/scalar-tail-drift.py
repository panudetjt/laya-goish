#!/usr/bin/env python3
"""Measure decision drift between two engine binaries (ticket 05 scalar-tail round).

Usage: scalar-tail-drift.py OLD_BIN NEW_BIN F16_LAYA F16_OPENTHAI [EXAMPLES_DIR]

Gates declared in .scratch/acceleration/issues/05-cpu-scalar-tail.md before
measuring: OpenThai outputs byte-identical to the old binary (conv1d channel
partitioning is bit-exact) except `usage`; Laya outputs (polynomial GELU +
SIMD norm reductions) within 5e-4 on every reported probability, confidence,
score, and entropy, with zero discrete flips (choice strings, booleans).
Exits non-zero on any breach.
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
PROB_GATE = 5e-4


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


def flat(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f"{path}[{i}]")
    else:
        yield path, o


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    old, new, laya, openthai = sys.argv[1:5]
    examples = sys.argv[5] if len(sys.argv) > 5 else "examples"
    models = {"laya": laya, "openthai": openthai}
    breaches = []
    worst = (0.0, "")
    for name in ["laya", "openthai"]:
        for req in CASES[name]:
            path = os.path.join(examples, f"{req}.json")
            for t in THREADS:
                a = decide(old, models[name], path, t)
                b = decide(new, models[name], path, t)
                sa, sb = strip_usage(json.loads(a)), strip_usage(json.loads(b))
                if json.dumps(sa, sort_keys=True) != json.dumps(sb, sort_keys=True):
                    if name == "openthai":
                        breaches.append(f"openthai/{req}@{t}t: NOT bit-identical")
                        continue
                    da, db = dict(flat(sa)), dict(flat(sb))
                    if set(da) != set(db):
                        breaches.append(f"laya/{req}@{t}t: field set changed")
                        continue
                    for k in da:
                        va, vb = da[k], db[k]
                        if isinstance(va, str) or isinstance(va, bool) or va is None:
                            if va != vb:
                                breaches.append(f"laya/{req}@{t}t: discrete flip {k}: {va} -> {vb}")
                        elif isinstance(va, (int, float)):
                            if ("probabilit" in k or k.endswith((".noul", ".confidence", ".score", ".entropy"))):
                                d = abs(float(va) - float(vb))
                                if d > worst[0]:
                                    worst = (d, f"laya/{req}@{t}t{k}")
                            elif float(va) != float(vb):
                                breaches.append(f"laya/{req}@{t}t: non-gated numeric changed {k}: {va} -> {vb}")
    if worst[0] > PROB_GATE:
        breaches.append(f"probability delta {worst[0]:.6f} > {PROB_GATE} at {worst[1]}")
    print(f"max gated |delta| = {worst[0]:.6f} ({worst[1] or 'n/a'}), gate {PROB_GATE}")
    if breaches:
        for b in breaches:
            print(f"BREACH: {b}")
        raise SystemExit(1)
    print("scalar-tail drift: PASS (OpenThai bit-identical, Laya within gates)")


if __name__ == "__main__":
    main()
