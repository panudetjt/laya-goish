#!/usr/bin/env python3
"""Measure Q8_0 vs F16 decision drift against the acceptance gates.

Usage: quantized-drift.py LAYA_BIN F16_LAYA Q8_LAYA F16_OPENTHAI Q8_OPENTHAI [EXAMPLES_DIR]

Gates declared in .scratch/inference-perf/issues/02-phase2-follow-ups.md:
no discrete choice flips, max |probability delta| <= 0.02, max |score delta|
<= 0.1 across the reference requests. Exits non-zero on any breach.
"""
import json
import subprocess
import sys
import os

REQUESTS = {
    "laya": ["refund", "urgent-score", "openthai-thai"],
    "openthai": ["refund", "openthai-thai", "openthai-mixed"],
}
PROB_GATES = 0.02
SCORE_GATE = 0.1


def decide(binary, model, request_path):
    out = subprocess.run(
        [binary, "decide", model, "--request", request_path, "--threads", "7"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def flat(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{path}.{k}")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, float(o)


def main():
    if len(sys.argv) < 6:
        raise SystemExit(__doc__)
    binary, f16_laya, q8_laya, f16_openthai, q8_openthai = sys.argv[1:6]
    examples = sys.argv[6] if len(sys.argv) > 6 else "examples"
    pairs = {
        "laya": (f16_laya, q8_laya),
        "openthai": (f16_openthai, q8_openthai),
    }
    breaches = []
    worst_prob = (0.0, "")
    worst_score = (0.0, "")
    for name, (f16, q8) in pairs.items():
        for req in REQUESTS[name]:
            path = os.path.join(examples, f"{req}.json")
            a = dict(flat(decide(binary, f16, path)))
            b = dict(flat(decide(binary, q8, path)))
            for k in a:
                if k not in b:
                    breaches.append(f"{name}/{req}: missing field {k}")
                    continue
                d = abs(a[k] - b[k])
                if k.endswith(".choice") and a[k] != b[k]:
                    breaches.append(f"{name}/{req}: choice flip {k}: {a[k]} -> {b[k]}")
                if ("probabilit" in k or k.endswith((".noul", ".confidence"))) and d > worst_prob[0]:
                    worst_prob = (d, f"{name}/{req}{k}")
                if k.endswith(".score"):
                    if d > worst_score[0]:
                        worst_score = (d, f"{name}/{req}{k}")
                    if d > SCORE_GATE:
                        breaches.append(f"{name}/{req}: score delta {d:.4f} > {SCORE_GATE} at {k}")
    if worst_prob[0] > PROB_GATES:
        breaches.append(f"probability delta {worst_prob[0]:.5f} > {PROB_GATES} at {worst_prob[1]}")
    print(f"max |dProb| = {worst_prob[0]:.5f} ({worst_prob[1] or 'n/a'})")
    print(f"max |dScore| = {worst_score[0]:.5f} ({worst_score[1] or 'n/a'})")
    if breaches:
        for b in breaches:
            print(f"BREACH: {b}")
        raise SystemExit(1)
    print("quantized drift: PASS")


if __name__ == "__main__":
    main()
