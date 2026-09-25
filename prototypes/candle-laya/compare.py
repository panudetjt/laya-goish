#!/usr/bin/env python3
"""Drift gates for the candle-laya prototype vs the shipped engine (ticket 05).

Semantics mirror validation/gpu-drift.py + quantized-drift.py:
  reference : ./laya decide <f16> --request CASE --threads N --raw  (saved in
              reference/*.json; contains input_ids/markers/logits/probabilities)
  candidate : candle-laya run MODEL --case reference/X.json (replays the SAME
              token ids and markers; recomputes the decision layer)
  gates     : 0 discrete choice flips, max |dProb| <= 0.02, max |dScore| <= 0.10

Usage: compare.py [PROTOTYPE_BIN] [MODEL] [THREADS]
Exit: non-zero on any gate breach.
"""
import json
import math
import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROTO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    PROJ, "prototypes/candle-laya/target/x86_64-unknown-linux-gnu/release/candle-laya")
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PROJ, "models/laya_english_f16.gguf")
THREADS = sys.argv[3] if len(sys.argv) > 3 else "12"
REF = os.path.join(PROJ, "prototypes/candle-laya/reference")
CASES = ["refund", "urgent-score", "openthai-thai", "openthai-mixed", "types", "unicode"]
PROB_GATE = 0.02
SCORE_GATE = 0.10


def main():
    mode = os.environ.get("WEIGHT_MODE", "f16")
    breaches = []
    worst_prob = (0.0, "")
    worst_score = (0.0, "")
    rows = []
    total_q = 0
    for case in CASES:
        ref_path = os.path.join(REF, f"{case}.json")
        ref = json.load(open(ref_path))
        out = subprocess.run(
            [PROTO, "run", MODEL, "--case", ref_path,
             "--threads", THREADS, "--mode", mode],
            capture_output=True, text=True, check=True, cwd=PROJ,
        )
        got = json.loads(out.stdout)
        for qid, ans_ref in ref["answers"].items():
            total_q += 1
            ans_got = got["answers"][qid]
            kind = ans_ref["type"]
            label = f"{case}/{qid}"
            row = {"case": label, "kind": kind}
            if kind == "choice":
                if ans_got.get("choice") != ans_ref["choice"]:
                    breaches.append(
                        f"{label}: choice flip {ans_ref['choice']} -> {ans_got.get('choice')}")
                row["choice_ref"] = ans_ref["choice"]
                row["choice_got"] = ans_got.get("choice")
            labels = list(ans_ref["probabilities"].keys())
            for i, lab in enumerate(labels):
                d = abs(ans_got["probabilities"][lab] - ans_ref["probabilities"][lab])
                if d > worst_prob[0]:
                    worst_prob = (d, f"{label}.probabilities.{lab}")
            for field in ("noul", "confidence"):
                if field in ans_ref:
                    d = abs(ans_got[field] - ans_ref[field])
                    if d > worst_prob[0]:
                        worst_prob = (d, f"{label}.{field}")
            if kind == "score":
                d = abs(ans_got["score"] - ans_ref["score"])
                row["score_ref"] = ans_ref["score"]
                row["score_got"] = ans_got["score"]
                if d > worst_score[0]:
                    worst_score = (d, f"{label}.score")
                if d > SCORE_GATE:
                    breaches.append(f"{label}: score delta {d:.4f} > {SCORE_GATE}")
            dl = max(abs(a - b) for a, b in zip(ans_got["logits"], ans_ref["logits"]))
            da = max(abs(a - b) for a, b in zip(ans_got["act_logits"], ans_ref["act_logits"]))
            row["max_dlogit"] = dl
            row["max_dact"] = da
            rows.append(row)
    for r in rows:
        extra = ""
        if "choice_ref" in r:
            extra = f" choice {r['choice_ref']}->{r['choice_got']}"
        if "score_ref" in r:
            extra += f" score {r['score_ref']:.4f}->{r['score_got']:.4f}"
        print(f"  {r['case']:<34} kind={r['kind']:<6} |dlogit|={r['max_dlogit']:.2e} "
              f"|dact|={r['max_dact']:.2e}{extra}")
    print()
    print(f"questions compared : {total_q} across {len(CASES)} cases (mode={mode})")
    print(f"max |dProbability| = {worst_prob[0]:.5f} (gate {PROB_GATE}) at {worst_prob[1] or 'n/a'}")
    print(f"max |dScore|       = {worst_score[0]:.5f} (gate {SCORE_GATE}) at {worst_score[1] or 'n/a'}")
    flips = [b for b in breaches if "flip" in b]
    print(f"choice flips       : {len(flips)}")
    if worst_prob[0] > PROB_GATE:
        breaches.append(f"probability delta {worst_prob[0]:.5f} > {PROB_GATE}")
    if breaches:
        for b in breaches:
            print(f"BREACH: {b}")
        raise SystemExit(1)
    print("candle-laya drift: PASS")


if __name__ == "__main__":
    main()
