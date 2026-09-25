#!/usr/bin/env python3
"""Drift gate for cross-ISA engine outputs (wayfinder ticket 17).

Compares a `laya-engine decide --raw` run against a reference dump produced
by the x86-64 engine binary on the same model + case. Cross-ISA outputs are
NEVER bit-identical (map decision): the gates are the map's tolerance bars —

  max |Δlogit|      ≤ 0.05   (logits + act_logits)
  max |Δp|          ≤ 0.02   (post-temperature probabilities / scores)
  choice flips      = 0

usage: drift.py GOT.json REF.json --label LABEL
exit 0 = PASS, 1 = breach (message on stdout)
"""
import json
import sys

DLOGIT_MAX = 0.05
DP_MAX = 0.02


def num(x):
    return float(x) if isinstance(x, (int, float)) else None


def main():
    argv = sys.argv[1:]
    label = "case"
    args = []
    it = iter(argv)
    for a in it:
        if a == "--label":
            label = next(it, "case")
        elif a.startswith("--label="):
            label = a[8:]
        else:
            args.append(a)
    if len(args) != 2:
        print(__doc__)
        return 2
    got = json.load(open(args[0]))
    ref = json.load(open(args[1]))
    ga, ra = got.get("answers", {}), ref.get("answers", {})
    if list(ga.keys()) != list(ra.keys()):
        print(f"FAIL {label}: answer ids differ: {list(ga)} vs {list(ra)}")
        return 1
    max_dlogit = max_dact = 0.0
    max_dp = 0.0
    flips = 0
    for qid in ra:
        g, r = ga[qid], ra[qid]
        for key in ("logits", "act_logits"):
            gl, rl = g.get(key), r.get(key)
            if gl is None and rl is None:
                continue  # answer type without raw logits (both sides)
            if gl is None or rl is None or len(gl) != len(rl):
                print(f"FAIL {label}/{qid}: {key} missing or length mismatch")
                return 1
            d = max((abs(float(a) - float(b)) for a, b in zip(gl, rl)), default=0.0)
            if key == "logits":
                max_dlogit = max(max_dlogit, d)
            else:
                max_dact = max(max_dact, d)
        gp, rp = g.get("probabilities", {}), r.get("probabilities", {})
        if list(gp.keys()) != list(rp.keys()):
            print(f"FAIL {label}/{qid}: probability labels differ")
            return 1
        max_dp = max(
            [max_dp]
            + [abs(float(gp[k]) - float(rp[k])) for k in rp]
        )
        if "choice" in r and g.get("choice") != r.get("choice"):
            flips += 1
        for key in ("score", "noul", "confidence"):
            if num(r.get(key)) is not None:
                max_dp = max(max_dp, abs(num(g.get(key)) - num(r.get(key))))
    ok = max_dlogit <= DLOGIT_MAX and max_dact <= DLOGIT_MAX and max_dp <= DP_MAX and flips == 0
    verdict = "PASS" if ok else "FAIL"
    print(
        f"{verdict} {label}: max|dlogit|={max_dlogit:.3e} "
        f"max|dact|={max_dact:.3e} max|dp|={max_dp:.3e} flips={flips}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
