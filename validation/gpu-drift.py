#!/usr/bin/env python3
"""GPU sidecar drift gates vs the CPU F16 engine (ticket 03).

Reference set — DECLARED BEFORE MEASURING (all on models/laya_english_f16.gguf,
covering all three question kinds, English + Thai, typed states, unicode):
  1. examples/refund.json            (noul)
  2. examples/urgent-score.json      (score + choice)
  3. examples/openthai-thai.json     (choice + score + noul, Thai state)
  4. examples/openthai-mixed.json    (mixed questions)
  5. validation/types.request.json   (typed/JSON state)
  6. validation/unicode.request.json (unicode edge cases)

Gates (declared in .scratch/acceleration/issues/03-gpu-sidecar-poc.md,
acknowledging GPU FMA order differs from the CPU kernels):
  - zero discrete choice flips
  - max |dProbability| <= 0.01   (probabilities.*, .noul, .confidence)
  - max |dScore|     <= 0.05     (.score fields)

The CPU reference is `./laya decide <f16> --request X --threads 16 --raw`,
which reports input_ids/markers/logits and the full decision JSON. The GPU
sidecar replays the SAME token ids and markers through its socket; the
harness recomputes the decision layer (temperature + softmax + score) from
the GPU logits with the engine's exact temperature selection rules and
compares at the decision level.

Usage: gpu-drift.py [--sock PATH] [--laya-bin ./laya] [--model models/laya_english_f16.gguf]
Exit: non-zero on any gate breach.
"""
import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [
    "examples/refund.json",
    "examples/urgent-score.json",
    "examples/openthai-thai.json",
    "examples/openthai-mixed.json",
    "validation/types.request.json",
    "validation/unicode.request.json",
]
PROB_GATE = 0.01
SCORE_GATE = 0.05


def sidecar_request(sock_path, payload, timeout=120.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock_path)
    s.sendall((json.dumps(payload) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(1 << 20)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode())


def cpu_decide(binary, model, request_path, threads):
    out = subprocess.run(
        [binary, "decide", model, "--request", request_path,
         "--threads", str(threads), "--raw"],
        capture_output=True, text=True, check=True, cwd=PROJ,
    )
    return json.loads(out.stdout)


def softmax(xs):
    m = max(xs)
    es = [math.exp(v - m) for v in xs]
    total = sum(es)
    return [e / total for e in es]


def temperature(kind, k, base, buckets):
    bucket = f"{kind}:" + ("2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+")
    t = buckets.get(bucket, base[kind])
    return max(t, 1e-3)


def gpu_decision(logits, kind, base, buckets):
    """Recompute the CPU engine's decision layer from raw logits."""
    k = len(logits)
    t = temperature(kind, k, base, buckets)
    p = softmax([v / t for v in logits])
    out = {}
    out["p"] = p  # ordered option probabilities
    if kind == "choice":
        # Engine tie-break (decision.rs max_by with then_with(b.cmp(&a))):
        # larger index wins on exact ties.
        best = max(range(k), key=lambda i: (p[i], i))
        out["choice"] = best
    elif kind == "score":
        out["score"] = sum(i * p[i] for i in range(k))
    else:
        out["noul"] = p[1]
        out["confidence"] = max(p[1], 1.0 - p[1])
        return out
    if kind != "noul":
        n = len(p)
        e = -sum(max(v, 1e-12) * math.log(max(v, 1e-12)) for v in p)
        out["confidence"] = min(max(1.0 - e / math.log(n), 0.0), 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/laya-gpu-sidecar.sock")
    ap.add_argument("--laya-bin", default="./laya")
    ap.add_argument("--model", default="models/laya_english_f16.gguf")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--sidecar-bin",
                    default="sidecar/target/x86_64-unknown-linux-gnu/release/laya-gpu-sidecar")
    args = ap.parse_args()

    # Engine temperatures from the model metadata (via the sidecar's info).
    info = subprocess.run(
        [args.sidecar_bin, "info", "--laya", args.model],
        capture_output=True, text=True, check=True, cwd=PROJ,
    )
    meta = json.loads(info.stdout)
    base = {
        "choice": meta["temperature"][0],
        "score": meta["temperature"][1],
        "noul": meta["temperature"][2],
    }
    buckets = {k: v for k, v in meta["temperature_by_options"].items()}

    # Ping the sidecar (it must already be running with the same model).
    pong = sidecar_request(args.sock, {"ping": 1})
    print(f"sidecar: {pong}")
    if not pong.get("pong"):
        raise SystemExit("sidecar did not answer ping")

    breaches = []
    worst_prob = (0.0, "")
    worst_score = (0.0, "")
    total_questions = 0

    for case in CASES:
        path = os.path.join(PROJ, case)
        ref = cpu_decide(args.laya_bin, args.model, path, args.threads)
        for qid, ans in ref["answers"].items():
            total_questions += 1
            kind = ans["type"]
            qtype = {"choice": 0, "score": 1, "noul": 2}[kind]
            ids = ans["input_ids"]
            markers = ans["markers"]
            resp = sidecar_request(args.sock, {
                "ids": ids, "markers": markers, "qtype": qtype,
            })
            if "error" in resp:
                breaches.append(f"{case}/{qid}: sidecar error: {resp['error']}")
                continue
            gpu = gpu_decision(resp["logits"], kind, base, buckets)
            label = f"{os.path.basename(case)}/{qid}"

            if kind == "choice":
                # CPU choice is the label text; the GPU choice index maps to
                # the same ordered options list (criteria insertion order).
                gpu_choice_label = list(ans["probabilities"].keys())[gpu["choice"]]
                if gpu_choice_label != ans["choice"]:
                    breaches.append(
                        f"{label}: choice flip {ans['choice']} -> {gpu_choice_label}")
            # probabilities: compare in the CPU's label order
            labels = list(ans["probabilities"].keys())
            for i, lab in enumerate(labels):
                d = abs(gpu["p"][i] - ans["probabilities"][lab])
                if d > worst_prob[0]:
                    worst_prob = (d, f"{label}.probabilities.{lab}")
            # noul / confidence
            for field in ("noul", "confidence"):
                if field in ans:
                    d = abs(gpu[field] - ans[field])
                    if d > worst_prob[0]:
                        worst_prob = (d, f"{label}.{field}")
            # score
            if kind == "score":
                d = abs(gpu["score"] - ans["score"])
                if d > worst_score[0]:
                    worst_score = (d, f"{label}.score")
                if d > SCORE_GATE:
                    breaches.append(f"{label}: score delta {d:.4f} > {SCORE_GATE}")
            # logits sanity vs the CPU raw logits (informational, not gated)
            dl = max(abs(a - b) for a, b in zip(resp["logits"], ans["logits"]))
            da = max(abs(a - b) for a, b in zip(resp["act"], ans["act_logits"]))
            print(f"  {label:<38} kind={kind:<6} |dlogit|={dl:.2e} |dact|={da:.2e}")

    if worst_prob[0] > PROB_GATE:
        breaches.append(
            f"probability delta {worst_prob[0]:.5f} > {PROB_GATE} at {worst_prob[1]}")
    print()
    print(f"questions compared : {total_questions} across {len(CASES)} cases")
    print(f"max |dProbability| = {worst_prob[0]:.5f} (gate {PROB_GATE}) at {worst_prob[1] or 'n/a'}")
    print(f"max |dScore|       = {worst_score[0]:.5f} (gate {SCORE_GATE}) at {worst_score[1] or 'n/a'}")
    if breaches:
        for b in breaches:
            print(f"BREACH: {b}")
        raise SystemExit(1)
    print("gpu sidecar drift: PASS")


if __name__ == "__main__":
    main()
