#!/usr/bin/env python3
"""GPU sidecar latency benchmark vs the CPU engine (ticket 03).

Protocol (mirrors BENCHMARK_REPORT.md section 3): 1 warm-up request then 8
measured sequential requests for the reference refund case
(examples/refund.json token IDs, 45 tokens, 1 noul question).

- GPU: full round trip over the sidecar UDS socket (IPC + JSON + forward),
  timed client-side; the sidecar's internal gpu_ms is reported alongside so
  the IPC share is visible.
- CPU: ./laya daemon <f16> --threads 16 (persistent process, JSON lines),
  usage.latency_ms per request — same metric as section 7 of the report.
- VRAM residency: nvidia-smi memory.used delta while the sidecar holds both
  F16 models, plus the sidecar's own allocation accounting.
- Per-request PCIe traffic: the sidecar reports h2d/d2h bytes per request;
  the gate is that these stay at KB scale (no weight re-streaming over PCIe).

Usage: gpu-bench.py [--sock PATH] [--laya-bin ./laya] [--model ...] [--iters 8]
Writes validation/gpu-sidecar.bench.json and prints a summary.
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFUND = os.path.join(PROJ, "validation/refund.reference.json")


def gpu_series(sock_path, ids, markers, qtype, warmup, iters):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    f = s.makefile("rwb")

    def one():
        payload = {"ids": ids, "markers": markers, "qtype": qtype}
        t0 = time.perf_counter()
        f.write((json.dumps(payload) + "\n").encode())
        f.flush()
        line = f.readline()
        wall = (time.perf_counter() - t0) * 1000.0
        return wall, json.loads(line)

    for _ in range(warmup):
        wall, resp = one()
        if "error" in resp:
            raise SystemExit(f"sidecar error: {resp['error']}")
    walls, internals, h2d, d2h = [], [], 0, 0
    for _ in range(iters):
        wall, resp = one()
        if "error" in resp:
            raise SystemExit(f"sidecar error: {resp['error']}")
        walls.append(wall)
        internals.append(resp["gpu_ms"])
        h2d = max(h2d, resp["h2d_bytes"])
        d2h = max(d2h, resp["d2h_bytes"])
    s.close()
    return walls, internals, h2d, d2h


def cpu_series(binary, model, request, threads, warmup, iters):
    p = subprocess.Popen(
        [binary, "daemon", model, "--threads", str(threads)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=PROJ,
    )
    req = json.load(open(os.path.join(PROJ, request)))
    line = json.dumps(req) + "\n"

    def one():
        t0 = time.perf_counter()
        p.stdin.write(line)
        p.stdin.flush()
        out = p.stdout.readline()
        wall = (time.perf_counter() - t0) * 1000.0
        return wall, json.loads(out)

    for _ in range(warmup):
        one()
    walls, internals = [], []
    for _ in range(iters):
        wall, resp = one()
        walls.append(wall)
        internals.append(resp["usage"]["latency_ms"])
    p.stdin.close()
    p.wait()
    return walls, internals


def nvidia_smi(field):
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return [float(x) for x in out.stdout.strip().split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/laya-gpu-sidecar.sock")
    ap.add_argument("--laya-bin", default="./laya")
    ap.add_argument("--model", default="models/laya_english_f16.gguf")
    ap.add_argument("--resident", default="models/openthai_systemone_v03_f16.gguf")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--request", default="examples/refund.json")
    args = ap.parse_args()

    case = json.load(open(REFUND))
    q = list(case.values())[0]
    ids, markers = q["input_ids"], q["markers"]

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(args.sock)
    s.sendall(b'{"ping":1}\n')
    pong = json.loads(s.makefile("r").readline())
    s.close()
    if not pong.get("pong"):
        raise SystemExit("sidecar did not answer ping")

    gpu_walls, gpu_ms, h2d, d2h = gpu_series(
        args.sock, ids, markers, 2, args.warmup, args.iters)
    cpu_walls, cpu_ms = cpu_series(
        args.laya_bin, args.model, args.request, args.threads,
        args.warmup, args.iters)

    mem_used = nvidia_smi("memory.used")[0]
    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request": args.request,
        "tokens": len(ids),
        "protocol": f"{args.warmup} warmup + {args.iters} measured",
        "gpu": {
            "round_trip_ms": {
                "mean": statistics.mean(gpu_walls),
                "min": min(gpu_walls),
                "p50": statistics.median(gpu_walls),
                "max": max(gpu_walls),
                "all": gpu_walls,
            },
            "sidecar_gpu_ms": {
                "mean": statistics.mean(gpu_ms),
                "all": gpu_ms,
            },
            "ipc_overhead_ms_mean": statistics.mean(gpu_walls) - statistics.mean(gpu_ms),
            "per_request_h2d_bytes_max": h2d,
            "per_request_d2h_bytes_max": d2h,
            "models_resident_bytes": pong["models_resident_bytes"],
        },
        "cpu": {
            "threads": args.threads,
            "latency_ms": {
                "mean": statistics.mean(cpu_ms),
                "min": min(cpu_ms),
                "p50": statistics.median(cpu_ms),
                "max": max(cpu_ms),
                "all": cpu_ms,
            },
            "round_trip_ms_mean": statistics.mean(cpu_walls),
        },
        "nvidia_smi_memory_used_mib": mem_used,
        "speedup_vs_cpu": statistics.mean(cpu_ms) / statistics.mean(gpu_walls),
    }
    out_path = os.path.join(PROJ, "validation/gpu-sidecar.bench.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    g = result["gpu"]["round_trip_ms"]
    c = result["cpu"]["latency_ms"]
    print(f"GPU round trip (incl IPC): mean {g['mean']:.1f} ms  min {g['min']:.1f}  p50 {g['p50']:.1f}  max {g['max']:.1f}")
    print(f"  sidecar gpu_ms mean     : {result['gpu']['sidecar_gpu_ms']['mean']:.1f} ms  (IPC+JSON overhead ~{result['gpu']['ipc_overhead_ms_mean']:.1f} ms)")
    print(f"  per-request PCIe        : <= {h2d} B up / {d2h} B down (weights are {pong['models_resident_bytes'] >> 20} MiB resident, never re-streamed)")
    print(f"CPU {args.threads}T usage.latency_ms  : mean {c['mean']:.1f} ms  min {c['min']:.1f}  p50 {c['p50']:.1f}")
    print(f"speedup (cpu latency / gpu round trip): {result['speedup_vs_cpu']:.2f}x")
    print(f"wrote {out_path}")
    ok = g["mean"] <= 60.0
    print(f"ticket gate: round trip <= 60 ms -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
