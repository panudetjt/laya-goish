#!/usr/bin/env python3
"""Bounded memory soak for the std engine's HTTP server — the
parameterized replica of validation/soak_http.py (frozen; it reads
server.pid of an existing goish server on port 8080). Same shape:
five rounds of sequential inference, concurrent overload, mixed HTTP
routes and oversized bodies, then idle observation; RSS/anon/swap/
threads/fd sampling each second. --http-only sends 100,000 mixed
requests (no inference), mirroring the original.

Usage: soak.py ENGINE_BIN LAYA_F16 [--http-only] [--rounds N]
Writes soak.report.json / churn.report.json next to this script.
"""
import argparse
import collections
import concurrent.futures
import http.client
import json
import pathlib
import subprocess
import socket
import threading
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bin", help="engine binary")
parser.add_argument("model", help="laya f16 GGUF")
parser.add_argument("--http-only", action="store_true", help="100,000 mixed HTTP requests without inference")
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()
ROOT = pathlib.Path(__file__).resolve().parents[2]
BIN = pathlib.Path(args.bin)
MODEL = pathlib.Path(args.model)
REPORT = pathlib.Path(__file__).resolve().parent / (
    "churn.report.json" if args.http_only else "soak.report.json"
)
REQ = (ROOT / "examples/refund.json").read_bytes()
OVERSIZE = b"x" * (1024 * 1024 + 1)
stop = threading.Event()
lock = threading.Lock()
counts = collections.Counter()
samples, checkpoints, errors = [], [], []
started = time.monotonic()
phase = "baseline"
finished = False

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    PORT = s.getsockname()[1]
srv_log = open(REPORT.with_suffix(".server.log"), "w")
server = subprocess.Popen(
    [str(BIN), "serve", str(MODEL), "--port", str(PORT), "--threads", str(args.threads)],
    stdout=srv_log,
    stderr=srv_log,
)
for _ in range(1200):
    if server.poll() is not None:
        raise SystemExit(f"soak server exited {server.returncode}")
    try:
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=5)
        c.request("GET", "/health")
        if c.getresponse().read():
            break
    except OSError:
        time.sleep(0.1)
PID = server.pid
PROC = pathlib.Path("/proc") / str(PID)


def fields(path):
    out = {}
    for line in path.read_text().splitlines():
        k, _, v = line.partition(":")
        if not k or not k[0].isalpha():
            continue
        try:
            out[k] = int(v.split()[0])
        except (ValueError, IndexError):
            pass
    return out


def snapshot(full=False):
    s = fields(PROC / "status")
    m = fields(pathlib.Path("/proc/meminfo"))
    out = dict(
        elapsed_s=round(time.monotonic() - started, 2),
        phase=phase,
        rss_kib=s["VmRSS"],
        anon_kib=s["RssAnon"],
        file_kib=s["RssFile"],
        swap_kib=s.get("VmSwap", 0),
        threads=s["Threads"],
        fds=len(list((PROC / "fd").iterdir())),
        available_kib=m["MemAvailable"],
    )
    if full:
        out["smaps_rollup"] = fields(PROC / "smaps_rollup")
    return out


def monitor():
    while not stop.wait(1):
        try:
            s = snapshot()
            samples.append(s)
            if s["available_kib"] < 700 * 1024 or s["rss_kib"] > baseline["rss_kib"] + 3 * 1024 * 1024:
                errors.append("Memory safety threshold reached; stopping load")
                stop.set()
        except Exception as e:
            errors.append(repr(e))
            stop.set()


def checkpoint(label):
    s = snapshot(True)
    s["label"] = label
    checkpoints.append(s)
    print(json.dumps({k: v for k, v in s.items() if k != "smaps_rollup"}), flush=True)


def call(method, path, body=None, expected=(200,), connection=None):
    if stop.is_set():
        raise RuntimeError("load stopped")
    c = connection or http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
    try:
        c.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
        r = c.getresponse()
        payload = r.read()
        with lock:
            counts[str(r.status)] += 1
        if r.status not in expected:
            raise AssertionError((path, r.status, payload[:300]))
        obj = json.loads(payload) if payload else None
        if path in ("/v1/decide", "/v1/systemone") and r.status == 200:
            assert obj["answers"] == answer, ("inference changed", obj)
        return obj
    finally:
        if connection is None:
            c.close()


def mixed_worker(n, persistent):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30) if persistent else None
    routes = [
        ("GET", "/health", None, (200,)),
        ("GET", "/v1/models", None, (200,)),
        ("GET", "/v1/presets", None, (200,)),
        ("GET", "/missing", None, (404,)),
        ("POST", "/v1/decide", b"{bad", (400,)),
        ("POST", "/v1/decide", b"{}", (422,)),
    ]
    try:
        for i in range(n):
            if stop.is_set():
                break
            call(*routes[i % len(routes)], connection=c)
    finally:
        if c:
            c.close()


def parallel(fn, args_list, workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fs = [pool.submit(fn, *a) for a in args_list]
        for f in fs:
            f.result()


def save():
    REPORT.write_text(
        json.dumps(
            dict(
                pid=PID,
                elapsed_s=round(time.monotonic() - started, 2),
                status=("failed" if errors else "passed" if finished else "running"),
                counts=dict(counts),
                errors=errors,
                checkpoints=checkpoints,
                samples=samples,
            ),
            indent=2,
        )
        + "\n"
    )


try:
    if args.http_only:
        answer = None
    else:
        answer = json.loads(
            subprocess.check_output(
                [str(BIN), "decide", str(MODEL), "--request", str(ROOT / "examples/refund.json")],
                stderr=subprocess.DEVNULL,
            )
        )["answers"]
    baseline = snapshot(True)
    checkpoint("initial")
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    phase = "warmup"
    for _ in range(0 if args.http_only else 5):
        call("POST", "/v1/decide", REQ)
    checkpoint("warm")
    for round_no in range(1, args.rounds + 1):
        phase = f"round-{round_no}-inference"
        for _ in range(0 if args.http_only else 20):
            call("POST", "/v1/decide", REQ)
        checkpoint(f"round-{round_no}-inference")
        phase = f"round-{round_no}-http"
        parallel(mixed_worker, [(1250 if args.http_only else 250, round_no % 2 == 0)] * 16, 16)
        phase = f"round-{round_no}-concurrent-inference"
        if not args.http_only:
            parallel(call, [("POST", "/v1/decide", REQ, (200, 429))] * 32, 8)
        phase = f"round-{round_no}-oversized"
        parallel(call, [("POST", "/v1/decide", OVERSIZE, (413,))] * 32, 8)
        time.sleep(2)
        checkpoint(f"round-{round_no}-end")
        save()
    phase = "idle"
    for i in range(3):
        time.sleep(10)
        checkpoint(f"idle-{(i + 1) * 10}s")
    call("GET", "/health")
    finished = True
except Exception as e:
    errors.append(repr(e))
    print("FAIL", repr(e), flush=True)
finally:
    stop.set()
    thread.join(timeout=2)
    server.terminate()
    try:
        server.wait(timeout=40)
    except subprocess.TimeoutExpired:
        server.kill()
    srv_log.close()
    save()
    print(json.dumps(dict(report=str(REPORT), counts=dict(counts), errors=errors, server_rc=server.returncode)), flush=True)
if errors or server.returncode != 0:
    raise SystemExit(1)
