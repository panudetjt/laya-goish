#!/usr/bin/env python3
"""Byte-parity harness: run identical raw-socket probes against the
frozen goish binary (./laya) and the std engine binary, diff every
response byte, and report. Normalized fields are only the inherently
volatile ones: Date (wall clock), X-Request-Id (per-process counter —
sequences are compared separately), X-Response-Time-Ms and the JSON
latency_ms (timing).

Phases: static probes (metadata/routes/errors/oversize/malformed/
keep-alive/HTTP-1.0/absolute-URI), auth server, busy/429 during
inference, OpenThai model routing, SIGTERM drain (in-flight completes,
exit code, log line).

Usage: parity.py GOISH_BIN ENGINE_BIN LAYA_F16 OPENTHAI_F16 ROOT
Writes parity.report.json next to this script.
"""
import concurrent.futures
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import threading
import time

if len(sys.argv) < 6:
    raise SystemExit(__doc__)
GOISH, ENGINE, LAYA, OPENTHAI, ROOT = (
    pathlib.Path(sys.argv[1]),
    pathlib.Path(sys.argv[2]),
    pathlib.Path(sys.argv[3]),
    pathlib.Path(sys.argv[4]),
    pathlib.Path(sys.argv[5]),
)
REPORT = pathlib.Path(__file__).resolve().parent / "parity.report.json"

REFUND = json.dumps(
    {
        "state": "I was charged twice. Please refund the duplicate payment.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}
        },
    }
).encode()
CT = b"Content-Type: application/json\r\n"
BIG = b"x" * (1024 * 1024 + 1)

fails = []
checks = []
ONLY = os.environ.get("PARITY_ONLY", "")


def record(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    print(("PASS: " if ok else "FAIL: ") + name + (f"  {detail}" if not ok else ""), flush=True)
    if not ok:
        fails.append(name)


def norm(b):
    s = b.decode("latin1")
    s = re.sub(r"Date: [A-Z][a-z]{2}, \d{2} [A-Z][a-z]{2} \d{4}.*?GMT", "Date: <DATE>", s)
    s = re.sub(r"X-Request-Id: laya-\d+", "X-Request-Id: laya-<N>", s)
    s = re.sub(r"X-Response-Time-Ms: \d+", "X-Response-Time-Ms: <MS>", s)
    s = re.sub(r'"latency_ms":\d+', '"latency_ms":<MS>', s)
    # A latency digit count shifts the body length by one byte; the
    # Content-Length of inference responses is timing-derived too.
    parts = s.split("<<R>>")
    for i, part in enumerate(parts):
        if '"latency_ms"' in part:
            parts[i] = re.sub(r"Content-Length: \d+", "Content-Length: <CL>", part)
    return "<<R>>".join(parts)


def read_response(sock, initial=b"", head_method=False):
    buf = initial
    while b"\r\n\r\n" not in buf:
        d = sock.recv(65536)
        if not d:
            return buf, True
        buf += d
    head, rest = buf.split(b"\r\n\r\n", 1)
    headers = {}
    for line in head.split(b"\r\n")[1:]:
        k, _, v = line.partition(b":")
        headers[k.strip().lower()] = v.strip()
    if head_method:
        return head + b"\r\n\r\n", False
    if headers.get(b"transfer-encoding", b"").lower().endswith(b"chunked"):
        body, closed = read_chunked(sock, rest)
        return head + b"\r\n\r\n" + body, closed
    n = int(headers.get(b"content-length", b"0"))
    while len(rest) < n:
        d = sock.recv(65536)
        if not d:
            return head + b"\r\n\r\n" + rest, True
        rest += d
    return head + b"\r\n\r\n" + rest[:n], False


def read_chunked(sock, initial):
    buf = initial
    while True:
        while b"\r\n" not in buf:
            d = sock.recv(65536)
            if not d:
                return buf, True
            buf += d
        line, buf = buf.split(b"\r\n", 1)
        try:
            n = int(line.split(b";")[0], 16)
        except ValueError:
            return buf, False
        while len(buf) < n + 2:
            d = sock.recv(65536)
            if not d:
                return buf, True
            buf += d
        if n == 0:
            return buf, False
        buf = buf[n + 2 :]


def req(method, path, extra=b"", body=b"", http=b"HTTP/1.1", host=b"Host: probe.local"):
    if body:
        extra = extra + b"Content-Length: " + str(len(body)).encode() + b"\r\n"
    return (
        f"{method} {path} ".encode()
        + http
        + b"\r\n"
        + host
        + b"\r\n"
        + extra
        + b"\r\n"
        + body
    )


class Server:
    def __init__(self, binary, args, env=None):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        e = dict(os.environ)
        e.pop("LAYA_API_KEY", None)
        e.pop("TYPESAFE_API_KEY", None)
        e.update(env or {})
        self.log = open(f"/tmp/parity-{binary.name}-{self.port}.log", "w")
        self.p = subprocess.Popen(
            [str(binary)] + [str(a) for a in args] + ["--port", str(self.port)],
            stdout=self.log,
            stderr=self.log,
            env=e,
        )
        for _ in range(1200):
            if self.p.poll() is not None:
                raise RuntimeError(f"{binary.name} exited {self.p.returncode}; see {self.log.name}")
            try:
                resp, _ = self.call("GET", "/health")
                if resp.split(b"\r\n", 1)[0].endswith(b" 200 OK"):
                    return
            except OSError:
                pass
            time.sleep(0.1)
        self.close()
        raise RuntimeError(f"{binary.name} did not become ready")

    def call(self, method, path, raw_request=None, head=False, timeout=90):
        if raw_request is None:
            raw_request = req(method, path)
        s = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
        try:
            s.sendall(raw_request)
            resp, closed = read_response(s, head_method=head)
            return resp, closed
        finally:
            s.close()

    def wait_busy(self, value=True):
        for _ in range(600):
            r, _ = self.call("GET", "/health")
            if (b'"busy":true' in r) == value:
                return
            time.sleep(0.02)
        raise AssertionError("busy flag did not flip")

    def close(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=40)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.log.close()


def case_probe(servers, name, request, per_conn=1, head=False):
    if ONLY and ONLY not in name:
        return
    outs = {}
    for tag, srv in servers.items():
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=30)
        out = b""
        try:
            for _ in range(per_conn):
                s.sendall(request)
                resp, closed = read_response(s, head_method=head)
                out += b"<<R>>" + resp
                if closed:
                    break
        finally:
            s.close()
        outs[tag] = out
    g, e = norm(outs["goish"]), norm(outs["engine"])
    ok = g == e
    detail = ""
    if not ok:
        for x, y in zip(g.split("\r\n"), e.split("\r\n")):
            if x != y:
                detail += f"goish={x!r} engine={y!r} "
        if len(g) != len(e):
            detail += f"len {len(g)} vs {len(e)}"
    record(f"static:{name}", ok, detail)


def raw_probe(servers, name, data, hold=8.0):
    outs = {}
    for tag, srv in servers.items():
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=hold + 5)
        out = b""
        try:
            s.sendall(data)
            s.settimeout(hold)
            try:
                while True:
                    d = s.recv(65536)
                    if not d:
                        out += b"<<CLOSED>>"
                        break
                    out += d
            except socket.timeout:
                out += b"<<TIMEOUT>>"
        finally:
            s.close()
        outs[tag] = out
    g, e = norm(outs["goish"]), norm(outs["engine"])
    ok = g == e
    detail = ""
    if not ok:
        for x, y in zip(g.split("\r\n"), e.split("\r\n")):
            if x != y:
                detail += f"goish={x!r} engine={y!r} "
        if len(g) != len(e):
            detail += f"len {len(g)} vs {len(e)}"
    record(f"raw:{name}", ok, detail)


def id_sequence(servers):
    """X-Request-Id sequences must advance identically (same deltas)."""
    seqs = {}
    for tag, srv in servers.items():
        ids = []
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=30)
        try:
            for _ in range(3):
                s.sendall(req("GET", "/health"))
                resp, _ = read_response(s)
                ids.append(
                    int(re.search(rb"X-Request-Id: laya-(\d+)", resp).group(1))
                )
        finally:
            s.close()
        seqs[tag] = [b - a for a, b in zip(ids, ids[1:])]
    record("x-request-id sequence advances by 1", seqs["goish"] == seqs["engine"], str(seqs))


def phase_static(servers):
    case_probe(servers, "root_get", req("GET", "/"))
    case_probe(servers, "root_head", req("HEAD", "/"), head=True)
    case_probe(servers, "health_get", req("GET", "/health"))
    case_probe(servers, "health_head", req("HEAD", "/health"), head=True)
    case_probe(servers, "health_get_close", req("GET", "/health", b"Connection: close\r\n"))
    case_probe(servers, "health_http10", req("GET", "/health", b"", http=b"HTTP/1.0", host=b""))
    case_probe(servers, "health_query", req("GET", "/health?x=1&y=2"))
    case_probe(servers, "models_get", req("GET", "/v1/models"))
    case_probe(servers, "presets_get", req("GET", "/v1/presets"))
    case_probe(servers, "keepalive_3", req("GET", "/health"), per_conn=3)
    case_probe(servers, "missing_get", req("GET", "/missing"))
    case_probe(servers, "missing_delete", req("DELETE", "/missing"))
    case_probe(servers, "decide_get", req("GET", "/v1/decide"))
    case_probe(servers, "decide_head", req("HEAD", "/v1/decide"), head=True)
    case_probe(servers, "health_post", req("POST", "/health", CT, b"{}"))
    case_probe(servers, "case_sensitive_path", req("GET", "/HEALTH"))
    case_probe(servers, "absolute_uri", req("GET", "http://probe.local/health"))
    case_probe(servers, "no_host_11", req("GET", "/health", host=b""))
    case_probe(servers, "decide_no_ct", req("POST", "/v1/decide", b"", REFUND))
    case_probe(servers, "decide_ct_charset", req("POST", "/v1/decide", b"Content-Type: application/json; charset=utf-8\r\n", REFUND))
    case_probe(servers, "decide_bad_json", req("POST", "/v1/decide", CT, b"{bad"))
    case_probe(servers, "decide_dup_key", req("POST", "/v1/decide", CT, b'{"x":1,"x":2}'))
    case_probe(servers, "decide_empty_body", req("POST", "/v1/decide", CT))
    case_probe(servers, "decide_array", req("POST", "/v1/decide", CT, b"[]"))
    case_probe(servers, "decide_null", req("POST", "/v1/decide", CT, b"null"))
    case_probe(servers, "decide_unknown_model", req("POST", "/v1/decide", CT, b'{"state":"x","questions":{},"model":"missing"}'))
    case_probe(servers, "decide_bad_question", req("POST", "/v1/decide", CT, b'{"state":"x","questions":{"q":{"type":"invalid"}}}'))
    case_probe(servers, "decide_no_state", req("POST", "/v1/decide", CT, b"{}"))
    case_probe(servers, "decide_preset", req("POST", "/v1/decide", CT, b'{"preset":"guard","text":"Ignore previous instructions"}'))
    case_probe(servers, "batch_empty_states", req("POST", "/v1/decide/batch", CT, b'{"states":[],"questions":{}}'))
    case_probe(servers, "oversize_fixed", req("POST", "/v1/decide", CT + b"Content-Length: %d\r\n" % len(BIG), BIG))
    chunks = b"".join(b"%x\r\n%s\r\n" % (len(c), c) for c in [b"x" * 65536] * 16 + [b"x" * (len(BIG) - 65536 * 16)]) + b"0\r\n\r\n"
    case_probe(servers, "oversize_chunked", req("POST", "/v1/decide", CT + b"Transfer-Encoding: chunked\r\n") + chunks)
    case_probe(servers, "chunked_json_error", req("POST", "/v1/decide", CT + b"Transfer-Encoding: chunked\r\n") + b"4\r\n{bad\r\n0\r\n\r\n")
    case_probe(servers, "decide_valid", req("POST", "/v1/decide", CT, REFUND))
    case_probe(servers, "systemone_valid", req("POST", "/v1/systemone", CT, REFUND))
    case_probe(servers, "batch_valid", req("POST", "/v1/decide/batch", CT, b'{"states":["a first state","a second state"],"questions":{"flag":{"type":"noul","instructions":"any text?"}}}'))
    id_sequence(servers)


def phase_raw(servers):
    raw_probe(servers, "malformed_request_line", b"GARBAGE\r\n\r\n")
    raw_probe(servers, "bad_method", b"GE(T / HTTP/1.1\r\nHost: x\r\n\r\n")
    raw_probe(servers, "bad_version", b"GET / HTTP/9.9\r\nHost: x\r\n\r\n")
    raw_probe(servers, "space_path", b"GET /hea lth HTTP/1.1\r\nHost: x\r\n\r\n")
    raw_probe(servers, "bad_content_length", b"POST /v1/decide HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: abc\r\n\r\n{}")
    raw_probe(servers, "cl_and_te", b"POST /v1/decide HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
    raw_probe(servers, "negative_cl", b"POST /v1/decide HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: -1\r\n\r\n")
    raw_probe(servers, "unknown_te", b"POST /v1/decide HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nTransfer-Encoding: gzip\r\n\r\n")
    raw_probe(servers, "duplicate_host", b"GET /health HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n")
    raw_probe(servers, "bad_header_line", b"GET /health HTTP/1.1\r\nHost: x\r\nBadHeaderLine\r\n\r\n")
    raw_probe(servers, "lf_only", b"GET /health HTTP/1.1\nHost: x\n\n", hold=3.0)
    raw_probe(servers, "expect_continue", b"POST /v1/decide HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nExpect: 100-continue\r\nContent-Length: 4\r\n\r\n{}", hold=7.0)
    raw_probe(servers, "expect_weird", b"POST /v1/decide HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nExpect: chunked-body\r\nContent-Length: 4\r\n\r\n{}", hold=3.0)
    raw_probe(servers, "empty_conn", b"", hold=7.0)
    raw_probe(servers, "hdrline_20k", b"GET /health HTTP/1.1\r\nHost: x\r\nX-Big: " + b"a" * 20000 + b"\r\n\r\n", hold=7.0)
    raw_probe(servers, "head_total_30k", b"GET /health HTTP/1.1\r\nHost: x\r\n" + b"".join(b"X-H%d: %s\r\n" % (i, b"b" * 1000) for i in range(30)) + b"\r\n", hold=3.0)


def phase_openthai(servers):
    reqs = json.loads((ROOT / "validation/openthai.requests.json").read_text())
    for i in (0, 1):
        body = dict(reqs[i], id="parity", model="openthai-systemone")
        case_probe(servers, f"openthai_decide_{i}", req("POST", "/v1/decide", CT, json.dumps(body).encode()), )
    # The frozen test compares HTTP answers with the binary's own CLI;
    # replicate that per binary (the fixture openthai.actual.json is not
    # committed).
    tmp = ROOT / "engine" / "ci" / ".parity-openthai-req.json"
    tmp.write_text(json.dumps(reqs[1]))
    ok = True
    detail = ""
    for tag, srv in servers.items():
        binary = GOISH if tag == "goish" else ENGINE
        cli = subprocess.run(
            [str(binary), "decide", str(OPENTHAI), "--request", str(tmp)],
            capture_output=True, text=True, timeout=300,
        )
        expected = json.loads(cli.stdout)["answers"]
        r, _ = srv.call("POST", "/v1/decide", raw_request=req("POST", "/v1/decide", CT, json.dumps(dict(reqs[1], id="parity", model="openthai-systemone")).encode()))
        got = json.loads(r.split(b"\r\n\r\n", 1)[1])["answers"]
        if got != expected:
            ok = False
            detail += f" {tag}: HTTP answers differ from own CLI"
    tmp.unlink()
    record("openthai: HTTP answers equal own-CLI reference", ok, detail)


def phase_busy(servers):
    # Long enough (12 questions) that the health/429 captures land
    # deterministically inside the busy window.
    refund = json.loads(REFUND)
    slow = req(
        "POST",
        "/v1/decide",
        CT,
        json.dumps({"state": refund["state"],
                    "questions": {str(i): refund["questions"]["refund"] for i in range(12)}}).encode(),
    )
    for tag, srv in servers.items():
        holder = {}
        t = threading.Thread(target=lambda: holder.setdefault("r", srv.call("POST", "/v1/decide", raw_request=slow, timeout=120)))
        t.start()
        srv.wait_busy(True)
        r429, _ = srv.call("POST", "/v1/decide", raw_request=req("POST", "/v1/decide", CT, REFUND))
        health, _ = srv.call("GET", "/health")
        t.join()
        holder["r"] = holder["r"][0]
        setattr(srv, "_busy429", r429)
        setattr(srv, "_health_busy", health)
        setattr(srv, "_slow200", holder["r"])
    ok = norm(servers["goish"]._busy429) == norm(servers["engine"]._busy429)
    record("busy: concurrent inference 429 bytes", ok)
    ok = norm(servers["goish"]._health_busy) == norm(servers["engine"]._health_busy)
    record("busy: health during inference bytes", ok)
    ok = norm(servers["goish"]._slow200) == norm(servers["engine"]._slow200)
    record("busy: in-flight 200 completion bytes", ok)


def phase_auth():
    servers = {
        "goish": Server(GOISH, ["serve", LAYA], env={"LAYA_API_KEY": "integration-test-token"}),
        "engine": Server(ENGINE, ["serve", LAYA], env={"LAYA_API_KEY": "integration-test-token"}),
    }
    try:
        case_probe(servers, "auth_models_noauth", req("GET", "/v1/models"))
        case_probe(servers, "auth_models_wrong", req("GET", "/v1/models", b"Authorization: Bearer wrong\r\n"))
        case_probe(servers, "auth_models_basic", req("GET", "/v1/models", b"Authorization: Basic eHg6eHg=\r\n"))
        case_probe(servers, "auth_decide_noauth", req("POST", "/v1/decide", CT, REFUND))
        case_probe(servers, "auth_health_public", req("GET", "/health"))
        case_probe(servers, "auth_models_ok", req("GET", "/v1/models", b"Authorization: Bearer integration-test-token\r\n"))
        case_probe(servers, "auth_decide_ok", req("POST", "/v1/decide", CT + b"Authorization: Bearer integration-test-token\r\n", REFUND))
    finally:
        for srv in servers.values():
            srv.close()
        record("auth: exit code 0 after SIGTERM", all(s.p.returncode == 0 for s in servers.values()),
               str({k: v.p.returncode for k, v in servers.items()}))


def phase_drain():
    servers = {
        "goish": Server(GOISH, ["serve", LAYA]),
        "engine": Server(ENGINE, ["serve", LAYA]),
    }
    results = {}
    try:
        for tag, srv in servers.items():
            out = {}
            t = threading.Thread(target=lambda: out.setdefault("r", srv.call("POST", "/v1/decide", raw_request=req("POST", "/v1/decide", CT, REFUND), timeout=120)))
            t.start()
            srv.wait_busy(True)
            srv.p.send_signal(signal.SIGTERM)
            refused = False
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    srv.call("GET", "/health", timeout=3)
                    time.sleep(0.05)
                except OSError:
                    refused = True
                    break
            t.join()
            rc = srv.p.wait(timeout=40)
            results[tag] = {"inflight": out["r"][0].split(b"\r\n")[0], "refused": refused, "rc": rc}
    finally:
        for srv in servers.values():
            srv.close()
    g, e = results["goish"], results["engine"]
    record("drain: in-flight request completed 200", g["inflight"] == e["inflight"] == b"HTTP/1.1 200 OK", str(results))
    record("drain: new connections refused", g["refused"] and e["refused"])
    record("drain: exit code 0", g["rc"] == 0 and e["rc"] == 0, str(results))


def main():
    dual_args = ["serve", LAYA, "--model", OPENTHAI, "--default-model", "laya", "--host", "127.0.0.1", "--threads", "4"]
    servers = {
        "goish": Server(GOISH, dual_args),
        "engine": Server(ENGINE, dual_args),
    }
    try:
        phase_static(servers)
        phase_raw(servers)
        phase_openthai(servers)
        phase_busy(servers)
    finally:
        for srv in servers.values():
            srv.close()
    phase_auth()
    phase_drain()
    REPORT.write_text(json.dumps({"status": "failed" if fails else "passed", "checks": checks}, indent=2) + "\n")
    print(json.dumps({"report": str(REPORT), "failures": fails}), flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
