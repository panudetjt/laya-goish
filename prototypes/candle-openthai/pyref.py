#!/usr/bin/env python3
"""Independent numpy reference of the OpenThai forward, written directly from
src/models/openthai/model.rs. Diagnostic only: locates which layer the candle
port diverges on. Run from prototypes/candle-openthai/.
Usage: python3 pyref.py CASE.json [--layers N]"""
import json
import math
import struct
import sys

import numpy as np

D = 1024
INTER = 3584
LAYERS = 24
FULL_HEADS = 8
FULL_HD = 256
KV_HEADS = 2
LH = 16
LD = 128
QKVW = 6144
VOCAB = 248339


def half(x):
    sign = -1.0 if x & 0x8000 else 1.0
    e = (x >> 10) & 31
    m = x & 1023
    if e == 0:
        return sign * m * (1.0 / 16777216.0)
    if e == 31:
        return sign * math.inf if m == 0 else math.nan
    return sign * math.ldexp(1024 + m, e - 25)


H = np.vectorize(np.float16)(0).dtype


def f16_arr(raw):
    return np.frombuffer(raw, dtype="<f2")


class Gguf:
    def __init__(self, path):
        self.b = open(path, "rb").read()
        p = 0
        assert self.b[0:4] == b"GGUF"
        p = 4
        (ver,) = struct.unpack_from("<I", self.b, p); p += 4
        assert ver == 3
        (nt,) = struct.unpack_from("<Q", self.b, p); p += 8
        (nk,) = struct.unpack_from("<Q", self.b, p); p += 8
        self.meta = {}
        for _ in range(nk):
            k, v, p = self.kv(p)
            self.meta[k] = v
        self.tensors = {}
        for _ in range(nt):
            (n,) = struct.unpack_from("<Q", self.b, p); p += 8
            name = self.b[p:p+n].decode(); p += n
            (nd,) = struct.unpack_from("<I", self.b, p); p += 4
            dims = []
            for _ in range(nd):
                (d,) = struct.unpack_from("<Q", self.b, p); p += 8
                dims.append(d)
            (kind,) = struct.unpack_from("<I", self.b, p); p += 4
            (off,) = struct.unpack_from("<Q", self.b, p); p += 8
            self.tensors[name] = (dims, kind, off)
        align = self.meta.get("general.alignment", 32)
        base = (p + align - 1) & ~(align - 1)
        self.base = base

    def kv(self, p):
        (n,) = struct.unpack_from("<Q", self.b, p); p += 8
        key = self.b[p:p+n].decode(); p += n
        (t,) = struct.unpack_from("<I", self.b, p); p += 4
        v, p = self.val(t, p)
        return key, v, p

    def val(self, t, p):
        if t == 8:
            (n,) = struct.unpack_from("<Q", self.b, p); p += 8
            s = self.b[p:p+n].decode(); p += n
            return s, p
        if t == 9:
            (vt,) = struct.unpack_from("<I", self.b, p); p += 4
            (n,) = struct.unpack_from("<Q", self.b, p); p += 8
            a = []
            for _ in range(n):
                v, p = self.val(vt, p)
                a.append(v)
            return a, p
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
               7: "<B", 10: "<Q", 11: "<q", 12: "<d"}[t]
        (v,) = struct.unpack_from(fmt, self.b, p); p += struct.calcsize(fmt)
        return v, p

    def raw(self, name):
        dims, kind, off = self.tensors[name]
        sizes = {0: 4, 1: 2, 8: 34 / 32}
        n = 1
        for d in dims:
            n *= d
        if kind == 0:
            sz = n * 4
        elif kind == 1:
            sz = n * 2
        else:
            sz = (n // 32) * 34
        return self.b[self.base + off:self.base + off + sz], dims, kind, n

    def mat(self, name, rows, cols):
        raw, dims, kind, n = self.raw(name)
        assert dims == [cols, rows], (name, dims)
        if kind == 0:
            return np.frombuffer(raw, dtype="<f4").reshape(rows, cols).astype(np.float32)
        if kind == 1:
            return f16_arr(raw).reshape(rows, cols).astype(np.float32)
        raise ValueError(kind)

    def vec(self, name):
        raw, dims, kind, n = self.raw(name)
        if kind == 0:
            return np.frombuffer(raw, dtype="<f4").astype(np.float32)
        if kind == 1:
            return f16_arr(raw).astype(np.float32)
        raise ValueError(kind)


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x):
    m = np.max(x)
    e = np.exp(x - m)
    return e / e.sum()


def rms(x, w, offset):
    # x: [s, k*len(w)] -> engine rms per len(w)-wide chunk
    n = len(w)
    xr = x.reshape(-1, n)
    scale = 1.0 / np.sqrt((xr.astype(np.float32) ** 2).sum(axis=-1, keepdims=True) / n + 1e-6)
    return (xr * scale * (offset + w)).reshape(x.shape)


class Model:
    def __init__(self, path):
        g = Gguf(path)
        self.g = g
        cfg = json.loads(g.meta["openthai.config"])
        c = cfg["text_config"]
        self.types = c["layer_types"]
        self.blocks = []
        for i in range(LAYERS):
            p = f"model.layers.{i}"
            if self.types[i] == "full_attention":
                mix = {
                    "kind": "full",
                    "q": g.mat(f"{p}.self_attn.q_proj.weight", 4096, D),
                    "k": g.mat(f"{p}.self_attn.k_proj.weight", 512, D),
                    "v": g.mat(f"{p}.self_attn.v_proj.weight", 512, D),
                    "o": g.mat(f"{p}.self_attn.o_proj.weight", D, 2048),
                    "qn": g.vec(f"{p}.self_attn.q_norm.weight"),
                    "kn": g.vec(f"{p}.self_attn.k_norm.weight"),
                }
            else:
                mix = {
                    "kind": "delta",
                    "qkv": g.mat(f"{p}.linear_attn.in_proj_qkv.weight", QKVW, D),
                    "z": g.mat(f"{p}.linear_attn.in_proj_z.weight", 2048, D),
                    "a": g.mat(f"{p}.linear_attn.in_proj_a.weight", LH, D),
                    "b": g.mat(f"{p}.linear_attn.in_proj_b.weight", LH, D),
                    "o": g.mat(f"{p}.linear_attn.out_proj.weight", D, 2048),
                    "conv": g.vec(f"{p}.linear_attn.conv1d.weight"),
                    "dt": g.vec(f"{p}.linear_attn.dt_bias"),
                    "alog": g.vec(f"{p}.linear_attn.A_log"),
                    "norm": g.vec(f"{p}.linear_attn.norm.weight"),
                }
            self.blocks.append({
                "n1": g.vec(f"{p}.input_layernorm.weight"),
                "n2": g.vec(f"{p}.post_attention_layernorm.weight"),
                "gate": g.mat(f"{p}.mlp.gate_proj.weight", INTER, D),
                "up": g.mat(f"{p}.mlp.up_proj.weight", INTER, D),
                "down": g.mat(f"{p}.mlp.down_proj.weight", D, INTER),
                "mix": mix,
            })
        self.emb = g.mat("model.embed_tokens.weight", VOCAB, D)
        self.norm = g.vec("model.norm.weight")
        self.head = g.mat("slot_head.weight", 256, D)
        self.bias = g.vec("slot_head.bias")
        self.temperature = np.exp(g.vec("log_temperature"))

    def full_attn(self, f, x):
        s = x.shape[0]
        qgate = x @ f["q"].T
        q = np.zeros((s, 2048), dtype=np.float32)
        gate = np.zeros((s, 2048), dtype=np.float32)
        for h in range(FULL_HEADS):
            q[:, h*256:(h+1)*256] = qgate[:, h*512:h*512+256]
            gate[:, h*256:(h+1)*256] = qgate[:, h*512+256:(h+1)*512]
        q = rms(q, f["qn"], 1.0)
        k = rms(x @ f["k"].T, f["kn"], 1.0)
        v = x @ f["v"].T
        # rope
        for buf, heads in ((q, FULL_HEADS), (k, KV_HEADS)):
            for i in range(s):
                for h in range(heads):
                    for j in range(32):
                        angle = i / math.pow(10000000.0, (2 * j) / 64.0)
                        co, si = math.cos(angle), math.sin(angle)
                        p = i * heads * 256 + h * 256 + j
                        a, b = buf.flat[p], buf.flat[p + 32]
                        buf.flat[p] = a * co - b * si
                        buf.flat[p + 32] = b * co + a * si
        out = np.zeros((s, 2048), dtype=np.float32)
        for i in range(s):
            for h in range(FULL_HEADS):
                kh = h // 4
                scores = (k[: i + 1, kh*256:(kh+1)*256] @ q[i, h*256:(h+1)*256]) / 16.0
                p = softmax(scores[: i + 1])
                for j in range(i + 1):
                    out[i, h*256:(h+1)*256] += p[j] * v[j, kh*256:(kh+1)*256]
        out = out * (1.0 / (1.0 + np.exp(-gate)))
        return out @ f["o"].T

    def delta(self, d, x):
        s = x.shape[0]
        raw = x @ d["qkv"].T
        z = x @ d["z"].T
        a = x @ d["a"].T
        b = x @ d["b"].T
        qkv = np.zeros_like(raw)
        convw = d["conv"].reshape(QKVW, 4)
        for i in range(s):
            acc = np.zeros(QKVW, dtype=np.float32)
            for j in range(4):
                if i + j >= 3:
                    acc += raw[i + j - 3] * convw[:, j]
            qkv[i] = acc / (1.0 + np.exp(-acc))
        out = np.zeros((s, 2048), dtype=np.float32)
        for h in range(LH):
            state = np.zeros((128, 128), dtype=np.float32)
            for i in range(s):
                q = qkv[i, h*128:(h+1)*128]
                k = qkv[i, 2048+h*128:2048+(h+1)*128]
                v = qkv[i, 4096+h*128:4096+(h+1)*128]
                qs = 1.0 / math.sqrt(float(q @ q) + 1e-6) / math.sqrt(128.0)
                ks = 1.0 / math.sqrt(float(k @ k) + 1e-6)
                decay = math.exp(-math.exp(d["alog"][h]) * math.log1p(math.exp(a[i, h] + d["dt"][h])))
                beta = 1.0 / (1.0 + math.exp(-b[i, h]))
                state *= decay
                delta = state.T @ (k * ks)
                delta = (v - delta) * beta
                o = out[i, h*128:(h+1)*128]
                for j in range(128):
                    key = k[j] * ks
                    query = q[j] * qs
                    state[j] += key * delta
                    o += state[j] * query
        out = rms(out, d["norm"], 0.0)
        out = out * silu(z)
        return out @ d["o"].T

    def forward(self, ids, positions, dump=None):
        x = self.emb[np.asarray(ids)]
        for i, blk in enumerate(self.blocks):
            h = rms(x, blk["n1"], 1.0)
            if blk["mix"]["kind"] == "full":
                y = self.full_attn(blk["mix"], h)
            else:
                y = self.delta(blk["mix"], h)
            x = x + y
            h = rms(x, blk["n2"], 1.0)
            m = silu(h @ blk["gate"].T) * (h @ blk["up"].T)
            x = x + m @ blk["down"].T
            if dump is not None:
                dump(i, x)
        ans = x[np.asarray(positions)]
        logits = rms(ans, self.norm, 1.0) @ self.head.T + self.bias
        return logits


def main():
    case = json.load(open(sys.argv[1]))
    ref = case["_raw"][0]
    ids, positions = ref["input_ids"], ref["answer_positions"]
    m = Model("../../models/openthai_systemone_v03_f16.gguf")
    dumpdir = sys.argv[2] if len(sys.argv) > 2 else None
    if dumpdir:
        import os
        os.makedirs(dumpdir, exist_ok=True)
        def dump(i, x):
            np.save(f"{dumpdir}/py_layer{i}.npy", x)
    else:
        dump = None
    logits = m.forward(ids, positions, dump=dump)
    rl = np.array(ref["logits"])
    print("max|dlogit| vs engine:", np.abs(logits.flatten() - rl).max())
    print("sample got:", logits[0, :4])
    print("sample ref:", rl[:4])


if __name__ == "__main__":
    main()
