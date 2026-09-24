#!/usr/bin/env python3
"""Requantize F16 matrices in a GGUF v3 file to Q8_0 (34 bytes / 32 values).

Usage: quantize-q8.py IN.gguf OUT.gguf [--min-elements N]

2-D F16 tensors with at least --min-elements (default 262144) elements and a
32-aligned leading dimension are converted; everything else (vectors, small
matrices, the per-request-decoded OpenThai embedding table) is copied
verbatim. Tensor data offsets are relative to the aligned end of the tensor
info table; that table keeps its byte length, so the data base is unchanged
and only post-quantization offsets shift.
"""
import struct
import sys

import numpy as np

Q8_0, F16, GGUF_V3 = 8, 1, 3
MIN_ELEMS = 262144
LAYOUT = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34), 12: (256, 144), 14: (256, 210)}
# Kept out of quantization: the OpenThai backend decodes embeddings per
# request and accepts F32/F16 only; the Laya embedding table adds direct
# input noise that pushed probability drift past the acceptance gate.
SKIP = {b"model.embed_tokens.weight", b"tok_emb.weight"}


def rx(f, n):
    b = f.read(n)
    if len(b) != n:
        raise SystemExit("truncated GGUF")
    return b


def ru32(f):
    return struct.unpack("<I", rx(f, 4))[0]


def ru64(f):
    return struct.unpack("<Q", rx(f, 8))[0]


def rstr(f):
    return rx(f, ru64(f))


def skip_value(f, t, depth=0):
    if depth > 4:
        raise SystemExit("GGUF metadata nesting too deep")
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t == 8:
        rstr(f)
    elif t == 9:
        it = ru32(f)
        n = ru64(f)
        if n > 1_000_000:
            raise SystemExit("GGUF array too large")
        for _ in range(n):
            skip_value(f, it, depth + 1)
    elif t in fixed:
        rx(f, fixed[t])
    else:
        raise SystemExit(f"unsupported GGUF metadata type {t}")


def quantize_q8_0(f16_bytes):
    x = np.frombuffer(f16_bytes, dtype="<f2").astype(np.float32)
    if x.size % 32:
        raise SystemExit("Q8_0 block misalignment")
    blocks = x.reshape(-1, 32)
    amax = np.abs(blocks).max(axis=1)
    d = (amax / 127.0).astype("<f2")
    dv = d.astype(np.float32)
    q = np.clip(np.rint(blocks / np.where(dv == 0, 1.0, dv)[:, None]), -127, 127)
    out = np.zeros((len(blocks), 34), dtype=np.uint8)
    out[:, :2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.astype(np.int8).view(np.uint8).reshape(-1, 32)
    return out.tobytes()


def align_up(x, a):
    return (x + a - 1) & ~(a - 1)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    min_elems = MIN_ELEMS
    if "--min-elements" in sys.argv:
        min_elems = int(sys.argv[sys.argv.index("--min-elements") + 1])

    with open(src, "rb") as f, open(dst, "wb") as out:
        if rx(f, 4) != b"GGUF" or ru32(f) != GGUF_V3:
            raise SystemExit("expected GGUF version 3")
        ntensors, nkv = ru64(f), ru64(f)
        kv_start = f.tell()
        alignment = 32
        for _ in range(nkv):
            name = rstr(f)
            t = ru32(f)
            before = f.tell()
            skip_value(f, t)
            if name == b"general.alignment" and t == 4:
                cur = f.tell()
                f.seek(before)
                alignment = ru32(f)
                f.seek(cur)
        kv_end = f.tell()
        f.seek(kv_start)
        kv_blob = rx(f, kv_end - kv_start)

        tensors = []
        for _ in range(ntensors):
            name = rstr(f)
            nd = ru32(f)
            dims = [ru64(f) for _ in range(nd)]
            kind = ru32(f)
            offset = ru64(f)
            tensors.append(
                {"name": name, "dims": dims, "kind": kind, "src_offset": offset}
            )
        info_end = f.tell()
        data_base = align_up(info_end, alignment)

        for t in tensors:
            count = 1
            for d in t["dims"]:
                count *= d
            src_block, src_size = LAYOUT[t["kind"]]
            t["src_len"] = (count // src_block) * src_size
            t["quantize"] = (
                t["kind"] == F16
                and len(t["dims"]) >= 2
                and all(d == 1 for d in t["dims"][2:])
                and count >= min_elems
                and t["dims"][0] % 32 == 0
                and t["name"] not in SKIP
            )
            if t["quantize"]:
                t["kind"] = Q8_0
            block, size = LAYOUT[t["kind"]]
            t["length"] = (count // block) * size

        cursor = data_base
        for t in tensors:
            cursor = align_up(cursor, alignment)
            t["offset"] = cursor - data_base
            cursor += t["length"]

        out.write(b"GGUF")
        out.write(struct.pack("<IQQ", GGUF_V3, ntensors, nkv))
        out.write(kv_blob)
        for t in tensors:
            out.write(struct.pack("<Q", len(t["name"])))
            out.write(t["name"])
            out.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]:
                out.write(struct.pack("<Q", d))
            out.write(struct.pack("<IQ", t["kind"], t["offset"]))
        out.write(b"\0" * (data_base - info_end))

        nq = 0
        written = 0
        for t in tensors:
            if t["offset"] > written:
                out.write(b"\0" * (t["offset"] - written))
                written = t["offset"]
            f.seek(data_base + t["src_offset"])
            raw = rx(f, t["src_len"])
            if t["quantize"]:
                data = quantize_q8_0(raw)
                nq += 1
            else:
                data = raw
            out.write(data)
            written += len(data)
        print(f"quantized {nq}/{ntensors} tensors -> {dst}")


if __name__ == "__main__":
    main()
