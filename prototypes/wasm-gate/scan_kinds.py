#!/usr/bin/env python3
# Scan GGUF tensor kinds: bytes by kind + every non-F16 tensor. Kinds: 0=F32 1=F16 8=Q8_0.
import struct, sys, collections

LAYOUT = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34), 12: (256, 144), 14: (256, 210)}

class R:
    def __init__(self, data):
        self.d = data
        self.p = 0
    def take(self, n):
        s = self.d[self.p:self.p+n]
        self.p += n
        return s
    def u32(self):
        return struct.unpack('<I', self.take(4))[0]
    def u64(self):
        return struct.unpack('<Q', self.take(8))[0]
    def string(self):
        n = self.u64()
        return self.take(n).decode()
    def skip_value(self, t, depth=0):
        if t in (0, 1, 7): self.take(1)
        elif t in (2, 3): self.take(2)
        elif t in (4, 5, 6): self.take(4)
        elif t == 8: self.take(self.u64())
        elif t == 9:
            et = self.u32(); n = self.u64()
            for _ in range(n): self.skip_value(et, depth + 1)
        elif t in (10, 11, 12): self.take(8)
        else: raise SystemExit(f'bad meta type {t}')

def scan(path):
    r = R(open(path, 'rb').read())
    assert r.take(4) == b'GGUF'
    ver = r.u32()
    nt, nk = r.u64(), r.u64()
    for _ in range(nk):
        k = r.string()
        t = r.u32()
        r.skip_value(t)
    kinds = collections.Counter()
    nonf16 = []
    for _ in range(nt):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        kind = r.u32()
        off = r.u64()
        blk, sz = LAYOUT[kind]
        cnt = 1
        for d in dims: cnt *= d
        nb = cnt // blk * sz
        kinds[kind] += nb
        if kind != 1:
            nonf16.append((name, kind, dims, nb))
    print(path.split('/')[-1], '| version', ver, '| tensors', nt, '| bytes by kind', dict(kinds))
    for name, kind, dims, nb in nonf16[:10]:
        print('   ', name, 'kind', kind, dims, nb)
    if len(nonf16) > 10:
        print(f'    ... {len(nonf16)-10} more non-F16 tensors')

for path in sys.argv[1:]:
    scan(path)
