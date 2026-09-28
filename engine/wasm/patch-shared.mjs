// Post-link patch for the threads artifact (ticket 24): marks the module's
// imported linear memory SHARED. wasm-ld only emits a shared memory import
// under --shared, which additionally demands PIC-built std (nightly
// -Zbuild-std); on stable we instead build with --import-memory
// --max-memory=4294967296 and flip the import's limits flags here.
//
// The patch is length-preserving: flags 0x01 (has-max) -> 0x03
// (shared+has-max); the declared maximum is already 65536 pages (4 GiB),
// exactly what the harness creates. Active data segments stay active —
// the harness instantiates every worker before the first wasm call, so
// their re-application only rewrites pristine zeros.
//
// usage: node patch-shared.mjs in.wasm out.wasm
import { readFileSync, writeFileSync } from 'node:fs';

const [inp, outp] = process.argv.slice(2);
const b = readFileSync(inp);

function leb(b, i) {
  let r = 0, s = 0;
  for (;;) {
    const x = b[i++];
    r |= (x & 0x7f) << s;
    if ((x & 0x80) === 0) return [r >>> 0, i];
    s += 7;
  }
}

if (b.readUInt32LE(0) !== 0x6d736100) throw new Error('not a wasm');
let i = 8;
let patched = false;
while (i < b.length) {
  const id = b[i];
  const [sz, cStart] = leb(b, i + 1);
  const cEnd = cStart + sz;
  if (id === 2) {
    const [nImports, p0] = leb(b, cStart);
    let p = p0;
    for (let k = 0; k < nImports; k++) {
      const [mLen, q0] = leb(b, p);
      const mod = b.toString('utf8', q0, q0 + mLen);
      p = q0 + mLen;
      const [fLen, q1] = leb(b, p);
      const field = b.toString('utf8', q1, q1 + fLen);
      p = q1 + fLen;
      const kind = b[p++];
      if (kind === 0x00) {
        const [, q] = leb(b, p); p = q;
      } else if (kind === 0x01) {
        p++;
        const tf = b[p++];
        const [, q] = leb(b, p); p = q;
        if (tf & 0x01) { const [, q2] = leb(b, p); p = q2; }
      } else if (kind === 0x02) {
        if (mod !== 'env' || field !== 'memory') throw new Error(`unexpected memory import ${mod}.${field}`);
        const flags = b[p];
        if (flags & 0x02) throw new Error('memory import already shared');
        if (!(flags & 0x01)) throw new Error('no declared maximum (build with --max-memory)');
        const [, q2] = leb(b, p + 1);
        const [maxPages, q3] = leb(b, q2);
        if (maxPages !== 65536) throw new Error(`unexpected declared max ${maxPages} pages (want 65536)`);
        b[p] = flags | 0x02;
        p = q3;
        patched = true;
      } else if (kind === 0x03) {
        p += 2;
      } else {
        throw new Error(`unknown import kind ${kind}`);
      }
    }
  }
  i = cEnd;
}
if (!patched) throw new Error('env.memory import not found');
writeFileSync(outp, b);
console.error(`patched ${outp}: env.memory shared, 65536-page max (${b.length} bytes)`);
