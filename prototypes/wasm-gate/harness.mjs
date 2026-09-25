// Node harness for the wasm32 gate (wayfinder ticket 08).
//
// Loads the wasm32-unknown-unknown build of this crate, copies a GGUF file
// into linear memory, runs the reference case forwards inside wasm, and
// measures: copy/load/parse/build times, linear-memory high-water (V8
// memory.buffer.byteLength is monotonic), allocator live/peak bytes, forward
// latency (inside-wasm timer around the real work), and logits drift vs the
// native candle reference dumps in native/.
//
// Usage:
//   node harness.mjs --model laya --gguf ../../models/laya_english_f16.gguf \
//     --mode f16 --case ../candle-laya/reference/refund.json \
//     --native native/laya_refund_f16.json --repeats 5 --out results/laya_f16.json
//   node harness.mjs --model openthai --gguf ../../models/openthai_systemone_v03_f16.gguf \
//     --mode f16 --case ../candle-openthai/reference/thai.json \
//     --native native/openthai_thai_f16.json --repeats 5 --out results/ot_f16.json
//   node harness.mjs --model dual --gguf ../../models/laya_english_f16.gguf \
//     --gguf2 ../../models/openthai_systemone_v03_q8_0.gguf --mode f16 --mode2 q80 \
//     --case ../candle-laya/reference/refund.json \
//     --case2 ../candle-openthai/reference/thai.json \
//     --native native/laya_refund_f16.json --native2 native/openthai_thai_f16.json \
//     --repeats 3 --out results/dual.json
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

const args = process.argv.slice(2);
const opt = (k, d) => {
  const i = args.indexOf(k);
  return i >= 0 ? args[i + 1] : d;
};
const num = (k, d) => Number(opt(k, d));

const MODEL = opt('--model', 'laya');
const MODE = opt('--mode', 'f16');
const MODE2 = opt('--mode2', 'f16');
const REPEATS = num('--repeats', 5);
const WARMUP = num('--warmup', 1);
const WASM = opt('--wasm', 'target/wasm32-unknown-unknown/release/wasm_gate.wasm');

const wbytes = readFileSync(WASM);
const t0 = performance.now();
const { instance } = await WebAssembly.instantiate(wbytes, {
  env: { now_ms: () => performance.now() },
});
const instantiateMs = performance.now() - t0;
const e = instance.exports;
const linear = () => e.memory.buffer.byteLength;

const result = {
  node: process.version,
  wasm_bytes: wbytes.length,
  instantiate_ms: +instantiateMs.toFixed(1),
  model: MODEL,
  mode: MODE,
  mode2: MODEL === 'dual' ? MODE2 : undefined,
  memory: {},
  runs: [],
};

function putU32(arr) {
  const ptr = e.wasm_alloc_u32(arr.length) >>> 0;
  new Uint32Array(e.memory.buffer, ptr, arr.length).set(arr);
  return ptr;
}

function lastError() {
  try {
    const p = e.wasm_last_error() >>> 0;
    const end = new Uint8Array(e.memory.buffer, p, 4096).indexOf(0);
    return end > 0 ? new TextDecoder().decode(new Uint8Array(e.memory.buffer, p, end)) : null;
  } catch {
    return null;
  }
}

function diag() {
  try {
    const p = e.wasm_diag() >>> 0;
    const end = new Uint8Array(e.memory.buffer, p, 4096).indexOf(0);
    return end > 0 ? new TextDecoder().decode(new Uint8Array(e.memory.buffer, p, end)) : null;
  } catch {
    return null;
  }
}

function loadModel(id, ggufPath, mode) {
  const buf = readFileSync(ggufPath);
  const t = performance.now();
  const ptr = e.wasm_alloc_u8(buf.length) >>> 0;
  new Uint8Array(e.memory.buffer, ptr, buf.length).set(buf);
  const copyMs = performance.now() - t;
  const tl = performance.now();
  let rc;
  try {
    rc = e.wasm_load(id, mode === 'f16' ? 0 : mode === 'convert' ? 1 : 2, ptr, buf.length);
  } catch (err) {
    throw new Error(`wasm_load trapped: ${lastError() ?? err}`);
  }
  const loadWallMs = performance.now() - tl;
  if (rc !== 0) {
    throw new Error(`wasm_load failed: ${lastError()}`);
  }
  return {
    gguf: ggufPath,
    gguf_bytes: buf.length,
    copy_ms: +copyMs.toFixed(1),
    parse_ms: +(Number(e.wasm_stat_us(1)) / 1000).toFixed(1),
    build_ms: +(Number(e.wasm_stat_us(2)) / 1000).toFixed(1),
    load_inside_ms: +(Number(e.wasm_stat_us(0)) / 1000).toFixed(1),
    load_wall_ms: +loadWallMs.toFixed(1),
    linear_after_load: linear(),
    live_alloc_after_load: e.wasm_stat_live_alloc() >>> 0,
    peak_alloc_after_load: e.wasm_stat_peak_alloc() >>> 0,
  };
}

function softmax(xs) {
  const m = Math.max(...xs);
  const es = xs.map((v) => Math.exp(v - m));
  const s = es.reduce((a, b) => a + b, 0);
  return es.map((v) => v / s);
}

function readOuts(packed) {
  const nOut = Number(packed & 0xffffffffn);
  const nOut2 = Number(packed >> 32n);
  const logits = Array.from(new Float32Array(e.memory.buffer, e.wasm_out_ptr() >>> 0, nOut));
  const act = nOut2 ? Array.from(new Float32Array(e.memory.buffer, e.wasm_out2_ptr() >>> 0, nOut2)) : [];
  return { logits, act };
}

function driftStats(wasm, native) {
  let maxDl = 0;
  let sumSq = 0;
  const n = Math.min(wasm.length, native.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(wasm[i] - native[i]);
    maxDl = Math.max(maxDl, d);
    sumSq += d * d;
  }
  return { n, max_dlogit: maxDl, rms_dlogit: Math.sqrt(sumSq / n) };
}

function forwardLoop(id, ids, aux, qtype, repeats) {
  const idsPtr = putU32(ids);
  const auxPtr = putU32(aux);
  const insideMs = [];
  const wallMs = [];
  let outs = null;
  let trapped = null;
  for (let i = 0; i < WARMUP + repeats; i++) {
    try {
      const t = performance.now();
      const packed = e.wasm_forward(id, idsPtr, ids.length, auxPtr, aux.length, qtype);
      const wall = performance.now() - t;
      if (packed === 0xffffffffffffffffn) {
        throw new Error(`wasm_forward failed: ${lastError()}`);
      }
      outs = readOuts(packed);
      if (i >= WARMUP) {
        insideMs.push(+(Number(e.wasm_stat_us(3)) / 1000).toFixed(1));
        wallMs.push(+wall.toFixed(1));
      }
    } catch (err) {
      trapped = `${lastError() ?? ''} | ${err}`;
      break;
    }
  }
  e.wasm_free_u32(idsPtr, ids.length);
  e.wasm_free_u32(auxPtr, aux.length);
  return { inside_ms: insideMs, wall_ms: wallMs, outs, trapped };
}

const mean = (v) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : null);

function layaRun(gguf, casePath, nativePath, mode) {
  const load = loadModel(0, gguf, mode);
  const cs = JSON.parse(readFileSync(casePath));
  const ans = Object.values(cs.answers)[0];
  const kind = { choice: 0, score: 1, noul: 2 }[ans.type];
  const ids = ans.input_ids;
  const markers = ans.markers;
  const { inside_ms, wall_ms, outs, trapped } = forwardLoop(0, ids, markers, kind, REPEATS);
  let drift = null;
  let dp = null;
  if (outs && !trapped) {
    const nat = JSON.parse(readFileSync(nativePath));
    const natAns = Object.values(nat.answers)[0];
    drift = driftStats(outs.logits, natAns.logits);
    drift.drift_act = driftStats(outs.act, natAns.act_logits);
    const T = e.wasm_laya_temp(kind, outs.logits.length);
    const pw = softmax(outs.logits.map((v) => v / T));
    const pn = softmax(natAns.logits.map((v) => v / T));
    dp = Math.max(...pw.map((v, i) => Math.abs(v - pn[i])));
  }
  return {
    model: 'laya',
    mode,
    tokens: ids.length,
    case: casePath,
    load,
    forwards: { warmup: WARMUP, repeats: REPEATS, inside_ms, wall_ms,
      mean_inside_ms: mean(inside_ms), trapped },
    drift_vs_native: drift && { ...drift, dp_after_temp: dp },
    engine_ref: { noul: ans.noul, p_true: ans.probabilities.true },
    logits: outs ? outs.logits : null,
    act_logits: outs ? outs.act : null,
  };
}

function openthaiQuestions(cs) {
  return Object.entries(cs.answers).map(([qid, a]) => {
    const labels = a.probabilities ? Object.keys(a.probabilities) : null;
    return { qid, kind: { choice: 0, score: 1, noul: 2 }[a.type],
      k: a.type === 'noul' ? 2 : labels.length, labels };
  });
}

function openthaiProbs(logitsRows, qs, temps) {
  // decision.rs: valid = row[..k]/T + row[255]/T; softmax; renormalize over
  // the k option slots; noul reads p[1] of the renormalized pair.
  return qs.map((q, i) => {
    const row = logitsRows[i];
    const T = temps[q.kind];
    const valid = [...row.slice(0, q.k).map((v) => v / T), row[255] / T];
    const p = softmax(valid);
    const sum = p.slice(0, q.k).reduce((a, b) => a + b, 0);
    const opts = p.slice(0, q.k).map((v) => v / sum);
    return { qid: q.qid, kind: q.kind, p: opts, abstain: p[q.k] };
  });
}

function openthaiRun(gguf, casePath, nativePath, mode, repeats = REPEATS, nativePath2 = null) {
  const load = loadModel(1, gguf, mode);
  const cs = JSON.parse(readFileSync(casePath));
  const raw = cs._raw[0];
  const ids = raw.input_ids;
  const positions = raw.answer_positions;
  const qs = openthaiQuestions(cs);
  const { inside_ms, wall_ms, outs, trapped } = forwardLoop(1, ids, positions, 0, repeats);
  let drift = null;
  let dp = null;
  let flips = null;
  if (outs && !trapped) {
    const nat = JSON.parse(readFileSync(nativePath));
    const natRows = qs.map((q) => nat.answers[q.qid].logits);
    const natFlat = natRows.flat();
    drift = driftStats(outs.logits, natFlat);
    const tp = e.wasm_alloc_u32(3) >>> 0;
    e.wasm_openthai_temps(tp);
    const t = Array.from(new Float32Array(e.memory.buffer, tp, 3));
    e.wasm_free_u32(tp, 3);
    const pw = openthaiProbs(chunk(outs.logits, 256), qs, t);
    const pn = openthaiProbs(natRows, qs, t);
    dp = 0;
    flips = 0;
    pw.forEach((w, i) => {
      if (w.kind === 0) {
        const bw = w.p.indexOf(Math.max(...w.p));
        const bn = pn[i].p.indexOf(Math.max(...pn[i].p));
        if (bw !== bn) flips++;
      }
      dp = Math.max(dp, ...w.p.map((v, j) => Math.abs(v - pn[i].p[j])));
    });
  }
  return {
    model: 'openthai',
    mode,
    tokens: ids.length,
    case: casePath,
    load,
    forwards: { warmup: WARMUP, repeats, inside_ms, wall_ms,
      mean_inside_ms: mean(inside_ms), trapped },
    drift_vs_native: drift && { ...drift, dp_after_temp: dp, choice_flips: flips },
    logits_head: outs ? outs.logits.slice(0, 6) : null,
    logits: outs ? outs.logits : null,
    drift_vs_native2: null,
  };
}

function driftAgainst(run, nativePath) {
  const cs = JSON.parse(readFileSync(run.case));
  const qs = openthaiQuestions(cs);
  const nat = JSON.parse(readFileSync(nativePath));
  const natRows = qs.map((q) => nat.answers[q.qid].logits);
  const d = driftStats(run.logits, natRows.flat());
  const tp = e.wasm_alloc_u32(3) >>> 0;
  e.wasm_openthai_temps(tp);
  const t = Array.from(new Float32Array(e.memory.buffer, tp, 3));
  e.wasm_free_u32(tp, 3);
  const pw = openthaiProbs(chunk(run.logits, 256), qs, t);
  const pn = openthaiProbs(natRows, qs, t);
  let dp = 0;
  let flips = 0;
  pw.forEach((w, i) => {
    if (w.kind === 0) {
      const bw = w.p.indexOf(Math.max(...w.p));
      const bn = pn[i].p.indexOf(Math.max(...pn[i].p));
      if (bw !== bn) flips++;
    }
    dp = Math.max(dp, ...w.p.map((v, j) => Math.abs(v - pn[i].p[j])));
  });
  return { ...d, dp_after_temp: dp, choice_flips: flips };
}

function chunk(a, n) {
  const out = [];
  for (let i = 0; i < a.length; i += n) out.push(a.slice(i, i + n));
  return out;
}

try {
  if (MODEL === 'laya') {
    result.runs.push(layaRun(opt('--gguf'), opt('--case'), opt('--native'), MODE));
  } else if (MODEL === 'openthai') {
    const run = openthaiRun(opt('--gguf'), opt('--case'), opt('--native'), MODE);
    if (opt('--native2') && run.logits) run.drift_vs_native2 = driftAgainst(run, opt('--native2'));
    result.runs.push(run);
  } else if (MODEL === 'dual') {
    const laya = layaRun(opt('--gguf'), opt('--case'), opt('--native'), MODE);
    result.runs.push(laya);
    result.linear_after_laya = linear();
    const ot = openthaiRun(opt('--gguf2'), opt('--case2'), opt('--native2'), MODE2, Math.min(REPEATS, 3));
    result.runs.push(ot);
  } else {
    throw new Error(`unknown --model ${MODEL}`);
  }
  result.memory.linear_peak = linear();
  result.memory.peak_alloc_bytes = e.wasm_stat_peak_alloc() >>> 0;
  result.memory.live_alloc_bytes = e.wasm_stat_live_alloc() >>> 0;
  result.memory.alloc_calls = e.wasm_stat_alloc_calls();
  const d = diag();
  if (d) result.memory.allocator_diag = d;
} catch (err) {
  result.fatal = String(err?.stack || err);
  result.memory.linear_at_fatal = linear();
  result.memory.peak_alloc_bytes = e.wasm_stat_peak_alloc() >>> 0;
}

const out = opt('--out');
if (out) {
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(result, null, 2));
}
console.log(JSON.stringify({ ...result, runs: result.runs?.map((r) => ({
  model: r.model, mode: r.mode, tokens: r.tokens,
  forwards: r.forwards, load: r.load, drift: r.drift_vs_native,
})) }, null, 2));
process.exit(result.fatal ? 2 : 0);
