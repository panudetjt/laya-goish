// Node driver for the wasm32 candle probe: instantiate, run gemm f32/f16,
// rayon fallback, report allocator peak + linear-memory growth.
import { readFileSync } from 'node:fs';

const wasmPath = new URL('./target/wasm32-unknown-unknown/release/wasm_probe.wasm', import.meta.url);
const bytes = readFileSync(wasmPath);
const t_inst0 = performance.now();
const { instance } = await WebAssembly.instantiate(bytes, {
  env: { now_ms: () => performance.now() },
});
const instMs = performance.now() - t_inst0;

const e = instance.exports;
const memBytes = () => e.memory.buffer.byteLength;

console.log(JSON.stringify({
  wasm_bytes: bytes.length,
  instantiate_ms: Number(instMs.toFixed(1)),
  initial_linear_mem_bytes: memBytes(),
}));

// gemm f32 256x1024x1024, 3 reps
let t = performance.now();
const f32ms = e.probe_gemm(256, 1024, 1024, 3);
const f32wall = performance.now() - t;
console.log(JSON.stringify({
  probe: 'gemm_f32_256x1024x1024', last_rep_inside_ms: Number(f32ms.toFixed(2)),
  wall_3reps_ms: Number(f32wall.toFixed(1)), linear_mem_bytes: memBytes(),
  peak_alloc_bytes: e.probe_stats(),
}));

// gemm f16 (gemm f16 path) 256x1024x1024, 3 reps
t = performance.now();
const f16ms = e.probe_gemm_f16(256, 1024, 1024, 3);
const f16wall = performance.now() - t;
console.log(JSON.stringify({
  probe: 'gemm_f16_256x1024x1024', last_rep_inside_ms: Number(f16ms.toFixed(2)),
  wall_3reps_ms: Number(f16wall.toFixed(1)), linear_mem_bytes: memBytes(),
  peak_alloc_bytes: e.probe_stats(),
}));

// rayon par_iter fallback: no trap/panic means the single-thread fallback works
const r = BigInt(e.probe_rayon(10_000_000));
console.log(JSON.stringify({
  probe: 'rayon_par_iter_10M', checksum: r.toString(), linear_mem_bytes: memBytes(),
}));

// alloc-shape bisect: node probe.mjs gguf_mb,layer_mb,steps,grow [...]
const shapes = process.argv.slice(2).map((a) => a.split(',').map(Number));
for (const [gguf, layer, steps, grow] of shapes) {
  try {
    const peak = e.probe_alloc_shape(gguf, layer, steps, grow);
    console.log(JSON.stringify({ shape: { gguf, layer, steps, grow }, ok: true, peak_alloc_mb: peak, linear_mb: memBytes() >> 20 }));
  } catch (err) {
    console.log(JSON.stringify({ shape: { gguf, layer, steps, grow }, ok: false, err: String(err).slice(0, 120), linear_mb: memBytes() >> 20 }));
  }
}
