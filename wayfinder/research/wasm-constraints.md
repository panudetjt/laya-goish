---
wayfinder: research
ticket: 04-wasm-constraints
date: 2026-09-24
---

# wasm32 constraints for sub-2GB weight inference

**Scope.** This is a go/no-go constraint sheet for compiling and running real inference for the Laya English encoder (~807 MiB F16) and OpenThai-SystemOne (~1.5 GiB F16, Gated-DeltaNet/GQA hybrid) in wasm32 in browsers and Node/Bun/Docker-wasm. Sources were accessed **2026-09-24**. “Last updated” or commit dates are included where the source exposes them. Model sizes are project inputs, not independently verified here.

## 1. MEMORY

### 1.1 wasm32 ceiling and practical headroom

1. **The architectural ceiling is 4 GiB, not a guaranteed allocatable heap.** WebAssembly pages are 64 KiB and the JS API caps an i32-addressed memory at 65,536 pages (4 GiB). `WebAssembly.Memory` may reserve the requested maximum and allocation can fail for host-resource reasons. [MDN WebAssembly.Memory](https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/JavaScript_interface/Memory) (accessed 2026-09-24; MDN commit `bdab3a1efc984f4915590ba0a3099442e5e6f675`, document modified 2026-05-14).

2. **V8/Chrome.** Current V8 source states 4 GiB/65,536 pages for wasm32 on 64-bit hosts and just under 2 GiB on 32-bit hosts; these are implementation maxima, while runtime allocation can be lower. [V8 `wasm-limits.h`](https://github.com/v8/v8/blob/361d00046cdd30d7c70a2cd313baa3ca95186524/src/wasm/wasm-limits.h) (accessed 2026-09-24; source commit 2026-09-24). V8’s 2020 explanation says 2–4 GiB required an explicit Emscripten `MAXIMUM_MEMORY=4GB` build setting at that time; this is a toolchain requirement, not evidence that modern Rust toolchains need a browser flag. [V8 4GB post](https://v8.dev/blog/4gb-wasm-memory) (accessed 2026-09-24; published 2020-05-14). **No Chrome runtime flag is required for ordinary wasm32 4-GiB memories in current V8; the relevant risk is virtual-address reservation and physical commit.**

3. **V8 has a large virtual reservation cost.** Node’s current documentation says its trap-handler optimization typically reserves an ~8-GiB virtual-memory cage per 32-bit Wasm memory (and ~16 GiB for wasm64); it can fail under `ulimit -v` or similar limits and can be disabled with `--disable-wasm-trap-handler` at a performance cost. [Node CLI, `--disable-wasm-trap-handler`](https://github.com/nodejs/node/blob/72bb834e247bd7a3c7e8b60cbcce346567e3cb48/doc/api/cli.md#--disable-wasm-trap-handler) (accessed 2026-09-24; source commit 2026-09-24). This is why “4 GiB wasm32” should be budgeted as a practical **2–4 GiB** envelope, not as 4 GiB available for weights.

4. **Firefox/SpiderMonkey.** SpiderMonkey documents reserving roughly 4 GiB of address space for a 32-bit memory on 64-bit systems, with guard space and fallback bounds checks on constrained platforms. [SpiderMonkey “Is Memory64 actually worth using?”](https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using.html) (accessed 2026-09-24; published 2025-01-15). Its implementation also tracks a per-process Wasm reservation limit and can throw OOM when reservations are exhausted. [Firefox `ArrayBufferObject.cpp`](https://github.com/mozilla-firefox/firefox/blob/c565130f5c47d68264b2d1e11a96da012f94e63b/js/src/vm/ArrayBufferObject.cpp) (accessed 2026-09-24; commit date not exposed by page). **There is no single portable Firefox “usable heap” number.**

5. **Safari/JavaScriptCore.** WebKit’s public feature/status material does not provide a stable, portable maximum usable wasm32 heap for Safari; OS/device address-space and process-memory pressure remain relevant. **OPEN QUESTION:** a current Safari/WebKit primary source giving a default maximum/flag for a 1.5-GiB wasm32 memory was not found. Do not use a Safari-specific ceiling in the gate; test current Safari versions and iOS separately.

6. **Node.** Node embeds V8, so wasm32’s module memory ceiling is the V8 ceiling, not the JavaScript old-space limit. Node’s worker threads and `SharedArrayBuffer` sharing are stable and no `--experimental-wasm-threads` flag is documented in Node 26. [Node worker threads](https://nodejs.org/docs/latest-v26.x/api/worker_threads.html#worker-threads) (accessed 2026-09-24; Node v26.10.0 docs). Node’s JS heap flags do not enlarge Wasm linear memory; process virtual-memory limits can still prevent allocation.

7. **Bun.** Bun documents `WebAssembly.Memory`, `fetch`, streams, workers, and related Web APIs, but does not publish a Bun-specific wasm32 memory maximum. [Bun Web APIs](https://bun.sh/docs/runtime/web-apis) (accessed 2026-09-24; Bun repository latest commit 2026-09-24). Bun uses JavaScriptCore, so the browser Safari/JSC uncertainty applies. **OPEN QUESTION:** obtain a version-pinned Bun/JSC allocation experiment before treating Bun as equivalent to Node.

8. **Wasmtime/Wasmer in Docker-wasm.** Native Wasm runtimes are not bound to the browser JS API’s 4-GiB address-space rule in the same way; their effective limit is the module’s declared limit plus host/runtime resource limits. Wasmtime exposes configurable memory reservation/resource controls and supports wasm32/wasm64 features through `Config`; its examples demonstrate growth and out-of-bounds traps. [Wasmtime `Config`](https://docs.wasmtime.dev/api/wasmtime/struct.Config.html) (accessed 2026-09-24; Wasmtime 50.0.0-dev docs, repository commit 2026-09-24), [linear-memory example](https://docs.wasmtime.dev/examples-memory.html) (accessed 2026-09-24). Container cgroups, `ulimit`, and runtime configuration are the practical limits. **OPEN QUESTION:** exact Wasmer default maximum for the selected Docker image; the public URL checked did not resolve.

### 1.2 memory64 status

9. The current WebAssembly proposals index places **Memory64 in the finished/spec track rather than the active-proposal list**, while the archived proposal records core implementation as done in V8/Chrome and Firefox, Safari unknown. [WebAssembly proposals index](https://github.com/WebAssembly/proposals/blob/f0db14a5555abf7b931667fd289755124a3bf37e/README.md) (accessed 2026-09-24; commit 2026-08-10), [Memory64 implementation status](https://github.com/WebAssembly/memory64/blob/9003cd5e24e53b84cd9027ea3dd7ae57159a6db1/proposals/memory64/Overview.md#implementation-status) (accessed 2026-09-24; repository archived 2025-01-15). The current JS API exposes `address: "i64"`; MDN documents it, with i64 initial/max values as BigInts. [MDN Memory constructor](https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/JavaScript_interface/Memory/Memory) (accessed 2026-09-24; modified 2026-05-14).

10. **Shipping is not the same as usable ML toolchain support.** V8 and Firefox implementation status is “done” for core Memory64 in the proposal record, Safari is `?`; table64 remains incomplete in V8/Firefox/Safari. The JS API documentation says Memory64 can address up to 16 GiB in current browser API descriptions, but a Rust application must be compiled for a memory64 ABI and all dependencies must support it. [SpiderMonkey analysis](https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using.html) (accessed 2026-09-24) reports a 16-GiB JS API constraint.

11. **Rust/framework status is a gate blocker, not a fallback.** A production wasm64 Rust target/cargo/wasm-pack path, including `std` and Burn/Candle tensor backends, was not established from primary sources in this investigation. **OPEN QUESTION:** verify current `rustc --print target-list`, `wasm-bindgen`, `wasm-pack`, Burn, and Candle support for a version-pinned `wasm64-unknown-unknown`/`wasm64v1` target. Neither Burn nor Candle’s cited browser examples advertise Memory64; their examples target ordinary wasm32. Do not assume Memory64 rescues this ticket.

## 2. WEIGHT LOADING AT 0.8–1.5 GiB

### 2.1 Loading strategies

12. `fetch()` plus `ReadableStream` can consume a response incrementally; the Wasm JS API’s streaming compilation methods avoid materializing the Wasm binary as a separate ArrayBuffer. For model weights, a custom loader can read chunks, parse tensor metadata, and copy/dequantize into preallocated Wasm memory. [MDN Wasm JS API](https://developer.mozilla.org/en-US/docs/WebAssembly/Guides/Using_the_JavaScript_API#streaming_the_webassembly_module) (accessed 2026-09-24; modified 2026-02-19), [Bun Web APIs](https://bun.sh/docs/runtime/web-apis) (accessed 2026-09-24).

13. **Streaming is not mmap.** Browser Wasm linear memory is a contiguous byte array; `WebAssembly.Memory` has `buffer` and `grow`, not file-backed mmap. OPFS can supply offset reads, but a model implementation must explicitly page/copy data into Wasm memory. MDN describes OPFS sync offset reads only in workers. [MDN OPFS](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system) (accessed 2026-09-24; modified 2025-07-14).

14. **Avoid `response.arrayBuffer()` for a 0.8–1.5 GiB file.** It creates a large host-side buffer before the loader can release it. Use a stream reader, bounded staging buffer, or worker OPFS reads. This is an engineering implication of the API model, not a browser promise that streaming eliminates all copies.

### 2.2 Persistent stores

15. OPFS and IndexedDB are origin-scoped storage, subject to quota and eviction. OPFS supports high-performance in-place files and synchronous `read()`/`write()` through `FileSystemSyncAccessHandle` in Web Workers. [MDN OPFS](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system) (accessed 2026-09-24; modified 2025-07-14).

16. Quotas are browser-dependent: current MDN reports Chromium origins up to about 60% of disk, Firefox best-effort as the smaller of 10% of disk or 10 GiB, and Safari browser apps about 60% of disk (with different WebView limits). `navigator.storage.estimate()` is only an estimate; best-effort data can be evicted under storage pressure and Safari may proactively evict script data after seven days without interaction. [MDN storage quotas](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria) (accessed 2026-09-24; modified 2026-01-05). A 1.5-GiB cache is therefore plausible but not durable unless persistence is requested and granted.

17. Browser HTTP cache and Cache Storage have no portable “large-file guarantee.” The HTTP cache is heuristic/browser-managed; Cache API writes are quota-governed. Transformers.js explicitly uses Cache API in browsers and a filesystem cache in Node, which is useful prior art but not evidence that every browser retains a 1.5-GiB response. [Transformers.js cache implementation](https://github.com/huggingface/transformers.js/blob/dce094eebf1675281a6f15bf0bc52b0c4c5c6210/packages/transformers/src/utils/cache.js) (accessed 2026-09-24; commit 2026-09-07). **OPEN QUESTION:** browser/version-specific Cache Storage behavior for 800 MiB–1.5 GiB objects; benchmark Chrome, Firefox, Safari with quota telemetry.

### 2.3 Prior art

18. **Transformers.js.** The project uses browser Cache API and Node filesystem caching; its current source streams a `Response.body` into a temporary file and atomically renames it, avoiding a second full host buffer. [Transformers.js `FileCache`](https://github.com/huggingface/transformers.js/blob/dce094eebf1675281a6f15bf0bc52b0c4c5c6210/packages/transformers/src/utils/cache/FileCache.js) (accessed 2026-09-24; commit 2026-09-07). Sharding and safetensors details should be treated as model/hub conventions; **OPEN QUESTION:** current default shard-size value in the exact Transformers.js release selected.

19. **Burn.** Burn’s browser MNIST example compiles the model parameters into a ~1.87-MB Wasm artifact: 1,509,747 bytes are parameters and 356,744 bytes are code/deserialization/math. It is a tiny embedded-weights demo, not evidence for external multi-GiB loading. [Burn MNIST web README](https://github.com/tracel-ai/burn/blob/862ef266ade864d6fc48df3687e9f8b911c7d05e/examples/mnist-inference-web/README.md#comparison) (accessed 2026-09-24; commit 2026-06-17).

20. **Candle.** Candle’s BERT browser example builds a wasm library and uses vanilla JS/WebWorkers; its README says assets are fetched from the web. Its current Qwen3 wasm UI fetches the model file with `response.arrayBuffer()` before constructing the model, a pattern unsuitable for 1.5 GiB without redesign. [Candle BERT WASM README](https://github.com/huggingface/candle/blob/7ad82b87e4c8dbbe8e6d84bcd05007c54675c310/candle-wasm-examples/bert/README.md) (accessed 2026-09-24; commit 2023-09-19), [Candle Qwen3 wasm loader](https://github.com/huggingface/candle/blob/a2029da3c6f95f348b131afd06eb358824106b2e/candle-wasm-examples/quant-qwen3/index.html) (accessed 2026-09-24; commit 2026-01-14).

21. **llama.cpp/wllama.** Wllama is shipped for browser Wasm SIMD inference, supports single/multi-thread builds, and explicitly recommends splitting models into chunks of at most 512 MB with parallel downloads. It states a 2-GiB single-file ArrayBuffer limitation and uses GGUF split files, not mmap. [wllama README](https://github.com/ngxson/wllama/blob/91f24918b64ca1cb471ab8dd69e78116b88ead65/README.md) (accessed 2026-09-24; commit 2026-08-23). It recommends Q4/Q5/Q6 quantization, showing that practical browser precedent solves memory with quantization plus sharding, not multi-GiB F16 tensors.

## 3. COMPUTE

### 3.1 Threads

22. Wasm threads use shared linear memory and atomics; the embedder creates/joins workers. [WebAssembly Threads proposal](https://github.com/WebAssembly/threads/blob/cc535ada1aa21cfaa3cabf3ac73b89acef78a0a0/proposals/threads/Overview.md) (accessed 2026-09-24; commit 2026-07-30).

23. In browsers, sharing `SharedArrayBuffer`/shared Wasm memory requires a secure, cross-origin-isolated document: `Cross-Origin-Opener-Policy: same-origin` plus `Cross-Origin-Embedder-Policy: require-corp` or `credentialless` (and compatible CORS/CORP on resources). [MDN COOP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cross-Origin-Opener-Policy) (accessed 2026-09-24; modified 2026-07-12), [MDN COEP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cross-Origin-Embedder-Policy) (accessed 2026-09-24; modified 2026-09-04), [MDN SharedArrayBuffer](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer) (accessed 2026-09-24; modified 2026-02-10). Safari supports SharedArrayBuffer only under the same security model; exact Safari-version thread performance/limits remain an OPEN QUESTION.

24. Node’s worker threads and SharedArrayBuffer are stable; no current Node flag is needed for ordinary worker-thread sharing. [Node worker threads](https://nodejs.org/docs/latest-v26.x/api/worker_threads.html#worker-threads) (accessed 2026-09-24). Bun advertises workers and streams but no version-pinned Wasm-thread guarantee; test the selected Bun release. [Bun Web APIs](https://bun.sh/docs/runtime/web-apis) (accessed 2026-09-24).

25. Wasmtime exposes a `wasm_threads` configuration capability and a multithreaded embedding example; native runtimes do not have COOP/COEP, but Docker cgroups and runtime configuration still govern memory and thread availability. [Wasmtime Config](https://docs.wasmtime.dev/api/wasmtime/struct.Config.html#method.wasm_threads) (accessed 2026-09-24; docs current to 50.0.0-dev), [multithreaded embedding](https://docs.wasmtime.dev/examples-multithreaded-embedding.html) (accessed 2026-09-24). **OPEN QUESTION:** exact Wasmer thread feature/default for the chosen image.

### 3.2 SIMD128 and F16

26. SIMD128 is standardized and broadly implemented; its portable type is a 128-bit `v128` with f32x4/f64x2 and integer lanes. The standard SIMD definition includes signed i16 dot product (`i32x4.dot_i16x8_s`) but not x86 AVX2’s full instruction set. [WebAssembly SIMD specification](https://github.com/WebAssembly/simd/blob/a78b98a6899c9e91a13095e560767af6e99d98fd/proposals/simd/SIMD.md) (accessed 2026-09-24; archived repository commit 2021-11-16), [MDN SIMD reference](https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/SIMD) (accessed 2026-09-24; MDN current page).

27. SIMD128 does **not** provide native f16 arithmetic. The WebAssembly proposals index lists FP16 as only Phase 2, not standardized/shipping. [WebAssembly proposals](https://github.com/WebAssembly/proposals/blob/f0db14a5555abf7b931667fd289755124a3bf37e/README.md#phase-2---proposed-spec-text-available-cg--wg) (accessed 2026-09-24; commit 2026-08-10). Relaxed SIMD is a separate proposal/feature; do not assume dot-product or f16 support from baseline SIMD128.

28. Chrome/V8, Firefox, Safari, and Node ship baseline SIMD128 in normal modern configurations, but the precise minimum version is not needed for this gate because the target runtime must still be feature-detected. Use `WebAssembly.validate()`/feature detection rather than a UA string. [WebAssembly feature-status page](https://webassembly.org/features/) (accessed 2026-09-24), [V8 SIMD implementation source](https://github.com/v8/v8/tree/361d00046cdd30d7c70a2cd313baa3ca95186524/src/wasm) (accessed 2026-09-24; source commit 2026-09-24). **OPEN QUESTION:** cite/version-pin the exact first Safari and Node SIMD releases if the product matrix requires minimum-version claims.

29. **F16 implication.** With F16 weights, arithmetic must either be scalar/software-converted to f32 or use a non-baseline extension; a common safe path is to keep the bytes as 16-bit storage and upconvert tiles to f32 for compute. If the whole 807-MiB or 1.5-GiB tensor set is expanded to f32 simultaneously, the weight working set becomes approximately 1.61 GiB or 3.0 GiB respectively, before activations, allocator fragmentation, temporary tensors, code, and input/output buffers. Streaming/tile-wise conversion can avoid the full 2x peak, but only if the backend never materializes all f32 weights.

## 4. REALISTIC PRECEDENT

30. **ModernBERT-class encoder.** Candle has a native ModernBERT example (including `modern-bert-large`), but its browser WASM examples cited here are BERT/SAM/Qwen demos; the sources do not document a browser run, latency, or memory number for a ~400M-parameter ModernBERT-class encoder. [Candle ModernBERT example](https://github.com/huggingface/candle/blob/main/candle-examples/examples/modernbert/README.md) (accessed 2026-09-24; current branch; commit date not exposed here), [Candle BERT WASM README](https://github.com/huggingface/candle/blob/7ad82b87e4c8dbbe8e6d84bcd05007c54675c310/candle-wasm-examples/bert/README.md) (accessed 2026-09-24; commit 2023-09-19). **Conclusion:** no primary-source evidence found of a shipped ~0.8-GiB F16 ModernBERT browser-Wasm inference deployment with documented latency.

31. **GDN/Mamba/RWKV/linear attention.** Candle contains native Mamba and RWKV examples, but no cited browser-Wasm deployment or measured browser memory/latency. [Candle Mamba example](https://github.com/huggingface/candle/blob/648596c07389f21564b17022c88b7a4faeaad2df/candle-examples/examples/mamba/README.md) (accessed 2026-09-24; commit 2025-04-03), [Candle RWKV example](https://github.com/huggingface/candle/blob/2bd0256aff1605b0514a68f4f38d833e1658f02f/candle-examples/examples/rwkv/README.md) (accessed 2026-09-24; commit 2026-02-17). **No Gated-DeltaNet wasm precedent was found.** This is an absence-of-evidence statement, not proof that no private port exists.

32. **Largest practical browser precedent.** Wllama’s own documented constraints are the strongest directly relevant evidence: browser Wasm SIMD, split files at 512 MB, multi-thread only with COOP/COEP, and quantized Q4–Q6 recommendation. It does not publish a controlled 1–3B model load-time/RSS/tokens-per-second table in the cited README. **OPEN QUESTION:** a reproducible, primary-source benchmark with model name, Q4 size, load time, peak RSS/linear memory, and tok/s for wllama; do not invent numbers.

33. WebLLM is not a wasm32 precedent: its documented architecture uses WebGPU for model inference, so its large-browser-model results do not satisfy this gate. **OPEN QUESTION:** add a version-pinned WebLLM source if WebGPU comparison is needed; it is intentionally not used as evidence for wasm32.

34. State-recurrent attention changes activations, not weight storage. GDN/linear attention can keep recurrent state proportional to hidden/head dimensions rather than sequence-length KV cache, which is favorable for long contexts; it does not make 1.5 GiB F16 weights fit in a 2-GiB practical heap, nor does it remove temporary f32 tensors during matrix operations. This is an algorithmic inference from the model architecture and must be validated against the actual OpenThai implementation.

## 5. VERDICT

### 5.1 Laya English encoder — 807 MiB F16

**Classification: FEASIBLE WITH KNOWN TECHNIQUES, but not FEASIBLE TODAY for an unmodified Burn/Candle “load whole F16 file then convert” path.**

- A 4-GiB wasm32 address ceiling leaves 3.2 GiB nominally after 807 MiB, but V8 may reserve large virtual cages, browsers may fail allocation under address/commit pressure, and the inference heap also needs activations, allocator metadata, code, tokenizer/input buffers, temporaries, and output. [V8 limits](https://github.com/v8/v8/blob/361d00046cdd30d7c70a2cd313baa3ca95186524/src/wasm/wasm-limits.h) (accessed 2026-09-24; commit 2026-09-24), [Node trap-handler note](https://github.com/nodejs/node/blob/72bb834e247bd7a3c7e8b60cbcce346567e3cb48/doc/api/cli.md#--disable-wasm-trap-handler) (accessed 2026-09-24).
- A naive F16-to-F32 full expansion is ~1.61 GiB, leaving ~2.39 GiB nominally; that can be workable on a generous desktop but is not a cross-browser guarantee. Streaming tiles, preallocating a bounded staging buffer, and/or quantizing weights to Q8/Q4 are the known techniques.
- Browser deployment additionally needs SIMD feature detection; threads are optional but require COOP/COEP and compatible cross-origin model assets. Wllama demonstrates the sharding/parallel-download/quantization pattern, not an F16 encoder. [wllama](https://github.com/ngxson/wllama/blob/91f24918b64ca1cb471ab8dd69e78116b88ead65/README.md) (accessed 2026-09-24; commit 2026-08-23).
- **Gate interpretation:** compile/run is plausible for a custom streaming/quantized backend in Node and desktop Chromium/Firefox, but the requested “both models, all browser runtimes” gate must not pass Laya until a real inference test demonstrates peak memory and Safari behavior.

### 5.2 OpenThai-SystemOne — 1.5 GiB F16 GDN/GQA hybrid

**Classification: BLOCKED for a portable wasm32 F16 migration; FEASIBLE WITH KNOWN TECHNIQUES only after model-specific quantization and a streaming/tiled backend, which is a new implementation rather than a routine Burn/Candle migration.**

- 1.5 GiB of bytes leaves only ~0.5 GiB under a 2-GiB practical budget; whole-model f32 expansion is ~3.0 GiB and leaves roughly 1.0 GiB under the absolute 4-GiB ceiling before runtime/activations. A naive loader can therefore fail during conversion even when the raw file fits.
- GDN’s recurrent state may be constant in sequence length, unlike a transformer KV cache, but its matrix/tensor intermediates and the 1.5-GiB weights remain. The absence of documented GDN wasm precedent increases compute/backend risk; there is no evidence here that Burn or Candle’s browser backends implement this exact hybrid.
- Sharding alone fixes download/host-buffer pressure, not the requirement that the active weights and compute temporaries fit in one wasm32 linear memory. Quantization plus tile-wise dequantization is required; likely Q8 at minimum for quality investigation, possibly lower with model validation.
- **Gate interpretation:** do not approve migration on the basis of “wasm32 supports 4 GiB.” Require a dedicated quantized kernel/backend prototype and real inference measurements first. If F16 fidelity is mandatory, classify as blocked for browser wasm32.

### 5.3 Browser vs Node/Bun/Docker-wasm

- **Browser:** hardest portability target: wasm32 memory/commit limits, Safari uncertainty, origin storage/eviction, COOP/COEP for threads, CORS/CORP for cross-origin assets, and no file-backed mmap.
- **Node:** removes COOP/COEP and gives filesystem access; V8’s wasm32 and virtual-cage constraints still apply. It is the easiest JS-host target for streaming local shards, but not automatically a 4-GiB usable heap.
- **Bun:** removes browser origin/header constraints and has streams/workers, but its JavaScriptCore-specific Wasm maximum and thread behavior are not sufficiently documented here; treat as a separate test target.
- **Docker-wasm/Wasmtime:** removes browser JS and origin constraints and can apply explicit runtime/container resource limits; a native host can use file I/O and runtime memory controls. That makes OpenThai more plausible there, but it does not satisfy the browser gate and does not create a portable Burn/Candle wasm32 solution.

## SOURCES

All links below were accessed **2026-09-24**. Dates in parentheses are publication, document-modified, repository-commit, or release dates as identified above.

- [WebAssembly proposals index](https://github.com/WebAssembly/proposals/blob/f0db14a5555abf7b931667fd289755124a3bf37e/README.md) — proposal phases/status (commit 2026-08-10).
- [Memory64 proposal status](https://github.com/WebAssembly/memory64/blob/9003cd5e24e53b84cd9027ea3dd7ae57159a6db1/proposals/memory64/Overview.md) — 32-bit/64-bit memory model and engine status (repository archived 2025-01-15).
- [WebAssembly feature status](https://webassembly.org/features/) — implementation-status dashboard (accessed 2026-09-24).
- [V8 4GB Wasm](https://v8.dev/blog/4gb-wasm-memory) — historical 2–4-GiB toolchain details (published 2020-05-14).
- [V8 wasm limits](https://github.com/v8/v8/blob/361d00046cdd30d7c70a2cd313baa3ca95186524/src/wasm/wasm-limits.h) — current V8 4-GiB wasm32/16-GiB wasm64 constants (commit 2026-09-24).
- [SpiderMonkey Memory64 analysis](https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using.html) — browser reservations and 16-GiB JS API note (published 2025-01-15).
- [MDN WebAssembly.Memory](https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/JavaScript_interface/Memory) — pages, i32/i64 address API and allocation behavior (modified 2026-05-14).
- [MDN Wasm JS API](https://developer.mozilla.org/en-US/docs/WebAssembly/Guides/Using_the_JavaScript_API) — streaming/non-streaming loading and linear memory (modified 2026-02-19).
- [MDN OPFS](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system) — OPFS worker sync access and quota relationship (modified 2025-07-14).
- [MDN storage quotas](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria) — quota, persistence, eviction (modified 2026-01-05).
- [MDN COOP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cross-Origin-Opener-Policy) and [MDN COEP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cross-Origin-Embedder-Policy) — cross-origin isolation headers (COOP modified 2026-07-12; COEP modified 2026-09-04).
- [MDN SharedArrayBuffer](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer) — SAB security and Wasm shared memory (modified 2026-02-10).
- [WebAssembly Threads](https://github.com/WebAssembly/threads/blob/cc535ada1aa21cfaa3cabf3ac73b89acef78a0a0/proposals/threads/Overview.md) — shared memory/atomics (commit 2026-07-30).
- [WebAssembly SIMD](https://github.com/WebAssembly/simd/blob/a78b98a6899c9e91a13095e560767af6e99d98fd/proposals/simd/SIMD.md) — v128/f32/f64/integer/dot-product operations (commit 2021-11-16).
- [Node CLI](https://github.com/nodejs/node/blob/72bb834e247bd7a3c7e8b60cbcce346567e3cb48/doc/api/cli.md) and [Node worker threads](https://nodejs.org/docs/latest-v26.x/api/worker_threads.html) — V8 cage behavior and stable workers/SAB (Node source/docs current 2026-09-24).
- [Wasmtime Config](https://docs.wasmtime.dev/api/wasmtime/struct.Config.html) and [linear memory example](https://docs.wasmtime.dev/examples-memory.html) — native runtime memory/features (docs current 2026-09-24).
- [Transformers.js cache](https://github.com/huggingface/transformers.js/blob/dce094eebf1675281a6f15bf0bc52b0c4c5c6210/packages/transformers/src/utils/cache.js) and [FileCache](https://github.com/huggingface/transformers.js/blob/dce094eebf1675281a6f15bf0bc52b0c4c5c6210/packages/transformers/src/utils/cache/FileCache.js) — browser Cache API and streamed filesystem cache (commit 2026-09-07).
- [Burn browser MNIST](https://github.com/tracel-ai/burn/blob/862ef266ade864d6fc48df3687e9f8b911c7d05e/examples/mnist-inference-web/README.md) — embedded tiny model size and wasm loading pattern (commit 2026-06-17).
- [Candle BERT WASM](https://github.com/huggingface/candle/blob/7ad82b87e4c8dbbe8e6d84bcd05007c54675c310/candle-wasm-examples/bert/README.md) — fetched assets/WebWorkers (commit 2023-09-19).
- [Candle Qwen3 WASM loader](https://github.com/huggingface/candle/blob/a2029da3c6f95f348b131afd06eb358824106b2e/candle-wasm-examples/quant-qwen3/index.html) — `arrayBuffer()` model loading (commit 2026-01-14).
- [Candle ModernBERT](https://github.com/huggingface/candle/blob/main/candle-examples/examples/modernbert/README.md), [Mamba](https://github.com/huggingface/candle/blob/648596c07389f21564b17022c88b7a4faeaad2df/candle-examples/examples/mamba/README.md), [RWKV](https://github.com/huggingface/candle/blob/2bd0256aff1605b0514a68f4f38d833e1658f02f/candle-examples/examples/rwkv/README.md) — native model precedents, not browser measurements (Mamba commit 2025-04-03; RWKV commit 2026-02-17).
- [wllama README](https://github.com/ngxson/wllama/blob/91f24918b64ca1cb471ab8dd69e78116b88ead65/README.md) — browser SIMD, COOP/COEP, 2-GiB single-file limit, 512-MB sharding, Q4–Q6 recommendation (commit 2026-08-23).

## OPEN QUESTIONS

1. What exact wasm32 maximum/commit behavior do current Safari 19/JSC and Bun release builds expose for a single 1.5-GiB memory?
2. What exact Rust nightly/stable targets and wasm-bindgen/wasm-pack versions support Memory64 end-to-end, and do Burn/Candle tensor backends compile/run there?
3. Which current Transformers.js release and model conversion pipeline provides the desired shard size and safetensors streaming behavior?
4. What are reproducible wllama Q4 1B measurements for load time, peak RSS/linear memory, and tokens/sec on the target machines?
5. Can OpenThai’s actual GDN implementation be tiled/quantized without accuracy loss, and what is its peak activation/state footprint?
