# 02: Phase 2 — Post-Phase-1 optimization round (pick from profile evidence)

**What to build:** After Phase 1 lands and the new baseline is measured, choose and ship the follow-up optimizations that profiling still justifies. Candidate menu, in expected-value order:

1. **Q8_0 quantized weights** — halves memory traffic again, but probabilities drift slightly; requires an explicit accuracy-tolerance decision and re-validation of the report's reference cases (today's posture is bit-identical determinism).
2. **Weighted work partition for hybrid P-core/E-core CPUs** — give E-cores narrower column slices and/or pin threads, shaving straggler wait at every matmul barrier.
3. **Vectorize scalar helpers** — attention dot-product, layer norm, GELU (currently scalar with per-element libm calls).
4. **Request-level batching** — fold concurrent requests into one GEMM for throughput under parallel load.

Items the post-Phase-1 profile does not justify are dropped with a written rationale rather than implemented.

**Blocked by:** 01 — Phase 1 memory-traffic rework (must land first: it changes the kernel seam and defines the baseline every item here is measured against).

**Status:** ready-for-agent

- [ ] Post-Phase-1 bottleneck re-measured (per-request latency + computed effective GB/s) and written down before any item is chosen
- [ ] The quantization question gets an explicit decision: adopt Q8_0 with a defined probability tolerance and re-validated English/Thai reference outputs, or reject with rationale (keeps the bit-identical posture)
- [ ] Every implemented item ships with before/after numbers from the same benchmark harness and unchanged-or-revalidated outputs
- [ ] No regression vs the Phase 1 result: 16T mean latency on both models not worse than Phase 1's recorded numbers
