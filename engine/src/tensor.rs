use alloc::{string::String, vec, vec::Vec};
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(any(target_arch = "wasm32", target_arch = "aarch64"))]
use core::sync::atomic::{AtomicBool, Ordering};
pub enum Storage {
    F32(Vec<f32>),
    F16(Vec<u8>),
    Q8(Vec<u8>),
}
pub struct Matrix {
    pub cols: usize,
    pub rows: usize,
    pub storage: Storage,
}
impl Matrix {
    /// Load weight bytes keeping F16 and Q8_0 tensors resident in their
    /// on-disk layout; F32 and other quantized kinds decode eagerly. F16 and
    /// Q8_0 residency cuts streamed weight bytes (2× and ~2× again) and RAM;
    /// the kernel converts on the fly. F16→F32 is exact, so F16 outputs stay
    /// bit-identical to eager decoding; Q8_0 outputs carry quantization error
    /// bounded by the checkpoint's quantizer.
    pub fn new(kind: u32, bytes: &[u8], rows: usize, cols: usize) -> Result<Self, String> {
        let n = rows
            .checked_mul(cols)
            .ok_or("matrix element count overflow")?;
        let storage = match kind {
            1 => {
                if bytes.len() != n.checked_mul(2).ok_or("matrix byte size overflow")? {
                    return Err("matrix byte length mismatch".into());
                }
                Storage::F16(bytes.to_vec())
            }
            8 => {
                if cols % 32 != 0 {
                    return Err("Q8_0 matrix requires 32-aligned columns".into());
                }
                if bytes.len()
                    != (n / 32)
                        .checked_mul(34)
                        .ok_or("matrix byte size overflow")?
                {
                    return Err("matrix byte length mismatch".into());
                }
                Storage::Q8(bytes.to_vec())
            }
            _ => {
                let v = crate::gguf::decode(kind, bytes)?;
                if v.len() != n {
                    return Err("matrix element count mismatch".into());
                }
                Storage::F32(v)
            }
        };
        Ok(Self {
            cols,
            rows,
            storage,
        })
    }
    /// Decode one stored row (for embedding gathers). Bit-identical to eager
    /// decode of the same tensor.
    pub fn row(&self, r: usize) -> Vec<f32> {
        let k = self.cols;
        match &self.storage {
            Storage::F32(v) => v[r * k..(r + 1) * k].to_vec(),
            Storage::F16(b) => (0..k)
                .map(|j| {
                    crate::gguf::half(u16::from_le_bytes([
                        b[(r * k + j) * 2],
                        b[(r * k + j) * 2 + 1],
                    ]))
                })
                .collect(),
            Storage::Q8(_) => (0..k).map(|j| self.value(r, j)).collect(),
        }
    }
    fn value(&self, c: usize, j: usize) -> f32 {
        let i = c * self.cols + j;
        match &self.storage {
            Storage::F32(v) => v[i],
            Storage::F16(b) => crate::gguf::half(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]])),
            Storage::Q8(b) => {
                let off = (i / 32) * 34;
                let d = crate::gguf::half(u16::from_le_bytes([b[off], b[off + 1]]));
                d * (b[off + 2 + i % 32] as i8) as f32
            }
        }
    }
    pub fn apply(&self, x: &[f32], bias: Option<&[f32]>, threads: usize) -> Vec<f32> {
        assert_eq!(x.len() % self.cols, 0);
        let m = x.len() / self.cols;
        let n = self.rows;
        let k = self.cols;
        assert!(bias.map(|b| b.len() == n).unwrap_or(true));
        let mut out = vec![0.; m * n];
        // Workers own disjoint output-feature slices (stored weight rows), so
        // the weight matrix streams from RAM once per row block no matter how
        // many workers run, and small token counts still engage every thread.
        let count = threads.min(n).max(1);
        let per = n.div_ceil(count);
        let xp = x.as_ptr() as usize;
        let (wp, kind) = match &self.storage {
            Storage::F32(v) => (v.as_ptr() as usize, 0u32),
            Storage::F16(v) => (v.as_ptr() as usize, 1),
            Storage::Q8(v) => (v.as_ptr() as usize, 8),
        };
        let op = out.as_mut_ptr() as usize;
        let bp = bias.map(|b| b.as_ptr() as usize).unwrap_or(0);
        // Same fan-out as the goish engine: count-1 team workers plus the
        // caller (t=0), same per/count arithmetic, same disjoint
        // output-feature ranges [t*per, (t+1)*per) — each output element is
        // computed entirely by one worker with an unchanged summation order,
        // so results are bit-identical.
        crate::team::fanout(count, &|t| {
            let c0 = (t * per).min(n);
            let c1 = ((t + 1) * per).min(n);
            if c1 > c0 {
                run_kernel(xp, wp, kind, op, bp, k, n, c0, c1, m);
            }
        });
        out
    }
}
// Goish alpha.14 preserves YMM state with XSAVE/XRSTOR during asynchronous
// preemption. SIMD kernels can remain preemptible while workers share a P.
#[inline(never)]
fn run_kernel(
    xp: usize,
    wp: usize,
    kind: u32,
    op: usize,
    bp: usize,
    k: usize,
    n: usize,
    c0: usize,
    c1: usize,
    m: usize,
) {
    unsafe {
        kernel(xp, wp, kind, op, bp, k, n, c0, c1, m);
    }
}
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn kernel(
    xp: usize,
    wp: usize,
    kind: u32,
    op: usize,
    bp: usize,
    k: usize,
    n: usize,
    c0: usize,
    c1: usize,
    m: usize,
) {
    let x = xp as *const f32;
    let o = op as *mut f32;
    let bias = bp as *const f32;
    // Row blocks of up to 12 amortize each streamed weight load across 12
    // token rows; output-feature slices [c0, c1) keep every worker's weight
    // traffic disjoint. Both preserve each output element's summation order,
    // so results are bit-identical to the previous 4-row row-partitioned
    // kernel.
    let mut r = 0;
    while r < m {
        let nr = (m - r).min(12);
        for c in c0..c1 {
            match nr {
                12 => block::<12>(x, wp, kind, o, bias, bp, k, n, c, r),
                11 => block::<11>(x, wp, kind, o, bias, bp, k, n, c, r),
                10 => block::<10>(x, wp, kind, o, bias, bp, k, n, c, r),
                9 => block::<9>(x, wp, kind, o, bias, bp, k, n, c, r),
                8 => block::<8>(x, wp, kind, o, bias, bp, k, n, c, r),
                7 => block::<7>(x, wp, kind, o, bias, bp, k, n, c, r),
                6 => block::<6>(x, wp, kind, o, bias, bp, k, n, c, r),
                5 => block::<5>(x, wp, kind, o, bias, bp, k, n, c, r),
                4 => block::<4>(x, wp, kind, o, bias, bp, k, n, c, r),
                3 => block::<3>(x, wp, kind, o, bias, bp, k, n, c, r),
                2 => block::<2>(x, wp, kind, o, bias, bp, k, n, c, r),
                _ => block::<1>(x, wp, kind, o, bias, bp, k, n, c, r),
            }
        }
        r += nr;
    }
}
/// Scalar reference kernel for targets without hand kernels yet (aarch64 —
/// CI compile gates, wayfinder ticket 17 lands the NEON one) and the wasm32
/// fallback for engines without simd128. Same per-output arithmetic as
/// self_test's reference (sequential sum over k, bias last), judged by the
/// tolerance gate — cross-ISA bit-identity is out of scope by map decision.
#[cfg(not(target_arch = "x86_64"))]
#[inline(never)]
unsafe fn kernel_scalar(
    xp: usize,
    wp: usize,
    kind: u32,
    op: usize,
    bp: usize,
    k: usize,
    n: usize,
    c0: usize,
    c1: usize,
    m: usize,
) {
    let x = xp as *const f32;
    let o = op as *mut f32;
    let bias = bp as *const f32;
    for r in 0..m {
        for c in c0..c1 {
            let mut v = 0f32;
            for j in 0..k {
                v += fetch_w(wp, kind, c * k + j) * *x.add(r * k + j);
            }
            if bp != 0 {
                v += *bias.add(c);
            }
            *o.add(r * n + c) = v;
        }
    }
}
// aarch64: NEON tier with the scalar reference as the selectable fallback
// (wayfinder ticket 17).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn kernel(
    xp: usize,
    wp: usize,
    kind: u32,
    op: usize,
    bp: usize,
    k: usize,
    n: usize,
    c0: usize,
    c1: usize,
    m: usize,
) {
    if NEON.load(Ordering::Relaxed) {
        kernel_neon(xp, wp, kind, op, bp, k, n, c0, c1, m);
    } else {
        kernel_scalar(xp, wp, kind, op, bp, k, n, c0, c1, m);
    }
}

// Remaining targets without hand kernels: scalar reference only.
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
)))]
#[inline(never)]
unsafe fn kernel(
    xp: usize,
    wp: usize,
    kind: u32,
    op: usize,
    bp: usize,
    k: usize,
    n: usize,
    c0: usize,
    c1: usize,
    m: usize,
) {
    kernel_scalar(xp, wp, kind, op, bp, k, n, c0, c1, m);
}

// ---------------------------------------------------------------------------
// aarch64 NEON kernel tier (wayfinder ticket 17).
//
// The grid — 12 f32x4 accumulators (row blocks of up to 12 tokens), k stepped
// 16 per iteration (unroll 4 packs), fused `vfmaq_f32`, multi-vector weight
// loads `vld1q_f32_x2/x4` — is lifted from gemm 0.19's NEON f32 microkernel
// grid (gemm-f32/src/microkernel.rs:358-448, `microkernel!(["neon"], 4, ..)`
// = unroll 4, MR ≤ 16, NR ≤ 4), and the control flow mirrors the x86-64
// kernel: 12-row blocking, per-k-step dequant-on-load of the F16/Q8_0-resident
// weights, per-row sequential horizontal sum, bias last. gemm's
// `vfmaq_laneq_f32` rhs-lane broadcast assumes packed N-contiguous operands;
// our streaming no-pack layout (weights k-contiguous per feature, activations
// k-contiguous per token) maps to the plain `vfmaq_f32` form of the same
// grid — the fused FMA is the `_mm256_fmadd_ps` analog and the win over the
// wasm tier's non-fused mul+add.
//
// F16→F32 dequant is fused in the k-loop via hardware FCVT
// (`vcvt_f32_f16`/`vcvt_high_f32_f16`, 8 halfs per 16-byte load — the exact
// `_mm256_cvtph_ps` analog): bit-exact with gguf::half for every normal,
// subnormal, zero, infinity and quiet-NaN half; signaling NaNs (absent from
// trained checkpoints — both shipped models scan clean) are quieted by
// hardware, and the self-test checks the exhaustive 65536-pattern set under
// NaN-aware compare. Q8_0 decodes with NEON widen chains
// (i8x16 → i16x8 → i32x4 → f32x4 × d), porting the scalar decode as-is.
//
// Scheduling follows the ticket-19 measurement: the outer loop is
// feature-tiled (~256 KiB of streamed weight bytes per tile) so 12-row
// blocks revisit the tile from L2 instead of re-streaming the matrix from
// DRAM — pure scheduling, per-output arithmetic untouched (outputs are
// disjoint across (row, feature)). aarch64 has no wasm memory ceiling: the
// threaded fan-out (threads-1 workers + caller) runs real OS threads here.
// Cross-ISA bit-identity is out of scope (map decision): the tolerance gate
// applies.
//
// Adapted from gemm 0.19 (https://github.com/sarah-ek/gemm), MIT license:
//
//   Copyright (c) 2021 sarah
//
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to
//   permit persons to whom the Software is furnished to do so, subject to
//   the following conditions:
//
//   The above copyright notice and this permission notice shall be included
//   in all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// NEON is part of the ARMv8-A baseline (no CPU query needed — unlike the
// x86-64 tier there is no AVX2-less aarch64), but the default-on atomic
// keeps the scalar fallback selectable for A/B measurement, mirroring the
// wasm tier's switch.
#[cfg(target_arch = "aarch64")]
static NEON: AtomicBool = AtomicBool::new(true);

/// Force the scalar kernel on/off on aarch64 (A/B switch; default on).
#[cfg(target_arch = "aarch64")]
pub fn set_aarch64_neon(on: bool) {
    NEON.store(on, Ordering::Relaxed);
}

// The FCVT instructions ARE baseline NEON (`vcvt_f32_f16` /
// `vcvt_high_f32_f16` — the `_mm256_cvtph_ps` analog), but Rust's intrinsic
// wrappers for the f16 types are nightly-only (`stdarch_neon_f16`), so the
// exact instructions are emitted via stable asm on u16-typed vectors.
// FCVTL converts the source register's low 4 halfs, FCVTL2 the high 4;
// IEEE-exact for normals, subnormals, zeros, infinities and NaNs (sNaNs
// quieted — see the self-test's exhaustive check).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fcvtl(v: uint16x8_t) -> float32x4_t {
    let out: float32x4_t;
    core::arch::asm!("fcvtl {0:v}.4s, {1:v}.4h", lateout(vreg) out, in(vreg) v);
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fcvtl2(v: uint16x8_t) -> float32x4_t {
    let out: float32x4_t;
    core::arch::asm!("fcvtl2 {0:v}.4s, {1:v}.8h", lateout(vreg) out, in(vreg) v);
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fcvtl4(v: uint16x4_t) -> float32x4_t {
    let out: float32x4_t;
    core::arch::asm!("fcvtl {0:v}.4s, {1:v}.4h", lateout(vreg) out, in(vreg) v);
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "neon")]
unsafe fn kernel_neon(
    xp: usize,
    wp: usize,
    kind: u32,
    op: usize,
    bp: usize,
    k: usize,
    n: usize,
    c0: usize,
    c1: usize,
    m: usize,
) {
    let x = xp as *const f32;
    let o = op as *mut f32;
    let bias = bp as *const f32;
    // Row blocks of up to 12 tokens amortize each streamed weight load
    // across 12 rows (12 f32x4 accumulators — 3 of gemm's 4-pack rows);
    // the feature tiling above each block keeps the streamed weight chunk
    // L2-resident (see the section comment).
    let wb = match kind {
        1 => k * 2,
        8 => k * 34 / 32,
        _ => k * 4,
    };
    let tile = (262144 / wb).clamp(1, 512);
    let mut ct = c0;
    while ct < c1 {
        let ce = (ct + tile).min(c1);
        let mut r = 0;
        while r < m {
            let nr = (m - r).min(12);
            for c in ct..ce {
                ndispatch(x, wp, kind, o, bias, bp, k, n, c, r, nr);
            }
            r += nr;
        }
        ct = ce;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ndispatch(
    x: *const f32,
    wp: usize,
    kind: u32,
    o: *mut f32,
    bias: *const f32,
    bp: usize,
    k: usize,
    n: usize,
    c: usize,
    r: usize,
    nr: usize,
) {
    match nr {
        12 => nblock::<12>(x, wp, kind, o, bias, bp, k, n, c, r),
        11 => nblock::<11>(x, wp, kind, o, bias, bp, k, n, c, r),
        10 => nblock::<10>(x, wp, kind, o, bias, bp, k, n, c, r),
        9 => nblock::<9>(x, wp, kind, o, bias, bp, k, n, c, r),
        8 => nblock::<8>(x, wp, kind, o, bias, bp, k, n, c, r),
        7 => nblock::<7>(x, wp, kind, o, bias, bp, k, n, c, r),
        6 => nblock::<6>(x, wp, kind, o, bias, bp, k, n, c, r),
        5 => nblock::<5>(x, wp, kind, o, bias, bp, k, n, c, r),
        4 => nblock::<4>(x, wp, kind, o, bias, bp, k, n, c, r),
        3 => nblock::<3>(x, wp, kind, o, bias, bp, k, n, c, r),
        2 => nblock::<2>(x, wp, kind, o, bias, bp, k, n, c, r),
        _ => nblock::<1>(x, wp, kind, o, bias, bp, k, n, c, r),
    }
}

/// aarch64 mirror of the x86-64/wasm32 block: one weight-column product for
/// NR token rows, dequant-on-load, per-row sequential horizontal sum
/// (4-lane NEON: ((l0+l1)+l2)+l3 — the lane order differs from x86's 8-lane
/// sum; the tolerance gate judges it), bias last.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn nblock<const NR: usize>(
    x: *const f32,
    wp: usize,
    kind: u32,
    o: *mut f32,
    bias: *const f32,
    bp: usize,
    k: usize,
    n: usize,
    c: usize,
    r: usize,
) {
    let mut a = [vdupq_n_f32(0.); NR];
    let base = c * k;
    let mut j = 0;
    if kind == 8 {
        // Q8_0: per 32 values, an F16 scale then 32 signed 8-bit offsets;
        // two 16-byte NEON loads cover the 32 offsets exactly (block = 34 B).
        while j + 32 <= k {
            let off = wp + ((base + j) / 32) * 34;
            let d = crate::gguf::half(u16::from_le_bytes([
                *(off as *const u8),
                *((off + 1) as *const u8),
            ]));
            let ds = vdupq_n_f32(d);
            for sub in 0..2 {
                let q = vld1q_s8((off + 2 + sub * 16) as *const i8);
                let w01 = vmovl_s8(vget_low_s8(q));
                let w23 = vmovl_s8(vget_high_s8(q));
                let v0 = vmulq_f32(ds, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w01))));
                let v1 = vmulq_f32(ds, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w01))));
                let v2 = vmulq_f32(ds, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w23))));
                let v3 = vmulq_f32(ds, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w23))));
                for t in 0..NR {
                    let xi = x.add((r + t) * k + j + sub * 16);
                    a[t] = vfmaq_f32(a[t], v0, vld1q_f32(xi));
                    a[t] = vfmaq_f32(a[t], v1, vld1q_f32(xi.add(4)));
                    a[t] = vfmaq_f32(a[t], v2, vld1q_f32(xi.add(8)));
                    a[t] = vfmaq_f32(a[t], v3, vld1q_f32(xi.add(12)));
                }
            }
            j += 32;
        }
    } else {
        // F32 weights load with gemm's multi-vector loads; F16 weights load
        // 8 halfs per 16-byte load and convert with hardware FCVT (both
        // widen halves interleaved with the NR accumulator updates).
        while j + 16 <= k {
            let (v0, v1, v2, v3) = if kind == 1 {
                let h0 = vld1q_u16((wp + (base + j) * 2) as *const u16);
                let h1 = vld1q_u16((wp + (base + j + 8) * 2) as *const u16);
                (fcvtl(h0), fcvtl2(h0), fcvtl(h1), fcvtl2(h1))
            } else {
                let w4 = vld1q_f32_x4((wp + (base + j) * 4) as *const f32);
                (w4.0, w4.1, w4.2, w4.3)
            };
            for t in 0..NR {
                let xi = x.add((r + t) * k + j);
                a[t] = vfmaq_f32(a[t], v0, vld1q_f32(xi));
                a[t] = vfmaq_f32(a[t], v1, vld1q_f32(xi.add(4)));
                a[t] = vfmaq_f32(a[t], v2, vld1q_f32(xi.add(8)));
                a[t] = vfmaq_f32(a[t], v3, vld1q_f32(xi.add(12)));
            }
            j += 16;
        }
        while j + 8 <= k {
            let (v0, v1) = if kind == 1 {
                let h0 = vld1q_u16((wp + (base + j) * 2) as *const u16);
                (fcvtl(h0), fcvtl2(h0))
            } else {
                let w2 = vld1q_f32_x2((wp + (base + j) * 4) as *const f32);
                (w2.0, w2.1)
            };
            for t in 0..NR {
                let xi = x.add((r + t) * k + j);
                a[t] = vfmaq_f32(a[t], v0, vld1q_f32(xi));
                a[t] = vfmaq_f32(a[t], v1, vld1q_f32(xi.add(4)));
            }
            j += 8;
        }
        while j + 4 <= k {
            let v = if kind == 1 {
                fcvtl4(vld1_u16((wp + (base + j) * 2) as *const u16))
            } else {
                vld1q_f32((wp + (base + j) * 4) as *const f32)
            };
            for t in 0..NR {
                a[t] = vfmaq_f32(a[t], v, vld1q_f32(x.add((r + t) * k + j)));
            }
            j += 4;
        }
    }
    for t in 0..NR {
        let lanes: [f32; 4] = core::mem::transmute(a[t]);
        let mut v = ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3];
        let mut q = j;
        while q < k {
            v += fetch_w(wp, kind, c * k + q) * *x.add((r + t) * k + q);
            q += 1;
        }
        if bp != 0 {
            v += *bias.add(c);
        }
        *o.add((r + t) * n + c) = v;
    }
}

// ---------------------------------------------------------------------------
// wasm32 SIMD128 kernel tier (wayfinder ticket 19).
//
// The grid shape — 12 f32x4 accumulators (row blocks of up to 12 tokens),
// 4-lane k steps unrolled 2, NON-fused mul+add — is lifted from gemm 0.19's
// simd128 microkernel grid, and the control flow mirrors the x86-64 kernel
// above: 12-row blocking, per-k-step dequant-on-load of the F16/Q8_0-resident
// weights, per-row sequential horizontal sum. F16 residency with in-kernel
// dequant is the point vs candle-wasm (gemm has no wasm f16 SIMD — its f16
// path measured 4.4× slower than f32); the f16→f32 widen below is
// hand-written and bit-exact with gguf::half for every input including
// subnormals and inf/nan (the wasm self-test checks all 65536 patterns).
// Cross-ISA bit-identity is out of scope (map decision): the tolerance gate
// applies.
//
// Adapted from gemm 0.19 (https://github.com/sarah-ek/gemm), MIT license:
//
//   Copyright (c) 2021 sarah
//
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to
//   permit persons to whom the Software is furnished to do so, subject to
//   the following conditions:
//
//   The above copyright notice and this permission notice shall be included
//   in all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Lifted pieces: the microkernel grid (gemm-f32/src/microkernel.rs:501-562,
// microkernel!(["simd128"], 2, ..x3x4) = 12 accumulators, unroll 2) and the
// runtime-dispatch pattern (gemm-common/src/gemm.rs:1038-1041). wasm32 has no
// CPU feature query: gemm gates simd128 on a default-on atomic, and so do we
// (engines lacking simd128 fail at WebAssembly.validate before these
// instructions ever run; the host can flip the flag off to force the scalar
// kernel for A/B measurement).
#[cfg(target_arch = "wasm32")]
static WASM_SIMD128: AtomicBool = AtomicBool::new(true);

// Set by the wasm loader's one-time scan of the F16 tensors: inf/nan halfs
// (e == 31) switch the kernel to the full widen. Trained checkpoints never
// contain them (both shipped models scan clean), but bit-exactness with
// gguf::half is guaranteed either way; subnormals and zeros (e == 0) are
// handled by BOTH widen paths.
#[cfg(target_arch = "wasm32")]
static F16_HAS_INF: AtomicBool = AtomicBool::new(false);

#[cfg(target_arch = "wasm32")]
pub fn set_wasm_f16_has_inf() {
    F16_HAS_INF.store(true, Ordering::Relaxed);
}

/// Force the scalar kernel on/off (wasm host A/B switch; default on).
#[cfg(target_arch = "wasm32")]
pub fn set_wasm_simd128(on: bool) {
    WASM_SIMD128.store(on, Ordering::Relaxed);
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
unsafe fn kernel(
    xp: usize,
    wp: usize,
    kind: u32,
    op: usize,
    bp: usize,
    k: usize,
    n: usize,
    c0: usize,
    c1: usize,
    m: usize,
) {
    if WASM_SIMD128.load(Ordering::Relaxed) {
        kernel_simd128(xp, wp, kind, op, bp, k, n, c0, c1, m);
    } else {
        kernel_scalar(xp, wp, kind, op, bp, k, n, c0, c1, m);
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
#[target_feature(enable = "simd128")]
unsafe fn kernel_simd128(
    xp: usize,
    wp: usize,
    kind: u32,
    op: usize,
    bp: usize,
    k: usize,
    n: usize,
    c0: usize,
    c1: usize,
    m: usize,
) {
    let x = xp as *const f32;
    let o = op as *mut f32;
    let bias = bp as *const f32;
    // Row blocks of up to 12 amortize each streamed weight load across 12
    // token rows — 12 f32x4 accumulators, the gemm simd128 grid budget. The
    // outer feature tiling is pure scheduling (outputs are disjoint across
    // (row, feature), so per-element arithmetic is untouched): it keeps each
    // streamed weight chunk L2-resident while every row block of the
    // single-threaded wasm kernel visits it, instead of re-streaming the
    // whole matrix from DRAM once per 12-row block (ceil(m/12) passes for
    // the x86 kernel's thread-sliced layout).
    let wb = match kind {
        1 => k * 2,
        8 => k * 34 / 32,
        _ => k * 4,
    };
    let tile = (262144 / wb).clamp(1, 512);
    let mut ct = c0;
    while ct < c1 {
        let ce = (ct + tile).min(c1);
        let mut r = 0;
        while r < m {
            let nr = (m - r).min(12);
            for c in ct..ce {
                wdispatch(x, wp, kind, o, bias, bp, k, n, c, r, nr);
            }
            r += nr;
        }
        ct = ce;
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn wdispatch(
    x: *const f32,
    wp: usize,
    kind: u32,
    o: *mut f32,
    bias: *const f32,
    bp: usize,
    k: usize,
    n: usize,
    c: usize,
    r: usize,
    nr: usize,
) {
    match nr {
        12 => wblock::<12>(x, wp, kind, o, bias, bp, k, n, c, r),
        11 => wblock::<11>(x, wp, kind, o, bias, bp, k, n, c, r),
        10 => wblock::<10>(x, wp, kind, o, bias, bp, k, n, c, r),
        9 => wblock::<9>(x, wp, kind, o, bias, bp, k, n, c, r),
        8 => wblock::<8>(x, wp, kind, o, bias, bp, k, n, c, r),
        7 => wblock::<7>(x, wp, kind, o, bias, bp, k, n, c, r),
        6 => wblock::<6>(x, wp, kind, o, bias, bp, k, n, c, r),
        5 => wblock::<5>(x, wp, kind, o, bias, bp, k, n, c, r),
        4 => wblock::<4>(x, wp, kind, o, bias, bp, k, n, c, r),
        3 => wblock::<3>(x, wp, kind, o, bias, bp, k, n, c, r),
        2 => wblock::<2>(x, wp, kind, o, bias, bp, k, n, c, r),
        _ => wblock::<1>(x, wp, kind, o, bias, bp, k, n, c, r),
    }
}

/// f16 → f32, 4 lanes, bit-exact with gguf::half for every input:
///   normal v: bits((v + 0x1C000) << 13) rebases the exponent by +112
///   e == 0:   t = the same bits; sub = (t − 2⁻¹⁵)·2 = m·2⁻²⁴ (exact:
///             Sterbenz on t ∈ [2⁻¹⁵, 2⁻¹⁴), then an exact doubling)
///   e == 31:  0x7F80_0000 | mant << 13 (inf/nan payload preserved)
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn widen4(p: *const u8) -> v128 {
    let h: [u64; 2] = [(p as *const u64).read_unaligned(), 0];
    widen_lanes(u32x4_extend_low_u16x8(core::mem::transmute(h)))
}

// Fast widen for inf/nan-free weights (load-time scan): identical to
// widen_lanes minus the e == 31 select.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn widen_fast(lanes: v128) -> v128 {
    let norm: v128 = core::mem::transmute(i32x4_shl(
        i32x4_add(v128_and(lanes, i32x4_splat(0x7FFF)), i32x4_splat(0x1C000)),
        13,
    ));
    let half_of_t = f32x4_sub(norm, f32x4_splat(3.0517578125e-5));
    let sub = f32x4_add(half_of_t, half_of_t);
    let r = v128_bitselect(
        sub,
        norm,
        i32x4_eq(v128_and(lanes, i32x4_splat(0x7C00)), i32x4_splat(0)),
    );
    v128_xor(
        r,
        v128_and(i32x4_shl(lanes, 16), i32x4_splat(-2147483648i32)),
    )
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn widen_sel(lanes: v128, fast: bool) -> v128 {
    if fast {
        widen_fast(lanes)
    } else {
        widen_lanes(lanes)
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn widen_lanes(lanes: v128) -> v128 {
    let norm: v128 = core::mem::transmute(i32x4_shl(
        i32x4_add(v128_and(lanes, i32x4_splat(0x7FFF)), i32x4_splat(0x1C000)),
        13,
    ));
    let half_of_t = f32x4_sub(norm, f32x4_splat(3.0517578125e-5));
    let sub = f32x4_add(half_of_t, half_of_t);
    let e = v128_and(lanes, i32x4_splat(0x7C00));
    let r = v128_bitselect(sub, norm, i32x4_eq(e, i32x4_splat(0)));
    let r = v128_bitselect(
        v128_or(
            i32x4_shl(v128_and(lanes, i32x4_splat(0x3FF)), 13),
            i32x4_splat(0x7F800000),
        ),
        r,
        i32x4_eq(e, i32x4_splat(0x7C00)),
    );
    v128_xor(
        r,
        v128_and(i32x4_shl(lanes, 16), i32x4_splat(-2147483648i32)),
    )
}

/// wasm32 mirror of the x86-64 block: one weight-column product for NR token
/// rows, dequant-on-load, per-row sequential horizontal sum, bias last.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn wblock<const NR: usize>(
    x: *const f32,
    wp: usize,
    kind: u32,
    o: *mut f32,
    bias: *const f32,
    bp: usize,
    k: usize,
    n: usize,
    c: usize,
    r: usize,
) {
    let mut a = [f32x4_splat(0.); NR];
    let base = c * k;
    let mut j = 0;
    let fast = kind != 1 || !F16_HAS_INF.load(Ordering::Relaxed);
    if kind == 8 {
        // Q8_0: per 32 values, an F16 scale then 32 signed 8-bit offsets;
        // two 16-byte loads cover the 32 offsets exactly (block = 34 B).
        while j + 32 <= k {
            let off = wp + ((base + j) / 32) * 34;
            let d = crate::gguf::half(u16::from_le_bytes([
                *(off as *const u8),
                *((off + 1) as *const u8),
            ]));
            let ds = f32x4_splat(d);
            for sub in 0..2 {
                let q = v128_load((off + 2 + sub * 16) as *const v128);
                let w01 = i16x8_extend_low_i8x16(q);
                let w23 = i16x8_extend_high_i8x16(q);
                let v0 = f32x4_mul(ds, f32x4_convert_i32x4(i32x4_extend_low_i16x8(w01)));
                let v1 = f32x4_mul(ds, f32x4_convert_i32x4(i32x4_extend_high_i16x8(w01)));
                let v2 = f32x4_mul(ds, f32x4_convert_i32x4(i32x4_extend_low_i16x8(w23)));
                let v3 = f32x4_mul(ds, f32x4_convert_i32x4(i32x4_extend_high_i16x8(w23)));
                for t in 0..NR {
                    let xi = x.add((r + t) * k + j + sub * 16);
                    a[t] = f32x4_add(a[t], f32x4_mul(v0, v128_load(xi as *const v128)));
                    a[t] = f32x4_add(a[t], f32x4_mul(v1, v128_load(xi.add(4) as *const v128)));
                    a[t] = f32x4_add(a[t], f32x4_mul(v2, v128_load(xi.add(8) as *const v128)));
                    a[t] = f32x4_add(a[t], f32x4_mul(v3, v128_load(xi.add(12) as *const v128)));
                }
            }
            j += 32;
        }
    } else {
        // 16-wide step: four weight vectors (four widen chains) interleaved
        // with the NR accumulator updates — hides the widen latency chain.
        while j + 16 <= k {
            let (v0, v1, v2, v3) = if kind == 1 {
                let h0 = v128_load((wp + (base + j) * 2) as *const v128);
                let h1 = v128_load((wp + (base + j + 8) * 2) as *const v128);
                (
                    widen_sel(u32x4_extend_low_u16x8(h0), fast),
                    widen_sel(u32x4_extend_high_u16x8(h0), fast),
                    widen_sel(u32x4_extend_low_u16x8(h1), fast),
                    widen_sel(u32x4_extend_high_u16x8(h1), fast),
                )
            } else {
                (
                    v128_load((wp + (base + j) * 4) as *const v128),
                    v128_load((wp + (base + j + 4) * 4) as *const v128),
                    v128_load((wp + (base + j + 8) * 4) as *const v128),
                    v128_load((wp + (base + j + 12) * 4) as *const v128),
                )
            };
            for t in 0..NR {
                let xi = x.add((r + t) * k + j);
                a[t] = f32x4_add(a[t], f32x4_mul(v0, v128_load(xi as *const v128)));
                a[t] = f32x4_add(a[t], f32x4_mul(v1, v128_load(xi.add(4) as *const v128)));
                a[t] = f32x4_add(a[t], f32x4_mul(v2, v128_load(xi.add(8) as *const v128)));
                a[t] = f32x4_add(a[t], f32x4_mul(v3, v128_load(xi.add(12) as *const v128)));
            }
            j += 16;
        }
        while j + 8 <= k {
            let v0;
            let v1;
            if kind == 1 {
                let h0 = v128_load((wp + (base + j) * 2) as *const v128);
                v0 = widen_sel(u32x4_extend_low_u16x8(h0), fast);
                v1 = widen_sel(u32x4_extend_high_u16x8(h0), fast);
            } else {
                v0 = v128_load((wp + (base + j) * 4) as *const v128);
                v1 = v128_load((wp + (base + j + 4) * 4) as *const v128);
            }
            for t in 0..NR {
                let xi = x.add((r + t) * k + j);
                a[t] = f32x4_add(a[t], f32x4_mul(v0, v128_load(xi as *const v128)));
                a[t] = f32x4_add(a[t], f32x4_mul(v1, v128_load(xi.add(4) as *const v128)));
            }
            j += 8;
        }
        while j + 4 <= k {
            let v = if kind == 1 {
                let h0 = v128_load((wp + (base + j) * 2) as *const v128);
                widen_sel(u32x4_extend_low_u16x8(h0), fast)
            } else {
                v128_load((wp + (base + j) * 4) as *const v128)
            };
            for t in 0..NR {
                a[t] = f32x4_add(
                    a[t],
                    f32x4_mul(v, v128_load(x.add((r + t) * k + j) as *const v128)),
                );
            }
            j += 4;
        }
    }
    for t in 0..NR {
        let lanes: [f32; 4] = core::mem::transmute(a[t]);
        let mut v = ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3];
        let mut q = j;
        while q < k {
            v += fetch_w(wp, kind, c * k + q) * *x.add((r + t) * k + q);
            q += 1;
        }
        if bp != 0 {
            v += *bias.add(c);
        }
        *o.add((r + t) * n + c) = v;
    }
}

/// One weight-column product for NR token rows starting at absolute row r.
/// Const NR keeps the accumulator array in YMM registers (fully unrolled).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn block<const NR: usize>(
    x: *const f32,
    wp: usize,
    kind: u32,
    o: *mut f32,
    bias: *const f32,
    bp: usize,
    k: usize,
    n: usize,
    c: usize,
    r: usize,
) {
    let mut a = [_mm256_setzero_ps(); NR];
    let base = c * k;
    let mut j = 0;
    if kind == 8 {
        // Q8_0: per 32 values, an F16 scale then 32 signed 8-bit offsets.
        while j + 32 <= k {
            let off = wp + ((base + j) / 32) * 34;
            let d = crate::gguf::half(u16::from_le_bytes([
                *(off as *const u8),
                *((off + 1) as *const u8),
            ]));
            let ds = _mm256_set1_ps(d);
            for sub in 0..4 {
                let q =
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64((off + 2 + sub * 8) as *const __m128i));
                let v = _mm256_mul_ps(ds, _mm256_cvtepi32_ps(q));
                for t in 0..NR {
                    a[t] =
                        _mm256_fmadd_ps(v, _mm256_loadu_ps(x.add((r + t) * k + j + sub * 8)), a[t]);
                }
            }
            j += 32;
        }
    } else {
        while j + 8 <= k {
            let v = if kind == 1 {
                _mm256_cvtph_ps(_mm_loadu_si128((wp + (base + j) * 2) as *const __m128i))
            } else {
                _mm256_loadu_ps((wp + (base + j) * 4) as *const f32)
            };
            for t in 0..NR {
                a[t] = _mm256_fmadd_ps(v, _mm256_loadu_ps(x.add((r + t) * k + j)), a[t]);
            }
            j += 8;
        }
    }
    for t in 0..NR {
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), a[t]);
        let mut v = lanes.iter().sum::<f32>();
        let mut q = j;
        while q < k {
            v += fetch_w(wp, kind, c * k + q) * *x.add((r + t) * k + q);
            q += 1;
        }
        if bp != 0 {
            v += *bias.add(c);
        }
        *o.add((r + t) * n + c) = v;
    }
}
#[inline(always)]
unsafe fn fetch_w(wp: usize, kind: u32, i: usize) -> f32 {
    match kind {
        1 => crate::gguf::half(u16::from_le_bytes([
            *((wp + i * 2) as *const u8),
            *((wp + i * 2 + 1) as *const u8),
        ])),
        8 => {
            let off = wp + (i / 32) * 34;
            let d = crate::gguf::half(u16::from_le_bytes([
                *(off as *const u8),
                *((off + 1) as *const u8),
            ]));
            d * (*((off + 2 + i % 32) as *const u8) as i8) as f32
        }
        _ => *((wp + i * 4) as *const f32),
    }
}
pub fn norm(x: &[f32], d: usize, w: Option<&[f32]>, b: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut y = vec![0.; x.len()];
    for (a, o) in x.chunks_exact(d).zip(y.chunks_exact_mut(d)) {
        unsafe { norm_chunk(a, o, w, b, eps) };
    }
    y
}
// Outputs must stay bit-identical to the scalar textbook form: mean/var
// reductions keep sequential summation order, and the affine is elementwise
// (sub, mul, mul, add — no reassociation, no fused contraction). Reductions
// and affine share one target_feature kernel so intrinsics stay in YMM
// registers (splitting the kernel doubled its measured cost).
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn norm_chunk(a: &[f32], o: &mut [f32], w: Option<&[f32]>, b: Option<&[f32]>, eps: f32) {
    let d = a.len();
    let mean = a.iter().sum::<f32>() / d as f32;
    let var = a
        .iter()
        .map(|v| {
            let z = v - mean;
            z * z
        })
        .sum::<f32>()
        / d as f32;
    let scale = 1. / libm::sqrtf(var + eps);
    let m = _mm256_set1_ps(mean);
    let sv = _mm256_set1_ps(scale);
    let one = _mm256_set1_ps(1.);
    let zero = _mm256_setzero_ps();
    let mut j = 0;
    while j + 8 <= d {
        let av = _mm256_loadu_ps(a[j..].as_ptr());
        let t = _mm256_mul_ps(_mm256_sub_ps(av, m), sv);
        let wv = w.map(|w| _mm256_loadu_ps(w[j..].as_ptr())).unwrap_or(one);
        let bv = b.map(|b| _mm256_loadu_ps(b[j..].as_ptr())).unwrap_or(zero);
        _mm256_storeu_ps(o[j..].as_mut_ptr(), _mm256_add_ps(_mm256_mul_ps(t, wv), bv));
        j += 8;
    }
    while j < d {
        o[j] =
            (a[j] - mean) * scale * w.map(|w| w[j]).unwrap_or(1.) + b.map(|b| b[j]).unwrap_or(0.);
        j += 1;
    }
}
// Scalar form for non-x86 targets: identical arithmetic to the textbook
// reference (sequential mean/var sums, elementwise sub/mul/mul/add affine),
// so outputs match the x86 kernel bit-for-bit by construction.
#[cfg(not(target_arch = "x86_64"))]
#[inline(never)]
unsafe fn norm_chunk(a: &[f32], o: &mut [f32], w: Option<&[f32]>, b: Option<&[f32]>, eps: f32) {
    let d = a.len();
    let mean = a.iter().sum::<f32>() / d as f32;
    let var = a
        .iter()
        .map(|v| {
            let z = v - mean;
            z * z
        })
        .sum::<f32>()
        / d as f32;
    let scale = 1. / libm::sqrtf(var + eps);
    for j in 0..d {
        o[j] =
            (a[j] - mean) * scale * w.map(|w| w[j]).unwrap_or(1.) + b.map(|b| b[j]).unwrap_or(0.);
    }
}
pub fn add(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for (a, b) in x.iter_mut().zip(y) {
        *a += b
    }
}
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1. + libm::tanhf(0.7978845608028654 * x * (1. + 0.044715 * x * x)))
}
pub fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = x.iter().map(|v| v - m).collect();
    let sum = katgpt_types::simd::simd_exp_sum_inplace(&mut p);
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in &mut p {
            *v *= inv;
        }
    }
    p
}
pub fn attention(
    qkv: &[f32],
    s: usize,
    d: usize,
    hd: usize,
    rope: Option<(&[f32], &[f32])>,
    window: Option<usize>,
) -> Vec<f32> {
    let heads = d / hd;
    let mut q = vec![0.; s * d];
    let mut k = q.clone();
    let mut v = q.clone();
    for i in 0..s {
        q[i * d..(i + 1) * d].copy_from_slice(&qkv[i * 3 * d..i * 3 * d + d]);
        k[i * d..(i + 1) * d].copy_from_slice(&qkv[i * 3 * d + d..i * 3 * d + 2 * d]);
        v[i * d..(i + 1) * d].copy_from_slice(&qkv[i * 3 * d + 2 * d..(i + 1) * 3 * d]);
    }
    if let Some((cos, sin)) = rope {
        for a in [&mut q, &mut k] {
            for i in 0..s {
                for h in 0..heads {
                    for j in 0..hd / 2 {
                        let p = i * d + h * hd + j;
                        let u = a[p];
                        let v = a[p + hd / 2];
                        a[p] = u * cos[i * hd + j] - v * sin[i * hd + j];
                        a[p + hd / 2] = v * cos[i * hd + j + hd / 2] + u * sin[i * hd + j + hd / 2];
                    }
                }
            }
        }
    }
    let mut out = vec![0.; s * d];
    let scale = 1. / libm::sqrtf(hd as f32);
    let mut score = vec![0.; s];
    for h in 0..heads {
        for i in 0..s {
            let lo = window.map(|w| i.saturating_sub(w)).unwrap_or(0);
            let hi = window.map(|w| (i + w + 1).min(s)).unwrap_or(s);
            for j in lo..hi {
                score[j] = dot(&q[i * d + h * hd..][..hd], &k[j * d + h * hd..][..hd]) * scale
            }
            let p = softmax(&score[lo..hi]);
            for (j, &p) in (lo..hi).zip(&p) {
                for z in 0..hd {
                    out[i * d + h * hd + z] += p * v[j * d + h * hd + z]
                }
            }
        }
    }
    out
}
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe { dot_kernel(a.as_ptr(), b.as_ptr(), a.len()) }
}
// Eight reassociated partial sums instead of the previous sequential sum;
// adoption was gated on measured drift against reference outputs.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_kernel(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc);
        i += 8;
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = lanes.iter().sum::<f32>();
    while i < n {
        s += *a.add(i) * *b.add(i);
        i += 1;
    }
    s
}
// Scalar form for non-x86 targets: sequential sum (the x86 kernel's 8
// reassociated partial sums are an x86-only adoption; cross-ISA outputs are
// judged by the tolerance gate). Also the wasm fallback when simd128 is off.
#[cfg(not(target_arch = "x86_64"))]
#[inline(never)]
unsafe fn dot_scalar(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut s = 0f32;
    for i in 0..n {
        s += *a.add(i) * *b.add(i);
    }
    s
}
// wasm32: 4-lane partial sums when simd128 is on (same dispatch pattern as
// the matmul kernel above).
#[cfg(target_arch = "wasm32")]
#[inline(never)]
unsafe fn dot_kernel(a: *const f32, b: *const f32, n: usize) -> f32 {
    if WASM_SIMD128.load(Ordering::Relaxed) {
        dot_simd128(a, b, n)
    } else {
        dot_scalar(a, b, n)
    }
}
// aarch64: NEON 4-lane partial sums when the tier is on (same dispatch
// pattern as the matmul kernel above).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn dot_kernel(a: *const f32, b: *const f32, n: usize) -> f32 {
    if NEON.load(Ordering::Relaxed) {
        dot_neon(a, b, n)
    } else {
        dot_scalar(a, b, n)
    }
}
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut acc = vdupq_n_f32(0.);
    let mut i = 0;
    while i + 8 <= n {
        acc = vfmaq_f32(acc, vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
        acc = vfmaq_f32(acc, vld1q_f32(a.add(i + 4)), vld1q_f32(b.add(i + 4)));
        i += 8;
    }
    while i + 4 <= n {
        acc = vfmaq_f32(acc, vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
        i += 4;
    }
    let lanes: [f32; 4] = core::mem::transmute(acc);
    let mut s = ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3];
    while i < n {
        s += *a.add(i) * *b.add(i);
        i += 1;
    }
    s
}
// Remaining targets without hand kernels: scalar reference only.
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
)))]
#[inline(never)]
unsafe fn dot_kernel(a: *const f32, b: *const f32, n: usize) -> f32 {
    dot_scalar(a, b, n)
}
#[cfg(target_arch = "wasm32")]
#[inline(never)]
#[target_feature(enable = "simd128")]
unsafe fn dot_simd128(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut acc = f32x4_splat(0.);
    let mut i = 0;
    while i + 8 <= n {
        acc = f32x4_add(
            acc,
            f32x4_mul(
                v128_load(a.add(i) as *const v128),
                v128_load(b.add(i) as *const v128),
            ),
        );
        acc = f32x4_add(
            acc,
            f32x4_mul(
                v128_load(a.add(i + 4) as *const v128),
                v128_load(b.add(i + 4) as *const v128),
            ),
        );
        i += 8;
    }
    while i + 4 <= n {
        acc = f32x4_add(
            acc,
            f32x4_mul(
                v128_load(a.add(i) as *const v128),
                v128_load(b.add(i) as *const v128),
            ),
        );
        i += 4;
    }
    let lanes: [f32; 4] = core::mem::transmute(acc);
    let mut s = ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3];
    while i < n {
        s += *a.add(i) * *b.add(i);
        i += 1;
    }
    s
}
#[cfg(target_arch = "x86_64")]
pub fn require_simd() -> Result<(), String> {
    let f = core::arch::x86_64::__cpuid(1);
    let f7 = core::arch::x86_64::__cpuid_count(7, 0);
    if f.ecx & (1 << 12) == 0
        || f.ecx & (1 << 27) == 0
        || f.ecx & (1 << 29) == 0
        || f7.ebx & (1 << 5) == 0
        || unsafe { core::arch::x86_64::_xgetbv(0) } & 6 != 6
    {
        return Err("CPU requires AVX2/FMA/F16C and OSXSAVE".into());
    }
    Ok(())
}
// aarch64: NEON is ARMv8-A baseline; probe via std's hwcaps query so a
// hypothetical non-conforming core fails loudly instead of executing
// UNDEFINED instructions.
#[cfg(target_arch = "aarch64")]
pub fn require_simd() -> Result<(), String> {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return Err("CPU lacks NEON (aarch64 baseline)".into());
    }
    Ok(())
}
// Remaining non-x86 targets run the scalar reference kernels above; per-arch
// capability checks land with their kernels.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn require_simd() -> Result<(), String> {
    Ok(())
}
pub fn self_test() -> Result<(), String> {
    require_simd()?;
    fn check(
        w: &Matrix,
        x: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        label: &str,
    ) -> Result<(), String> {
        let cols = w.cols;
        let n = w.rows;
        for threads in [1usize, 2, 4, 16] {
            let y = w.apply(x, bias, threads);
            for r in 0..rows {
                for c in 0..n {
                    let ref_row: Vec<_> = (0..cols).map(|j| w.value(c, j)).collect();
                    let mut want = dot(&x[r * cols..][..cols], &ref_row);
                    if let Some(b) = bias {
                        want += b[c];
                    }
                    if (y[r * n + c] - want).abs() > 0.0001 {
                        return Err(format!("{label} matrix kernel mismatch"));
                    }
                }
            }
        }
        Ok(())
    }
    let w = Matrix {
        cols: 17,
        rows: 11,
        storage: Storage::F32((0..187).map(|i| (i as f32 - 83.) / 71.).collect()),
    };
    let x: Vec<f32> = (0..153).map(|i| (i as f32 - 71.) / 43.).collect();
    let b = vec![0.25; 11];
    check(&w, &x, Some(&b), 9, "F32")?;
    // Half-exact values so F16 storage round-trips exactly; 13 rows cross the
    // 12-row register-block boundary. Q8_0 uses a 0.25 block scale so the
    // quarter values encode to exact int8 offsets.
    let to_half = |v: f32| -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        if v == 0. {
            return sign;
        }
        let exp = (((bits >> 23) & 0xff) as i32 - 112) as u16;
        let mant = ((bits >> 13) & 1023) as u16;
        sign | (exp << 10) | mant
    };
    let qv = |i: usize| ((i % 15) as f32 - 7.) * 0.25;
    let wf = Matrix {
        cols: 17,
        rows: 11,
        storage: Storage::F16(
            (0..187)
                .map(|i| to_half(qv(i)))
                .flat_map(|h| h.to_le_bytes())
                .collect(),
        ),
    };
    let xf: Vec<f32> = (0..13 * 17).map(|i| qv(i + 5)).collect();
    let bf = vec![0.25; 11];
    check(&wf, &xf, Some(&bf), 13, "F16")?;
    check(&wf, &xf[..17].to_vec(), None, 1, "F16 single-row")?;
    let mut q8 = Vec::with_capacity(11 * 34);
    for blk in (0..32 * 11).step_by(32) {
        q8.extend_from_slice(&0x3800u16.to_le_bytes());
        for e in 0..32 {
            q8.push((qv(blk + e) / 0.25) as i8 as u8);
        }
    }
    let wq = Matrix {
        cols: 32,
        rows: 11,
        storage: Storage::Q8(q8),
    };
    let xq: Vec<f32> = (0..13 * 32).map(|i| qv(i + 5)).collect();
    let bq = vec![0.25; 11];
    check(&wq, &xq, Some(&bq), 13, "Q8_0")?;
    check(&wq, &xq[..32].to_vec(), None, 1, "Q8_0 single-row")?;
    // With half-exact values the F16 and F32 kernels must agree bit-for-bit.
    let w32 = Matrix {
        cols: 17,
        rows: 11,
        storage: Storage::F32((0..187).map(|i| qv(i)).collect()),
    };
    for threads in [1usize, 2, 4, 16] {
        let a = wf.apply(&xf, Some(&bf), threads);
        let b = w32.apply(&xf, Some(&bf), threads);
        if a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits()) {
            return Err("F16 kernel is not bit-identical to F32".into());
        }
    }
    if (crate::gguf::half(0x3c00) - 1.).abs() > 1e-8 || crate::gguf::half(1) != 1. / 16777216. {
        return Err("half conversion mismatch".into());
    }
    // wasm32 simd128 tier: the f16→f32 widen must be bit-exact with
    // gguf::half over EVERY half bit pattern (65536 values, subnormal and
    // inf/nan included), and the scalar fallback kernel must stay correct.
    #[cfg(target_arch = "wasm32")]
    {
        for h in 0..=u16::MAX {
            let b = [
                h.to_le_bytes(),
                h.to_le_bytes(),
                h.to_le_bytes(),
                h.to_le_bytes(),
            ]
            .concat();
            let v = unsafe { widen4(b.as_ptr()) };
            let lanes: [u32; 4] = unsafe { core::mem::transmute(v) };
            let want = crate::gguf::half(h).to_bits();
            if lanes.iter().any(|&l| l != want) {
                return Err(format!("widen4 not bit-exact at half 0x{h:04x}"));
            }
            // the scan-gated fast path must agree wherever it is used
            // (every pattern without e == 31)
            if h & 0x7C00 != 0x7C00 {
                let v = unsafe {
                    widen_sel(
                        u32x4_extend_low_u16x8(v128_load(b.as_ptr() as *const v128)),
                        true,
                    )
                };
                let lanes: [u32; 4] = unsafe { core::mem::transmute(v) };
                if lanes.iter().any(|&l| l != want) {
                    return Err(format!("widen_fast not bit-exact at half 0x{h:04x}"));
                }
            }
        }
        set_wasm_simd128(false);
        let scalar_ok = check(&w, &x, Some(&b), 9, "F32 scalar-fallback");
        set_wasm_simd128(true);
        scalar_ok?;
    }
    // aarch64 NEON tier: the hardware f16→f32 conversion (FCVT) must be
    // bit-exact with gguf::half over every half bit pattern except
    // signaling NaNs (e==31 with the quiet bit clear — trained checkpoints
    // never contain them), which hardware quiets: qNaN/inf payloads
    // transfer exactly, sNaNs are compared as NaNs. The scalar fallback
    // kernel must stay correct with the tier switched off.
    #[cfg(target_arch = "aarch64")]
    {
        for h in 0..=u16::MAX {
            let b = [
                h.to_le_bytes(),
                h.to_le_bytes(),
                h.to_le_bytes(),
                h.to_le_bytes(),
            ]
            .concat();
            let v = unsafe { fcvtl4(vld1_u16(b.as_ptr() as *const u16)) };
            let lanes: [u32; 4] = unsafe { core::mem::transmute(v) };
            let want = crate::gguf::half(h);
            let wb = want.to_bits();
            let ok = if want.is_nan() {
                if h & 0x0200 != 0 {
                    lanes.iter().all(|&l| l == wb)
                } else {
                    lanes.iter().all(|&l| f32::from_bits(l).is_nan())
                }
            } else {
                lanes.iter().all(|&l| l == wb)
            };
            if !ok {
                return Err(format!("FCVT f16->f32 mismatch at half 0x{h:04x}"));
            }
        }
        set_aarch64_neon(false);
        let scalar_ok = check(&w, &x, Some(&b), 9, "F32 scalar-fallback");
        set_aarch64_neon(true);
        scalar_ok?;
    }
    // Layer norm must be BIT-identical to the textbook scalar form (zero
    // drift is a hard requirement): widths exercise lanes and tails
    // (768 = Laya hidden, 17 odd), with and without affine parameters.
    let w768: Option<Vec<f32>> = Some((0..768).map(|j| 1. + ((j % 7) as f32) / 49.).collect());
    let b768: Option<Vec<f32>> = Some((0..768).map(|j| ((j % 5) as f32 - 2.) / 64.).collect());
    for (w, b) in [(w768, b768), (None, None)] {
        let d = if w.is_some() { 768 } else { 17 };
        let x: Vec<f32> = (0..d * 3).map(|j| ((j % 29) as f32 - 14.) / 13.).collect();
        let got = norm(&x, d, w.as_deref(), b.as_deref(), 1e-5);
        for c in 0..3 {
            let a = &x[c * d..(c + 1) * d];
            let mean = a.iter().sum::<f32>() / d as f32;
            let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let scale = 1. / libm::sqrtf(var + 1e-5);
            for j in 0..d {
                let want = (a[j] - mean) * scale * w.as_deref().map(|w| w[j]).unwrap_or(1.)
                    + b.as_deref().map(|b| b[j]).unwrap_or(0.);
                if got[c * d + j].to_bits() != want.to_bits() {
                    return Err("norm kernel is not bit-identical to scalar form".into());
                }
            }
        }
    }
    Ok(())
}
