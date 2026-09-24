use alloc::{string::String, sync::Arc, vec, vec::Vec};
use core::arch::x86_64::*;
use goish::{go, sync::WaitGroup};
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
                if bytes.len()
                    != n
                        .checked_mul(2)
                        .ok_or("matrix byte size overflow")?
                {
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
        let wg = Arc::new(WaitGroup::new());
        wg.Add((count - 1) as i64);
        // Each worker owns disjoint output columns. Inputs live through Wait().
        // Kernels cannot panic after these validated dimensions; no pointer escapes.
        for t in 1..count {
            let wg = wg.clone();
            let c0 = (t * per).min(n);
            let c1 = ((t + 1) * per).min(n);
            go!(move || {
                if c1 > c0 {
                    run_kernel(xp, wp, kind, op, bp, k, n, c0, c1, m);
                }
                wg.Done();
            });
        }
        run_kernel(xp, wp, kind, op, bp, k, n, 0, per.min(n), m);
        wg.Wait();
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
/// One weight-column product for NR token rows starting at absolute row r.
/// Const NR keeps the accumulator array in YMM registers (fully unrolled).
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
                let q = _mm256_cvtepi8_epi32(_mm_loadl_epi64((off + 2 + sub * 8) as *const __m128i));
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
            o[j] = (a[j] - mean) * scale * w.map(|w| w[j]).unwrap_or(1.)
                + b.map(|b| b[j]).unwrap_or(0.)
        }
    }
    y
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
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<_> = x.iter().map(|v| libm::expf(v - m)).collect();
    let sum = p.iter().sum::<f32>();
    for v in &mut p {
        *v /= sum
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
    Ok(())
}
