// OpenThai-SystemOne: candle port of src/models/openthai/model.rs (Gated-
// DeltaNet hybrid). Op order, eps, RoPE pairing, GQA mapping, conv1d taps and
// the per-head scalar GDN recurrence mirror the engine 1:1; only the gemm /
// attention kernels differ (candle gemm instead of the engine's hand-written
// AVX2 kernels), which is the quantity under measurement.
use crate::gguf::Gguf;
use candle_core::cpu_backend::CpuStorage;
use candle_core::{CustomOp1, DType, Device, Layout, Shape, Tensor};
use rayon::prelude::*;
use serde_json::Value;
use std::sync::Arc;

pub const D: usize = 1024;
pub const INTER: usize = 3584;
pub const LAYERS: usize = 24;
pub const FULL_HEADS: usize = 8;
pub const FULL_HD: usize = 256;
pub const KV_HEADS: usize = 2;
pub const GQA: usize = FULL_HEADS / KV_HEADS;
pub const LIN_HEADS: usize = 16;
pub const LIN_D: usize = 128;
pub const QKV_W: usize = 6144; // 3 * LIN_HEADS * LIN_D
pub const CONV_K: usize = 4;
pub const VOCAB: usize = 248339;
pub const ANSWER_TOK: u32 = 248082;
const ROPE_HALF: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightMode {
    // F16-resident weights; activations cast to F16 per matmul (gemm f16,
    // f32 accumulators). RAM parity with the engine's F16 residency, but the
    // engine keeps activations f32.
    F16,
    // F16-resident weights, dequantized to F32 on every matmul. Closest to
    // the engine's in-kernel conversion (weights converted, activations f32).
    Convert,
    // Weights upcast to F32 once at load (2x RAM; engine does NOT do this).
    Upcast,
}

impl WeightMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "f16" => Ok(Self::F16),
            "convert" => Ok(Self::Convert),
            "upcast" => Ok(Self::Upcast),
            _ => Err(format!("unknown weight mode {s}")),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GdnImpl {
    Scalar,
    Tensor,
}

pub struct Linear {
    w: Tensor, // [rows, cols], row-major GGUF layout
    b: Option<Tensor>,
    mode: WeightMode,
}

struct Full {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    qn: Vec<f32>,
    kn: Vec<f32>,
}

struct Delta {
    qkv: Linear,
    z: Linear,
    a: Linear,
    b: Linear,
    o: Linear,
    conv: Vec<f32>,
    dt: Vec<f32>,
    alog: Vec<f32>,
    norm: Vec<f32>,
}

enum Mixer {
    Full(Full),
    Delta(Delta),
}

struct Block {
    n1: Vec<f32>,
    n2: Vec<f32>,
    gate: Linear,
    up: Linear,
    down: Linear,
    mix: Mixer,
}

pub struct Model {
    pub layers: usize,
    pub full_layers: Vec<usize>,
    emb: Tensor, // [VOCAB, D]
    blocks: Vec<Block>,
    norm: Vec<f32>,
    head: Linear,
    pub temperature: [f32; 3],
    pub max_tokens: usize,
    pub max_state: usize,
    rope_cos: Arc<Vec<f32>>, // [max_tokens, ROPE_HALF]
    rope_sin: Arc<Vec<f32>>,
}

pub struct Req {
    pub ids: Vec<u32>,
    pub positions: Vec<usize>,
}


fn sigmoid(x: f32) -> f32 {
    1. / (1. + libm::expf(-x))
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    if x > 20. {
        x
    } else {
        libm::log1pf(libm::expf(x))
    }
}

// Engine tensor::softmax, exact: max-sub, exp, sequential sum, divide.
pub fn softmax_f32(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = x.iter().map(|v| libm::expf(v - m)).collect();
    let sum = p.iter().sum::<f32>();
    for v in &mut p {
        *v /= sum
    }
    p
}

// Engine tensor::dot(x, x): 8 reassociated partial sums like the engine's
// dot_kernel; ulp-level differences from its horizontal add are acceptable
// (gemm drift dominates).
fn self_dot(a: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut i = 0;
    while i + 8 <= a.len() {
        for (s, x) in acc.iter_mut().zip(&a[i..i + 8]) {
            *s += x * x;
        }
        i += 8;
    }
    let mut t = acc.iter().sum::<f32>();
    for &x in &a[i..] {
        t += x * x;
    }
    t
}

// Engine rms(): per d-wide chunk, y = x * (1/sqrt(dot/d + 1e-6)) * (offset + w).
struct RmsNorm {
    w: Vec<f32>,
    offset: f32,
}

impl CustomOp1 for RmsNorm {
    fn name(&self) -> &'static str {
        "openthai-rms"
    }
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let vs = storage.as_slice::<f32>()?;
        let (start, len) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::RequiresContiguous { op: self.name() })?;
        let x = &vs[start..start + len];
        let d = self.w.len();
        let mut out = vec![0f32; len];
        let (w, off) = (&self.w, self.offset);
        out.par_chunks_exact_mut(d)
            .zip(x.par_chunks_exact(d))
            .for_each(|(o, a)| {
                let scale = 1. / libm::sqrtf(self_dot(a) / d as f32 + 1e-6);
                for j in 0..d {
                    o[j] = a[j] * scale * (off + w[j]);
                }
            });
        let shape = layout.shape().to_owned();
        Ok((CpuStorage::F32(out), shape))
    }
}

fn rms(x: &Tensor, w: &[f32], offset: f32) -> Result<Tensor, candle_core::Error> {
    x.contiguous()?
        .apply_op1_no_bwd(&RmsNorm { w: w.to_vec(), offset })
}


// Engine conv_channels: qkv[i][c] = silu(sum_{t in max(lo,i-3)..=i}
// raw[t][c] * conv[c*4 + (t-i+3)]), lo = segment start (single request: 0).
struct ConvSilu {
    conv: Vec<f32>,
    lo: Vec<usize>,
}

impl CustomOp1 for ConvSilu {
    fn name(&self) -> &'static str {
        "openthai-conv-silu"
    }
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let vs = storage.as_slice::<f32>()?;
        let (start, len) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::RequiresContiguous { op: self.name() })?;
        let raw = &vs[start..start + len];
        let c = self.conv.len() / CONV_K;
        let mut out = vec![0f32; len];
        let (conv, lo) = (&self.conv, &self.lo);
        out.par_chunks_exact_mut(c)
            .enumerate()
            .for_each(|(i, o)| {
                let t0 = lo[i].max(i.saturating_sub(CONV_K - 1));
                for ch in 0..c {
                    let mut v = 0.;
                    for t in t0..=i {
                        v += raw[t * c + ch] * conv[ch * CONV_K + (CONV_K + t - i - 1)];
                    }
                    o[ch] = silu(v);
                }
            });
        let shape = layout.shape().to_owned();
        Ok((CpuStorage::F32(out), shape))
    }
}

// Engine rope loop: pairs (j, j+32) inside the first 64 dims of each 256-wide
// head; angle = pos / 1e7^(2j/64); pos is the position within the request
// (stacked batches restart at each segment).
struct RopeRotate {
    cos: Arc<Vec<f32>>,
    sin: Arc<Vec<f32>>,
    pos: Vec<usize>,
}

impl CustomOp1 for RopeRotate {
    fn name(&self) -> &'static str {
        "openthai-rope"
    }
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let vs = storage.as_slice::<f32>()?;
        let (start, len) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::RequiresContiguous { op: self.name() })?;
        let x = &vs[start..start + len];
        let mut out = x.to_vec();
        let (cos, sin, pos) = (&self.cos, &self.sin, &self.pos);
        let width = *layout.dims().last().unwrap();
        let heads = width / FULL_HD;
        out.par_chunks_exact_mut(FULL_HD)
            .enumerate()
            .for_each(|(r, o)| {
                let p = pos[r / heads];
                for j in 0..ROPE_HALF {
                    let (co, si) = (cos[p * ROPE_HALF + j], sin[p * ROPE_HALF + j]);
                    let a = o[j];
                    let b = o[j + ROPE_HALF];
                    o[j] = a * co - b * si;
                    o[j + ROPE_HALF] = b * co + a * si;
                }
            });
        let shape = layout.shape().to_owned();
        Ok((CpuStorage::F32(out), shape))
    }
}

// softmax over the last dim (candle's own softmax path is a composed,
// single-threaded chain; this mirrors the engine's max-sub/exp/sum/divide).
struct SoftmaxLast;

impl CustomOp1 for SoftmaxLast {
    fn name(&self) -> &'static str {
        "openthai-softmax"
    }
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let vs = storage.as_slice::<f32>()?;
        let (start, len) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::RequiresContiguous { op: self.name() })?;
        let x = &vs[start..start + len];
        let d = *layout.dims().last().unwrap();
        let mut out = vec![0f32; len];
        out.par_chunks_exact_mut(d)
            .zip(x.par_chunks_exact(d))
            .for_each(|(o, a)| {
                let p = softmax_f32(a);
                o.copy_from_slice(&p);
            });
        let shape = layout.shape().to_owned();
        Ok((CpuStorage::F32(out), shape))
    }
}

// o = x * silu(g)  (MLP join, GDN z-gate).
struct SiluMul;

impl CustomOp1 for SiluMul {
    fn name(&self) -> &'static str {
        "openthai-silu-mul"
    }
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        two_input_op(storage, layout, self.name(), |x, g| x * silu(g))
    }
}

// o = x * sigmoid(g)  (full-attention output gate).
struct SigMul;

impl CustomOp1 for SigMul {
    fn name(&self) -> &'static str {
        "openthai-sig-mul"
    }
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        two_input_op(storage, layout, self.name(), |x, g| x * sigmoid(g))
    }
}

// SiluMul/SigMul consume the [x..g] concatenation along the last dim; split
// at the midpoint inside the op so both halves stay parallel and fused.
fn two_input_op(
    storage: &CpuStorage,
    layout: &Layout,
    op: &'static str,
    f: impl Fn(f32, f32) -> f32 + Sync + Send,
) -> candle_core::Result<(CpuStorage, Shape)> {
    let vs = storage.as_slice::<f32>()?;
    let (start, len) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::RequiresContiguous { op })?;
    let v = &vs[start..start + len];
    let d = *layout.dims().last().unwrap();
    let h = d / 2;
    let rows = v.len() / d;
    let mut out = vec![0f32; rows * h];
    out.par_chunks_exact_mut(h).enumerate().for_each(|(r, o)| {
        let xs = &v[r * d..r * d + h];
        let gs = &v[r * d + h..(r + 1) * d];
        for j in 0..h {
            o[j] = f(xs[j], gs[j]);
        }
    });
    let dims: Vec<usize> = layout
        .dims()
        .iter()
        .enumerate()
        .map(|(i, &dd)| if i == layout.dims().len() - 1 { h } else { dd })
        .collect();
    Ok((CpuStorage::F32(out), Shape::from_dims(&dims)))
}

fn silu_mul(x: &Tensor, g: &Tensor) -> Result<Tensor, candle_core::Error> {
    let j = Tensor::cat(&[x.contiguous()?, g.contiguous()?], 1)?;
    j.apply_op1_no_bwd(&SiluMul)
}

fn sig_mul(x: &Tensor, g: &Tensor) -> Result<Tensor, candle_core::Error> {
    let j = Tensor::cat(&[x.contiguous()?, g.contiguous()?], 1)?;
    j.apply_op1_no_bwd(&SigMul)
}

impl Linear {
    fn run(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let out = match (self.w.dtype(), self.mode) {
            (DType::F16, WeightMode::F16) => x
                .to_dtype(DType::F16)?
                .matmul(&self.w.t()?)?
                .to_dtype(DType::F32)?,
            (DType::F16, WeightMode::Convert) => {
                x.matmul(&self.w.to_dtype(DType::F32)?.t()?)?
            }
            _ => x.matmul(&self.w.t()?)?,
        };
        match &self.b {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}

fn check_matrix(g: &Gguf, key: &str, rows: usize, cols: usize) -> Result<(), String> {
    let t = g
        .tensors
        .get(key)
        .ok_or_else(|| format!("missing tensor {key}"))?;
    if t.dims != [cols, rows] {
        return Err(format!("{key}: unexpected matrix shape {:?}", t.dims));
    }
    Ok(())
}

fn linear(g: &Gguf, key: &str, rows: usize, cols: usize, mode: WeightMode) -> Result<Linear, String> {
    check_matrix(g, &format!("{key}.weight"), rows, cols)?;
    let name = format!("{key}.weight");
    let w = match mode {
        WeightMode::F16 => {
            let bits = g.f16_bits(&name).map_err(|e| e.to_string())?;
            let v: Vec<half::f16> = bits.into_iter().map(half::f16::from_bits).collect();
            Tensor::from_vec(v, (rows, cols), &Device::Cpu).map_err(|e| e.to_string())?
        }
        _ => {
            let v = g.floats(&name).map_err(|e| e.to_string())?;
            Tensor::from_vec(v, (rows, cols), &Device::Cpu).map_err(|e| e.to_string())?
        }
    };
    Ok(Linear {
        w,
        b: None,
        mode,
    })
}

fn vector(g: &Gguf, key: &str, n: usize) -> Result<Vec<f32>, String> {
    let t = g
        .tensors
        .get(key)
        .ok_or_else(|| format!("missing tensor {key}"))?;
    if t.dims.iter().product::<usize>() != n {
        return Err(format!("{key}: unexpected vector shape"));
    }
    g.floats(key).map_err(|e| e.to_string())
}

impl Model {
    pub fn load(g: &Gguf, mode: WeightMode) -> Result<Self, String> {
        if g.meta_str("general.architecture") != "openthai_systemone" {
            return Err("expected an OpenThai-SystemOne GGUF".into());
        }
        let cfg: Value =
            serde_json::from_str(g.meta_str("openthai.config")).map_err(|e| e.to_string())?;
        let c = &cfg["text_config"];
        for (k, n) in [
            ("hidden_size", D),
            ("intermediate_size", INTER),
            ("num_hidden_layers", LAYERS),
            ("head_dim", FULL_HD),
            ("num_attention_heads", FULL_HEADS),
            ("num_key_value_heads", KV_HEADS),
            ("linear_conv_kernel_dim", CONV_K),
            ("linear_key_head_dim", LIN_D),
            ("linear_value_head_dim", LIN_D),
            ("linear_num_key_heads", LIN_HEADS),
            ("linear_num_value_heads", LIN_HEADS),
            ("vocab_size", VOCAB),
        ] {
            if c[k].as_f64() != Some(n as f64) {
                return Err(format!("unsupported OpenThai config {k}"));
            }
        }
        if cfg["n_slots"].as_f64() != Some(256.)
            || cfg["abstain_slot"].as_f64() != Some(255.)
            || cfg["answer_token_id"].as_f64() != Some(ANSWER_TOK as f64)
            || c["rms_norm_eps"].as_f64() != Some(1e-6)
            || c["rope_parameters"]["rope_theta"].as_f64() != Some(10000000.)
            || c["rope_parameters"]["partial_rotary_factor"].as_f64() != Some(0.25)
            || c["rope_parameters"]["rope_type"].as_str() != Some("default")
        {
            return Err("unsupported OpenThai head/normalization/rotary configuration".into());
        }
        let types: Vec<&str> = c["layer_types"]
            .as_array()
            .ok_or("layer_types missing")?
            .iter()
            .map(|v| v.as_str().unwrap_or(""))
            .collect();
        if types.len() != LAYERS {
            return Err("unexpected layer count".into());
        }
        let max_tokens = g.meta_num("openthai.context_length", 4096);
        let max_state = g.meta_num("openthai.state_length", 2048);
        let mut blocks = Vec::new();
        let mut full_layers = Vec::new();
        for (i, ty) in types.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let lin = |tail: &str, r, c| linear(g, &format!("{p}.{tail}"), r, c, mode);
            let mix = match *ty {
                "full_attention" => {
                    full_layers.push(i);
                    Mixer::Full(Full {
                        q: lin("self_attn.q_proj", FULL_HEADS * 2 * FULL_HD, D)?,
                        k: lin("self_attn.k_proj", KV_HEADS * FULL_HD, D)?,
                        v: lin("self_attn.v_proj", KV_HEADS * FULL_HD, D)?,
                        o: lin("self_attn.o_proj", D, FULL_HEADS * FULL_HD)?,
                        qn: vector(g, &format!("{p}.self_attn.q_norm.weight"), FULL_HD)?,
                        kn: vector(g, &format!("{p}.self_attn.k_norm.weight"), FULL_HD)?,
                    })
                }
                "linear_attention" => Mixer::Delta(Delta {
                    qkv: lin("linear_attn.in_proj_qkv", QKV_W, D)?,
                    z: lin("linear_attn.in_proj_z", LIN_HEADS * LIN_D, D)?,
                    a: lin("linear_attn.in_proj_a", LIN_HEADS, D)?,
                    b: lin("linear_attn.in_proj_b", LIN_HEADS, D)?,
                    o: lin("linear_attn.out_proj", D, LIN_HEADS * LIN_D)?,
                    conv: vector(g, &format!("{p}.linear_attn.conv1d.weight"), QKV_W * CONV_K)?,
                    dt: vector(g, &format!("{p}.linear_attn.dt_bias"), LIN_HEADS)?,
                    alog: vector(g, &format!("{p}.linear_attn.A_log"), LIN_HEADS)?,
                    norm: vector(g, &format!("{p}.linear_attn.norm.weight"), LIN_D)?,
                }),
                _ => return Err("unsupported OpenThai layer type".into()),
            };
            blocks.push(Block {
                n1: vector(g, &format!("{p}.input_layernorm.weight"), D)?,
                n2: vector(g, &format!("{p}.post_attention_layernorm.weight"), D)?,
                gate: lin("mlp.gate_proj", INTER, D)?,
                up: lin("mlp.up_proj", INTER, D)?,
                down: lin("mlp.down_proj", D, INTER)?,
                mix,
            });
        }
        check_matrix(g, "model.embed_tokens.weight", VOCAB, D)?;
        let emb = match g.tensors.get("model.embed_tokens.weight").unwrap().kind {
            1 if mode == WeightMode::F16 => {
                let bits = g.f16_bits("model.embed_tokens.weight").map_err(|e| e.to_string())?;
                let v: Vec<half::f16> = bits.into_iter().map(half::f16::from_bits).collect();
                Tensor::from_vec(v, (VOCAB, D), &Device::Cpu).map_err(|e| e.to_string())?
            }
            k if k == 0 || k == 1 => {
                let v = g
                    .floats("model.embed_tokens.weight")
                    .map_err(|e| e.to_string())?;
                Tensor::from_vec(v, (VOCAB, D), &Device::Cpu).map_err(|e| e.to_string())?
            }
            k => return Err(format!("embedding kind {k} unsupported")),
        };
        let mut head = linear(g, "slot_head", 256, D, mode)?;
        head.b = Some(
            Tensor::from_vec(
                vector(g, "slot_head.bias", 256)?,
                256,
                &Device::Cpu,
            )
            .map_err(|e| e.to_string())?,
        );
        let temperature = vector(g, "log_temperature", 3)?
            .iter()
            .map(|v| libm::expf(*v))
            .collect::<Vec<_>>();
        let mut rope_cos = vec![0f32; max_tokens * ROPE_HALF];
        let mut rope_sin = vec![0f32; max_tokens * ROPE_HALF];
        for i in 0..max_tokens {
            for j in 0..ROPE_HALF {
                let angle = i as f32 / libm::powf(10000000., (2 * j) as f32 / 64.);
                rope_cos[i * ROPE_HALF + j] = libm::cosf(angle);
                rope_sin[i * ROPE_HALF + j] = libm::sinf(angle);
            }
        }
        Ok(Self {
            layers: LAYERS,
            full_layers,
            emb,
            blocks,
            norm: vector(g, "model.norm.weight", D)?,
            head,
            temperature: [temperature[0], temperature[1], temperature[2]],
            max_tokens,
            max_state,
            rope_cos: Arc::new(rope_cos),
            rope_sin: Arc::new(rope_sin),
        })
    }

    // Full (GQA) mixer: partial RoPE (32 pairs in the first 64 dims of each
    // 256-wide head), q/k RMS-normed per head, causal softmax over j <= i,
    // q head h reads kv head h/4, output gated by sigmoid of the q_proj gate
    // half.
    fn full_attn(
        &self,
        f: &Full,
        h: &Tensor,
        pos: &[usize],
        segs: &[(usize, usize)],
    ) -> Result<Tensor, candle_core::Error> {
        let s = h.dims()[0];
        let qgate = f.q.run(h)?;
        let qg = qgate.reshape((s, FULL_HEADS, 2, FULL_HD))?;
        let q = qg
            .narrow(2, 0, 1)?
            .contiguous()?
            .reshape((s, FULL_HEADS * FULL_HD))?;
        let gate = qg
            .narrow(2, 1, 1)?
            .contiguous()?
            .reshape((s, FULL_HEADS * FULL_HD))?;
        let q = rms(&q, &f.qn, 1.)?;
        let k = rms(&f.k.run(h)?, &f.kn, 1.)?;
        let v = f.v.run(h)?;
        let rope = |a: &Tensor| -> Result<Tensor, candle_core::Error> {
            a.contiguous()?.apply_op1_no_bwd(&RopeRotate {
                cos: self.rope_cos.clone(),
                sin: self.rope_sin.clone(),
                pos: pos.to_vec(),
            })
        };
        let q = rope(&q)?;
        let k = rope(&k)?;
        let mut seg_outs = Vec::with_capacity(segs.len());
        for &(st, n) in segs {
            let qs = q
                .narrow(0, st, n)?
                .reshape((n, FULL_HEADS, FULL_HD))?
                .transpose(0, 1)?
                .contiguous()?;
            let ks = k
                .narrow(0, st, n)?
                .reshape((n, KV_HEADS, FULL_HD))?
                .transpose(0, 1)?
                .contiguous()?;
            let vs = v
                .narrow(0, st, n)?
                .reshape((n, KV_HEADS, FULL_HD))?
                .transpose(0, 1)?
                .contiguous()?;
            let mut m = vec![f32::NEG_INFINITY; n * n];
            for i in 0..n {
                for j in 0..=i {
                    m[i * n + j] = 0.;
                }
            }
            let mask = Tensor::from_vec(m, (1, n, n), &Device::Cpu)?;
            let mut head_outs = Vec::with_capacity(KV_HEADS);
            for kh in 0..KV_HEADS {
                let qg = qs.narrow(0, kh * GQA, GQA)?;
                // candle matmul needs equal-rank operands.
                let kt = ks.narrow(0, kh, 1)?.squeeze(0)?.t()?.contiguous()?;
                let ktb = kt.unsqueeze(0)?.broadcast_as((GQA, FULL_HD, n))?;
                let sc = qg
                    .matmul(&ktb)?
                    .affine(1.0 / 16.0, 0.0)?
                    .broadcast_add(&mask)?;
                let p = sc
                    .contiguous()?
                    .apply_op1_no_bwd(&SoftmaxLast)?;
                let vv = vs.narrow(0, kh, 1)?.squeeze(0)?.contiguous()?;
                let vvb = vv.unsqueeze(0)?.broadcast_as((GQA, n, FULL_HD))?;
                head_outs.push(p.matmul(&vvb)?);
            }
            let o = Tensor::cat(&head_outs, 0)?;
            seg_outs.push(
                o.transpose(0, 1)?
                    .contiguous()?
                    .reshape((n, FULL_HEADS * FULL_HD))?,
            );
        }
        let out = Tensor::cat(&seg_outs, 0)?.contiguous()?;
        let out = sig_mul(&out, &gate)?;
        f.o.run(&out)
    }

    fn delta(
        &self,
        d: &Delta,
        h: &Tensor,
        lo: &[usize],
        segs: &[(usize, usize)],
        impl_: GdnImpl,
    ) -> Result<Tensor, candle_core::Error> {
        let s = h.dims()[0];
        let raw = d.qkv.run(h)?;
        let z = d.z.run(h)?;
        let a = d.a.run(h)?;
        let b = d.b.run(h)?;
        let qkv = raw
            .contiguous()?
            .apply_op1_no_bwd(&ConvSilu {
                conv: d.conv.clone(),
                lo: lo.to_vec(),
            })?;
        let out = match impl_ {
            GdnImpl::Scalar | GdnImpl::Tensor => {
                let qkv_v = qkv.flatten_all()?.to_vec1::<f32>()?;
                let a_v = a.flatten_all()?.to_vec1::<f32>()?;
                let b_v = b.flatten_all()?.to_vec1::<f32>()?;
                if impl_ == GdnImpl::Tensor {
                    gdn_tensor_loop(&qkv, &a_v, &b_v, &d.alog, &d.dt, segs)?
                } else {
                    // One job per (segment, head); heads own disjoint 128-wide
                    // output ranges, mirroring the engine's per-head
                    // parallelism. Jobs return their slice; the scatter below
                    // is sequential.
                    let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
                    for &(st, n) in segs {
                        for head in 0..LIN_HEADS {
                            pairs.push((st, n, head));
                        }
                    }
                    let parts: Vec<Vec<f32>> = pairs
                        .par_iter()
                        .map(|&(st, n, head)| {
                            let mut o = vec![0f32; n * LIN_D];
                            delta_head(head, st, n, &qkv_v, &a_v, &b_v, &d.alog, &d.dt, &mut o);
                            o
                        })
                        .collect();
                    let mut out = vec![0f32; s * LIN_HEADS * LIN_D];
                    for (part, &(st, n, head)) in parts.iter().zip(&pairs) {
                        for t in 0..n {
                            let dst = (st + t) * LIN_HEADS * LIN_D + head * LIN_D;
                            out[dst..dst + LIN_D].copy_from_slice(&part[t * LIN_D..(t + 1) * LIN_D]);
                        }
                    }
                    Tensor::from_vec(out, (s, LIN_HEADS * LIN_D), &Device::Cpu)?
                }
            }
        };
        let out = rms(&out, &d.norm, 0.)?;
        let out = silu_mul(&out, &z)?;
        d.o.run(&out)
    }

    pub fn forward(
        &self,
        ids: &[u32],
        positions: &[usize],
    ) -> Result<Vec<f32>, String> {
        let reqs = [Req {
            ids: ids.to_vec(),
            positions: positions.to_vec(),
        }];
        self.forward_batch(&reqs, GdnImpl::Scalar)
            .map_err(|e| e.to_string())
            .map(|mut v| v.remove(0))
    }

    pub fn forward_batch(
        &self,
        reqs: &[Req],
        impl_: GdnImpl,
    ) -> Result<Vec<Vec<f32>>, candle_core::Error> {
        let s: usize = reqs.iter().map(|r| r.ids.len()).sum();
        if reqs
            .iter()
            .any(|r| r.ids.is_empty() || r.ids.len() > self.max_tokens)
        {
            return Err(candle_core::Error::Msg("invalid sequence length".into()));
        }
        let mut seg_start = Vec::with_capacity(reqs.len());
        let mut acc = 0usize;
        for r in reqs {
            seg_start.push(acc);
            acc += r.ids.len();
        }
        let segs: Vec<(usize, usize)> = reqs
            .iter()
            .zip(&seg_start)
            .map(|(r, &st)| (st, r.ids.len()))
            .collect();
        let mut ids_all = Vec::with_capacity(s);
        let mut pos = Vec::with_capacity(s);
        let mut lo = Vec::with_capacity(s);
        for (ri, r) in reqs.iter().enumerate() {
            ids_all.extend_from_slice(&r.ids);
            for p in 0..r.ids.len() {
                pos.push(p);
                lo.push(segs[ri].0);
            }
        }
        let ids_t = Tensor::from_vec(ids_all, s, &Device::Cpu)?;
        let mut x = self.emb.index_select(&ids_t, 0)?.to_dtype(DType::F32)?;
        for b in &self.blocks {
            let h = rms(&x, &b.n1, 1.)?;
            let y = match &b.mix {
                Mixer::Full(f) => self.full_attn(f, &h, &pos, &segs)?,
                Mixer::Delta(d) => self.delta(d, &h, &lo, &segs, impl_)?,
            };
            x = (&x + &y)?;
            let h = rms(&x, &b.n2, 1.)?;
            let g = b.gate.run(&h)?;
            let u = b.up.run(&h)?;
            // engine: silu(gate) * up — silu applies to the gate output.
            let m = silu_mul(&u, &g)?;
            x = (&x + &b.down.run(&m)?)?;
        }
        let mut outs = Vec::with_capacity(reqs.len());
        for (ri, r) in reqs.iter().enumerate() {
            if r.positions.iter().any(|&p| p >= r.ids.len()) {
                return Err(candle_core::Error::Msg("invalid readout position".into()));
            }
            let rows: Vec<u32> = r.positions.iter().map(|&p| (seg_start[ri] + p) as u32).collect();
            let rt = Tensor::from_vec(rows, r.positions.len(), &Device::Cpu)?;
            let a = x.index_select(&rt, 0)?;
            let an = rms(&a, &self.norm, 1.)?;
            outs.push(
                self.head
                    .run(&an)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
        }
        Ok(outs)
    }
}

// Bit-mirrors the engine's delta_head: same op order, same libm calls, so
// given identical qkv/a/b inputs the output is bit-identical to the engine.
// Writes the local [n, 128] slice for tokens st..st+n of head h.
fn delta_head(
    h: usize,
    st: usize,
    n: usize,
    qkv: &[f32],
    a: &[f32],
    b: &[f32],
    alog: &[f32],
    dt: &[f32],
    out: &mut [f32],
) {
    let mut state = vec![0f32; LIN_D * LIN_D];
    for i in st..st + n {
        let q = &qkv[i * QKV_W + h * LIN_D..][..LIN_D];
        let k = &qkv[i * QKV_W + 2048 + h * LIN_D..][..LIN_D];
        let v = &qkv[i * QKV_W + 4096 + h * LIN_D..][..LIN_D];
        let qs = 1. / libm::sqrtf(self_dot(q) + 1e-6) / libm::sqrtf(128.);
        let ks = 1. / libm::sqrtf(self_dot(k) + 1e-6);
        let decay =
            libm::expf(-libm::expf(alog[h]) * softplus(a[i * LIN_HEADS + h] + dt[h]));
        let beta = sigmoid(b[i * LIN_HEADS + h]);
        let mut delta = [0f32; LIN_D];
        for v in state.iter_mut() {
            *v *= decay;
        }
        for j in 0..LIN_D {
            let key = k[j] * ks;
            for d in 0..LIN_D {
                delta[d] += state[j * LIN_D + d] * key;
            }
        }
        for d in 0..LIN_D {
            delta[d] = (v[d] - delta[d]) * beta;
        }
        let o = &mut out[(i - st) * LIN_D..][..LIN_D];
        for j in 0..LIN_D {
            let key = k[j] * ks;
            let query = q[j] * qs;
            for d in 0..LIN_D {
                state[j * LIN_D + d] += key * delta[d];
                o[d] += state[j * LIN_D + d] * query;
            }
        }
    }
}

// Per-position candle-tensor GDN loop (single segment): all 16 heads batched
// into each op, state kept as a [16,128,128] Tensor. Semantically identical
// to delta_head; kept only to measure what a "pure tensor ops" GDN costs in
// candle (op dispatch + broadcast materialization traffic). Do not use.
fn gdn_tensor_loop(
    qkv: &Tensor,
    a: &[f32],
    b: &[f32],
    alog: &[f32],
    dt: &[f32],
    segs: &[(usize, usize)],
) -> Result<Tensor, candle_core::Error> {
    if segs.len() != 1 {
        return Err(candle_core::Error::Msg(
            "tensor GDN variant supports a single segment".into(),
        ));
    }
    let s = qkv.dims()[0];
    let dev = qkv.device();
    let mut state = Tensor::zeros((LIN_HEADS, LIN_D, LIN_D), DType::F32, dev)?;
    let mut outs = Vec::with_capacity(s);
    for i in 0..s {
        let row = qkv.narrow(0, i, 1)?;
        let q = row
            .narrow(1, 0, LIN_HEADS * LIN_D)?
            .reshape((LIN_HEADS, LIN_D))?;
        let k = row
            .narrow(1, LIN_HEADS * LIN_D, LIN_HEADS * LIN_D)?
            .reshape((LIN_HEADS, LIN_D))?;
        let v = row
            .narrow(1, 2 * LIN_HEADS * LIN_D, LIN_HEADS * LIN_D)?
            .reshape((LIN_HEADS, LIN_D))?;
        // Per-head gates in scalar Rust (libm, engine formulas); the heavy
        // state math below stays in tensor ops.
        let mut decay = vec![0f32; LIN_HEADS];
        let mut beta = vec![0f32; LIN_HEADS];
        for h in 0..LIN_HEADS {
            decay[h] =
                libm::expf(-libm::expf(alog[h]) * softplus(a[i * LIN_HEADS + h] + dt[h]));
            beta[h] = sigmoid(b[i * LIN_HEADS + h]);
        }
        let qs = dot_rows(&q)?.affine((1. / libm::sqrtf(128.)) as f64, 0.)?;
        let ks = dot_rows(&k)?;
        let scale_rows = |x: &Tensor, w: &Tensor| -> Result<Tensor, candle_core::Error> {
            x.broadcast_mul(&w.reshape((LIN_HEADS, 1))?.broadcast_as(x.shape())?)
        };
        let kw = scale_rows(&k, &ks)?;
        let qw = scale_rows(&q, &qs)?;
        let decay_t = Tensor::from_vec(decay, LIN_HEADS, dev)?
            .reshape((LIN_HEADS, 1, 1))?
            .broadcast_as(state.shape())?;
        state = (&state * &decay_t)?;
        let kb = kw
            .reshape((LIN_HEADS, LIN_D, 1))?
            .broadcast_as(state.shape())?;
        let mut delta = state.mul(&kb)?.sum(1)?;
        delta = (&v - &delta)?.broadcast_mul(&Tensor::from_vec(beta, LIN_HEADS, dev)?.reshape((LIN_HEADS, 1))?)?;
        let db = delta
            .reshape((LIN_HEADS, 1, LIN_D))?
            .broadcast_as(state.shape())?;
        state = (&state + &kb.mul(&db)?)?;
        let qb = qw
            .reshape((LIN_HEADS, LIN_D, 1))?
            .broadcast_as(state.shape())?;
        outs.push(state.mul(&qb)?.sum(1)?);
    }
    let o = Tensor::cat(&outs, 0)?;
    o.reshape((s, LIN_HEADS * LIN_D))
}

// Per-row inv-sqrt of (dot + 1e-6) for the tensor GDN loop.
fn dot_rows(x: &Tensor) -> Result<Tensor, candle_core::Error> {
    x.mul(x)?
        .sum(1)?
        .affine(1.0, 1e-6)?
        .sqrt()?
        .recip()
}
