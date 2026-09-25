// Laya model: candle port of src/models/laya/model.rs (op order, eps, GeGLU,
// final norm + head readout mirrored 1:1; weight residency selectable).
use crate::gguf::Gguf;
use candle_core::cpu_backend::CpuStorage;
use candle_core::{CustomOp1, DType, Device, Layout, Shape, Tensor, D};
use rayon::prelude::*;
use serde_json::Value;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightMode {
    // F16-resident weights; activations cast to F16 per matmul (gemm f16,
    // f32 accumulators). RAM parity with the engine's F16 residency.
    F16,
    // F16-resident weights, dequantized to F32 on every matmul (streams like
    // the engine's in-kernel conversion, at extra conversion traffic).
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

pub struct Linear {
    w: Tensor, // [out, in], row-major GGUF layout
    b: Option<Tensor>,
}

pub struct Norm {
    w: Option<Tensor>,
    b: Option<Tensor>,
    eps: f32,
}

pub struct Layer {
    pub n1: Option<Norm>,
    pub qkv: Linear,
    pub wo: Linear,
    pub n2: Norm,
    pub wi: Linear,
    pub down: Linear,
    pub full: bool,
}

pub struct BatchReq {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: usize,
}

pub struct Model {
    pub layers: usize,
    pub d: usize,
    pub hd: usize,
    pub heads: usize,
    emb: Tensor, // [vocab, d]
    embnorm: Norm,
    finalnorm: Norm,
    pub trunk: Vec<Layer>,
    pub head: Vec<Layer>,
    type_emb: Tensor, // [3, d]
    scorenorm: Norm,
    score1: Linear,
    score2: Linear,
    act1: Linear,
    act2: Linear,
    cos_full: Tensor, // [513, hd]
    sin_full: Tensor,
    cos_slide: Tensor,
    sin_slide: Tensor,
}

fn tensor_name(graph: &Value, id: &Value) -> String {
    graph["tensors"][id.to_string()]["name"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

fn linear(g: &Gguf, graph: &Value, key: &str, mode: WeightMode) -> Result<Linear, String> {
    let weight = format!("{key}.weight");
    let t = g
        .tensors
        .get(&weight)
        .ok_or_else(|| format!("missing {weight}"))?;
    let cols = t.dims[0];
    let rows = *t.dims.get(1).unwrap_or(&1);
    if t.dims.iter().skip(2).any(|&d| d != 1) {
        return Err("non-matrix linear".into());
    }
    let node = graph["nodes"]
        .as_array()
        .ok_or("graph nodes missing")?
        .iter()
        .find(|n| {
            n["opcode"].as_f64() == Some(29.0)
                && n["inputs"]
                    .as_array()
                    .and_then(|a| a.first())
                    .map(|i| tensor_name(graph, i) == weight)
                    .unwrap_or(false)
        })
        .ok_or_else(|| format!("graph lacks linear {key}"))?;
    let mut biases = Vec::new();
    if node["inputs"].as_array().map(|a| a.len()).unwrap_or(0) > 2 {
        biases.push(tensor_name(graph, &node["inputs"].as_array().unwrap()[2]));
    }
    let out_id = node["outputs"].as_array().and_then(|a| a.first()).cloned();
    if let Some(out_id) = out_id {
        for n in graph["nodes"].as_array().unwrap() {
            if n["opcode"].as_f64() == Some(200.0)
                && n["inputs"].as_array().and_then(|a| a.first()) == Some(&out_id)
            {
                biases.push(tensor_name(graph, &n["inputs"].as_array().unwrap()[1]));
            }
        }
    }
    let b = if biases.is_empty() {
        None
    } else {
        let mut out = vec![0f32; rows];
        for name in biases {
            let v = g.floats(&name).map_err(|e| e.to_string())?;
            if v.len() != rows {
                return Err("linear bias shape mismatch".into());
            }
            for (a, b) in out.iter_mut().zip(&v) {
                *a += b;
            }
        }
        Some(Tensor::from_vec(out, rows, &Device::Cpu).map_err(|e| e.to_string())?)
    };
    let v = g.floats(&weight).map_err(|e| e.to_string())?;
    if v.len() != rows * cols {
        return Err("linear weight element mismatch".into());
    }
    let wt = Tensor::from_vec(v, (rows, cols), &Device::Cpu).map_err(|e| e.to_string())?;
    let w = match mode {
        WeightMode::F16 | WeightMode::Convert => {
            wt.to_dtype(DType::F16).map_err(|e| e.to_string())?
        }
        WeightMode::Upcast => wt,
    };
    Ok(Linear { w, b })
}

fn norm(g: &Gguf, graph: &Value, i: usize) -> Result<Norm, String> {
    let key = if i == 0 {
        "layer_norm".to_string()
    } else {
        format!("layer_norm_{i}")
    };
    let node = graph["nodes"]
        .as_array()
        .ok_or("graph nodes missing")?
        .iter()
        .find(|n| n["name"].as_str() == Some(key.as_str()) && n["opcode"].as_f64() == Some(201.0))
        .ok_or_else(|| format!("missing {key}"))?;
    let ins = node["inputs"].as_array().unwrap();
    let get = |idx: usize| -> Result<Option<Tensor>, String> {
        if idx < ins.len() {
            let v = g
                .floats(&tensor_name(graph, &ins[idx]))
                .map_err(|e| e.to_string())?;
            let n = v.len();
            Ok(Some(
                Tensor::from_vec(v, n, &Device::Cpu).map_err(|e| e.to_string())?,
            ))
        } else {
            Ok(None)
        }
    };
    let eps = node["attributes"]["eps"].as_f64().unwrap_or(0.0) as f32;
    if !(eps > 0.0 && eps < 0.1) {
        return Err("invalid layer norm epsilon".into());
    }
    Ok(Norm {
        w: get(1)?,
        b: get(2)?,
        eps,
    })
}

fn norm_run(n: &Norm, x: &Tensor, d: usize) -> Result<Tensor, candle_core::Error> {
    // Custom op: candle's stock CPU elementwise/broadcast path is
    // single-threaded; this mirrors the engine's per-row math (sequential
    // mean/var sums) with the engine's disjoint-row worker parallelism.
    let w = match &n.w {
        Some(t) => t.to_vec1::<f32>()?,
        None => vec![1f32; d],
    };
    let b = match &n.b {
        Some(t) => t.to_vec1::<f32>()?,
        None => vec![0f32; d],
    };
    x.contiguous()?
        .apply_op1_no_bwd(&RowNorm { w, b, eps: n.eps })
}

struct RowNorm {
    w: Vec<f32>,
    b: Vec<f32>,
    eps: f32,
}

impl CustomOp1 for RowNorm {
    fn name(&self) -> &'static str {
        "laya-row-norm"
    }
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        let vs = storage.as_slice::<f32>()?;
        let (start, len) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::RequiresContiguous { op: self.name() })?;
        let x = &vs[start..start + len];
        let d = self.w.len();

        let mut out = vec![0f32; len];
        let (w, b, eps) = (&self.w, &self.b, self.eps);
        out.par_chunks_exact_mut(d)
            .zip(x.par_chunks_exact(d))
            .for_each(|(o, a)| {
                let mean = a.iter().sum::<f32>() / d as f32;
                let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
                let scale = 1.0 / (var + eps).sqrt();
                for j in 0..d {
                    o[j] = (a[j] - mean) * scale * w[j] + b[j];
                }
            });
        let shape = layout.shape().to_owned();
        Ok((CpuStorage::F32(out), shape))
    }
}

struct GeGlu;

impl CustomOp1 for GeGlu {
    fn name(&self) -> &'static str {
        "laya-geglu"
    }
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        let vs = storage.as_slice::<f32>()?;
        let (start, len) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::RequiresContiguous { op: self.name() })?;
        let up = &vs[start..start + len];
        let f = layout.dims()[1] / 2;
        let rows = up.len() / (2 * f);
        let mut out = vec![0f32; rows * f];
        out.par_chunks_exact_mut(f).enumerate().for_each(|(i, o)| {
            let g = &up[i * 2 * f..i * 2 * f + f];
            let u = &up[i * 2 * f + f..(i + 1) * 2 * f];
            for j in 0..f {
                let x = g[j];
                let t = 0.7978845608028654f32 * x * (1.0 + 0.044715 * x * x);
                o[j] = 0.5 * x * (1.0 + t.tanh()) * u[j];
            }
        });
        let dims: Vec<usize> = layout
            .dims()
            .iter()
            .enumerate()
            .map(|(i, &dd)| if i == 1 { f } else { dd })
            .collect();
        Ok((CpuStorage::F32(out), Shape::from_dims(&dims)))
    }
}

struct SoftmaxLast;

struct RopeRotate {
    cos: std::sync::Arc<Vec<f32>>, // [s, hd]
    sin: std::sync::Arc<Vec<f32>>,
    s: usize,
}

impl CustomOp1 for RopeRotate {
    fn name(&self) -> &'static str {
        "laya-rope"
    }
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        let vs = storage.as_slice::<f32>()?;
        let (start, len) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::RequiresContiguous { op: self.name() })?;
        let x = &vs[start..start + len];
        let hd = layout.dims().last().copied().unwrap_or(64);
        let mut out = x.to_vec();
        let (cos, sin, s) = (&self.cos, &self.sin, self.s);
        out.par_chunks_exact_mut(hd).enumerate().for_each(|(r, o)| {
            let pos = r % s;
            let half = hd / 2;
            for j in 0..half {
                let u = o[j];
                let w = o[j + half];
                o[j] = u * cos[pos * hd + j] - w * sin[pos * hd + j];
                o[j + half] = w * cos[pos * hd + j + half] + u * sin[pos * hd + j + half];
            }
        });
        let shape = layout.shape().to_owned();
        Ok((CpuStorage::F32(out), shape))
    }
}

impl CustomOp1 for SoftmaxLast {
    fn name(&self) -> &'static str {
        "laya-softmax"
    }
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
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
                let m = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for (y, &v) in o.iter_mut().zip(a) {
                    *y = (v - m).exp();
                    sum += *y;
                }
                for y in o.iter_mut() {
                    *y /= sum;
                }
            });
        let shape = layout.shape().to_owned();
        Ok((CpuStorage::F32(out), shape))
    }
}

impl Linear {
    fn run(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let out = match self.w.dtype() {
            DType::F16 => x
                .to_dtype(DType::F16)?
                .matmul(&self.w.t()?)?
                .to_dtype(DType::F32)?,
            _ => x.matmul(&self.w.t()?)?,
        };
        match &self.b {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}

// Engine tensor::gelu: 0.5*x*(1 + tanh(0.7978845608028654*x*(1+0.044715*x*x)))
fn gelu_tanh(x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let xsq = x.mul(x)?;
    let t = xsq.affine(0.044715, 1.0)?;
    let z = x.mul(&t)?.affine(0.7978845608028654, 0.0)?;
    let th = z.tanh()?;
    x.affine(0.5, 0.0)?.mul(&((&th + 1.0)?))
}

// Engine tensor::softmax, exact: max-sub, exp, sequential sum, divide.
pub fn softmax_f32(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let sum = p.iter().sum::<f32>();
    for v in &mut p {
        *v /= sum
    }
    p
}

fn softmax_last(t: &Tensor) -> Result<Tensor, candle_core::Error> {
    t.contiguous()?.apply_op1_no_bwd(&SoftmaxLast)
}

impl Model {
    pub fn load(g: &Gguf, mode: WeightMode) -> Result<Self, String> {
        if g.meta_str("general.architecture") != "ggmlc"
            || !g.meta_str("laya.checkpoint").contains("laya")
        {
            return Err("expected a compiled Laya GGUF".into());
        }
        let graph: Value =
            serde_json::from_str(g.meta_str("ggmlc.graph_spec")).map_err(|e| e.to_string())?;
        let mut layers = 0;
        while g.tensors.contains_key(&format!("qkv.{layers}.weight")) {
            layers += 1
        }
        let mut heads = 0;
        while g.tensors.contains_key(&format!("head_qkv.{heads}.weight")) {
            heads += 1
        }
        if layers == 0 || layers > 64 || heads != 2 {
            return Err("unsupported Laya encoder/head layout".into());
        }
        let et = g.tensors.get("tok_emb.weight").ok_or("missing embedding")?;
        let d = et.dims[0];
        let vocab = et.dims[1];
        let hd = g
            .tensors
            .get("cos_full")
            .ok_or("missing RoPE tables")?
            .dims[0];
        if d == 0 || hd == 0 || hd % 2 != 0 || d % hd != 0 {
            return Err("invalid attention dimensions".into());
        }
        let attns: Vec<&Value> = graph["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|n| n["opcode"].as_f64() == Some(74.0))
            .collect();
        if attns.len() != layers + heads {
            return Err("unexpected attention graph".into());
        }
        let fullmask = tensor_name(
            &graph,
            attns[0]["inputs"].as_array().and_then(|a| a.last()).unwrap(),
        );
        let mut trunk = Vec::new();
        for i in 0..layers {
            trunk.push(Layer {
                n1: if i == 0 {
                    None
                } else {
                    Some(norm(g, &graph, 2 * i)?)
                },
                qkv: linear(g, &graph, &format!("qkv.{i}"), mode)?,
                wo: linear(g, &graph, &format!("wo.{i}"), mode)?,
                n2: norm(g, &graph, 2 * i + 1)?,
                wi: linear(g, &graph, &format!("mlp_wi.{i}"), mode)?,
                down: linear(g, &graph, &format!("mlp_wo.{i}"), mode)?,
                full: tensor_name(
                    &graph,
                    attns[i]["inputs"].as_array().and_then(|a| a.last()).unwrap(),
                ) == fullmask,
            });
        }
        let mut head = Vec::new();
        for i in 0..heads {
            head.push(Layer {
                n1: Some(norm(g, &graph, 2 * layers + 1 + 2 * i)?),
                qkv: linear(g, &graph, &format!("head_qkv.{i}"), mode)?,
                wo: linear(g, &graph, &format!("head_wo.{i}"), mode)?,
                n2: norm(g, &graph, 2 * layers + 2 + 2 * i)?,
                wi: linear(g, &graph, &format!("head_fc1.{i}"), mode)?,
                down: linear(g, &graph, &format!("head_fc2.{i}"), mode)?,
                full: true,
            });
        }
        let embv = g.floats("tok_emb.weight").map_err(|e| e.to_string())?;
        let emb =
            Tensor::from_vec(embv, (vocab, d), &Device::Cpu).map_err(|e| e.to_string())?;
        let emb = match mode {
            WeightMode::F16 | WeightMode::Convert => {
                emb.to_dtype(DType::F16).map_err(|e| e.to_string())?
            }
            WeightMode::Upcast => emb,
        };
        let te = g.floats("type_emb.weight").map_err(|e| e.to_string())?;
        let type_emb =
            Tensor::from_vec(te, (3, d), &Device::Cpu).map_err(|e| e.to_string())?;
        let rope = |name: &str| -> Result<Tensor, String> {
            let v = g.floats(name).map_err(|e| e.to_string())?;
            let pos = v.len() / hd;
            Tensor::from_vec(v, (pos, hd), &Device::Cpu).map_err(|e| e.to_string())
        };
        Ok(Self {
            layers,
            d,
            hd,
            heads: d / hd,

            emb,
            embnorm: norm(g, &graph, 0)?,
            finalnorm: norm(g, &graph, 2 * layers)?,
            trunk,
            head,
            type_emb,
            scorenorm: norm(g, &graph, 2 * layers + 1 + 2 * heads)?,
            score1: linear(g, &graph, "scorer_fc1", mode)?,
            score2: linear(g, &graph, "scorer_fc2", mode)?,
            act1: linear(g, &graph, "act_fc1", mode)?,
            act2: linear(g, &graph, "act_fc2", mode)?,
            cos_full: rope("cos_full")?,
            sin_full: rope("sin_full")?,
            cos_slide: rope("cos_slide")?,
            sin_slide: rope("sin_slide")?,
        })
    }

    // Single-segment attention over qkv [s, 3d]; mirrors tensor::attention
    // (RoPE table indexing, scale, softmax) with the engine's ±64 window
    // semantics. windowed=false matches the engine's full-attention layers.
    fn attention_one(
        &self,
        qkv: &Tensor,
        s: usize,
        rope: Option<(&Tensor, &Tensor)>,
        windowed: bool,
    ) -> Result<Tensor, candle_core::Error> {
        let d = self.d;
        let hd = self.hd;
        let h = self.heads;
        let window = 64usize;
        let split = |off: usize| -> Result<Tensor, candle_core::Error> {
            qkv
                .narrow(1, off, d)?
                .reshape((s, h, hd))?
                .transpose(0, 1)?
                .contiguous()
        };
        let mut q = split(0)?;
        let mut k = split(d)?;
        let v = split(2 * d)?;
        if let Some((cos, sin)) = rope {
            // Engine pairs (j, j+hd/2) per head with table row `pos`:
            // a[j] = u*cos[j] - v*sin[j]; a[j+hd/2] = v*cos[j+32] + u*sin[j+32]
            let rot = |a: &mut Tensor| -> Result<(), candle_core::Error> {
                let op = RopeRotate {
                    cos: std::sync::Arc::new(cos.flatten_all()?.to_vec1::<f32>()?),
                    sin: std::sync::Arc::new(sin.flatten_all()?.to_vec1::<f32>()?),
                    s,
                };
                *a = a.apply_op1_no_bwd(&op)?;
                Ok(())
            };
            rot(&mut q)?;
            rot(&mut k)?;
        }
        let scale = 1.0 / (self.hd as f64).sqrt();
        let attn = |q_c: &Tensor, k_c: &Tensor, v_c: &Tensor, mask: Option<&Tensor>| -> Result<Tensor, candle_core::Error> {
            let kt = k_c.transpose(D::Minus1, D::Minus2)?;
            let mut scores = q_c.matmul(&kt)?.affine(scale, 0.0)?;
            if let Some(m) = mask {
                scores = scores.broadcast_add(m)?;
            }
            let p = softmax_last(&scores)?;
            p.matmul(v_c)
        };
        let out = if windowed && s > window + 1 {
            // Chunk queries; keys span the union window so softmax sees
            // exactly the engine's per-query [i-64, i+64] key set.
            let mut outs = Vec::new();
            let mut c0 = 0usize;
            while c0 < s {
                let nq = window.min(s - c0);
                let k0 = c0.saturating_sub(window);
                let k1 = (c0 + nq + window).min(s);
                let kl = k1 - k0;
                let mut m = vec![f32::NEG_INFINITY; nq * kl];
                for i in 0..nq {
                    let lo = (c0 + i).saturating_sub(window);
                    let hi = ((c0 + i) + window + 1).min(s);
                    for kj in lo..hi {
                        m[i * kl + (kj - k0)] = 0.0;
                    }
                }
                let mt = Tensor::from_vec(m, (nq, kl), &Device::Cpu)?;
                outs.push(attn(
                    &q.narrow(1, c0, nq)?,
                    &k.narrow(1, k0, kl)?,
                    &v.narrow(1, k0, kl)?,
                    Some(&mt),
                )?);
                c0 += nq;
            }
            Tensor::cat(&outs, 1)?
        } else {
            // s <= 65 windowed (window covers everything) or full attention.
            attn(&q, &k, &v, None)?
        };
        out.transpose(0, 1)?.reshape((s, self.d))?.contiguous()
    }

    // Attention over a stacked batch, run per segment with local RoPE
    // positions — for in-segment pairs identical to running each request
    // alone (the engine's exact per-request math).
    fn attention_segments(
        &self,
        qkv: &Tensor,
        segs: &[(usize, usize)],
        rope_tables: Option<(&Tensor, &Tensor)>,
        windowed: bool,
    ) -> Result<Tensor, candle_core::Error> {
        let mut outs = Vec::with_capacity(segs.len());
        for &(st, n) in segs {
            let seg = qkv.narrow(0, st, n)?;
            let rope = match rope_tables {
                Some((c, s)) => Some((c.narrow(0, 0, n)?, s.narrow(0, 0, n)?)),
                None => None,
            };
            let rope_ref = rope.as_ref().map(|(c, s)| (c, s));
            outs.push(self.attention_one(&seg, n, rope_ref, windowed)?);
        }
        Tensor::cat(&outs, 0)
    }

    pub fn forward(
        &self,
        ids: &[u32],
        markers: &[usize],
        qtype: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let reqs = [BatchReq {
            ids: ids.to_vec(),
            markers: markers.to_vec(),
            qtype,
        }];
        self.forward_batch(&reqs)
            .map_err(|e| e.to_string())
            .map(|mut v| v.remove(0))
    }

    // Batched forward: requests stacked along dim 0 with block-diagonal
    // attention masks and per-request RoPE positions. For in-segment query/key
    // pairs the math is identical to running each request alone.
    pub fn forward_batch(
        &self,
        reqs: &[BatchReq],
    ) -> Result<Vec<(Vec<f32>, Vec<f32>)>, candle_core::Error> {
        let d = self.d;
        let s: usize = reqs.iter().map(|r| r.ids.len()).sum();
        let mut seg_start = Vec::with_capacity(reqs.len());
        let mut acc = 0;
        for r in reqs {
            seg_start.push(acc);
            acc += r.ids.len();
        }
        let mut ids_all = Vec::with_capacity(s);
        let mut qtypes = Vec::with_capacity(s);
        for r in reqs {
            ids_all.extend_from_slice(&r.ids);
            qtypes.extend(std::iter::repeat(r.qtype as u32).take(r.ids.len()));
        }
        let ids_t = Tensor::from_vec(ids_all, s, &Device::Cpu)?;
        let qtypes_t = Tensor::from_vec(qtypes, s, &Device::Cpu)?;

        let segs: Vec<(usize, usize)> = reqs
            .iter()
            .zip(&seg_start)
            .map(|(r, &st)| (st, r.ids.len()))
            .collect();

        let emb = self.emb.index_select(&ids_t, 0)?;
        let mut h = norm_run(&self.embnorm, &emb.to_dtype(DType::F32)?, d)?;
        for l in &self.trunk {
            let x = match &l.n1 {
                Some(n) => norm_run(n, &h, d)?,
                None => h.clone(),
            };
            let qkv = l.qkv.run(&x)?;
            let rope = if l.full {
                Some((&self.cos_full, &self.sin_full))
            } else {
                Some((&self.cos_slide, &self.sin_slide))
            };
            let a = self.attention_segments(&qkv, &segs, rope, !l.full)?;
            h = (&h + &l.wo.run(&a)?)?;
            let n = norm_run(&l.n2, &h, d)?;
            let up = l.wi.run(&n)?;
            let m = up.apply_op1_no_bwd(&GeGlu)?;
            h = (&h + &l.down.run(&m)?)?;
        }
        h = norm_run(&self.finalnorm, &h, d)?;
        let type_rows = self.type_emb.index_select(&qtypes_t, 0)?;
        h = (&h + &type_rows)?;
        for l in &self.head {
            let x = norm_run(l.n1.as_ref().unwrap(), &h, d)?;
            let qkv = l.qkv.run(&x)?;
            let a = self.attention_segments(&qkv, &segs, None, false)?;
            h = (&h + &l.wo.run(&a)?)?;
            let x = norm_run(&l.n2, &h, d)?;
            let f = l.wi.run(&x)?.clamp(0.0, f64::INFINITY)?;
            h = (&h + &l.down.run(&f)?)?;
        }

        let mut out = Vec::with_capacity(reqs.len());
        for (r, st) in reqs.iter().zip(&seg_start) {
            let marker_ids: Vec<u32> = r.markers.iter().map(|&m| (st + m) as u32).collect();
            let mt = Tensor::from_vec(marker_ids, r.markers.len(), &Device::Cpu)?;
            let gathered = h.index_select(&mt, 0)?;
            let x = norm_run(&self.scorenorm, &gathered, d)?;
            let x = gelu_tanh(&self.score1.run(&x)?)?;
            let logits: Vec<f32> = self
                .score2
                .run(&x)?
                .to_vec2::<f32>()?
                .into_iter()
                .flatten()
                .collect();

            // Act head: pooled = first row of this request + engine features.
            let p0 = softmax_f32(&logits);
            let top1 = p0.iter().copied().fold(0f32, f32::max);
            let top2 = p0
                .iter()
                .map(|&p| p * (top1 - p) / ((top1 - p) + 1e-6))
                .fold(0f32, f32::max);
            let k = r.markers.len().max(2) as f32;
            let ent = -p0.iter().map(|&p| p * p.max(1e-9).ln()).sum::<f32>() / k.ln();
            let mut pooled = h.narrow(0, *st, 1)?.to_vec2::<f32>()?.remove(0);
            pooled.extend_from_slice(&[top1, top1 - top2, ent, k / 255.0]);
            let pt = Tensor::from_vec(pooled, (1, d + 4), &Device::Cpu)?;
            let x = gelu_tanh(&self.act1.run(&pt)?)?;
            let act = self.act2.run(&x)?.to_vec2::<f32>()?.remove(0);
            out.push((logits, act));
        }
        Ok(out)
    }
}
