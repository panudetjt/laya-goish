//! Scalar CPU oracle for debugging the GPU kernels: a direct transcription
//! of the CPU engine's forward (src/models/laya/model.rs + src/tensor.rs)
//! with plain sequential loops. Deliberately a SEPARATE implementation from
//! laya.rs — the oracle must not share the loader/forward it validates. Validated by reproducing the committed
//! reference logits for the refund case; used by `stage-check` to localize
//! divergence between the GPU pipeline and the CPU semantics stage by stage.
use crate::gguf::{half, graph_tensor_name, jget, GgufFile, OP_ATTENTION, OP_FUSED_BIAS, OP_LAYER_NORM, OP_LINEAR};
use serde_json::Value as J;
use std::collections::HashMap;

struct OLinear {
    v: Vec<f32>,
    rows: usize,
    cols: usize,
    bias: Option<Vec<f32>>,
}

struct ONorm {
    w: Option<Vec<f32>>,
    b: Option<Vec<f32>>,
    eps: f32,
}

struct OLayer {
    n1: Option<ONorm>,
    qkv: OLinear,
    wo: OLinear,
    n2: ONorm,
    wi: OLinear,
    down: OLinear,
    full: bool,
    is_head: bool,
}

pub struct Oracle {
    d: usize,
    hd: usize,
    emb: Vec<f32>, // [vocab][d] decoded
    embnorm: ONorm,
    finalnorm: ONorm,
    trunk: Vec<OLayer>,
    head: Vec<OLayer>,
    type_emb: Vec<f32>,
    scorenorm: ONorm,
    score1: OLinear,
    score2: OLinear,
    act1: OLinear,
    act2: OLinear,
    cos_full: Vec<f32>,
    sin_full: Vec<f32>,
    cos_slide: Vec<f32>,
    sin_slide: Vec<f32>,
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608028654 * x * (1.0 + 0.044715 * x * x)).tanh())
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let sum = p.iter().sum::<f32>();
    for v in &mut p {
        *v /= sum;
    }
    p
}

fn norm_rows(x: &[f32], n: &ONorm, d: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for (a, o) in x.chunks_exact(d).zip(y.chunks_exact_mut(d)) {
        let mean = a.iter().sum::<f32>() / d as f32;
        let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let scale = 1.0 / (var + n.eps).sqrt();
        for j in 0..d {
            o[j] = (a[j] - mean) * scale * n.w.as_ref().map(|w| w[j]).unwrap_or(1.0)
                + n.b.as_ref().map(|b| b[j]).unwrap_or(0.0);
        }
    }
    y
}

fn apply(l: &OLinear, x: &[f32]) -> Vec<f32> {
    let m = x.len() / l.cols;
    let mut out = vec![0f32; m * l.rows];
    for mi in 0..m {
        for n in 0..l.rows {
            let mut acc = 0f32;
            let xr = &x[mi * l.cols..(mi + 1) * l.cols];
            let wr = &l.v[n * l.cols..(n + 1) * l.cols];
            for j in 0..l.cols {
                acc += xr[j] * wr[j];
            }
            if let Some(b) = &l.bias {
                acc += b[n];
            }
            out[mi * l.rows + n] = acc;
        }
    }
    out
}

/// Direct transcription of tensor::attention (rope, window, serial softmax).
fn attention(
    qkv: &[f32],
    s: usize,
    d: usize,
    hd: usize,
    rope: Option<(&[f32], &[f32])>,
    window: Option<usize>,
) -> Vec<f32> {
    let heads = d / hd;
    let mut q = vec![0f32; s * d];
    let mut k = q.clone();
    let mut v = q.clone();
    for i in 0..s {
        q[i * d..(i + 1) * d].copy_from_slice(&qkv[i * 3 * d..i * 3 * d + d]);
        k[i * d..(i + 1) * d].copy_from_slice(&qkv[i * 3 * d + d..i * 3 * d + 2 * d]);
        v[i * d..(i + 1) * d].copy_from_slice(&qkv[i * 3 * d + 2 * d..i * 3 * d + 3 * d]);
    }
    if let Some((cos, sin)) = rope {
        for a in [&mut q, &mut k] {
            for i in 0..s {
                for h in 0..heads {
                    for j in 0..hd / 2 {
                        let p = i * d + h * hd + j;
                        let u = a[p];
                        let vv = a[p + hd / 2];
                        a[p] = u * cos[i * hd + j] - vv * sin[i * hd + j];
                        a[p + hd / 2] = vv * cos[i * hd + j + hd / 2] + u * sin[i * hd + j + hd / 2];
                    }
                }
            }
        }
    }
    let mut out = vec![0f32; s * d];
    let scale = 1.0 / (hd as f32).sqrt();
    let mut score = vec![0f32; s];
    for h in 0..heads {
        for i in 0..s {
            let lo = window.map(|w| i.saturating_sub(w)).unwrap_or(0);
            let hi = window.map(|w| (i + w + 1).min(s)).unwrap_or(s);
            for j in lo..hi {
                let mut dot = 0f32;
                for z in 0..hd {
                    dot += q[i * d + h * hd + z] * k[j * d + h * hd + z];
                }
                score[j] = dot * scale;
            }
            let p = softmax(&score[lo..hi]);
            for (j, &pj) in (lo..hi).zip(&p) {
                for z in 0..hd {
                    out[i * d + h * hd + z] += pj * v[j * d + h * hd + z];
                }
            }
        }
    }
    out
}

impl Oracle {
    pub fn load(g: &GgufFile) -> Result<Oracle, String> {
        let graph = g.graph()?;
        let tensors = jget(&graph, "tensors").as_object().cloned().ok_or("tensors")?;
        let nodes = jget(&graph, "nodes").as_array().ok_or("nodes")?;
        let mut layers = 0;
        while g.tensors.contains_key(&format!("qkv.{layers}.weight")) {
            layers += 1;
        }
        let mut heads = 0;
        while g.tensors.contains_key(&format!("head_qkv.{heads}.weight")) {
            heads += 1;
        }
        let attns: Vec<&J> = nodes
            .iter()
            .filter(|n| jget(n, "opcode").as_f64() == Some(OP_ATTENTION))
            .collect();
        let fullmask = attns[0]
            .get("inputs")
            .and_then(|v| v.as_array())
            .and_then(|a| a.last())
            .cloned()
            .ok_or("mask")?;
        let dec = |name: &str| -> Result<Vec<f32>, String> {
            let (t, b) = g.data(name)?;
            match t.kind {
                0 => Ok(b
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()),
                1 => Ok(b
                    .chunks_exact(2)
                    .map(|c| half(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect()),
                k => Err(format!("oracle: kind {k} for {name}")),
            }
        };
        let mk_norm = |i: usize| -> Result<ONorm, String> {
            let key = if i == 0 { "layer_norm".into() } else { format!("layer_norm_{i}") };
            let node = nodes
                .iter()
                .find(|n| jget(n, "name").as_str() == Some(key.as_str()) && jget(n, "opcode").as_f64() == Some(OP_LAYER_NORM))
                .ok_or_else(|| format!("missing {key}"))?;
            let ins = node.get("inputs").and_then(|v| v.as_array()).unwrap();
            let getf = |idx: usize| -> Result<Option<Vec<f32>>, String> {
                if idx < ins.len() {
                    Ok(Some(dec(graph_tensor_name(&tensors, &ins[idx]))?))
                } else {
                    Ok(None)
                }
            };
            Ok(ONorm {
                w: getf(1)?,
                b: getf(2)?,
                eps: jget(jget(node, "attributes"), "eps").as_f64().unwrap_or(0.0) as f32,
            })
        };
        let mk_linear = |key: &str| -> Result<OLinear, String> {
            let weight = format!("{key}.weight");
            let (t, _) = g.data(&weight)?;
            let cols = t.dims[0];
            let rows = *t.dims.get(1).unwrap_or(&1);
            let node = nodes
                .iter()
                .find(|n| {
                    jget(n, "opcode").as_f64() == Some(OP_LINEAR)
                        && n.get("inputs").and_then(|i| i.as_array()).and_then(|a| a.first())
                            .map(|i| graph_tensor_name(&tensors, i) == weight)
                            .unwrap_or(false)
                })
                .ok_or_else(|| format!("graph lacks linear {key}"))?;
            let ins = node.get("inputs").and_then(|v| v.as_array()).unwrap();
            let mut biases: Vec<String> = Vec::new();
            if ins.len() > 2 {
                biases.push(graph_tensor_name(&tensors, &ins[2]).to_string());
            }
            for n in nodes {
                if jget(n, "opcode").as_f64() == Some(OP_FUSED_BIAS) {
                    let ni = n.get("inputs").and_then(|v| v.as_array()).unwrap();
                    let no = node.get("outputs").and_then(|v| v.as_array()).unwrap();
                    if ni.first() == no.first() {
                        biases.push(graph_tensor_name(&tensors, &ni[1]).to_string());
                    }
                }
            }
            let mut bias = None;
            if !biases.is_empty() {
                let mut out = vec![0f32; rows];
                for name in biases {
                    let v = dec(&name)?;
                    for (a, b) in out.iter_mut().zip(&v) {
                        *a += *b;
                    }
                }
                bias = Some(out);
            }
            Ok(OLinear {
                v: dec(&weight)?,
                rows,
                cols,
                bias,
            })
        };
        let mut trunk = Vec::new();
        for i in 0..layers {
            trunk.push(OLayer {
                n1: if i == 0 { None } else { Some(mk_norm(2 * i)?) },
                qkv: mk_linear(&format!("qkv.{i}"))?,
                wo: mk_linear(&format!("wo.{i}"))?,
                n2: mk_norm(2 * i + 1)?,
                wi: mk_linear(&format!("mlp_wi.{i}"))?,
                down: mk_linear(&format!("mlp_wo.{i}"))?,
                full: attns[i]
                    .get("inputs")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.last())
                    .cloned()
                    == Some(fullmask.clone()),
                is_head: false,
            });
        }
        let mut head = Vec::new();
        for i in 0..heads {
            head.push(OLayer {
                n1: Some(mk_norm(2 * layers + 1 + 2 * i)?),
                qkv: mk_linear(&format!("head_qkv.{i}"))?,
                wo: mk_linear(&format!("head_wo.{i}"))?,
                n2: mk_norm(2 * layers + 2 + 2 * i)?,
                wi: mk_linear(&format!("head_fc1.{i}"))?,
                down: mk_linear(&format!("head_fc2.{i}"))?,
                full: true,
                is_head: true,
            });
        }
        Ok(Oracle {
            d: g.tensors.get("tok_emb.weight").unwrap().dims[0],
            hd: g.tensors.get("cos_full").unwrap().dims[0],
            emb: dec("tok_emb.weight")?,
            embnorm: mk_norm(0)?,
            finalnorm: mk_norm(2 * layers)?,
            trunk,
            head,
            type_emb: dec("type_emb.weight")?,
            scorenorm: mk_norm(2 * layers + 1 + 2 * heads)?,
            score1: mk_linear("scorer_fc1")?,
            score2: mk_linear("scorer_fc2")?,
            act1: mk_linear("act_fc1")?,
            act2: mk_linear("act_fc2")?,
            cos_full: dec("cos_full")?,
            sin_full: dec("sin_full")?,
            cos_slide: dec("cos_slide")?,
            sin_slide: dec("sin_slide")?,
        })
    }

    /// Full forward with named stage snapshots (matching the GPU trace).
    pub fn forward_traced(
        &self,
        ids: &[u32],
        markers: &[u32],
        qtype: u32,
    ) -> (Vec<f32>, Vec<f32>, HashMap<String, Vec<f32>>) {
        let mut t: HashMap<String, Vec<f32>> = HashMap::new();
        let s = ids.len();
        let d = self.d;
        let mut h: Vec<f32> = Vec::with_capacity(s * d);
        for &id in ids {
            let r = id as usize;
            h.extend_from_slice(&self.emb[r * d..(r + 1) * d]);
        }
        t.insert("emb".into(), h.clone());
        h = norm_rows(&h, &self.embnorm, d);
        t.insert("embnorm".into(), h.clone());
        for (li, l) in self.trunk.iter().enumerate() {
            let x = if let Some(n1) = &l.n1 {
                let xn = norm_rows(&h, n1, d);
                t.insert(format!("t{li}.xn"), xn.clone());
                xn
            } else {
                h.clone()
            };
            let qkv = apply(&l.qkv, &x);
            t.insert(format!("t{li}.qkv"), qkv.clone());
            let (cos, sin) = if l.full {
                (&self.cos_full, &self.sin_full)
            } else {
                (&self.cos_slide, &self.sin_slide)
            };
            let a = attention(
                &qkv,
                s,
                d,
                self.hd,
                Some((cos, sin)),
                if l.full { None } else { Some(64) },
            );
            t.insert(format!("t{li}.attn"), a.clone());
            let wo = apply(&l.wo, &a);
            t.insert(format!("t{li}.wo"), wo.clone());
            for (hh, w) in h.iter_mut().zip(&wo) {
                *hh += w;
            }
            t.insert(format!("t{li}.h1"), h.clone());
            let xn = norm_rows(&h, &l.n2, d);
            t.insert(format!("t{li}.n2"), xn.clone());
            let up = apply(&l.wi, &xn);
            t.insert(format!("t{li}.up"), up.clone());
            let f = l.down.cols;
            let mut mid = vec![0f32; s * f];
            for k in 0..s * f {
                let i = k / f;
                let j = k - i * f;
                mid[k] = gelu(up[i * 2 * f + j]) * up[i * 2 * f + f + j];
            }
            t.insert(format!("t{li}.mid"), mid.clone());
            let dl = apply(&l.down, &mid);
            for (hh, w) in h.iter_mut().zip(&dl) {
                *hh += w;
            }
            t.insert(format!("t{li}.h2"), h.clone());
        }
        h = norm_rows(&h, &self.finalnorm, d);
        t.insert("finalnorm".into(), h.clone());
        let q = qtype as usize;
        for row in h.chunks_exact_mut(d) {
            for (a, b) in row.iter_mut().zip(&self.type_emb[q * d..(q + 1) * d]) {
                *a += b;
            }
        }
        t.insert("typeemb".into(), h.clone());
        for (li, l) in self.head.iter().enumerate() {
            let xn = norm_rows(&h, l.n1.as_ref().unwrap(), d);
            t.insert(format!("h{li}.xn"), xn.clone());
            let qkv = apply(&l.qkv, &xn);
            let a = attention(&qkv, s, d, self.hd, None, None);
            t.insert(format!("h{li}.attn"), a.clone());
            let wo = apply(&l.wo, &a);
            for (hh, w) in h.iter_mut().zip(&wo) {
                *hh += w;
            }
            t.insert(format!("h{li}.h1"), h.clone());
            let xn = norm_rows(&h, &l.n2, d);
            let mut up = apply(&l.wi, &xn);
            for v in &mut up {
                *v = v.max(0.0);
            }
            t.insert(format!("h{li}.up"), up.clone());
            let dl = apply(&l.down, &up);
            for (hh, w) in h.iter_mut().zip(&dl) {
                *hh += w;
            }
            t.insert(format!("h{li}.h2"), h.clone());
        }
        let kk = markers.len();
        let mut gathered = Vec::with_capacity(kk * d);
        for &p in markers {
            gathered.extend_from_slice(&h[p as usize * d..(p as usize + 1) * d]);
        }
        t.insert("gathered".into(), gathered.clone());
        let sx = norm_rows(&gathered, &self.scorenorm, d);
        t.insert("scorenorm".into(), sx.clone());
        let mut s1 = apply(&self.score1, &sx);
        for v in &mut s1 {
            *v = gelu(*v);
        }
        t.insert("score1".into(), s1.clone());
        let logits = apply(&self.score2, &s1);
        t.insert("logits".into(), logits.clone());
        let p = softmax(&logits);
        let top1 = p.iter().copied().fold(0f32, f32::max);
        let top2 = p
            .iter()
            .map(|&v| v * (top1 - v) / ((top1 - v) + 1e-6))
            .fold(0f32, f32::max);
        let k = kk.max(2) as f32;
        let ent = -p.iter().map(|v| v * v.max(1e-9).ln()).sum::<f32>() / k.ln();
        let mut pooled = h[..d].to_vec();
        pooled.extend_from_slice(&[top1, top1 - top2, ent, k / 255.0]);
        t.insert("pooled".into(), pooled.clone());
        let mut a1 = apply(&self.act1, &pooled);
        for v in &mut a1 {
            *v = gelu(*v);
        }
        t.insert("act1".into(), a1.clone());
        let act = apply(&self.act2, &a1);
        (logits, act.to_vec(), t)
    }
}
