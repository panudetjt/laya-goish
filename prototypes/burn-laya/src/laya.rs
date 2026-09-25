// Laya model: burn 0.21 port of prototypes/candle-laya/src/laya.rs (which is
// itself a 1:1 port of the engine's src/models/laya/model.rs op order),
// loaded from converted safetensors weights instead of GGUF.
//
// Hand-written forward using stock burn tensor ops only — the point of this
// prototype is measuring what the framework's CPU kernels (burn-flex/gemm)
// and the wgpu backend do on their own. LayerNorm/GeGLU/RoPE/softmax are
// composed from primitives exactly like the candle port did pre-custom-op.
use crate::weights::Weights;
use burn::tensor::backend::Backend;
use burn::tensor::{activation, DType, Int, Tensor, TensorData};

pub struct Linear<B: Backend> {
    w: Tensor<B, 2>, // [out, in], row-major
    b: Option<Tensor<B, 1>>,
}

pub struct Norm<B: Backend> {
    w: Option<Tensor<B, 1>>,
    b: Option<Tensor<B, 1>>,
    eps: f32,
}

pub struct Layer<B: Backend> {
    pub n1: Option<Norm<B>>,
    pub qkv: Linear<B>,
    pub wo: Linear<B>,
    pub n2: Norm<B>,
    pub wi: Linear<B>,
    pub down: Linear<B>,
    pub full: bool,
}

pub struct BatchReq {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: usize,
}

pub struct Model<B: Backend> {
    pub layers: usize,
    pub d: usize,
    pub hd: usize,
    pub heads: usize,
    // vocab embedding kept in `chunks` row-blocks: needed on llvmpipe, where
    // a single >128MB storage buffer exceeds the device limit
    emb_chunks: Vec<Tensor<B, 2>>,
    embnorm: Norm<B>,
    finalnorm: Norm<B>,
    pub trunk: Vec<Layer<B>>,
    pub head: Vec<Layer<B>>,
    type_emb: Tensor<B, 2>, // [3, d]
    scorenorm: Norm<B>,
    score1: Linear<B>,
    score2: Linear<B>,
    act1: Linear<B>,
    act2: Linear<B>,
    cos_full: Tensor<B, 2>, // [513, hd]
    sin_full: Tensor<B, 2>,
    cos_slide: Tensor<B, 2>,
    sin_slide: Tensor<B, 2>,
    f16_compute: bool,
}

impl<B: Backend> Linear<B> {
    fn new(w: &Weights, key: &str, device: &B::Device) -> Result<Self, String> {
        let wt = w
            .matrix(&format!("{key}.weight"), device)
            .map_err(|e| format!("{key}: {e}"))?;
        let b = match w.vector(&format!("{key}.bias"), device) {
            Ok(b) => Some(b),
            Err(_) => None,
        };
        Ok(Self { w: wt, b })
    }

    // F16-resident weights; matmuls run in F16 with F32 accumulation (gemm
    // f16 semantics), same as the candle f16-mode prototype.
    fn into_f16(mut self) -> Self {
        self.w = self.w.cast(DType::F16);
        self
    }

    fn run(&self, x: Tensor<B, 2>, f16: bool) -> Tensor<B, 2> {
        let out = if f16 {
            x.cast(DType::F16)
                .matmul(self.w.clone().swap_dims(0, 1))
                .cast(DType::F32)
        } else {
            x.matmul(self.w.clone().swap_dims(0, 1))
        };
        match &self.b {
            Some(b) => {
                let [s, n] = out.dims();
                out + b.clone().reshape([1, n]).expand([s, n])
            }
            None => out,
        }
    }
}

impl<B: Backend> Norm<B> {
    fn new(w: &Weights, name: &str, _d: usize, device: &B::Device) -> Result<Self, String> {
        Ok(Self {
            w: w.vector_opt(&format!("{name}.weight"), device),
            b: w.vector_opt(&format!("{name}.bias"), device),
            eps: w.norm_eps(name),
        })
    }

    // Engine layer_norm: per-row mean, biased var, scale = 1/sqrt(var+eps),
    // y = (x-mean)*scale*w + b — composed from stock burn ops.
    fn run(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [s, d] = x.dims();
        let mean = x.clone().sum_dim(1).div_scalar(d as f64); // [s, 1]
        let xc = x - mean.expand([s, d]);
        let sq = xc.clone().mul(xc.clone());
        let var = sq.sum_dim(1).div_scalar(d as f64); // [s, 1]
        let scale = var.add_scalar(self.eps as f64).sqrt().recip();
        let mut y = xc.mul(scale.expand([s, d]));
        if let Some(w) = &self.w {
            y = y.mul(w.clone().reshape([1, d]).expand([s, d]));
        }
        if let Some(b) = &self.b {
            y = y + b.clone().reshape([1, d]).expand([s, d]);
        }
        y
    }
}

// Engine tensor::gelu: 0.5*x*(1 + tanh(0.7978845608028654*x*(1+0.044715*x*x)))
fn gelu_tanh<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let xsq = x.clone().mul(x.clone());
    let t = xsq.mul_scalar(0.044715).add_scalar(1.0);
    let z = x.clone().mul(t).mul_scalar(0.7978845608028654);
    let th = z.tanh();
    x.mul_scalar(0.5).mul(th.add_scalar(1.0))
}

// Engine RoPE pairing: a[j] = u*cos[j] - v*sin[j]; a[j+hd/2] = v*cos[j+hd/2]
// + u*sin[j+hd/2], table row = position. q: [h, s, hd], tables: [s, hd].
fn rope_rotate<B: Backend>(
    q: Tensor<B, 3>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
    s: usize,
    hd: usize,
) -> Tensor<B, 3> {
    let h = q.dims()[0];
    let half = hd / 2;
    let c1: Tensor<B, 3> = cos.clone().slice([0..s, 0..half]).reshape([1, s, half]);
    let c2: Tensor<B, 3> = cos.clone().slice([0..s, half..hd]).reshape([1, s, half]);
    let s1: Tensor<B, 3> = sin.clone().slice([0..s, 0..half]).reshape([1, s, half]);
    let s2: Tensor<B, 3> = sin.clone().slice([0..s, half..hd]).reshape([1, s, half]);
    let q1 = q.clone().slice([0..h, 0..s, 0..half]);
    let q2 = q.clone().slice([0..h, 0..s, half..hd]);
    let c1e = c1.expand([h, s, half]);
    let c2e = c2.expand([h, s, half]);
    let s1e = s1.expand([h, s, half]);
    let s2e = s2.expand([h, s, half]);
    let r1 = q1.clone().mul(c1e).sub(q2.clone().mul(s1e));
    let r2 = q2.mul(c2e).add(q1.mul(s2e));
    Tensor::cat(vec![r1, r2], 2)
}

// Engine softmax, exact op semantics (used for the decision layer only).
pub fn softmax_f32(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let sum = p.iter().sum::<f32>();
    for v in &mut p {
        *v /= sum;
    }
    p
}

impl<B: Backend> Model<B> {
    fn dbg(stage: &str, t: &Tensor<B, 2>) {
        if std::env::var("LAYA_DEBUG").is_err() {
            return;
        }
        let data = t.clone().slice([0..1.min(t.dims()[0]), 0..t.dims()[1].min(8)]).into_data();
        let v: Vec<f32> = data.to_vec::<f32>().unwrap_or_default();
        let all = t.clone().into_data();
        let av: Vec<f32> = all.to_vec::<f32>().unwrap_or_default();
        let n = av.len().max(1) as f32;
        let mean = av.iter().sum::<f32>() / n;
        let amax = av.iter().fold(0f32, |m, x| m.max(x.abs()));
        eprintln!("DBG {stage:<22} mean={mean:9.4} max|x|={amax:9.4} row0[..8]={v:?}");
    }

    pub fn load(
        w: &Weights,
        f16_compute: bool,
        emb_chunks: usize,
        device: &B::Device,
    ) -> Result<Self, String> {
        let layers = w.layers;
        let d = w.d;
        let hd = w.hd;
        let head_layers = w.head_layers;
        let full = &w.full_layers;
        let layer = |n1: Option<Norm<B>>,
                     qkv: &str,
                     wo: &str,
                     n2i: usize,
                     wi: &str,
                     down: &str,
                     full: bool|
         -> Result<Layer<B>, String> {
            let lin = |key: &str| -> Result<Linear<B>, String> {
                let mut l = Linear::new(w, key, device)?;
                if f16_compute {
                    l = l.into_f16();
                }
                Ok(l)
            };
            Ok(Layer {
                n1,
                qkv: lin(qkv)?,
                wo: lin(wo)?,
                n2: Norm::new(w, &format!("layer_norm_{n2i}"), d, device)?,
                wi: lin(wi)?,
                down: lin(down)?,
                full,
            })
        };
        let mut trunk = Vec::new();
        for i in 0..layers {
            trunk.push(layer(
                if i == 0 {
                    None
                } else {
                    Some(Norm::new(w, &format!("layer_norm_{}", 2 * i), d, device)?)
                },
                &format!("qkv.{i}"),
                &format!("wo.{i}"),
                2 * i + 1,
                &format!("mlp_wi.{i}"),
                &format!("mlp_wo.{i}"),
                full.contains(&i),
            )?);
        }
        let mut head = Vec::new();
        for i in 0..head_layers {
            head.push(layer(
                Some(Norm::new(
                    w,
                    &format!("layer_norm_{}", 2 * layers + 1 + 2 * i),
                    d,
                    device,
                )?),
                &format!("head_qkv.{i}"),
                &format!("head_wo.{i}"),
                2 * layers + 2 + 2 * i,
                &format!("head_fc1.{i}"),
                &format!("head_fc2.{i}"),
                true,
            )?);
        }
        let vocab = w.manifest["vocab"].as_u64().ok_or("vocab")? as usize;
        let emb_split: Vec<Tensor<B, 2>> = if emb_chunks > 1 {
            (0..emb_chunks)
                .map(|i| {
                    let b = vocab * i / emb_chunks;
                    let e = vocab * (i + 1) / emb_chunks;
                    let t = w
                        .matrix_rows::<B>("tok_emb.weight", b, e - b, device)
                        .map_err(|err| format!("tok_emb: {err}"))?;
                    Ok(if f16_compute { t.cast(DType::F16) } else { t })
                })
                .collect::<Result<_, String>>()?
        } else {
            let emb = if f16_compute {
                // F16-resident like the engine: select rows first, upcast the
                // gathered activations only (candle f16-mode semantics).
                w.matrix::<B>("tok_emb.weight", device)
                    .map_err(|e| format!("tok_emb: {e}"))?
                    .cast(DType::F16)
            } else {
                w.matrix("tok_emb.weight", device)
                    .map_err(|e| format!("tok_emb: {e}"))?
            };
            vec![emb]
        };
        let rope = |name: &str| w.matrix(name, device);
        Ok(Self {
            layers,
            d,
            hd,
            heads: d / hd,
            emb_chunks: emb_split,
            embnorm: Norm::new(w, "layer_norm", d, device)?,
            finalnorm: Norm::new(w, &format!("layer_norm_{}", 2 * layers), d, device)?,
            trunk,
            head,
            type_emb: w
                .matrix("type_emb.weight", device)
                .map_err(|e| format!("type_emb: {e}"))?,
            scorenorm: Norm::new(
                w,
                &format!("layer_norm_{}", 2 * layers + 1 + 2 * head_layers),
                d,
                device,
            )?,
            score1: {
                let mut l = Linear::new(w, "scorer_fc1", device)?;
                if f16_compute {
                    l = l.into_f16();
                }
                l
            },
            score2: {
                let mut l = Linear::new(w, "scorer_fc2", device)?;
                if f16_compute {
                    l = l.into_f16();
                }
                l
            },
            act1: {
                let mut l = Linear::new(w, "act_fc1", device)?;
                if f16_compute {
                    l = l.into_f16();
                }
                l
            },
            act2: {
                let mut l = Linear::new(w, "act_fc2", device)?;
                if f16_compute {
                    l = l.into_f16();
                }
                l
            },
            cos_full: rope("cos_full")?,
            sin_full: rope("sin_full")?,
            cos_slide: rope("cos_slide")?,
            sin_slide: rope("sin_slide")?,
            f16_compute,
        })
    }

    // Single-segment attention over qkv [s, 3d]; mirrors tensor::attention
    // (RoPE table indexing, scale, softmax) with the engine's ±64 window.
    fn attention_one(
        &self,
        qkv: Tensor<B, 2>,
        s: usize,
        rope: Option<(&Tensor<B, 2>, &Tensor<B, 2>)>,
        windowed: bool,
    ) -> Tensor<B, 2> {
        let d = self.d;
        let hd = self.hd;
        let h = self.heads;
        let window = 64usize;
        let split = |off: usize| -> Tensor<B, 3> {
            qkv.clone()
                .slice([0..s, off..off + d])
                .reshape([s, h, hd])
                .swap_dims(0, 1)
        };
        let mut q = split(0);
        let mut k = split(d);
        let v = split(2 * d);
        if let Some((cos, sin)) = rope {
            if std::env::var("LAYA_DEBUG").is_ok() {
                let c0: Vec<f32> = cos.clone().slice([0..2, 0..4]).into_data().to_vec().unwrap();
                let s0: Vec<f32> = sin.clone().slice([0..2, 0..4]).into_data().to_vec().unwrap();
                let qpre: Vec<f32> = q
                    .clone()
                    .slice([0..1, 0..2, 0..4])
                    .into_data()
                    .to_vec()
                    .unwrap();
                eprintln!("DBG rope cos[0..2][0..4]={c0:?} sin={s0:?} q[0]h0[..4]={qpre:?}");
            }
            q = rope_rotate(q, cos, sin, s, hd);
            k = rope_rotate(k, cos, sin, s, hd);
            if std::env::var("LAYA_DEBUG").is_ok() {
                let qpost: Vec<f32> = q
                    .clone()
                    .slice([0..1, 0..2, 0..4])
                    .into_data()
                    .to_vec()
                    .unwrap();
                let kpost: Vec<f32> = k
                    .clone()
                    .slice([0..1, 0..2, 0..4])
                    .into_data()
                    .to_vec()
                    .unwrap();
                eprintln!("DBG rope q'[0]h0[..4]={qpost:?} k'[0]h0[..4]={kpost:?}");
            }
        }
        let scale = 1.0f64 / (hd as f64).sqrt();
        let attn = |qc: Tensor<B, 3>,
                    kc: Tensor<B, 3>,
                    vc: Tensor<B, 3>,
                    mask: Option<Tensor<B, 2>>|
         -> Tensor<B, 3> {
            let kt = kc.swap_dims(1, 2);
            let mut scores = qc.matmul(kt).mul_scalar(scale);
            if let Some(m) = mask {
                let [nq, kl] = m.dims();
                scores = scores + m.reshape([1, nq, kl]).expand([h, nq, kl]);
            }
            activation::softmax(scores, 2).matmul(vc)
        };
        let out = if windowed && s > window + 1 {
            let mut outs: Vec<Tensor<B, 3>> = Vec::new();
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
                let device = q.device();
                let mt: Tensor<B, 2> =
                    Tensor::<B, 1>::from_floats(m.as_slice(), &device).reshape([nq, kl]);
                outs.push(attn(
                    q.clone().narrow(1, c0, nq),
                    k.clone().narrow(1, k0, kl),
                    v.clone().narrow(1, k0, kl),
                    Some(mt),
                ));
                c0 += nq;
            }
            Tensor::cat(outs, 1)
        } else {
            attn(q, k, v, None)
        };
        out.swap_dims(0, 1).reshape([s, d])
    }

    // Attention over a stacked batch, per segment with local RoPE positions.
    fn attention_segments(
        &self,
        qkv: Tensor<B, 2>,
        segs: &[(usize, usize)],
        rope_tables: Option<(&Tensor<B, 2>, &Tensor<B, 2>)>,
        windowed: bool,
    ) -> Tensor<B, 2> {
        let mut outs = Vec::with_capacity(segs.len());
        for &(st, n) in segs {
            let seg = qkv.clone().narrow(0, st, n);
            let rope = match rope_tables {
                Some((c, s)) => Some((
                    c.clone().narrow(0, 0, n),
                    s.clone().narrow(0, 0, n),
                )),
                None => None,
            };
            let rope_ref = rope.as_ref().map(|(c, s)| (c, s));
            outs.push(self.attention_one(seg, n, rope_ref, windowed));
        }
        Tensor::cat(outs, 0)
    }

    // Row-gather from the (possibly chunked) embedding table. Chunks are a
    // no-op single block on CPU; on llvmpipe they dodge the ~128MB storage
    // buffer cap. Math identical to one select over the full table.
    fn gather_emb(&self, ids: &Tensor<B, 1, Int>, s: usize) -> Tensor<B, 2> {
        if self.emb_chunks.len() == 1 {
            return self.emb_chunks[0].clone().select(0, ids.clone());
        }
        let device = ids.device();
        let d = self.d;
        let mut out: Option<Tensor<B, 2>> = None;
        let mut base = 0usize;
        for chunk in &self.emb_chunks {
            let h = chunk.dims()[0];
            let raw = ids.clone().sub_scalar(base as i64);
            let zero = Tensor::<B, 1, Int>::zeros([s], &device);
            let hi = Tensor::<B, 1, Int>::full([s], (h - 1) as i64, &device);
            let clamped = raw
                .clone()
                .mask_where(raw.clone().lower_elem(0), zero)
                .mask_where(raw.clone().greater_equal_elem(h as i64), hi);
            let rows = chunk.clone().select(0, clamped);
            let inside = raw
                .clone()
                .lower_elem(h as i64)
                .bool_and(raw.greater_equal_elem(0));
            let [n] = inside.dims();
            let mask2d = inside.reshape([n, 1]).expand([n, d]);
            let zeros = Tensor::<B, 2>::zeros([s, d], &device).cast(rows.dtype());
            let contribution = zeros.mask_where(mask2d, rows);
            out = Some(match out {
                None => contribution,
                Some(acc) => acc + contribution,
            });
            base += h;
        }
        out.expect("emb chunks nonempty")
    }

    pub fn forward(
        &self,
        ids: &[u32],
        markers: &[usize],
        qtype: usize,
        device: &B::Device,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let reqs = [BatchReq {
            ids: ids.to_vec(),
            markers: markers.to_vec(),
            qtype,
        }];
        self.forward_batch(&reqs, device)
            .map(|mut v| v.remove(0))
    }

    pub fn forward_batch(
        &self,
        reqs: &[BatchReq],
        device: &B::Device,
    ) -> Result<Vec<(Vec<f32>, Vec<f32>)>, String> {
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
            qtypes.extend(std::iter::repeat(r.qtype as u64).take(r.ids.len()));
        }
        let ids_t: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(ids_all, [s]).convert::<i64>(),
            device,
        );
        let qtypes_t: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(qtypes, [s]).convert::<i64>(),
            device,
        );
        let segs: Vec<(usize, usize)> = reqs
            .iter()
            .zip(&seg_start)
            .map(|(r, &st)| (st, r.ids.len()))
            .collect();

        let emb_rows = self.gather_emb(&ids_t, s);
        let emb = if self.f16_compute {
            emb_rows.cast(DType::F32)
        } else {
            emb_rows
        };
        let mut h = self.embnorm.run(emb);
        Self::dbg("after embnorm", &h);
        for (li, l) in self.trunk.iter().enumerate() {
            let x = match &l.n1 {
                Some(n) => n.run(h.clone()),
                None => h.clone(),
            };
            let qkv = l.qkv.run(x, self.f16_compute);
            if li < 2 {
                Self::dbg(&format!("L{li} qkv"), &qkv);
            }
            let rope = if l.full {
                Some((&self.cos_full, &self.sin_full))
            } else {
                Some((&self.cos_slide, &self.sin_slide))
            };
            let a = self.attention_segments(qkv, &segs, rope, !l.full);
            if li < 2 {
                Self::dbg(&format!("L{li} attn out"), &a);
            }
            h = h.clone() + l.wo.run(a, self.f16_compute);
            let n = l.n2.run(h.clone());
            let up = l.wi.run(n, self.f16_compute);
            let m = geglu(up, d);
            h = h.clone() + l.down.run(m, self.f16_compute);
            if li < 2 {
                Self::dbg(&format!("L{li} out"), &h);
            }
        }
        h = self.finalnorm.run(h);
        Self::dbg("after finalnorm", &h);
        let type_rows = self.type_emb.clone().select(0, qtypes_t.clone());
        h = h + type_rows;
        for (hi, l) in self.head.iter().enumerate() {
            let x = l.n1.as_ref().unwrap().run(h.clone());
            let qkv = l.qkv.run(x, self.f16_compute);
            Self::dbg(&format!("head{hi} qkv"), &qkv);
            let a = self.attention_segments(qkv, &segs, None, false);
            Self::dbg(&format!("head{hi} attn"), &a);
            h = h.clone() + l.wo.run(a, self.f16_compute);
            let x = l.n2.run(h.clone());
            let f = activation::relu(l.wi.run(x, self.f16_compute));
            h = h.clone() + l.down.run(f, self.f16_compute);
            Self::dbg(&format!("head{hi} out"), &h);
        }

        let mut out = Vec::with_capacity(reqs.len());
        for (r, st) in reqs.iter().zip(&seg_start) {
            let marker_ids: Vec<u64> = r.markers.iter().map(|&m| (st + m) as u64).collect();
            let mt: Tensor<B, 1, Int> = Tensor::from_data(
                TensorData::new(marker_ids, [r.markers.len()]).convert::<i64>(),
                device,
            );
            let gathered = h.clone().select(0, mt);
            let x = self.scorenorm.run(gathered);
            let x = gelu_tanh(self.score1.run(x, self.f16_compute));
            let logits_v: Vec<f32> = self
                .score2
                .run(x, self.f16_compute)
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| e.to_string())?;

            // Act head: pooled = first row of this request + engine features.
            let p0 = softmax_f32(&logits_v);
            let top1 = p0.iter().copied().fold(0f32, f32::max);
            let top2 = p0
                .iter()
                .map(|&p| p * (top1 - p) / ((top1 - p) + 1e-6))
                .fold(0f32, f32::max);
            let k = r.markers.len().max(2) as f32;
            let ent = -p0.iter().map(|&p| p * p.max(1e-9).ln()).sum::<f32>() / k.ln();
            let mut pooled: Vec<f32> = h
                .clone()
                .narrow(0, *st, 1)
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| e.to_string())?;
            pooled.extend_from_slice(&[top1, top1 - top2, ent, k / 255.0]);
            let pt: Tensor<B, 2> = Tensor::<B, 1>::from_floats(pooled.as_slice(), device)
                .reshape([1, d + 4]);
            let x = gelu_tanh(self.act1.run(pt, self.f16_compute));
            let act_v: Vec<f32> = self
                .act2
                .run(x, self.f16_compute)
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| e.to_string())?;
            out.push((logits_v, act_v));
        }
        Ok(out)
    }
}

// Engine GeGLU: gelu(x[:, :f]) * x[:, f:], engine gelu-tanh constants.
fn geglu<B: Backend>(up: Tensor<B, 2>, d: usize) -> Tensor<B, 2> {
    let [s, two_f] = up.dims();
    let _ = d;
    let f = two_f / 2;
    let g = up.clone().slice([0..s, 0..f]);
    let u = up.slice([0..s, f..two_f]);
    let gg = gelu_tanh(g);
    gg.mul(u)
}
