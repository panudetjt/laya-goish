//! Forward-pass orchestration: one complete Laya forward per call, exactly
//! the op sequence of the CPU engine's Model::forward
//! (src/models/laya/model.rs), executed as CUDA kernels over
//! VRAM-resident weights. Per-request PCIe traffic is counted; weights are
//! never re-uploaded (they live in the model region uploaded at load time).
use crate::cuda::{Cuda, DevMem, Function};
use crate::laya::{LayaGpu, Linear, Norm};
use std::ffi::c_void;
use std::time::Instant;

pub struct Kernels {
    matmul: Function,
    layernorm: Function,
    attention: Function,
    embed: Function,
    geglu: Function,
    relu: Function,
    gelu: Function,
    add: Function,
    add_typeemb: Function,
    gather: Function,
}

impl Kernels {
    pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<Kernels, String> {
        let m = cuda.load_module(cubin)?;
        Ok(Kernels {
            matmul: m.function("matmul_f16")?,
            layernorm: m.function("layernorm")?,
            attention: m.function("attention")?,
            embed: m.function("embed_gather")?,
            geglu: m.function("geglu")?,
            relu: m.function("relu_inplace")?,
            gelu: m.function("gelu_inplace")?,
            add: m.function("add_inplace")?,
            add_typeemb: m.function("add_typeemb")?,
            gather: m.function("gather_rows")?,
        })
    }
}

struct Buffers {
    ids: DevMem,
    markers: DevMem,
    h_a: DevMem,
    h_b: DevMem,
    xn: DevMem,
    qkv: DevMem,
    attn: DevMem,
    tmp: DevMem,
    up: DevMem,
    mid: DevMem,
    /// k x d rows for the scorer path (also score1 output).
    sc_in: DevMem,
    sc_x: DevMem,
    pooled: DevMem,
    ax: DevMem,
    actout: DevMem,
    logits: DevMem,
}

pub struct Runner {
    cuda: Cuda,
    k: Kernels,
    pub m: LayaGpu,
    buf: Buffers,
    /// Debug stage capture (stage-check): set to Some(vec![]) before forward.
    pub trace: Option<std::cell::RefCell<Vec<(String, Vec<f32>)>>>,
    rope_full_elems: usize,
    rope_slide_elems: usize,
    pub last_h2d: u64,
    pub last_d2h: u64,
    pub last_gpu_us: u128,
}

fn pi32(v: &mut i32) -> *mut c_void {
    v as *mut i32 as *mut c_void
}
fn pf32(v: &mut f32) -> *mut c_void {
    v as *mut f32 as *mut c_void
}
fn pu64(v: &mut u64) -> *mut c_void {
    v as *mut u64 as *mut c_void
}

impl Runner {
    pub fn new(cuda: &Cuda, m: LayaGpu, cubin: &[u8]) -> Result<Runner, String> {
        let k = Kernels::load(cuda, cubin)?;
        let d = m.d;
        let s = m.max_len;
        let max_up = m
            .trunk
            .iter()
            .chain(m.head.iter())
            .map(|l| l.wi.rows)
            .max()
            .unwrap_or(d * 2);
        let max_mid_f = m.trunk.iter().map(|l| l.down.cols).max().unwrap_or(d);
        let rope_full_elems = m.rope_full_elems;
        let rope_slide_elems = m.rope_slide_elems;
        let buf = Buffers {
            ids: cuda.alloc(s * 4)?,
            markers: cuda.alloc(16 * 4)?,
            h_a: cuda.alloc(s * d * 4)?,
            h_b: cuda.alloc(s * d * 4)?,
            xn: cuda.alloc(s * d * 4)?,
            qkv: cuda.alloc(s * 3 * d * 4)?,
            attn: cuda.alloc(s * d * 4)?,
            tmp: cuda.alloc(s * d * 4)?,
            up: cuda.alloc(s * max_up * 4)?,
            mid: cuda.alloc(s * max_mid_f * 4)?,
            sc_in: cuda.alloc(16 * d * 4)?,
            sc_x: cuda.alloc(16 * d * 4)?,
            pooled: cuda.alloc((d + 4) * 4)?,
            ax: cuda.alloc(256 * 4)?,
            actout: cuda.alloc(2 * 4)?,
            logits: cuda.alloc(16 * 4)?,
        };
        // Invariant: clone the Cuda handle AFTER every allocation so the
        // cloned allocation counter reflects the true total (Cell clones by
        // value; later allocs through another clone would not be seen here).
        Ok(Runner {
            cuda: cuda.clone(),
            k,
            m,
            buf,
            trace: None,
            rope_full_elems,
            rope_slide_elems,
            last_h2d: 0,
            last_d2h: 0,
            last_gpu_us: 0,
        })
    }

    pub fn cuda_allocated(&self) -> usize {
        self.cuda.allocated.get()
    }

    fn trace_stage(&self, name: &str, ptr: u64, elems: usize) -> Result<(), String> {
        if self.trace.is_none() {
            return Ok(());
        }
        self.cuda.synchronize()?;
        let mut b = vec![0u8; elems * 4];
        self.cuda.d2h(ptr, &mut b)?;
        let v: Vec<f32> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if let Some(t) = &self.trace {
            t.borrow_mut().push((name.to_string(), v));
        }
        Ok(())
    }

    fn ln(&self, out: u64, a: u64, n: &Norm, rows: usize) -> Result<(), String> {
        let mut d = self.m.d as i32;
        let mut out = out;
        let mut a = a;
        let mut w = n.w;
        let mut b = n.b;
        let mut rows = rows as i32;
        let mut eps = n.eps;
        let grid = ((rows.max(1) as u32).div_ceil(256), 1, 1);
        let mut params = vec![
            pu64(&mut out),
            pu64(&mut a),
            pu64(&mut w),
            pu64(&mut b),
            pi32(&mut rows),
            pi32(&mut d),
            pf32(&mut eps),
        ];
        self.cuda
            .launch(&self.k.layernorm, grid, (256, 1, 1), 0, &mut params)
    }

    fn mm(&self, out: u64, x: u64, l: &Linear, m_rows: usize) -> Result<(), String> {
        let mut out = out;
        let mut x = x;
        let mut w = l.w;
        let mut bias = self.m.bias_ptr(l);
        let mut m = m_rows as i32;
        let mut n = l.rows as i32;
        let mut kk = l.cols as i32;
        let mut wf = if l.kind == 1 { 1i32 } else { 0i32 };
        let grid = ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1);
        let mut params = vec![
            pu64(&mut out),
            pu64(&mut x),
            pu64(&mut w),
            pu64(&mut bias),
            pi32(&mut m),
            pi32(&mut n),
            pi32(&mut kk),
            pi32(&mut wf),
        ];
        self.cuda
            .launch(&self.k.matmul, grid, (16, 16, 1), 0, &mut params)
    }

    fn elem(&self, f: &Function, args: Vec<*mut c_void>, n: usize) -> Result<(), String> {
        let grid = ((n.max(1) as u32).div_ceil(256), 1, 1);
        let mut args = args;
        self.cuda.launch(f, grid, (256, 1, 1), 0, &mut args)
    }

    fn attn(&self, cos: u64, sin: u64, window: i32, s: usize) -> Result<(), String> {
        let mut out = self.buf.attn.ptr;
        let mut qkv = self.buf.qkv.ptr;
        let mut cos = cos;
        let mut sin = sin;
        let mut s = s as i32;
        let mut d = self.m.d as i32;
        let mut hd = self.m.hd as i32;
        let mut scale = 1.0f32 / (self.m.hd as f32).sqrt();
        let mut window = window;
        let mut heads = (self.m.d / self.m.hd) as i32;
        let n = (heads * s).max(1) as usize;
        let warps_per_block = 8usize; // 256 threads
        let grid = ((n as u32).div_ceil(warps_per_block as u32), 1, 1);
        let shared = (warps_per_block * s as usize * 4) as u32;
        let mut params = vec![
            pu64(&mut out),
            pu64(&mut qkv),
            pu64(&mut cos),
            pu64(&mut sin),
            pi32(&mut s),
            pi32(&mut d),
            pi32(&mut hd),
            pf32(&mut scale),
            pi32(&mut window),
            pi32(&mut heads),
        ];
        self.cuda
            .launch(&self.k.attention, grid, (256, 1, 1), shared, &mut params)
    }

    fn add_into_h(&self, h: u64, y: u64, n: usize) -> Result<(), String> {
        let mut x = h;
        let mut y = y;
        let mut n = n as i32;
        let params = vec![pu64(&mut x), pu64(&mut y), pi32(&mut n)];
        self.elem(&self.k.add, params, n as usize)
    }

    /// One complete forward pass. Returns (logits, act) exactly like the CPU
    /// Model::forward: logits has one entry per marker, act has two.
    pub fn forward(
        &mut self,
        ids: &[u32],
        markers: &[u32],
        qtype: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let t0 = Instant::now();
        let s = ids.len();
        let d = self.m.d;
        if s == 0 || s > self.m.max_len || qtype >= 3 || markers.is_empty() || markers.len() > 16 {
            return Err("invalid inference input shape".into());
        }
        if markers.iter().any(|&p| p as usize >= s) {
            return Err("markers exceed sequence".into());
        }
        if ids.iter().any(|&id| id as usize >= self.m.vocab) {
            return Err("token ID out of range".into());
        }
        if self.rope_full_elems < s * self.m.hd || self.rope_slide_elems < s * self.m.hd {
            return Err("RoPE table too short".into());
        }
        let kk = markers.len();
        let mut h2d: u64 = 0;
        let mut d2h: u64 = 0;

        // Per-request uploads: token ids + marker positions only.
        let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.cuda.h2d(self.buf.ids.ptr, &id_bytes)?;
        h2d += id_bytes.len() as u64;
        let mk_bytes: Vec<u8> = markers.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.cuda.h2d(self.buf.markers.ptr, &mk_bytes)?;
        h2d += mk_bytes.len() as u64;

        // h = emb[ids] (F16 gather) ; h = embnorm(h)
        {
            let mut out = self.buf.h_a.ptr;
            let mut emb = self.m.emb;
            let mut idp = self.buf.ids.ptr;
            let mut s_i = s as i32;
            let mut d_i = d as i32;
            let params = vec![
                pu64(&mut out),
                pu64(&mut emb),
                pu64(&mut idp),
                pi32(&mut s_i),
                pi32(&mut d_i),
            ];
            self.elem(&self.k.embed, params, s * d)?;
        }
        self.trace_stage("emb", self.buf.h_a.ptr, s * d)?;
        self.ln(self.buf.h_b.ptr, self.buf.h_a.ptr, &self.m.embnorm, s)?;
        self.trace_stage("embnorm", self.buf.h_b.ptr, s * d)?;
        let (mut h, alt) = (self.buf.h_b.ptr, self.buf.h_a.ptr);

        // Trunk layers.
        for (li, l) in self.m.trunk.iter().enumerate() {
            let x_src = match &l.n1 {
                Some(n1) => {
                    self.ln(self.buf.xn.ptr, h, n1, s)?;
                    self.trace_stage(&format!("t{li}.xn"), self.buf.xn.ptr, s * d)?;
                    self.buf.xn.ptr
                }
                None => h,
            };
            self.mm(self.buf.qkv.ptr, x_src, &l.qkv, s)?;
            self.trace_stage(&format!("t{li}.qkv"), self.buf.qkv.ptr, s * 3 * d)?;
            let (cos, sin) = if l.full {
                (self.m.cos_full, self.m.sin_full)
            } else {
                (self.m.cos_slide, self.m.sin_slide)
            };
            self.attn(cos, sin, if l.full { 0 } else { 64 }, s)?;
            self.trace_stage(&format!("t{li}.attn"), self.buf.attn.ptr, s * d)?;
            self.mm(self.buf.tmp.ptr, self.buf.attn.ptr, &l.wo, s)?;
            self.trace_stage(&format!("t{li}.wo"), self.buf.tmp.ptr, s * d)?;
            self.add_into_h(h, self.buf.tmp.ptr, s * d)?;
            self.trace_stage(&format!("t{li}.h1"), h, s * d)?;
            self.ln(self.buf.xn.ptr, h, &l.n2, s)?;
            self.trace_stage(&format!("t{li}.n2"), self.buf.xn.ptr, s * d)?;
            self.mm(self.buf.up.ptr, self.buf.xn.ptr, &l.wi, s)?;
            self.trace_stage(&format!("t{li}.up"), self.buf.up.ptr, s * l.wi.rows)?;
            let f = l.down.cols;
            {
                let mut m = self.buf.mid.ptr;
                let mut up = self.buf.up.ptr;
                let mut s_i = s as i32;
                let mut f_i = f as i32;
                let params = vec![pu64(&mut m), pu64(&mut up), pi32(&mut s_i), pi32(&mut f_i)];
                self.elem(&self.k.geglu, params, s * f)?;
            }
            self.trace_stage(&format!("t{li}.mid"), self.buf.mid.ptr, s * f)?;
            self.mm(self.buf.tmp.ptr, self.buf.mid.ptr, &l.down, s)?;
            self.add_into_h(h, self.buf.tmp.ptr, s * d)?;
            self.trace_stage(&format!("t{li}.h2"), h, s * d)?;
        }

        // h = finalnorm(h) (ping-pong); h += type_emb[qtype]
        self.ln(alt, h, &self.m.finalnorm, s)?;
        h = alt;
        self.trace_stage("finalnorm", h, s * d)?;
        {
            let mut hh = h;
            let mut te = self.m.type_emb;
            let mut q = qtype as i32;
            let mut s_i = s as i32;
            let mut d_i = d as i32;
            let params = vec![
                pu64(&mut hh),
                pu64(&mut te),
                pi32(&mut q),
                pi32(&mut s_i),
                pi32(&mut d_i),
            ];
            self.elem(&self.k.add_typeemb, params, s * d)?;
        }
        self.trace_stage("typeemb", h, s * d)?;

        // Head layers: full attention, no RoPE; ReLU MLP.
        for (li, l) in self.m.head.iter().enumerate() {
            self.ln(self.buf.xn.ptr, h, l.n1.as_ref().unwrap(), s)?;
            self.trace_stage(&format!("h{li}.xn"), self.buf.xn.ptr, s * d)?;
            self.mm(self.buf.qkv.ptr, self.buf.xn.ptr, &l.qkv, s)?;
            self.attn(0, 0, 0, s)?;
            self.trace_stage(&format!("h{li}.attn"), self.buf.attn.ptr, s * d)?;
            self.mm(self.buf.tmp.ptr, self.buf.attn.ptr, &l.wo, s)?;
            self.add_into_h(h, self.buf.tmp.ptr, s * d)?;
            self.trace_stage(&format!("h{li}.h1"), h, s * d)?;
            self.ln(self.buf.xn.ptr, h, &l.n2, s)?;
            self.mm(self.buf.up.ptr, self.buf.xn.ptr, &l.wi, s)?;
            let f = l.down.cols;
            {
                let mut x = self.buf.up.ptr;
                let mut n = (s * f) as i32;
                let params = vec![pu64(&mut x), pi32(&mut n)];
                self.elem(&self.k.relu, params, s * f)?;
            }
            self.trace_stage(&format!("h{li}.up"), self.buf.up.ptr, s * f)?;
            self.mm(self.buf.tmp.ptr, self.buf.up.ptr, &l.down, s)?;
            self.add_into_h(h, self.buf.tmp.ptr, s * d)?;
            self.trace_stage(&format!("h{li}.h2"), h, s * d)?;
        }

        // Scorer: gather marker rows, scorenorm, score1 + gelu, score2.
        {
            let mut out = self.buf.sc_in.ptr;
            let mut hh = h;
            let mut mk = self.buf.markers.ptr;
            let mut k_i = kk as i32;
            let mut d_i = d as i32;
            let params = vec![
                pu64(&mut out),
                pu64(&mut hh),
                pu64(&mut mk),
                pi32(&mut k_i),
                pi32(&mut d_i),
            ];
            self.elem(&self.k.gather, params, kk * d)?;
        }
        self.trace_stage("gathered", self.buf.sc_in.ptr, kk * d)?;
        self.ln(self.buf.sc_x.ptr, self.buf.sc_in.ptr, &self.m.scorenorm, kk)?;
        self.trace_stage("scorenorm", self.buf.sc_x.ptr, kk * d)?;
        self.mm(self.buf.sc_in.ptr, self.buf.sc_x.ptr, &self.m.score1, kk)?;
        {
            let mut x = self.buf.sc_in.ptr;
            let mut n = (kk * d) as i32;
            let params = vec![pu64(&mut x), pi32(&mut n)];
            self.elem(&self.k.gelu, params, kk * d)?;
        }
        self.trace_stage("score1", self.buf.sc_in.ptr, kk * d)?;
        self.mm(self.buf.logits.ptr, self.buf.sc_in.ptr, &self.m.score2, kk)?;
        self.trace_stage("logits", self.buf.logits.ptr, kk)?;

        // Download logits + h row 0 (for the pooled act vector), then run
        // the tiny tail on the host with the CPU engine's exact op order.
        self.cuda.synchronize()?;
        let mut log_bytes = vec![0u8; kk * 4];
        self.cuda.d2h(self.buf.logits.ptr, &mut log_bytes)?;
        d2h += log_bytes.len() as u64;
        let logits: Vec<f32> = log_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut h0_bytes = vec![0u8; d * 4];
        self.cuda.d2h(h, &mut h0_bytes)?;
        d2h += h0_bytes.len() as u64;
        let h0: Vec<f32> = h0_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // tensor::softmax (serial max/exp/sum/div, CPU op order)
        let mut m = f32::NEG_INFINITY;
        for v in &logits {
            m = m.max(*v);
        }
        let p: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
        let sum: f32 = p.iter().sum();
        let p: Vec<f32> = p.iter().map(|v| v / sum).collect();
        let top1 = p.iter().copied().fold(0f32, f32::max);
        let top2 = p
            .iter()
            .map(|&v| v * (top1 - v) / ((top1 - v) + 1e-6))
            .fold(0f32, f32::max);
        let k = kk.max(2) as f32;
        let ent = -p.iter().map(|v| v * v.max(1e-9).ln()).sum::<f32>() / k.ln();

        // pooled = h[0..d] ++ [top1, top1-top2, ent, k/255]
        let mut pooled = h0;
        pooled.extend_from_slice(&[top1, top1 - top2, ent, k / 255.]);
        let pooled_bytes: Vec<u8> = pooled.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.cuda.h2d(self.buf.pooled.ptr, &pooled_bytes)?;
        h2d += pooled_bytes.len() as u64;

        self.mm(self.buf.ax.ptr, self.buf.pooled.ptr, &self.m.act1, 1)?;
        {
            let mut x = self.buf.ax.ptr;
            let mut n = self.m.act1.rows as i32;
            let params = vec![pu64(&mut x), pi32(&mut n)];
            self.elem(&self.k.gelu, params, self.m.act1.rows)?;
        }
        self.trace_stage("act1", self.buf.ax.ptr, self.m.act1.rows)?;
        self.mm(self.buf.actout.ptr, self.buf.ax.ptr, &self.m.act2, 1)?;
        self.cuda.synchronize()?;
        let mut act_bytes = vec![0u8; 8];
        self.cuda.d2h(self.buf.actout.ptr, &mut act_bytes)?;
        d2h += 8;
        let act: Vec<f32> = act_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        if logits.iter().chain(&act).any(|v| !v.is_finite()) {
            return Err("non-finite model output".into());
        }
        self.last_h2d = h2d;
        self.last_d2h = d2h;
        self.last_gpu_us = t0.elapsed().as_micros();
        Ok((logits, act))
    }
}
