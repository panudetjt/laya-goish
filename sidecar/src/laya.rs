//! Laya model loading for the GPU sidecar. This walks ggmlc.graph_spec
//! EXACTLY as the CPU engine does (src/models/laya/model.rs): norms by
//! opcode-201 node name, linears by opcode-29 with their first input being
//! the weight tensor, folded biases from a third linear input plus any
//! opcode-200 (fused bias/GELU) consumer, and full-vs-sliding attention from
//! the attention nodes' mask input identity.
//!
//! The whole GGUF tensor data section is uploaded verbatim into one device
//! region; kernels address weights (F16), norms/rope tables (F32) and the
//! embedding table at their file offsets. Only the folded bias SUMS (where
//! the CPU engine adds several vectors) are computed on the host and
//! uploaded to a small auxiliary buffer, in the CPU engine's summation order.
use crate::cuda::{Cuda, DevMem};
use crate::gguf::{graph_tensor_name, jget, GgufFile, OP_ATTENTION, OP_FUSED_BIAS, OP_LAYER_NORM, OP_LINEAR};
use serde_json::Value as J;
use serde_json::Map;

pub struct Linear {
    /// Device pointer to the weight rows [rows][cols] inside the region.
    /// GGML kind of the weight tensor: 1 = F16 rows, 0 = F32 rows
    /// (scorer_fc2 is the one F32 linear in the Laya checkpoint).
    pub w: u64,
    pub kind: u32,
    pub rows: usize,
    pub cols: usize,
    /// Element offset of the summed f32 bias in the aux buffer (None = no bias).
    pub bias_off: Option<usize>,
}

pub struct Norm {
    pub w: u64, // 0 = none
    pub b: u64, // 0 = none
    pub eps: f32,
}

pub struct Layer {
    pub n1: Option<Norm>,
    pub qkv: Linear,
    pub wo: Linear,
    pub n2: Norm,
    pub wi: Linear,
    pub down: Linear,
    pub full: bool,
    pub is_head: bool,
}

#[allow(dead_code)]
pub struct LayaGpu {
    pub layers: usize,
    pub d: usize,
    pub hd: usize,
    pub max_len: usize,
    pub vocab: usize,
    pub emb: u64, // F16 [vocab][d]
    pub embnorm: Norm,
    pub finalnorm: Norm,
    pub trunk: Vec<Layer>,
    pub head: Vec<Layer>,
    pub type_emb: u64, // F16 [3][d]
    pub scorenorm: Norm,
    pub score1: Linear,
    pub score2: Linear,
    pub act1: Linear,
    pub act2: Linear,
    pub cos_full: u64,
    pub sin_full: u64,
    pub cos_slide: u64,
    pub sin_slide: u64,
    pub rope_full_elems: usize,
    pub rope_slide_elems: usize,
    /// The uploaded data section (kept alive; all u64 pointers above point
    /// into it).
    pub region: DevMem,
    /// Folded-bias f32 buffer.
    pub aux: DevMem,
}

impl LayaGpu {
    /// Resolve a linear's bias device pointer (0 = none).
    pub fn bias_ptr(&self, l: &Linear) -> u64 {
        l.bias_off
            .map(|off| self.aux.ptr + (off * 4) as u64)
            .unwrap_or(0)
    }
}

fn linear(
    g: &GgufFile,
    graph: &J,
    tensors: &Map<String, J>,
    key: &str,
    region: &DevMem,
    aux: &mut Vec<f32>,
) -> Result<Linear, String> {
    let weight = format!("{key}.weight");
    let (t, _) = g.data(&weight)?;
    let cols = t.dims[0];
    let rows = *t.dims.get(1).unwrap_or(&1);
    if t.dims.iter().skip(2).any(|&d| d != 1) {
        return Err("non-matrix linear".into());
    }
    let nodes = jget(graph, "nodes")
        .as_array()
        .ok_or("graph nodes missing")?;
    let node = nodes
        .iter()
        .find(|n| {
            jget(n, "opcode").as_f64() == Some(OP_LINEAR)
                && n.get("inputs")
                    .and_then(|i| i.as_array())
                    .and_then(|a| a.first())
                    .map(|i| graph_tensor_name(tensors, i) == weight)
                    .unwrap_or(false)
        })
        .ok_or_else(|| format!("graph lacks linear {key}"))?;
    let mut biases: Vec<String> = Vec::new();
    let ins = node.get("inputs").and_then(|v| v.as_array()).unwrap();
    if ins.len() > 2 {
        biases.push(graph_tensor_name(tensors, &ins[2]).to_string());
    }
    for n in nodes {
        if jget(n, "opcode").as_f64() == Some(OP_FUSED_BIAS) {
            let n_ins = n.get("inputs").and_then(|v| v.as_array()).unwrap();
            let n_outs = node.get("outputs").and_then(|v| v.as_array()).unwrap();
            if n_ins.first() == n_outs.first() {
                biases.push(graph_tensor_name(tensors, &n_ins[1]).to_string());
            }
        }
    }
    let bias_off = if biases.is_empty() {
        None
    } else {
        // CPU engine: start at zero and add each folded vector in order.
        let mut out = vec![0f32; rows];
        for name in biases {
            let v = g.floats(&name)?;
            if v.len() != rows {
                return Err("linear bias shape mismatch".into());
            }
            for (a, b) in out.iter_mut().zip(&v) {
                *a += *b;
            }
        }
        let off = aux.len();
        aux.extend_from_slice(&out);
        Some(off)
    };
    Ok(Linear {
        w: region.ptr + (t.offset - g.data_base) as u64,
        kind: t.kind,
        rows,
        cols,
        bias_off,
    })
}

fn norm(
    g: &GgufFile,
    graph: &J,
    tensors: &Map<String, J>,
    region: &DevMem,
    i: usize,
) -> Result<Norm, String> {
    let key = if i == 0 {
        "layer_norm".to_string()
    } else {
        format!("layer_norm_{i}")
    };
    let nodes = jget(graph, "nodes").as_array().ok_or("graph nodes missing")?;
    let node = nodes
        .iter()
        .find(|n| {
            jget(n, "name").as_str() == Some(key.as_str()) && jget(n, "opcode").as_f64() == Some(OP_LAYER_NORM)
        })
        .ok_or_else(|| format!("missing {key}"))?;
    let ins = node.get("inputs").and_then(|v| v.as_array()).unwrap();
    let dev = |idx: usize| -> Result<u64, String> {
        if idx < ins.len() {
            let name = graph_tensor_name(tensors, &ins[idx]);
            let (t, _) = g.data(name)?;
            Ok(region.ptr + (t.offset - g.data_base) as u64)
        } else {
            Ok(0)
        }
    };
    let eps = jget(jget(node, "attributes"), "eps")
        .as_f64()
        .unwrap_or(0.0) as f32;
    if !(eps > 0.0 && eps < 0.1) {
        return Err("invalid layer norm epsilon".into());
    }
    Ok(Norm {
        w: dev(1)?,
        b: dev(2)?,
        eps,
    })
}

fn tensor_dev(g: &GgufFile, region: &DevMem, name: &str) -> Result<u64, String> {
    let (t, _) = g.data(name)?;
    Ok(region.ptr + (t.offset - g.data_base) as u64)
}

impl LayaGpu {
    pub fn load(cuda: &Cuda, g: &GgufFile) -> Result<LayaGpu, String> {
        if g.get("general.architecture").as_str() != Some("ggmlc")
            || !g
                .get("laya.checkpoint")
                .as_str()
                .unwrap_or("")
                .contains("laya")
        {
            return Err("expected a compiled Laya GGUF".into());
        }
        if g.num("laya.max_opts", 16) != 16 {
            return Err("expected 16-option Laya head".into());
        }
        let graph = g.graph()?;
        let tensors: Map<String, J> = jget(&graph, "tensors")
            .as_object()
            .cloned()
            .ok_or("graph tensors missing")?;
        let nodes = jget(&graph, "nodes").as_array().ok_or("graph nodes missing")?;

        let mut layers = 0;
        while g.tensors.contains_key(&format!("qkv.{layers}.weight")) {
            layers += 1;
        }
        let mut heads = 0;
        while g.tensors.contains_key(&format!("head_qkv.{heads}.weight")) {
            heads += 1;
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
        let attns: Vec<&J> = nodes
            .iter()
            .filter(|n| jget(n, "opcode").as_f64() == Some(OP_ATTENTION))
            .collect();
        if attns.len() != layers + heads {
            return Err("unexpected attention graph".into());
        }
        let fullmask = attns[0]
            .get("inputs")
            .and_then(|v| v.as_array())
            .and_then(|a| a.last())
            .cloned()
            .ok_or("attention has no mask")?;

        // Upload the whole tensor data section first; all pointers below
        // reference it. This is the one-time PCIe transfer: no per-request
        // weight traffic afterwards.
        let region_len = g.bytes.len() - g.data_base;
        let region = cuda.alloc(region_len)?;
        cuda.upload(&region, &g.bytes[g.data_base..])?;

        let mut aux: Vec<f32> = Vec::new();
        let mut mk_linear = |key: &str| linear(g, &graph, &tensors, key, &region, &mut aux);
        let mk_norm = |i: usize| norm(g, &graph, &tensors, &region, i);

        let mut trunk = Vec::new();
        for i in 0..layers {
            trunk.push(Layer {
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
            head.push(Layer {
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
        for l in trunk.iter().chain(head.iter()) {
            // Trunk MLP is GeGLU (wi rows = 2x down cols); head MLP is
            // plain ReLU (wi rows = down cols).
            let mlp_fanout = if l.is_head { 1 } else { 2 };
            if l.qkv.cols != d
                || l.qkv.rows != 3 * d
                || l.wo.cols != d
                || l.wo.rows != d
                || l.wi.cols != d
                || l.down.rows != d
                || l.wi.rows != l.down.cols * mlp_fanout
            {
                return Err("inconsistent layer dimensions".into());
            }
        }
        if g.floats("type_emb.weight")?.len() != 3 * d {
            return Err("type embedding shape mismatch".into());
        }
        let score1 = mk_linear("scorer_fc1")?;
        let score2 = mk_linear("scorer_fc2")?;
        let act1 = mk_linear("act_fc1")?;
        let act2 = mk_linear("act_fc2")?;
        if score1.cols != d
            || score1.rows != d
            || score2.cols != d
            || score2.rows != 1
            || act1.cols != d + 4
            || act2.cols != act1.rows
            || act2.rows != 2
        {
            return Err("decision head shape mismatch".into());
        }
        let aux_mem = {
            let mem = cuda.alloc(aux.len() * 4)?;
            let bytes: Vec<u8> = aux.iter().flat_map(|v| v.to_le_bytes()).collect();
            cuda.upload(&mem, &bytes)?;
            mem
        };
        Ok(LayaGpu {
            layers,
            d,
            hd,
            max_len: g.num("laya.max_len", 512),
            vocab,
            emb: tensor_dev(g, &region, "tok_emb.weight")?,
            embnorm: mk_norm(0)?,
            finalnorm: mk_norm(2 * layers)?,
            trunk,
            head,
            type_emb: tensor_dev(g, &region, "type_emb.weight")?,
            scorenorm: mk_norm(2 * layers + 1 + 2 * heads)?,
            score1,
            score2,
            act1,
            act2,
            cos_full: tensor_dev(g, &region, "cos_full")?,
            sin_full: tensor_dev(g, &region, "sin_full")?,
            cos_slide: tensor_dev(g, &region, "cos_slide")?,
            sin_slide: tensor_dev(g, &region, "sin_slide")?,
            rope_full_elems: g.tensors.get("cos_full").map(|t| t.len / 4).unwrap_or(0),
            rope_slide_elems: g.tensors.get("cos_slide").map(|t| t.len / 4).unwrap_or(0),
            region,
            aux: aux_mem,
        })
    }
}

/// Upload a second model's tensor data section purely for VRAM residency
/// (ticket gate: both F16 models stay resident; no forward pass is ported
/// for it in this PoC).
pub struct Resident {
    pub name: String,
    pub region: DevMem,
}

pub fn load_resident(cuda: &Cuda, g: &GgufFile, path: &str) -> Result<Resident, String> {
    let name = g.get("general.name").as_str().unwrap_or(path).to_string();
    let region_len = g.bytes.len() - g.data_base;
    let region = cuda.alloc(region_len)?;
    cuda.upload(&region, &g.bytes[g.data_base..])?;
    Ok(Resident { name, region })
}
