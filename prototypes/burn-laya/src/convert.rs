// GGUF -> safetensors converter for the Laya checkpoint.
//
// Burn has no GGUF reader (upstream issue #1187), so this is the migration
// cost the ticket asks to measure: parse the GGUF (crate::gguf, same std port
// the candle prototype uses), resolve the ggmlc graph exactly like
// src/models/laya/model.rs once, and write a safetensors file whose tensors
// are already in row-major [rows, cols] linear layout with folded biases:
//
//   <key>.weight  F16 or F32  [out, in]   (row-major)
//   <key>.bias    F32         [out]       (third input + opcode-200 fused
//                                          consumers, summed in engine order)
//   layer_norm{i}(.weight|.bias)?  F32    (only embnorm/finalnorm carry
//                                          affines; eps lives in the manifest)
//   tok_emb.weight [vocab, d], type_emb.weight [3, d],
//   cos_full/sin_full/cos_slide/sin_slide [513, hd]
//
// Model shape facts (full-attention trunk layers, eps, dims) are stored in
// the safetensors `__metadata__` block as JSON strings.
use crate::gguf::Gguf;
use safetensors::tensor::Dtype;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

fn tensor_name(graph: &Value, id: &Value) -> String {
    graph["tensors"][id.to_string()]["name"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

// Graph-resolved bias names for a linear, engine order: the matmul node's
// third input first, then every opcode-200 (fused add) consumer of the
// node's output. Mirrors prototypes/candle-laya/src/laya.rs::linear.
fn bias_names(graph: &Value, key: &str) -> Result<Vec<String>, String> {
    let weight = format!("{key}.weight");
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
    Ok(biases)
}

fn norm_node(_g: &Gguf, graph: &Value, i: usize) -> Result<Value, String> {
    let key = if i == 0 {
        "layer_norm".to_string()
    } else {
        format!("layer_norm_{i}")
    };
    graph["nodes"]
        .as_array()
        .ok_or("graph nodes missing")?
        .iter()
        .find(|n| n["name"].as_str() == Some(key.as_str()) && n["opcode"].as_f64() == Some(201.0))
        .cloned()
        .ok_or_else(|| format!("missing {key}"))
}

pub struct Converted {
    pub f32_path: String,
    pub f16_path: String,
    pub convert_ms: f64,
    pub f32_bytes: u64,
    pub f16_bytes: u64,
}

pub fn convert(gguf_path: &str, out_dir: &str) -> Result<Converted, String> {
    let t0 = Instant::now();
    let g = Gguf::open(gguf_path).map_err(|e| e.to_string())?;
    if g.meta_str("general.architecture") != "ggmlc"
        || !g.meta_str("laya.checkpoint").contains("laya")
    {
        return Err("expected a compiled Laya GGUF".into());
    }
    let graph: Value =
        serde_json::from_str(g.meta_str("ggmlc.graph_spec")).map_err(|e| e.to_string())?;

    // Same layout discovery as the candle prototype.
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
    let d = g.tensors.get("tok_emb.weight").ok_or("missing embedding")?.dims[0];
    let hd = g.tensors.get("cos_full").ok_or("missing RoPE tables")?.dims[0];

    // Full-vs-sliding attention from mask identity (attn opcode 74, last input).
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
        &attns[0]["inputs"].as_array().and_then(|a| a.last()).unwrap(),
    );
    let mut full = Vec::new();
    for i in 0..layers {
        full.push(
            tensor_name(
                &graph,
                &attns[i]["inputs"].as_array().and_then(|a| a.last()).unwrap(),
            ) == fullmask,
        );
    }

    // Linear keys with every weight the forward needs.
    let mut lin_keys = Vec::new();
    for i in 0..layers {
        for k in ["qkv", "wo", "mlp_wi", "mlp_wo"] {
            lin_keys.push(format!("{k}.{i}"));
        }
    }
    for i in 0..heads {
        for k in ["head_qkv", "head_wo", "head_fc1", "head_fc2"] {
            lin_keys.push(format!("{k}.{i}"));
        }
    }
    for k in ["scorer_fc1", "scorer_fc2", "act_fc1", "act_fc2"] {
        lin_keys.push(k.to_string());
    }

    // Norm eps map (all 1e-5 in this checkpoint, but read per node like the engine).
    let mut eps = serde_json::Map::new();
    let mut norm_affines = Vec::new();
    for i in 0..(2 * layers + 1 + 2 * heads + 1) {
        let node = norm_node(&g, &graph, i)?;
        let e = node["attributes"]["eps"].as_f64().unwrap_or(0.0);
        if !(e > 0.0 && e < 0.1) {
            return Err("invalid layer norm epsilon".into());
        }
        let name = node["name"].as_str().unwrap_or("").to_string();
        eps.insert(name.clone(), json!(e));
        let ins = node["inputs"].as_array().unwrap();
        if ins.len() > 1 {
            norm_affines.push(name);
        }
    }

    // ---- Emit tensors -----------------------------------------------------
    // name -> (dtype, shape, bytes little-endian)
    let mut f32_tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut f16_tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();

    fn push_f32(
        name: String,
        v: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<(String, Dtype, Vec<usize>, Vec<u8>), String> {
        if shape.iter().product::<usize>() != v.len() {
            return Err(format!("tensor {name}: shape/len mismatch"));
        }
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in &v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        Ok((name, Dtype::F32, shape, bytes))
    }

    // Linear weights + folded biases. GGUF dims [cols, rows] (fastest first):
    // rows = out features, cols = in features; flat data is rows of cols.
    for key in &lin_keys {
        let wname = format!("{key}.weight");
        let t = g
            .tensors
            .get(&wname)
            .ok_or_else(|| format!("missing {wname}"))?;
        let cols = t.dims[0];
        let rows = *t.dims.get(1).unwrap_or(&1);
        let v = g.floats(&wname).map_err(|e| e.to_string())?;
        if v.len() != rows * cols {
            return Err("linear weight element mismatch".into());
        }
        let mut f16b = Vec::with_capacity(v.len() * 2);
        for x in &v {
            f16b.extend_from_slice(&half::f16::from_f32(*x).to_le_bytes());
        }
        f16_tensors.push((wname.clone(), Dtype::F16, vec![rows, cols], f16b));
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in &v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        f32_tensors.push((wname, Dtype::F32, vec![rows, cols], bytes));

        let biases = bias_names(&graph, key)?;
        if !biases.is_empty() {
            let mut out = vec![0f32; rows];
            for name in biases {
                let bv = g.floats(&name).map_err(|e| e.to_string())?;
                if bv.len() != rows {
                    return Err("linear bias shape mismatch".into());
                }
                for (a, b) in out.iter_mut().zip(&bv) {
                    *a += b;
                }
            }
            let n = out.len();
            let mut bytes = Vec::with_capacity(n * 4);
            for x in &out {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            f16_tensors.push((format!("{key}.bias"), Dtype::F32, vec![n], bytes.clone()));
            f32_tensors.push((format!("{key}.bias"), Dtype::F32, vec![n], bytes));
        }
    }

    // Embeddings. tok_emb GGUF dims [d, vocab]; data is vocab rows of d.
    let embt = g.tensors.get("tok_emb.weight").unwrap();
    let vocab = embt.dims[1];
    let embv = g.floats("tok_emb.weight").map_err(|e| e.to_string())?;
    let mut f16b = Vec::with_capacity(embv.len() * 2);
    for x in &embv {
        f16b.extend_from_slice(&half::f16::from_f32(*x).to_le_bytes());
    }
    f16_tensors.push(("tok_emb.weight".into(), Dtype::F16, vec![vocab, d], f16b));
    f32_tensors.push(push_f32("tok_emb.weight".into(), embv, vec![vocab, d])?);
    let te = g.floats("type_emb.weight").map_err(|e| e.to_string())?;
    let t = push_f32("type_emb.weight".into(), te, vec![3, d])?;
    f16_tensors.push(t.clone());
    f32_tensors.push(t);

    // RoPE tables: flat F32 vec, [pos, hd] rows.
    for name in ["cos_full", "sin_full", "cos_slide", "sin_slide"] {
        let v = g.floats(name).map_err(|e| e.to_string())?;
        let pos = v.len() / hd;
        let t = push_f32(name.into(), v, vec![pos, hd])?;
            f16_tensors.push(t.clone());
            f32_tensors.push(t);
    }

    // Norm affines (only embnorm/finalnorm in this checkpoint).
    for name in &norm_affines {
        // candle norm(): input 1 = weight, input 2 = bias.
        let node = graph["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["name"].as_str() == Some(name.as_str()))
            .ok_or("norm node vanished")?;
        let ins = node["inputs"].as_array().unwrap();
        if ins.len() > 1 {
            let w = g
                .floats(&tensor_name(&graph, &ins[1]))
                .map_err(|e| e.to_string())?;
            let t = push_f32(format!("{name}.weight"), w, vec![d])?;
                f16_tensors.push(t.clone());
                f32_tensors.push(t);
        }
        if ins.len() > 2 {
            let b = g
                .floats(&tensor_name(&graph, &ins[2]))
                .map_err(|e| e.to_string())?;
            let t = push_f32(format!("{name}.bias"), b, vec![d])?;
                f16_tensors.push(t.clone());
                f32_tensors.push(t);
        }
    }

    // Manifest in safetensors __metadata__.
    let meta = json!({
        "architecture": g.meta_str("general.architecture"),
        "name": g.meta_str("general.name"),
        "layers": layers,
        "head_layers": heads,
        "d": d,
        "hd": hd,
        "heads": d / hd,
        "vocab": vocab,
        "max_len": g.meta_num("laya.max_len", 512),
        "full_attention_trunk_layers": (0..layers).filter(|&i| full[i]).collect::<Vec<_>>(),
        "norm_eps": Value::Object(eps),
        "temperature": g.meta_str("laya.temperature"),
        "temperature_by_options": g.meta_str("laya.temperature_by_options"),
    })
    .to_string();

    let f32_path = format!("{out_dir}/laya_english_f32.safetensors");
    let f16_path = format!("{out_dir}/laya_english_f16.safetensors");
    write_safetensors(&f32_tensors, &f32_path, &meta)?;
    write_safetensors(&f16_tensors, &f16_path, &meta)?;
    let f32_bytes = std::fs::metadata(&f32_path).map_err(|e| e.to_string())?.len();
    let f16_bytes = std::fs::metadata(&f16_path).map_err(|e| e.to_string())?.len();
    Ok(Converted {
        f32_path,
        f16_path,
        convert_ms: t0.elapsed().as_secs_f64() * 1e3,
        f32_bytes,
        f16_bytes,
    })
}

fn write_safetensors(
    tensors: &[(String, Dtype, Vec<usize>, Vec<u8>)],
    path: &str,
    manifest: &str,
) -> Result<(), String> {
    use safetensors::serialize_to_file;
    use safetensors::tensor::TensorView;
    let views: Vec<(String, TensorView)> = tensors
        .iter()
        .map(|(name, dt, shape, bytes)| {
            let v = TensorView::new(*dt, shape.clone(), bytes.as_slice())
                .map_err(|e| format!("view {name}: {e:?}"))?;
            Ok((name.clone(), v))
        })
        .collect::<Result<_, String>>()?;
    let mut seen = std::collections::HashSet::new();
    for (n, _) in &views {
        if !seen.insert(n.clone()) {
            return Err(format!("duplicate tensor name {n}"));
        }
    }
    let mut header = HashMap::new();
    header.insert("manifest".to_string(), manifest.to_string());
    serialize_to_file(views, Some(header), std::path::Path::new(path))
        .map_err(|e| format!("write {path}: {e:?}"))
}
