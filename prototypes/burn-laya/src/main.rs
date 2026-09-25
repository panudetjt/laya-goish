mod convert;
mod decision;
mod gguf;
mod laya;
mod weights;

use burn::tensor::backend::BackendTypes;
use decision::{decide, Answer, Temps};
use laya::{BatchReq, Model};
use serde_json::{json, Value};
use std::time::Instant;
use weights::Weights;

type B = burn::backend::Flex;
type W = burn::backend::Wgpu;

struct CaseQ {
    qid: String,
    ids: Vec<u32>,
    markers: Vec<usize>,
    qtype: usize,
    labels: Vec<String>,
}

fn load_case(path: &str) -> Result<Vec<CaseQ>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (qid, ans) in v["answers"].as_object().ok_or("no answers")? {
        let kind = match ans["type"].as_str().unwrap_or("") {
            "choice" => 0,
            "score" => 1,
            "noul" => 2,
            t => return Err(format!("bad type {t}")),
        };
        let ids = ans["input_ids"]
            .as_array()
            .ok_or("no input_ids (use --raw references)")?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as u32)
            .collect();
        let markers = ans["markers"]
            .as_array()
            .ok_or("no markers")?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as usize)
            .collect();
        let labels: Vec<String> = ans["probabilities"]
            .as_object()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        out.push(CaseQ {
            qid: qid.clone(),
            ids,
            markers,
            qtype: kind,
            labels,
        });
    }
    Ok(out)
}

fn rss_field(field: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn peak_rss_mib() -> f64 {
    rss_field("VmHWM:")
}

fn cur_rss_mib() -> f64 {
    rss_field("VmRSS:")
}

fn set_threads(n: usize) {
    // burn-flex elementwise/reduce ops ride the global rayon pool; gemm's
    // rayon backend does too. Must precede the first pool use.
    std::env::set_var("RAYON_NUM_THREADS", n.to_string());
}

fn opts(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn infer_f16(w: &Weights) -> bool {
    matches!(
        w.dtype("qkv.0.weight"),
        Some(safetensors::tensor::Dtype::F16)
    )
}

fn load_model(
    w: &Weights,
    mode: Option<String>,
    emb_chunks: usize,
) -> Result<(Model<B>, bool, String), String> {
    let f16 = match mode.as_deref() {
        Some("f16") | Some("convert") => true,
        Some("f32") | Some("upcast") => false,
        None => infer_f16(w),
        Some(m) => return Err(format!("unknown mode {m}")),
    };
    let mode_str = if f16 { "f16" } else { "f32" }.to_string();
    let device: <B as BackendTypes>::Device = Default::default();
    Ok((Model::load(w, f16, emb_chunks, &device)?, f16, mode_str))
}

fn run_case(
    m: &Model<B>,
    temps: &Temps,
    qs: &[CaseQ],
    device: &<B as BackendTypes>::Device,
) -> Result<Value, String> {
    let mut answers = serde_json::Map::new();
    let mut tokens = 0usize;
    for q in qs {
        tokens += q.ids.len();
        let t0 = Instant::now();
        let (logits, act) = m.forward(&q.ids, &q.markers, q.qtype, device)?;
        let fwd = t0.elapsed().as_secs_f64() * 1e3;
        let ans = Answer {
            kind: q.qtype,
            labels: q.labels.clone(),
        };
        let v = decide(&logits, &act, &ans, temps);
        let mut o = v.as_object().unwrap().clone();
        o.insert("forward_ms".into(), json!(fwd));
        answers.insert(q.qid.clone(), Value::Object(o));
    }
    Ok(json!({"answers": Value::Object(answers), "usage": {"input_tokens": tokens}}))
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: burn-laya convert|info|run|bench|bench-batch|wgpu-probe [MODEL|GGUF] [opts]\n\
                 opts: --case F | --cases F,F,... | --mode f32|f16 (default: file dtype)\n\
                       --threads N | --warmup N | --repeats N | --batch N | --out DIR";
    if args.len() < 2 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let cmd = args[1].clone();

    if cmd == "convert" {
        let gguf_path = args.get(2).cloned().unwrap_or_default();
        let out_dir = opts(&args, "--out").unwrap_or_else(|| ".".into());
        if gguf_path.is_empty() {
            eprintln!("convert GGUF --out DIR");
            std::process::exit(2);
        }
        match convert::convert(&gguf_path, &out_dir) {
            Ok(c) => println!(
                "{}",
                json!({
                    "gguf": gguf_path,
                    "f32_file": c.f32_path, "f32_mib": c.f32_bytes as f64 / 1048576.0,
                    "f16_file": c.f16_path, "f16_mib": c.f16_bytes as f64 / 1048576.0,
                    "convert_ms": c.convert_ms,
                })
            ),
            Err(e) => {
                eprintln!("convert: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    let model_path = args.get(2).cloned().unwrap_or_default();
    if model_path.is_empty() {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    if let Some(t) = opts(&args, "--threads").and_then(|s| s.parse::<usize>().ok()) {
        set_threads(t);
    }
    let warmup: usize = opts(&args, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let repeats: usize = opts(&args, "--repeats")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let t0 = Instant::now();
    let w = match Weights::open(&model_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("weights: {e}");
            std::process::exit(1);
        }
    };
    let device: <B as BackendTypes>::Device = Default::default();
    let emb_chunks: usize = opts(&args, "--emb-chunks")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let (m, f16, mode_str) = match load_model(&w, opts(&args, "--mode"), emb_chunks) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("model: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "loaded {} layers={} d={} heads={}x{} mode={} load_ms={:.0} peak_rss={:.0}MiB",
        model_path,
        m.layers,
        m.d,
        m.heads,
        m.hd,
        mode_str,
        t0.elapsed().as_secs_f64() * 1e3,
        peak_rss_mib()
    );
    let temps = Temps::new(&w.temperature, &w.temperature_by_options);
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());

    match cmd.as_str() {
        "info" => {
            println!(
                "{}",
                json!({
                    "file": model_path,
                    "name": w.manifest["name"],
                    "architecture": w.manifest["architecture"],
                    "layers": m.layers,
                    "d": m.d,
                    "heads": m.heads,
                    "head_dim": m.hd,
                    "vocab": w.manifest["vocab"],
                    "max_len": w.manifest["max_len"],
                    "full_attention_trunk_layers": w.full_layers,
                    "temperature": serde_json::from_str::<Value>(&w.temperature).unwrap_or(Value::Null),
                    "temperature_by_options": serde_json::from_str::<Value>(&w.temperature_by_options).unwrap_or(Value::Null),
                })
            );
        }
        "run" => {
            let case = opts(&args, "--case").unwrap_or_else(|| {
                eprintln!("--case required");
                std::process::exit(2);
            });
            let qs = match load_case(&case) {
                Ok(q) => q,
                Err(e) => {
                    eprintln!("case: {e}");
                    std::process::exit(1);
                }
            };
            let out = match run_case(&m, &temps, &qs, &device) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("forward: {e}");
                    std::process::exit(1);
                }
            };
            println!("{out}");
        }
        "bench" => {
            let case = opts(&args, "--case").unwrap_or_else(|| {
                eprintln!("--case required");
                std::process::exit(2);
            });
            let qs = load_case(&case).expect("case");
            let q = &qs[0];
            let ans = Answer {
                kind: q.qtype,
                labels: q.labels.clone(),
            };
            let mut fwd_ms = Vec::new();
            let mut e2e_ms = Vec::new();
            for i in 0..(warmup + repeats) {
                let t0 = Instant::now();
                let (logits, act) = m.forward(&q.ids, &q.markers, q.qtype, &device).expect("forward");
                let t1 = Instant::now();
                let _ = decide(&logits, &act, &ans, &temps);
                let t2 = Instant::now();
                if i >= warmup {
                    fwd_ms.push((t1 - t0).as_secs_f64() * 1e3);
                    e2e_ms.push((t2 - t0).as_secs_f64() * 1e3);
                }
            }
            println!(
                "{}",
                json!({
                    "case": case, "tokens": q.ids.len(), "mode": mode_str,
                    "threads": threads,
                    "forward_ms": fwd_ms, "e2e_ms": e2e_ms,
                    "forward_mean_ms": mean(&fwd_ms),
                    "forward_min_ms": fwd_ms.iter().cloned().fold(f64::INFINITY, f64::min),
                    "e2e_mean_ms": mean(&e2e_ms),
                    "peak_rss_mib": peak_rss_mib(),
                    "steady_rss_mib": cur_rss_mib(),
                })
            );
        }
        "bench-batch" => {
            let cases = opts(&args, "--cases").unwrap_or_else(|| {
                eprintln!("--cases required");
                std::process::exit(2);
            });
            let batch: usize = opts(&args, "--batch")
                .and_then(|s| s.parse().ok())
                .unwrap_or(4);
            let mut pool = Vec::new();
            for c in cases.split(',') {
                pool.extend(load_case(c).expect("case"));
            }
            if pool.is_empty() {
                eprintln!("no questions loaded");
                std::process::exit(2);
            }
            let reqs: Vec<BatchReq> = (0..batch)
                .map(|i| {
                    let q = &pool[i % pool.len()];
                    BatchReq {
                        ids: q.ids.clone(),
                        markers: q.markers.clone(),
                        qtype: q.qtype,
                    }
                })
                .collect();
            let answers: Vec<Answer> = (0..batch)
                .map(|i| Answer {
                    kind: pool[i % pool.len()].qtype,
                    labels: pool[i % pool.len()].labels.clone(),
                })
                .collect();
            let tokens: usize = reqs.iter().map(|r| r.ids.len()).sum();

            let singles: Vec<(Vec<f32>, Vec<f32>)> = reqs
                .iter()
                .map(|r| m.forward(&r.ids, &r.markers, r.qtype, &device).expect("single fwd"))
                .collect();
            let batched = m.forward_batch(&reqs, &device).expect("batch fwd");
            let mut max_dl = 0f32;
            for (a, b) in singles.iter().zip(&batched) {
                for (x, y) in a.0.iter().zip(&b.0) {
                    max_dl = max_dl.max((x - y).abs());
                }
            }

            let mut batch_fwd = Vec::new();
            let mut batch_e2e = Vec::new();
            for i in 0..(warmup + repeats) {
                let t0 = Instant::now();
                let out = m.forward_batch(&reqs, &device).expect("batch fwd");
                let t1 = Instant::now();
                for ((l, a), ans) in out.iter().zip(&answers) {
                    decide(l, a, ans, &temps);
                }
                let t2 = Instant::now();
                if i >= warmup {
                    batch_fwd.push((t1 - t0).as_secs_f64() * 1e3);
                    batch_e2e.push((t2 - t0).as_secs_f64() * 1e3);
                }
            }
            let mut seq_fwd = Vec::new();
            for i in 0..(warmup + repeats) {
                let t0 = Instant::now();
                for r in &reqs {
                    m.forward(&r.ids, &r.markers, r.qtype, &device).expect("fwd");
                }
                let t1 = Instant::now();
                if i >= warmup {
                    seq_fwd.push((t1 - t0).as_secs_f64() * 1e3);
                }
            }
            let bf = mean(&batch_fwd);
            let sf = mean(&seq_fwd);
            println!(
                "{}",
                json!({
                    "batch": batch, "tokens_total": tokens, "mode": mode_str,
                    "threads": threads,
                    "batch_fwd_ms": batch_fwd, "batch_fwd_mean_ms": bf,
                    "batch_e2e_mean_ms": mean(&batch_e2e),
                    "batch_req_per_s": batch as f64 / (bf / 1e3),
                    "seq_fwd_ms": seq_fwd, "seq_fwd_mean_ms": sf,
                    "seq_req_per_s": batch as f64 / (sf / 1e3),
                    "speedup_batch_vs_seq": sf / bf,
                    "batched_vs_single_max_dlogit": max_dl,
                    "peak_rss_mib": peak_rss_mib(),
                })
            );
        }
        "wgpu-probe" => {
            wgpu_probe(&args, warmup, repeats);
        }
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    }
    let _ = f16;
}

// wgpu leg: enumerate the adapters burn's own wgpu stack can see on this
// machine, then run the forward on the burn-wgpu backend whatever the
// adapter turns out to be (llvmpipe expected on this WSL2 deployment).
fn wgpu_probe(args: &[String], warmup: usize, repeats: usize) {
    let case = opts(args, "--case").expect("--case required");
    let adapters = enumerate_wgpu_adapters();
    for a in &adapters {
        eprintln!(
            "wgpu adapter: {} | {} | {:?} | backend {:?}",
            a.0, a.1, a.2, a.3
        );
    }

    let model_path = args.get(2).cloned().unwrap_or_default();
    let w = Weights::open(&model_path).expect("weights");
    // f32 compute on wgpu: the f16 path trips a Fusion-IR DTypeMismatch on
    // this backend build, and f32 is the conservative stock configuration.
    let f16 = match opts(args, "--mode").as_deref() {
        Some("f16") => true,
        _ => false,
    };
    let t0 = Instant::now();
    let device: <W as BackendTypes>::Device = Default::default();
    let m: Model<W> = Model::load(&w, f16, 4, &device).expect("model");
    eprintln!(
        "wgpu model loaded mode={} load_ms={:.0}",
        if f16 { "f16" } else { "f32" },
        t0.elapsed().as_secs_f64() * 1e3
    );
    let temps = Temps::new(&w.temperature, &w.temperature_by_options);

    let qs = load_case(&case).expect("case");
    let q = &qs[0];
    let t0 = Instant::now();
    let (logits, act) = m.forward(&q.ids, &q.markers, q.qtype, &device).expect("forward");
    let first = t0.elapsed().as_secs_f64() * 1e3;
    let ans = Answer {
        kind: q.qtype,
        labels: q.labels.clone(),
    };
    let v = decide(&logits, &act, &ans, &temps);
    let mut fwd_ms = Vec::new();
    for i in 0..(warmup + repeats) {
        let t0 = Instant::now();
        let (l, a) = m.forward(&q.ids, &q.markers, q.qtype, &device).expect("forward");
        let t1 = Instant::now();
        let _ = decide(&l, &a, &ans, &temps);
        if i >= warmup {
            fwd_ms.push((t1 - t0).as_secs_f64() * 1e3);
        }
    }
    println!(
        "{}",
        json!({
            "case": case, "tokens": q.ids.len(), "mode": if f16 { "f16" } else { "f32" },
            "adapters": adapters.iter().map(|a| json!({
                "vendor": a.0, "name": a.1, "device_type": a.2, "backend": a.3,
            })).collect::<Vec<_>>(),
            "first_forward_ms": first,
            "forward_ms": fwd_ms,
            "forward_mean_ms": mean(&fwd_ms),
            "answer": v,
        })
    );
}

type AdapterInfo = (String, String, String, String);

fn enumerate_wgpu_adapters() -> Vec<AdapterInfo> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::all();
    let instance = wgpu::Instance::new(desc);
    let mut out = Vec::new();
    for a in pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all())) {
        let i = a.get_info();
        out.push((
            format!("{:#x}", i.vendor),
            i.name.clone(),
            format!("{:?}", i.device_type),
            format!("{:?}", i.backend),
        ));
    }
    out
}
