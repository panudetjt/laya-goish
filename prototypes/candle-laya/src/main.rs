mod decision;
mod gguf;
mod laya;

use laya::{BatchReq, Model, WeightMode};
use decision::Answer;
use serde_json::{json, Value};
use std::time::Instant;

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
    // Must precede the first rayon/candle pool use.
    std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    std::env::set_var("CANDLE_NUM_THREADS", n.to_string());
}

fn opts(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn run_case(m: &Model, temps: &decision::Temps, qs: &[CaseQ]) -> Result<Value, String> {
    let mut answers = serde_json::Map::new();
    let mut tokens = 0usize;
    for q in qs {
        tokens += q.ids.len();
        let t0 = Instant::now();
        let (logits, act) = m.forward(&q.ids, &q.markers, q.qtype)?;
        let fwd = t0.elapsed().as_secs_f64() * 1e3;
        let ans = Answer {
            kind: q.qtype,
            labels: q.labels.clone(),
        };
        let v = decision::decide(&logits, &act, &ans, temps);
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
    let usage = "usage: candle-laya info|run|bench|bench-batch MODEL [opts]\n\
                 opts: --case F | --cases F,F,... | --mode f16|convert|upcast\n\
                       --threads N | --warmup N | --repeats N | --batch N";
    if args.len() < 3 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let cmd = args[1].clone();
    let model_path = args[2].clone();
    let mode =
        WeightMode::parse(&opts(&args, "--mode").unwrap_or_else(|| "f16".into()))
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2);
            });
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
    let g = match gguf::Gguf::open(&model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gguf: {e}");
            std::process::exit(1);
        }
    };
    let m = match Model::load(&g, mode) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("model: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "loaded {} layers={} d={} heads={}x{} mode={:?} load_ms={:.0} peak_rss={:.0}MiB",
        model_path,
        m.layers,
        m.d,
        m.heads,
        m.hd,
        mode,
        t0.elapsed().as_secs_f64() * 1e3,
        peak_rss_mib()
    );
    let temps = decision::Temps::load(&g);
    let name = g.meta_str("general.name").to_string();
    let arch = g.meta_str("general.architecture").to_string();
    let vocab = g
        .tensors
        .get("tok_emb.weight")
        .map(|t| t.dims[1])
        .unwrap_or(0);
    let max_len = g.meta_num("laya.max_len", 512);
    let temp_str = g.meta_str("laya.temperature").to_string();
    let temp_buckets_str = g.meta_str("laya.temperature_by_options").to_string();
    drop(g);

    match cmd.as_str() {
        "info" => {
            let full: Vec<usize> = m
                .trunk
                .iter()
                .enumerate()
                .filter(|(_, l)| l.full)
                .map(|(i, _)| i)
                .collect();
            println!(
                "{}",
                json!({
                    "name": name,
                    "architecture": arch,
                    "layers": m.layers,
                    "d": m.d,
                    "heads": m.heads,
                    "head_dim": m.hd,
                    "vocab": vocab,
                    "max_len": max_len,
                    "full_attention_trunk_layers": full,
                    "temperature": serde_json::from_str::<Value>(&temp_str).unwrap_or(Value::Null),
                    "temperature_by_options": serde_json::from_str::<Value>(&temp_buckets_str).unwrap_or(Value::Null),
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
            let out = match run_case(&m, &temps, &qs) {
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
            // One question per iteration, matching the engine's protocol.
            let q = &qs[0];
            let ans = Answer {
                kind: q.qtype,
                labels: q.labels.clone(),
            };
            let mut fwd_ms = Vec::new();
            let mut e2e_ms = Vec::new();
            for i in 0..(warmup + repeats) {
                let t0 = Instant::now();
                let (logits, act) = m.forward(&q.ids, &q.markers, q.qtype).expect("forward");
                let t1 = Instant::now();
                let _ = decision::decide(&logits, &act, &ans, &temps);
                let t2 = Instant::now();
                if i >= warmup {
                    fwd_ms.push((t1 - t0).as_secs_f64() * 1e3);
                    e2e_ms.push((t2 - t0).as_secs_f64() * 1e3);
                }
            }
            println!(
                "{}",
                json!({
                    "case": case, "tokens": q.ids.len(), "mode": format!("{:?}", mode),
                    "threads": std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into()),
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

            // Correctness sanity: batched logits vs single-run logits.
            let singles: Vec<(Vec<f32>, Vec<f32>)> = reqs
                .iter()
                .map(|r| m.forward(&r.ids, &r.markers, r.qtype).expect("single fwd"))
                .collect();
            let batched = m.forward_batch(&reqs).expect("batch fwd");
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
                let out = m.forward_batch(&reqs).expect("batch fwd");
                let t1 = Instant::now();
                for ((l, a), ans) in out.iter().zip(&answers) {
                    decision::decide(l, a, ans, &temps);
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
                    m.forward(&r.ids, &r.markers, r.qtype).expect("fwd");
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
                    "batch": batch, "tokens_total": tokens, "mode": format!("{:?}", mode),
                    "threads": std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into()),
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
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    }
}
