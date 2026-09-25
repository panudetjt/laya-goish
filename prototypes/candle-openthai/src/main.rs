mod decision;
mod gguf;
mod openthai;

use openthai::{GdnImpl, Model, Req, WeightMode};
use serde_json::{json, Value};
use std::time::Instant;

struct CaseQ {
    qid: String,
    kind: usize,
    k: usize,
    labels: Vec<String>,
}

struct Case {
    qs: Vec<CaseQ>,
    // One (ids, positions) per permutation pass.
    passes: Vec<(Vec<u32>, Vec<usize>)>,
    ref_logits: Vec<Vec<f32>>,
}

fn load_case(path: &str) -> Result<Case, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut qs = Vec::new();
    for (qid, ans) in v["answers"].as_object().ok_or("no answers")? {
        let kind = match ans["type"].as_str().unwrap_or("") {
            "choice" => 0,
            "score" => 1,
            "noul" => 2,
            t => return Err(format!("bad type {t}")),
        };
        let labels: Vec<String> = ans["probabilities"]
            .as_object()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        let k = if kind == 2 { 2 } else { labels.len() };
        qs.push(CaseQ {
            qid: qid.clone(),
            kind,
            k,
            labels,
        });
    }
    let mut passes = Vec::new();
    let mut ref_logits = Vec::new();
    for p in v["_raw"].as_array().ok_or("no _raw (use --raw references)")? {
        let ids = p["input_ids"]
            .as_array()
            .ok_or("no input_ids")?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as u32)
            .collect();
        let positions = p["answer_positions"]
            .as_array()
            .ok_or("no answer_positions")?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as usize)
            .collect();
        let logits = p["logits"]
            .as_array()
            .ok_or("no logits")?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();
        passes.push((ids, positions));
        ref_logits.push(logits);
    }
    Ok(Case {
        qs,
        passes,
        ref_logits,
    })
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

fn run_case(m: &Model, case: &Case, gdn: GdnImpl) -> Result<Value, String> {
    let qids: Vec<String> = case.qs.iter().map(|q| q.qid.clone()).collect();
    let mut logits_per_pass = Vec::new();
    let mut fwd_ms = Vec::new();
    for (ids, positions) in &case.passes {
        let t0 = Instant::now();
        let logits = m
            .forward_batch(
                &[Req {
                    ids: ids.clone(),
                    positions: positions.clone(),
                }],
                gdn,
            )
            .expect("forward")
            .remove(0);
        fwd_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        logits_per_pass.push(logits);
    }
    let answers = decision::decide(&qids, &case.qs_summary(), &logits_per_pass, &m.temperature);
    let tokens = case.passes[0].0.len();
    let max_dl = case
        .ref_logits
        .iter()
        .zip(&logits_per_pass)
        .map(|(r, g)| r.iter().zip(g).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max))
        .fold(0f32, f32::max);
    Ok(json!({
        "answers": answers,
        "usage": {
            "input_tokens": tokens,
            "passes": case.passes.len(),
            "forward_ms": fwd_ms,
            "max_dlogit_vs_raw": max_dl,
        },
    }))
}

impl Case {
    fn qs_summary(&self) -> Vec<decision::Question> {
        self.qs
            .iter()
            .map(|q| decision::Question {
                kind: q.kind,
                k: q.k,
                labels: q.labels.clone(),
            })
            .collect()
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: candle-openthai info|run|bench|bench-batch MODEL [opts]\n\
                 opts: --case F | --cases F,F,... | --mode f16|convert|upcast\n\
                       --threads N | --warmup N | --repeats N | --batch N\n\
                       --gdn scalar|tensor";
    if args.len() < 3 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let cmd = args[1].clone();
    let model_path = args[2].clone();
    let mode = WeightMode::parse(&opts(&args, "--mode").unwrap_or_else(|| "f16".into()))
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2);
        });
    let gdn = match opts(&args, "--gdn").unwrap_or_else(|| "scalar".into()).as_str() {
        "tensor" => GdnImpl::Tensor,
        _ => GdnImpl::Scalar,
    };
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
    let name = g.meta_str("general.name").to_string();
    let arch = g.meta_str("general.architecture").to_string();
    drop(g);
    eprintln!(
        "loaded {} layers={} full_layers={:?} mode={:?} load_ms={:.0} peak_rss={:.0}MiB",
        model_path,
        m.layers,
        m.full_layers,
        mode,
        t0.elapsed().as_secs_f64() * 1e3,
        peak_rss_mib()
    );

    match cmd.as_str() {
        "info" => {
            println!(
                "{}",
                json!({
                    "name": name,
                    "architecture": arch,
                    "layers": m.layers,
                    "full_attention_layers": m.full_layers,
                    "linear_attention_layers": m.layers - m.full_layers.len(),
                    "hidden": openthai::D,
                    "gdn_heads": openthai::LIN_HEADS,
                    "gdn_dim": openthai::LIN_D,
                    "gqa_heads": openthai::FULL_HEADS,
                    "gqa_kv_heads": openthai::KV_HEADS,
                    "gqa_head_dim": openthai::FULL_HD,
                    "max_tokens": m.max_tokens,
                    "max_state": m.max_state,
                    "temperature": m.temperature,
                })
            );
        }
        "run" => {
            let case = opts(&args, "--case").unwrap_or_else(|| {
                eprintln!("--case required");
                std::process::exit(2);
            });
            let c = match load_case(&case) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("case: {e}");
                    std::process::exit(1);
                }
            };
            let out = match run_case(&m, &c, gdn) {
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
            let c = load_case(&case).expect("case");
            let qids: Vec<String> = c.qs.iter().map(|q| q.qid.clone()).collect();
            let summary = c.qs_summary();
            let mut fwd_ms = Vec::new();
            let mut e2e_ms = Vec::new();
            for i in 0..(warmup + repeats) {
                let t0 = Instant::now();
                let mut logits = Vec::new();
                for (ids, positions) in &c.passes {
                    logits.push(
                        m.forward_batch(
                            &[Req {
                                ids: ids.clone(),
                                positions: positions.clone(),
                            }],
                            gdn,
                        )
                        .expect("forward")
                        .remove(0),
                    );
                }
                let t1 = Instant::now();
                let _ = decision::decide(&qids, &summary, &logits, &m.temperature);
                let t2 = Instant::now();
                if i >= warmup {
                    fwd_ms.push((t1 - t0).as_secs_f64() * 1e3);
                    e2e_ms.push((t2 - t0).as_secs_f64() * 1e3);
                }
            }
            println!(
                "{}",
                json!({
                    "case": case, "tokens": c.passes[0].0.len(), "passes": c.passes.len(),
                    "mode": format!("{:?}", mode), "gdn": format!("{:?}", gdn),
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
                pool.extend(load_case(c).expect("case").passes);
            }
            if pool.is_empty() {
                eprintln!("no requests loaded");
                std::process::exit(2);
            }
            let reqs: Vec<Req> = (0..batch)
                .map(|i| {
                    let (ids, positions) = &pool[i % pool.len()];
                    Req {
                        ids: ids.clone(),
                        positions: positions.clone(),
                    }
                })
                .collect();
            let tokens: usize = reqs.iter().map(|r| r.ids.len()).sum();

            let singles: Vec<Vec<f32>> = reqs
                .iter()
                .map(|r| m.forward(&r.ids, &r.positions).expect("single fwd"))
                .collect();
            let batched = m.forward_batch(&reqs, GdnImpl::Scalar).expect("batch fwd");
            let mut max_dl = 0f32;
            for (a, b) in singles.iter().zip(&batched) {
                for (x, y) in a.iter().zip(b) {
                    max_dl = max_dl.max((x - y).abs());
                }
            }

            let mut batch_fwd = Vec::new();
            for i in 0..(warmup + repeats) {
                let t0 = Instant::now();
                let _ = m.forward_batch(&reqs, GdnImpl::Scalar).expect("batch fwd");
                let t1 = Instant::now();
                if i >= warmup {
                    batch_fwd.push((t1 - t0).as_secs_f64() * 1e3);
                }
            }
            let mut seq_fwd = Vec::new();
            for i in 0..(warmup + repeats) {
                let t0 = Instant::now();
                for r in &reqs {
                    m.forward(&r.ids, &r.positions).expect("fwd");
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
