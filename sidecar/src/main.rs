//! laya-gpu-sidecar: standard-OS Rust process that keeps the Laya F16
//! weights resident in GPU memory (CUDA, RTX 5060 Ti via WSL2 libcuda) and
//! answers complete forward passes — token IDs + answer positions in,
//! per-question logits out — over a Unix-domain-socket JSON-lines protocol,
//! one round trip per request. The CPU binary is untouched.
mod cuda;
mod forward;
mod gguf;
mod laya;
mod oracle;

use forward::Runner;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "laya-gpu-sidecar\n  serve --laya GGUF [--resident GGUF] [--sock PATH] [--cubin PATH]\n  bench --laya GGUF [--resident GGUF] --case JSON [--qtype N] [--warmup N] [--iters N] [--cubin PATH]\n  info --laya GGUF\n  stage-check --laya GGUF --case JSON [--qtype N] (debug oracle)"
    );
    std::process::exit(2);
}

fn opt(a: &[String], k: &str) -> Option<String> {
    a.iter()
        .position(|s| s == k)
        .and_then(|i| a.get(i + 1))
        .cloned()
}

/// Extract (ids, markers) from a request/case JSON value. Malformed entries
/// are an error, never silently dropped.
fn extract_ids_markers(v: &Value) -> Result<(Vec<u32>, Vec<u32>), String> {
    let arr = |key: &str| -> Result<Vec<u32>, String> {
        let a = v
            .get(key)
            .and_then(|x| x.as_array())
            .ok_or_else(|| format!("missing {key}"))?;
        a.iter()
            .enumerate()
            .map(|(i, x)| {
                x.as_u64()
                    .filter(|n| *n <= u32::MAX as u64)
                    .map(|n| n as u32)
                    .ok_or_else(|| format!("{key}[{i}] is not a u32"))
            })
            .collect()
    };
    Ok((arr("ids")?, arr("markers")?))
}

fn usize_opt(a: &[String], k: &str) -> Result<Option<usize>, String> {
    opt(a, k)
        .map(|v| v.parse::<usize>().map_err(|_| format!("bad {k}")))
        .transpose()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("laya-gpu-sidecar: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("info") => info(&args),
        Some("serve") => serve(&args),
        Some("bench") => bench(&args),
        Some("stage-check") => stage_check(&args),
        _ => usage(),
    }
}

fn open_runner(args: &[String]) -> Result<(cuda::Cuda, Runner), String> {
    let laya_path = opt(args, "--laya").ok_or("missing --laya")?;
    let cubin_path = opt(args, "--cubin").unwrap_or_else(|| {
        "sidecar/kernels/laya.cubin".to_string()
    });
    let cubin = std::fs::read(&cubin_path).map_err(|e| format!("{cubin_path}: {e}"))?;
    let cuda = cuda::Cuda::init()?;
    let t0 = Instant::now();
    let g = gguf::GgufFile::open(&laya_path)?;
    let m = laya::LayaGpu::load(&cuda, &g)?;
    let mut resident_bytes = 0usize;
    if let Some(res) = opt(args, "--resident") {
        let rg = gguf::GgufFile::open(&res)?;
        let r = laya::load_resident(&cuda, &rg, &res)?;
        resident_bytes = r.region.bytes;
        eprintln!(
            "resident: {} ({} MiB held, never touched per-request)",
            r.name,
            resident_bytes >> 20
        );
    }
    let runner = Runner::new(&cuda, m, &cubin)?;
    eprintln!(
        "loaded {} in {:.2}s; device={}, driver={}, VRAM held: {} MiB (models {} + activations/buffers)",
        laya_path,
        t0.elapsed().as_secs_f64(),
        cuda.device_name,
        cuda.driver_version,
        cuda.allocated.get() >> 20,
        (runner.m.region.bytes + resident_bytes) >> 20,
    );
    Ok((cuda, runner))
}

fn info(args: &[String]) -> Result<(), String> {
    let laya_path = opt(args, "--laya").ok_or("missing --laya")?;
    eprintln!("opening {laya_path}");
    let g = gguf::GgufFile::open(&laya_path)?;
    eprintln!("opened; {} tensors, {} kv", g.tensors.len(), g.meta.len());
    let graph = g.graph()?;
    eprintln!("graph parsed");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "file": laya_path,
            "architecture": g.get("general.architecture"),
            "tensors": g.tensors.len(),
            "d": g.tensors.get("tok_emb.weight").map(|t| t.dims[0]),
            "vocab": g.tensors.get("tok_emb.weight").map(|t| t.dims[1]),
            "hd": g.tensors.get("cos_full").map(|t| t.dims[0]),
            "layers": g.tensors.keys().filter(|n| n.starts_with("qkv.") && n.ends_with(".weight")).count(),
            "graph_nodes": graph.get("nodes").and_then(|v| v.as_array()).map(|a| a.len()),
            "max_len": g.num("laya.max_len", 512),
            "temperature": serde_json::from_str::<Value>(
                g.get("laya.temperature").as_str().unwrap_or("[]")
            ).unwrap_or(Value::Null),
            "temperature_by_options": serde_json::from_str::<Value>(
                g.get("laya.temperature_by_options").as_str().unwrap_or("{}")
            ).unwrap_or(Value::Null),
        }))
        .map_err(|e| e.to_string())?
    );
    Ok(())
}

fn serve(args: &[String]) -> Result<(), String> {
    let sock_path = opt(args, "--sock").unwrap_or_else(|| "/tmp/laya-gpu-sidecar.sock".into());
    let (_cuda, mut runner) = open_runner(args)?;
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).map_err(|e| format!("{sock_path}: {e}"))?;
    eprintln!(
        "serving on {sock_path} (JSON lines: {{ids, markers, qtype}} -> {{logits, act}}); one connection at a time"
    );
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        serve_conn(stream, &mut runner);
    }
    Ok(())
}

fn serve_conn(stream: std::os::unix::net::UnixStream, runner: &mut Runner) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(c) => c,
        Err(_) => return,
    });
    let mut stream = stream;
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return,
            Ok(_) => {}
            Err(_) => return,
        }
        let resp: Value = match serde_json::from_str(line.trim()) {
            Ok(req) => handle(req, runner),
            Err(e) => json!({"error": format!("bad request: {e}")}),
        };
        let mut out = serde_json::to_string(&resp).unwrap_or_default();
        out.push('\n');
        if stream.write_all(out.as_bytes()).is_err() || stream.flush().is_err() {
            return;
        }
    }
}

fn handle(req: Value, runner: &mut Runner) -> Value {
    if req.get("ping").is_some() {
        return json!({
            "pong": 1,
            "models_resident_bytes": runner.m.region.bytes,
        });
    }
    let (ids, markers) = match extract_ids_markers(&req) {
        Ok(v) => v,
        Err(e) => return json!({"error": e}),
    };
    let qtype = req.get("qtype").and_then(|v| v.as_u64()).unwrap_or(2) as u32;
    match runner.forward(&ids, &markers, qtype) {
        Ok((logits, act)) => json!({
            "logits": logits,
            "act": act,
            "gpu_ms": runner.last_gpu_us as f64 / 1000.0,
            "h2d_bytes": runner.last_h2d,
            "d2h_bytes": runner.last_d2h,
        }),
        Err(e) => json!({"error": e}),
    }
}

fn stage_check(args: &[String]) -> Result<(), String> {
    let case_path = opt(args, "--case").unwrap_or_else(|| "validation/refund.reference.json".into());
    let qtype = opt(args, "--qtype")
        .map(|v| v.parse::<u32>().map_err(|_| "bad --qtype".to_string()))
        .transpose()?
        .unwrap_or(2);
    let laya_path = opt(args, "--laya").ok_or("missing --laya")?;
    let (_cuda, mut runner) = open_runner(args)?;
    let case: Value = serde_json::from_str(
        &std::fs::read_to_string(&case_path).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let q = case.as_object().and_then(|o| o.values().next()).cloned().ok_or("empty case")?;
    let mapped = {
        let mut o = serde_json::Map::new();
        o.insert("ids".into(), q.get("input_ids").cloned().ok_or("ids")?);
        o.insert("markers".into(), q.get("markers").cloned().ok_or("markers")?);
        Value::Object(o)
    };
    let (ids, markers) = extract_ids_markers(&mapped)?;
    eprintln!("decoding oracle weights (scalar, one-off)...");
    let g = gguf::GgufFile::open(&laya_path)?;
    let orc = oracle::Oracle::load(&g)?;
    eprintln!("oracle forward (slow scalar)...");
    let (o_logits, o_act, stages) = orc.forward_traced(&ids, &markers, qtype);
    eprintln!("oracle logits = {o_logits:?}  act = {o_act:?}");
    runner.trace = Some(std::cell::RefCell::new(Vec::new()));
    let (g_logits, g_act) = runner.forward(&ids, &markers, qtype)?;
    eprintln!("gpu   logits = {g_logits:?}  act = {g_act:?}");
    let trace = runner
        .trace
        .take()
        .map(|t| t.into_inner())
        .unwrap_or_default();
    let mut bad = 0usize;
    for (name, gv) in &trace {
        let ov = match stages.get(name) {
            Some(v) if v.len() == gv.len() => v,
            _ => {
                eprintln!("{name}: MISSING/size-mismatch in oracle");
                bad += 1;
                continue;
            }
        };
        let dmax = gv.iter().zip(ov).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let flag = if dmax > 0.05 { " <<<<" } else { "" };
        if dmax > 0.05 || name.contains("logits") || name == "act1" {
            eprintln!("{name:>14}: max|d| = {dmax:>12.6}{flag}");
        }
        if dmax > 0.05 {
            bad += 1;
        }
    }
    eprintln!("stages over tolerance: {bad}/{}", trace.len());
    Ok(())
}

fn bench(args: &[String]) -> Result<(), String> {
    let case_path = opt(args, "--case").unwrap_or_else(|| {
        "validation/refund.reference.json".into()
    });
    let qtype = opt(args, "--qtype")
        .map(|v| v.parse::<u32>().map_err(|_| "bad --qtype".to_string()))
        .transpose()?
        .unwrap_or(2);
    let warmup = usize_opt(args, "--warmup")?.unwrap_or(1);
    let iters = usize_opt(args, "--iters")?.unwrap_or(8);
    let (_cuda, mut runner) = open_runner(args)?;
    let case: Value = serde_json::from_str(
        &std::fs::read_to_string(&case_path).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let q = case
        .as_object()
        .and_then(|o| o.values().next())
        .cloned()
        .ok_or("case file has no question entry")?;
    let mapped = {
        let mut o = serde_json::Map::new();
        o.insert("ids".into(), q.get("input_ids").cloned().ok_or("case lacks input_ids")?);
        o.insert("markers".into(), q.get("markers").cloned().ok_or("case lacks markers")?);
        Value::Object(o)
    };
    let (ids, markers) = extract_ids_markers(&mapped)?;
    let ref_logits: Option<Vec<f32>> = q
        .get("logits")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_f64().map(|x| x as f32)).collect());
    let ref_act: Option<Vec<f32>> = q
        .get("act_logits")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_f64().map(|x| x as f32)).collect());

    for _ in 0..warmup {
        runner.forward(&ids, &markers, qtype)?;
    }
    {
        let (lg, ac) = runner.forward(&ids, &markers, qtype)?;
        eprintln!("gpu logits = {lg:?}");
        eprintln!("gpu act    = {ac:?}");
        eprintln!("ref logits = {ref_logits:?}");
        eprintln!("ref act    = {ref_act:?}");
    }
    let mut times = Vec::new();
    let mut worst_logit = 0f32;
    let mut worst_act = 0f32;
    for _ in 0..iters {
        let t = Instant::now();
        let (logits, act) = runner.forward(&ids, &markers, qtype)?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        if let Some(rl) = &ref_logits {
            worst_logit = worst_logit
                .max(logits.iter().zip(rl).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max));
            if let Some(ra) = &ref_act {
                worst_act = worst_act
                    .max(act.iter().zip(ra).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max));
            }
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
    let out = json!({
        "case": case_path,
        "qtype": qtype,
        "tokens": ids.len(),
        "markers": markers,
        "iters": iters,
        "mean_ms": mean,
        "min_ms": times[0],
        "p50_ms": times[times.len() / 2],
        "max_ms": times[times.len() - 1],
        "all_ms": times,
        "last_h2d_bytes": runner.last_h2d,
        "last_d2h_bytes": runner.last_d2h,
        "vram_total_bytes": runner.cuda_allocated(),
        "vram_models_bytes": runner.m.region.bytes,
        "max_dlogit": worst_logit,
        "max_dact": worst_act,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    Ok(())
}
