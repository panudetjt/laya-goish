// Decision layer: port of src/models/laya/decision.rs math (temperature
// buckets, softmax, choice tie-break, score, confidence, act_probability).
use crate::gguf::Gguf;
use crate::laya::softmax_f32;

pub struct Temps {
    base: [f32; 3],
    buckets: std::collections::HashMap<String, f32>,
}

impl Temps {
    pub fn load(g: &Gguf) -> Self {
        let base_v: Vec<f32> = serde_json::from_str(g.meta_str("laya.temperature"))
            .unwrap_or_else(|_| vec![1.0, 1.0, 1.0]);
        let mut base = [1.0f32; 3];
        for (i, v) in base_v.iter().take(3).enumerate() {
            base[i] = *v;
        }
        let buckets = serde_json::from_str::<std::collections::HashMap<String, f32>>(
            g.meta_str("laya.temperature_by_options"),
        )
        .unwrap_or_default();
        Self { base, buckets }
    }

    fn temp(&self, kind: usize, k: usize) -> f32 {
        let bucket = format!(
            "{}:{}",
            ["choice", "score", "noul"][kind],
            if k <= 2 {
                "2"
            } else if k <= 5 {
                "3-5"
            } else if k <= 10 {
                "6-10"
            } else {
                "11+"
            }
        );
        self.buckets
            .get(&bucket)
            .copied()
            .unwrap_or(self.base[kind])
            .max(1e-3)
    }
}

// Engine decision::confidence (f64 entropy form, choice/score kinds).
fn confidence(p: &[f32]) -> f32 {
    if p.len() < 2 {
        return 1.0;
    }
    let e = -p
        .iter()
        .map(|&v| {
            let x = (v as f64).max(1e-12);
            x * x.ln()
        })
        .sum::<f64>();
    (1.0 - e / (p.len() as f64).ln()).clamp(0.0, 1.0) as f32
}

pub struct Answer {
    pub kind: usize,
    pub labels: Vec<String>,
}

// Computes the engine's answer fields from raw (logits, act). Labels come
// from the reference case (engine order).
pub fn decide(logits: &[f32], act: &[f32], ans: &Answer, temps: &Temps) -> serde_json::Value {
    let kind = ans.kind;
    let k = logits.len();
    let temp = temps.temp(kind, k);
    let p = softmax_f32(&logits.iter().map(|v| v / temp).collect::<Vec<f32>>());
    let ap = softmax_f32(act)[0];
    let mut a = serde_json::Map::new();
    a.insert("type".into(), ["choice", "score", "noul"][kind].into());
    let mut probs = serde_json::Map::new();
    for (i, &v) in p.iter().enumerate() {
        let label = if kind == 1 {
            i.to_string()
        } else {
            ans.labels
                .get(i)
                .cloned()
                .unwrap_or_else(|| i.to_string())
        };
        probs.insert(label, (v as f64).into());
    }
    match kind {
        0 => {
            // Engine: (0..k).max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap()
            //                         .then_with(|| b.cmp(&a)))
            // verified empirically: plain argmax, first index wins exact ties.
            let mut best = 0usize;
            for i in 1..k {
                if p[i] > p[best] {
                    best = i;
                }
            }
            a.insert(
                "choice".into(),
                ans.labels
                    .get(best)
                    .cloned()
                    .unwrap_or_else(|| best.to_string())
                    .into(),
            );
        }
        1 => {
            let score: f64 = p
                .iter()
                .enumerate()
                .map(|(i, &p)| i as f64 * (p as f64))
                .sum();
            a.insert("score".into(), score.into());
        }
        _ => {
            a.insert("noul".into(), (p[1] as f64).into());
        }
    }
    a.insert(
        "confidence".into(),
        (if kind == 2 {
            p[1].max(1.0 - p[1])
        } else {
            confidence(&p)
        } as f64)
            .into(),
    );
    a.insert("probabilities".into(), serde_json::Value::Object(probs));
    a.insert(
        "action".into(),
        serde_json::json!({"act_probability": ap as f64}),
    );
    a.insert(
        "logits".into(),
        serde_json::Value::Array(logits.iter().map(|&v| (v as f64).into()).collect()),
    );
    a.insert(
        "act_logits".into(),
        serde_json::Value::Array(act.iter().map(|&v| (v as f64).into()).collect()),
    );
    serde_json::Value::Object(a)
}
