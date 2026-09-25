// Decision layer: port of src/models/openthai/decision.rs math (slot softmax
// with abstain slot 255, temperature per kind, choice-permutation averaging
// over cyclic orders, entropy confidence, abstain, score).
use crate::openthai::softmax_f32;
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub struct Question {
    pub kind: usize, // 0 choice, 1 score, 2 noul
    pub k: usize,
    pub labels: Vec<String>,
}

// Engine orders(): cyclic option permutations used when a request averages
// multiple passes (k >= 11 options, or order_invariant true).
pub fn orders(k: usize, n: usize) -> Vec<Vec<usize>> {
    let n = n.min(k).max(1);
    let offsets: BTreeSet<_> = (0..n)
        .map(|j| {
            let a = j * k;
            let d = a / n;
            let rem = a % n;
            (d + usize::from(rem * 2 > n || rem * 2 == n && d % 2 == 1)) % k
        })
        .collect();
    offsets
        .into_iter()
        .map(|off| (0..k).map(|i| (i + off) % k).collect())
        .collect()
}

fn entropy_confidence(p: &[f32]) -> f32 {
    let k = p.len();
    if k <= 1 {
        return 1.;
    }
    let entropy = -p
        .iter()
        .map(|v| v.max(1e-12) * libm::logf(v.max(1e-12)))
        .sum::<f32>();
    (1. - entropy / libm::logf(k as f32)).clamp(0., 1.)
}

// logits_per_pass: one Vec (nq*256, engine row order = question order) per
// permutation pass. temperature: exp(log_temperature) per kind.
pub fn decide(
    qids: &[String],
    qs: &[Question],
    logits_per_pass: &[Vec<f32>],
    temperature: &[f32; 3],
) -> Value {
    let n = logits_per_pass.len();
    let mut probs: Vec<Vec<f32>> = qs.iter().map(|q| vec![0.; q.k]).collect();
    let mut abstain = vec![0f32; qs.len()];
    for (pass, logits) in logits_per_pass.iter().enumerate() {
        for (i, q) in qs.iter().enumerate() {
            let k = q.k;
            let row = &logits[i * 256..(i + 1) * 256];
            let temp = temperature[q.kind];
            let mut valid: Vec<f32> = row[..k].iter().map(|v| v / temp).collect();
            valid.push(row[255] / temp);
            let p = softmax_f32(&valid);
            let sum = p[..k].iter().sum::<f32>().max(1e-12);
            let ord = if q.kind == 0 {
                let all = orders(k, n);
                all[pass % all.len()].clone()
            } else {
                (0..k).collect()
            };
            for slot in 0..k {
                probs[i][ord[slot]] += p[slot] / sum / n as f32;
            }
            abstain[i] += p[k] / n as f32;
        }
    }
    let mut answers = serde_json::Map::new();
    for (i, q) in qs.iter().enumerate() {
        let p = &mut probs[i];
        let sum = p.iter().sum::<f32>().max(1e-12);
        for v in p.iter_mut() {
            *v /= sum;
        }
        let k = p.len();
        let logits = &logits_per_pass[0][i * 256..(i + 1) * 256];
        let mut fields = vec![("type", json!(["choice", "score", "noul"][q.kind]))];
        if q.kind == 2 {
            fields.push(("noul", json!(p[1] as f64)));
        } else {
            let mut probabilities = serde_json::Map::new();
            for (j, lab) in q.labels.iter().enumerate() {
                probabilities.insert(lab.clone(), json!(p[j] as f64));
            }
            fields.push(("probabilities", Value::Object(probabilities)));
            fields.push(("confidence", json!(entropy_confidence(p) as f64)));
            if q.kind == 0 {
                let mut best = 0;
                for j in 1..k {
                    if p[j] > p[best] {
                        best = j;
                    }
                }
                fields.push(("choice", json!(q.labels[best].clone())));
                fields.push(("abstain", json!(abstain[i] as f64)));
            } else {
                let score: f32 = p.iter().enumerate().map(|(j, v)| j as f32 * v).sum();
                fields.push(("score", json!(score as f64)));
            }
        }
        fields.push((
            "logits",
            Value::Array(logits.iter().map(|&v| json!(v as f64)).collect()),
        ));
        answers.insert(qids[i].clone(), Value::Object(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect()));
    }
    Value::Object(answers)
}
