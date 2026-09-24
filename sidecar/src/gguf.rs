//! GGUF v3 reader (std port of the CPU engine's src/gguf.rs reader).
//! Parses metadata into serde_json::Value; the numeric conversions match the
//! CPU engine's Reader::value exactly (all GGUF scalar types become f64
//! numbers, which is lossless for every value this model's metadata holds).
use serde_json::{Map, Value as J};
use std::collections::BTreeMap;

pub struct TensorMeta {
    pub dims: Vec<usize>,
    pub kind: u32,
    /// Absolute byte offset of the tensor data in the file.
    pub offset: usize,
    /// Byte length of the stored tensor data.
    pub len: usize,
}

pub struct GgufFile {
    pub bytes: Vec<u8>,
    pub meta: BTreeMap<String, J>,
    pub tensors: BTreeMap<String, TensorMeta>,
    /// Byte offset where the tensor data section starts.
    pub data_base: usize,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("GGUF length overflow")?;
        let s = self.b.get(self.p..end).ok_or("truncated GGUF")?;
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        if n > 32 * 1024 * 1024 {
            return Err("GGUF string too large".into());
        }
        std::str::from_utf8(self.take(n)?)
            .map(|s| s.to_string())
            .map_err(|_| "invalid GGUF UTF-8".into())
    }
    fn value(&mut self, t: u32, depth: usize) -> Result<J, String> {
        if depth > 4 {
            return Err("GGUF metadata nesting too deep".into());
        }
        Ok(match t {
            0 => J::from(self.take(1)?[0] as u8 as f64),
            1 => J::from(self.take(1)?[0] as i8 as f64),
            2 => J::from(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as f64),
            3 => J::from(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as f64),
            4 => J::from(self.u32()? as f64),
            5 => J::from(self.u32()? as i32 as f64),
            6 => number(f32::from_bits(self.u32()?) as f64),
            7 => J::Bool(self.take(1)?[0] != 0),
            8 => J::String(self.string()?),
            9 => {
                let t = self.u32()?;
                let n = self.u64()? as usize;
                if n > 1_000_000 {
                    return Err("GGUF array too large".into());
                }
                let mut a = Vec::with_capacity(n);
                for _ in 0..n {
                    a.push(self.value(t, depth + 1)?);
                }
                J::Array(a)
            }
            10 => J::from(self.u64()? as f64),
            11 => J::from(self.u64()? as i64 as f64),
            12 => number(f64::from_bits(self.u64()?)),
            _ => return Err(format!("unsupported GGUF metadata type {t}")),
        })
    }
}

fn number(v: f64) -> J {
    // GGUF float metadata in these models (eps, temperatures, scales) is
    // always finite and well inside f64 precision.
    serde_json::Number::from_f64(v).map(J::Number).unwrap_or(J::Null)
}

/// Block size and stored byte size per GGML tensor type (mirrors
/// src/gguf.rs::layout in the CPU engine).
pub fn layout(kind: u32) -> Result<(usize, usize), String> {
    match kind {
        0 => Ok((1, 4)),
        1 => Ok((1, 2)),
        2 => Ok((32, 18)),
        8 => Ok((32, 34)),
        12 => Ok((256, 144)),
        14 => Ok((256, 210)),
        _ => Err(format!("unsupported GGML tensor type {kind}")),
    }
}

impl GgufFile {
    pub fn open(path: &str) -> Result<GgufFile, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let mut r = Reader {
            b: &bytes,
            p: 0,
        };
        if r.take(4)? != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        if r.u32()? != 3 {
            return Err("expected GGUF version 3".into());
        }
        let nt = r.u64()?;
        let nk = r.u64()?;
        if nt > 100000 || nk > 100000 {
            return Err("GGUF table too large".into());
        }
        let mut meta = BTreeMap::new();
        for _ in 0..nk {
            let k = r.string()?;
            let t = r.u32()?;
            let v = r.value(t, 0)?;
            if meta.insert(k, v).is_some() {
                return Err("duplicate GGUF metadata key".into());
            }
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..nt {
            let name = r.string()?;
            let nd = r.u32()?;
            if !(1..=4).contains(&nd) {
                return Err("invalid tensor rank".into());
            }
            let mut dims = Vec::new();
            let mut count = 1usize;
            for _ in 0..nd {
                let d = r.u64()? as usize;
                if d == 0 {
                    return Err("empty tensor".into());
                }
                count = count.checked_mul(d).ok_or("tensor size overflow")?;
                dims.push(d);
            }
            let kind = r.u32()?;
            let offset = r.u64()? as usize;
            let (block, size) = layout(kind)?;
            if dims[0] % block != 0 {
                return Err("invalid quantized row alignment".into());
            }
            let len = (count / block)
                .checked_mul(size)
                .ok_or("tensor byte size overflow")?;
            if tensors.insert(name, TensorMeta { dims, kind, offset, len }).is_some() {
                return Err("duplicate tensor".into());
            }
        }
        let align = meta
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        if align == 0 || !align.is_power_of_two() || align > 4096 {
            return Err("invalid GGUF alignment".into());
        }
        let base = (r.p.checked_add(align - 1).ok_or("offset overflow")?) & !(align - 1);
        for t in tensors.values_mut() {
            if t.offset % align != 0 {
                return Err("unaligned GGUF tensor".into());
            }
            t.offset = t.offset.checked_add(base).ok_or("offset overflow")?;
            if t.offset.checked_add(t.len).ok_or("length overflow")? > bytes.len() {
                return Err("truncated tensor data".into());
            }
        }
        Ok(GgufFile {
            bytes,
            meta,
            tensors,
            data_base: base,
        })
    }

    pub fn get(&self, k: &str) -> &J {
        self.meta.get(k).unwrap_or(&J::Null)
    }

    pub fn num(&self, k: &str, default: usize) -> usize {
        self.meta
            .get(k)
            .and_then(|v| v.as_f64())
            .map(|v| v as usize)
            .unwrap_or(default)
    }

    pub fn graph(&self) -> Result<J, String> {
        let s = self
            .get("ggmlc.graph_spec")
            .as_str()
            .ok_or("missing ggmlc.graph_spec")?;
        serde_json::from_str(s).map_err(|e| format!("graph_spec parse: {e}"))
    }

    pub fn data(&self, name: &str) -> Result<(&TensorMeta, &[u8]), String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        Ok((
            t,
            &self.bytes[t.offset..t.offset + t.len],
        ))
    }

    /// Decode a tensor to f32 (F32/F16 paths are all the Laya metadata
    /// tensors need; matches the CPU engine's gguf::decode for these kinds).
    pub fn floats(&self, name: &str) -> Result<Vec<f32>, String> {
        let (t, b) = self.data(name)?;
        decode(t.kind, b)
    }
}

/// Exact port of the CPU engine's gguf::half (bit-manipulation F16 -> F32).
pub fn half(x: u16) -> f32 {
    let sign = ((x as u32) & 0x8000) << 16;
    let e = (x >> 10) & 31;
    let m = x & 1023;
    if e == 0 {
        if m == 0 {
            f32::from_bits(sign)
        } else {
            let v = (m as f32) * (1.0 / 16777216.0);
            if sign != 0 {
                -v
            } else {
                v
            }
        }
    } else if e == 31 {
        f32::from_bits(sign | 0x7f800000 | ((m as u32) << 13))
    } else {
        f32::from_bits(sign | (((e as u32) + 112) << 23) | ((m as u32) << 13))
    }
}

pub fn decode(kind: u32, b: &[u8]) -> Result<Vec<f32>, String> {
    let (block, size) = layout(kind)?;
    let mut out = vec![0f32; b.len() / size * block];
    match kind {
        0 => {
            for (x, c) in out.iter_mut().zip(b.chunks_exact(4)) {
                *x = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        1 => {
            for (x, c) in out.iter_mut().zip(b.chunks_exact(2)) {
                *x = half(u16::from_le_bytes(c.try_into().unwrap()));
            }
        }
        8 => {
            for (o, c) in out.chunks_exact_mut(32).zip(b.chunks_exact(34)) {
                let d = half(u16::from_le_bytes([c[0], c[1]]));
                for j in 0..32 {
                    o[j] = d * (c[2 + j] as i8 as f32);
                }
            }
        }
        _ => return Err(format!("sidecar decode: unsupported tensor kind {kind}")),
    }
    Ok(out)
}

/// Convenience: object-get helper mirroring the CPU engine's J::get on
/// graph_spec values (string keys only).
pub fn jget<'a>(v: &'a J, key: &str) -> &'a J {
    match v {
        J::Object(m) => m.get(key).unwrap_or(&J::Null),
        _ => &J::Null,
    }
}

/// ggmlc graph_spec node opcodes this loader walks (mirrors the CPU
/// engine's src/models/laya/model.rs).
pub const OP_LINEAR: f64 = 29.0;
pub const OP_FUSED_BIAS: f64 = 200.0;
pub const OP_LAYER_NORM: f64 = 201.0;
pub const OP_ATTENTION: f64 = 74.0;

/// Tensor name for a graph tensor id (graph.tensors."<id>".name).
pub fn graph_tensor_name<'a>(tensors: &'a Map<String, J>, id: &J) -> &'a str {
    let key = format!("{}", id.as_f64().map(|v| v as i64).unwrap_or(-1));
    tensors
        .get(&key)
        .map(|t| t.get("name").and_then(|n| n.as_str()).unwrap_or(""))
        .unwrap_or("")
}
