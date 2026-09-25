// GGUF v3 reader, std port of the engine's src/gguf.rs parsing rules.
use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Debug)]
pub struct GgufTensor {
    pub dims: Vec<usize>,
    pub kind: u32,
    pub offset: usize,
    pub len: usize,
}

pub struct Gguf {
    data: Vec<u8>,
    pub meta: serde_json::Map<String, serde_json::Value>,
    pub tensors: HashMap<String, GgufTensor>,
}

#[derive(Debug)]
pub struct GgufError(pub String);

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for GgufError {
    fn from(s: &str) -> Self {
        GgufError(s.to_string())
    }
}

type R<T> = Result<T, GgufError>;
fn err<T>(msg: impl Into<String>) -> R<T> {
    Err(GgufError(msg.into()))
}

fn layout(kind: u32) -> R<(usize, usize)> {
    match kind {
        0 => Ok((1, 4)),
        1 => Ok((1, 2)),
        2 => Ok((32, 18)),
        8 => Ok((32, 34)),
        12 => Ok((256, 144)),
        14 => Ok((256, 210)),
        _ => err(format!("unsupported GGML tensor type {kind}")),
    }
}

// Bit-exact f16 -> f32, identical to the engine's gguf::half.
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

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        let end = self.p.checked_add(n).ok_or("GGUF length overflow")?;
        let s = self.b.get(self.p..end).ok_or("truncated GGUF")?;
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> R<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> R<String> {
        let n = self.u64()? as usize;
        if n > 32 * 1024 * 1024 {
            return err("GGUF string too large");
        }
        std::str::from_utf8(self.take(n)?)
            .map(|s| s.to_string())
            .map_err(|_| GgufError("invalid GGUF UTF-8".into()))
    }
    fn value(&mut self, t: u32, depth: usize) -> R<serde_json::Value> {
        if depth > 4 {
            return err("GGUF metadata nesting too deep");
        }
        Ok(match t {
            0 => self.take(1)?[0].into(),
            1 => (self.take(1)?[0] as i8).into(),
            2 => u16::from_le_bytes(self.take(2)?.try_into().unwrap()).into(),
            3 => i16::from_le_bytes(self.take(2)?.try_into().unwrap()).into(),
            4 => self.u32()?.into(),
            5 => (self.u32()? as i32).into(),
            6 => (f32::from_bits(self.u32()?) as f64).into(),
            7 => (self.take(1)?[0] != 0).into(),
            8 => self.string()?.into(),
            9 => {
                let t = self.u32()?;
                let n = self.u64()? as usize;
                if n > 1_000_000 {
                    return err("GGUF array too large");
                }
                let mut a = Vec::with_capacity(n);
                for _ in 0..n {
                    a.push(self.value(t, depth + 1)?);
                }
                serde_json::Value::Array(a)
            }
            10 => self.u64()?.into(),
            11 => (self.u64()? as i64).into(),
            12 => f64::from_bits(self.u64()?).into(),
            _ => return err(format!("unsupported GGUF metadata type {t}")),
        })
    }
}

impl Gguf {
    pub fn open(path: &str) -> R<Self> {
        let data = std::fs::read(path).map_err(|e| GgufError(format!("read {path}: {e}")))?;
        let mut r = Reader { b: &data, p: 0 };
        if r.take(4)? != b"GGUF" {
            return err("not a GGUF file");
        }
        if r.u32()? != 3 {
            return err("expected GGUF version 3");
        }
        let nt = r.u64()?;
        let nk = r.u64()?;
        if nt > 100000 || nk > 100000 {
            return err("GGUF table too large");
        }
        let mut meta = serde_json::Map::new();
        for _ in 0..nk {
            let k = r.string()?;
            let t = r.u32()?;
            let v = r.value(t, 0)?;
            if meta.insert(k, v).is_some() {
                return err("duplicate GGUF metadata key");
            }
        }
        let mut tensors = HashMap::new();
        for _ in 0..nt {
            let name = r.string()?;
            let nd = r.u32()?;
            if !(1..=4).contains(&nd) {
                return err("invalid tensor rank");
            }
            let mut dims = Vec::new();
            let mut count = 1usize;
            for _ in 0..nd {
                let d = r.u64()? as usize;
                if d == 0 {
                    return err("empty tensor");
                }
                count = count.checked_mul(d).ok_or("tensor size overflow")?;
                dims.push(d)
            }
            let kind = r.u32()?;
            let offset = r.u64()? as usize;
            let (block, size) = layout(kind)?;
            if dims[0] % block != 0 {
                return err("invalid quantized row alignment");
            }
            let len = (count / block).checked_mul(size).ok_or("byte size overflow")?;
            if tensors
                .insert(name, GgufTensor { dims, kind, offset, len })
                .is_some()
            {
                return err("duplicate GGUF tensor");
            }
        }
        let align = meta
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        if align == 0 || !align.is_power_of_two() || align > 4096 {
            return err("invalid GGUF alignment");
        }
        let base = r.p.checked_add(align - 1).ok_or("offset overflow")? & !(align - 1);
        for t in tensors.values_mut() {
            if t.offset % align != 0 {
                return err("unaligned GGUF tensor");
            }
            t.offset = t.offset.checked_add(base).ok_or("offset overflow")?;
            if t.offset.checked_add(t.len).ok_or("length overflow")? > data.len() {
                return err("truncated tensor data");
            }
        }
        Ok(Self { data, meta, tensors })
    }

    pub fn meta_str(&self, k: &str) -> &str {
        self.meta.get(k).and_then(|v| v.as_str()).unwrap_or("")
    }
    pub fn meta_num(&self, k: &str, default: usize) -> usize {
        self.meta
            .get(k)
            .and_then(|v| v.as_f64())
            .map(|v| v as usize)
            .unwrap_or(default)
    }

    pub fn floats(&self, name: &str) -> R<Vec<f32>> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError(format!("missing tensor {name}")))?;
        let b = &self.data[t.offset..t.offset + t.len];
        let (block, size) = layout(t.kind)?;
        let mut out = vec![0f32; b.len() / size * block];
        match t.kind {
            0 => {
                for (x, c) in out.iter_mut().zip(b.chunks_exact(4)) {
                    *x = f32::from_le_bytes(c.try_into().unwrap())
                }
            }
            1 => {
                for (x, c) in out.iter_mut().zip(b.chunks_exact(2)) {
                    *x = half(u16::from_le_bytes([c[0], c[1]]))
                }
            }
            8 => {
                for (o, c) in out.chunks_exact_mut(32).zip(b.chunks_exact(34)) {
                    let d = half(u16::from_le_bytes([c[0], c[1]]));
                    for j in 0..32 {
                        o[j] = d * (c[2 + j] as i8 as f32)
                    }
                }
            }
            _ => return err(format!("tensor {name}: unsupported dtype {}", t.kind)),
        }
        Ok(out)
    }
}
