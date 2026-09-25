// safetensors weight loader for the converted Laya checkpoint.
//
// Tensors are built straight from the mmap'd file bytes (one copy into
// TensorData) with the recorded dtype, so the F16 file keeps weights
// F16-resident on burn-flex.
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor, TensorData};
use serde_json::Value;
use std::collections::HashMap;

pub struct Weights {
    st: safetensors::SafeTensors<'static>,
    _mmap: memmap2::Mmap, // holds the 'static borrow
    pub manifest: Value,
    pub layers: usize,
    pub head_layers: usize,
    pub d: usize,
    pub hd: usize,
    pub full_layers: Vec<usize>,
    pub temperature: String,
    pub temperature_by_options: String,
    norm_eps: HashMap<String, f32>,
}

impl Weights {
    pub fn open(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| e.to_string())? };
        let raw: &'static [u8] = unsafe { std::mem::transmute(&mmap[..]) };
        let st = safetensors::SafeTensors::deserialize(raw)
            .map_err(|e| format!("safetensors {path}: {e}"))?;
        let (_, header) = safetensors::SafeTensors::read_metadata(raw)
            .map_err(|e| format!("safetensors {path}: {e}"))?;
        let manifest_str = header
            .metadata()
            .as_ref()
            .and_then(|m| m.get("manifest"))
            .cloned()
            .ok_or("weights lack __metadata__.manifest (run `convert` first)")?;
        let manifest: Value =
            serde_json::from_str(&manifest_str).map_err(|e| format!("manifest: {e}"))?;
        let norm_eps =
            serde_json::from_value::<HashMap<String, f32>>(manifest["norm_eps"].clone())
                .unwrap_or_default();
        Ok(Self {
            st,
            _mmap: mmap,
            layers: manifest["layers"].as_u64().ok_or("layers")? as usize,
            head_layers: manifest["head_layers"].as_u64().ok_or("head_layers")? as usize,
            d: manifest["d"].as_u64().ok_or("d")? as usize,
            hd: manifest["hd"].as_u64().ok_or("hd")? as usize,
            full_layers: manifest["full_attention_trunk_layers"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
                .unwrap_or_default(),
            temperature: manifest["temperature"].as_str().unwrap_or("").to_string(),
            temperature_by_options: manifest["temperature_by_options"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            norm_eps,
            manifest,
        })
    }

    pub fn norm_eps(&self, norm_name: &str) -> f32 {
        self.norm_eps.get(norm_name).copied().unwrap_or(1e-5)
    }

    pub fn dtype(&self, name: &str) -> Option<safetensors::tensor::Dtype> {
        self.st.tensor(name).ok().map(|t| t.dtype())
    }

    pub fn data(&self, name: &str) -> Result<TensorData, String> {
        let view = self
            .st
            .tensor(name)
            .map_err(|e| format!("missing tensor {name}: {e}"))?;
        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = match view.dtype() {
            safetensors::tensor::Dtype::F32 => DType::F32,
            safetensors::tensor::Dtype::F16 => DType::F16,
            d => return Err(format!("tensor {name}: unsupported safetensors dtype {d:?}")),
        };
        Ok(TensorData::from_bytes_vec(view.data().to_vec(), shape, dtype))
    }

    pub fn matrix<B: Backend>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>, String> {
        let data = self.data(name)?;
        if data.shape.len() != 2 {
            return Err(format!("tensor {name}: expected rank 2"));
        }
        Ok(Tensor::<B, 2>::from_data(data, device))
    }

    // Rows [row_start, row_start+len) of a rank-2 tensor, built without ever
    // materializing the full tensor on the device (llvmpipe buffer limits).
    pub fn matrix_rows<B: Backend>(
        &self,
        name: &str,
        row_start: usize,
        len: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>, String> {
        let view = self
            .st
            .tensor(name)
            .map_err(|e| format!("missing tensor {name}: {e}"))?;
        let shape = view.shape().to_vec();
        if shape.len() != 2 || row_start + len > shape[0] {
            return Err(format!("tensor {name}: bad row range"));
        }
        let dtype = match view.dtype() {
            safetensors::tensor::Dtype::F32 => DType::F32,
            safetensors::tensor::Dtype::F16 => DType::F16,
            d => return Err(format!("tensor {name}: unsupported dtype {d:?}")),
        };
        let esize = if dtype == DType::F32 { 4 } else { 2 };
        let cols = shape[1];
        let bytes = view.data();
        let s = row_start * cols * esize;
        let rows_bytes = bytes[s..s + len * cols * esize].to_vec();
        Ok(Tensor::<B, 2>::from_data(
            TensorData::from_bytes_vec(rows_bytes, vec![len, cols], dtype),
            device,
        ))
    }

    pub fn vector<B: Backend>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 1>, String> {
        let data = self.data(name)?;
        if data.shape.len() != 1 {
            return Err(format!("tensor {name}: expected rank 1"));
        }
        Ok(Tensor::<B, 1>::from_data(data, device).cast(DType::F32))
    }

    pub fn vector_opt<B: Backend>(&self, name: &str, device: &B::Device) -> Option<Tensor<B, 1>> {
        if self.st.tensor(name).is_err() {
            return None;
        }
        self.vector(name, device).ok()
    }
}
