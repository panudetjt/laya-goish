use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use std::io::{Read, Write};
pub fn write(fd: i32, s: &str) {
    let mut b = s.as_bytes();
    while !b.is_empty() {
        let n = if fd == 1 {
            std::io::stdout().write(b)
        } else {
            std::io::stderr().write(b)
        };
        match n {
            Ok(0) => break,
            Ok(n) => b = &b[n..],
            Err(_) => break,
        }
    }
}
pub fn out(s: &str) {
    write(1, s);
    write(1, "\n");
}
pub fn log(s: &str) {
    write(2, s);
    write(2, "\n");
}
pub fn read(path: &str) -> Result<String, String> {
    let v = std::fs::read(path).map_err(|e| crate::format!("cannot read {path}: {e:?}"))?;
    String::from_utf8(v).map_err(|_| "file is not UTF-8".to_string())
}
pub fn args() -> Vec<String> {
    std::env::args().collect()
}
pub fn line() -> Option<String> {
    let mut v = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let n = std::io::stdin().read(&mut byte).ok()?;
        if n == 0 {
            return if v.is_empty() {
                None
            } else {
                String::from_utf8(v).ok()
            };
        }
        if byte[0] == b'\n' {
            return String::from_utf8(v).ok();
        }
        v.push(byte[0]);
        if v.len() > 1048576 {
            return None;
        }
    }
}
#[cfg(not(target_arch = "wasm32"))]
pub struct Mapping(memmap2::Mmap);
#[cfg(not(target_arch = "wasm32"))]
impl Mapping {
    pub fn open(path: &str) -> Result<Self, String> {
        let f = std::fs::File::open(path).map_err(|e| crate::format!("open {path}: {e}"))?;
        let n = f
            .metadata()
            .map_err(|_| "empty/unreadable model".to_string())?
            .len();
        if n == 0 {
            return Err("empty/unreadable model".into());
        }
        let map = unsafe { memmap2::Mmap::map(&f) }.map_err(|_| "mmap failed".to_string())?;
        Ok(Self(map))
    }
    pub fn bytes(&self) -> &[u8] {
        &self.0[..]
    }
}
// wasm32 has no filesystem to mmap: the host copies GGUF bytes into linear
// memory and the loader takes ownership (engine/wasm ABI, ticket 19). Same
// parse path via Gguf::from_bytes.
#[cfg(target_arch = "wasm32")]
pub struct Mapping(Vec<u8>);
#[cfg(target_arch = "wasm32")]
impl Mapping {
    pub fn open(path: &str) -> Result<Self, String> {
        Err(format!("no filesystem on wasm32 (cannot open {path})"))
    }
    pub fn from_vec(v: Vec<u8>) -> Result<Self, String> {
        if v.is_empty() {
            return Err("empty/unreadable model".into());
        }
        Ok(Self(v))
    }
    pub fn bytes(&self) -> &[u8] {
        &self.0
    }
    /// Drop the owned GGUF bytes. Laya's forward path reads only metadata
    /// after build, so the wasm loader releases the mapping to halve live
    /// bytes (linear memory never shrinks; the pages return to the wasm
    /// allocator's free list for reuse by forward activations). OpenThai
    /// gathers embedding rows from the mapping every forward and must keep it.
    pub fn release(&mut self) {
        self.0 = Vec::new();
    }
}
