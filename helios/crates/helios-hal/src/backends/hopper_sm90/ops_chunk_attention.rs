//! The chunked prefill attention kernel as a single op on host slices, for verification: no model is needed and
//! the context can be as long as the device memory allows (the untiled kernel stops at 8192).

use std::ffi::c_void;

use anyhow::{anyhow, Result};

use super::backend::HopperBackend;
use super::ops::ELEMENTWISE_IMAGE;

impl HopperBackend {
    /// Attention of `positions.len()` query rows (`q`: `[rows, heads, head_dim]` bf16 bits) over a cache of `len`
    /// positions (`k`, `v`: `[len, kv_heads, head_dim]`), with the kernel the prefill chunks use. Row `t` attends to
    /// cache positions `0..=positions[t]`; every position must be below `len`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_chunk_bf16(
        &self,
        q: &[u16],
        k: &[u16],
        v: &[u16],
        positions: &[u32],
        len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<u16>> {
        let rows = positions.len();
        if rows == 0 || len == 0 || !matches!(head_dim, 64 | 128) || kv_heads == 0 || heads % kv_heads != 0 {
            return Err(anyhow!("unsupported attention shape: {rows} rows, len {len}, {heads} heads, {kv_heads} KV heads, head_dim {head_dim}"));
        }
        if q.len() != rows * heads * head_dim || k.len() != len * kv_heads * head_dim || v.len() != k.len() {
            return Err(anyhow!("attention input sizes do not match the shape"));
        }
        if positions.iter().any(|&p| p as usize >= len) {
            return Err(anyhow!("a query position is beyond the {len} cached positions"));
        }
        let mut dev = self.device.lock().expect("device mutex poisoned");
        let kernel = self.kernel(&dev, "elementwise", ELEMENTWISE_IMAGE, "attention_chunk_bf16")?;
        let bytes = |x: &[u16]| -> Vec<u8> { x.iter().flat_map(|v| v.to_le_bytes()).collect() };

        let mut allocated: Vec<u64> = Vec::new();
        let result = (|| -> Result<Vec<u16>> {
            let mut buffer = |dev: &mut crate::cuda::CudaDevice, n: usize| -> Result<u64> {
                let p = dev.alloc(n.max(1))?;
                allocated.push(p);
                Ok(p)
            };
            let (q_ptr, k_ptr, v_ptr) = (buffer(&mut dev, q.len() * 2)?, buffer(&mut dev, k.len() * 2)?, buffer(&mut dev, v.len() * 2)?);
            let out_ptr = buffer(&mut dev, rows * heads * head_dim * 2)?;
            let pos_ptr = buffer(&mut dev, rows * 4)?;
            dev.htod(q_ptr, &bytes(q))?;
            dev.htod(k_ptr, &bytes(k))?;
            dev.htod(v_ptr, &bytes(v))?;
            dev.htod(pos_ptr, &positions.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>())?;

            let (h, kvh, hd) = (heads as i32, kv_heads as i32, head_dim as i32);
            let mut args: Vec<*mut c_void> = vec![
                &q_ptr as *const u64 as *mut c_void,
                &k_ptr as *const u64 as *mut c_void,
                &v_ptr as *const u64 as *mut c_void,
                &out_ptr as *const u64 as *mut c_void,
                &pos_ptr as *const u64 as *mut c_void,
                &h as *const i32 as *mut c_void,
                &kvh as *const i32 as *mut c_void,
                &hd as *const i32 as *mut c_void,
            ];
            let shared = ((head_dim + 128 + 4) * 4) as u32;
            // Safety: the argument list mirrors the kernel's signature.
            unsafe { kernel.launch(&dev, ((rows * heads) as u32, 1, 1), (128, 1, 1), shared, &mut args)? };
            dev.synchronize()?;
            let mut raw = vec![0u8; rows * heads * head_dim * 2];
            dev.dtoh(&mut raw, out_ptr)?;
            Ok(raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
        })();
        for p in allocated {
            let _ = dev.free(p);
        }
        result
    }
}
