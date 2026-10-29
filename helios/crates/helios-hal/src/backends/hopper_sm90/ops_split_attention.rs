//! The split-K decode attention as a single op on host slices, for verification: no model is
//! needed and the context can be longer than the untiled prefill kernel allows.

use std::ffi::c_void;

use anyhow::{anyhow, Result};

use super::backend::HopperBackend;
use super::ops::ELEMENTWISE_PTX;

/// Threads per block of `attention_split_bf16`; the kernel's tile size.
const SPLIT_BLOCK: u32 = 128;

/// How many pieces the context is cut into: enough blocks (KV heads x splits) to give every SM one.
pub(super) fn split_count(sm_count: u32, kv_heads: usize) -> usize {
    (sm_count as usize).div_ceil(kv_heads.max(1)).clamp(1, 256)
}

impl HopperBackend {
    /// Attention of one query token (`q`: `[heads, head_dim]` bf16 bits) over a cache of `len`
    /// positions (`k`, `v`: `[len, kv_heads, head_dim]`), with the split-K kernels the decode
    /// step uses. The query sits at position `len - 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_split_bf16(
        &self,
        q: &[u16],
        k: &[u16],
        v: &[u16],
        len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<u16>> {
        if len == 0 || !matches!(head_dim, 32 | 64 | 128) || kv_heads == 0 || heads % kv_heads != 0 || heads / kv_heads > 8 {
            return Err(anyhow!("unsupported attention shape: len {len}, {heads} heads, {kv_heads} KV heads, head_dim {head_dim}"));
        }
        if q.len() != heads * head_dim || k.len() != len * kv_heads * head_dim || v.len() != k.len() {
            return Err(anyhow!("attention input sizes do not match the shape"));
        }
        let mut dev = self.device.lock().expect("device mutex poisoned");
        let split = self.kernel(&dev, "elementwise", ELEMENTWISE_PTX, "attention_split_bf16")?;
        let combine = self.kernel(&dev, "elementwise", ELEMENTWISE_PTX, "attention_combine_bf16")?;
        let splits = split_count(dev.multiprocessor_count(), kv_heads);
        let bytes = |x: &[u16]| -> Vec<u8> { x.iter().flat_map(|v| v.to_le_bytes()).collect() };

        let mut allocated: Vec<u64> = Vec::new();
        let result = (|| -> Result<Vec<u16>> {
            let mut buffer = |dev: &mut crate::cuda::CudaDevice, n: usize| -> Result<u64> {
                let p = dev.alloc(n.max(1))?;
                allocated.push(p);
                Ok(p)
            };
            let (q_ptr, k_ptr, v_ptr) = (buffer(&mut dev, q.len() * 2)?, buffer(&mut dev, k.len() * 2)?, buffer(&mut dev, v.len() * 2)?);
            let out_ptr = buffer(&mut dev, heads * head_dim * 2)?;
            let ws_ptr = buffer(&mut dev, heads * splits * (head_dim + 2) * 4)?;
            let pos_ptr = buffer(&mut dev, 4)?;
            dev.htod(q_ptr, &bytes(q))?;
            dev.htod(k_ptr, &bytes(k))?;
            dev.htod(v_ptr, &bytes(v))?;
            dev.htod(pos_ptr, &((len - 1) as u32).to_le_bytes())?;

            let (h, kvh, hd, sp) = (heads as i32, kv_heads as i32, head_dim as i32, splits as i32);
            let mut split_args: Vec<*mut c_void> = vec![
                &q_ptr as *const u64 as *mut c_void,
                &k_ptr as *const u64 as *mut c_void,
                &v_ptr as *const u64 as *mut c_void,
                &ws_ptr as *const u64 as *mut c_void,
                &pos_ptr as *const u64 as *mut c_void,
                &h as *const i32 as *mut c_void,
                &kvh as *const i32 as *mut c_void,
                &hd as *const i32 as *mut c_void,
                &sp as *const i32 as *mut c_void,
            ];
            let mut combine_args: Vec<*mut c_void> = vec![
                &ws_ptr as *const u64 as *mut c_void,
                &out_ptr as *const u64 as *mut c_void,
                &pos_ptr as *const u64 as *mut c_void,
                &h as *const i32 as *mut c_void,
                &hd as *const i32 as *mut c_void,
                &sp as *const i32 as *mut c_void,
            ];
            // Safety: the argument lists mirror the kernels' signatures.
            unsafe {
                split.launch(&dev, ((kv_heads * splits) as u32, 1, 1), (SPLIT_BLOCK, 1, 1), 0, &mut split_args)?;
                combine.launch(&dev, (heads as u32, 1, 1), (head_dim as u32, 1, 1), 0, &mut combine_args)?;
            }
            dev.synchronize()?;
            let mut raw = vec![0u8; heads * head_dim * 2];
            dev.dtoh(&mut raw, out_ptr)?;
            Ok(raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
        })();
        for p in allocated {
            let _ = dev.free(p);
        }
        result
    }
}
