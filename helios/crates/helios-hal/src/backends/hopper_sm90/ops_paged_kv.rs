//! The page-major KV kernels as host ops, for verification (ADR 0002): K and V rows appended through a device page table into one pool
//! whose page holds `[layer][k|v][token][kv_head][head_dim]`, and the prefill attention reading them back through the table. A page table
//! entry of -1 (`NO_PAGE`) makes every kernel skip its access. The pool pages are placed by a shuffled table, so nothing is contiguous.

use std::ffi::c_void;

use anyhow::{anyhow, Result};

use super::backend::HopperBackend;
use super::ops::ELEMENTWISE_IMAGE;
use super::ops_paged_attention::PAGE_TOKENS;

fn bytes(x: &[u16]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl HopperBackend {
    /// Append `rows` (K and V, `[rows, kv_heads * head_dim]` bf16 bits) at `positions` through `table` into a fresh pool of `pages` pages
    /// and `layers` layers, as layer `layer`, with the prefill-chunk kernel (`chunk` true) or, one row at a time, the decode kernel. Returns the
    /// whole pool, for the host to check each row against the layout formula.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_paged_bf16(&self, chunk: bool, pages: usize, layers: usize, layer: usize, kv_heads: usize, head_dim: usize, table: &[i32], k: &[u16], v: &[u16], positions: &[u32]) -> Result<Vec<u16>> {
        let kv_dim = kv_heads * head_dim;
        let rows = positions.len();
        if rows == 0 || layer >= layers || k.len() != rows * kv_dim || v.len() != k.len() {
            return Err(anyhow!("unsupported KV append shape: {rows} rows, layer {layer} of {layers}, kv_dim {kv_dim}"));
        }
        let mut dev = self.device.lock().expect("device mutex poisoned");
        let name = if chunk { "kv_append_chunk_paged_bf16" } else { "kv_append_paged_bf16" };
        let kernel = self.kernel(&dev, "elementwise", ELEMENTWISE_IMAGE, name)?;
        let pool_elems = pages * layers * 2 * PAGE_TOKENS * kv_dim;
        let mut allocated: Vec<u64> = Vec::new();
        let result = (|| -> Result<Vec<u16>> {
            let mut buffer = |dev: &mut crate::cuda::CudaDevice, n: usize| -> Result<u64> {
                let p = dev.alloc(n.max(4))?;
                allocated.push(p);
                Ok(p)
            };
            let pool = buffer(&mut dev, pool_elems * 2)?;
            let (table_p, k_p, v_p, pos_p, rows_p) = (buffer(&mut dev, table.len() * 4)?, buffer(&mut dev, k.len() * 2)?, buffer(&mut dev, v.len() * 2)?, buffer(&mut dev, rows * 4)?, buffer(&mut dev, 4)?);
            dev.htod(pool, &vec![0xEEu8; pool_elems * 2])?;
            dev.htod(table_p, &table.iter().flat_map(|e| e.to_le_bytes()).collect::<Vec<u8>>())?;
            dev.htod(k_p, &bytes(k))?;
            dev.htod(v_p, &bytes(v))?;
            dev.htod(pos_p, &positions.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>())?;
            dev.htod(rows_p, &(rows as i32).to_le_bytes())?;
            let (kv_dim_i, layer_i, layers_i) = (kv_dim as i32, layer as i32, layers as i32);
            let ptr = |p: &u64| p as *const u64 as *mut c_void;
            let int = |v: &i32| v as *const i32 as *mut c_void;
            if chunk {
                let mut args = vec![ptr(&pool), ptr(&table_p), ptr(&k_p), ptr(&v_p), ptr(&pos_p), ptr(&rows_p), int(&kv_dim_i), int(&layer_i), int(&layers_i)];
                // Safety: the argument list mirrors the kernel's signature.
                unsafe { kernel.launch(&dev, (rows as u32, 1, 1), (256, 1, 1), 0, &mut args)? };
            } else {
                // The decode kernel reads one position from device memory and appends one row: run it once per row.
                for r in 0..rows {
                    let row_k = k_p + (r * kv_dim * 2) as u64;
                    let row_v = v_p + (r * kv_dim * 2) as u64;
                    let at = pos_p + (r * 4) as u64;
                    let mut args = vec![ptr(&pool), ptr(&table_p), ptr(&row_k), ptr(&row_v), ptr(&at), int(&kv_dim_i), int(&layer_i), int(&layers_i)];
                    // Safety: as above.
                    unsafe { kernel.launch(&dev, (kv_dim.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &mut args)? };
                }
            }
            dev.synchronize()?;
            let mut raw = vec![0u8; pool_elems * 2];
            dev.dtoh(&mut raw, pool)?;
            Ok(raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
        })();
        for p in allocated {
            let _ = dev.free(p);
        }
        result
    }

    /// The prefill attention reading a page-major pool through a page table: `k` and `v` (`[len, kv_heads, head_dim]`) are scattered into a pool
    /// of `layers` layers as layer `layer` along `table` (logical page -> physical page, `table.len() * 64 >= len`), and the kernel
    /// `attention_mma_paged_{head_dim}_bf16` runs over it. Same arguments and output as [`Self::attention_mma_bf16`], which it must equal bit for bit.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_mma_paged_bf16(&self, layers: usize, layer: usize, table: &[i32], q: &[u16], k: &[u16], v: &[u16], positions: &[u32], len: usize, heads: usize, kv_heads: usize, head_dim: usize) -> Result<Vec<u16>> {
        let rows = positions.len();
        let row = kv_heads * head_dim;
        let pages = table.iter().copied().filter(|&p| p >= 0).max().map_or(0, |m| m as usize + 1);
        if rows == 0 || len == 0 || !matches!(head_dim, 64 | 128) || kv_heads == 0 || heads % kv_heads != 0 || layer >= layers || table.len() * PAGE_TOKENS < len || pages == 0 {
            return Err(anyhow!("unsupported paged attention shape: {rows} rows, len {len}, {heads}/{kv_heads} heads, head_dim {head_dim}, layer {layer} of {layers}"));
        }
        if q.len() != rows * heads * head_dim || k.len() != len * row || v.len() != k.len() || positions.iter().any(|&p| p as usize >= len) {
            return Err(anyhow!("paged attention input sizes do not match the shape"));
        }
        let mut dev = self.device.lock().expect("device mutex poisoned");
        let kernel = self.kernel(&dev, "elementwise", ELEMENTWISE_IMAGE, &format!("attention_mma_paged_{head_dim}_bf16"))?;
        let page_elems = PAGE_TOKENS * row;
        let mut allocated: Vec<u64> = Vec::new();
        let result = (|| -> Result<Vec<u16>> {
            let mut buffer = |dev: &mut crate::cuda::CudaDevice, n: usize| -> Result<u64> {
                let p = dev.alloc(n.max(4))?;
                allocated.push(p);
                Ok(p)
            };
            let pool = buffer(&mut dev, pages * layers * 2 * page_elems * 2)?;
            let (q_p, out_p, pos_p, table_p) = (buffer(&mut dev, q.len() * 2)?, buffer(&mut dev, rows * heads * head_dim * 2)?, buffer(&mut dev, rows * 4)?, buffer(&mut dev, table.len() * 4)?);
            dev.htod(pool, &vec![0xEEu8; pages * layers * 2 * page_elems * 2])?;
            for (logical, &physical) in table.iter().enumerate() {
                if physical < 0 || logical * PAGE_TOKENS >= len {
                    continue;
                }
                let upto = ((logical + 1) * PAGE_TOKENS).min(len);
                let used = (upto - logical * PAGE_TOKENS) * row;
                let slot = |which: usize| pool + (((physical as usize * layers + layer) * 2 + which) * page_elems * 2) as u64;
                dev.htod(slot(0), &bytes(&k[logical * page_elems..logical * page_elems + used]))?;
                dev.htod(slot(1), &bytes(&v[logical * page_elems..logical * page_elems + used]))?;
            }
            dev.htod(q_p, &bytes(q))?;
            dev.htod(pos_p, &positions.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>())?;
            dev.htod(table_p, &table.iter().flat_map(|e| e.to_le_bytes()).collect::<Vec<u8>>())?;
            let (h, kvh, rows_i, layer_i, layers_i) = (heads as i32, kv_heads as i32, rows as i32, layer as i32, layers as i32);
            let ptr = |p: &u64| p as *const u64 as *mut c_void;
            let int = |v: &i32| v as *const i32 as *mut c_void;
            let mut args = vec![ptr(&q_p), ptr(&pool), ptr(&out_p), ptr(&pos_p), int(&rows_i), int(&h), int(&kvh), ptr(&table_p), int(&layer_i), int(&layers_i)];
            let (blocks, kv_rows) = super::prefill_plan::mma_grid(rows, heads, kv_heads).ok_or_else(|| anyhow!("heads do not divide over the KV heads"))?;
            // Safety: the argument list mirrors the kernel's signature.
            unsafe { kernel.launch(&dev, (blocks as u32, kv_rows as u32, 1), (128, 1, 1), 0, &mut args)? };
            dev.synchronize()?;
            let mut raw = vec![0u8; rows * heads * head_dim * 2];
            dev.dtoh(&mut raw, out_p)?;
            Ok(raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
        })();
        for p in allocated {
            let _ = dev.free(p);
        }
        result
    }

    /// Launch the three paged kernels against a table whose entries are all `NO_PAGE` and a pool address that is not mapped: they must skip every
    /// access and return without a fault (the warm-up launches before the KV pool exists rely on it). `Ok` after the device synchronised.
    pub fn paged_kernels_skip_a_table_without_pages(&self, kv_heads: usize, head_dim: usize, heads: usize) -> Result<()> {
        let kv_dim = kv_heads * head_dim;
        let mut dev = self.device.lock().expect("device mutex poisoned");
        let append = self.kernel(&dev, "elementwise", ELEMENTWISE_IMAGE, "kv_append_paged_bf16")?;
        let append_chunk = self.kernel(&dev, "elementwise", ELEMENTWISE_IMAGE, "kv_append_chunk_paged_bf16")?;
        let mma = self.kernel(&dev, "elementwise", ELEMENTWISE_IMAGE, &format!("attention_mma_paged_{head_dim}_bf16"))?;
        let split = self.kernel(&dev, "elementwise", ELEMENTWISE_IMAGE, &format!("attention_split_paged_pm_{head_dim}_bf16"))?;
        let mut allocated: Vec<u64> = Vec::new();
        let result = (|| -> Result<()> {
            let mut buffer = |dev: &mut crate::cuda::CudaDevice, n: usize| -> Result<u64> {
                let p = dev.alloc(n.max(4))?;
                allocated.push(p);
                Ok(p)
            };
            let (table, k, v, pos, rows, q, out, ws) = (buffer(&mut dev, 64)?, buffer(&mut dev, kv_dim * 2)?, buffer(&mut dev, kv_dim * 2)?, buffer(&mut dev, 4)?, buffer(&mut dev, 4)?, buffer(&mut dev, heads * head_dim * 2)?, buffer(&mut dev, heads * head_dim * 2)?, buffer(&mut dev, heads * 8 * (head_dim + 2) * 4)?);
            dev.htod(table, &vec![0xFFu8; 64])?; // sixteen entries of -1
            dev.htod(pos, &0i32.to_le_bytes())?;
            dev.htod(rows, &1i32.to_le_bytes())?;
            let unmapped: u64 = 0x7000_0000_0000;
            let (kv_dim_i, zero, layers_i, h, kvh, one, eight) = (kv_dim as i32, 0i32, 4i32, heads as i32, kv_heads as i32, 1i32, 8i32);
            let ptr = |p: &u64| p as *const u64 as *mut c_void;
            let int = |v: &i32| v as *const i32 as *mut c_void;
            // Safety: the argument lists mirror the kernels' signatures; the unmapped pool address is never dereferenced because every table entry is -1.
            unsafe {
                append.launch(&dev, (kv_dim.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &mut vec![ptr(&unmapped), ptr(&table), ptr(&k), ptr(&v), ptr(&pos), int(&kv_dim_i), int(&zero), int(&layers_i)])?;
                append_chunk.launch(&dev, (1, 1, 1), (256, 1, 1), 0, &mut vec![ptr(&unmapped), ptr(&table), ptr(&k), ptr(&v), ptr(&pos), ptr(&rows), int(&kv_dim_i), int(&zero), int(&layers_i)])?;
                let (blocks, kv_rows) = super::prefill_plan::mma_grid(1, heads, kv_heads).ok_or_else(|| anyhow!("heads do not divide over the KV heads"))?;
                mma.launch(&dev, (blocks as u32, kv_rows as u32, 1), (128, 1, 1), 0, &mut vec![ptr(&q), ptr(&unmapped), ptr(&out), ptr(&pos), int(&one), int(&h), int(&kvh), ptr(&table), int(&zero), int(&layers_i)])?;
                split.launch(&dev, ((kv_heads * 8) as u32, 1, 1), (128, 1, 1), 0, &mut vec![ptr(&q), ptr(&unmapped), ptr(&unmapped), ptr(&ws), ptr(&pos), int(&h), int(&kvh), int(&eight), ptr(&table), int(&zero), int(&layers_i)])?;
            }
            dev.synchronize()
        })();
        for p in allocated {
            let _ = dev.free(p);
        }
        result
    }
}
