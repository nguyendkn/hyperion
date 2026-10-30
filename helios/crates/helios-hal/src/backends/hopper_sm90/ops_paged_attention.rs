//! Decode attention reading K and V through a page table, for the page-layout benchmark of ADR 0002: the same split-K kernel with three
//! ways of addressing a position (one contiguous buffer per layer; 64-token pages placed by a table inside each layer's pool, "layer-major";
//! one pool whose pages hold every layer, "page-major"). The data, the page order (a seeded shuffle, so pages are not contiguous) and the
//! arithmetic are the same in all three, so their outputs must be bit-identical and only the time differs. Not on any serving path.

use std::ffi::c_void;

use anyhow::{anyhow, Result};

use super::backend::HopperBackend;
use super::ops::ELEMENTWISE_IMAGE;
use super::ops_split_attention::split_count;

/// Tokens per page of the benchmark (ADR 0002's default `page_tokens`).
pub const PAGE_TOKENS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageLayout {
    /// One contiguous `[position][kv_head][head_dim]` buffer per layer, as the decode step uses today.
    Contiguous,
    /// Pages of 64 positions per layer, placed by the page table.
    LayerMajor,
    /// One pool; a page holds `[layer][k|v][token][kv_head][head_dim]`.
    PageMajor,
}

/// Result of [`HopperBackend::bench_decode_attention_layout`].
pub struct LayoutBench {
    /// Milliseconds for one sweep over all layers (one split and one combine launch per layer), median of the timed sweeps.
    pub ms_per_sweep: f64,
    /// The attention output of layer 0, to compare between layouts bit for bit.
    pub layer0_output: Vec<u16>,
}

fn noise(n: usize, seed: u32) -> Vec<u16> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            crate::reference::f32_to_bf16(((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * 2.0)
        })
        .collect()
}

/// A seeded permutation of `0..n` (Fisher-Yates over a 64-bit LCG).
fn shuffled(n: usize, seed: u64) -> Vec<i32> {
    let mut order: Vec<i32> = (0..n as i32).collect();
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    for i in (1..n).rev() {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        order.swap(i, (state >> 33) as usize % (i + 1));
    }
    order
}

impl HopperBackend {
    /// Time the decode attention of one query token over a context of `ctx` positions (a multiple of 64) in every one of `layers` layers,
    /// with K and V laid out as `layout`; `sweeps` timed sweeps after 3 warm-ups, CUDA events around each. `heads` query heads over `kv_heads`.
    #[allow(clippy::too_many_arguments)]
    pub fn bench_decode_attention_layout(&self, layout: PageLayout, ctx: usize, layers: usize, heads: usize, kv_heads: usize, head_dim: usize, sweeps: usize) -> Result<LayoutBench> {
        if ctx == 0 || ctx % PAGE_TOKENS != 0 || !matches!(head_dim, 64 | 128) || kv_heads == 0 || heads % kv_heads != 0 || heads / kv_heads > 8 || layers == 0 {
            return Err(anyhow!("unsupported benchmark shape: ctx {ctx}, {layers} layers, {heads}/{kv_heads} heads, head_dim {head_dim}"));
        }
        let mut dev = self.device.lock().expect("device mutex poisoned");
        let (split_name, split_args) = match layout {
            PageLayout::Contiguous => (format!("attention_split_{head_dim}_bf16"), 8),
            PageLayout::LayerMajor => (format!("attention_split_paged_lm_{head_dim}_bf16"), 11),
            PageLayout::PageMajor => (format!("attention_split_paged_pm_{head_dim}_bf16"), 11),
        };
        let split = self.kernel(&dev, "elementwise", ELEMENTWISE_IMAGE, &split_name)?;
        let combine = self.kernel(&dev, "elementwise", ELEMENTWISE_IMAGE, "attention_combine_bf16")?;
        let splits = split_count(dev.multiprocessor_count(), kv_heads);
        let (row, pages) = (kv_heads * head_dim, ctx / PAGE_TOKENS);
        let page_elems = PAGE_TOKENS * row;
        let to_bytes = |x: &[u16]| -> Vec<u8> { x.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let table = shuffled(pages, 7);

        let mut allocated: Vec<u64> = Vec::new();
        let result = (|| -> Result<LayoutBench> {
            let mut buffer = |dev: &mut crate::cuda::CudaDevice, bytes: usize| -> Result<u64> {
                let p = dev.alloc(bytes.max(1))?;
                allocated.push(p);
                Ok(p)
            };
            let q_ptr = buffer(&mut dev, heads * head_dim * 2)?;
            let out_ptr = buffer(&mut dev, heads * head_dim * 2)?;
            let ws_ptr = buffer(&mut dev, heads * splits * (head_dim + 2) * 4)?;
            let pos_ptr = buffer(&mut dev, 4)?;
            let table_ptr = buffer(&mut dev, pages * 4)?;
            dev.htod(q_ptr, &to_bytes(&noise(heads * head_dim, 1)))?;
            dev.htod(pos_ptr, &((ctx - 1) as i32).to_le_bytes())?;
            dev.htod(table_ptr, &table.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;

            // Per layer: (K base, V base); for page-major both are the pool.
            let mut bases: Vec<(u64, u64)> = Vec::with_capacity(layers);
            let pool = if layout == PageLayout::PageMajor { Some(buffer(&mut dev, pages * layers * 2 * page_elems * 2)?) } else { None };
            for layer in 0..layers {
                let k = noise(ctx * row, 100 + 2 * layer as u32);
                let v = noise(ctx * row, 101 + 2 * layer as u32);
                match layout {
                    PageLayout::Contiguous => {
                        let (kp, vp) = (buffer(&mut dev, ctx * row * 2)?, buffer(&mut dev, ctx * row * 2)?);
                        dev.htod(kp, &to_bytes(&k))?;
                        dev.htod(vp, &to_bytes(&v))?;
                        bases.push((kp, vp));
                    }
                    PageLayout::LayerMajor => {
                        let (kp, vp) = (buffer(&mut dev, ctx * row * 2)?, buffer(&mut dev, ctx * row * 2)?);
                        for (logical, &physical) in table.iter().enumerate() {
                            let at = physical as u64 * (page_elems * 2) as u64;
                            dev.htod(kp + at, &to_bytes(&k[logical * page_elems..(logical + 1) * page_elems]))?;
                            dev.htod(vp + at, &to_bytes(&v[logical * page_elems..(logical + 1) * page_elems]))?;
                        }
                        bases.push((kp, vp));
                    }
                    PageLayout::PageMajor => {
                        let pool = pool.expect("allocated above");
                        for (logical, &physical) in table.iter().enumerate() {
                            let slot = |which: usize| pool + (((physical as usize * layers + layer) * 2 + which) * page_elems * 2) as u64;
                            dev.htod(slot(0), &to_bytes(&k[logical * page_elems..(logical + 1) * page_elems]))?;
                            dev.htod(slot(1), &to_bytes(&v[logical * page_elems..(logical + 1) * page_elems]))?;
                        }
                        bases.push((pool, pool));
                    }
                }
            }

            let (h, kvh, hd, sp, lay) = (heads as i32, kv_heads as i32, head_dim as i32, splits as i32, layers as i32);
            let ptr_arg = |p: &u64| p as *const u64 as *mut c_void;
            let int_arg = |v: &i32| v as *const i32 as *mut c_void;
            let sweep = |dev: &crate::cuda::CudaDevice| -> Result<()> {
                for (layer, (kb, vb)) in bases.iter().enumerate() {
                    let layer_i = layer as i32;
                    let mut args: Vec<*mut c_void> = vec![ptr_arg(&q_ptr), ptr_arg(kb), ptr_arg(vb), ptr_arg(&ws_ptr), ptr_arg(&pos_ptr), int_arg(&h), int_arg(&kvh), int_arg(&sp)];
                    if split_args == 11 {
                        args.extend([ptr_arg(&table_ptr), int_arg(&layer_i), int_arg(&lay)]);
                    }
                    let mut combine_args: Vec<*mut c_void> = vec![ptr_arg(&ws_ptr), ptr_arg(&out_ptr), ptr_arg(&pos_ptr), int_arg(&h), int_arg(&hd), int_arg(&sp)];
                    // Safety: the argument lists mirror the kernels' signatures.
                    unsafe {
                        split.launch(dev, ((kv_heads * splits) as u32, 1, 1), (128, 1, 1), 0, &mut args)?;
                        combine.launch(dev, (heads as u32, 1, 1), (head_dim as u32, 1, 1), 0, &mut combine_args)?;
                    }
                }
                Ok(())
            };
            // The output of layer 0 alone, for the bit-identity check: run just that layer once.
            {
                let (kb, vb) = bases[0];
                let layer_i = 0i32;
                let mut args: Vec<*mut c_void> = vec![ptr_arg(&q_ptr), ptr_arg(&kb), ptr_arg(&vb), ptr_arg(&ws_ptr), ptr_arg(&pos_ptr), int_arg(&h), int_arg(&kvh), int_arg(&sp)];
                if split_args == 11 {
                    args.extend([ptr_arg(&table_ptr), int_arg(&layer_i), int_arg(&lay)]);
                }
                let mut combine_args: Vec<*mut c_void> = vec![ptr_arg(&ws_ptr), ptr_arg(&out_ptr), ptr_arg(&pos_ptr), int_arg(&h), int_arg(&hd), int_arg(&sp)];
                unsafe {
                    split.launch(&dev, ((kv_heads * splits) as u32, 1, 1), (128, 1, 1), 0, &mut args)?;
                    combine.launch(&dev, (heads as u32, 1, 1), (head_dim as u32, 1, 1), 0, &mut combine_args)?;
                }
                dev.synchronize()?;
            }
            let mut raw = vec![0u8; heads * head_dim * 2];
            dev.dtoh(&mut raw, out_ptr)?;
            let layer0_output = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

            for _ in 0..3 {
                sweep(&dev)?;
            }
            dev.synchronize()?;
            let (start, end) = (crate::cuda::event::Event::new_timed(&dev)?, crate::cuda::event::Event::new_timed(&dev)?);
            let mut times = Vec::with_capacity(sweeps);
            for _ in 0..sweeps {
                start.record(&dev)?;
                sweep(&dev)?;
                end.record(&dev)?;
                end.wait()?;
                times.push(crate::cuda::event::Event::elapsed_ms(&start, &end)?);
            }
            times.sort_by(|a, b| a.partial_cmp(b).expect("finite times"));
            Ok(LayoutBench { ms_per_sweep: times[times.len() / 2], layer0_output })
        })();
        for p in allocated {
            let _ = dev.free(p);
        }
        result
    }
}
