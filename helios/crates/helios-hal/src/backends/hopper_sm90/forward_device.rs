//! The forward pass with everything on the device: activations, the per-layer K/V cache and
//! the RoPE tables live in GPU memory, every op is one launch on a device pointer (custom
//! kernels and cuBLAS share a stream, so nothing waits between ops), and only the final logits
//! travel back to the host. It computes exactly what the host-sliced path in `forward.rs`
//! computes, with the same kernels and the same rounding, and a hardware test requires the
//! logits to be bit-identical to that path.

use std::cell::RefCell;
use std::ffi::c_void;
use std::time::Instant;

use anyhow::{anyhow, Result};

use super::backend::{HopperBackend, ROW_BLOCK, RMSNORM_PTX};
use super::forward::Qwen2Dims;
use super::ops::ELEMENTWISE_PTX;
use super::decode_profile::DecodeProfile;
use crate::cuda::profile::StepMarks;
use crate::cuda::{CudaDevice, GraphExec, Kernel};
use crate::reference::{bf16_to_f32, f32_to_bf16, rope_tables};
use crate::sampling::{sample, Rng, SamplingConfig};
use crate::traits::StopReason;

const ATTENTION_BLOCK: u32 = 128;
const ELEMENT_BLOCK: u32 = 256;
const ROPE_BLOCK: u32 = 64;
/// Longest sequence the untiled attention kernel's shared memory allows.
const MAX_SEQ: usize = 8192;

/// One kernel argument.
enum Arg {
    Ptr(u64),
    Int(i32),
    Float(f32),
}

/// Launch `kernel`. Every argument is stored in an 8-byte slot and the kernel reads its own
/// width from the slot's first bytes (little endian), which is how the driver expects them.
fn launch(dev: &CudaDevice, kernel: Kernel, grid: u32, block: u32, shared: u32, args: &[Arg]) -> Result<()> {
    let slots: Vec<u64> = args
        .iter()
        .map(|a| match a {
            Arg::Ptr(p) => *p,
            Arg::Int(v) => u64::from(*v as u32),
            Arg::Float(v) => u64::from(v.to_bits()),
        })
        .collect();
    let mut params: Vec<*mut c_void> = slots.iter().map(|s| s as *const u64 as *mut c_void).collect();
    // Safety: every call site passes the arguments in the kernel's signature order.
    unsafe { kernel.launch(dev, (grid, 1, 1), (block, 1, 1), shared, &mut params) }
}

fn grid_for(n: usize) -> Result<u32> {
    Ok(u32::try_from(n.div_ceil(ELEMENT_BLOCK as usize))?)
}

/// Device pointers of one transformer layer's weights.
struct LayerWeights {
    input_norm: u64,
    q_w: u64,
    q_b: u64,
    k_w: u64,
    k_b: u64,
    v_w: u64,
    v_b: u64,
    o_w: u64,
    post_norm: u64,
    gate: u64,
    up: u64,
    down: u64,
}

struct Kernels {
    rmsnorm: Kernel,
    embedding: Kernel,
    add: Kernel,
    swiglu: Kernel,
    rope: Kernel,
    attention: Kernel,
    attention_decode: Kernel,
    kv_append: Kernel,
    bias_cast: Kernel,
}

/// Device buffers of one generation, freed together.
struct Arena {
    allocated: Vec<u64>,
}

impl Arena {
    fn alloc(&mut self, dev: &mut CudaDevice, bytes: usize) -> Result<u64> {
        let p = dev.alloc(bytes.max(1))?;
        self.allocated.push(p);
        Ok(p)
    }

    fn free(&mut self, dev: &mut CudaDevice) {
        for p in self.allocated.drain(..) {
            let _ = dev.free(p);
        }
    }
}

struct Buffers {
    ids: u64,
    positions: u64,
    h: u64,
    x: u64,
    q: u64,
    k: u64,
    v: u64,
    attn: u64,
    o: u64,
    gate: u64,
    up: u64,
    act: u64,
    down: u64,
    y_f32: u64,
    logits: u64,
    cos: u64,
    sin: u64,
    /// Per layer: (K cache, V cache), `[cap_seq, kv_heads, head_dim]` bf16 each.
    kv: Vec<(u64, u64)>,
}

/// How decode steps (one token after the prompt) are issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeMode {
    /// Every step launches its kernels one by one, the arguments carrying the sequence length.
    Direct,
    /// The first decode step runs directly (it also warms cuBLAS up); the next one is captured
    /// into a CUDA graph, and every later step replays that graph. The per-token values (token
    /// id, position) live in device memory, so the same graph is valid for every step.
    Graph,
}

impl HopperBackend {
    /// Greedy or sampled generation entirely on the device; see the module docs. Same contract as
    /// [`HopperBackend::generate_streaming_host`]. Decode steps are launched one by one.
    pub fn generate_streaming(
        &self,
        d: &Qwen2Dims,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        sampling: &SamplingConfig,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<StopReason> {
        self.run_generation(d, prompt, max_new, eos, sampling, on_token, DecodeMode::Direct, None, None)
    }

    /// Greedy generation of `max_new` tokens with the decode steps profiled; see [`DecodeProfile`].
    pub fn generate_profiled(&self, d: &Qwen2Dims, prompt: &[u32], max_new: usize, mode: DecodeMode) -> Result<DecodeProfile> {
        let mut profile = DecodeProfile::default();
        self.run_generation(d, prompt, max_new, None, &SamplingConfig::greedy(), &mut |_| {}, mode, None, Some(&mut profile))?;
        Ok(profile)
    }

    /// Like [`Self::generate_streaming`] with an explicit decode mode; when `trace` is given, the
    /// logits of every step (prompt first) are appended to it. For verification and measurement.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_traced(
        &self,
        d: &Qwen2Dims,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        sampling: &SamplingConfig,
        mode: DecodeMode,
        trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<StopReason> {
        self.run_generation(d, prompt, max_new, eos, sampling, &mut |_| {}, mode, trace, None)
    }

    /// Logits of the last prompt token through the device pipeline (prefill only); the
    /// verification entry point that is compared bit for bit with the host path.
    pub fn forward_prefill_logits_device(&self, d: &Qwen2Dims, prompt: &[u32]) -> Result<Vec<f32>> {
        let mut trace = Vec::new();
        self.run_generation(d, prompt, 0, None, &SamplingConfig::greedy(), &mut |_| {}, DecodeMode::Direct, Some(&mut trace), None)?;
        trace.into_iter().next().ok_or_else(|| anyhow!("no logits produced"))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_generation(
        &self,
        d: &Qwen2Dims,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        sampling: &SamplingConfig,
        on_token: &mut dyn FnMut(u32),
        mode: DecodeMode,
        trace: Option<&mut Vec<Vec<f32>>>,
        profile: Option<&mut DecodeProfile>,
    ) -> Result<StopReason> {
        if prompt.is_empty() {
            return Err(anyhow!("no prompt tokens"));
        }
        let cap_seq = prompt.len() + max_new;
        if cap_seq > MAX_SEQ {
            return Err(anyhow!("prompt + max_new = {cap_seq} exceeds the untiled attention kernel's limit of {MAX_SEQ}"));
        }

        let mut dev = self.device.lock().expect("device mutex poisoned");
        {
            let mut cublas = self.cublas.lock().expect("cublas mutex poisoned");
            if cublas.is_none() {
                *cublas = Some(crate::cuda::Cublas::new(&dev)?);
            }
        }
        let kernels = Kernels {
            rmsnorm: self.kernel(&dev, "rmsnorm", RMSNORM_PTX, "rmsnorm_bf16")?,
            embedding: self.kernel(&dev, "elementwise", ELEMENTWISE_PTX, "embedding_gather_bf16")?,
            add: self.kernel(&dev, "elementwise", ELEMENTWISE_PTX, "add_bf16")?,
            swiglu: self.kernel(&dev, "elementwise", ELEMENTWISE_PTX, "swiglu_bf16")?,
            rope: self.kernel(&dev, "elementwise", ELEMENTWISE_PTX, "rope_bf16")?,
            attention: self.kernel(&dev, "elementwise", ELEMENTWISE_PTX, "attention_bf16")?,
            attention_decode: self.kernel(&dev, "elementwise", ELEMENTWISE_PTX, "attention_decode_bf16")?,
            kv_append: self.kernel(&dev, "elementwise", ELEMENTWISE_PTX, "kv_append_bf16")?,
            bias_cast: self.kernel(&dev, "elementwise", ELEMENTWISE_PTX, "bias_cast_bf16")?,
        };
        let (embed, final_norm, layers) = {
            let weights = self.weights.lock().expect("weights mutex poisoned");
            let weights = weights.as_ref().ok_or_else(|| anyhow!("no weights loaded"))?;
            let ptr = |name: &str| -> Result<u64> {
                weights.pointer(name).map(|(p, _)| p).ok_or_else(|| anyhow!("weight {name} is not loaded"))
            };
            let mut layers = Vec::with_capacity(d.layers);
            for l in 0..d.layers {
                let n = |part: &str| format!("layers.{l}.{part}");
                layers.push(LayerWeights {
                    input_norm: ptr(&n("input_layernorm.weight"))?,
                    q_w: ptr(&n("self_attn.q_proj.weight"))?,
                    q_b: ptr(&n("self_attn.q_proj.bias"))?,
                    k_w: ptr(&n("self_attn.k_proj.weight"))?,
                    k_b: ptr(&n("self_attn.k_proj.bias"))?,
                    v_w: ptr(&n("self_attn.v_proj.weight"))?,
                    v_b: ptr(&n("self_attn.v_proj.bias"))?,
                    o_w: ptr(&n("self_attn.o_proj.weight"))?,
                    post_norm: ptr(&n("post_attention_layernorm.weight"))?,
                    gate: ptr(&n("mlp.gate_proj.weight"))?,
                    up: ptr(&n("mlp.up_proj.weight"))?,
                    down: ptr(&n("mlp.down_proj.weight"))?,
                });
            }
            (ptr("embed_tokens.weight")?, ptr("norm.weight")?, layers)
        };

        let mut arena = Arena { allocated: Vec::new() };
        let result = self.generate_on_device(&mut dev, &mut arena, d, &kernels, embed, final_norm, &layers, prompt, max_new, cap_seq, eos, sampling, on_token, mode, trace, profile);
        arena.free(&mut dev);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_on_device(
        &self,
        dev: &mut CudaDevice,
        arena: &mut Arena,
        d: &Qwen2Dims,
        k: &Kernels,
        embed: u64,
        final_norm: u64,
        layers: &[LayerWeights],
        prompt: &[u32],
        max_new: usize,
        cap_seq: usize,
        eos: Option<u32>,
        sampling: &SamplingConfig,
        on_token: &mut dyn FnMut(u32),
        mode: DecodeMode,
        mut trace: Option<&mut Vec<Vec<f32>>>,
        mut profile: Option<&mut DecodeProfile>,
    ) -> Result<StopReason> {
        let (hidden, heads, kv_heads, hd) = (d.hidden, d.heads, d.kv_heads, d.head_dim);
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let t_max = prompt.len();
        let widest = d.intermediate.max(hidden).max(q_dim).max(kv_dim);

        let bf = 2usize;
        let b = Buffers {
            ids: arena.alloc(dev, t_max * 4)?,
            positions: arena.alloc(dev, t_max * 4)?,
            h: arena.alloc(dev, t_max * hidden * bf)?,
            x: arena.alloc(dev, t_max * hidden * bf)?,
            q: arena.alloc(dev, t_max * q_dim * bf)?,
            k: arena.alloc(dev, t_max * kv_dim * bf)?,
            v: arena.alloc(dev, t_max * kv_dim * bf)?,
            attn: arena.alloc(dev, t_max * q_dim * bf)?,
            o: arena.alloc(dev, t_max * hidden * bf)?,
            gate: arena.alloc(dev, t_max * d.intermediate * bf)?,
            up: arena.alloc(dev, t_max * d.intermediate * bf)?,
            act: arena.alloc(dev, t_max * d.intermediate * bf)?,
            down: arena.alloc(dev, t_max * hidden * bf)?,
            y_f32: arena.alloc(dev, t_max * widest * 4)?,
            logits: arena.alloc(dev, d.vocab * 4)?,
            cos: arena.alloc(dev, cap_seq * hd * bf)?,
            sin: arena.alloc(dev, cap_seq * hd * bf)?,
            kv: {
                let mut kv = Vec::with_capacity(d.layers);
                for _ in 0..d.layers {
                    kv.push((arena.alloc(dev, cap_seq * kv_dim * bf)?, arena.alloc(dev, cap_seq * kv_dim * bf)?));
                }
                kv
            },
        };
        let (cos, sin) = rope_tables(cap_seq, hd, d.rope_theta);
        let to_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        dev.htod(b.cos, &to_bytes(&cos))?;
        dev.htod(b.sin, &to_bytes(&sin))?;

        let cublas_guard = self.cublas.lock().expect("cublas mutex poisoned");
        let cublas = cublas_guard.as_ref().expect("created above");

        // The launches of one forward step over `t` tokens placed at `start..start + t`, with the
        // token ids and positions already in `b.ids` / `b.positions`. Nothing here allocates,
        // synchronises or reads back, so it can be recorded into a CUDA graph. With `dynamic`
        // (decode only, `t == 1`) the cache row and the attention range come from `b.positions`
        // instead of from launch arguments, so one recording stays valid for every position.
        let launches = |dev: &CudaDevice, t: usize, start: usize, dynamic: bool, marks: Option<&RefCell<StepMarks>>| -> Result<()> {
            let seq = start + t;
            let mark = |group: &'static str| -> Result<()> {
                match marks {
                    Some(m) => m.borrow_mut().mark(dev, group),
                    None => Ok(()),
                }
            };
            let (hidden_i, hd_i, heads_i, kv_heads_i) =
                (i32::try_from(hidden)?, i32::try_from(hd)?, i32::try_from(heads)?, i32::try_from(kv_heads)?);
            let ti = u32::try_from(t)?;
            let norm = |input: u64, weight: u64, output: u64, rows: u32| {
                launch(dev, k.rmsnorm, rows, ROW_BLOCK, ROW_BLOCK * 4, &[Arg::Ptr(input), Arg::Ptr(weight), Arg::Ptr(output), Arg::Int(hidden_i), Arg::Float(d.rms_eps)])
            };
            // y = bf16(x * W^T + bias), x [t, inn] -> out [t, out]
            let linear = |w: u64, bias: u64, x: u64, out_ptr: u64, out: usize, inn: usize| -> Result<()> {
                cublas.linear_bf16_f32(w, x, b.y_f32, t, out, inn)?;
                let n = t * out;
                launch(dev, k.bias_cast, grid_for(n)?, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.y_f32), Arg::Ptr(bias), Arg::Ptr(out_ptr), Arg::Int(i32::try_from(n)?), Arg::Int(i32::try_from(out)?)])
            };

            launch(dev, k.embedding, ti, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.ids), Arg::Ptr(embed), Arg::Ptr(b.h), Arg::Int(hidden_i)])?;
            mark("embedding")?;
            for (l, w) in layers.iter().enumerate() {
                norm(b.h, w.input_norm, b.x, ti)?;
                mark("norm")?;
                linear(w.q_w, w.q_b, b.x, b.q, q_dim, hidden)?;
                linear(w.k_w, w.k_b, b.x, b.k, kv_dim, hidden)?;
                linear(w.v_w, w.v_b, b.x, b.v, kv_dim, hidden)?;
                mark("qkv_proj")?;
                // RoPE in place: each thread owns a (i, i + half) pair, so reading and writing
                // the same buffer is safe.
                launch(dev, k.rope, ti * u32::try_from(heads)?, ROPE_BLOCK, 0, &[Arg::Ptr(b.q), Arg::Ptr(b.cos), Arg::Ptr(b.sin), Arg::Ptr(b.positions), Arg::Ptr(b.q), Arg::Int(heads_i), Arg::Int(hd_i)])?;
                launch(dev, k.rope, ti * u32::try_from(kv_heads)?, ROPE_BLOCK, 0, &[Arg::Ptr(b.k), Arg::Ptr(b.cos), Arg::Ptr(b.sin), Arg::Ptr(b.positions), Arg::Ptr(b.k), Arg::Int(kv_heads_i), Arg::Int(hd_i)])?;
                mark("rope")?;
                // Append this step's K and V to the layer's cache, then attend over it.
                let (k_cache, v_cache) = b.kv[l];
                if dynamic {
                    launch(dev, k.kv_append, grid_for(kv_dim)?, ELEMENT_BLOCK, 0, &[Arg::Ptr(k_cache), Arg::Ptr(v_cache), Arg::Ptr(b.k), Arg::Ptr(b.v), Arg::Ptr(b.positions), Arg::Int(i32::try_from(kv_dim)?)])?;
                    mark("kv_append")?;
                    launch(
                        dev,
                        k.attention_decode,
                        u32::try_from(heads)?,
                        ATTENTION_BLOCK,
                        u32::try_from((cap_seq + ATTENTION_BLOCK as usize) * 4)?,
                        &[Arg::Ptr(b.q), Arg::Ptr(k_cache), Arg::Ptr(v_cache), Arg::Ptr(b.attn), Arg::Ptr(b.positions), Arg::Int(i32::try_from(cap_seq)?), Arg::Int(heads_i), Arg::Int(kv_heads_i), Arg::Int(hd_i)],
                    )?;
                    mark("attention")?;
                } else {
                    let offset = (start * kv_dim * bf) as u64;
                    dev.dtod_async(k_cache + offset, b.k, t * kv_dim * bf)?;
                    dev.dtod_async(v_cache + offset, b.v, t * kv_dim * bf)?;
                    mark("kv_append")?;
                    launch(
                        dev,
                        k.attention,
                        ti * u32::try_from(heads)?,
                        ATTENTION_BLOCK,
                        u32::try_from((seq + ATTENTION_BLOCK as usize) * 4)?,
                        &[Arg::Ptr(b.q), Arg::Ptr(k_cache), Arg::Ptr(v_cache), Arg::Ptr(b.attn), Arg::Int(i32::try_from(seq)?), Arg::Int(heads_i), Arg::Int(kv_heads_i), Arg::Int(hd_i), Arg::Int(i32::try_from(start)?)],
                    )?;
                    mark("attention")?;
                }
                linear(w.o_w, 0, b.attn, b.o, hidden, q_dim)?;
                mark("o_proj")?;
                launch(dev, k.add, grid_for(t * hidden)?, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.h), Arg::Ptr(b.o), Arg::Ptr(b.h), Arg::Int(i32::try_from(t * hidden)?)])?;
                mark("residual")?;

                norm(b.h, w.post_norm, b.x, ti)?;
                mark("norm")?;
                linear(w.gate, 0, b.x, b.gate, d.intermediate, hidden)?;
                linear(w.up, 0, b.x, b.up, d.intermediate, hidden)?;
                mark("mlp_gate_up")?;
                launch(dev, k.swiglu, grid_for(t * d.intermediate)?, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.gate), Arg::Ptr(b.up), Arg::Ptr(b.act), Arg::Int(i32::try_from(t * d.intermediate)?)])?;
                mark("mlp_act")?;
                linear(w.down, 0, b.act, b.down, hidden, d.intermediate)?;
                mark("mlp_down")?;
                launch(dev, k.add, grid_for(t * hidden)?, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.h), Arg::Ptr(b.down), Arg::Ptr(b.h), Arg::Int(i32::try_from(t * hidden)?)])?;
                mark("residual")?;
            }

            // Final norm of the last row only, then the tied output head.
            let last_row = b.h + ((t - 1) * hidden * bf) as u64;
            norm(last_row, final_norm, b.x, 1)?;
            mark("norm")?;
            cublas.linear_bf16_f32(embed, b.x, b.logits, 1, d.vocab, hidden)?;
            mark("lm_head")
        };
        // Host to device: the token ids and their positions (the only per-step inputs).
        let upload = |dev: &CudaDevice, ids: &[u32], start: usize| -> Result<()> {
            let positions: Vec<u32> = (start as u32..(start + ids.len()) as u32).collect();
            dev.htod(b.ids, &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            dev.htod(b.positions, &positions.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
        };
        // Wait for the queued step and read its logits back, rounded to bf16 like the host path.
        let read_logits = |dev: &CudaDevice| -> Result<Vec<f32>> {
            let mut raw = vec![0u8; d.vocab * 4];
            dev.dtoh(&mut raw, b.logits)?;
            Ok(raw.chunks_exact(4).map(|c| bf16_to_f32(f32_to_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))).collect())
        };
        let download = |dev: &CudaDevice| -> Result<Vec<f32>> {
            dev.synchronize()?;
            read_logits(dev)
        };
        let step = |dev: &CudaDevice, ids: &[u32], start: usize| -> Result<Vec<f32>> {
            upload(dev, ids, start)?;
            launches(dev, ids.len(), start, false, None)?;
            download(dev)
        };

        let mut rng = Rng::new(sampling.seed);
        let mut logits = step(dev, prompt, 0)?;
        if let Some(trace) = trace.as_mut() {
            trace.push(logits.clone());
        }
        let mut position = prompt.len();
        let mut graph: Option<GraphExec> = None;
        let mut warmed_up = false;
        for produced in 0..max_new {
            let sampled_at = Instant::now();
            let next = sample(&logits, sampling, &mut rng);
            if produced > 0 {
                if let Some(p) = profile.as_mut() {
                    p.host_sample_ms += sampled_at.elapsed().as_secs_f64() * 1000.0;
                }
            }
            on_token(next);
            if Some(next) == eos {
                return Ok(StopReason::EndOfSequence);
            }
            if produced + 1 < max_new {
                let timed = profile.is_some() && !(mode == DecodeMode::Graph && !warmed_up);
                let t0 = Instant::now();
                upload(dev, &[next], position)?;
                let t1 = Instant::now();
                let marks = if timed && mode == DecodeMode::Direct { Some(RefCell::new(StepMarks::start(dev)?)) } else { None };
                if mode == DecodeMode::Graph {
                    if !warmed_up {
                        // Launched directly: cuBLAS may allocate its workspace for the one-token
                        // shapes on first use, which is not allowed while a capture is open.
                        launches(dev, 1, position, true, None)?;
                        warmed_up = true;
                    } else {
                        if graph.is_none() {
                            graph = Some(GraphExec::capture(dev, || launches(dev, 1, position, true, None))?);
                        }
                        graph.as_ref().expect("captured above").launch(dev)?;
                    }
                } else {
                    launches(dev, 1, position, false, marks.as_ref())?;
                }
                let t2 = Instant::now();
                dev.synchronize()?;
                let t3 = Instant::now();
                logits = read_logits(dev)?;
                let t4 = Instant::now();
                if timed {
                    if let Some(p) = profile.as_mut() {
                        let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1000.0;
                        p.steps += 1;
                        p.host_upload_ms += ms(t0, t1);
                        p.host_issue_ms += ms(t1, t2);
                        p.host_wait_ms += ms(t2, t3);
                        p.host_read_ms += ms(t3, t4);
                        if let Some(m) = &marks {
                            for (group, gpu_ms) in m.borrow().read()? {
                                p.add_gpu(group, gpu_ms);
                            }
                        }
                    }
                }
                if let Some(trace) = trace.as_mut() {
                    trace.push(logits.clone());
                }
                position += 1;
            }
        }
        Ok(StopReason::MaxTokens)
    }
}
