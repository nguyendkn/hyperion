//! The decode session: everything a generation needs, built once when the model is loaded.
//!
//! Activations, the per-layer K/V cache, the RoPE tables, the cuBLAS workspace, the kernel modules and
//! the CUDA graphs of a decode step are all created in [`Session::build`]. A generation afterwards only
//! uploads the prompt, launches work and reads token ids back; it allocates nothing, loads no module and
//! captures no graph (the device refuses those while a generation runs, see `cuda::work_guard`). The K/V
//! cache is reused between generations: stale rows beyond the current length are never read, because
//! attention is bounded by the position held on the device.
//!
//! A prompt is prefilled in chunks (`prefill_plan`): the activation buffers are sized for the largest chunk, not
//! for the context, and one graph per chunk capacity is captured at load. The capacity (longest prompt plus
//! generated tokens) is therefore the smaller of the model's maximum and what the free device memory allows
//! after the weights loaded; nothing else limits the context.

use std::cell::RefCell;
use std::ffi::c_void;

use anyhow::{anyhow, Result};

use super::backend::{HopperBackend, ROW_BLOCK, RMSNORM_IMAGE};
use super::forward::Qwen2Dims;
use super::ops::ELEMENTWISE_IMAGE;
use super::forward_device::DecodeMode;
use super::ops_split_attention::split_count;
use super::prefill_plan::{capacities, largest_chunk, plan_chunks, PrefillError};
use crate::cuda::event::Event;
use crate::cuda::profile::StepMarks;
use crate::cuda::{Cublas, CudaDevice, GraphExec, Kernel};
use crate::reference::{bf16_to_f32, f32_to_bf16, rope_tables};

pub(super) const ATTENTION_BLOCK: u32 = 128;
pub(super) const ELEMENT_BLOCK: u32 = 256;
pub(super) const ROPE_BLOCK: u32 = 64;
pub(super) const ARGMAX_BLOCK: u32 = 1024;
/// Threads of the split-K decode attention block (the kernel's tile size).
pub(super) const SPLIT_BLOCK: u32 = 128;
/// Threads of the chunked prefill attention block (the kernel's key tile).
pub(super) const CHUNK_ATTENTION_BLOCK: u32 = 128;
/// Shared floats of the chunked prefill attention kernel besides the query: a key tile of scores and 4 reduction slots.
const CHUNK_ATTENTION_FIXED_FLOATS: usize = 128 + 4;
/// The fixed workspace given to cuBLAS, so none of its calls allocates.
const CUBLAS_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
/// A model must fit at least this many tokens of context, or loading it is an error.
const MIN_CAPACITY: usize = 16;

/// What one run of [`Session::launches`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Step {
    /// One decode token: cache row and attention range come from `positions` on the device.
    Decode,
    /// A prefill chunk of a fixed row count: positions and the number of real rows come from the device, so one
    /// recording serves every chunk of that capacity.
    Chunk,
    /// The whole prompt in one launch at `start..start + t`, the attention kernel that needs shared memory
    /// proportional to the context. Kept as the reference the chunked path is tested against; limited to the
    /// largest chunk capacity by the activation buffers.
    Untiled { start: usize },
}

/// One kernel argument.
pub(super) enum Arg {
    Ptr(u64),
    Int(i32),
    Float(f32),
}

/// Launch `kernel`. Every argument is stored in an 8-byte slot and the kernel reads its own
/// width from the slot's first bytes (little endian), which is how the driver expects them.
pub(super) fn launch(dev: &CudaDevice, kernel: Kernel, grid: u32, block: u32, shared: u32, args: &[Arg]) -> Result<()> {
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
pub(super) struct LayerWeights {
    input_norm: u64,
    q_w: u64,
    q_b: u64,
    k_w: u64,
    k_b: u64,
    v_w: u64,
    v_b: u64,
    /// Q, K and V weight rows (and biases) concatenated at load: one matrix multiply serves a decode step.
    qkv_w: u64,
    qkv_b: u64,
    o_w: u64,
    post_norm: u64,
    gate: u64,
    up: u64,
    down: u64,
}

pub(super) struct Kernels {
    rmsnorm: Kernel,
    embedding: Kernel,
    add: Kernel,
    swiglu: Kernel,
    rope: Kernel,
    attention: Kernel,
    attention_split: Kernel,
    attention_combine: Kernel,
    attention_chunk: Kernel,
    kv_append: Kernel,
    kv_append_chunk: Kernel,
    argmax: Kernel,
    bias_cast: Kernel,
}

pub(super) struct Buffers {
    pub(super) ids: u64,
    pub(super) positions: u64,
    /// Number of real rows of the current prefill chunk (the rest is padding).
    rows: u64,
    h: u64,
    x: u64,
    q: u64,
    k: u64,
    v: u64,
    /// The fused Q|K|V output of a decode step (`q_dim + 2 * kv_dim` bf16 values); `q`, `k` and `v` are its parts.
    qkv: u64,
    attn: u64,
    o: u64,
    gate: u64,
    up: u64,
    act: u64,
    down: u64,
    y_f32: u64,
    logits: u64,
    /// Greedy tokens in the order they were chosen, written by the argmax kernel.
    pub(super) ring: u64,
    /// Where the first token of the current generation lands in `ring` (read by the argmax kernel).
    pub(super) ring_base: u64,
    /// Per query head and split: (max, sum of exp, weighted V) of the split-K decode attention.
    attn_ws: u64,
    cos: u64,
    sin: u64,
    /// Per layer: (K cache, V cache), `[capacity, kv_heads, head_dim]` bf16 each.
    kv: Vec<(u64, u64)>,
}

/// Bytes of device memory a session needs besides the weights, as `(fixed, per token of capacity)`. The activation
/// buffers are sized for `chunk` rows (the largest prefill chunk), so they are part of the fixed cost; what grows with
/// the context is the K/V cache, the RoPE tables and the token ring.
fn session_bytes(d: &Qwen2Dims, splits: usize, chunk: usize) -> (usize, usize) {
    let (q_dim, kv_dim) = (d.heads * d.head_dim, d.kv_heads * d.head_dim);
    let widest = d.intermediate.max(d.hidden).max(q_dim).max(kv_dim);
    let bf = 2;
    let per_chunk_row = 8 // ids and positions
        + 4 * d.hidden * bf // h, x, o, down
        + 2 * q_dim * bf // q, attn
        + 2 * kv_dim * bf // k, v
        + 3 * d.intermediate * bf // gate, up, act
        + widest * 4; // y_f32
    let per_token = 4 // ring
        + 2 * d.head_dim * bf // cos, sin
        + d.layers * 2 * kv_dim * bf; // K and V cache
    let qkv_dim = q_dim + 2 * kv_dim;
    // The concatenated copies of the weights a decode step reads, and the buffer it writes.
    let fused = d.layers * (qkv_dim * d.hidden * bf + qkv_dim * bf) + qkv_dim * bf;
    let fixed = chunk * per_chunk_row + fused + d.vocab * 4 + d.heads * splits * (d.head_dim + 2) * 4 + 4 + 4 + CUBLAS_WORKSPACE_BYTES;
    (fixed, per_token)
}

/// The longest sequence a session can hold: the model's maximum, or what fits in nine tenths of `free_bytes` (the
/// rest is left to the driver and cuBLAS), whichever is smaller. An error when not even [`MIN_CAPACITY`] tokens fit.
pub(super) fn plan_capacity(d: &Qwen2Dims, max_seq_len: usize, free_bytes: u64, splits: usize) -> Result<usize> {
    let (fixed, per_token) = session_bytes(d, splits, largest_chunk(max_seq_len));
    let budget = (free_bytes - free_bytes / 10) as usize;
    let by_memory = budget.saturating_sub(fixed) / per_token.max(1);
    let capacity = max_seq_len.min(by_memory);
    if capacity < MIN_CAPACITY {
        return Err(anyhow!(
            "not enough free device memory for a decode session: {:.2} GiB free after the weights, {:.2} GiB of fixed buffers and {} bytes per token of context are needed",
            free_bytes as f64 / (1u64 << 30) as f64,
            fixed as f64 / (1u64 << 30) as f64,
            per_token
        ));
    }
    Ok(capacity)
}

/// Everything a generation needs; see the module docs.
pub(super) struct Session {
    pub(super) dims: Qwen2Dims,
    /// Longest sequence (prompt plus generated tokens) the K/V cache holds.
    pub(super) capacity: usize,
    /// Largest prefill chunk: the row count the activation buffers were sized for.
    pub(super) chunk_rows: usize,
    kernels: Kernels,
    layers: Vec<LayerWeights>,
    embed: u64,
    head: u64,
    final_norm: u64,
    pub(super) b: Buffers,
    splits: usize,
    /// A decode step followed by the argmax that chains it to the next one.
    graph_chain: Option<GraphExec>,
    /// A decode step that leaves its logits for the host (sampling, tracing, profiling).
    graph_logits: Option<GraphExec>,
    /// One prefill-chunk recording per chunk capacity (128, 256, ... `chunk_rows`).
    graph_chunks: Vec<(usize, GraphExec)>,
    /// Orders the host's reads of chained tokens (a small ring, reused).
    pub(super) events: Vec<Event>,
    /// Start and end of a timed window of decode steps.
    pub(super) window: (Event, Event),
    allocated: Vec<u64>,
}

// Safety: the kernel handles, device pointers, graphs and events in a session are only used behind the
// backend's mutexes (device first, then session), one generation at a time.
unsafe impl Send for Session {}

impl Session {
    /// Build the session for the weights already loaded in `backend`: allocate every buffer, load the
    /// kernels, give cuBLAS its workspace, and warm up and capture the decode graphs. On failure the
    /// buffers allocated so far are freed.
    pub(super) fn build(backend: &HopperBackend, dev: &mut CudaDevice, dims: Qwen2Dims, max_seq_len: usize) -> Result<Session> {
        let (hd, heads, kv_heads) = (dims.head_dim, dims.heads, dims.kv_heads);
        // The split-K decode attention has kernels for head dimensions 64 and 128 and serves at most
        // 8 query heads per KV head; anything else is refused, never mis-computed.
        if !matches!(hd, 64 | 128) || kv_heads == 0 || heads % kv_heads != 0 || heads / kv_heads > 8 {
            return Err(anyhow!("decode attention supports head_dim 64 or 128 and at most 8 query heads per KV head; got head_dim {hd}, {heads} heads, {kv_heads} KV heads"));
        }
        // One block per (KV head, split): enough splits to give every SM a block, whatever the context.
        let splits = split_count(dev.multiprocessor_count(), kv_heads);
        let (free, total) = dev.memory_info()?;
        let capacity = plan_capacity(&dims, max_seq_len, free, splits)?;
        let chunk_rows = largest_chunk(max_seq_len);
        tracing::info!(
            capacity,
            chunk_rows,
            free_gib = free as f64 / (1u64 << 30) as f64,
            total_gib = total as f64 / (1u64 << 30) as f64,
            "decode session sized from the free device memory after the weights loaded"
        );
        let mut taken = Vec::new();
        match Self::build_inner(backend, dev, dims, capacity, chunk_rows, splits, &mut taken) {
            Ok(session) => Ok(session),
            Err(error) => {
                for pointer in taken.drain(..) {
                    let _ = dev.free(pointer);
                }
                Err(error)
            }
        }
    }

    fn build_inner(backend: &HopperBackend, dev: &mut CudaDevice, d: Qwen2Dims, capacity: usize, chunk_rows: usize, splits: usize, taken: &mut Vec<u64>) -> Result<Session> {
        let (hidden, heads, kv_heads, hd) = (d.hidden, d.heads, d.kv_heads, d.head_dim);
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let t = chunk_rows; // rows of every activation buffer
        let bf = 2usize;

        let mut alloc = |dev: &mut CudaDevice, bytes: usize| -> Result<u64> {
            let pointer = dev.alloc(bytes.max(1))?;
            taken.push(pointer);
            Ok(pointer)
        };
        let widest = d.intermediate.max(hidden).max(q_dim).max(kv_dim);
        let b = Buffers {
            ids: alloc(dev, t * 4)?,
            positions: alloc(dev, t * 4)?,
            rows: alloc(dev, 4)?,
            h: alloc(dev, t * hidden * bf)?,
            x: alloc(dev, t * hidden * bf)?,
            q: alloc(dev, t * q_dim * bf)?,
            k: alloc(dev, t * kv_dim * bf)?,
            v: alloc(dev, t * kv_dim * bf)?,
            qkv: alloc(dev, (q_dim + 2 * kv_dim) * bf)?,
            attn: alloc(dev, t * q_dim * bf)?,
            o: alloc(dev, t * hidden * bf)?,
            gate: alloc(dev, t * d.intermediate * bf)?,
            up: alloc(dev, t * d.intermediate * bf)?,
            act: alloc(dev, t * d.intermediate * bf)?,
            down: alloc(dev, t * hidden * bf)?,
            y_f32: alloc(dev, t * widest * 4)?,
            logits: alloc(dev, d.vocab * 4)?,
            ring: alloc(dev, capacity * 4)?,
            ring_base: alloc(dev, 4)?,
            attn_ws: alloc(dev, heads * splits * (hd + 2) * 4)?,
            cos: alloc(dev, capacity * hd * bf)?,
            sin: alloc(dev, capacity * hd * bf)?,
            kv: {
                let mut kv = Vec::with_capacity(d.layers);
                for _ in 0..d.layers {
                    kv.push((alloc(dev, capacity * kv_dim * bf)?, alloc(dev, capacity * kv_dim * bf)?));
                }
                kv
            },
        };
        let workspace = alloc(dev, CUBLAS_WORKSPACE_BYTES)?;

        let (cos, sin) = rope_tables(capacity, hd, d.rope_theta);
        let to_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        dev.htod(b.cos, &to_bytes(&cos))?;
        dev.htod(b.sin, &to_bytes(&sin))?;

        let mut cublas_slot = backend.cublas.lock().expect("cublas mutex poisoned");
        if cublas_slot.is_none() {
            *cublas_slot = Some(Cublas::new(dev)?);
        }
        let cublas = cublas_slot.as_ref().expect("created above");
        cublas.set_workspace(workspace, CUBLAS_WORKSPACE_BYTES)?;

        let kernels = Kernels {
            rmsnorm: backend.kernel(dev, "rmsnorm", RMSNORM_IMAGE, "rmsnorm_bf16")?,
            embedding: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, "embedding_gather_bf16")?,
            add: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, "add_bf16")?,
            swiglu: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, "swiglu_bf16")?,
            rope: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, "rope_bf16")?,
            attention: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, "attention_bf16")?,
            attention_split: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, &format!("attention_split_{hd}_bf16"))?,
            attention_combine: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, "attention_combine_bf16")?,
            attention_chunk: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, "attention_chunk_bf16")?,
            kv_append: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, "kv_append_bf16")?,
            kv_append_chunk: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, "kv_append_chunk_bf16")?,
            argmax: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, "argmax_chain_bf16")?,
            bias_cast: backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, "bias_cast_bf16")?,
        };
        let (embed, head, final_norm, layers) = {
            let weights = backend.weights.lock().expect("weights mutex poisoned");
            let weights = weights.as_ref().ok_or_else(|| anyhow!("no weights loaded"))?;
            let ptr = |name: &str| -> Result<u64> {
                weights.pointer(name).map(|(p, _)| p).ok_or_else(|| anyhow!("weight {name} is not loaded"))
            };
            let mut layers = Vec::with_capacity(d.layers);
            for l in 0..d.layers {
                let n = |part: &str| format!("layers.{l}.{part}");
                let (qkv_dim, hidden_bytes) = (q_dim + 2 * kv_dim, hidden * bf);
                let qkv_w = alloc(dev, qkv_dim * hidden_bytes)?;
                let qkv_b = alloc(dev, qkv_dim * bf)?;
                // Row-major [out, hidden] matrices stack by concatenating their bytes.
                let mut at = (0usize, 0usize);
                for (part, rows) in [("q_proj", q_dim), ("k_proj", kv_dim), ("v_proj", kv_dim)] {
                    dev.dtod_async(qkv_w + at.0 as u64, ptr(&n(&format!("self_attn.{part}.weight")))?, rows * hidden_bytes)?;
                    dev.dtod_async(qkv_b + at.1 as u64, ptr(&n(&format!("self_attn.{part}.bias")))?, rows * bf)?;
                    at = (at.0 + rows * hidden_bytes, at.1 + rows * bf);
                }
                layers.push(LayerWeights {
                    qkv_w,
                    qkv_b,
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
            (ptr("embed_tokens.weight")?, ptr(d.head_weight())?, ptr("norm.weight")?, layers)
        };

        let events = (0..4).map(|_| Event::new(dev)).collect::<Result<Vec<_>>>()?;
        let window = (Event::new_timed(dev)?, Event::new_timed(dev)?);
        let mut session = Session {
            dims: d,
            capacity,
            chunk_rows,
            kernels,
            layers,
            embed,
            head,
            final_norm,
            b,
            splits,
            graph_chain: None,
            graph_logits: None,
            graph_chunks: Vec::new(),
            events,
            window,
            allocated: Vec::new(),
        };
        session.capture_graphs(dev, cublas)?;
        session.allocated = std::mem::take(taken);
        Ok(session)
    }

    /// Warm cuBLAS up with every shape a generation uses (each prefill chunk capacity and the one-token step), then
    /// capture the chunk graphs and the two decode graphs and move them to the device. The graphs read the tokens,
    /// positions, row count and ring base from device memory, so they are valid for every generation and position.
    fn capture_graphs(&mut self, dev: &CudaDevice, cublas: &Cublas) -> Result<()> {
        let sizes = capacities(self.chunk_rows);
        for &rows in &sizes {
            self.upload_chunk(dev, &vec![0u32; rows], 0, 1, rows)?;
            self.launches(dev, cublas, rows, Step::Chunk, None)?;
            dev.synchronize()?;
        }
        // The one-token shapes, so cuBLAS has nothing left to set up when a capture is open.
        self.upload(dev, &[0u32], 0)?;
        dev.htod(self.b.ring_base, &0u32.to_le_bytes())?;
        self.launches(dev, cublas, 1, Step::Decode, None)?;
        self.pick(dev)?;
        dev.synchronize()?;
        let mut chunk_graphs = Vec::with_capacity(sizes.len());
        for &rows in &sizes {
            self.upload_chunk(dev, &vec![0u32; rows], 0, 1, rows)?;
            let graph = GraphExec::capture(dev, || self.launches(dev, cublas, rows, Step::Chunk, None))?;
            graph.upload(dev)?;
            chunk_graphs.push((rows, graph));
        }
        self.upload(dev, &[0u32], 0)?;
        let chain = GraphExec::capture(dev, || {
            self.launches(dev, cublas, 1, Step::Decode, None)?;
            self.pick(dev)
        })?;
        let logits = GraphExec::capture(dev, || self.launches(dev, cublas, 1, Step::Decode, None))?;
        chain.upload(dev)?;
        logits.upload(dev)?;
        dev.synchronize()?;
        self.graph_chunks = chunk_graphs;
        self.graph_chain = Some(chain);
        self.graph_logits = Some(logits);
        Ok(())
    }

    /// Release the graphs, events and every device buffer.
    pub(super) fn free(&mut self, dev: &mut CudaDevice) {
        let _ = dev.synchronize();
        self.graph_chain = None;
        self.graph_logits = None;
        self.graph_chunks.clear();
        self.events.clear();
        for pointer in self.allocated.drain(..) {
            let _ = dev.free(pointer);
        }
    }

    pub(super) fn chain_graph(&self) -> Result<&GraphExec> {
        self.graph_chain.as_ref().ok_or_else(|| anyhow!("the decode graph was not captured when the model was loaded"))
    }

    pub(super) fn logits_graph(&self) -> Result<&GraphExec> {
        self.graph_logits.as_ref().ok_or_else(|| anyhow!("the decode graph was not captured when the model was loaded"))
    }

    /// Host to device: the token ids and their positions (the only per-step inputs).
    pub(super) fn upload(&self, dev: &CudaDevice, ids: &[u32], start: usize) -> Result<()> {
        let positions: Vec<u32> = (start as u32..(start + ids.len()) as u32).collect();
        dev.htod(self.b.ids, &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        dev.htod(self.b.positions, &positions.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
    }

    /// The logits of the last step, read back and rounded to bf16 like the host path.
    pub(super) fn read_logits(&self, dev: &CudaDevice) -> Result<Vec<f32>> {
        let mut raw = vec![0u8; self.dims.vocab * 4];
        dev.dtoh(&mut raw, self.b.logits)?;
        Ok(raw.chunks_exact(4).map(|c| bf16_to_f32(f32_to_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))).collect())
    }

    /// Choose the next token from the logits on the device and chain it to the next step.
    pub(super) fn pick(&self, dev: &CudaDevice) -> Result<()> {
        let b = &self.b;
        launch(dev, self.kernels.argmax, 1, ARGMAX_BLOCK, 0, &[Arg::Ptr(b.logits), Arg::Int(i32::try_from(self.dims.vocab)?), Arg::Ptr(b.ids), Arg::Ptr(b.positions), Arg::Ptr(b.ring), Arg::Ptr(b.ring_base)])
    }

    /// The launches of one forward step over `t` rows, with the token ids and positions already in `ids` /
    /// `positions`; see [`Step`]. Nothing here allocates, synchronises or reads back, so it can be recorded into a
    /// CUDA graph. A decode step and a prefill chunk take the cache row and the attention range from `positions` on
    /// the device instead of from launch arguments, so one recording stays valid for every position. A chunk ends
    /// after its last layer: the final norm and output head run on one row of it, by [`Session::logits_tail`].
    pub(super) fn launches(&self, dev: &CudaDevice, cublas: &Cublas, t: usize, step: Step, marks: Option<&RefCell<StepMarks>>) -> Result<()> {
        let (k, b, d, splits) = (&self.kernels, &self.b, &self.dims, self.splits);
        let (start, dynamic, chunk) = match step {
            Step::Decode => (0, true, false),
            Step::Chunk => (0, false, true),
            Step::Untiled { start } => (start, false, false),
        };
        let (hidden, heads, kv_heads, hd) = (d.hidden, d.heads, d.kv_heads, d.head_dim);
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let bf = 2usize;
        let seq = start + t;
        let mark = |group: &'static str| -> Result<()> {
            match marks {
                Some(m) => m.borrow_mut().mark(dev, group),
                None => Ok(()),
            }
        };
        let (hidden_i, hd_i, heads_i, kv_heads_i) = (i32::try_from(hidden)?, i32::try_from(hd)?, i32::try_from(heads)?, i32::try_from(kv_heads)?);
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

        launch(dev, k.embedding, ti, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.ids), Arg::Ptr(self.embed), Arg::Ptr(b.h), Arg::Int(hidden_i)])?;
        mark("embedding")?;
        for (l, w) in self.layers.iter().enumerate() {
            norm(b.h, w.input_norm, b.x, ti)?;
            mark("norm")?;
            // A decode step is one row: Q, K and V are the three consecutive parts of one fused projection.
            let (q_p, k_p, v_p) = if dynamic {
                let qkv_dim = q_dim + 2 * kv_dim;
                cublas.linear_bf16_f32(w.qkv_w, b.x, b.y_f32, 1, qkv_dim, hidden)?;
                launch(dev, k.bias_cast, grid_for(qkv_dim)?, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.y_f32), Arg::Ptr(w.qkv_b), Arg::Ptr(b.qkv), Arg::Int(i32::try_from(qkv_dim)?), Arg::Int(i32::try_from(qkv_dim)?)])?;
                (b.qkv, b.qkv + (q_dim * bf) as u64, b.qkv + ((q_dim + kv_dim) * bf) as u64)
            } else {
                linear(w.q_w, w.q_b, b.x, b.q, q_dim, hidden)?;
                linear(w.k_w, w.k_b, b.x, b.k, kv_dim, hidden)?;
                linear(w.v_w, w.v_b, b.x, b.v, kv_dim, hidden)?;
                (b.q, b.k, b.v)
            };
            mark("qkv_proj")?;
            // RoPE in place: each thread owns a (i, i + half) pair, so reading and writing
            // the same buffer is safe.
            launch(dev, k.rope, ti * u32::try_from(heads)?, ROPE_BLOCK, 0, &[Arg::Ptr(q_p), Arg::Ptr(b.cos), Arg::Ptr(b.sin), Arg::Ptr(b.positions), Arg::Ptr(q_p), Arg::Int(heads_i), Arg::Int(hd_i)])?;
            launch(dev, k.rope, ti * u32::try_from(kv_heads)?, ROPE_BLOCK, 0, &[Arg::Ptr(k_p), Arg::Ptr(b.cos), Arg::Ptr(b.sin), Arg::Ptr(b.positions), Arg::Ptr(k_p), Arg::Int(kv_heads_i), Arg::Int(hd_i)])?;
            mark("rope")?;
            // Append this step's K and V to the layer's cache, then attend over it.
            let (k_cache, v_cache) = b.kv[l];
            if chunk {
                launch(dev, k.kv_append_chunk, ti, ELEMENT_BLOCK, 0, &[Arg::Ptr(k_cache), Arg::Ptr(v_cache), Arg::Ptr(k_p), Arg::Ptr(v_p), Arg::Ptr(b.positions), Arg::Ptr(b.rows), Arg::Int(i32::try_from(kv_dim)?)])?;
                mark("kv_append")?;
                launch(
                    dev,
                    k.attention_chunk,
                    ti * u32::try_from(heads)?,
                    CHUNK_ATTENTION_BLOCK,
                    u32::try_from((hd + CHUNK_ATTENTION_FIXED_FLOATS) * 4)?,
                    &[Arg::Ptr(q_p), Arg::Ptr(k_cache), Arg::Ptr(v_cache), Arg::Ptr(b.attn), Arg::Ptr(b.positions), Arg::Int(heads_i), Arg::Int(kv_heads_i), Arg::Int(hd_i)],
                )?;
                mark("attention")?;
            } else if dynamic {
                launch(dev, k.kv_append, grid_for(kv_dim)?, ELEMENT_BLOCK, 0, &[Arg::Ptr(k_cache), Arg::Ptr(v_cache), Arg::Ptr(k_p), Arg::Ptr(v_p), Arg::Ptr(b.positions), Arg::Int(i32::try_from(kv_dim)?)])?;
                mark("kv_append")?;
                launch(
                    dev,
                    k.attention_split,
                    u32::try_from(kv_heads * splits)?,
                    SPLIT_BLOCK,
                    0,
                    &[Arg::Ptr(q_p), Arg::Ptr(k_cache), Arg::Ptr(v_cache), Arg::Ptr(b.attn_ws), Arg::Ptr(b.positions), Arg::Int(heads_i), Arg::Int(kv_heads_i), Arg::Int(i32::try_from(splits)?)],
                )?;
                mark("attention")?;
                launch(dev, k.attention_combine, u32::try_from(heads)?, u32::try_from(hd)?, 0, &[Arg::Ptr(b.attn_ws), Arg::Ptr(b.attn), Arg::Ptr(b.positions), Arg::Int(heads_i), Arg::Int(hd_i), Arg::Int(i32::try_from(splits)?)])?;
                mark("attention_combine")?;
            } else {
                let offset = (start * kv_dim * bf) as u64;
                dev.dtod_async(k_cache + offset, k_p, t * kv_dim * bf)?;
                dev.dtod_async(v_cache + offset, v_p, t * kv_dim * bf)?;
                mark("kv_append")?;
                launch(
                    dev,
                    k.attention,
                    ti * u32::try_from(heads)?,
                    ATTENTION_BLOCK,
                    u32::try_from((seq + ATTENTION_BLOCK as usize) * 4)?,
                    &[Arg::Ptr(q_p), Arg::Ptr(k_cache), Arg::Ptr(v_cache), Arg::Ptr(b.attn), Arg::Int(i32::try_from(seq)?), Arg::Int(heads_i), Arg::Int(kv_heads_i), Arg::Int(hd_i), Arg::Int(i32::try_from(start)?)],
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

        if chunk {
            return Ok(());
        }
        // Final norm of the last row only, then the output head.
        let last_row = b.h + ((t - 1) * hidden * bf) as u64;
        norm(last_row, self.final_norm, b.x, 1)?;
        mark("norm")?;
        cublas.linear_bf16_f32(self.head, b.x, b.logits, 1, d.vocab, hidden)?;
        mark("lm_head")
    }

    /// Host to device for a prefill chunk of `capacity` rows: `ids` (real tokens, then zeros), their positions
    /// `start..`, padding rows repeating the last real position, and the number of real rows.
    pub(super) fn upload_chunk(&self, dev: &CudaDevice, ids: &[u32], start: usize, valid: usize, capacity: usize) -> Result<()> {
        let mut padded = ids.to_vec();
        padded.resize(capacity, 0);
        let positions: Vec<u32> = (0..capacity).map(|i| (start + i.min(valid - 1)) as u32).collect();
        let bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        dev.htod(self.b.ids, &bytes(&padded))?;
        dev.htod(self.b.positions, &bytes(&positions))?;
        dev.htod(self.b.rows, &(valid as i32).to_le_bytes())
    }

    /// Final norm and output head of row `row` of the activations, leaving the logits in `b.logits`.
    pub(super) fn logits_tail(&self, dev: &CudaDevice, cublas: &Cublas, row: usize) -> Result<()> {
        let (b, d) = (&self.b, &self.dims);
        let source = b.h + (row * d.hidden * 2) as u64;
        launch(dev, self.kernels.rmsnorm, 1, ROW_BLOCK, ROW_BLOCK * 4, &[Arg::Ptr(source), Arg::Ptr(self.final_norm), Arg::Ptr(b.x), Arg::Int(i32::try_from(d.hidden)?), Arg::Float(d.rms_eps)])?;
        cublas.linear_bf16_f32(self.head, b.x, b.logits, 1, d.vocab, d.hidden)
    }

    fn chunk_graph(&self, capacity: usize) -> Result<&GraphExec> {
        self.graph_chunks.iter().find(|(rows, _)| *rows == capacity).map(|(_, g)| g).ok_or_else(|| PrefillError::NoGraph { capacity }.into())
    }

    /// Run the prompt through the model in chunks (see `prefill_plan`) and leave the logits of its last token in
    /// `b.logits`. Each chunk is a recorded graph (`Graph`) or direct launches (`Direct`); both run the same kernels in
    /// the same order. Synchronises before it returns.
    pub(super) fn prefill(&self, dev: &CudaDevice, cublas: &Cublas, ids: &[u32], mode: DecodeMode) -> Result<()> {
        let chunks = plan_chunks(ids.len(), self.chunk_rows)?;
        for c in &chunks {
            self.upload_chunk(dev, &ids[c.start..c.start + c.valid], c.start, c.valid, c.capacity)?;
            match mode {
                DecodeMode::Graph => self.chunk_graph(c.capacity)?.launch(dev)?,
                DecodeMode::Direct => self.launches(dev, cublas, c.capacity, Step::Chunk, None)?,
            }
            // The next chunk's inputs overwrite this one's: wait for it.
            dev.synchronize()?;
        }
        let last = chunks.last().expect("a non-empty prompt has a chunk");
        self.logits_tail(dev, cublas, last.valid - 1)?;
        dev.synchronize()
    }

    /// The reference prefill: the whole prompt in one launch with the untiled attention kernel. Only for tests,
    /// and only for prompts that fit one chunk (the activation buffers' size).
    pub(super) fn prefill_untiled(&self, dev: &CudaDevice, cublas: &Cublas, ids: &[u32]) -> Result<()> {
        if ids.is_empty() || ids.len() > self.chunk_rows {
            return Err(anyhow!("the untiled reference prefill takes 1 to {} tokens, got {}", self.chunk_rows, ids.len()));
        }
        self.upload(dev, ids, 0)?;
        self.launches(dev, cublas, ids.len(), Step::Untiled { start: 0 }, None)?;
        dev.synchronize()
    }

    /// [`Session::prefill`] timed: `(GPU milliseconds between two events around it, wall milliseconds)`.
    pub(super) fn prefill_timed(&self, dev: &CudaDevice, cublas: &Cublas, ids: &[u32], mode: DecodeMode) -> Result<(f64, f64)> {
        let started = std::time::Instant::now();
        self.window.0.record(dev)?;
        self.prefill(dev, cublas, ids, mode)?;
        self.window.1.record(dev)?;
        self.window.1.wait()?;
        let wall = started.elapsed().as_secs_f64() * 1000.0;
        Ok((Event::elapsed_ms(&self.window.0, &self.window.1)?, wall))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen2.5-7B as published: 28 layers, 28 heads, 4 KV heads, head_dim 128, MLP 18944, vocab 152064.
    fn qwen2_7b() -> Qwen2Dims {
        Qwen2Dims { layers: 28, hidden: 3584, intermediate: 18944, heads: 28, kv_heads: 4, head_dim: 128, vocab: 152_064, rope_theta: 1e6, rms_eps: 1e-6, tied_head: false }
    }

    /// Qwen2.5-0.5B: 24 layers, 14 heads, 2 KV heads, head_dim 64, MLP 4864, vocab 151936.
    fn qwen2_half_b() -> Qwen2Dims {
        Qwen2Dims { layers: 24, hidden: 896, intermediate: 4864, heads: 14, kv_heads: 2, head_dim: 64, vocab: 151_936, rope_theta: 1e6, rms_eps: 1e-6, tied_head: true }
    }

    const GIB: u64 = 1 << 30;

    #[test]
    fn plenty_of_memory_gives_the_models_maximum_nothing_else_limits_the_context() {
        assert_eq!(plan_capacity(&qwen2_7b(), 131_072, 60 * GIB, 66).unwrap(), 131_072, "the model maximum binds");
        assert_eq!(plan_capacity(&qwen2_half_b(), 4096, 70 * GIB, 66).unwrap(), 4096, "the model maximum binds");
        // More than the old 8192 untiled limit: tiled prefill removed it.
        assert!(plan_capacity(&qwen2_7b(), 32_768, 60 * GIB, 66).unwrap() > 8192);
    }

    #[test]
    fn little_memory_limits_the_context_by_what_a_token_costs() {
        let d = qwen2_7b();
        let (fixed, per_token) = session_bytes(&d, 66, 4096);
        // KV alone is 28 layers x 2 x (4 x 128) x 2 bytes = 57 344 bytes per token; the RoPE tables and the token ring
        // add a few hundred. The activations are a fixed cost of the largest chunk, not of the context.
        assert!((57_344..58_500).contains(&per_token), "{per_token}");
        assert!(fixed > 800 << 20 && fixed < 3 << 30, "{fixed}");
        let free = (fixed as u64 + per_token as u64 * 1000) * 10 / 9 + 10; // room for about 1000 tokens
        let capacity = plan_capacity(&d, 131_072, free, 66).unwrap();
        assert!((990..=1010).contains(&capacity), "{capacity}");
    }

    #[test]
    fn a_device_that_cannot_hold_a_minimum_context_is_an_error_not_a_tiny_session() {
        // 40 MiB free: the 32 MiB cuBLAS workspace and the logits leave room for fewer than 16 tokens.
        let error = plan_capacity(&qwen2_7b(), 131_072, 40 << 20, 66).unwrap_err().to_string();
        assert!(error.contains("not enough free device memory"), "{error}");
    }

    #[test]
    fn the_session_budget_for_the_two_models_in_use_fits_the_hardware_they_run_on() {
        // The 7B weights take about 15.2 GB of an H100's 80, so well over 60 GB is free: the model's full 131 072 tokens fit.
        let (fixed, per_token) = session_bytes(&qwen2_7b(), 66, 4096);
        let total = fixed + per_token * 131_072;
        assert!(total < 12 * (1 << 30), "7B session at 131072 tokens: {:.2} GiB", total as f64 / (1u64 << 30) as f64);
        let (fixed, per_token) = session_bytes(&qwen2_half_b(), 66, 4096);
        assert!(fixed + per_token * 32_768 < 2 * (1 << 30));
    }

    #[test]
    fn the_activation_cost_depends_on_the_chunk_and_not_on_the_context() {
        let d = qwen2_7b();
        let (small, per_token_small) = session_bytes(&d, 66, 128);
        let (large, per_token_large) = session_bytes(&d, 66, 4096);
        assert_eq!(per_token_small, per_token_large);
        assert!(large > small + (800 << 20), "a 4096-row chunk needs far more activation memory than a 128-row one");
    }
}
