//! The decode session: everything a generation or a plan step needs, built once when the graphs are captured.
//!
//! Activations, the RoPE tables, the per-slot page tables, the cuBLAS workspace, the kernel modules and the CUDA graphs (one per prefill chunk
//! capacity, one per decode bucket) are created in [`Session::build`]. The K/V cache is not here: it is the page-major pool of ADR 0002, a virtual
//! range reserved before the graphs are captured (its base address is baked into them) and mapped once afterwards. A step afterwards only uploads
//! small inputs (positions, slots, token ids, page-table deltas), launches work and reads token ids back; it allocates nothing, loads no module
//! and captures no graph (the device refuses those while a step runs, see `cuda::work_guard`).
//!
//! A prompt is prefilled in chunks (`prefill_plan`): the activation buffers are sized for the largest chunk, not for the context. A decode step
//! is a bucket of rows (1, 2, 4, ... `max_batch`); row r belongs to slot `slots[r]` (-1: masked), reads its own page table and position, and its
//! result does not depend on the other rows. Page tables live on the device, one per slot, and are changed by the deltas of a plan.

use std::cell::RefCell;
use std::ffi::c_void;

use anyhow::{anyhow, Result};

use super::backend::{HopperBackend, ROW_BLOCK, RMSNORM_IMAGE};
use super::forward::Qwen2Dims;
use super::forward_device::DecodeMode;
use super::ops::ELEMENTWISE_IMAGE;
use super::ops_split_attention::split_count;
use super::prefill_plan::{capacities, largest_chunk, mma_grid, plan_chunks, PrefillError};
use crate::cuda::event::Event;
use crate::cuda::profile::StepMarks;
use crate::cuda::{Cublas, CudaDevice, GraphExec, Kernel};
use crate::reference::rope_tables;

pub(super) const ELEMENT_BLOCK: u32 = 256;
pub(super) const ROPE_BLOCK: u32 = 64;
pub(super) const ARGMAX_BLOCK: u32 = 1024;
/// Threads of the split-K decode attention block (the kernel's tile size).
pub(super) const SPLIT_BLOCK: u32 = 128;
/// Threads of the tensor-core prefill attention block (4 warps of 16 virtual rows).
pub(super) const MMA_ATTENTION_BLOCK: u32 = 128;
/// The fixed workspace given to cuBLAS, so none of its calls allocates.
const CUBLAS_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
/// Tokens per page of the pool; the kernels have it built in.
pub(super) const PAGE_TOKENS: usize = 64;
/// Entries of the per-row token ring the argmax kernel writes (indexed by position modulo this).
pub(super) const RING: usize = 64;
/// Table deltas a step may carry beyond one full table (a decode step adds at most one page per row).
const DELTA_SLACK: usize = 64;

/// What one run of [`Session::launches`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Step {
    /// One decode step of the bucket's rows: slot, position and input token come from the device.
    Decode,
    /// A prefill chunk of a fixed row count: positions and the number of real rows come from the device, so one recording serves every chunk of
    /// that capacity.
    Chunk,
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
    launch_grid(dev, kernel, (grid, 1), block, shared, args)
}

/// [`launch`] with a two-dimensional grid.
pub(super) fn launch_grid(dev: &CudaDevice, kernel: Kernel, grid: (u32, u32), block: u32, shared: u32, args: &[Arg]) -> Result<()> {
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
    unsafe { kernel.launch(dev, (grid.0, grid.1, 1), (block, 1, 1), shared, &mut params) }
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
    /// Per-head RMSNorm weights of Q and K (Qwen3); 0 when the model has none.
    q_norm: u64,
    k_norm: u64,
    /// Gate and up weight rows concatenated at load: one matrix multiply serves the MLP input of a decode step.
    gate_up: u64,
    o_w: u64,
    post_norm: u64,
    gate: u64,
    up: u64,
    down: u64,
}

pub(super) struct Kernels {
    rmsnorm: Kernel,
    /// `h += delta` followed by the RMSNorm of the new `h`, in one pass (bit-identical to `add` then `rmsnorm`).
    add_rmsnorm: Kernel,
    embedding: Kernel,
    add: Kernel,
    swiglu: Kernel,
    rope: Kernel,
    /// Prefill attention on tensor cores reading the pool through a page table (`attention_mma_paged_{head_dim}_bf16`).
    attention_mma_paged: Kernel,
    kv_append_chunk_paged: Kernel,
    bias_cast: Kernel,
    // Decode over slots.
    gather_inputs: Kernel,
    apply_deltas: Kernel,
    qkv_split: Kernel,
    swiglu_rows: Kernel,
    kv_append_rows: Kernel,
    attention_split_rows: Kernel,
    attention_combine_rows: Kernel,
    argmax_rows: Kernel,
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
    /// The fused Q|K|V output of a decode step, `[max_batch][q_dim + 2 * kv_dim]` bf16 values.
    qkv: u64,
    /// The fused gate|up output of a decode step, `[max_batch][2 * intermediate]` bf16 values.
    gate_up: u64,
    attn: u64,
    o: u64,
    gate: u64,
    up: u64,
    act: u64,
    down: u64,
    y_f32: u64,
    /// `[max_batch][vocab]` f32: row r holds the logits of decode row r (a prefill tail uses row 0).
    logits: u64,
    /// Per decode row and split: (max, sum of exp, weighted V) of the split-K decode attention.
    attn_ws: u64,
    cos: u64,
    sin: u64,
    /// `[max_batch]` i32: the slot of each decode row, -1 for a masked row.
    pub(super) slots: u64,
    /// `[max_batch]` u32: the token each slot feeds into its next step.
    pub(super) slot_token: u64,
    /// `[max_batch][RING]` u32: the tokens the argmax kernel chose, by row and position modulo [`RING`].
    pub(super) tokens_ring: u64,
    /// `[max_batch][stride]` i32: the page table of every slot (NO_PAGE = -1).
    pub(super) tables: u64,
    /// `[stride]` i32: the table of the slot a prefill chunk belongs to, copied in before the chunk (its graphs read this address).
    stage: u64,
    /// `(word, page)` pairs of the table deltas of a step, uploaded before the apply kernel runs.
    deltas: u64,
}

/// Bytes of device memory a session needs besides the weights and the KV pool: the activation buffers (sized for the larger of the largest
/// prefill chunk and the largest decode bucket), the decode-row buffers, the RoPE tables, the page tables and the cuBLAS workspace.
pub(super) fn session_bytes(d: &Qwen2Dims, splits: usize, chunk: usize, max_batch: usize, max_seq_len: usize) -> usize {
    let (q_dim, kv_dim) = (d.heads * d.head_dim, d.kv_heads * d.head_dim);
    let widest = d.intermediate.max(d.hidden).max(q_dim).max(kv_dim);
    let bf = 2;
    let rows = chunk.max(max_batch);
    let per_row = 8 // ids and positions
        + 4 * d.hidden * bf // h, x, o, down
        + 2 * q_dim * bf // q, attn
        + 2 * kv_dim * bf // k, v
        + 3 * d.intermediate * bf // gate, up, act
        + widest * 4; // y_f32
    let decode = max_batch * ((q_dim + 2 * kv_dim) * bf + 2 * d.intermediate * bf + d.vocab * 4 + d.heads * splits * (d.head_dim + 2) * 4 + 4 + 4 + RING * 4);
    let stride = max_seq_len.div_ceil(PAGE_TOKENS);
    let tables = (max_batch + 1) * stride * 4 + (stride + DELTA_SLACK) * 8;
    let rope = max_seq_len * 2 * d.head_dim * bf;
    rows * per_row + decode + tables + rope + CUBLAS_WORKSPACE_BYTES + 4
}

/// Everything a generation or a plan step needs; see the module docs.
pub(super) struct Session {
    pub(super) dims: Qwen2Dims,
    /// Longest sequence (prompt plus generated tokens) a slot's page table covers: the model's maximum.
    pub(super) capacity: usize,
    /// Largest prefill chunk: the row count the activation buffers were sized for.
    pub(super) chunk_rows: usize,
    /// Largest decode bucket, and the slots (one per row of it).
    pub(super) max_batch: usize,
    /// Base address of the pool's reserved range: baked into every captured graph.
    pool_base: u64,
    /// Pages of a slot's table (`capacity` in pages).
    pub(super) stride: usize,
    /// Pages the pool was mapped with (0 until `commit_kv`).
    pub(super) pool_pages: usize,
    /// Device memory the capture used beyond the buffers (graph memory, driver state), measured for the pool reserve.
    pub(super) capture_growth: u64,
    kernels: Kernels,
    layers: Vec<LayerWeights>,
    embed: u64,
    head: u64,
    final_norm: u64,
    pub(super) b: Buffers,
    splits: usize,
    /// One decode step of a bucket followed by the argmax that chains it to the next one, by bucket size.
    graph_decode: Vec<(usize, GraphExec)>,
    /// One prefill-chunk recording per chunk capacity (128, 256, ... `chunk_rows`).
    graph_chunks: Vec<(usize, GraphExec)>,
    /// Orders the host's reads of chained tokens (a small ring, reused).
    pub(super) events: Vec<Event>,
    /// Start and end of a timed window of decode steps.
    pub(super) window: (Event, Event),
    allocated: Vec<u64>,
}

// Safety: the kernel handles, device pointers, graphs and events in a session are only used behind the
// backend's mutexes (device first, then session), one step at a time.
unsafe impl Send for Session {}

impl Session {
    /// Build the session for the weights already loaded in `backend`: allocate every buffer, load the kernels, give cuBLAS its workspace, and warm up
    /// and capture the graphs against the pool base address. The pool is not mapped yet: the paged kernels skip a table of NO_PAGE entries, so the
    /// warm-up launches touch no pool memory. On failure the buffers allocated so far are freed.
    pub(super) fn build(backend: &HopperBackend, dev: &mut CudaDevice, dims: Qwen2Dims, max_seq_len: usize, max_batch: usize, pool_base: u64) -> Result<Session> {
        let (hd, heads, kv_heads) = (dims.head_dim, dims.heads, dims.kv_heads);
        // The split-K decode attention has kernels for head dimensions 64 and 128 and serves at most
        // 8 query heads per KV head; anything else is refused, never mis-computed.
        if !matches!(hd, 64 | 128) || kv_heads == 0 || heads % kv_heads != 0 || heads / kv_heads > 8 {
            return Err(anyhow!("decode attention supports head_dim 64 or 128 and at most 8 query heads per KV head; got head_dim {hd}, {heads} heads, {kv_heads} KV heads"));
        }
        // One block per (KV head, split): enough splits to give every SM a block, whatever the context.
        let splits = split_count(dev.multiprocessor_count(), kv_heads);
        let chunk_rows = largest_chunk(max_seq_len);
        let (free, total) = dev.memory_info()?;
        let need = session_bytes(&dims, splits, chunk_rows, max_batch, max_seq_len);
        if need as u64 > free {
            return Err(anyhow!("not enough free device memory for a decode session: {:.2} GiB needed for buffers, {:.2} GiB free after the weights", need as f64 / (1u64 << 30) as f64, free as f64 / (1u64 << 30) as f64));
        }
        tracing::info!(
            max_seq_len,
            chunk_rows,
            max_batch,
            session_gib = need as f64 / (1u64 << 30) as f64,
            free_gib = free as f64 / (1u64 << 30) as f64,
            total_gib = total as f64 / (1u64 << 30) as f64,
            "decode session sized"
        );
        let mut taken = Vec::new();
        match Self::build_inner(backend, dev, dims, max_seq_len, chunk_rows, max_batch, splits, pool_base, &mut taken) {
            Ok(session) => Ok(session),
            Err(error) => {
                for pointer in taken.drain(..) {
                    let _ = dev.free(pointer);
                }
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_inner(backend: &HopperBackend, dev: &mut CudaDevice, d: Qwen2Dims, capacity: usize, chunk_rows: usize, max_batch: usize, splits: usize, pool_base: u64, taken: &mut Vec<u64>) -> Result<Session> {
        let (hidden, heads, kv_heads, hd) = (d.hidden, d.heads, d.kv_heads, d.head_dim);
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let t = chunk_rows.max(max_batch); // rows of every activation buffer
        let bf = 2usize;
        let stride = capacity.div_ceil(PAGE_TOKENS);

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
            qkv: alloc(dev, max_batch * (q_dim + 2 * kv_dim) * bf)?,
            gate_up: alloc(dev, max_batch * 2 * d.intermediate * bf)?,
            attn: alloc(dev, t * q_dim * bf)?,
            o: alloc(dev, t * hidden * bf)?,
            gate: alloc(dev, t * d.intermediate * bf)?,
            up: alloc(dev, t * d.intermediate * bf)?,
            act: alloc(dev, t * d.intermediate * bf)?,
            down: alloc(dev, t * hidden * bf)?,
            y_f32: alloc(dev, t * widest * 4)?,
            logits: alloc(dev, max_batch * d.vocab * 4)?,
            attn_ws: alloc(dev, max_batch * heads * splits * (hd + 2) * 4)?,
            cos: alloc(dev, capacity * hd * bf)?,
            sin: alloc(dev, capacity * hd * bf)?,
            slots: alloc(dev, max_batch * 4)?,
            slot_token: alloc(dev, max_batch * 4)?,
            tokens_ring: alloc(dev, max_batch * RING * 4)?,
            tables: alloc(dev, max_batch * stride * 4)?,
            stage: alloc(dev, stride * 4)?,
            deltas: alloc(dev, (stride + DELTA_SLACK) * 8)?,
        };
        let workspace = alloc(dev, CUBLAS_WORKSPACE_BYTES)?;

        let (cos, sin) = rope_tables(capacity, hd, d.rope_theta);
        let to_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        dev.htod(b.cos, &to_bytes(&cos))?;
        dev.htod(b.sin, &to_bytes(&sin))?;
        // Every table starts empty (NO_PAGE = -1, all bits set), so the warm-up launches skip the pool.
        dev.htod(b.tables, &vec![0xFFu8; max_batch * stride * 4])?;
        dev.htod(b.stage, &vec![0xFFu8; stride * 4])?;
        dev.htod(b.slot_token, &vec![0u8; max_batch * 4])?;

        let mut cublas_slot = backend.cublas.lock().expect("cublas mutex poisoned");
        if cublas_slot.is_none() {
            *cublas_slot = Some(Cublas::new(dev)?);
        }
        let cublas = cublas_slot.as_ref().expect("created above");
        cublas.set_workspace(workspace, CUBLAS_WORKSPACE_BYTES)?;

        let kernel = |name: &str| backend.kernel(dev, "elementwise", ELEMENTWISE_IMAGE, name);
        let kernels = Kernels {
            rmsnorm: backend.kernel(dev, "rmsnorm", RMSNORM_IMAGE, "rmsnorm_bf16")?,
            add_rmsnorm: backend.kernel(dev, "rmsnorm", RMSNORM_IMAGE, "add_rmsnorm_bf16")?,
            embedding: kernel("embedding_gather_bf16")?,
            add: kernel("add_bf16")?,
            swiglu: kernel("swiglu_bf16")?,
            rope: kernel("rope_bf16")?,
            attention_mma_paged: kernel(&format!("attention_mma_paged_{hd}_bf16"))?,
            kv_append_chunk_paged: kernel("kv_append_chunk_paged_bf16")?,
            bias_cast: kernel("bias_cast_bf16")?,
            gather_inputs: kernel("gather_inputs_rows")?,
            apply_deltas: kernel("apply_page_deltas")?,
            qkv_split: kernel("qkv_split_rows_bf16")?,
            swiglu_rows: kernel("swiglu_rows_bf16")?,
            kv_append_rows: kernel("kv_append_rows_paged_bf16")?,
            attention_split_rows: kernel(&format!("attention_split_rows_{hd}_bf16"))?,
            attention_combine_rows: kernel("attention_combine_rows_bf16")?,
            argmax_rows: kernel("argmax_rows_bf16")?,
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
                // The fused matrices were laid out by the upload (`fused_groups`): Q, K, V (and their biases) and gate, up sit back to
                // back in one allocation each, and the separate tensors are views into them. The bias is 0 for a model without one.
                let group = |name: &str| weights.group(&n(name)).map(|(p, _)| p).ok_or_else(|| anyhow!("fused weights {} were not laid out by the upload", n(name)));
                let qkv_w = group("qkv.weight")?;
                let qkv_b = if d.qkv_bias { group("qkv.bias")? } else { 0 };
                let gate_up = group("gate_up.weight")?;
                layers.push(LayerWeights {
                    qkv_w,
                    qkv_b,
                    gate_up,
                    input_norm: ptr(&n("input_layernorm.weight"))?,
                    q_w: ptr(&n("self_attn.q_proj.weight"))?,
                    q_norm: if d.qk_norm { ptr(&n("self_attn.q_norm.weight"))? } else { 0 },
                    k_norm: if d.qk_norm { ptr(&n("self_attn.k_norm.weight"))? } else { 0 },
                    q_b: if d.qkv_bias { ptr(&n("self_attn.q_proj.bias"))? } else { 0 },
                    k_w: ptr(&n("self_attn.k_proj.weight"))?,
                    k_b: if d.qkv_bias { ptr(&n("self_attn.k_proj.bias"))? } else { 0 },
                    v_w: ptr(&n("self_attn.v_proj.weight"))?,
                    v_b: if d.qkv_bias { ptr(&n("self_attn.v_proj.bias"))? } else { 0 },
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
            max_batch,
            pool_base,
            stride,
            pool_pages: 0,
            capture_growth: 0,
            kernels,
            layers,
            embed,
            head,
            final_norm,
            b,
            splits,
            graph_decode: Vec::new(),
            graph_chunks: Vec::new(),
            events,
            window,
            allocated: Vec::new(),
        };
        session.capture_graphs(dev, cublas)?;
        session.allocated = std::mem::take(taken);
        Ok(session)
    }

    /// The decode buckets of this session, ascending.
    pub(super) fn buckets(&self) -> Vec<usize> {
        crate::decode_buckets::buckets(self.max_batch).expect("max_batch was checked at build")
    }

    /// Warm cuBLAS up with every shape a step uses (each prefill chunk capacity, each decode bucket), then capture the chunk graphs and the decode
    /// graphs and move them to the device. The graphs read tokens, positions, slots and the row count from device memory and the pool and tables
    /// from fixed addresses, so they are valid for every step. All page tables are empty here: the paged kernels skip them.
    fn capture_graphs(&mut self, dev: &CudaDevice, cublas: &Cublas) -> Result<()> {
        let sizes = capacities(self.chunk_rows);
        let buckets = self.buckets();
        for &rows in &sizes {
            self.upload_chunk(dev, &vec![0u32; rows], 0, 1, rows)?;
            self.launches(dev, cublas, rows, Step::Chunk, None)?;
            dev.synchronize()?;
        }
        for &bucket in &buckets {
            self.stage_rows(dev, &vec![(0i32, 0u32); bucket], bucket)?;
            self.launches(dev, cublas, bucket, Step::Decode, None)?;
            self.pick(dev, bucket)?;
            dev.synchronize()?;
        }
        let (before, _) = dev.memory_info()?;
        let mut chunk_graphs = Vec::with_capacity(sizes.len());
        for &rows in &sizes {
            self.upload_chunk(dev, &vec![0u32; rows], 0, 1, rows)?;
            let graph = GraphExec::capture(dev, || self.launches(dev, cublas, rows, Step::Chunk, None))?;
            graph.upload(dev)?;
            chunk_graphs.push((rows, graph));
        }
        let mut decode_graphs = Vec::with_capacity(buckets.len());
        for &bucket in &buckets {
            let (free_before, _) = dev.memory_info()?;
            self.stage_rows(dev, &vec![(0i32, 0u32); bucket], bucket)?;
            let graph = GraphExec::capture(dev, || {
                self.launches(dev, cublas, bucket, Step::Decode, None)?;
                self.pick(dev, bucket)
            })?;
            graph.upload(dev)?;
            let (free_after, _) = dev.memory_info()?;
            tracing::info!(bucket, graph_mib = free_before.saturating_sub(free_after) as f64 / (1u64 << 20) as f64, "decode graph captured");
            decode_graphs.push((bucket, graph));
        }
        dev.synchronize()?;
        let (after, _) = dev.memory_info()?;
        self.capture_growth = before.saturating_sub(after);
        self.graph_chunks = chunk_graphs;
        self.graph_decode = decode_graphs;
        Ok(())
    }

    /// Release the graphs, events and every device buffer.
    pub(super) fn free(&mut self, dev: &mut CudaDevice) {
        let _ = dev.synchronize();
        self.graph_decode.clear();
        self.graph_chunks.clear();
        self.events.clear();
        for pointer in self.allocated.drain(..) {
            let _ = dev.free(pointer);
        }
    }

    pub(super) fn decode_graph(&self, bucket: usize) -> Result<&GraphExec> {
        self.graph_decode.iter().find(|(b, _)| *b == bucket).map(|(_, g)| g).ok_or_else(|| anyhow!("no decode graph for a bucket of {bucket} rows was captured when the model was loaded"))
    }

    /// Tell the device which slot and position each decode row has: `rows` are `(slot, position)`, padded to `bucket` with masked rows (slot -1,
    /// position 0, a valid index for the RoPE tables). The input tokens are in `slot_token`.
    pub(super) fn stage_rows(&self, dev: &CudaDevice, rows: &[(i32, u32)], bucket: usize) -> Result<()> {
        let mut slots = vec![-1i32; bucket];
        let mut positions = vec![0u32; bucket];
        for (r, &(slot, position)) in rows.iter().enumerate() {
            slots[r] = slot;
            positions[r] = position;
        }
        dev.htod(self.b.slots, &slots.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        dev.htod(self.b.positions, &positions.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
    }

    /// Set the token `slot` feeds into its next decode step.
    pub(super) fn set_slot_token(&self, dev: &CudaDevice, slot: usize, token: u32) -> Result<()> {
        dev.htod(self.b.slot_token + (slot * 4) as u64, &token.to_le_bytes())
    }

    /// Apply page-table deltas given as `(word, page)` pairs (see `PageTableLayout::encode`), before the step that reads them.
    pub(super) fn apply_deltas(&self, dev: &CudaDevice, pairs: &[u32]) -> Result<()> {
        let n = pairs.len() / 2;
        if n == 0 {
            return Ok(());
        }
        if n > self.stride + DELTA_SLACK {
            return Err(anyhow!("{n} page-table deltas in one step; a step holds at most {}", self.stride + DELTA_SLACK));
        }
        dev.htod(self.b.deltas, &pairs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        launch(dev, self.kernels.apply_deltas, u32::try_from(n.div_ceil(ELEMENT_BLOCK as usize))?, ELEMENT_BLOCK, 0, &[Arg::Ptr(self.b.tables), Arg::Ptr(self.b.deltas), Arg::Int(i32::try_from(n)?)])
    }

    /// The f32 logits of decode row `row` after a step.
    pub(super) fn read_logits(&self, dev: &CudaDevice, row: usize) -> Result<Vec<f32>> {
        let mut raw = vec![0u8; self.dims.vocab * 4];
        dev.dtoh(&mut raw, self.b.logits + (row * self.dims.vocab * 4) as u64)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// The token the argmax kernel chose for `row` when it sampled at `position`.
    pub(super) fn read_token(&self, dev: &CudaDevice, row: usize, position: usize) -> Result<u32> {
        let mut raw = [0u8; 4];
        dev.dtoh(&mut raw, self.b.tokens_ring + ((row * RING + (position & (RING - 1))) * 4) as u64)?;
        Ok(u32::from_le_bytes(raw))
    }

    /// Choose the next token of every active row from the logits on the device and chain it to the next step.
    pub(super) fn pick(&self, dev: &CudaDevice, rows: usize) -> Result<()> {
        let b = &self.b;
        launch(dev, self.kernels.argmax_rows, u32::try_from(rows)?, ARGMAX_BLOCK, 0, &[Arg::Ptr(b.logits), Arg::Int(i32::try_from(self.dims.vocab)?), Arg::Ptr(b.slots), Arg::Ptr(b.slot_token), Arg::Ptr(b.tokens_ring), Arg::Ptr(b.positions)])
    }

    /// The launches of one forward step over `t` rows; see [`Step`]. For a decode step `t` is the bucket and the token ids are gathered from the
    /// slots on the device. Nothing here allocates, synchronises or reads back, so it can be recorded into a CUDA graph. A chunk ends after its
    /// last layer: the final norm and output head run on one row of it, by [`Session::logits_tail`].
    pub(super) fn launches(&self, dev: &CudaDevice, cublas: &Cublas, t: usize, step: Step, marks: Option<&RefCell<StepMarks>>) -> Result<()> {
        let (k, b, d, splits) = (&self.kernels, &self.b, &self.dims, self.splits);
        let decode = step == Step::Decode;
        let chunk = !decode;
        let (hidden, heads, kv_heads, hd) = (d.hidden, d.heads, d.kv_heads, d.head_dim);
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let bf = 2usize;
        let mark = |group: &'static str| -> Result<()> {
            match marks {
                Some(m) => m.borrow_mut().mark(dev, group),
                None => Ok(()),
            }
        };
        let (hidden_i, hd_i, heads_i, kv_heads_i) = (i32::try_from(hidden)?, i32::try_from(hd)?, i32::try_from(heads)?, i32::try_from(kv_heads)?);
        let (stride_i, layers_i, kv_dim_i) = (i32::try_from(self.stride)?, i32::try_from(d.layers)?, i32::try_from(kv_dim)?);
        let ti = u32::try_from(t)?;
        let norm = |input: u64, weight: u64, output: u64, rows: u32| {
            launch(dev, k.rmsnorm, rows, ROW_BLOCK, ROW_BLOCK * 4, &[Arg::Ptr(input), Arg::Ptr(weight), Arg::Ptr(output), Arg::Int(hidden_i), Arg::Float(d.rms_eps)])
        };
        let norm_heads = |x: u64, weight: u64, rows: u32| {
            launch(dev, k.rmsnorm, rows, ROW_BLOCK, ROW_BLOCK * 4, &[Arg::Ptr(x), Arg::Ptr(weight), Arg::Ptr(x), Arg::Int(hd_i), Arg::Float(d.rms_eps)])
        };
        // h = bf16(h + delta), then x = RMSNorm(h) * weight: the residual add and the norm that follows it.
        let add_norm = |delta: u64, weight: u64| {
            launch(dev, k.add_rmsnorm, ti, ROW_BLOCK, ROW_BLOCK * 4, &[Arg::Ptr(b.h), Arg::Ptr(delta), Arg::Ptr(weight), Arg::Ptr(b.x), Arg::Int(hidden_i), Arg::Float(d.rms_eps)])
        };
        // y = bf16(x * W^T + bias), x [t, inn] -> out [t, out]
        let linear = |w: u64, bias: u64, x: u64, out_ptr: u64, out: usize, inn: usize| -> Result<()> {
            cublas.linear_bf16_f32(w, x, b.y_f32, t, out, inn)?;
            let n = t * out;
            launch(dev, k.bias_cast, grid_for(n)?, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.y_f32), Arg::Ptr(bias), Arg::Ptr(out_ptr), Arg::Int(i32::try_from(n)?), Arg::Int(i32::try_from(out)?)])
        };

        // A decode step's projections go through cuBLASLt when it is available: bf16 out, bias in the epilogue, one launch, the heuristic's
        // first algorithm for the shape (M = the bucket).
        let lt = if decode { cublas.lt() } else { None };
        // `project` is `linear` for a step that is not a decode step or when cuBLASLt is absent.
        let project = |w: u64, bias: u64, x: u64, out_ptr: u64, out: usize, inn: usize| -> Result<()> {
            match lt {
                Some(lt) => lt.linear_bf16(w, x, out_ptr, bias, t, out, inn),
                None => linear(w, bias, x, out_ptr, out, inn),
            }
        };

        if decode {
            launch(dev, k.gather_inputs, 1, 32, 0, &[Arg::Ptr(b.slots), Arg::Ptr(b.slot_token), Arg::Ptr(b.ids), Arg::Int(i32::try_from(t)?)])?;
        }
        launch(dev, k.embedding, ti, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.ids), Arg::Ptr(self.embed), Arg::Ptr(b.h), Arg::Int(hidden_i)])?;
        mark("embedding")?;
        norm(b.h, self.layers[0].input_norm, b.x, ti)?;
        mark("norm")?;
        for (l, w) in self.layers.iter().enumerate() {
            let layer_i = i32::try_from(l)?;
            // A decode step of one row: Q, K and V are the three consecutive parts of one fused projection. Several rows: the fused projection
            // is cut into Q, K and V rows by one small kernel.
            let (q_p, k_p, v_p) = if decode {
                let qkv_dim = q_dim + 2 * kv_dim;
                project(w.qkv_w, w.qkv_b, b.x, b.qkv, qkv_dim, hidden)?;
                if t == 1 {
                    (b.qkv, b.qkv + (q_dim * bf) as u64, b.qkv + ((q_dim + kv_dim) * bf) as u64)
                } else {
                    launch_grid(dev, k.qkv_split, (grid_for(qkv_dim)?, ti), ELEMENT_BLOCK, 0, &[Arg::Ptr(b.qkv), Arg::Ptr(b.q), Arg::Ptr(b.k), Arg::Ptr(b.v), Arg::Int(i32::try_from(q_dim)?), Arg::Int(kv_dim_i)])?;
                    (b.q, b.k, b.v)
                }
            } else {
                linear(w.q_w, w.q_b, b.x, b.q, q_dim, hidden)?;
                linear(w.k_w, w.k_b, b.x, b.k, kv_dim, hidden)?;
                linear(w.v_w, w.v_b, b.x, b.v, kv_dim, hidden)?;
                (b.q, b.k, b.v)
            };
            mark("qkv_proj")?;
            if d.qk_norm {
                // Per-head RMSNorm of Q and K before RoPE (Qwen3): the row norm kernel with head_dim as the row width, in place.
                norm_heads(q_p, w.q_norm, ti * u32::try_from(heads)?)?;
                norm_heads(k_p, w.k_norm, ti * u32::try_from(kv_heads)?)?;
                mark("qk_norm")?;
            }
            // RoPE in place: each thread owns a (i, i + half) pair, so reading and writing the same buffer is safe.
            launch(dev, k.rope, ti * u32::try_from(heads)?, ROPE_BLOCK, 0, &[Arg::Ptr(q_p), Arg::Ptr(b.cos), Arg::Ptr(b.sin), Arg::Ptr(b.positions), Arg::Ptr(q_p), Arg::Int(heads_i), Arg::Int(hd_i)])?;
            launch(dev, k.rope, ti * u32::try_from(kv_heads)?, ROPE_BLOCK, 0, &[Arg::Ptr(k_p), Arg::Ptr(b.cos), Arg::Ptr(b.sin), Arg::Ptr(b.positions), Arg::Ptr(k_p), Arg::Int(kv_heads_i), Arg::Int(hd_i)])?;
            mark("rope")?;
            // Append this step's K and V to the pool through the page table(s), then attend over it.
            if chunk {
                launch(dev, k.kv_append_chunk_paged, ti, ELEMENT_BLOCK, 0, &[Arg::Ptr(self.pool_base), Arg::Ptr(b.stage), Arg::Ptr(k_p), Arg::Ptr(v_p), Arg::Ptr(b.positions), Arg::Ptr(b.rows), Arg::Int(kv_dim_i), Arg::Int(layer_i), Arg::Int(layers_i)])?;
                mark("kv_append")?;
                // Tensor-core attention: one block per 64 (chunk row, query head) pairs of a KV head, all K/V tiles loaded once per group.
                let (blocks, kv_rows) = mma_grid(t, heads, kv_heads).ok_or_else(|| anyhow!("{heads} heads do not divide over {kv_heads} KV heads"))?;
                launch_grid(
                    dev,
                    k.attention_mma_paged,
                    (u32::try_from(blocks)?, u32::try_from(kv_rows)?),
                    MMA_ATTENTION_BLOCK,
                    0,
                    &[Arg::Ptr(q_p), Arg::Ptr(self.pool_base), Arg::Ptr(b.attn), Arg::Ptr(b.positions), Arg::Int(i32::try_from(t)?), Arg::Int(heads_i), Arg::Int(kv_heads_i), Arg::Ptr(b.stage), Arg::Int(layer_i), Arg::Int(layers_i)],
                )?;
                mark("attention")?;
            } else {
                launch_grid(dev, k.kv_append_rows, (grid_for(kv_dim)?, ti), ELEMENT_BLOCK, 0, &[Arg::Ptr(self.pool_base), Arg::Ptr(b.tables), Arg::Int(stride_i), Arg::Ptr(b.slots), Arg::Ptr(k_p), Arg::Ptr(v_p), Arg::Ptr(b.positions), Arg::Int(kv_dim_i), Arg::Int(layer_i), Arg::Int(layers_i)])?;
                mark("kv_append")?;
                launch_grid(
                    dev,
                    k.attention_split_rows,
                    (u32::try_from(kv_heads * splits)?, ti),
                    SPLIT_BLOCK,
                    0,
                    &[Arg::Ptr(q_p), Arg::Ptr(self.pool_base), Arg::Ptr(b.attn_ws), Arg::Ptr(b.positions), Arg::Int(heads_i), Arg::Int(kv_heads_i), Arg::Int(i32::try_from(splits)?), Arg::Ptr(b.tables), Arg::Int(stride_i), Arg::Ptr(b.slots), Arg::Int(layer_i), Arg::Int(layers_i)],
                )?;
                mark("attention")?;
                launch_grid(dev, k.attention_combine_rows, (u32::try_from(heads)?, ti), u32::try_from(hd)?, 0, &[Arg::Ptr(b.attn_ws), Arg::Ptr(b.attn), Arg::Ptr(b.positions), Arg::Ptr(b.slots), Arg::Int(heads_i), Arg::Int(hd_i), Arg::Int(i32::try_from(splits)?)])?;
                mark("attention_combine")?;
            }
            project(w.o_w, 0, b.attn, b.o, hidden, q_dim)?;
            mark("o_proj")?;
            add_norm(b.o, w.post_norm)?;
            mark("residual_norm")?;

            // A decode step of one row: gate and up are the two consecutive halves of one fused projection; several rows use the fused SwiGLU.
            if decode {
                project(w.gate_up, 0, b.x, b.gate_up, 2 * d.intermediate, hidden)?;
                mark("mlp_gate_up")?;
                if t == 1 {
                    launch(dev, k.swiglu, grid_for(d.intermediate)?, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.gate_up), Arg::Ptr(b.gate_up + (d.intermediate * bf) as u64), Arg::Ptr(b.act), Arg::Int(i32::try_from(d.intermediate)?)])?;
                } else {
                    launch_grid(dev, k.swiglu_rows, (grid_for(d.intermediate)?, ti), ELEMENT_BLOCK, 0, &[Arg::Ptr(b.gate_up), Arg::Ptr(b.act), Arg::Int(i32::try_from(d.intermediate)?)])?;
                }
            } else {
                linear(w.gate, 0, b.x, b.gate, d.intermediate, hidden)?;
                linear(w.up, 0, b.x, b.up, d.intermediate, hidden)?;
                mark("mlp_gate_up")?;
                launch(dev, k.swiglu, grid_for(t * d.intermediate)?, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.gate), Arg::Ptr(b.up), Arg::Ptr(b.act), Arg::Int(i32::try_from(t * d.intermediate)?)])?;
            }
            mark("mlp_act")?;
            project(w.down, 0, b.act, b.down, hidden, d.intermediate)?;
            mark("mlp_down")?;
            match self.layers.get(l + 1) {
                // The residual add and the next layer's input norm.
                Some(next) => add_norm(b.down, next.input_norm)?,
                None => launch(dev, k.add, grid_for(t * hidden)?, ELEMENT_BLOCK, 0, &[Arg::Ptr(b.h), Arg::Ptr(b.down), Arg::Ptr(b.h), Arg::Int(i32::try_from(t * hidden)?)])?,
            }
            mark("residual_norm")?;
        }

        if chunk {
            return Ok(());
        }
        // Final norm of every row, then the output head: logits [t, vocab].
        norm(b.h, self.final_norm, b.x, ti)?;
        mark("norm")?;
        cublas.linear_bf16_f32(self.head, b.x, b.logits, t, d.vocab, hidden)?;
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

    /// Final norm and output head of row `row` of the activations, leaving the logits in row 0 of `b.logits`.
    pub(super) fn logits_tail(&self, dev: &CudaDevice, cublas: &Cublas, row: usize) -> Result<()> {
        let (b, d) = (&self.b, &self.dims);
        let source = b.h + (row * d.hidden * 2) as u64;
        launch(dev, self.kernels.rmsnorm, 1, ROW_BLOCK, ROW_BLOCK * 4, &[Arg::Ptr(source), Arg::Ptr(self.final_norm), Arg::Ptr(b.x), Arg::Int(i32::try_from(d.hidden)?), Arg::Float(d.rms_eps)])?;
        cublas.linear_bf16_f32(self.head, b.x, b.logits, 1, d.vocab, d.hidden)
    }

    fn chunk_graph(&self, capacity: usize) -> Result<&GraphExec> {
        self.graph_chunks.iter().find(|(rows, _)| *rows == capacity).map(|(_, g)| g).ok_or_else(|| PrefillError::NoGraph { capacity }.into())
    }

    /// Copy the page table of `slot` to the staging table the chunk graphs read.
    fn stage_table(&self, dev: &CudaDevice, slot: usize) -> Result<()> {
        dev.dtod_async(self.b.stage, self.b.tables + (slot * self.stride * 4) as u64, self.stride * 4)
    }

    /// Run `ids` (positions `start..`) through the model for `slot` in chunks (see `prefill_plan`), appending their K and V to the slot's pages
    /// (which must be in its table), and leave the logits of the last token in row 0 of `b.logits`. When `sample` the greedy token of those logits
    /// is chosen too, stored as the slot's next input and in the row ring (at the position of the last token), and `positions[0]` is left
    /// at the position after it. Each chunk is a recorded graph (`Graph`) or direct launches (`Direct`); both run the same kernels in the same
    /// order. Synchronises before it returns.
    pub(super) fn prefill(&self, dev: &CudaDevice, cublas: &Cublas, slot: usize, ids: &[u32], start: usize, mode: DecodeMode, sample: bool) -> Result<()> {
        let chunks = plan_chunks(ids.len(), self.chunk_rows)?;
        self.stage_table(dev, slot)?;
        for c in &chunks {
            self.upload_chunk(dev, &ids[c.start..c.start + c.valid], start + c.start, c.valid, c.capacity)?;
            match mode {
                DecodeMode::Graph => self.chunk_graph(c.capacity)?.launch(dev)?,
                DecodeMode::Direct => self.launches(dev, cublas, c.capacity, Step::Chunk, None)?,
            }
            // The next chunk's inputs overwrite this one's: wait for it.
            dev.synchronize()?;
        }
        let last = chunks.last().expect("a non-empty prompt has a chunk");
        self.logits_tail(dev, cublas, last.valid - 1)?;
        if sample {
            self.stage_rows(dev, &[(i32::try_from(slot)?, u32::try_from(start + ids.len() - 1)?)], 1)?;
            self.pick(dev, 1)?;
        }
        dev.synchronize()
    }

    /// Give slot 0 the first `tokens.div_ceil(64)` pages of the pool, in order: how a generation outside a scheduler uses the pool (the
    /// scheduler and the `generate` path never share it). A typed refusal when the pool is not mapped or has fewer pages.
    pub(super) fn assign_slot0(&self, dev: &CudaDevice, tokens: usize) -> Result<()> {
        let pages = tokens.div_ceil(PAGE_TOKENS);
        if self.pool_pages == 0 {
            return Err(anyhow!("the KV pool is not mapped: run reserve_kv, capture_graphs and commit_kv before generating"));
        }
        if tokens > self.capacity || pages > self.pool_pages {
            return Err(PrefillError::ContextTooLong { requested: tokens, capacity: self.capacity.min(self.pool_pages * PAGE_TOKENS) }.into());
        }
        let pairs: Vec<u32> = (0..pages as u32).flat_map(|i| [i, i]).collect();
        self.apply_deltas(dev, &pairs)
    }

    /// [`Session::prefill`] of a whole prompt for `slot`, timed: `(GPU milliseconds between two events around it, wall milliseconds)`.
    pub(super) fn prefill_timed(&self, dev: &CudaDevice, cublas: &Cublas, slot: usize, ids: &[u32], mode: DecodeMode) -> Result<(f64, f64)> {
        let started = std::time::Instant::now();
        self.window.0.record(dev)?;
        self.prefill(dev, cublas, slot, ids, 0, mode, false)?;
        self.window.1.record(dev)?;
        self.window.1.wait()?;
        let wall = started.elapsed().as_secs_f64() * 1000.0;
        Ok((Event::elapsed_ms(&self.window.0, &self.window.1)?, wall))
    }

    /// Run one decode step of `bucket` rows already staged (`stage_rows`, `set_slot_token`): the recorded graph or direct launches.
    pub(super) fn decode_step(&self, dev: &CudaDevice, cublas: &Cublas, bucket: usize, mode: DecodeMode, marks: Option<&RefCell<StepMarks>>) -> Result<()> {
        match mode {
            DecodeMode::Graph => self.decode_graph(bucket)?.launch(dev),
            DecodeMode::Direct => {
                self.launches(dev, cublas, bucket, Step::Decode, marks)?;
                self.pick(dev, bucket)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen2_7b() -> Qwen2Dims {
        Qwen2Dims { layers: 28, hidden: 3584, intermediate: 18944, heads: 28, kv_heads: 4, head_dim: 128, vocab: 152064, rope_theta: 1e6, rms_eps: 1e-6, tied_head: false, qkv_bias: true, qk_norm: false }
    }

    fn qwen2_half_b() -> Qwen2Dims {
        Qwen2Dims { layers: 24, hidden: 896, intermediate: 4864, heads: 14, kv_heads: 2, head_dim: 64, vocab: 151936, rope_theta: 1e6, rms_eps: 1e-6, tied_head: true, qkv_bias: true, qk_norm: false }
    }

    #[test]
    fn the_session_budget_for_the_two_models_in_use_fits_the_hardware_they_run_on() {
        // 7B on an 80 GB card, the largest chunk (4096), a batch of 32 and the full 131072-token context; 0.5B at 32768 tokens.
        let big = session_bytes(&qwen2_7b(), 132, 4096, 32, 131_072);
        assert!(big < 3 << 30, "{} MiB", big >> 20);
        let small = session_bytes(&qwen2_half_b(), 132, 4096, 32, 32_768);
        assert!(small < 1 << 30, "{} MiB", small >> 20);
    }

    #[test]
    fn the_activation_cost_depends_on_the_chunk_and_the_batch_and_the_context_costs_only_its_tables() {
        let d = qwen2_7b();
        let chunk_small = session_bytes(&d, 132, 128, 1, 8192);
        let chunk_large = session_bytes(&d, 132, 4096, 1, 8192);
        assert!(chunk_large > chunk_small + (512 << 20), "a 4096-row chunk needs much more than a 128-row one");
        let context_small = session_bytes(&d, 132, 4096, 1, 8192);
        let context_large = session_bytes(&d, 132, 4096, 1, 131_072);
        // The context adds RoPE tables and a page table only (about 2 * head_dim * 2 bytes and 4 / 64 bytes per token): no KV cache in the session.
        assert!(context_large - context_small < 100 << 20, "{} MiB", (context_large - context_small) >> 20);
        assert!(session_bytes(&d, 132, 4096, 32, 8192) > session_bytes(&d, 132, 4096, 1, 8192), "decode rows add buffers");
    }
}
