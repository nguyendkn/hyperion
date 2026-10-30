# 0002: KV cache memory is one reserved virtual range per rank, addressed by page index

Status: **accepted** (owner review, 2026-10-30). Design only: nothing here is implemented, and the trait change it needs is made once, together with every other (ADR 0006, step C3). Plan: phase 12 step 6, phase 11 (tiered store), phase 13 D3; owner decision 17 (page-major layout).

## Context
- Today (`session.rs`) the K/V cache is one contiguous buffer per layer, `[capacity, kv_heads, head_dim]` bf16, allocated when the model loads, with `capacity = min(model maximum, what 90 percent of the free memory after the weights allows)` (the 8192 limit went with tiled prefill). The trait method `allocate_kv_cache(num_blocks, block_size) -> Vec<DevicePointer>` is not used by any real backend.
- That shape cannot serve what the plan needs: prefix sharing and tier moves (phase 07, 11) work on pages; continuous batching needs sequences to own pages, not a whole buffer; tensor parallelism needs the same page count on every rank; and the size must come from the memory that is really free after the weights and the graphs exist.
- Measured 2026-10-30 (H100, Qwen2.5-7B): 52.3 GiB free after the 14.19 GiB of weights and the session's activation buffers; capacity 131 072 tokens at 57 344 bytes of KV per token.

## Decision
1. **The order is fixed, and nothing is mapped while serving.** At load, in this order:
   1. weights are uploaded;
   2. **reserve** the virtual range for the whole KV pool (`cuMemAddressReserve`), with no physical memory behind it;
   3. **capture every graph** (prefill chunk capacities, decode batch buckets) against that range, so their kernels hold its base address;
   4. **measure** the free memory (`cuMemGetInfo`) now that the weights, buffers and graph memory exist;
   5. compute `pages = floor((free - margin) / page_bytes)`, `margin` being what the driver, cuBLAS and NCCL still need (measured: the free memory before and after capturing every graph, plus a fixed minimum, recorded in the ledger), rounded down to the allocation granule (`cuMemGetAllocationGranularity`);
   6. **map all physical memory at once** (`cuMemCreate`, `cuMemMap`, `cuMemSetAccess`);
   7. the engine is **Ready**.
   There is no map-on-demand, no growth and no shrinking after Ready: this is the "no new GPU work while serving" rule applied to memory. A pool that is full is a typed refusal (or an eviction the scheduler orders), never a mapping. **The trait expresses the order** (ADR 0006): reserving, capturing and committing are separate calls, and the type that allows a step exists only after the commit (type-state), so a backend cannot serve with an unmapped or half-mapped pool.
2. **Page-index addressing.** A page is `page_tokens` tokens, **64 by default** (a power of two, a load-time setting; decoupled from the hash block of ADR 0008, whose default is also 64). Kernels receive the pool base address and a device-resident page table; the address of a page is `base + index * page_bytes`. No `Vec<DevicePointer>`. The decode graphs read the page table from device memory that is updated by deltas (ADR 0007).
3. **Page-major layout (decision 17: DECIDED page-major, measured 2026-10-30).** A page holds every layer's K and V for its tokens: `[page][layer][k|v][token][kv_head][head_dim]`. A tier move (device to host, disk, another node) is one contiguous copy per page. **Budget: 5 percent.** Page-major is adopted when decode attention at 8k and 32k contexts is within 5 percent of the layer-major layout on an H100 (attention is 13 to 25 percent of the step there, so 5 percent of it is about 1 percent of the step); otherwise pages stay layer-major (each layer's slab its own pool) and tier moves pay per-layer batched copies. **Measurement `[RUN on H100]` (pod `pod-20261030-1047`, build `8dd5bf1`, 7B shape 28/4/128, split-K decode attention, one sweep over all layers, ms): context 8192: contiguous 1.470, layer-major pages 1.507, page-major 1.530, page-major / layer-major 1.0152; context 32768: 4.845, 4.976, 5.012, ratio 1.0073. Both inside the 5 percent budget, so the layout is page-major (rule applied by the run's authorization A5).** Page-major costs 1.5 percent and 0.7 percent of the attention time, below 0.4 percent of the decode step.
4. **Tensor parallel: identical page counts.** Every rank reserves the same `page_bytes` per token for its own KV heads, and the page count is the minimum over ranks of what each can hold after step 4, so one page table is valid on every rank (phase 13 D3). The minimum is agreed through the control plane before step 6.
5. **Ownership.** The pool belongs to the backend of the rank; `helios-core` sees page indices and counts only (rule 1). The prefix index (phase 07) stores page indices and hashes, never addresses.

## Why over the alternatives
- Per-sequence contiguous buffers (today): fragmentation, no sharing, no tiering.
- Many small `cuMemAlloc` pages: no contiguous virtual range, so kernels need a pointer per page, and allocation while serving is forbidden.
- A single big `cuMemAlloc` pool: cannot be sized after the graphs exist without a second allocation, and cannot be extended.
- Mapping on demand: an out-of-memory or a driver stall in the middle of serving, and a graph replay that races the map.

## Consequences
- The trait changes (ADR 0006): `allocate_kv_cache` is replaced by `reserve_kv`, graph capture against the reservation, and `commit_kv`.
- The session's contiguous K/V buffers, `kv_append_bf16` and the split-K and chunk attention change to page-table addressing: a kernel and graph rework with its own bit-identity and long-context checks (the 32768 kernel test exists).
- Prefill chunks write whole pages (a chunk capacity is a multiple of `page_tokens`).
- `HardwareInfo` carries the free amount at call time as well as the total.

## Verification plan (before the trait change is implemented)
`cuMemGetInfo` and granule size on the H100 (7B session); the margin measured over all graphs; a page-strided attention microbenchmark at 8k and 32k against layer-major, which decides the layout; map time for the whole pool; a full pool answered with a typed refusal.
