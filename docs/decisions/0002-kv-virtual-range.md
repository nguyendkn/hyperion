# 0002: KV cache memory is one reserved virtual range per rank, addressed by page index

Status: **proposed**, needs owner sign-off (not accepted, nothing implemented). Plan: phase 12 step 6, phase 11 (tiered store), phase 13 D3; owner decision 17 (page-major layout). Part of one trait design with ADR 0006: the trait change that this ADR needs is NOT made separately.

## Context
- Today (`session.rs`) the K/V cache is one contiguous buffer per layer, `[capacity, kv_heads, head_dim]` bf16, allocated when the model loads, with `capacity = min(model maximum, 8192, what 90 percent of the free memory after the weights allows)`. The trait method `allocate_kv_cache(num_blocks, block_size) -> Vec<DevicePointer>` (one pointer per block) is not used by any real backend.
- That shape cannot serve what the plan needs: prefix sharing and tier moves (phase 07, 11) work on pages; continuous batching needs sequences to own pages, not a whole buffer; tensor parallelism needs the same page count on every rank; and the size must come from the memory that is really free after the weights and the graphs exist, not from a total.
- Facts so far: `cuMemGetInfo` after the weights load has been written but never run on a GPU (ledger: `[NEEDS-GPU]`, closed by the 7B run). The 8192-token limit comes from the untiled prefill kernel (phase 12 step 7), not from memory.

## Decision (proposed)
1. **One virtual range per rank.** At load, after the weights and after the decode and prefill graphs were captured, reserve a virtual address range for the whole KV pool (`cuMemAddressReserve`) and back it with physical memory in granules (`cuMemCreate` + `cuMemMap`, granule from `cuMemGetAllocationGranularity`). The pool's size is `pages = floor((free - reserve) / page_bytes)` rounded down to the granule, where `free` comes from `cuMemGetInfo` and `reserve` is a driver margin (for graph memory growth, cuBLAS and NCCL buffers) that is measured, not guessed: the free memory before and after capturing every graph, plus a fixed minimum, recorded in the ledger. Physical pages are mapped on demand up to that size, so an idle pool costs nothing and growth never moves existing pages.
2. **Page-index addressing.** A page is a fixed number of tokens (`page_tokens`, a power of two). Kernels receive the pool base address and a device array of page indices per sequence; the address of a page is `base + index * page_bytes`. No `Vec<DevicePointer>`. The decode graphs read the page table from device buffers that the host updates before replay, like the position today.
3. **Page-major layout (answer to decision 17, recommended).** A page holds every layer's K and V for its tokens: `[page][layer][k|v][token][kv_head][head_dim]`. A tier move (device to host, to disk, to another node) is one contiguous copy per page instead of one per layer, and it is the layout NVIDIA's KV block manager uses, so a Dynamo worker can exchange pages without a transposition. Cost: attention reads one layer's keys as `page_tokens * kv_heads * head_dim` contiguous elements per page, strided by `page_bytes`. For Qwen2.5-7B with 64-token pages that is 64 KiB contiguous per (layer, page), large enough to stay efficient. **Condition: measure first.** Accept page-major only if decode attention at 8k and 32k contexts is within 5 percent of the layer-major layout on an H100; otherwise keep layer-major pages (each layer's slab is its own pool) and pay per-layer batched copies for tier moves.
4. **Tensor parallel: identical page counts.** Every rank reserves the same `page_bytes` per token for its own KV heads, and the pool page count is the minimum over ranks of what each can hold, so one page table is valid on every rank (phase 13 D3).
5. **Ownership.** The pool belongs to the backend of the rank; `helios-core` sees page indices and counts only (rule 1). The prefix index (phase 07) stores page indices and hashes; it never sees addresses.

## Why over the alternatives
- Per-sequence contiguous buffers (today): fragmentation, no sharing, no tiering.
- Many small `cuMemAlloc` pages: no contiguous virtual range, so kernels need a pointer per page, and allocation at serving time is forbidden by the "no new GPU work while serving" rule.
- A single big `cuMemAlloc` pool: works, but cannot grow or shrink physical backing and wastes memory when the free amount changes between loads.

## Consequences
- The trait changes (ADR 0006): `allocate_kv_cache` is replaced by planning and reserving a pool; this happens together with every other trait change, once.
- The session's contiguous K/V buffers, `kv_append_bf16` (row index from position) and the split-K attention (rows addressed by position) change to page-table addressing: a kernel and graph rework with its own bit-identity and long-context checks (kernel-level 32768 test already exists).
- Prefill chunks (phase 12 step 7) write whole pages.
- The memory report (`HardwareInfo`) must carry the free amount after load, not only the total.

## Open questions for the owner
- `page_tokens` (64 matches decision 10's candidate hash block; they stay decoupled either way).
- Is the 5 percent attention budget for page-major right?
- Decision 17 itself.

## Verification plan (before accepting)
`cuMemGetInfo` and granule size on the H100 (7B session); a page-strided attention microbenchmark at 8k and 32k against layer-major; map and unmap time per granule; behaviour when the pool is mapped to 100 percent and a sequence needs a page (a typed refusal, never a silent eviction).
