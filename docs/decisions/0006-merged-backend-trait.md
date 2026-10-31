# 0006: one merged change of the `ComputeBackend` trait

Status: **accepted** (owner review, 2026-10-30) as the design of the single change. **Implemented 2026-10-30 (autonomous run W1.1, commit `fa92cc9`)** exactly as designed: stub complete, sm90 implements `capabilities`, `initialize_device`, `load_model`, `generate`, `shutdown` and returns `HalError::Unsupported` for `reserve_kv`, `capture_graphs`, `commit_kv`, `submit_step`, `take_output` until the page-index KV pool (ADR 0002) exists; sm120 returns `Unsupported` for the chain. Plan: phase 05 (decoupling, capabilities), phase 12 step 6 (ADR 0002), phase 13 D3 (ranks, device group, communicator), phase 07 (StepPlan). Related: ADR 0004 (ranks), 0005 (manifest), 0007 (StepPlan), 0009 (NCCL).

## Context
The trait is the most stability-sensitive file in the workspace and changing it twice is the costly mistake. Everything that wants to change it is known now:
- `load_model_weights(&Path, &Manifest)` ties every backend to `v1::Manifest` (phase 05 requirement 4).
- `allocate_kv_cache`, `execute_prefill`, `execute_decode_step` were never wired on sm90: the real path is the optional `generate`. Their shape (one pointer per block, a step number) does not fit page-index KV (ADR 0002) or a scheduler that emits one plan per step (ADR 0007).
- Multi-GPU needs per-rank backends, a device group and graph-capturable collectives.
- Capabilities (arch, formats, memory kind) are a table inside `triage.rs`.
- `HardwareInfo` carries only the total device memory; KV sizing needs the free amount after load.

## Decision: one change, with the load order in the types
```rust
pub trait ComputeBackend: Send + Sync {            // ONE INSTANCE PER RANK
    fn capabilities(&self) -> Capabilities;         // data, not allowlists: arch, formats, memory kind, max graph captures, peer access, collective kinds
    fn initialize_device(&self, rank: &RankInfo) -> Result<HardwareInfo>;   // HardwareInfo gains free_vram_bytes (at call time)

    // The load order of ADR 0002 as a type-state chain: each step consumes or borrows the token the previous one made.
    fn load_model(&self, config: &BackendConfig, weights: &WeightSource) -> Result<LoadedModel>;     // replaces load_model_weights(&Path, &Manifest)
    fn reserve_kv(&self, model: &LoadedModel, spec: &KvSpec) -> Result<KvReservation>;                // a virtual range only: no physical memory
    fn capture_graphs(&self, model: &LoadedModel, kv: &KvReservation) -> Result<GraphSet>;            // prefill chunks, decode buckets; warms cuBLAS, NCCL
    fn commit_kv(&self, kv: KvReservation, graphs: &GraphSet, policy: &MemoryPolicy) -> Result<KvPool>; // measures the free memory, maps ALL of it, consumes the reservation

    // Serving needs a KvPool, which only commit_kv makes: a backend cannot serve with an unmapped or half-mapped pool, and there is
    // no call that maps memory afterwards.
    fn submit_step(&self, pool: &KvPool, plan: &StepPlan) -> Result<StepTicket>;      // enqueues the step; returns at once
    fn take_output(&self, ticket: StepTicket) -> Result<StepOutput>;                   // the result of a submitted step; may be asked for one step late

    fn generate(&self, ...) -> Result<StopReason> { Err(HalError::Unsupported("generate")) }  // kept, default-implemented, until the scheduler uses StepPlan
    fn collective(&self) -> Option<&dyn Communicator> { None }   // graph-capturable all-reduce, all-gather, reduce-scatter, send/recv on the rank's stream
    fn shutdown(&self) -> Result<()> { Ok(()) }
}
```
- **`submit_step` and `take_output`** replace `execute_step`. Submitting returns a ticket as soon as the step is enqueued; a row whose next input is the token the device just sampled carries no token id (ADR 0007: `FromDevice(slot)`), so the host does not wait to learn the token before enqueuing the next step. The host collects the output one step behind, exactly as the token ring of the sm90 decode chain does today. A backend that cannot overlap (the stub) completes inside `submit_step` and `take_output` returns at once; the contract allows both.
- `BackendConfig` is a small struct (model dimensions, tensor format, target arch, payload files, deployment `{tp_size, pp_size, rank}`) built from `&Manifest` by `BackendConfig::from` inside `helios-hal`; the backend never sees `v1::Manifest`. `WeightSource` names the payload files and, under tensor parallelism, which slice of each tensor this rank reads and how the fused groups are laid out (ADR 0005).
- `RankInfo { ordinal, node_rank, numa_node, peers }`, `DeviceGroup { ranks, topology }`: built by the factory from probing; a single-rank group is the default; `helios-core` sees ranks only through these types.
- Errors: `HalError::Unsupported(op)` is typed; nothing panics.
- `create_backend` keeps its shape (additive: it returns a `DeviceGroup` of backends where there was one); `helios-cli` is not edited by this change except where `create_backend`'s return type grows.

## Multi-rank stub for L0 tests (accepted)
A stub backend with N ranks (one thread each) and CPU collectives that implement the real semantics (sum all-reduce, concatenating all-gather, and so on) is acceptable as a **named test double**: it tests the control plane, the plan broadcast, sharding plans and failure handling (a rank dies, a rank is slow, rendezvous timeout, a version mismatch), and nothing else. Every result it produces is tagged **`[RUN on stub]`** in the ledger and in reports, a new tag in `docs/testing-layers.md`; it is never evidence about a GPU, and no claim of speed or numerics rests on it.

## Why over the alternatives
- Two or three smaller trait changes (decouple now, KV later, ranks later): each is a breaking change for every backend and for `helios-core`.
- A new trait next to the old one: two contracts to keep in step, and the old one has no real implementation to protect.
- Keeping `Manifest` in the trait: blocks a second vendor and couples every backend to the signed on-disk format.
- One `execute_step` that returns the output: forces the host to wait for the step before the next can be enqueued, and loses the on-GPU token chain.

## Consequences
- Breaking for `helios-hal` backends (stub, sm90, sm120) and for the places in `helios-core` that call the removed methods (none reachable today: the scheduler uses `generate`).
- Backend internals (sessions, graphs, kernels) are untouched by this change; phase 12 steps 4, 7 and K stay inside the backend and land before it (done).
- Triage consumes `Capabilities` instead of `format_table()`; the committed JSON dump and the Ignition cross-check follow.

## Verification plan
Compile-level test that a stub variant with different capabilities needs no edit under `helios-core`; a test that `submit_step` cannot be called without a `KvPool` (compile-fail test); the existing triage matrix migrated; a two-rank stub group exercised through the trait only (`[RUN on stub]`); then the 2-GPU pod (phase 13 M1) for the real communicator.

## Clarification 2026-10-31 (no trait change)
The multi-rank stub implements `Communicator` unchanged (`helios-hal/src/multi_rank_comm.rs`): a buffer is an opaque `u64`, a device address for NCCL and a handle into a host arena for the stub. Only the doc comment of the trait says so; no signature changed, so the narrow amendment of the 2026-10-31 run (A3) was not used. The stub moves `F32` buffers only; other element types are `HalError::Unsupported`.

## Amendment 2026-11-01: `KvSpec.max_batch` (data field, no method change; owner-approved)
`KvSpec` gains `max_batch: usize` (default 1): the largest decode bucket, a power of two from 1 to 32 (`--max-batch`). `reserve_kv` and `capture_graphs` read it: a decode graph is captured for each of 1, 2, 4, ... `max_batch` before the pool is mapped, so the graph memory is part of what `commit_kv` measures. No method signature of `ComputeBackend` or `Communicator` changes; a backend that has no batched path ignores the field and reports `Capabilities` accordingly. The bucket list itself is computed by `helios_hal::decode_buckets`, shared by the backend and the scheduler.
