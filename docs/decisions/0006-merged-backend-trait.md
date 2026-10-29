# 0006: one merged change of the `ComputeBackend` trait

Status: **proposed**, needs owner sign-off (not accepted; the trait is NOT changed by this document). Plan: phase 05 (decoupling, capabilities), phase 12 step 6 (ADR 0002), phase 13 D3 (ranks, device group, communicator), phase 07 (StepPlan). Related: ADR 0004 (ranks), 0005 (manifest), 0007 (StepPlan), 0009 (NCCL).

## Context
The trait is the most stability-sensitive file in the workspace and changing it twice is the costly mistake. Everything that wants to change it is known now:
- `load_model_weights(&Path, &Manifest)` ties every backend to `v1::Manifest` (phase 05 requirement 4).
- `allocate_kv_cache`, `execute_prefill`, `execute_decode_step` were never wired on sm90: the real path is the optional `generate`. Their shape (one pointer per block, a step number) does not fit page-index KV (ADR 0002) or a scheduler that emits one plan per step (ADR 0007).
- Multi-GPU (owner scope, 2026-10-29) needs per-rank backends, a device group and graph-capturable collectives.
- Capabilities (arch, formats, memory kind) are today a table inside `triage.rs`, migrated to the backend by phase 05.
- `HardwareInfo` carries only the total device memory; KV sizing needs the free amount after load.

## Decision (proposed): one change, additive where possible, with the old unwired methods removed at the same time
```rust
pub trait ComputeBackend: Send + Sync {            // ONE INSTANCE PER RANK
    fn capabilities(&self) -> Capabilities;         // data, not allowlists: arch, formats, memory kind, max graph captures, peer access, collective kinds
    fn initialize_device(&self, rank: &RankInfo) -> Result<HardwareInfo>;   // HardwareInfo gains free_vram_bytes (at call time)
    fn load_model(&self, config: &BackendConfig, weights: &WeightSource) -> Result<()>;   // replaces load_model_weights(&Path, &Manifest)
    fn plan_kv(&self, spec: &KvSpec) -> Result<KvPlan>;       // ADR 0002: pages that fit after the weights and graphs
    fn reserve_kv(&self, plan: &KvPlan) -> Result<KvPool>;    // one virtual range, page-index addressed
    fn execute_step(&self, plan: &StepPlan) -> Result<StepOutput>;   // ADR 0007: replaces execute_prefill / execute_decode_step
    fn generate(&self, ...) -> Result<StopReason> { Err(HalError::Unsupported("generate")) }  // kept, default-implemented, until the scheduler uses StepPlan
    fn collective(&self) -> Option<&dyn Communicator> { None }   // graph-capturable all-reduce, all-gather, reduce-scatter, send/recv on the rank's stream
    fn shutdown(&self) -> Result<()> { Ok(()) }
}
```
- `BackendConfig` is a small struct (model dimensions, tensor format, target arch, payload files, deployment `{tp_size, pp_size, rank}`) built from `&Manifest` by `BackendConfig::from` inside `helios-hal`; the backend never sees `v1::Manifest`. `WeightSource` names the payload files (ADR 0003) and, under tensor parallelism, which slice of each tensor this rank reads (ADR 0005).
- `RankInfo { ordinal, node_rank, numa_node, peers }`, `DeviceGroup { ranks, topology }`: built by the factory from probing; a single-rank group is the default; `helios-core` sees ranks only through these types.
- Errors: `HalError::Unsupported(op)` is typed; nothing panics.
- `create_backend` keeps its shape (additive: it returns a `DeviceGroup` of backends where there was one); `helios-cli` is not edited by this change except where `create_backend`'s return type grows.

## Multi-rank stub for L0 tests
A stub backend with N ranks (one thread each) and CPU collectives that implement the real semantics (sum all-reduce, concatenating all-gather, and so on). It exists only to test the control plane, the StepPlan broadcast, sharding plans and failure handling (a rank dies, a rank is slow, rendezvous timeout); it never stands in for GPU results (rule: no fakes on main applies to claims, not to a test double that is named and documented as one; confirm with the owner).

## Why over the alternatives
- Two or three smaller trait changes (decouple now, KV later, ranks later): each is a breaking change for every backend and for `helios-core`; the plan's explicit warning.
- A new trait next to the old one: two contracts to keep in step, and the old one has no real implementation to protect.
- Keeping `Manifest` in the trait: blocks a second vendor and couples every backend to the signed on-disk format.

## Consequences
- Breaking for `helios-hal` backends (stub, sm90, sm120) and for the places in `helios-core` that call the removed methods (none of them reachable today: the scheduler uses `generate`).
- Backend internals (sessions, graphs, kernels) are untouched by this change; phase 12 steps 4, 7 and K stay inside the backend and land before it.
- Triage consumes `Capabilities` instead of `format_table()`; the committed JSON dump and the Ignition cross-check follow.

## Open questions for the owner
- Approve the sketch above as the single change (and the removal of the three unwired methods).
- Is a named test double (multi-rank stub with CPU collectives) acceptable under "no mocks or fakes", given what it may and may not be used to claim?
- Decisions 20 (shard at load) and 22 (NCCL only) shape `WeightSource` and `Communicator`.

## Verification plan
Compile-level test that a stub variant with different capabilities needs no edit under `helios-core`; the existing triage matrix migrated; a two-rank stub group exercised through the trait only; then the 2-GPU pod (phase 13 M1) for the real communicator.
