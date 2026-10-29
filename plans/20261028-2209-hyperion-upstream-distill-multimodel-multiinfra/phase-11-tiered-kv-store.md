# Phase 11: Tiered KV store (GPU HBM, host RAM, NVMe)

## Context Links
- Promotes the phase 09 backlog row "Tiered cache offload". This file replaces that row's description; the row stays and points here.
- Depends on: phase 12 step 6 KV memory design (docs/decisions/0002-kv-virtual-range.md, incl. plan.md decision 17), phase 07 prefix cache (chained hash, pin counts, eviction, cache identity, `Restore` phase, transfer tickets, location-agnostic index), phase 12 step 7 (tiled prefill beyond 8192 tokens), decode batching (phase 12 step 8 GPU side + phase 07 scheduler side).
- Ideas only, no copied code: LMCache (Apache-2.0); upstream spark-local-inference `docs/prefix-cache-offload.md`, `docs/runtime-package-design.md` (memory planning), `helios/src/cache/hash.rs`.
- CLAUDE.md rules 1, 2; invariant "after Ready: no GPU allocation, no compilation, no capture".

## Overview
Priority P2. Status: gated (see Entry Criteria). Wave: W5 (plan.md). Depends: 07, 12 (steps 6, 7, 8). Effort: ~10-14d across sub-phases A-E.
Goal: let one Hyperion server use the whole machine for KV state. GPU HBM holds active and hot pages, NUMA-local pinned host RAM holds warm pages, NVMe holds cold pages and survives restarts. Each resource is used only where a measurement shows benefit, inside configured budgets.

## Key Insights
- This is a storage layer for KV pages, not an alternative cache. Every request computes with KV on the GPU; tiers decide where idle pages live.
- The placement axis is expected reuse and memory pressure, NOT prompt length. A 30k-token prompt used once gains nothing; a reused system prompt or a parked agent session gains a lot.
- The "hybrid" decision is per prefix and cost-based: restore from a tier when measured restore time < measured recompute (prefill) time for those tokens, otherwise recompute.
- Value grows with KV bytes per token (estimates, to be measured): Qwen2.5-0.5B ~12 KB/token, Qwen2-7B ~56 KB/token. An 80 GB H100 holds millions of tokens for small models, so this phase only pays off with larger models or many long sessions kept alive.
- Most of LMCache's reasons for an out-of-process sidecar (Python GIL, cross-engine sharing) do not apply to a single Rust process. A shared sidecar becomes relevant only when several engine processes on one node share one host cache (sub-phase F).
- On unified-memory systems (e.g. DGX Spark class), host RAM is the same physical pool: an L2 tier adds no capacity there. Capabilities must report memory kind, and L2 is disabled when memory is unified.

## Entry Criteria (all must hold, each with measured evidence; until then only sub-phase A may run)
1. Prerequisites landed: VMM-backed paged KV (phase 12 step 6 design implemented via phase 05 or its follow-up), phase 07 prefix cache with pin/evict and cache identity, tiled prefill > 8192 (phase 12 step 7), decode batching (phase 12 step 8 + phase 07).
2. Under a recorded agent-like workload on the target model, the GPU KV pool reaches its high watermark and evicts.
3. A measurable share of evicted prefixes is requested again later (report the reuse rate).
4. For the target model, measured restore (host RAM -> GPU) is faster than measured prefill for the same tokens.

## Hardware roles (what "use the whole server" means here)
| Resource | Role | Notes |
|---|---|---|
| GPU HBM | L1: active pages + hot cached prefixes | VMM range from phase 12 step 6; sized from free memory after weights, minus reserve and driver margin |
| GPU copy engines + PCIe | Async page moves on a dedicated low-priority stream | Must not slow decode beyond a declared bound (measured) |
| Host DRAM | L2: pinned page slab, NUMA-local to the GPU | Allocated once at startup; huge pages where available |
| CPU cores | I/O threads, hashing/checksums, tier policy | Pinned to cores on the GPU's NUMA node; never on the decode critical path |
| NVMe | L3: cold pages, persistence across restart | O_DIRECT + io_uring on Linux; striping across drives when several are configured |

## Requirements
1. **Location-agnostic index.** Block key = chained prefix hash (phase 07) under the cache identity (model id, layout fingerprint, payload id, target_arch, tensor_format, domain tag). Each entry records: parent, token ids (verified on hit to rule out hash collisions), residency set {gpu, host, disk}, pin count, last-access tick, hit count, byte size, checksum. Adding a tier never changes the key.
2. **Transfer unit = KV page across all layers.** Transfers move whole pages (target 1-4 MB per transfer). Requires the phase 12 step 6 layout decision below (plan.md decision 17).
3. **Per rank under TP (phase 13).** Each rank holds its own KV shard (split by KV heads). L2 slabs are per rank, in the NUMA node of that rank's GPU (HGX nodes usually split GPUs across two CPU sockets). The index and L3 are per engine. A block is restorable only if every rank holds its shard.
3. **L2 host tier.**
   - One pinned slab per GPU, allocated at startup on the GPU's NUMA node (`/sys/bus/pci/devices/<bdf>/numa_node`), registered once, split into page-sized slots.
   - Demotion: on GPU pressure, evict unpinned LRU leaves. Copy to host only when the reuse score is above a threshold; otherwise drop. Free the GPU page only after the copy event completes.
   - Promotion: on a host hit, admission plans a `Restore` (phase 07 phase and transfer ticket), copies host to GPU into reserved pages, then admits.
4. **L3 NVMe tier.**
   - Log-structured segment files (e.g. 256 MiB), preallocated, written with O_DIRECT and aligned buffers through L2 staging slots.
   - Write-behind only for pages above a reuse threshold, with a write budget per hour to limit drive wear.
   - Reads go NVMe -> L2 slot -> GPU.
   - Windows: positional I/O without io_uring (correct but slower); recorded in the ledger.
5. **Persistence.** Index snapshot plus segments survive restart. On startup, keep only entries whose cache identity matches the loaded package. Checksums are verified on read; a mismatch drops the entry and counts it.
6. **Restore-vs-recompute decision.** Uses measured bandwidths (per tier) and measured prefill rate (per model and capacity), refreshed from runtime counters. No hand-tuned constants.
7. **Budgets and invariants.**
   - Flags: `--kv-host-bytes`, `--kv-disk-path`, `--kv-disk-bytes`. L3 is off unless a path is given.
   - All host and disk space is reserved at startup; nothing is allocated in steady state.
   - Default host budget leaves headroom for the OS (cap at a fraction of NUMA-local free memory, stated in docs).
8. **Observability.** `/metrics` counters per tier: hits, misses, bytes moved, restore latency histogram, recompute avoided (tokens), evictions, checksum failures, write budget used.

## Architecture
- New crate `helios-kvstore`, CUDA-free:
  - index and policy;
  - host slab accounting;
  - disk segment store;
  - I/O thread pool.

  It talks to the GPU only through HAL copy primitives (rule 1).
- Transfers run through per-path queues (device->host, host->disk, host->device, disk->device), an idea from Dynamo KVBM, so a slow disk never blocks GPU restores.
- HAL additions (additive trait methods, default "unsupported"):
  - pinned host slab allocation;
  - page copy GPU<->host on a dedicated low-priority stream, returning a completion ticket (event).
- helios-core scheduler:
  - asks kvstore for placement on lookup;
  - runs `Restore` admissions with transfer tickets;
  - never blocks the decode loop on I/O.
- Sub-phase A reuses the phase 07 simulator with a tier model.

## Inputs required from phase 12 step 6 (decide in ADR 0002, not here)
- **Page-major layout:** all layers of one KV page contiguous, so a page moves with one copy. Adopt it only if attention cost with this layout is unchanged when measured. Otherwise transfers need batched copies (one per layer per page) and this phase's transfer cost doubles in complexity.
- Page size in tokens, so that bytes per page land in the 1-4 MB range for the target model.

## Sub-phases
- **A. Workload + simulation (CPU only; may start before entry criteria).**
  - Record agent-like traces: multi-turn sessions, tool-result appends, idle gaps.
  - Extend the phase 07 simulator with L2/L3 tiers parameterized by bandwidth and latency.
  - Report predicted hit rates and time saved versus recompute, for the 0.5B and 7B class models.

  Output decides whether B-E are worth building.
- **B. HAL primitives [RUN on H100].** NUMA-local pinned slab, copy stream, ticketed async copies. Measure:
  - H2D/D2H bandwidth from a local vs a remote NUMA node;
  - decode slowdown while copies run.
- **C. L2 host tier.** Index residency, demotion, promotion, restore-vs-recompute, metrics.
- **D. L3 NVMe tier.** Segment store, io_uring/O_DIRECT, write budget, persistence and restart, integrity.
- **E. Session parking.** Optional client hint to raise the reuse score of a conversation prefix. Only if A shows that implicit LRU plus hit counts are not enough.
- **F. Later (not this phase).**
  - Cross-process shared host cache on multi-GPU nodes (sidecar with shared memory).
  - GPUDirect Storage (NVMe -> GPU without a CPU bounce).
  - FP8 storage for cold tiers: this changes numerics, so it needs its own decision.

## Related Code Files
- Create: `helios/crates/helios-kvstore/**`, `docs/decisions/0003-tiered-kv-store.md`, `docs/kv-tiers.md`, tests listed below, `helios-core/tests/sim/tiers.rs`.
- Modify: helios-hal (traits + cuda module: pinned slab, copy stream), helios-core (scheduler admission, cache residency, metrics), helios-cli (flags, startup reservation).
- Append: phase 11 section in `docs/unverified-ledger.md`.

## Implementation Steps
1. Sub-phase A; stop and report. Proceed only if the entry criteria are met and A predicts a benefit.
2. Tests first for kvstore on CPU:
   - index collisions (token verification);
   - residency transitions;
   - pin safety;
   - segment write/read with real files;
   - crash mid-write;
   - corrupted segment;
   - identity mismatch after restart;
   - hostile index files.
3. Sub-phase B with `*_hw` tests.
4. Sub-phase C; then D; then E only if justified.
5. Each sub-phase: one commit series, measurements in the report, ledger rows updated.

## Todo
- [ ] A traces + sim
- [ ] entry criteria evidence
- [ ] B HAL primitives
- [ ] C L2
- [ ] D L3
- [ ] E parking (conditional)
- [ ] docs + ADR

## Success Criteria
- [RUN on H100] A session whose pages were demoted to L2 (and separately to L3), then restored, decodes **bit-identically** to the same session never evicted. The KV bytes are the same, so any difference is a bug.
- [RUN on H100] Time to first token for resuming a long session (8k and 32k history): restore path vs recompute path, both measured. The policy picks the faster one in each case.
- [RUN on H100] Decode time per token while background copies run stays within a bound declared before measuring.
- [RUN] After Ready, no host or GPU allocation (test or assertion). Restart with L3 restores valid entries and drops identity-mismatched or corrupted ones.
- [RUN] Windows: L2/L3 logic tests pass. io_uring path is [RUN on Linux] only.

## Test Strategy & Quality Gate
- Pyramid 70/20/10. L0 (CPU): kvstore unit and integration with real files, simulator. L1 (`*_hw`): copy correctness and bandwidth, restore bit-identity, interference. L3 bench: TTFT resume and per-token time under copy load.
- Evidence: phase 01 command templates; benchmark protocol from the optimization task (release build, 10 runs, median with min/max, A/B interleaved, GPU clocks recorded).

## ADR Rationale
- Context: agent workloads keep long histories; GPU memory bounds how many stay resident.
- Decision: one location-agnostic index, tiers as residency, cost-based restore-vs-recompute, CUDA-free store crate, in-process.
- Why not the alternatives:
  - Prompt-size switching ignores reuse.
  - An out-of-process sidecar now solves problems we do not have (no GIL, one process).
  - Writing every page to disk wastes drives and bandwidth.

## Risk Assessment
- **Wrong reuse means silent wrong output.** Mitigations: cache identity, chained hash, token verification on hit, checksums, bit-identity tests.
- **Pinning large slabs starves the OS.** Mitigations: budgets, NUMA-local caps, documented defaults.
- **Copy traffic slows decode.** Mitigations: low-priority stream, measured interference bound.
- **NVMe wear.** Mitigations: reuse threshold, write budget.
- **Complexity before need.** Mitigation: entry criteria plus sub-phase A gate.

## Security Considerations
- KV pages derive from user prompts. L3 files are created 0600 in a service-owned directory, and a purge command removes them. No sharing across cache identities.
- Disk data is never trusted: checksums plus identity checks, and parser hardening for index and segment files (same posture as helios-pkg).
- Encryption at rest is out of scope for now. Document it and record it as an open decision.

## Open Decisions (owner; numbered as in plan.md)
14. L3 persistence across restarts: default off with an opt-in path? (Rec: yes.)
15. Encryption at rest for L3? (Rec: not now; 0600 + purge; revisit before multi-user deployments.)
16. Explicit session hint in the API? (Rec: only if sub-phase A shows a need.)
17. Page-major KV layout in phase 12 step 6? (Rec: yes, if attention cost is unchanged.)
18. Multi-GPU node sharing (sub-phase F)? (Rec: per-rank NUMA-local tiers inside one engine first, which phase 13 needs anyway; cross-process sharing only for several data-parallel engines on one node.)

## Next Steps
Run sub-phase A as soon as the phase 07 simulator exists. It is CPU-only and decides whether the rest of this phase is worth building.
