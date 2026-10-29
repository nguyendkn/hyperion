# Phase 13: Multi-GPU and multi-node foundation

## Context Links
- Owner direction (2026-10-29): Hyperion targets many H100s per node and many nodes. Single-GPU is the first milestone, not the scope.
- Supersedes the single-GPU assumptions behind plan.md decisions 3, 7 and 18, phase 05's "abstraction only" multi-GPU note, and phase 11 sub-phase F's "later".
- Ideas only, no copied code:
  - upstream spark-local-inference `docs/same-process-tp.md`, `docs/multi-host.md`, `docs/core-design.md`;
  - NVIDIA Dynamo (KVBM, KV-aware router, custom-backend KV events);
  - NCCL user guide (CUDA graph capture).
- Related phases: 03 (spec before freeze), 05 (trait), 07 (scheduler, prefix cache), 11 (tiers), 12 (kernels, KV design).

## Overview
Priority P1 for design items (they gate freezes); P2 for implementation. Status: pending. Track: D1-D6 run on Windows beside phase 12; M1-M4 on GPU pods. Depends: D2 lands through phase 03; D3 lands with phase 05 + phase 12 step 6 as one trait change; D4/D5 land through phase 07; M1 needs D1-D3 and phase 12 step 4.
Goal: make every contract that is expensive to change later (manifest, backend trait, scheduler output, KV layout and hashing, process model) rank-aware now, then prove tensor parallelism on one node before going multi-node.
Effort:
- design items D1-D6: ~4-6d (mostly docs, ADRs and spec fields);
- implementation M1-M4: ~15-25d (estimate, not measured).

## Key Insights
- Kernels written for one GPU are reused by every rank, so phase 12 continues. What must change now are the **contracts around the kernels**. Each one gets harder to change after more backends, packages and clients depend on it.
- With multi-GPU a stated requirement, rank fields and a device group are no longer speculative abstractions. The earlier "not now" recommendations assumed a single-GPU scope.
- Hyperion captures graphs at load time, unlike upstream, whose package fixes the rank count. So one package can serve TP 1/2/4/8 if weights are sharded at load from canonical tensors. This is a real advantage; keep it.
- NCCL graph capture is a collective property: every rank must capture the same collectives and replay the same graph. The scheduler must therefore make one decision per step and broadcast it (SPMD). Ranks must never decide independently.
- On an HGX H100 node (8 GPUs, NVSwitch) NCCL is the right default transport: NVLink inside the node, InfiniBand across nodes, NVLS when available. Custom small-message all-reduce is a later optimization, gated on measurement.

## Parallelism map (what runs where)
| Dimension | Where | When |
|---|---|---|
| Tensor parallel (TP 2/4/8) | Inside one node over NVLink/NVSwitch | M1-M2 (first) |
| Data parallel replicas | Across GPUs or nodes, independent engines | M3 (cluster layer) |
| Pipeline parallel | Across nodes, for models larger than one node | M4, only when a target model needs it |
| Expert parallel (MoE) | Inside or across nodes | Backlog (needs MoE models) |
| Context parallel (ring attention) | For very long contexts | Backlog (needs measured need) |
| Disaggregated prefill/decode | Separate prefill and decode workers, KV transfer | Backlog (via Dynamo/NIXL) |

## Design items (do now; each ends in an ADR or a spec PR and needs owner sign-off)
- **D1. Process and rank model.**
  - One Hyperion process per node, one execution thread and one CUDA context per GPU rank. Each rank has its own VMM arena, streams, modules and graphs.
  - Each node is a separate process; node rank 0 runs HTTP and the scheduler and coordinates peers through a rendezvous.
  - Graphs with NCCL operations are launched from their rank's own thread. NCCL documents deadlocks when graphs of several communicators are launched from one thread.
- **D2. Manifest (phase 03, before freeze).**
  - Weights stay canonical (unsharded safetensors).
  - Each compiler (Ignition) declares a sharding plan per tensor class: column-split, row-split, replicated, vocab-split. It also declares the TP sizes it supports, derived from head counts. KV heads fewer than TP size need replication; make that rule explicit.
  - Optional `deployment` (tp_size, pp_size) stays a load-time choice, not a package property.
  - Rejection: a requested TP size not in the supported list fails with a typed error before CUDA init.
- **D3. Backend trait (merge with phase 05 decoupling and phase 12 step 6 KV design into ONE trait change).**
  - Backends are per rank.
  - A `DeviceGroup` (ranks, topology, NUMA node per GPU) is built by the factory.
  - A `Communicator` capability (all-reduce, all-gather, reduce-scatter, send/recv) is enqueued on the rank's stream and must be graph-capturable.
  - KV allocation is page-index based (ADR 0002) with identical page counts on every rank, limited by the smallest rank budget.
  - Changing the trait twice is the costly mistake to avoid.
- **D4. Scheduler output is a serializable `StepPlan` (phase 07).**
  - The scheduler (rank 0 only) emits per step: graph/capacity id, metadata buffer contents, page table deltas, sampling params.
  - Every rank applies the same plan.
  - With one rank this is a local call; with many it is broadcast over the control plane.
  - This keeps phase 07's scheduler valid for multi-GPU without a rewrite.
- **D5. KV hashing and events compatible with a cluster router.**
  - Keep the chained prefix hash (parent link) and token verification (phase 07).
  - Emit block stored/removed/cleared events with monotonically increasing ids and the hash block size, so a KV-aware router can track cached prefixes per worker. This matches what Dynamo expects from custom engines.
  - Under TP the index is per engine (rank 0), not per rank.
- **D6. Transport loading.**
  - NCCL is loaded at runtime (dlopen `libnccl.so.2`), like the CUDA driver (phase 12 step 1), with a typed error if absent.
  - Record the NCCL version at startup. No link-time dependency.

## Implementation milestones (each [RUN] on real hardware; record topology, driver, NCCL version)
- **M1. TP=2 on one node (2-GPU pod).**
  - Shard at load from the canonical package; all-reduce after attention output and MLP down projections; vocab-parallel logits with gather.
  - Graph capture including NCCL.
  - Correctness: TP=2 logits vs TP=1 within a tolerance declared before running (reduction order changes); TP=2 run-to-run bit-identical.
  - Model: Qwen2.5-0.5B for plumbing (2 KV heads), then Qwen2-7B (4 KV heads) for TP 2/4.
- **M2. TP=8 on a full HGX node.**
  - A model that needs it (70B class, 8 KV heads).
  - Measure per-token time and all-reduce share (nsys).
  - Evaluate NVLS and, only if the all-reduce share is high at decode, a custom one-shot all-reduce.
- **M3. Multiple nodes, data parallel first.**
  - Independent engines per node (or per GPU group) behind a KV-aware router.
  - Prefer integrating as a worker of NVIDIA Dynamo (engine-agnostic router, planner, KVBM, NIXL) over building a router. A minimal built-in router is acceptable only if the Dynamo path is blocked.
  - Owner decision 19.
- **M4. Pipeline parallel across nodes.** Only when a target model does not fit one node. Stage boundaries at layer groups; activations over InfiniBand via NCCL send/recv; graphs per stage.

## Effects on other phases (hand-offs)
- **03:** add D2 fields before the first release (decision 3 recommendation changes to "add now").
- **05:** trait work becomes the merged D3 change, done once.
- **07:** D4 `StepPlan` and D5 events; prefix index per engine.
- **11:**
  - L2 is per rank, in the NUMA node of that rank's GPU (HGX nodes usually split GPUs across two CPU sockets);
  - the L3 index is per engine;
  - sub-phase F (cross-process sharing) matters mainly for data-parallel engines on one node.
- **12:** step 6 ADR follows D3; kernels stay per GPU; step 8 batch buckets are shared by all ranks.

## Testing strategy without many GPUs
- **L0 (CPU):** a multi-rank stub backend (N threads, one per rank) with CPU collectives that implement the real semantics. It is used only to test the control plane, the StepPlan broadcast, sharding plans, and failure handling (a rank dies, a rank is slow, rendezvous timeout).
- **L1 (hardware):**
  - 2-GPU pod (M1);
  - 8-GPU node (M2);
  - 2 nodes with InfiniBand (M3/M4).

  Every claim is tagged with the topology it ran on.
- **Failure semantics:** if any rank fails, the engine stops serving with a typed error (no partial results). Restarts are handled by the supervisor or cluster layer.

## Success Criteria
- [RUN] The design items D1-D6 are accepted as ADRs. The spec carries D2 fields before freeze. One trait change covers D3.
- [RUN on 2xH100] M1 correctness as defined above.
- [RUN on 8xH100] M2 per-token time and all-reduce share reported.
- [RUN on 2 nodes] M3: requests routed by prefix overlap. A second request with the same long prefix reaches the worker holding it (measured TTFT difference).

## Risk Assessment
- **Over-design before evidence.** Mitigation: D-items are contracts only; M-items need hardware runs.
- **NCCL plus graphs deadlocks or hangs.** Mitigations: thread per rank, uniform capture, timeouts on rendezvous and replay.
- **Numerics differ across TP sizes.** Mitigation: declared tolerances; never compare tokens at long context.
- **Cluster layer scope creep.** Mitigation: Dynamo integration first (decision 19).

## Security Considerations
- Rendezvous and control plane on a private network; authenticated peers (shared secret or mTLS) before multi-tenant use.
- No inbound ports beyond rank 0's API and the documented rendezvous ports.

## Open Decisions (owner; numbering continues plan.md)
19. Cluster layer: integrate as an NVIDIA Dynamo worker vs a built-in router? Rec: Dynamo first.
20. Package and TP: shard at load from canonical weights (one package for all TP sizes) vs pre-sharded packages per TP size? Rec: shard at load.
21. First multi-GPU target model for M2 (70B class)? Owner picks based on the workloads that matter.
22. Collective transport: NCCL only, or NCCL plus a custom small-message all-reduce later? Rec: NCCL only until M2 measurements.
