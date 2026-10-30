# 0004: process and rank model for many GPUs and many nodes

Status: **accepted** (owner review, 2026-10-30). Design only: nothing is implemented. Plan: phase 13 D1. Related: ADR 0006 (trait), 0007 (StepPlan), 0009 (NCCL).

## Context
Owner scope (2026-10-29): many GPUs per node and many nodes. Today one process drives one GPU through one `HopperBackend` with one device mutex. NCCL documents that launching graphs that contain collectives of several communicators from a single thread can deadlock, and a captured graph that contains a collective must be launched by every rank in step.

## Decision
- **One Hyperion process per node, one execution thread and one CUDA context per GPU rank** (owner decision, 2026-10-30). Each rank owns its own device mutex, stream, kernel modules, cuBLAS handle, VMM arena (ADR 0002) and CUDA graphs. Graphs that contain NCCL operations are launched only from their rank's thread.
- **Node rank 0 runs the HTTP server, the tokenizer and the scheduler.** Other nodes run the rank threads and a control-plane client; they never listen on the public API.
- **SPMD by construction.** The scheduler decides once per step and every rank applies the same plan (ADR 0007); ranks never decide independently, because a collective captured in a graph is a property of all ranks.
- **Rendezvous.** At start every rank announces itself: node rank, local ordinal, driver version, NCCL version, free memory, topology, and the NCCL environment (ADR 0009). The group is formed when the expected size is reached, with a timeout and a typed error that names the missing ranks; a rank whose driver or NCCL version differs from rank 0's is refused with a typed error naming both. TP size and PP size are a load-time deployment choice, not part of the package (ADR 0005).
- **Failure semantics.** If any rank fails (error, hang past a replay timeout, lost peer) the engine stops serving with a typed error. No partial results, no silent continuation on fewer ranks. Restart is the supervisor's job.
- **Single GPU stays the degenerate case**: a group of one rank; no collective, no control plane.
- **Test double.** A multi-rank stub with one thread per rank and CPU collectives is accepted for L0 tests of the control plane, the plan broadcast, sharding and failure handling; every result it produces is labelled `[RUN on stub]` and is never evidence about a GPU (ADR 0006, `docs/testing-layers.md`).

## Why over the alternatives
- One process per GPU: simpler isolation, but every step needs cross-process KV and plan transport; the same-process path (shared host buffers, one scheduler) is what a node wants. Across nodes a process per node is needed anyway, and that is the model above.
- Rank-local scheduling: needs agreement on every step; one decision broadcast is simpler and is what graph replay requires.

## Consequences
- Rank-aware contracts (ADR 0006) and the stub.
- The scheduler's output is a serialisable StepPlan (ADR 0007).
- A control plane (private network, authenticated peers before any multi-tenant use) with the smallest possible port set: rank 0's API and the documented rendezvous ports.

## Verification plan
The stub with CPU collectives for the control plane and failure paths (a rank dies, a rank is slow, rendezvous timeout, a version mismatch); a 2-GPU pod for the thread-per-rank launch of captured NCCL graphs (phase 13 M1); the hang case with a deliberately absent rank.
