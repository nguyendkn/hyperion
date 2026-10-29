# 0004: process and rank model for many GPUs and many nodes

Status: **proposed**, needs owner sign-off (not accepted). Plan: phase 13 D1. Related: ADR 0006 (trait), 0007 (StepPlan), 0009 (NCCL). Nothing is implemented.

## Context
Owner scope (2026-10-29): many GPUs per node and many nodes. Today one process drives one GPU through one `HopperBackend` with one device mutex. NCCL documents that launching graphs that contain collectives of several communicators from a single thread can deadlock, and a captured graph that contains a collective must be launched by every rank in step.

## Decision (proposed)
- **One Hyperion process per node, one execution thread and one CUDA context per GPU rank.** Each rank owns its own device mutex, stream, kernel modules, cuBLAS handle, VMM arena and CUDA graphs. Graphs that contain NCCL operations are launched only from their rank's thread.
- **Node rank 0 runs the HTTP server, the tokenizer and the scheduler.** Other nodes run the rank threads and a control-plane client; they never listen on the public API.
- **SPMD by construction.** The scheduler decides once per step and every rank applies the same decision (ADR 0007); ranks never decide independently, because a collective captured in a graph is a property of all ranks.
- **Rendezvous.** At start every rank announces itself (node rank, local ordinal, driver and NCCL versions, free memory, topology); the group is formed when the expected size is reached, with a timeout and a typed error that names the missing ranks. TP size and PP size are a load-time deployment choice, not part of the package (ADR 0005).
- **Failure semantics.** If any rank fails (error, hang past a replay timeout, lost peer) the engine stops serving with a typed error. No partial results, no silent continuation on fewer ranks. Restart is the supervisor's job.
- **Single GPU stays the degenerate case**: a group of one rank; no collective, no control plane.

## Why over the alternatives
- One process per GPU: simpler isolation, but every step needs cross-process KV and plan transport, and the same-process path (shared host buffers, one scheduler) is what a single node wants. Revisit for multi-node, where processes per node are needed anyway.
- Rank-local scheduling: needs agreement on every step; one decision broadcast is simpler and is what graph replay requires.

## Consequences
- Rank-aware contracts (ADR 0006) and an L0 test double with threads per rank (ADR 0006, "multi-rank stub").
- The scheduler's output becomes a serialisable StepPlan (ADR 0007).
- A control plane (private network, authenticated peers before multi-tenant use) with the smallest possible port set: rank 0's API and the documented rendezvous ports.

## Open questions for the owner
- Same-process threads per node (proposed) versus one process per GPU.
- Decision 19 (Dynamo worker versus a built-in router) decides what sits above this model across nodes.

## Verification plan
The multi-rank stub with CPU collectives for the control plane and failure paths (a rank dies, a rank is slow, rendezvous timeout); 2-GPU pod for the thread-per-rank launch of captured NCCL graphs (phase 13 M1); the hang case with a deliberately absent rank.
