# 0009: NCCL is loaded at run time and used inside graphs

Status: **proposed**, needs owner sign-off (not accepted, nothing implemented). Plan: phase 13 D6 and milestones M1 to M3; owner decision 22 (NCCL only, or a custom small-message all-reduce later). Related: ADR 0004, 0006.

## Context
Tensor parallelism needs all-reduce after the attention output and MLP down projections, and all-gather for vocabulary-parallel logits, inside the captured decode and prefill graphs. On an HGX H100 node (NVLink and NVSwitch) NCCL is the right default transport; across nodes it uses InfiniBand. The CUDA driver and cuBLAS are already loaded at run time (no link-time dependency, typed `LoadError`); NCCL should follow the same pattern.

## Decision (proposed)
- **NCCL is loaded with `dlopen`/`LoadLibrary` (`libnccl.so.2`; `nccl.dll` is not a target), hand-declared, no new crate**, with explicit symbol names (`ncclCommInitRank`, `ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`, `ncclSend`, `ncclRecv`, `ncclGetVersion`, ...). A missing library or symbol is a typed `LoadError`; a single-rank run never touches it.
- **The NCCL version is recorded at startup** (and in the ledger for every multi-GPU result) together with the driver, CUDA and topology (`nvidia-smi topo -m`).
- **Collectives are enqueued on the rank's stream and are graph-capturable** (the `Communicator` capability of ADR 0006). Every rank captures the same collectives in the same order; each rank launches its graphs from its own thread (ADR 0004). Replay and rendezvous have timeouts so a hang becomes a typed error.
- **NCCL only for now (decision 22).** A custom one-shot small-message all-reduce is considered only if measurements on the 8-GPU node (phase 13 M2) show the all-reduce share of decode is high.
- **Determinism.** NCCL's reduction order can differ between TP sizes: correctness is TP=2 versus TP=1 within a declared tolerance, and TP=2 run to run bit-identical; tokens are never compared at long context.

## Why over the alternatives
- Link-time NCCL: a build would need the toolkit and a GPU again, losing the Windows builds that need neither.
- A crate wrapping NCCL: a new dependency for a handful of calls.
- Own collectives over peer access: more kernel work with no measured need.

## Consequences
- A new `cuda/`-style module for the NCCL table and unit tests of the loader (missing library, missing symbol) that run on Windows.
- The two-GPU pod is the first place anything real runs (M1); until then only the loader logic is testable.

## Open questions for the owner
- Decision 22; and whether Windows or non-NVIDIA transports are in scope (proposed: no).

## Verification plan
Loader tests on Windows; on 2 H100s: communicator creation, an all-reduce correctness test against a host sum, a graph that captures an all-reduce and replays it 1000 times bit-identically; a hang test with one rank withheld that must end in a typed timeout.
