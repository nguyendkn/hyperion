# 0009: NCCL is loaded at run time and used inside graphs

Status: **accepted** (owner review, 2026-10-30). Design only: nothing is implemented. Plan: phase 13 D6 and milestones M1 to M3; owner decision 22. Related: ADR 0004, 0006.

## Context
Tensor parallelism needs all-reduce after the attention output and MLP down projections, and all-gather for vocabulary-parallel logits, inside the captured decode and prefill graphs. On an HGX H100 node (NVLink and NVSwitch) NCCL is the right default transport; across nodes it uses InfiniBand. The CUDA driver and cuBLAS are already loaded at run time (no link-time dependency, typed `LoadError`); NCCL follows the same pattern.

## Decision
- **NCCL only (decision 22, 2026-10-30).** No custom all-reduce for now; revisit only if measurements on the 8-GPU node (phase 13 M2) show the all-reduce share of decode is high.
- **NCCL is loaded with `dlopen`/`LoadLibrary` (`libnccl.so.2`; `nccl.dll` is not a target), hand-declared, no new crate**, with explicit symbol names (`ncclCommInitRank`, `ncclGroupStart`, `ncclGroupEnd`, `ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`, `ncclSend`, `ncclRecv`, `ncclGetVersion`, `ncclGetUniqueId`, ...). A missing library or symbol is a typed `LoadError`; a single-rank run never touches it.
- **Communicator initialisation is grouped.** The rank threads of one process (ADR 0004: one thread per GPU) create their communicators inside one `ncclGroupStart` / `ncclGroupEnd` bracket (or one `ncclCommInitAll` for a single-node group), never as independent blocking `ncclCommInitRank` calls from separate threads, which can deadlock. The unique id is made on rank 0 and distributed over the control plane; for several nodes each node's threads form one group and the `ncclCommInitRank` calls rendezvous through the id.
- **Rendezvous refuses a mismatch.** Every rank announces its NCCL version (`ncclGetVersion`) and its driver version (ADR 0004); the group is not formed, with a typed error naming the ranks and both versions, when they differ from rank 0's.
- **The NCCL environment is recorded.** At startup every rank logs the NCCL variables it runs with (`NCCL_DEBUG`, `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_P2P_*`, `NCCL_IB_*`, `NCCL_NET_*`, `NCCL_SOCKET_IFNAME`, `NCCL_CUMEM_*`, `NCCL_GRAPH_*`, and any other `NCCL_*` that is set) together with the NCCL version, the driver, the CUDA version and `nvidia-smi topo -m`, and the rendezvous compares the set across ranks and warns on a difference. Every multi-GPU result in the ledger names them.
- **Collectives are enqueued on the rank's stream and are graph-capturable** (the `Communicator` capability of ADR 0006). Every rank captures the same collectives in the same order; each rank launches its graphs from its own thread. Replay and rendezvous have timeouts so a hang becomes a typed error.
- **Determinism.** NCCL's reduction order can differ between TP sizes: correctness is TP=2 versus TP=1 within a declared tolerance, and TP=2 run to run bit-identical; tokens are never compared at long context.

## Why over the alternatives
- Link-time NCCL: a build would need the toolkit and a GPU again, losing the Windows builds that need neither.
- A crate wrapping NCCL: a new dependency for a handful of calls.
- Own collectives over peer access: more kernel work with no measured need.
- Independent `ncclCommInitRank` per thread: documented deadlock risk.

## Consequences
- A new `cuda/`-style module for the NCCL table and unit tests of the loader (missing library, missing symbol) that run on Windows.
- The two-GPU pod is the first place anything real runs (M1); until then only the loader logic is testable.

## Verification plan
Loader tests on Windows; on 2 H100s: grouped communicator creation from two threads, an all-reduce correctness test against a host sum, a graph that captures an all-reduce and replays it 1000 times bit-identically; a hang test with one rank withheld that must end in a typed timeout; a rendezvous test with a deliberately different NCCL version string (stub, `[RUN on stub]`) and the recorded environment in the log.
