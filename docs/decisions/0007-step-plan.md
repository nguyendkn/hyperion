# 0007: the scheduler's output is a serialisable StepPlan

Status: **proposed**, needs owner sign-off (not accepted, nothing implemented). Plan: phase 13 D4 and phase 07 (serving). Related: ADR 0004 (ranks), 0006 (trait), 0002 (page tables).

## Context
Today `helios-core::scheduler` runs one request at a time and calls `generate`, which owns the sequence state inside the backend. With batching, prefix sharing and several ranks, the scheduler must say what each step does, and every rank must do the same thing: a decode graph that contains a collective is launched by all ranks together (ADR 0004). A plan that exists only as arguments of a local call cannot cross threads or nodes.

## Decision (proposed)
- **The scheduler (rank 0 only) emits one `StepPlan` per step.** Contents, all plain data with a stable encoding:
  - the **capacity id**: which captured graph runs (decode batch bucket 1/2/4 or a prefill chunk capacity, powers of two from 128 to 4096, ADR on tiled prefill) and how many rows are active;
  - the **metadata buffer contents**: per active row the token id (or the chunk's token ids), position, sequence length, sampling parameters, and the row's page table (indices, ADR 0002);
  - **page-table deltas**: pages newly assigned or released since the previous step;
  - the **expected outputs**: which rows return a token id (device sampling, only ids cross the bus) and which return logits (verification);
  - a **step number** and a **plan hash** so a rank can detect a gap or a divergence.
- **Every rank applies the same plan.** With one rank it is a local call (no encoding cost); with several it is broadcast over the control plane before the step. A rank that cannot apply a plan reports a typed error, and the engine stops (ADR 0004).
- **Backend entry point:** `ComputeBackend::execute_step(&StepPlan) -> StepOutput` (ADR 0006) replaces `execute_prefill` and `execute_decode_step`. Inactive rows of a bucket are masked by the plan, never by shape changes, so graphs are not re-captured.
- **Prefix cache** (phase 07) works on page indices and chained hashes and lives with the scheduler; the backend never sees hashes.

## Why over the alternatives
- Keep `generate` as the only entry point: no batching, no sharing, no multi-rank.
- Let each rank run its own scheduler: ranks would have to agree on every step anyway; one decision plus a broadcast is simpler and is what graph replay needs.
- Pass plans as backend-specific structs: the control plane would need a serialiser per backend.

## Consequences
- `helios-core/scheduler.rs` is rewritten around plans (phase 07); `generate` stays as a default-implemented convenience until then.
- The decode graphs' fixed inputs (token, position, ring base today) grow into a small metadata buffer written before replay; prefill capacities and decode buckets each get a graph captured at load.
- The simulator of phase 07 can replay recorded traffic against the scheduler and compare plans, with no GPU.

## Open questions for the owner
- Encoding (a compact binary format versus JSON for the first version; the plan is off the per-token hot path only for batch 1 on one rank).
- Whether device sampling parameters (temperature, top-p, top-k, seed) belong in the plan or in per-request state kept on the device (the plan then carries only a request id).

## Verification plan
Unit tests of plan construction and round trip with the multi-rank stub (ADR 0006); a replay test: the same trace produces the same plans; on a GPU, a plan applied by one and by two ranks gives the same tokens (phase 13 M1).
