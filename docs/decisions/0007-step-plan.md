# 0007: the scheduler's output is a serialisable StepPlan

Status: **accepted** (owner review, 2026-10-30). Design only: nothing is implemented. Plan: phase 13 D4 and phase 07 (serving). Related: ADR 0004 (ranks), 0006 (trait), 0002 (page tables).

## Context
Today `helios-core::scheduler` runs one request at a time and calls `generate`, which owns the sequence state inside the backend. With batching, prefix sharing and several ranks, the scheduler must say what each step does, and every rank must do the same thing: a decode graph that contains a collective is launched by all ranks together (ADR 0004). A plan that exists only as arguments of a local call cannot cross threads or nodes.

## Decision
- **The scheduler (rank 0 only) emits one `StepPlan` per step.** Contents, all plain data:
  - the **capacity id**: which captured graph runs (decode batch bucket 1/2/4/... or a prefill chunk capacity, powers of two from 128 to 4096) and how many rows are active;
  - per active row: the **slot id** (the sequence's state on the device), the **row's token count** (see below), where its input comes from (`Tokens(ids)` or **`FromDevice(slot)`**: the token the device sampled in the previous step, so a greedy or sampled chain never waits for the host), the position of its first token, and its page-table length;
  - **page-table deltas**: the device holds each slot's page table (ADR 0002); the plan carries only `(slot, index in the table, page)` entries for pages assigned or released since the previous step, never a whole table per row;
  - **slot initialisation**, once per request, in the plan of the step that admits it: sampling parameters (temperature, top-p, top-k, seed, penalties), the stop conditions' device-side parts, the maximum length;
  - the **expected outputs**: which rows return a token id and which return logits (verification);
  - a **step number** and a **plan hash** so a rank can detect a gap or a divergence.
- **Sampling state is per slot, on the device.** The random state and the parameters live in device memory per slot, are set by the slot initialisation and advanced by the sampling kernel; the plan never repeats them. Under tensor parallelism every rank holds the same state and draws identically.
- **Multiple tokens per row are allowed.** A row carries `n >= 1` tokens: a decode row has 1, a prefill chunk row has up to the chunk capacity, a speculative or multi-token row has several. The graphs and the plan are written for `n` rows of tokens, not for "one token per sequence".
- **Binary encoding.** A compact, versioned, fixed-width little-endian format with a header (format version, step number, plan hash); no JSON on the plan path. A golden-file test pins the encoding; a decoder refuses an unknown version with a typed error.
- **Every rank applies the same plan.** With one rank it is a local call (no encoding cost); with several it is broadcast over the control plane before the step. A rank that cannot apply a plan reports a typed error, and the engine stops (ADR 0004).
- **Backend entry points** (ADR 0006): `submit_step(&KvPool, &StepPlan) -> StepTicket` enqueues the step and returns at once; `take_output(StepTicket) -> StepOutput` returns the result and may be asked for **one step late**. The on-GPU token chain survives: the next plan names `FromDevice(slot)` and the host reads ids a step behind, as the token ring of the sm90 decode chain does now. Inactive rows of a bucket are masked by the plan, never by shape changes, so graphs are not re-captured.
- **Prefix cache** (phase 07) works on page indices and chained hashes and lives with the scheduler; the backend never sees hashes.

## Why over the alternatives
- Keep `generate` as the only entry point: no batching, no sharing, no multi-rank.
- Let each rank run its own scheduler: ranks would have to agree on every step anyway; one decision plus a broadcast is simpler and is what graph replay needs.
- A whole page table per row in every plan: the plan grows with the context length and is rebuilt every step; deltas keep it proportional to what changed.
- Sampling parameters in every plan: bytes repeated for every token, and a place for ranks to disagree.
- A step API that returns the output: the host would wait for each token before it could enqueue the next step.

## Consequences
- `helios-core/scheduler.rs` is rewritten around plans (phase 07); `generate` stays as a default-implemented convenience until then.
- The decode graphs' fixed inputs (token, position, ring base today) grow into a device-resident slot table (page tables, sampling state, positions) written by plan deltas before replay; prefill capacities and decode buckets each get a graph captured at load.
- The simulator of phase 07 can replay recorded traffic against the scheduler and compare plans, with no GPU.

## Verification plan
Unit tests of plan construction, delta application (a table rebuilt from deltas equals the scheduler's), the binary round trip and the golden file; a replay test: the same trace produces the same plans; the multi-rank stub (`[RUN on stub]`) for the broadcast and divergence detection; on a GPU, a plan applied by one and by two ranks gives the same tokens (phase 13 M1), and a `FromDevice` chain equals the host-fed chain.
