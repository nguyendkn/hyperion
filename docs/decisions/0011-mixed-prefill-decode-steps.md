# 0011: mixed prefill + decode steps (one flattened token batch)

Status: **accepted** (owner, 2026-11-05) with the four edits listed under "Owner decisions" below; nothing in this ADR is implemented yet. Changes ADR 0007 (`StepPlan`) in one version bump
(v1 -> v2) that also carries presence/frequency penalties, so the encoding is bumped once. No trait change (ADR 0006): `submit_step` / `take_output`
already take a `StepPlan`.

## Context
Today a step is either `Capacity::Decode { bucket }` (one token per row, up to 32 rows) or `Capacity::Prefill { rows }` (one row, a chunk). While a
long prompt is prefilled every running sequence waits a whole chunk, so the inter-token latency (ITL) of a decode row is the chunk time plus a decode
step. Measured (7B, H100, greedy, `docs/performance-log.md`, 8192-token prompts in, 256 out, concurrency 8): ITL p90 is 28.2 ms at a busy chunk of 128,
31.0 ms at 256, 37.9 ms at 512, 44.1 ms at 1024 and 149.1 ms uncapped, against a target of 3 x the 7.263 ms bucket-8 decode step = 22.1 ms. No chunk
size meets the target for 8192-token prompts and the small chunks cost TTFT (7.2 s p50 at 128 against 1.1 s uncapped): alternating chunks and decode
steps has a floor of one chunk plus one decode step per token. A mixed step removes the floor: the weights are read once for the decode rows and the
prefill tokens together, so the decode rows ride on the prefill step instead of waiting for it.

## Decision

### 1. The step is a flattened token batch
One forward over `T` tokens (the step's total). Per token: its sequence slot and position (derived from `Row { slot, position, n_tokens }`, which already
describes both a decode row, `n_tokens = 1`, and a prefill row, `n_tokens = chunk`). Embedding, QKV, o_proj, MLP, norms and the head are GEMMs/element-wise
over `T` rows exactly as the prefill graph is today. Attention runs per row against that sequence's page table:
* decode rows (one query token): the existing split-KV paged decode kernels (`attention_split_paged_pm_*` + combine);
* prefill rows (a chunk of queries): the existing paged MMA prefill attention, one row group per prefill row.
Both kernels already take page tables and positions per row; the new work is launching both in one graph, with the two row sets given by a small
device-side descriptor array (`row_kind`, `first_token`, `n_tokens`, `position`, `slot`). Sampling and the greedy argmax run on the *last* token of each
row that asks for a token (`Want::Token`), as today. KV append is per token with the slot's table (as `kv_append_rows_paged`).

### 2. Graph capacities by total token count
The graph is chosen by `T`: **`Capacity::Mixed { tokens }`** with `tokens` in {64, 128, 256, 384, 512, 768, 1024} (the decode-only buckets
{1, 2, 4, 8, 16, 32} stay as the decode-only graphs: they are the same flattened batch with `T = rows` and keep their measured speed and bit-exact
behavior; mixed capacities start at 64). Rows are padded to the capacity with masked tokens (`slot = -1`, as masked rows today). The list is
short on purpose: every capacity is a graph (the current prefill ladder 128..4096 is 6 graphs and decode buckets are 6; mixed adds 8, each a few MiB of
graph memory plus the shared activation workspace sized for the largest capacity). The 1024 limit follows from the budget in section 3; prompts that
should go faster than 1024 tokens per step when no decode row waits keep using the existing prefill capacities (2048, 4096). `capture_graphs` captures
all of them before Ready; nothing is captured afterwards (invariant).

### 3. Scheduler policy
* A **token budget** `B` per step (a CLI flag, `--step-token-budget`, default chosen by the sweep in the verification plan; 384 from the estimate below).
* **Decode rows are always included** (one token each, at most `max_batch` = 32).
* The **prefill chunk fills the remainder**: the oldest waiting prompt first (the policy of `StepScheduler` today), `min(remaining prompt, B - decode_rows)`
  tokens, rounded down to a multiple of 64 (the page size) unless it finishes the prompt, so a chunk ends on a page boundary and the prefix cache can
  take whole pages. A second prompt may fill what is left of a step only if the first finishes in it.
* **Fairness / starvation bounds:** decode rows are never delayed by prefill (they are in every step); a waiting prompt gets at least `B - 32` tokens
  per step, so the worst-case time to first token of a prompt of `P` tokens behind `Q` earlier queued tokens is `(Q + P) / (B - decode_rows)` steps
  (the same estimator that drives the 429 of `--first-token-timeout`, with the measured mixed step time in place of the alternating sum). When no decode
  row exists the step is a plain prefill with the largest capacity that fits (unchanged).
* The step graph is chosen as the smallest `Mixed` capacity >= `T`; a step with only decode rows uses the decode bucket graph.

### 4. `StepPlan` v2 (one bump)
* New `Capacity::Mixed { tokens: u32 }` (tag 2).
* `SamplingInit` gains `presence_penalty: f32` and `frequency_penalty: f32` (OpenAI/vLLM semantics: `z_i -= presence * [count_i > 0] + frequency * count_i`
  applied before the repetition penalty and the temperature; **the counts are of generated (output) tokens only**, the prompt does not count; the repetition
  penalty keeps its scope, prompt and output together); the device keeps per-slot token counts next to the `seen` bitmap
  (`slots * vocab * 2 bytes`, 16-bit saturating counts: 304 KB per slot for a 152k vocabulary, 9.7 MB at 32 slots; a `seen`-only fast path when both
  are 0). The sampler reference gets the same terms in the same operation order, and the exact-match test grows two modes.
* Rows are unchanged. `Row.want = Logits` stays verification-only.
* Compatibility rule: a v2 build **reads v1** plans (decoding fills the new fields with 0, `Capacity` has no `Mixed`); a v1 build refuses v2 with
  `UnsupportedVersion` (as it already does). Golden files: `step_plan_v1.bin` is kept and decoded by the v2 reader; `step_plan_v2.bin` (with `Mixed` and
  the penalties) is added; both are byte-compared in `step.rs` tests. The plan hash (SHA-256 prefix of the body) is computed over the new body.

### 5. Tensor parallelism
The same flattened step runs on every rank: the same plan is broadcast (ADR 0004/0007), every rank holds its shard of QKV/gate-up and the per-rank KV
(its heads); the two all-reduces per layer move `T x hidden` activations, so the communication per step grows with `T` and the mixed capacity's graph
contains the all-reduces of its own size (captured per capacity, as ADR 0009 requires). Row-level work (sampling on the gathered logits, argmax) is
unchanged. No rank-specific scheduling.

### 6. Speculative-decode readiness
A row with `n_tokens = k + 1` and `Want::Token` over all of them (verification of `k` draft tokens) is a row like any other in the flattened batch:
attention is the prefill-style kernel with a causal mask over the row's own tokens and its page table. The plan therefore needs one more field later
(`Want::Tokens(n)` per row), a v3 matter; v2 does not preclude it because rows already carry `n_tokens` and `position`.

### 7. Expected effect (an ESTIMATE, not a measurement)
Two measured points give the cost of a step over `T` tokens for the 7B at 8192-token context (all from `docs/performance-log.md`, 2026-11-01):
the bucket-8 decode step is 7.263 ms and a 128-row chunk costs about 8 ms (the weights, 15 GB at about 3.35 TB/s, dominate: about 4.5 ms of the 6.8 ms
bucket-1 step); ITL p90 at a busy chunk of 512 is 37.9 ms, i.e. about 37.9 - 7.3 = 30 ms for a 512-row chunk. A straight line through (128 rows, 8 ms) and
(512 rows, 30 ms) gives about 0.057 ms per row. A mixed step of 8 decode rows and `B - 8` prefill tokens then costs about
`8 + 0.057 (B - 128)` ms (plus about 0.4 ms for the decode rows' attention): the ITL of a decode row is that step time, no decode step added on top.
**ITL <= 22.1 ms holds for `B` up to about 370 tokens** (the default 384 is just above, and would be validated, not assumed). At `B = 368`: about 360
prompt tokens per 22 ms step = 16 k tokens/s of prefill against 512 / (7.3 + 30) ms = 13.7 k tokens/s with alternating 512-row chunks, so the TTFT of an
8192-token prompt is about 23 steps = 0.5 s (against 2.3 s p50 measured with busy 512 at c=8). Assumptions that can be wrong: linearity of the step cost
between 128 and 512 rows, the attention cost of 8 decode rows against 8k-token contexts inside the prefill-shaped graph, and the overhead of the
descriptor array. The verification plan below measures them first.

### 8. Verification plan (declared before implementation)
* **Oracle per row inside mixed steps:** a decode row and a prefill row run in the same mixed step against the same compact oracle prompts: the logits of
  every `Want::Logits` row stay within the existing tolerance of the single-row oracle (0.5B 0.3851, 7B 2.5574, Qwen3 0.4088; binding rules unchanged)
  and the chunk-by-chunk prefill output equals the whole-prompt prefill token.
* **Row isolation:** a row's output is bit-identical whoever shares the step (extends `a_row_is_isolated_from_its_neighbours...`, the same capacity,
  different neighbours); a masked row writes nothing.
* **Graph == direct** per mixed capacity (every capacity launched directly and as a graph, bit-identical, as `graph_decode_is_bit_identical...`).
* **Greedy decode-only regression:** decode-only steps still use the bucket graphs; budget +1.5% (+3% at 16000) against the current table at buckets
  1/8/32 and contexts 5/1024/4096/8000/16000.
* **ITL/TTFT sweep:** 7B, 8192/256 and 512/256, concurrency 1-32, `B` in {128, 256, 384, 512, 768}: ITL p90 against 22.1 ms, TTFT p50, tokens/s, with
  the measured cost curve of a mixed step against the estimate above.
* **Sampling:** exact match of the device sampler (with penalties) against the host reference inside mixed steps.
* **StepPlan:** golden files v1 and v2, version refusal, hash.

## Consequences
* Removes the alternation floor; the ITL target for long prompts becomes reachable by policy (one flag) instead of by trading away TTFT.
* More graphs captured (8 mixed capacities); graph memory and capture time grow, within the invariant (all before Ready).
* The prefill and decode attention kernels run in one graph with a descriptor array; the paged kernels are reused, so the new risk is the descriptor
  plumbing and the shared workspace sizing, not new math.
* Not decided here: preemption of a waiting sequence, bucket sizes above 32, logprobs (`Want::Logprobs`), speculative decoding. The v2 encoding does
  not block them.

## Owner decisions (2026-11-05, recorded)
1. Approved with four edits, all applied above: (1) no `Mixed` capacity of 32 (mixed capacities 64 to 1024; decode-only buckets stay 1-32); (2) presence and
   frequency penalties count output tokens only, the repetition penalty keeps prompt + output (also in `sampler_reference.rs` and `docs/sampler-v2.md`);
   (3) startup check: `--step-token-budget` must be at least `max_batch + 64`, otherwise the CLI refuses to start with a typed error naming both values;
   (4) the per-slot 16-bit output-token counts (9.7 MB at 32 slots for a 152k vocabulary) are accepted and allocated before Ready.
2. The default budget starts at 384 and is set by the sweep of the verification plan.
3. StepPlan v1 -> v2 is one bump exactly as section 4 says (with edit 2).
