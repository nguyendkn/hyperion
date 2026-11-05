# Requests from the speech track to the LLM track

No LLM-owned file is edited by the speech track (run protocol, section 4). Each request: what, why, smallest change.

## R1: embedding prefill in StepPlan v2 (owner direction 2026-11-05: ship it with the v2 bump of ADR 0011)
**Why.** Qwen3-ASR (ADR 0013) replaces placeholder positions of the decoder input with audio-encoder frames; the TTS backbone (ADR 0014) sums several codebook embeddings per frame. `StepPlan` `Input` is `Tokens` or `FromDevice` only (`helios-hal/src/step.rs`).

**Proposed addendum to ADR 0011 (StepPlan v2): add `Input::Embeddings` in the same bump.**
- The embeddings sit in a device buffer owned by the decoder session, allocated before Ready with a declared maximum size; the plan carries only an offset and a row count.
- The embedding gather kernel reads a per-token source flag (token id or embedding buffer) and runs in the existing graphs; no new graph.
- **Prefix cache (correctness trap).** ADR 0008 hashes token ids. Positions filled from embeddings have only placeholder ids, so two different audio inputs with the same placeholder ids would hash equal and a later request would reuse the earlier request's KV: a wrong answer with no error. One of these is mandatory, recorded in ADR 0008: (a) the identity of the embedding span (sha256 of the input audio or its features) enters the hash chain, or (b) blocks containing embedding tokens are never cached.
- Test: two different audios with the same placeholder ids never hit each other's cache.

## R3: return the final hidden row of a step (TTS backbone)
**Why.** The VieNeu acoustic decoder consumes the backbone's final hidden state per frame (ADR 0014 amendment 2), not logits. **Smallest change:** a `Want::Hidden` value next to `Want::Token`/`Want::Logits` in the StepPlan row, which makes the step write the post-final-norm hidden row of that position into an output buffer the plan names (or the step output); no new graph if the existing final-norm output is copied out before the head GEMM. Test: the returned row equals the reference's `last_hidden_state` row within the declared tolerance. Needs R1 (embedding input) too; both ride the StepPlan v2 bump.

## R2 (CLOSED 2026-11-05): MRoPE variant is not needed
Verified in the pinned reference code: for audio plus text the three position components are identical, and the interleaved-mrope step returns the first one unchanged, so plain rotate-half RoPE at theta 1e6 is bit-identical (ADR 0013 amendment). Nothing to change in the LLM RoPE kernel.

## Not requested
Cross-attention, paged-attention changes, sampler changes: the chosen families need none.
