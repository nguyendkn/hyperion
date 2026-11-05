# Requests from the speech track to the LLM track

No LLM-owned file is edited by the speech track (run protocol, section 4). Each request: what, why, smallest change.

## R1: embedding prefill in StepPlan v2 (owner direction 2026-11-05: ship it with the v2 bump of ADR 0011)
**Why.** Qwen3-ASR (ADR 0013) replaces placeholder positions of the decoder input with audio-encoder frames; the TTS backbone (ADR 0014) sums several codebook embeddings per frame. `StepPlan` `Input` is `Tokens` or `FromDevice` only (`helios-hal/src/step.rs`).

**Proposed addendum to ADR 0011 (StepPlan v2): add `Input::Embeddings` in the same bump.**
- The embeddings sit in a device buffer owned by the decoder session, allocated before Ready with a declared maximum size; the plan carries only an offset and a row count.
- The embedding gather kernel reads a per-token source flag (token id or embedding buffer) and runs in the existing graphs; no new graph.
- **Prefix cache (correctness trap).** ADR 0008 hashes token ids. Positions filled from embeddings have only placeholder ids, so two different audio inputs with the same placeholder ids would hash equal and a later request would reuse the earlier request's KV: a wrong answer with no error. One of these is mandatory, recorded in ADR 0008: (a) the identity of the embedding span (sha256 of the input audio or its features) enters the hash chain, or (b) blocks containing embedding tokens are never cached.
- Test: two different audios with the same placeholder ids never hit each other's cache.

## R2 (open, to be confirmed by reading the pinned Hugging Face code): MRoPE variant
Qwen3-ASR's decoder config has `mrope_interleaved [24,20,20]`. If for audio-only input the three position components do not reduce to plain one-dimensional RoPE, the RoPE kernel (LLM-owned) needs a variant. Status: unchecked (S1.2).

## Not requested
Cross-attention, paged-attention changes, sampler changes: the chosen families need none.
