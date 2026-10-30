# 0005: manifest fields for tensor parallelism (sharding plan, supported TP sizes, deployment)

Status: **accepted** (owner review, 2026-10-30). Design only: the `specification/` edit and the loader are a separate step (phase 03, second group), and nothing here is implemented. Plan: phase 13 D2 and phase 03 requirement 4; owner decisions 3 ("add now") and 20 (shard at load). Related: ADR 0003 (payload table), 0006, 0010.

## Context
`specification/v1` stays editable only until the first released package. Fields that describe how a model may be split across GPUs are cheap to add now and force a `v2` later. Weights stay canonical (unsharded safetensors, ADR 0003); Hyperion captures graphs at load, so one package can serve several TP sizes if each rank reads its own slices at load.

## Decision
1. **Sharding plan per tensor class, declared by the compiler (Ignition).** Each layer entry says how its tensor may be split: `column` (output dimension: Q, K, V, gate, up), `row` (input dimension: attention output, MLP down), `replicated` (norms) or `vocab` (embedding and output head). With the shape this is all a loader needs to compute a rank's slice, including slices that lie across several payload files.
2. **Supported TP sizes, derived and recorded.** A TP size `t` is supported only when **`t` divides `num_attention_heads`** and the KV heads either divide by `t` or `t` is a multiple of `num_kv_heads` (then each KV head is replicated on `t / num_kv_heads` ranks, the rule stated in the manifest). The compiler writes the list; a request outside it fails with a typed error before any CUDA initialisation. **Qwen2.5-7B (28 heads, 4 KV heads) supports TP 1, 2 and 4 and refuses 8** (28 is not divisible by 8); Qwen2.5-0.5B (14 heads, 2 KV heads) supports 1 and 2; Qwen3-0.6B (16 heads, 8 KV heads) supports 1, 2, 4 and 8.
3. **Vocabulary padding.** A vocab-split embedding and head need the vocabulary to divide by `t`: the loader pads it to a multiple of `t * 64` rows, the padded embedding rows are zero, and the padded logit columns are masked to `-inf` before any softmax, argmax or sampling (so a padded id can never be produced). `vocab_size` in the manifest is the real one; the padding is a load-time fact, not stored.
4. **Biases.** A bias of a column-split projection (Q, K, V) is split with its output rows. A bias of a row-split projection (none in Qwen2 or Qwen3; any future one) is added **exactly once, after the all-reduce**, never per rank.
5. **Shared embedding.** When the vocabulary is split, the embedding slice a rank holds is the same buffer its output-head slice reads (tied embeddings): one copy per rank, not two. The lookup is a masked gather (ids outside the rank's slice give zeros) followed by the all-reduce; the logits are produced vocab-parallel and gathered (ADR 0009) or reduced for the argmax.
6. **Fused QKV and gate/up per rank.** The fused matrices of a decode step (ADR 0010's `qkv` and `gate_up` groups) are built **from the rank's own head slices**: `[q_rank | k_rank | v_rank]` and `[gate_rank | up_rank]`, placed in one allocation at upload. A fused matrix never spans ranks.
7. **`deployment` is not a package property.** The runtime takes `{tp_size, pp_size}` at load (decision 20: one package for every TP size, sharded at load).
8. **Pipeline-parallel boundaries are chosen at load**, from `pp_size` and the layer count (balanced layer groups; the embedding on the first stage, the final norm and head on the last), not declared in the manifest.
9. **Additive and optional.** The new fields are optional with defaults that mean "single GPU, no split"; `deny_unknown_fields` stays.
10. **Where the numbers come from.** The classes follow the architecture mapper; a test shards and re-assembles every tensor of the committed sharded fixture and compares bytes.

## Why over the alternatives
- Pre-sharded packages per TP size: faster load, but a package and a signature per TP size (decision 20's other option).
- Leaving it to the runtime to infer from tensor names: the inference lives in the runtime and differs per architecture.
- Waiting for the first release: a required field later is `v2`.

## Consequences
- Spec, JSON schema (regenerated) and the Ignition writer change together, with golden fixtures and a Rust/Python round trip, in the phase 03 pre-freeze window.
- The loader (phase 06) reads rank slices; it needs the payload table (done) and the vocabulary padding rule.
- Attention and MLP kernels see per-rank head counts; `heads / kv_heads` per rank stays at most 8 or the kernel refuses.

## Verification plan
Golden manifests with and without the new fields; a unit test that computes every rank's slice for TP 1, 2, 4 on the real Qwen2.5-7B shard headers (and the refusal of TP 8 and TP 3), checks the slices tile each tensor exactly once, and checks that a replicated KV head is read by the right ranks; the vocabulary padding mask on a tiny model; refusal of an unsupported TP size before CUDA init.
