# 0005: manifest fields for tensor parallelism (sharding plan, supported TP sizes, deployment)

Status: **proposed**, needs owner sign-off before any spec change (not accepted; `specification/` is NOT edited by this document). Plan: phase 13 D2 and phase 03 requirement 4; owner decisions 3 (now "add now") and 20 (shard at load). Related: ADR 0003 (payload table), 0006.

## Context
`specification/v1` stays editable only until the first released package. Fields that describe how a model may be split across GPUs are cheap to add now and force a `v2` later. Weights stay canonical (unsharded safetensors, ADR 0003); Hyperion captures graphs at load, so one package can serve TP 1, 2, 4 or 8 if each rank reads its own slices at load.

## Decision (proposed)
1. **Sharding plan per tensor class, declared by the compiler (Ignition).** Each layer entry says how its tensor may be split: `column` (output dimension, for example Q, K, V, gate, up), `row` (input dimension, for example attention output and MLP down), `replicated` (norms, biases that are not split) or `vocab` (embedding and output head). Together with the shape this is all a loader needs to compute a rank's slice, including slices that lie across several payload files.
2. **Supported TP sizes, derived and recorded.** The compiler computes the set of TP sizes the model allows from its head counts (a TP size must divide the attention head count; with fewer KV heads than TP size the KV heads are replicated, the rule stated explicitly: `replicas = tp / kv_heads`), and writes it to the manifest. A request outside the list fails with a typed error before any CUDA initialisation.
3. **`deployment` is not a package property.** The runtime takes `{tp_size, pp_size}` at load (decision 20: one package for every TP size, sharded at load). The manifest only constrains what is allowed.
4. **Additive and optional.** All new fields are optional with defaults that mean "single GPU, no split"; older manifests parse unchanged; `deny_unknown_fields` stays.
5. **Where the numbers come from.** Q, K, V, gate and up are `column`, o_proj and down are `row`, embeddings and the output head are `vocab`, norms are `replicated`: derived from the architecture mapper, checked by a test that shards and re-assembles a tensor of the committed sharded fixture and compares bytes.

## Why over the alternatives
- Pre-sharded packages per TP size: faster load, but a package and a signature per TP size (decision 20's other option).
- Leaving it to the runtime to infer from tensor names: the inference lives in the runtime and differs per architecture; the compiler already knows each architecture.
- Waiting for the first release: a required field later is `v2`.

## Consequences
- Spec, JSON schema (regenerated) and the Ignition writer change together, with golden fixtures and a Rust/Python round trip, in the phase 03 pre-freeze window.
- The loader (phase 06) reads rank slices; it needs the payload table already present.
- Sign-off is needed on field names and on the KV-head replication rule before implementation.

## Open questions for the owner
- Decision 20 (shard at load; recommended) and decision 3 ("add now").
- Field names and the `replicated` rule for biases of column-split projections (a bias is split with its output rows).
- Whether pipeline-parallel stage boundaries (layer groups) are declared by the compiler or chosen at load.

## Verification plan
Golden manifests with and without the new fields; a unit test that computes every rank's slice for TP 1, 2, 4, 8 on the real Qwen2.5-7B shard headers (4 KV heads: TP 8 needs replication) and checks the slices tile each tensor exactly once; refusal of an unsupported TP size before CUDA init.
