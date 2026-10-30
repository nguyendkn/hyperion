# 0010: spec group 1, model constants and provenance in the signed manifest

Status: **accepted for group 1** (owner authorisation A2 of the 2026-10-30 run). Parallelism fields, sharding plans, `deployment` and capabilities are NOT part of this decision; they wait for the owner's review of ADR 0005. Plan: phase 03, `docs/model-constants-audit.md`.

## Context
`specification/v1` is editable until the first released package. The runtime read hidden size, MLP width, RoPE base, norm epsilon, tied head and the EOS token from a `config.json` beside the package, outside the signature, and hard-coded the variant of every operation. Both are things a release cannot change without a `v2`.

## Decision
- **Constants move into `ModelConfig`** (all required, no default): `hidden_size`, `intermediate_size`, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`, `qkv_bias`, `qk_norm`, `activation`, `rope_style`; `attention_scale` is optional (absent = `1/sqrt(head_dim)`). `activation` and `rope_style` are enums with only the supported variant. The runtime refuses a combination its kernels do not implement with a typed `UnsupportedModel` (QK-norm, no QKV bias, another attention scale).
- **Generation:** `generation.eos_token_ids` (union of `config.json` and `generation_config.json`, deduplicated, at least one) and optional `sampling_defaults`. **`chat_template`** is the template text plus the bos/eos token texts, inside the signed manifest. **`qualified_checkpoint`** is `{repo, revision}` (40-hex commit); the weights' hashes are already in `payloads`, so they are not repeated.
- **Contract revision:** `contract_revision` (now 1) is required; `Manifest::from_json_bytes` checks `manifest_version` and the revision before any other field and returns `UnsupportedContractVersion` with a "re-export the package with Ignition >= 0.2.0" hint. v1 is not frozen, so an older manifest is refused, not defaulted.
- **`cache_identity()` is computed, never stored:** SHA-256 over a domain tag, the payload hashes, the layout fingerprint, `target_arch`, `tensor_format` and the model-config hash. The signature, chat template and EOS list are outside it. The SHA-256 is written out in the specification crate (NIST vectors plus Python `hashlib` vectors in tests) because adding `sha2` would be a new dependency for a crate the runtime links.
- **Ignition** emits everything from real Hugging Face files and refuses a missing source (no RoPE base, no epsilon, no tied-head flag, no EOS anywhere, a non-silu activation, partial rotary, scaled RoPE, a list of chat templates). Both `rope_theta` and transformers 5's `rope_parameters` are read. Its version is 0.2.0.

## Owner decisions recorded as recommended
1. Generic decoder versus per-architecture: **per-architecture Python**, descriptor fields grown additively; revisit at the third architecture.
3. Parallelism fields in v1: **add now, but fields pending ADR 0005** (not in this change).
5. Publisher identity: **pinned keys plus a local trust store**.
7. One deployment per bundle: **one package per target arch**; TP size chosen at load.
20. Shard at load from canonical weights: **yes; fields pending ADR 0005**.

## Consequences
Packages written before this change must be re-exported. The loader reads the dimensions from a signed source; a changed `rope_theta` now breaks the signature once signing exists. The e2e stub fixture, the shared golden manifests, the tiny sharded fixture and the schema were regenerated together.
