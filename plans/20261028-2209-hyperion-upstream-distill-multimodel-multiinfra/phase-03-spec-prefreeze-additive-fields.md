# Phase 03: Spec pre-freeze additive fields (decision gate)

## Context Links
- CLAUDE.md rule 4 (v1 immutable after first release), schema decisions; research-02 ideas 1,5; research-03 idea 4; research-04 idea 7; research-01 ideas 2,9,11
- Files: `specification/src/{lib,schema}.rs`, `specification/src/v1/mod.rs`, `specification/tests/manifest.rs`, `specification/schema/manifest.v1.json`, `ignition/.../manifest_schema.py`

## Overview
Priority P1. Pending. Depends: 01. Wave: W2. Effort: 3d. v1 is not frozen: the cheapest moment to add every field later phases need. Fields are OPTIONAL/additive; shapes tied to open decisions 1, 3, 5, 6, 7 are drafted but need user sign-off before merge (decision 1 shapes req 1: tagged `attention`/descriptor growth in ModelConfig). This phase is the single place those five are asked; phase 06 reads the ADR and does not re-ask. NO manifest cost-catalog/graph field: the cost catalog is a Rust-side table (phase 07), outside the manifest.

## Key Insights
- After first released package, adding a required field = v2. Optional fields with serde defaults stay v1-compatible only if older readers tolerate them; `deny_unknown_fields` means readers are strict, so decide now.
- Signature covers stored manifest bytes (unchanged).

## Requirements (candidate fields)
1. `ModelConfig` additions: `attention` variants extensible (tagged, e.g. window); `eos_token_ids`, `chat_template` (payload ref), `sampling_defaults` + clamp ranges; per-model, authored not derived from generation_config.
2. `qualified_checkpoint { repo, revision, arch, stored_bytes, formats }` provenance.
3. Payload identity: per-payload sha256 of bytes only; `locations` listed separately and NOT in signed body (decision 6 affects layout).
   - **Done 2026-10-29 (ADR 0003, decision 6 answered: safetensors):** manifest `payloads` table (`file`, `byte_length`, `sha256`) in the signed body, `layers[].payload` index, compatible default of one `model.safetensors`; bare file names only. Remote `locations` stay outside the signed body and are not implemented.
4. Parallelism (phase 13 D2; decision 3 now "add now", decision 20):
   - per-tensor sharding plan declared by the compiler (column-split, row-split, replicated, vocab-split);
   - supported TP sizes derived from head counts, with an explicit KV-head replication rule when kv_heads < tp_size;
   - optional `deployment { tp_size, pp_size }` as a load-time choice, not a package property.

   Weights stay canonical (one package serves TP 1/2/4/8). A requested TP size outside the supported list is refused with a typed error before CUDA init.
5. `cache_identity` inputs: model id hash, layout fingerprint, payload id (per-payload sha256 from item 3), target_arch, tensor_format, domain-separation tag. Research-03 idea 4's "state mode ordinary/spec" input is deliberately dropped (speculation skipped).
6. Capabilities advertised to model card: max_concurrency, default_max_tokens, max_top_logprobs (0 for now).
7. Contract-version rejection requirement (owner: this phase; idea from research-04 idea 5): a reader meeting an unsupported contract version returns a typed error whose message says "re-export bundle" before any load; phase 06 implements it in helios-pkg by reference and tests it too. Acceptance test here (specification crate): fixture with unknown `manifest_version` -> `UnsupportedContractVersion` error whose text contains "re-export"; Python side rejects the same fixture.

## Architecture
Rust structs remain source of truth; schemars regenerates JSON; Ignition validates with `jsonschema`; shared fixtures under `specification/fixtures/v1` grow with each field (positive + negative).

## Related Code Files
- Modify: specification/src/v1/mod.rs, schema.rs, tests/manifest.rs, schema/manifest.v1.json (regenerated), ignition manifest_schema.py + tests/test_manifest.py, fixtures/v1/*
- Create: fixtures/v1/optional-fields-*.json, negative fixtures (unknown field, bad hash, unsupported manifest_version), docs/decisions/ ADR (new file), docs/spec-freeze-checklist.md (new file)
- Append: 03 named section in docs/unverified-ledger.md (append-only, after 08 landed; rank/deployment shape rows); no other section touched.

## Implementation Steps
1. Present candidate field list + shapes to user (open decisions 1,3,5,6,7); proceed only on answer; record answers in an ADR under `docs/decisions/`.
2. Tests first: fixtures + Rust parse/serialize round-trip + Python validate, both directions.
3. Add fields as `Option` with serde defaults; regenerate schema (`cargo run -p hyperion-specification --features schema --bin gen-schema`); drift check = `cargo test -p hyperion-specification --features schema` then `git diff --exit-code specification/schema` (`task schema:check` is the same, optional).
4. Doc: freeze checklist (what "first released package" means) in new `docs/spec-freeze-checklist.md`.

## Todo
- [ ] user sign-off (1,3,5,6,7)  [ ] fixtures  [ ] structs  [ ] schema regen  [ ] Python side  [ ] ADR

## Success Criteria
- [RUN] `cargo test -p hyperion-specification --features schema` incl. drift `git diff --exit-code`; `pytest ignition/tests`.
- [RUN] Old v1 fixtures from commit 33420be still parse (compat test passes in `cargo test -p hyperion-specification --features schema`).
- [RUN] unsupported-contract-version fixture -> typed error containing "re-export" (Rust test + Python test).

## Test Strategy & Quality Gate
- Spec: Goal = freeze-ready manifest. AC: Given old fixture Then parses; Given unknown field Then rejected both sides; Given bad payload hash Then rejected. I/O: manifest JSON per schema. Out of scope: package container (06), runtime use of fields.
- Pyramid 75/20/5. E2E: Ignition writes manifest from real HF config fixture -> Rust parses -> Rust re-emits same bytes check.
- Coverage diff >=90/75. Evidence: `cargo test -p hyperion-specification --features schema`; phase-01 templates `CMD-PY-COV`, `CMD-RUST-COV` (`-p hyperion-specification --features schema`), `CMD-DIFF` (both forms); pytest-cov from phase 01.

## ADR Rationale
Context: later phases need fields; freeze makes additions costly. Decision: add optional fields now, gated by user sign-off on rank/placement shape. Why over alternatives: add on demand (forces v2 or breaks strict readers); hard-fix TP shape blind (no multi-GPU run exists to validate shape).

## Risk Assessment
Wrong shape for rank/deployment fields: mitigate by keeping them optional and logging in the unverified ledger. Schema churn invalidates committed fixtures: regenerate once.

## Security Considerations
Hash fields length/charset validated; no path components trusted in locations.

## Next Steps
Gate for 04, 05, 06, 07, 13 (04/05/06 read its ADR; 04 reads decision 1; 13 reads the parallelism fields). Out of scope: using the fields at runtime.
