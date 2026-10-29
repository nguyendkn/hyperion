# Goal: Hyperion upstream-distill roadmap (multi-model, multi-infra)

> **Execution flow: /vk-hoh-run** — spec = `plans/20261028-2209-hyperion-upstream-distill-multimodel-multiinfra/spec.md`, loops = 3.
> Inner loop: /vk-hoh-plan → /vk-hoh-dev → /vk-hoh-qa per loop × T.

## Mission
Implement phases 01-08 of the plan so a single-GPU path is real, models/infra/package/serving breadth is added without breaking `core` rules, and every claim is either run here or ledgered as unverified.

## Context & Key Files
- Plan: `plans/20261028-2209-hyperion-upstream-distill-multimodel-multiinfra/plan.md`; phases `phase-01..09-*.md` (same dir)
- Research: `research/researcher-0{1..4}-*.md`; validation: `reports/validation-round-*.md`
- Real-GPU findings (H100, Qwen3-0.6B): `reports/gpu-survey-h100-qwen3-0.6b.md` (merged into phase 01)
- GPU track outside this goal: phase 10 (first forward, done) and phase 12 (decode performance); phase 11 (tiered KV store) is entry-gated; phase 13 (multi-GPU, multi-node) runs its design items beside this goal and lands them through phases 03, 05, 06, 07
- Repo rules: `CLAUDE.md`; code: `helios/crates/*`, `specification/`, `ignition/`

## Requirements
**Must do:**
- Follow waves: W1 (01, 08) → W2 (02, 03) → W3 (04, 05, 06) → W4 (07); respect file ownership in plan.md's matrix and the ledger rule.
- TDD per phase Test Strategy; use the CMD-RUST-COV / CMD-PY-COV / CMD-DIFF templates from phase 01.
- Add the two survey items: bump `tokenizers` (0.19 fails on Qwen3 tokenizer.json) with a Qwen3 fixture test; make Ignition compilers check `architectures`/`model_type`.
- Report design choices you make on your own.

**Must not:**
- Copy upstream code or text (ideas only); add mocks/fakes; panic on bad input.
- Decide open decisions 1-11 in plan.md: phase 03 needs the user's answers on 1,3,5,6,7 first (ask, then proceed).
- Edit `helios-core` for hardware specifics; break manifest v1 consumers.

## Success Criteria
- `cargo test --workspace --features stub`, `cargo check --workspace`, `pytest ignition/tests`, `cargo test -p hyperion-specification --features schema` all exit 0.
- `cargo test -p helios-hal` >= 12 tests; backend whose arch != probed GPU is refused.
- `/v1/chat/completions` + `/health` work; over-long prompt -> 400; default bind 127.0.0.1.
- Qwen3-0.6B tokenizer loads in helios-core; Qwen3 config is not accepted by the qwen2 compiler.
- `docs/unverified-ledger.md` lists every `[TYPE-CHECK]`/`[NEEDS-GPU]` item.

## Out of Scope
- Phase 09 backlog (speculation, multi-host, ...), phase 11 (tiered KV store, entry-gated) and the GPU track (phases 10, 12), which run in their own sessions on H100 pods.
- sm120/sm90 runtime claims without a GPU (staging H100 pods may be used for hardware-gated tests only).

## Verification
```bash
cargo test --workspace --features stub && cargo check --workspace
pytest ignition/tests
cargo test -p hyperion-specification --features schema
```
