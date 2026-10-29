# Phase 04: Model breadth (Ignition + oracle)

## Context Links
- research-01 sections 1-4 (ideas 1,2,3,4,5,7,8,9,11,15); CLAUDE.md rule 3
- Files: `ignition/src/hyperion_ignition/compiler/{base,registry,model_config}.py`, `compiler/architectures/{llama,qwen,deepseek}.py`, `ignition/tests/`, `specification/fixtures/`

## Overview
Priority P2. Pending. Depends: 03. Wave: W3. Effort: 6d. Add models cheaply and verifiably without GPU. Today: llama, qwen, deepseek files exist. Goal: declare, map, and prove each architecture on CPU.

## Key Insights
- Upstream breadth = many per-model files over shared components; cost per model stays large. Our lever: declaration layer + oracle + descriptor, not config-only magic.
- Open decision 1 (rec: per-arch Python) and decision 2 (rec: Python HF oracle, no Rust CPU backend) are assumed at their recommendations; if the user chooses otherwise, re-plan this phase before starting.
- Components layer only at 3rd+ real architecture (YAGNI); registry-by-import stays (rule 3).

## Requirements
1. `declaration.py`: frozen dataclasses (QualifiedCheckpoint, SamplingDefaults, ArchitectureDecl) importing no torch; NEW cli.py `list` and `fit` modes (memory-fit estimate; invoked via `hyperion-compile`) work without PyTorch.
2. Qualified-checkpoint record per arch (repo, revision, bytes, target_arch, formats); unqualified combination refused with supported list.
3. Weight mapping: explicit table; build FAILS on missing tensor AND on unmapped tensor.
4. Teacher-forced logit-trace oracle: versioned artifact (model id, revision, prompt tokens, teacher tokens, logits) under `specification/fixtures/logit-trace/`, produced by HF forward in Ignition, consumed by Helios tests later (and by Ignition's own reference).
5. Chat template, EOS ids, sampling defaults + ranges go into manifest (fields from phase 03); Helios holds no per-model policy.
6. Architecture descriptor strategy: ModelConfig grows by tagged variants (never model names). Generic-decoder question = open decision 1; plan assumes its recommendation (per-arch Python).

## Architecture
declaration (torch-free) -> architectures/<name>.py (torch allowed, registered via import in `__init__.py`) -> mapper (strict) -> manifest writer. Oracle generator is a build-time tool; artifact format is JSON + float32 npy/safetensors blob.

## Related Code Files
- Create: 04 ledger section appended to docs/unverified-ledger.md (after 08 landed; oracle skip + Helios-vs-oracle row), compiler/declaration.py, compiler/weight_mapping.py, compiler/oracle/logit_trace.py, tests/test_declaration_no_torch.py, test_weight_mapping_strict.py, test_logit_trace_format.py, test_logit_oracle.py (skipped when `oracle` extra absent); fixtures/logit-trace/README + one tiny-model trace; docs/add-a-model.md (new file)
- Modify: architectures/{llama,qwen,deepseek}.py (add decl + mapping), cli.py (`list`, `fit`; this phase owns cli.py), registry.py, ignition/pyproject.toml (new optional extra `oracle = [torch, transformers, safetensors, numpy]`, reason: HF forward is the canonical oracle and is build-time only, so it must not enter the base or `[test]` deps; plus two hand-offs from phase 06 in the same file: the `[project.scripts]` line `hyperion-sign = "hyperion_ignition.signing_cli:main"` and base dependency `cryptography` (Ed25519 signing; maintained, wheel-distributed, no libsodium build; 06 imports it); re-run `pip install -e "ignition[test]"` after editing)

## Implementation Steps
1. Tests first: importing declaration with torch blocked (sys.modules) works; strict mapper rejects injected extra/missing tensor.
2. Implement declaration + mapper; migrate qwen then llama; deepseek only declared if mapping complete, else marked unqualified.
3. Logit-trace: generate from a tiny real HF checkpoint (random-init tiny config is real forward, not fake data; state this); define tolerance fields per trace.
4. Qualified-checkpoint refusal path in cli.
5. Doc: "add a model" checklist (7 steps from research-01 sec 2, own wording) in new `docs/add-a-model.md`.
6. pyproject: add `oracle` extra, the `hyperion-sign` script line and the `cryptography` dep; oracle test uses `pytest.mark.skipif(importlib.util.find_spec("torch") is None)`; the skip is reported in the unverified ledger.

## Todo
- [ ] declaration  [ ] strict mapper  [ ] oracle format+tool  [ ] qualify 2 archs  [ ] checklist doc

## Success Criteria
- [RUN] `pytest ignition/tests`; mapper test fails on both missing and unmapped.
- [RUN, requires `pip install -e "ignition[oracle]"`] HF-vs-Ignition logit agreement within the trace's declared tolerance on the tiny model (CPU); The gate REPORTS the skip when the extra is absent (skipped test listed in the ledger, never counted as pass); the extra must be installed only to claim "oracle evidence"; without it that claim is not made.
- [NEEDS-GPU] Helios-vs-oracle comparison: no Rust test is added here (no backend forward exists to call; a stub would be a placeholder). Tracked as a phase 09 item; pass condition for this phase = a ledger row for it exists in docs/unverified-ledger.md.

## Test Strategy & Quality Gate
- Spec: Goal = new arch is addable with proof. AC: Given checkpoint with extra tensor Then build error naming it; Given torch absent Then `hyperion-compile list` works; Given trace Then re-running HF reproduces logits within tol. I/O: trace artifact schema v1. Out of scope: GPU numerics, MoE/MLA kernels, vision, speculation.
- Pyramid 75/20/5. E2E: `hyperion-compile` (existing console script; cli.py owned here) on a tiny HF model -> manifest validates -> trace verifies. `list`/`fit` are NEW cli.py subcommands added by this phase.
- Coverage diff >=90/75. Evidence: phase-01 templates `CMD-PY-COV` and `CMD-DIFF` (Python form).

## ADR Rationale
Context: breadth goal vs 3 archs and no GPU. Decision: declaration layer + strict mapping + CPU logit oracle; defer components layer and generic decoder. Why over alternatives: component layer now (premature, 2-3 archs), descriptor-generic decoder now (needs Rust runtime and oracle first; user decision 1), skipping oracle (no proof of correctness on this host).

## Risk Assessment
Tiny-model oracle may hide quantization/large-model issues: label as structural proof only. Tolerance choice arbitrary: record per-trace, never loosen to pass. HF version drift: pin transformers version in trace metadata.

## Security Considerations
No `trust_remote_code`; reject checkpoints with pickle weights (safetensors only).

## Next Steps
Feeds 09 (Helios-vs-oracle comparison, per-layer residual comparison). Out of scope: new architectures beyond what a user requests (ask first).
