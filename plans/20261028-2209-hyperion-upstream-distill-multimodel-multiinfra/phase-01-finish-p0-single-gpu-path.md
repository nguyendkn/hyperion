# Phase 01: Finish P0 so a single-GPU path is real

## Context Links
- `docs/project-overview-pdr.md` section 9 items 1, 7 and FR-8; CLAUDE.md rules 5-7; research-02 idea 4 (capacity errors)
- Files: `helios/crates/helios-hal/src/{backend_factory,triage,traits}.rs`, `helios-cli/src/main.rs`, `helios-core/src/api.rs`, `Taskfile.yml`, `Cargo.toml`, `ignition/pyproject.toml`

## Overview
Priority P1. Status pending. Depends: none. Wave: W1. Effort: 4d (incl. H100 survey items). Make load path honest and fix the cheap correctness bugs the H100 survey found (reports/gpu-survey-h100-qwen3-0.6b.md): probe real arch for every backend, exact match, format check, `mark_loaded` only after real load, safe defaults, and install the dev tooling every later gate uses.

## Key Insights
- Capability table (arch -> supported tensor_formats) lives in ONE place (Rust), Python recipes cross-checked by a test (commit 2c40cf7 already has `supported_formats` on recipes).
- Qwen2-7B max_seq_len 131072 => ~7.5 GB KV/sequence; runtime limit must be separate from model limit.
- `cargo test -p helios-core --features stub` is invalid (`stub` lives on helios-hal); use `cargo test --workspace --features stub` or per-crate without the flag.

- H100 survey [V]: `BlackwellSm120` was accepted on `sm_90` (`outcome=Exact`) because triage compares manifest vs probed cap only; `tokenizers` 0.19.1 cannot parse Qwen3's tokenizer.json (HTTP 422); Ignition wrote `model_id: qwen2` for a Qwen3 config.

## Requirements
- Func: factory probes real arch (sm120, sm90, stub reports `stub`); triage refuses unless `target_arch` equals probed exactly (rule 6); refuses when `tensor_format` not in backend formats (Hopper never FP4, rule 7); refusal text = needed vs got + fix hint; typed error, no panic.
- main.rs: TODAY the chat route is `/chat/completions` at root (`api::router().with_state(app_state)`, main.rs:163; api.rs:60-61; no nest exists). NEW: nest the chat router under `/v1` so the public path is `/v1/chat/completions`; NEW `GET /health` stays at root: 200 when Ready/Serving, 503 otherwise (api.rs); `mark_loaded` after load success only; bind `127.0.0.1` default; default log `info`; `--max-seq-len` runtime cap, error if > model max. This phase owns the capacity error (cap/KV fit) returned before the socket opens; phase 07 owns only SSE/429 semantics. `/v1/models` is phase 07, not here.
- Triage also compares the backend's OWN arch with the probed arch: `BlackwellSm120` on `sm_90` is refused (real-log test case); `create_backend` falls through sm120 -> sm90 -> stub on refusal instead of returning at the first candidate.
- Tokenizer: bump workspace `tokenizers` 0.19 -> 0.22; re-check the `onig` feature workaround (helios-core/Cargo.toml) and drop it if 0.22 no longer needs it (state the evidence); real model files are NOT committed: they are pinned in `testdata/models.json` (repo, revision, sha256), live in gitignored `testdata/models/<id>/` and are fetched by `scripts/fetch-model-fixtures.py`; tests are generic over the registry (no per-model test files) and skip visibly when files are absent; a generic test encodes a multilingual prompt with every fetched tokenizer.
- Ignition: each compiler declares the HF `architectures` it accepts; a config whose `architectures` is absent from the list is refused with the supported list (an unsupported family such as Qwen3 is refused until its compiler + QK-norm in `ModelConfig` exist; registry `compiler: null` encodes that); CLI exits non-zero via ClickException and writes nothing.
- Non-func: backend selection order sm120 -> sm90 -> stub unchanged.

## Architecture
`backend_factory` builds `ProbedBackend { arch, formats }` -> `triage(&Manifest, &ProbedBackend) -> Result<(), TriageError>` (pure fn, unit-testable without GPU). `ProbedBackend`/`triage` are internal to helios-hal; `create_backend` is the only cli-facing API. Format table `fn supported_formats(arch) -> &[TensorFormat]` single source; JSON dump of the table (committed `helios-hal/tests/data/format-table.json`, regenerated and asserted equal by a helios-hal test) asserted by the Python test. Phase 05 supersedes `ProbedBackend` and this table with per-backend `Capabilities` and migrates the table + cross-check test; 01 keeps its table until then.

## Related Code Files
- Modify: helios-core/Cargo.toml + root Cargo.toml (`tokenizers` bump, `onig` check), `helios-core/tests/model_tokenizers.rs` (generic), `testdata/{models.json,README.md}`, `scripts/fetch-model-fixtures.py`, ignition/.../compiler/architectures/{llama,qwen,deepseek}.py + compiler/base.py (accepted `architectures`; sequenced hand-off to 04, which owns compiler/** in W3), backend_factory.rs, triage.rs, backends/{stub,hopper_sm90/backend,blackwell_sm120/backend}.rs (probe fn only), helios-cli/main.rs (nest chat router under `/v1`), helios-cli/Cargo.toml (`[features]` stub/sm120/sm90 forwarding), helios-hal/build.rs + helios-hal/Cargo.toml (`cc` optional-dep -> [build-dependencies] only if build.rs calls it; see step 5), .gitignore (add `coverage*.xml`, `.coverage`), helios-core/src/api.rs (add `/health`; fix stale doc comment at api.rs:58-59 "Mounted under /v1 by the CLI" to match the real nest), Taskfile.yml (add `check:sm120` only; `check`, `test`, `schema:check` already exist), root Cargo.toml (drop unused deps; workspace `members` gains `helios/crates/*`), ignition/pyproject.toml (`[test]` extra gains pytest-cov)
- Create: helios-hal/tests/triage_matrix.rs, helios-hal/tests/data/format-table.json (+ the test that writes/verifies it), ledger-row hand-off list for 08 (no edit to docs/ here), ignition/tests/test_format_table_crosscheck.py, ignition/tests/test_architecture_check.py + generic ignition/tests/test_model_fixtures.py (real configs via the registry), helios-cli/tests/e2e_stub.rs + helios-cli/tests/fixtures/stub-manifest.json (paths via `env!("CARGO_MANIFEST_DIR")`, binary via `env!("CARGO_BIN_EXE_helios-cli")`)

## Implementation Steps
1. Dev tooling (do first, report each install and why): `cargo install cargo-llvm-cov --locked` + `rustup component add llvm-tools-preview` and `rustup +nightly component add llvm-tools-preview` (branch coverage needs nightly, quality-gate.md section 5); add `pytest-cov` to ignition `[test]` extra THEN re-run `pip install -e "ignition[test]"`; `pip install diff-cover` (CLI, not a project dep; diff-cover accepts lcov, verified; use `cargo llvm-cov --cobertura` only as the route to a real branch diff gate, since `--branch-coverage` needs Cobertura XML); go-task OPTIONAL. If an install fails, record coverage as unverified; never hand-compute.
2. Write failing triage matrix tests (arch x format x manifest) first.
3. Implement probe + table; wire factory; keep sm120 FFI cfg-gated.
4. Refactor main.rs: load-then-mark, bind/log/limit flags, capacity error before socket; nest chat router under `/v1` (fix api.rs:58-59 comment); add `/health` at root.
5. Add `[features]` forwarding in helios-cli (fixes `build:sm90`/`build:sm120` passing `--features` to a crate with none); move `cc` from the optional [dependencies] entry to [build-dependencies] only if build.rs actually calls it (the `cc::Build` call at helios-hal/build.rs is commented out today; sm90 feature wiring unchanged); remove unused deps (list each with reason); add `check:sm120` task.
6. Python cross-check test reads Rust table dump (`cargo run` bin or committed JSON) and compares to recipes.
7. Hand-off to 08 (01 is the only phase that hands rows to 08): list ledger rows (rows verified on H100 on 2026-10-28, tag `[RUN on H100]`: sm120 backend links with the CUDA 12.4 toolkit, device probe reports `sm_90`, SIGTERM shutdown returns GPU memory to 0 MiB, idle context 518 MiB; still [NEEDS-GPU]: sm90 TensorRT-LLM path, any kernel; Rust branch-gate UNVERIFIED if it occurs; 01 never edits docs/unverified-ledger.md, it records rows in its own report); 08 consumes them after this step (intra-W1 order).

8. Survey items: bump `tokenizers` + fixture test; compilers' `architectures` check + tests (real Qwen3 config refused, real Llama/Qwen2 configs still accepted); factory fall-through + `BlackwellSm120`-on-`sm_90` refusal test.

## Todo
- [ ] dev tooling  [ ] triage tests  [ ] probe+table  [ ] main.rs + /health  [ ] Cargo/Taskfile  [ ] cross-check test  [ ] e2e

## Success Criteria
- [RUN] `cargo test -p helios-hal` reports >=12 passed (named matrix cases: arch match/mismatch x fp4/fp8/bf16 x sm90/sm120/stub, plus unknown arch); `cargo check --workspace`; `pytest ignition/tests` green.
- [TYPE-CHECK] `cargo check -p helios-hal --features sm120`; real sm120/sm90 probe = [NEEDS-GPU].
- [RUN] Every fetched real `tokenizer.json` loads and encodes a multilingual prompt under `tokenizers` 0.22; `hyperion-compile --model qwen2` on the real Qwen3 config exits non-zero listing supported architectures; triage test: backend arch `sm_120` vs probed `sm_90` -> `ArchMismatch`.
- [RUN] helios-cli e2e: mismatched arch/format exits non-zero with message, no panic; load failure leaves `/health` 503.

## Test Strategy & Quality Gate
- Spec: Goal = refuse unsafe loads, serve only after load. AC: Given manifest sm90+fp4 When triage Then `FormatUnsupported`; Given arch mismatch Then `ArchMismatch`; Given load failure Then state != Ready and `GET /health` 503. I/O: `triage(&Manifest,&ProbedBackend)->Result<(),TriageError>`. Out of scope: multi-GPU, real kernels.
- Pyramid 80/15/5 (Rust workspace). E2E: `helios-cli` binary against `stub-manifest.json`: start, GET `/health` 200, POST `/v1/chat/completions` (valid only after the `/v1` nest) returns the documented 501/503, bad-arch manifest refused.
- Coverage: diff line >=90 / branch >=75 (quality-gate.md section 5), measured with the shared templates below.
- **Shared command templates (SSOT; other phases cite by name, add only `-p`/scope args):**
  - `CMD-RUST-COV`: `cargo +nightly llvm-cov --workspace --features stub --branch --lcov --output-path target/coverage/<phase>/lcov.info`; branch table: `cargo +nightly llvm-cov report --branch`. Scoped runs replace `--workspace --features stub` by `-p <crate> [--features <f>]` (e.g. `-p hyperion-specification --features schema`); never combine `--workspace` with `-p`. Each phase writes under its own `target/coverage/<phase>/` (gitignored via target/). If `grep -c '^BRDA' <lcov>` is 0 or `--branch` is rejected, the gate states "branch gate UNVERIFIED on this host" and records the row per the plan.md ledger rule (line gate still enforced).
  - `CMD-PY-COV`: `pytest ignition/tests --cov=hyperion_ignition --cov-branch --cov-report=xml:target/coverage/<phase>/coverage-<phase>.xml`; Python branch % = `coverage report` right after the same run (or `branch-rate` in the XML).
  - `CMD-DIFF`: `diff-cover target/coverage/<phase>/lcov.info --compare-branch=origin/main --fail-under=90` (Rust) or `diff-cover target/coverage/<phase>/coverage-<phase>.xml --compare-branch=origin/main --fail-under=90` (Python). With lcov, diff-cover reports lines only (lcov-specific caveat); branch % for changed files is read from the `CMD-RUST-COV` branch table / the `coverage report` branch column, never hand-computed. If `origin/main` is absent, report UNVERIFIED.
- Evidence (no `task` needed): `cargo test --workspace --features stub`; `CMD-RUST-COV`; `CMD-DIFF`; `CMD-PY-COV`.

## ADR Rationale
Context: format/arch rules are scattered. Decision: one Rust table + pure triage function, Python cross-checks. Why over alternatives: duplicating tables in recipes only (drift, found after commit 2c40cf7) or generating Rust from Python (violates Rust-as-source-of-truth).

## Risk Assessment
Existing behaviour change (bind address) breaks local scripts: note in commit message. Probe on sm90/sm120 unrunnable here: ledger entry via 08 hand-off.

## Security Considerations
Default loopback bind; no auth added yet (phase 07 decides bearer). Triage errors must not leak paths. `/health` exposes state only.

## Next Steps
Unblocks 02, 03, 05, 10. Out of scope: truncation/in-flight (02), cache (07), trait decoupling (05).
