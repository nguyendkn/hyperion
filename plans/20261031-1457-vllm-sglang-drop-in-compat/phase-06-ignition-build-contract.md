# Phase 06: Ignition non-interactive build contract

## Context Links
- `ignition/src/hyperion_ignition/cli.py` (54 lines: `--model id --hw --source --out file --hf-repo --hf-revision`; exit 0/1/2; `with_suffix`; weight blob + signature not yet written), `compiler/{registry.py, architectures/*}` (llama, qwen, qwen3, deepseek), `manifest_schema.py`
- research-03 section 3; ADR 0003, 0010; old plan phase 04 (owns Ignition cli.py there: hand-off), old phase 06 (signing)

## Overview
Priority P1. Status pending. Wave 1 (Python only; parallel with 02; 05 follows in W1b). Add a machine-driveable `hyperion-build` entry so Rust can run Ignition as a subprocess with stable exit codes and JSON-lines progress, writing a complete bundle directory.

## Key Insights
- Today `cli.py` takes a registry id, not an HF id/dir; one `--out` file, sidecar derived by `with_suffix` (breaks on dotted names).
- Weights stay as source safetensors (ADR 0003): build = manifest + payload table (size, sha256) + link/copy of shards and tokenizer assets into the dir.
- Unverified (phase 01 closes): does `compile()` copy assets; Ignition imports (torch eager?). If torch is imported, make it lazy; serve-time env should need only python + click + jsonschema (+safetensors header read).
- Ignition must never touch the network (downloader is Rust); subprocess env sets `HF_HUB_OFFLINE=1`.

## Requirements
Functional:
1. New console script `hyperion-build` (new file `build_cli.py`, `cli.py` untouched except import reuse) : `--source <dir> --out-dir <dir> --hw <arch, always explicit: Rust passes `RuntimeConfig.target_arch`; no GPU probe in Python> [--model <id>|auto] [--version-json] [--hf-repo --hf-revision] [--dtype auto|bf16] [--chat-template-file F] [--link-mode symlink|hardlink|copy] [--known-sha256 FILE=HEX ...] [--progress-json] [--sign-key PATH]`.
2. `--model auto`: read `config.json` `architectures[0]` -> registry id via explicit table (`Qwen2ForCausalLM`->qwen2, `Qwen3ForCausalLM`->qwen3, `LlamaForCausalLM`->llama3, ...; no decorator scanning, rule 3). Unknown -> exit 10 naming supported list.
3. Output dir contents: `bundle.manifest.json` (+ empty-placeholder `bundle.hyperion` until `helios-pkg`), shard links/copies (bare names), `tokenizer.json`, `tokenizer_config.json`, `BUILD.json` written last.
4. Stable exit codes: 0 ok; 10 unsupported architecture; 11 source missing/incomplete; 12 unsupported feature (rope_scaling, sliding window per CLAUDE.md, trust_remote_code-only archs); 13 no disk space; 14 manifest schema validation failed; 15 hash mismatch vs `--known-sha256`; 1 other; 2 usage.
5. `--progress-json`: stdout JSON lines `{"event":"stage","name","done","total"}`, final `{"event":"done","manifest"}` or `{"event":"error","code","message"}`; all human text to stderr.
6. Signing (CD3): `--sign-key` optional; without it manifest is unsigned and `BUILD.json` records `signed:false`. Real signing = old phase 06; this phase only passes the flag through when that module exists (no stub signer).
7. Hash reuse (CD4): files named in `--known-sha256` skip hashing but are size-checked; others hashed.
8. `--version-json` prints `{"ignition_version":"X"}` and exits 0 (phase 07 compares it with the Rust-embedded `EXPECTED_IGNITION_VERSION`; mismatch = build error).
Non-functional: python files <200 lines; ruff/format per repo; Python stdlib+click+jsonschema only in the build path.

## Architecture
Rust(07) -> `python -I -m hyperion_ignition.build_cli ... --progress-json` (cwd = temp dir, env offline) -> stdout events -> Rust log + health; exit code -> typed error -> message with remedy. Same pipeline as `cli.py` (`build_compiler` -> `build_recipe` -> `compile` -> `validate_manifest`), refactored into a shared `build_core.py` (<200 lines) used by both CLIs (DRY).

## Related Code Files
Create: `ignition/src/hyperion_ignition/{build_cli.py, build_core.py, arch_map.py, exit_codes.py, progress.py, linking.py}`; `ignition/tests/{test_build_cli.py, test_arch_map.py, test_exit_codes.py, test_linking.py}`; `specification/fixtures/build/exit-codes.json` (read by Rust 07 tests; cross-checked like format table), `docs/decisions/0013-auto-build-and-signing-policy.md` (created here with the subprocess-contract section; 07 appends signing and Building sections).
Modify: `ignition/pyproject.toml` (+script `hyperion-build`; hand-off with old 04/06 owners), `cli.py` (use `build_core`, behaviour unchanged). Delete: none.

## Implementation Steps
1. Resolve phase-01 local-code answers (assets copying, imports). 2. Extract `build_core.py`; existing cli tests stay green. 3. `arch_map.py` + test that every registered compiler is reachable and every mapped id is registered. 4. `exit_codes.py` enum + exception->code mapping; exit-code JSON for Rust. 5. `progress.py` JSON lines with throttling. 6. `linking.py` fallback chain with real tmp-dir tests (symlink may skip on Windows without privilege: tagged). 7. `build_cli.py`; out-dir built under `.tmp` name by caller; BUILD.json last. 8. Tests with real tiny fixtures (`testdata/models/<id>` via registry, no per-model files in engine; missing -> visible skip, `HYPERION_REQUIRE_MODEL_FIXTURES=1` fails). 9. Golden manifest equality vs `cli.py` output for same source (parity).

## Todo List
- [ ] build_core extract  - [ ] arch map  - [ ] exit codes  - [ ] progress json  - [ ] link modes  - [ ] CLI  - [ ] parity test  - [ ] CD3/CD4 answers wired

## Success Criteria
- `hyperion-build` on the fixture model yields a manifest identical to `hyperion-compile` for the same inputs; passes `validate_manifest`. `[RUN]`
- Each exit code 10-15 reproducible by a test; stdout contains only valid JSON lines with `--progress-json`. `[RUN]`
- Unknown architecture message lists supported ids. `[RUN]`
- `python -I -c "import hyperion_ignition.build_cli"` works with torch absent. `[RUN]`

## Test Strategy & Quality Gate
- G/W/T: G = machine-driveable, deterministic Ignition build; W = parity manifest, exit codes 10-15, JSON-lines only on stdout, torch-free import; T = `test_build_cli`, `test_arch_map`, `test_exit_codes`, `test_linking`, e2e `build_cli_subprocess`. Step 0: `pip install -e "ignition[test]" diff-cover`.
- Spec: Input source dir+flags. Output dir+exit code+events. Edge: sharded index, dotted dir names, rope_scaling present, existing out-dir. Error: missing tokenizer, unsupported arch, bad sha, disk full (simulated by tiny tmpfs only where available: tagged).
- Pyramid: 70% unit, 25% integration (full build of fixture), e2e `build_cli_subprocess`: subprocess run from a different cwd with `-I`, parse JSON lines, check dir contents.
- Coverage: `CMD-PY-COV` (`--cov=hyperion_ignition`, phase dir `compat-06`) + `CMD-DIFF` Python form (plan.md); diff >=90% line / >=75% branch.
- Evidence: `pytest ignition/tests/test_build_cli.py ignition/tests/test_arch_map.py ignition/tests/test_exit_codes.py ignition/tests/test_linking.py`, then the CMD-PY-COV run.

## ADR Rationale
Written here as section 1 of ADR 0013. Chosen: subprocess with stable exit codes (shared `exit-codes.json`), JSON-lines progress on stdout / human text on stderr, out-dir contract with `BUILD.json` last, `--version-json` handshake, arch map in Python (rule 3 compatible). Rejected: Rust maps architectures (model knowledge belongs to Ignition architectures files); embedding Python (breaks no-Python runtime); parsing human stderr (brittle). Consequence: exit codes and JSON fields are a versioned contract that 07 and Python tests both read.

## Risk Assessment
Old phase 04 also edits Ignition cli/pyproject: sequence, not parallel (hand-off). Hashing 15 GB slow: progress events + CD4. Symlink privilege on Windows.

## Security Considerations
Source dir untrusted: safetensors header parse only (no pickle), size sanity caps, no `trust_remote_code`, no network, out paths confined to out-dir (reject `..`/absolute payload names). Signing key path read once, never logged.

## Next Steps
Feeds 07 (subprocess driver). Old phase 06 plugs signing; phase 10 includes build timing on pod.
