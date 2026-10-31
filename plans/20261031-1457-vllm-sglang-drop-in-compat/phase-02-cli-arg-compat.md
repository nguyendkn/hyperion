# Phase 02: CLI arg-compat layer

## Context Links
- `gap-analysis.md` (flag families); phase-01 pins doc + fixtures; `helios/crates/helios-cli/src/main.rs` (Cli: --model --port --bind --max-seq-len, `resolve_max_seq_len`)
- ADR 0002 (KV sizing after free-memory measurement), 0005 (TP chosen at load), 0010 (max seq len is runtime-only)

## Overview
Priority P1. Status pending. Wave 1. New module `helios-cli/src/compat/` turns any vLLM or SGLang launch line into one `RuntimeConfig` (+ diagnostics), through one flag table that also generates `docs/vllm-sglang-compat-matrix.md`. Also makes `helios-cli` a lib + bin crate (`src/lib.rs`, `[lib]`) so integration tests reach `compat`, `model_source` (05) and `autobuild` (07), and owns `helios-cli/Cargo.toml` in W1. `lib.rs` declares ALL modules (compat, serve_flow, model_source, autobuild); 02 creates empty stub `mod.rs` for model_source (05 replaces), autobuild (07 replaces) and `compat/sglang_shim.rs` (09 replaces), so no other phase edits lib.rs.

## Key Insights
- clap derive cannot express ~150 flags with per-flag status, aliases, underscore/hyphen variants, k/M/G suffixes, `--x/--no-x` pairs, dotted JSON args. Table-driven parser instead; clap stays for legacy `--model <pkg>`.
- One table = three outputs: parser, `--help`, matrix doc. Drift test makes docs un-driftable (same pattern as `schema:check`).
- vLLM `-tp`-style single-dash long aliases must parse. Unknown flag -> exit 2 + closest-name hint (argparse parity, verified in 01).

## Requirements
Functional:
1. Subcommands: `hyperion serve [model] [flags]` (vLLM), `hyperion launch_server [flags]` (SGLang style, `--model-path`), legacy `hyperion --model <pkg>` still works (no regression), `hyperion compat-matrix` (prints/generates doc), `hyperion build` / `hyperion cache` routed by main.rs dispatch to stub handlers `run_build` / `run_cache` in `cli_cmds.rs` (typed PlannedFeature, exit 2, naming phase 07 / 05; 05 fills `run_cache`, 07 fills `run_build`).
2. Table entry: `name, aliases[], origin{vllm,sglang,both}, arity{bool,value,list,json,size}, maps_to (RuntimeConfig field | none), status{Supported,Planned(phase),Warn(reason),Reject(reason,phase)}, value_rules (accepted values -> status per value, e.g. dtype bfloat16 S / float16 R)`.
3. Value-level status (not only flag-level): `--tensor-parallel-size 1` S, `2` Planned(old13)->Reject typed error; `--dtype auto|bfloat16` S; `--quantization fp8` R; `--kv-cache-dtype auto` S, `fp8` R; `--enforce-eager` W (cannot change output); `--trust-remote-code` W; `--gpu-memory-utilization`/`--mem-fraction-static` planned(old 12) = rejected at the 02 report stage (exit 2, before any download) naming the phase; prefix-cache toggle: enable W (matches behaviour), disable R, core untouched.
4. `--config file.yaml` (keys = flag names, precedence per 01), env vars: `VLLM_API_KEY`->api-key, `VLLM_LOGGING_LEVEL`, `HF_TOKEN`, `HF_HOME`, `HF_ENDPOINT`, `HF_HUB_OFFLINE`, `CUDA_VISIBLE_DEVICES` (device probe respects), `SGLANG_*` and other `VLLM_*`: if set and not in table -> warn naming var. Precedence: CLI > env > config > default.
5. Style detection by subcommand/first token decides defaults (CD9): serve -> port 8000, launch_server -> 30000, legacy -> 8080. Host default stays `127.0.0.1` until api-key auth ships and owner decides (vLLM defaults differ: phase-01 pins-doc row "default host").
6. Positional model wins over `--model` per 01; `--model-path` alias.
7. Report: before any download, print a one-screen summary: accepted / warned / rejected flags; exit 2 on any reject OR planned flag (planned flags are rejected here, before any download, with `PlannedFeature{flag,phase}`), listing ALL at once.
7b. `--dry-run`: parse + report + exit code, JSON diagnostics (`--dry-run-json`), no download/bind (used by phase 10 swap test).
8. `--gpu-memory-utilization`/`--mem-fraction-static` parse into `kv_fraction` but status is planned(old 12) (CD10); `--max-model-len`/`--context-length` -> `max_seq_len` (k/M suffix, `auto`, -1). Hyperion-only flag `--target-arch <sm_xx|stub>` (origin `hyperion`) overrides the device probe for builds (07; Windows has no driver).
Non-functional: every file <200 lines; no new heavy deps. CD12 list declared HERE once (reason per dep in ADR 0011): YAML crate `serde_norway` or `serde_yml` (serde_yaml archived; choose after checking maintenance), `sha2` (05 cache key/hash), `fs4` (05 lock/disk), `libc` (unix) + `windows-sys` (job object; 07 process-group kill). The HTTP client (CD1) is the only dep 05 appends, after 02 merges. No panics on any argv.

## Architecture
```
argv/env/config -> normalize (hyphen/underscore, =, suffixes, -no-) -> lookup table -> Diagnostic list + RuntimeConfig
                                                                      -> Reject? exit 2 : continue to model_source (05) / serve
flags_*.rs (static tables by family) --> docs generator (tests/compat_matrix.rs rewrites docs/..matrix.md; CI git diff --exit-code)
```
`RuntimeConfig` (the ONE complete struct; later phases only read it) = {model_ref, bundle (Option path, `--bundle`), served_names[], host, port, api_keys[], max_seq_len, tp_size, kv_fraction, max_num_seqs, prefix_cache, dtype, quant, kv_dtype, seed, revision, download_dir, cache_dir (`--cache-dir`/`HYPERION_CACHE`), hf_token (masked type), offline, link_mode (symlink|hardlink|copy), chat_template (Option file or inline), target_arch (Option; probe result if None), build_timeout, exit_on_build_failure (default true), enable_metrics, enable_sglang_api, log_level, style (`LaunchStyle` vllm|sglang|legacy), dry_run, extra_warnings}. Reader map (SSOT, const table in `compat/readers.rs`: field -> reader phase): model_ref, revision, download_dir, hf_token: 05; cache_dir, offline: 05 + 07; chat_template: 05 (cache key) + 06/07 (`--chat-template-file`); dtype, quant: 05 cache key only; link_mode: 07 (passed to Ignition); bundle, build_timeout, exit_on_build_failure, target_arch: 07; max_num_seqs: 07 (flow passes it to the scheduler queue/slot config); host, port, log_level, tp_size (validated ==1), prefix_cache (enable = identity): 02 `serve_flow.rs`; kv_dtype (`auto` only): 02 value rules; max_seq_len: 02 (`resolve_max_seq_len`) + 03; kv_fraction: none (planned(old 12), never Supported); api_keys, served_names, enable_metrics, style, enable_sglang_api, seed: 03 (`ApiConfig` mapping in `serve_flow.rs`); 04 via ApiConfig only; 09: style, enable_sglang_api. Hardware-agnostic: helios-core untouched (rule 1); `Arc<Metrics>` is built in core by 04 (03 leaves `AppState.metrics = None`), so no CLI wiring is owned elsewhere.

## Related Code Files
Create: `helios-cli/src/compat/{mod.rs, readers.rs, sglang_shim.rs (empty stub; 09 fills), table.rs, flags_model.rs, flags_parallel.rs, flags_memory.rs, flags_sched.rs, flags_api.rs, flags_misc.rs, parse.rs, values.rs, config_file.rs, env.rs, style.rs, report.rs, matrix_doc.rs}`; `helios-cli/src/{lib.rs, serve_flow.rs, cli_cmds.rs (stub `run_build`/`run_cache`), model_source/mod.rs (empty stub), autobuild/mod.rs (empty stub)}`, `helios-cli/src/compat/sections.rs` (generator globs `docs/compat-sections/*.json`, rendered in filename order), `helios-cli/tests/{compat_parse.rs, compat_golden.rs, compat_matrix.rs, compat_e2e.rs}`; `helios-cli/tests/compat-cmdlines/{vllm,sglang}/*.txt` (real command lines from docs/README/recipes, one per file, with expected `.json` outcome); `docs/vllm-sglang-compat-matrix.md` (generated), `docs/decisions/0011-compat-flag-policy.md`.
Modify: `helios-cli/src/main.rs` (today 493 lines, already over the 200 rule: move the existing serve logic into `serve_flow.rs` as a pure move first, leave main.rs as dispatch <=30 lines), `helios-cli/Cargo.toml` (+`[lib]`, +bin `hyperion`, CD12 deps above), `Taskfile.yml` (`compat:check`).
Delete: none.

## Implementation Steps
1. Write ADR 0011: statuses, never-silently-wrong rule, value-level statuses, exit codes (2 usage/reject), CD9/CD10/CD11/CD12 options + recs; owner answers recorded. 2. `table.rs` types + `parse.rs` normalizer with unit tests (suffixes, `--no-x`, dotted JSON, repeated list flags like `--served-model-name a b`). 3. Fill flag tables from phase-01 verified lists, family by family (test per family: every research-listed flag present). 4. `values.rs` value rules. 5. `config_file.rs`, `env.rs`, precedence merge. 6. `style.rs` + defaults. 7. `report.rs` summary + exit codes (reject and planned both exit 2 before any download). 8. `matrix_doc.rs` generator + drift test. 9. Golden tests: parse the real command lines, including the owner's example (`serve Qwen/Qwen2.5-7B --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.9 --port 8000 --dtype bfloat16 --served-model-name X --api-key K --max-num-seqs N --enable-prefix-caching --quantization fp8 --kv-cache-dtype fp8` -> expected: rejects = tp 2, quantization fp8, kv-cache-dtype fp8 (R) and gpu-memory-utilization 0.9 (planned(old 12)), each with reason+phase; and the same line with those four dropped is fully accepted) and SGLang `--model-path --tp --mem-fraction-static --context-length --host --port`. 10. Wire dispatch in main.rs (serve, launch_server, legacy, compat-matrix, build, cache -> `cli_cmds.rs` stubs); create lib.rs with all module declarations and the stub files; keep legacy tests green. 11. Property/fuzz test: random argv never panics.

## Todo List
- [ ] ADR 0011 + owner answers  - [ ] table+parser  - [ ] families filled  - [ ] config/env  - [ ] matrix generator + drift CI  - [ ] golden lines  - [ ] fuzz  - [ ] main.rs dispatch

## Success Criteria
- Every flag listed in pins doc is in the table with a status (test enumerates the pins doc list; missing = fail). `[RUN]`
- Golden lines produce expected RuntimeConfig/diagnostics. Unknown flag -> exit 2 + hint. `[RUN]`
- `cargo test -p helios-cli` + `git diff --exit-code docs/vllm-sglang-compat-matrix.md` after regenerate pass. `[RUN]`
- Legacy `helios-cli --model pkg` behaviour unchanged. `[RUN]`
- Every Supported flag's `maps_to` field has a named reader phase in the reader map (test over `readers.rs`: a Supported field missing from the map fails; the reader phases' own tests prove the read). `[RUN]`

## Test Strategy & Quality Gate
- G/W/T: G = every launch line classified, never silently wrong; W = golden lines, drift test, fuzz, legacy unchanged; T = `compat_parse`, `compat_golden`, `compat_matrix`, `compat_e2e`. Step 0: `pip install -e "ignition[test]" diff-cover` (diff-cover serves every CMD-DIFF).
- Spec: Input argv/env/YAML. Output RuntimeConfig + diagnostics + exit code. Edge: both `--model` and positional; `--max-model-len 32k`; `--no-enable-prefix-caching`; underscore flags; env+CLI conflict. Error: unknown flag, bad value, reject flag (all rejects listed together).
- Pyramid: ~70% unit (parser, values, precedence), ~25% integration (golden cmdlines, matrix drift, fuzz), 1 e2e `serve_owner_example` (`compat_e2e.rs`): spawn built binary with the owner example, assert stderr+exit code 2, then with fixed line assert it reaches the model-resolution stage (stub backend).
- Coverage: diff line >=90%, branch >=75% (`CMD-RUST-COV` scoped `-p helios-cli`, `CMD-DIFF`, both in plan.md); no assertion-free tests.
- Evidence: `cargo test -p helios-cli --test compat_parse --test compat_golden --test compat_matrix --test compat_e2e`; `CMD-RUST-COV -p helios-cli` (phase dir `compat-02`); `CMD-DIFF` (plan.md templates, origin/main precondition).

## ADR Rationale
ADR 0011. Chosen: single static table, value-level statuses, reject-all-before-work. Rejected: clap derive with hidden ignored flags (statuses invisible, docs drift); accept-everything-and-warn (violates never-silently-wrong: `--quantization fp8` run as bf16 changes numerics/memory). Consequence: table must track upstream releases; phase 10 re-check task.

## Risk Assessment
Upstream flag drift (mitigate: pins + dated matrix). Too many rejects makes "drop-in" feel false: matrix doc states the gap honestly; planned phases listed. Value semantics wrong (e.g. mem-fraction) -> documented difference + CD10. Windows `CUDA_VISIBLE_DEVICES` handling untestable w/o GPU `[NEEDS-GPU]`.

## Security Considerations
`--api-key` value must not be logged or echoed in the report (mask); `HF_TOKEN` likewise; `--config` path read-only, size-capped; YAML parsed with no anchors-bomb (size/depth cap); argv from env expansion none.

## Next Steps
Feeds 03 (ApiConfig mapping), 05 (model_ref, hf env), 07. 09 fills the `compat/sglang_shim.rs` stub. ONE generator (this phase); 03/04/09 register sections only by adding `docs/compat-sections/<NN>-<name>.json`; CLI status flips are edits to the flag tables here.
