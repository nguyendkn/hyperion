---
title: "vLLM / SGLang drop-in launch compatibility"
description: "Swap `vllm serve` / `sglang.launch_server` for `hyperion` with every arg unchanged: each flag supported, planned, warned or rejected with a clear error; auto-build + cache of the signed bundle; OpenAI/vLLM/SGLang HTTP parity."
status: pending
priority: P1
effort: ~40-44d (Windows layers 01-07 ~25d; GPU-gated 08, 10 ~12-16d; 09 optional ~3d)
lane: high-risk
branch: main
tags: [api, cli, compat, vllm, sglang, auto-build, cache, metrics, feature, critical]
created: 2026-10-31
---

# Drop-in compat: `hyperion serve <model> <vllm/sglang args>`

Goal: user replaces `vllm serve ...` or `python -m sglang.launch_server ...` by `hyperion ...`, all other args unchanged. Contract: **never silently wrong**. Every flag is exactly one of `supported` | `planned(phase)` | `accept-and-warn` (cannot change output, e.g. `--disable-log-requests`, `--enforce-eager`, `--trust-remote-code true`; research-01 says these must not error; a model that truly needs remote code fails typed at build, Ignition exit 12) | `reject` (error names flag, reason, fixing phase; exit 2 before any download/build). `planned(phase)` = parsed, then rejected like `reject`: exit 2 at the 02 report stage before any download, typed `PlannedFeature{flag,phase}` naming the phase (HTTP request fields: 400); never a no-op. Prefix-cache toggle: accept-and-warn only when the value equals current behaviour (enabled), else reject; core untouched.

Owner decision 2026-10-31: model loading = AUTO-BUILD + CACHE. `hyperion serve <hf-id|path>`: first run calls Ignition (optional Python subprocess, build time only) -> signed bundle in a cache dir; later runs load directly. Helios stays Python-free; rules 1-7 of project CLAUDE.md unchanged.

Rules: ideas only from vLLM/SGLang docs (memory: upstream-ideas-only); no mocks on main; `[U]`/`[MEM]` tags in research are UNVERIFIED until phase 01 closes them. Evidence tags (SSOT): `[RUN]` executed on this Windows host; `[RUN on stub]` against the stub backend (named test double); `[RUN on local HTTP server]` against a real local protocol peer; `[TYPE-CHECK]` compiled only; `[NEEDS-GPU]` / `[NEEDS-NETWORK]` / `[NEEDS-TOOL]` not runnable here; `blocked-by sm90 KV` = needs old-plan paged-KV on sm90. Related plan: `plans/20261028-2209-hyperion-upstream-distill-multimodel-multiinfra/` (old 06 signing, old 07 serving, old 13 multi-GPU) - referenced by id, not duplicated.

## Shared command templates (SSOT; phases cite by name, add only scope args)
Step 0 once per host: `pip install -e "ignition[test]" diff-cover` (pytest, pytest-cov, diff-cover not installed here). Precondition for every CMD-DIFF: `git rev-parse origin/main` succeeds, else report "diff gate UNVERIFIED".
- `CMD-RUST-COV`: `cargo +nightly llvm-cov -p <crate> [--features <f>] --branch --lcov --output-path target/coverage/<phase>/lcov.info` (never combine `--workspace` with `-p`); branch table `cargo +nightly llvm-cov report --branch`; if `grep -c '^BRDA' <lcov>` is 0 state "branch gate UNVERIFIED on this host".
- `CMD-PY-COV`: `pytest <tests> --cov=<pkg|scripts/compat> --cov-branch --cov-report=xml:target/coverage/<phase>/coverage-<phase>.xml`; branch % from `coverage report`.
- `CMD-DIFF`: `diff-cover target/coverage/<phase>/{lcov.info|coverage-<phase>.xml} --compare-branch=origin/main --fail-under=90` (lcov = lines only; branch >=75% read from the coverage tables).
Each phase Test Strategy carries a `G/W/T` line mapping Goal (G), Acceptance (W), and Test (T) ids.

## Phases
| # | Phase | Depends | Wave | Status | Effort |
|---|-------|---------|------|--------|--------|
| 01 | [Verify [U]/[MEM] claims, pin vLLM+SGLang versions, record real fixtures](phase-01-verify-pin-fixtures.md) | - | W0 | pending | 2d |
| 02 | [CLI arg-compat layer: flag table, RuntimeConfig, lib+serve_flow, matrix generator](phase-02-cli-arg-compat.md) | 01 | W1 | pending | 4d |
| 05 | [Model resolver, HF downloader, cache key/layout/lock/gc](phase-05-model-resolve-download-cache.md) | 01,02 | W1b | pending | 4d |
| 06 | [Ignition non-interactive build contract](phase-06-ignition-build-contract.md) | 01 | W1 | pending | 3d |
| 03 | [HTTP API parity: schemas, SamplingParams/wire, SamplingCaps, auth, errors, models, tokenize, /version](phase-03-http-api-parity.md) | 02 | W2 | pending | 6d |
| 04 | [Metrics: vllm:* / sglang:* Prometheus](phase-04-metrics.md) | 03 | W3 | pending | 2d |
| 07 | [Auto-build orchestration + Building readiness](phase-07-autobuild-orchestration.md) | 02,03,05,06 | W3 | pending | 4d |
| 08 | [Sampling features needing backend/StepPlan (OWNER-GATED, GPU)](phase-08-sampling-backend-gated.md) | 03; old 07 + sm90 KV | W4 | gated | 8-12d |
| 09 | [SGLang native endpoints + launch_server shim (optional)](phase-09-sglang-native-optional.md) | 03,04 | W4 | optional | 3d |
| 10 | [Verification on H100 + swap test + gap closure](phase-10-verification-swap-test.md) | all | W5 | pending | 4d |

Honest gap table (flag family vs engine today): [gap-analysis.md](gap-analysis.md). Research: `research/researcher-01..03-*.md`.

## Parallel Execution Matrix
| Unit | Owns | Depends On | Wave | Verify |
|---|---|---|---|---|
| phase-01 | docs/vllm-sglang-pins.md, gap-analysis.md, helios-cli/tests/compat-fixtures/**, scripts/compat/{capture-fixtures.sh,check-fixtures.py,tests/test_check_fixtures.py} | - | 0 | pytest scripts/compat |
| phase-02 | helios-cli/{Cargo.toml (sole W1 owner), src/lib.rs (declares ALL CLI modules: compat, serve_flow, cli_cmds, model_source, autobuild), src/main.rs (dispatch: serve, launch_server, legacy, compat-matrix, build, cache), src/cli_cmds.rs (stub `run_build`/`run_cache`), src/serve_flow.rs, src/compat/** (incl. empty stub sglang_shim.rs), src/model_source/mod.rs + src/autobuild/mod.rs (empty stubs)}, tests/compat_*.rs, tests/compat-cmdlines/**, Taskfile.yml, docs/vllm-sglang-compat-matrix.md (generator), docs/decisions/0011 | 01 | 1 | cargo test -p helios-cli --test compat_parse |
| phase-05 | helios-cli/src/model_source/** (replaces 02's mod.rs stub; no lib.rs edit), src/cli_cmds.rs (ONLY `run_cache` body), tests/{model_resolve,hf_download_local_server,cache_key,cache_lock}.rs, docs/decisions/0012; Cargo.toml: ONE HTTP-client dep line appended after 02 merged (CD1) | 01, 02 (merged) | 1b | cargo test -p helios-cli --test model_resolve |
| phase-06 | ignition/** (build_cli.py etc., pyproject.toml), specification/fixtures/build/exit-codes.json, docs/decisions/0013 (creates; subprocess contract) | 01 | 1 | pytest ignition/tests |
| phase-03 | helios-core/src/{api/** (incl. AppState with `metrics: Option<Arc<Metrics>>` = None),sampling_wire.rs (SamplingCaps + from_capabilities hook),scheduler.rs,scheduler_plans.rs,step_scheduler.rs (ignore_eos/seed host-side only),error.rs,lib.rs (declares ALL core modules: metrics, api_sglang, api_sglang_wire),metrics/mod.rs (stub `pub struct Metrics;`),api_sglang.rs + api_sglang_wire.rs (empty stubs)}, helios-core/tests/api_*.rs, helios-cli/src/serve_flow.rs (ApiConfig incl. version, from_capabilities call), scripts/compat/{openai_client_smoke.py,tests/test_openai_client_smoke.py}, docs/compat-sections/03-api-fields.json, regenerated docs/vllm-sglang-compat-matrix.md, gap-analysis.md, docs/unverified-ledger.md, ADR 0011 addendum | 02 | 2 | cargo test -p helios-core --test api_chat_schema |
| phase-04 | helios-core/src/metrics/** (replaces 03's stub), api/mod.rs (`AppState::new` builds Metrics, after 03), step_scheduler.rs + scheduler.rs (hook calls), api/routes.rs (mount, after 03), tests/{metrics_encode,metrics_hooks (holds e2e `metrics_scrape_during_smoke`),metrics_names_fixture,metrics_bench}.rs, docs/compat-sections/04-metrics.json, regenerated docs/vllm-sglang-compat-matrix.md, ADR 0011 addendum; no lib.rs edit (03 declared `metrics`) | 03 | 3 | cargo test -p helios-core --test metrics_hooks |
| phase-07 | helios-cli/src/{autobuild/** (replaces 02's mod.rs stub; no lib.rs edit),serve_flow.rs (after 03), cli_cmds.rs (ONLY `run_build` body)}, tests/{autobuild_*,building_health}.rs, docs/decisions/0013 (appends), docs/serving-in-kubernetes.md, docs/unverified-ledger.md (append section) | 02,03,05,06 | 3 | cargo test -p helios-cli --test autobuild_subprocess |
| phase-08 | helios-hal/src/{sampling.rs,step.rs,traits.rs}, backends/**, kernels/*.cu, helios-core/src/{step_scheduler.rs,scheduler_plans.rs}, helios-core/src/sampling_wire.rs (ONLY the body of `SamplingCaps::from_capabilities`), docs/compat-sections/03-api-fields.json (flips rows) + regenerated docs/vllm-sglang-compat-matrix.md (08 regenerates first), helios-hal/tests/sampling_features_hw.rs, helios-core/tests/sampling_stub.rs, ADR 0014 | 03 + owner OK (CD7) | 4 | stub tests + GPU |
| phase-09 | helios-core/src/{api_sglang.rs,api_sglang_wire.rs} (replace 03's stubs), api/routes.rs (mount, after 03/04), tests/api_sglang.rs, helios-cli/src/compat/sglang_shim.rs (replaces 02's stub; no compat/mod.rs or lib.rs edit), docs/compat-sections/09-sglang-routes.json, regenerated docs/vllm-sglang-compat-matrix.md, scripts/compat/sglang-shim/ (only if CD14=B), ADR 0011 addendum | 03,04 | 4 | cargo test -p helios-core --test api_sglang |
| phase-10 | scripts/compat/{swap_test.py,battery.py,schema_diff.py,run_clients.sh,prom_check.py,schema-allow.json}, scripts/compat/tests/test_schema_diff.py, scripts/compat/launch-lines/**, docs/{h100-test-runbook,performance-log,unverified-ledger,compat-verification-report}.md, gap-analysis.md; consumes (never edits) 03's openai_client_smoke.py and 01's check-fixtures.py | all | 5 | pod |

Ordering notes: W4 matrix doc: 08 regenerates and commits docs/vllm-sglang-compat-matrix.md first; 09 regenerates after 08 merges (or after 08 is skipped/gated), never concurrently. helios-cli/Cargo.toml: 02 owns it in W1 (`serde_norway`/`serde_yml`, `sha2`, `fs4`, `libc`+`windows-sys` for 07's process-group kill, `[lib]`); 05 appends the HTTP client only after 02 merged. main.rs: 02 only (dispatch <=30 lines, routes serve, launch_server, legacy, compat-matrix, build, cache; `build`/`cache` handlers live in `cli_cmds.rs`: 02 stubs them, 05 fills `run_cache`, 07 fills `run_build`); all growth lives in `serve_flow.rs` (02 creates, 03 W2 adds ApiConfig mapping, 07 W3 adds autobuild flow). Core `AppState` (api/mod.rs): 03 adds `metrics: Option<Arc<Metrics>>` = None and a stub type `Metrics` (metrics/mod.rs); 04 owns `Metrics` and the `AppState::new` line that builds it from `ApiConfig.enable_metrics/style`, so 04 never touches helios-cli.
Module declarations: `helios-cli/src/lib.rs` is created by 02 declaring every CLI module with empty stub files (cli_cmds, model_source, autobuild, compat/sglang_shim); `helios-core/src/lib.rs` is owned by 03 declaring metrics, api_sglang, api_sglang_wire with stubs; 05, 07, 04, 09 replace stub bodies (05/07 also cli_cmds.rs handler bodies) and never edit lib.rs/mod declarations. Wave 1b: 05 starts after 02 merges (needs `[lib]`, `RuntimeConfig`, Cargo.toml). `scripts/compat/openai_client_smoke.py` is created by 03 (W2); 04, 10 consume it. `gap-analysis.md` row edits are sequential by wave: 01, 03, 10. Cross-crate fixtures are read as `env!("CARGO_MANIFEST_DIR")/../helios-cli/tests/compat-fixtures`. ADR 0011 addenda (03,04,09) and ledger appends are sequential by wave. Matrix doc = ONE generator (02, `compat/matrix_doc.rs`); other phases only add a data file `docs/compat-sections/<NN>-<name>.json` (03 api fields, 04 metrics, 09 sglang routes); 08 edits 03's file (W4, after 03). `SamplingParams`/`sampling_wire.rs` extension (seed, top_k, min_p, penalties, stop, ignore_eos, logprobs, n, min_tokens, logit_bias, stop_token_ids; additive Option fields) and the `SamplingCaps::from_capabilities` hook (default all-false, called in `serve_flow.rs`) = 03 only; 08 owns backend/StepPlan behaviour of top_k/min_p/penalties/logprobs and only the body of `from_capabilities` (after CD7).

## Owner decisions (flagged, NOT decided here; recommendation in each phase)
CD1 downloader dep (reqwest by hand vs `hf-hub` vs Python `snapshot_download`) - ph05. CD2 cache write layout (HF-compatible vs own) - ph05. CD3 signing of auto-built bundles (unsigned+warn / per-machine TOFU key / env key) - ph07 (old 06 trust store). CD4 trust HF LFS sha256 vs rehash - decided ph05 (ADR 0012), implemented ph06 (`--known-sha256`). CD5 `Building` state: CLI pre-engine router (A, rec; Building lives only there) vs new `EngineStatus` (touches state.rs) - ph07. CD6 helios-core touch scope (api, metrics) - ph03/04. CD7 StepPlan/ComputeBackend change for min_p, top_k, penalties, logprobs - ph08 (CLAUDE.md: needs owner approval); until approved ph03 uses a core-side `SamplingCaps` (gated fields false), no trait change. CD8 policy for unsupported sampling fields: reject 400 (rec) vs warn - ph03. CD9 default host/port per invocation style (vllm 0.0.0.0:8000, sglang 127.0.0.1:30000, today 127.0.0.1:8080 loopback because no auth until api-key) - ph02. CD10 `--gpu-memory-utilization` / `--mem-fraction-static` semantics vs ADR 0002 KV sizing - ph02; initial status planned(old 12). CD11 binary name `hyperion` next to `helios-cli`; `serve` subcommand coexists with legacy `--model <pkg>` - ph02. CD12 new deps list - declared once in ph02 (YAML, sha2, fs4, libc/windows-sys); HTTP client per CD1 appended by ph05 after 02 merges; reasons in ADR 0011/0012/0013. CD13 pinned vLLM/SGLang versions - ph01. CD14 `python -m sglang.launch_server` pip shim package - ph09. CD15 (fix-pass design choices, reviewable): Ignition version = Rust-embedded expected-version constant checked against the Python package at build time, cache-hit path Python-free (ph05/07); `/version` single-owned by ph03; `/tokenize` cap 1 MiB text / 262144 tokens, body cap 32 MiB (ph03); error JSON per style/route (vLLM nested, SGLang per phase-01 fixture); metric labels per style (vLLM `model_name,engine`; SGLang `model_name,engine_type`); `target_arch` = device probe or `--target-arch` override (ph07).

## Unresolved Questions
1. Which vLLM and SGLang versions are the target? (CD13; ph01 proposes, owner confirms)
2. Is a Python+Ignition install acceptable in the serving image, or is a prebuilt `--bundle` the production path (auto-build = dev convenience)?
3. Is TP>1 expected to work at first release of compat? Today: design-only (old 13), flag is `planned(13)` and rejected with a typed error.
4. fp8 weights/KV (`--quantization fp8`, `--kv-cache-dtype fp8`): no quantizer exists (tensor_format = bf16 written). Reject until a quantizer phase exists; is that scope for this plan? Assumed no.

## Automatic Validation
**Rounds run:** 5 (deep r1 + 4 delta). **Outcome:** ESCALATED at cap (r5: 1 BLOCKER) -> user chose "main agent fixes once"; W4 08/09 matrix-doc ordering note + 4 MINOR fixed manually, NOT re-validated (no round 6).
| R | Type | Struct | Consist | Fidel | Test | BLOCKERs | Clean |
|---|---|---|---|---|---|---|---|
| 1 | deep | 1B/5m | 5B/6m | 0B/3 | 0B/6m | 6 | no |
| 2 | delta | 2B/4m | 4B/7m | carried | carried | 6 | no |
| 3 | delta | 1B/3m | 0B/5m | carried | carried | 1 | no |
| 4 | delta | PASS | carried | carried | carried | 0 | yes |
| 5 | delta | 1B/5m | carried | carried | carried | 1 | no |
Artifacts: `reports/validation-round-*.md`.
