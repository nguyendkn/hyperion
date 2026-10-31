# Phase 01: Verify unverified claims, pin versions, record real fixtures

## Context Links
- `research/researcher-01-vllm-surface.md`, `researcher-02-sglang-surface.md`, `researcher-03-hf-fetch-build-cache.md` (tags [U]/[MEM] = unverified)
- `gap-analysis.md`; project CLAUDE.md; memory: upstream ideas only (read docs/behaviour, never copy code or text)

## Overview
Priority P1. Status pending. Wave 0. Freeze the facts every later phase builds on: pinned vLLM/SGLang releases, verified flag lists, endpoint/metric/error shapes, HF Hub behaviour. Output = `docs/vllm-sglang-pins.md` + recorded real-server fixtures under `helios/crates/helios-cli/tests/compat-fixtures/` (own recorded files, not upstream source).

## Key Insights
- Research was budget-limited: vLLM endpoints page unread; SGLang sampling/openai/bench pages unread; HF [MEM] items never fetched. Nothing may enter the flag table as `supported` before its row is verified.
- Ground truth is the tool itself: `vllm serve --help=all`, `python -m sglang.launch_server --help`, a running server's `/openapi.json`, `/metrics`, `/v1/models`, error bodies. Read-only observation, ideas only.
- vLLM runs CPU-only builds? [U]: if not, record on the H100 pod (phase 10 runbook) and keep this phase NEEDS-GPU for fixture capture only; help text and docs need no GPU.

## Requirements
Functional: (1) pick and pin target versions (rec: latest stable of each at 2026-10-31 minus none; owner confirms, CD13); (2) close every [U]/[MEM] item below with evidence or mark it "unverifiable, behaviour = X, reason"; (3) record fixtures. Non-functional: every claim cites doc URL+date or captured file; none left tagged [U] in the pins doc.

Verification checklist (each becomes a row `claim | source | result`):
- vLLM: positional `model_tag` vs `--model` precedence; `--config` key spelling (dash/underscore) + precedence; unknown-flag behaviour (exit code, message); `--help` group syntax; `--gpu-memory-utilization` default; `--enable-prefix-caching` default; all flags added after `--max-num-active-seqs`; speculative/compilation flags; default `--host`.
- vLLM HTTP: error JSON shape per invocation style and route (vLLM nested `error{}` on /v1/*; SGLang OpenAI routes and native routes: nested or flat) and 401 body, as fixture files `error-*.json` per style; which paths skip auth; extra-field policy (ignore/warn/reject); `/v1/models` shape; `/tokenize` `/detokenize` bodies; stream chunk order + usage chunk; `kv_cache_usage_perc` vs `gpu_cache_usage_perc` rename version; metric label sets per style (`model_name,engine` vs `model_name,engine_type`); `_total` suffix on prefix_cache_*; lm-eval `echo`/`logprobs` request shape.
- SGLang: `sglang serve` real? flag-compatible?; `--disable-radix-cache`, `--pipeline-parallel-size`, `--max-prefill-tokens`, `--config` YAML key form; `/generate` stream cumulative vs delta + `meta_info`; `/server_info` fields read by `bench_serving`; readiness endpoint used by `popen_launch_server`/`wait_for_server`; startup log line; error shape on native routes.
- HF [MEM]: `X-Repo-Commit`, `api/models/{id}/revision/{rev}` (+`?blobs=true`) fields; `HF_HUB_CACHE` vs `HF_HOME` vs `TRANSFORMERS_CACHE` precedence; xet-backed files vs plain `resolve` + `Range`; Authorization stripped on cross-host redirect by `reqwest`; gated-repo error codes (`X-Error-Code`).
- Local code (not [U], but unread by research): does `compiler/` copy tokenizer/shards or only write the manifest; Ignition's Python imports (torch?); registry ids vs `architectures`; Rust MSRV and `File::lock` availability; whether `StepPlan` slot init already carries top-k/penalties (ADR 0007 text says yes, hal `SamplingConfig` has only temperature/top_p/seed); whether `ignore_eos` and a request `seed` can be honoured host-side (read at plan time: EOS check `step_scheduler.rs:249`, seed = request_id at `scheduler.rs:225`).

## Architecture
No code. Files: `docs/vllm-sglang-pins.md` (versions, URLs, verification table, residual unknowns); fixtures dir: `vllm-<ver>/{help-all.txt,models.json,chat-nonstream.json,chat-stream.sse,completions.json,error-400.json,error-401.json,error-404.json,metrics.txt,tokenize.json}`, same for `sglang-<ver>/` plus `generate-*.json`, `server_info.json`, `get_model_info.json`. Fixtures are the schema-diff baseline for phase 10 and golden inputs for 02/03/04.

## Related Code Files
Create: `docs/vllm-sglang-pins.md`; `helios/crates/helios-cli/tests/compat-fixtures/**`; `scripts/compat/capture-fixtures.sh` (launches pinned server with a tiny model, curls endpoints, saves files; runs on pod), `scripts/compat/check-fixtures.py`, `scripts/compat/tests/test_check_fixtures.py`. Modify: `gap-analysis.md` (refresh rows; edits sequential 01, 03, 10); `plans/.../research/*` none (append corrections to the pins doc, not to research). Delete: none.

## Implementation Steps
1. Propose versions; get owner OK (CD13). 2. Install pinned vLLM and SGLang in throwaway venvs under `%TEMP%\hyperion-scratch\` (python `-I`; never in repo root). 3. Dump `--help` outputs to fixtures; diff against research tables; add missing flags to the pins doc. 4. Fetch the unread doc pages (endpoints, sampling params, openai API, bench) with WebFetch; record. 5. Run the checklist rows that need a live server via `capture-fixtures.sh` (pod or CPU build). 6. Probe HF behaviour with `curl -sI` against a small public repo (e.g. Qwen2.5-0.5B pinned in `testdata/models.json`). 7. Read the local-code items (compiler imports, registry, MSRV, hal sampling/StepPlan) and write answers; also check that Ignition has a hardware recipe for the stub backend's arch id (`triage.rs` id vs `ignition/.../hardware/`): if none, record in the pins doc that phase 07 uses a documented build override for the Windows/stub path. 8. Update `gap-analysis.md` rows whose status changed; publish residual unknowns.

## Todo List
- [ ] versions pinned + owner OK  - [ ] flag lists dumped  - [ ] checklist table 100% resolved  - [ ] fixtures committed  - [ ] gap-analysis.md refreshed

## Success Criteria
- `docs/vllm-sglang-pins.md` has zero `[U]`/`[MEM]` tags; each row has source+date. Doc-page rows and local-code rows `[RUN]`; `--help` dumps and HF probes `[NEEDS-NETWORK]` (pip install, curl); live-server fixture capture `[NEEDS-NETWORK]` + `[NEEDS-GPU]` unless a CPU build runs the tiny model. Rows that cannot be captured here are marked "unverifiable here" with a conservative status, never `supported`.
- `python -I scripts/compat/check-fixtures.py` confirms every fixture file listed in the pins doc exists and parses (JSON/SSE). `[RUN]`.

## Test Strategy & Quality Gate
- G/W/T: G = frozen verified facts; W = checklist 100% resolved, fixtures parse; T = `test_check_fixtures.py` + capture e2e below. Step 0: `pip install -e "ignition[test]" diff-cover`.
- Spec: Input = pinned versions + checklist. Output = verified table + fixtures. Edge = flag removed/renamed between versions (record both). Error = unverifiable item gets "unverifiable" row and a conservative status (reject, never accept).
- Pyramid: docs/data phase; 1 unit-style test (`check-fixtures.py`), e2e scenario `capture_fixtures_pod` (`capture-fixtures.sh` against pinned vLLM on pod yields all listed files; `[NEEDS-GPU]`/`[NEEDS-NETWORK]`).
- Coverage: N/A for docs; the check script is covered by its own pytest (`CMD-PY-COV` with `--cov=scripts/compat`, `CMD-DIFF`; >=90% line / >=75% branch).
- Evidence: `pytest scripts/compat/tests -q`; `python -I scripts/compat/check-fixtures.py`.

## ADR Rationale
No ADR; pins doc is the record. Decision CD13 (versions) recorded there. Alternative "research-only, trust memory" rejected: the compat matrix would encode guesses and users hit silent mismatches.

## Risk Assessment
Versions drift quickly: pin + re-verify task each release (phase 10 todo). vLLM CPU build may not run the tiny model: capture on pod. Fixtures from a real server may embed machine paths: scrub before commit.

## Security Considerations
Venvs from PyPI are untrusted code: isolated scratch dir, no repo cwd, `-I`. HF token not used (public repo only); never commit tokens in captured headers.

## Next Steps
Unblocks 02, 06 (W1) and 05 (W1b). Hands fixtures to 03, 04, 09, 10.
