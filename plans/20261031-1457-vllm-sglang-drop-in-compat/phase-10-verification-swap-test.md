# Phase 10: Verification (real clients on H100, swap test, gap closure)

## Context Links
- `docs/h100-test-runbook.md` (one `CARGO_TARGET_DIR` per build, revision + binary sha256 first, CUDA-event timing), `docs/unverified-ledger.md`, `gap-analysis.md`, phase-01 fixtures and pins, `docs/vllm-sglang-compat-matrix.md`
- Clients: `openai` python, `vllm bench serve`, `lm-evaluation-harness` (`local-chat-completions`, `local-completions`), `sglang.bench_serving`, Prometheus

## Overview
Priority P1. Status pending. Wave 5. Prove drop-in claims with the real clients and a differential "swap test", Windows-runnable parts first, H100 last. Ends with the gap table refreshed and every claim tagged `[RUN]` / `[TYPE-CHECK]` / `[NEEDS-GPU]` in the ledger.

## Key Insights
- Swap test compares behaviour, not tokens: for each launch line, did hyperion accept/warn/reject exactly as the matrix says, and are response schemas key-set-equal to the pinned vLLM/SGLang? Token equality is a different test (phase 10 oracle work, old plan).
- Real vLLM/SGLang needed as baseline: installed on the pod (or fixtures from phase 01 for the Windows layer).
- Honest scope: TP>1, fp8, sampling extras may still be rejected: those rows are asserted as rejects, which is a pass for "never silently wrong".

## Requirements
Functional:
1. `scripts/compat/swap_test.py`: input = file of launch arg strings (+style); for each: (a) hyperion `--dry-run` (parse only, exit code + diagnostics JSON; implemented and owned by 02, not modified here) compared to the matrix expectation; (b) optional live mode: launch vLLM/SGLang and hyperion with the same args (tiny model, pod), wait ready, run a fixed request battery (models, chat, stream+usage, completions, tokenize, bad key, unknown model, over-length, metrics), diff response key sets recursively, report added/missing keys, status codes, error shape.
2. Request battery from phase-01 fixtures as the offline baseline (`--baseline fixtures` for Windows).
3. Client runs on H100 (Qwen2.5-0.5B first, then 7B per runbook): openai client smoke; `vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --dataset-name random --num-prompts 200 ...` with `--ignore-eos` only while the gap row for ignore_eos is S (03, host-side); if 03 downgraded it to planned(08) the flag is dropped and the run is recorded as such; lm-eval `local-chat-completions` generate task + (if 08 done) `local-completions` loglikelihood; `sglang.bench_serving` backend `sglang-oai-chat` (+ native if 09); Prometheus scrape and dashboard-name check; K8s-style probe sim for Building (startupProbe) with a real 7B first-run build timing.
4. Auto-build on pod: first-run time (download, hash, build), second-run start-to-ready; record in `docs/performance-log.md`.
5. Release checklist: re-run pins check on new vLLM/SGLang versions; regenerate matrix.
Non-functional: scripts <200 lines each; results written under `target/compat-results/<rev>/` with revision and sha256 header.

## Architecture
`scripts/compat/{swap_test.py, battery.py, schema_diff.py, run_clients.sh, prom_check.py, schema-allow.json}`; consumes (never edits) `openai_client_smoke.py` (03) and `check-fixtures.py` (01); `schema_diff.py` = pure function (key-set tree diff) unit-tested; live launcher reuses runbook env conventions.

## Related Code Files
Create: `scripts/compat/{swap_test.py, battery.py, schema_diff.py, run_clients.sh, prom_check.py, schema-allow.json}` (explicit files, no glob), `scripts/compat/tests/test_schema_diff.py`, `scripts/compat/launch-lines/{vllm,sglang}.txt`, `docs/compat-verification-report.md` (results). Modify: `docs/h100-test-runbook.md` (append compat session section), `docs/unverified-ledger.md` (append section), `docs/performance-log.md` (append H100 compat bench rows), `gap-analysis.md` (final status). Delete: none.

## Implementation Steps
1. `schema_diff.py` + tests (RUN). 2. Battery + offline baseline run against stub server on Windows (RUN on stub). 3. `swap_test.py` dry-run mode over launch-line corpus incl. owner example; assert matrix agreement (RUN). 4. Pod session per runbook: print rev + binary sha256; build once per commit with separate target dirs; run live swap test against pinned vLLM and SGLang. 5. Client runs (list above). 6. Prometheus scrape with a real Prometheus container/binary, name check against dashboards' minimum set. 7. First-run vs cached 7B timings. 8. Write report; ledger rows with tags; refresh gap table; list remaining rejects with their phase.

## Todo List
- [ ] schema_diff  - [ ] battery+offline run  - [ ] dry-run corpus  - [ ] pod: vllm bench  - [ ] pod: lm-eval  - [ ] pod: sglang bench  - [ ] pod: openai  - [ ] Prometheus  - [ ] autobuild timings  - [ ] report+ledger

## Success Criteria
- Offline: swap dry-run corpus 100% matches matrix expectation; schema diff vs fixtures: 0 missing keys on supported routes, extra keys only if listed in `scripts/compat/schema-allow.json`. `[RUN]`
- Pod (all `[NEEDS-GPU]`, blocked-by sm90 KV paged path + old 13 M1): `vllm bench serve` completes 200 prompts against hyperion with 0 failed requests; `sglang.bench_serving sglang-oai-chat` same; openai client scenarios pass; lm-eval generate task runs. `[NEEDS-GPU]`
- Live swap: for lines classified supported both servers start and responses are key-set-compatible; for rejected lines hyperion exits 2 with the documented message and no download. `[NEEDS-GPU]`
- Prometheus scrape ok; required metric names present. `[NEEDS-GPU]`
- Any failing client row becomes either a fix item or a documented reject in the matrix (never silent). `[RUN]` (doc check)

## Test Strategy & Quality Gate
- G/W/T: G = drop-in claims proven by real clients; W = dry-run corpus 100%, schema diff, client runs; T = `scripts/compat/tests/test_schema_diff.py`, `swap_test.py --mode dry-run`, e2e `swap-owner-example`. Step 0: `pip install -e "ignition[test]" diff-cover`.
- Spec: Input launch lines+battery. Output pass/fail table + diffs. Edge: unknown flag, flag order, `--config`. Error: server fails to start, auth mismatch.
- Pyramid: unit (schema_diff), integration (stub live battery), e2e (pod clients) - this phase IS the e2e layer; named scenario `swap-owner-example`.
- Coverage: diff >=90%/>=75% for scripts/compat python (`CMD-PY-COV` with `--cov=scripts/compat`, phase dir `compat-10`; `CMD-DIFF`, plan.md); pod runs report results not coverage.
- Evidence: `pytest scripts/compat/tests -q`; `python -I scripts/compat/swap_test.py --mode dry-run --lines scripts/compat/launch-lines/vllm.txt`; pod commands recorded in the report.

## ADR Rationale
No ADR (verification). Alternative "token-equality swap test" rejected: depends on sampling/numerics (separate oracle track) and would hide schema/flag issues behind numeric noise.

## Risk Assessment
Pod cost/time (batch all clients in one session). Upstream tools change CLI: pinned versions. vLLM needs a GPU build matching pod CUDA. Results on one GPU, batch sizes small: claims limited to what ran.

## Security Considerations
API keys in test lines are dummy; tokens never printed; scratch venvs isolated; scripts run with `-I`.

## Next Steps
Feeds release notes + re-check schedule per upstream release; open rows return to owning phases (02/03/08).
