# Phase 04: Prometheus metrics (vllm:* and sglang:*)

## Context Links
- researcher-01 section 4, researcher-02 section 3 (metric names; versions verified in phase 01); `helios-core/src/{step_scheduler.rs, scheduler.rs, cache.rs, kv/}` (prefix cache, queue)
- Consumers: Dynamo/llm-d/KServe/vllm-router dashboards, `vllm bench serve` (does not need /metrics), Prometheus scrape

## Overview
Priority P2. Status pending. Wave 3. Add a small in-process metrics registry in helios-core and expose `GET /metrics` in Prometheus text format with the vLLM names (always) and SGLang names (when launched sglang-style or `--enable-metrics`).

## Key Insights
- Minimum useful set (research-01): running, waiting, kv_cache_usage_perc (+ legacy `gpu_cache_usage_perc` alias), prompt/generation tokens total, TTFT, e2e, ITL, request_success_total{finished_reason}, prefix cache queries/hits.
- Names drift per release: names come from a table verified in phase 01, not from memory; metric table doubles as docs section.
- Hand-rolled registry (counters, gauges, fixed-bucket histograms, text encoder) ~250 lines avoids a dependency; hand-rolled = no new dependency (CD6 addendum, rec); the `prometheus` crate is the alternative and WOULD be a new dependency needing its own CD12 entry (reason in ADR 0011), not chosen; escape/format rules covered by tests against `promtool`-style parser in a test.

## Requirements
Functional: (1) registry with labels per style (vLLM style: `model_name`, `engine="0"`; SGLang style: `model_name`, `engine_type` per phase-01 fixtures; verified, not assumed); (2) hooks: scheduler (running/waiting gauges, token counters, per-request timestamps -> TTFT/ITL/e2e/queue/prefill/decode histograms), KV (usage fraction from pool/ledger; 0 when backend has no pool, documented), prefix cache (queries/hits), request finish reasons; (3) `/metrics` text with `# HELP/# TYPE`, `_total` suffix rules per fixture; (4) sglang family `sglang:*` (`num_running_reqs`, `num_queue_reqs`, `token_usage`, `cache_hit_rate`, `prompt_tokens_total`, `generation_tokens_total`, TTFT/e2e/ITL hist) emitted only when style=sglang or flag; (5) `--disable-log-stats`/`--enable-metrics` honoured via `ApiConfig.enable_metrics` (02 `RuntimeConfig` -> 03 `ApiConfig`); (6) metrics overhead: no lock on hot path (atomics).
Non-functional: files <200 lines; no allocation per token.

## Architecture
`metrics/{mod.rs, registry.rs, histogram.rs, encode.rs, names_vllm.rs, names_sglang.rs, hooks.rs}`; 03 adds `AppState.metrics: Option<Arc<Metrics>>` (None) and a stub `Metrics`; this phase replaces `metrics/mod.rs` and edits `AppState::new` (api/mod.rs) to build `Arc<Metrics>` from `ApiConfig {enable_metrics, style}`: no helios-cli or main.rs/serve_flow.rs change in this phase; scheduler/step_scheduler call `hooks::*` at admit, first token, token, finish; `/metrics` handler encodes snapshot.

## Related Code Files
Create: `helios-core/src/metrics/**`, `helios-core/tests/{metrics_encode.rs, metrics_hooks.rs, metrics_names_fixture.rs}`. Create also: `docs/compat-sections/04-metrics.json` (+ regenerated `docs/vllm-sglang-compat-matrix.md`) (names table; matrix generator from 02 renders it), `helios-core/tests/metrics_bench.rs`. Modify: `helios-core/src/{step_scheduler.rs, scheduler.rs (hook calls only), api/mod.rs (`AppState::new` builds Metrics, after 03), api/routes.rs (route mount, after 03)}` (no lib.rs edit: 03 declared `metrics`), `docs/decisions/0011-compat-flag-policy.md` (metrics addendum). Delete: none.

## Implementation Steps
1. Registry + histogram + encoder with unit tests (bucket math, label escaping, NaN/inf). 2. Name tables from phase-01 `metrics.txt` fixtures (read via `env!("CARGO_MANIFEST_DIR")/../helios-cli/tests/compat-fixtures`); test: every fixture metric name in the required minimum set is present in our output (names+types+labels, values ignored). 3. Hooks in scheduler, driven by the deterministic simulator (`simulator.rs`) for exact-value tests. 4. Mount `/metrics`. 5. KV usage source. 6. sglang names gated on style. 7. Parse our output with a strict text-format parser in a test; if `promtool` exists, run `promtool check metrics` (`[NEEDS-TOOL]`). 8. Regenerate and commit `docs/vllm-sglang-compat-matrix.md` after writing `04-metrics.json`; e2e `metrics_scrape_during_smoke` lives in `helios-core/tests/metrics_hooks.rs`.

## Todo List
- [ ] registry  - [ ] names from fixtures  - [ ] hooks  - [ ] /metrics route  - [ ] sglang set  - [ ] simulator-driven value tests  - [ ] overhead check

## Success Criteria
- Simulator run of N requests yields exact counter values (tokens, success by reason) and histogram counts. `[RUN]`
- Required-minimum names/types/labels present vs fixture. `[RUN]`
- Prometheus scrape of a running stub server parses (strict parser; real Prometheus on pod in phase 10). `[RUN]` / `[NEEDS-GPU]` for KV usage on real pool.
- Hot-path microbench (`cargo test --release -p helios-core --test metrics_bench -- --ignored`; debug builds are not asserted): recording a token < 100 ns amortized over 10M iterations, no lock. `[RUN]` in release only

## Test Strategy & Quality Gate
- G/W/T: G = scrapeable vLLM/SGLang-named metrics; W = exact simulator counts, name/type/label presence, strict parse; T = `metrics_encode`, `metrics_hooks`, `metrics_names_fixture`, `metrics_bench`, e2e `metrics_scrape_during_smoke`.
- Spec: Input scheduler events. Output Prometheus text. Edge: zero requests, abort, 429 reject (counted? per fixture `admission_rejections_total`), sglang style. Error: histogram overflow bucket, concurrent updates.
- Pyramid: 70% unit, 25% integration (simulator, HTTP), e2e `metrics_scrape_during_smoke`: scrape `/metrics` before and after an `openai_client_smoke.py` run (script created by 03) on the stub server; counter deltas equal the request/token counts the script reports.
- Coverage: diff >=90% line / >=75% branch. Evidence: `cargo test -p helios-core --test metrics_encode --test metrics_hooks --test metrics_names_fixture`; `CMD-RUST-COV -p helios-core`, `CMD-DIFF` (plan.md).

## ADR Rationale
No ADR (additive observability). CD6 (core touch) and the hand-rolled registry choice (no new dependency, so no CD12 entry) recorded in the ADR 0011 metrics addendum (listed under Modify). Alternative: metrics sidecar in helios-cli only (rejected: scheduler timestamps live in core).

## Risk Assessment
Name drift between releases (fixtures + phase 10 recheck). Exposing both `kv_cache_usage_perc` and `gpu_cache_usage_perc` duplicates series (accepted for old dashboards). KV usage meaningless on stub: documented, value 0.

## Security Considerations
`/metrics` unauthenticated like upstream (documented); labels never contain prompt text or keys; model name label sanitized.

## Next Steps
Phase 09 adds nothing here except style flag use; phase 10 scrapes on pod.
