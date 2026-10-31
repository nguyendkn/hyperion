# Phase 09: SGLang native endpoints + launch_server shim (optional, last)

## Context Links
- researcher-02 (native routes, `/generate` shapes, readiness polling; all [U] until phase 01), phase-01 `sglang-*/` fixtures, phases 02 (style detection, `launch_server` subcommand), 03 (SamplingParams, ApiConfig), 04 (`sglang:*` metrics)

## Overview
Priority P3. Status optional. Wave 4 (parallel with 08; file-disjoint). Serve the SGLang-native HTTP routes and make the exact line `python -m sglang.launch_server --model-path ...` replaceable. OpenAI routes already work (03); this only adds what SGLang-only clients (`sglang.bench_serving --backend sglang`, `sglang.test` helpers) call.

## Key Insights
- Safest SGLang bench target is `--backend sglang-oai-chat` (OpenAI routes only); native `/generate` is for the `sglang` backend.
- Classic trap: `/generate` stream text is cumulative vs delta (phase 01 verifies); error shape on native routes differs from OpenAI shape.
- `python -m sglang.launch_server` cannot be intercepted by a Rust binary: options (CD14): (A) document `hyperion launch_server <same args>` (rec); (B) ship a tiny pip package `hyperion-sglang-shim` providing module `sglang.launch_server` that execs `hyperion launch_server "$@"` (name-squats the `sglang` namespace: conflicts if real sglang installed; opt-in venv only); owner decides.

## Requirements
Functional: `POST /generate` (`text`|`input_ids`, `sampling_params` {max_new_tokens, temperature, top_p, stop, ...; non-supported non-neutral -> 400 same policy as 03}, `stream`, batch list), response `{text, meta_info{id, finish_reason{type,...}, prompt_tokens, completion_tokens, cached_tokens, e2e_latency}}`, stream per verified shape; `GET /get_model_info`, `GET /server_info` (fields read by bench_serving: `max_total_num_tokens` etc., from our KV pool/ledger), `GET /health_generate` (1-token generate; 503 until loaded; during a build the 07 Building router answers 503 for it when `enable_sglang_api`), `GET /ready`, `GET /ping`, `POST /flush_cache` (200 if idle else per 01), `/tokenize` `/detokenize` (already 03); startup log line for log-grepping scripts (phase 07). `--api-key` on all but health routes per 01.
Non-functional: files <200 lines; native routes only mounted when `RuntimeConfig.enable_sglang_api` (SGLang style or `--enable-sglang-api`; 02) -> `ApiConfig.enable_sglang_api` (03); same condition as 07's Building-router answers.

## Architecture
`helios-core/src/api_sglang.rs` (+ `api_sglang_wire.rs`) maps native bodies to the same `SamplingParams` + ingress path as 03 (no second engine path); response encoders produce native shapes. `helios-cli/src/compat/sglang_shim.rs`: argv0/`launch_server` normalization (module-invocation style `-m sglang.launch_server` tokens tolerated if the shim passes them).

## Related Code Files
Create (replace 03's empty stubs; no lib.rs edit): `helios-core/src/{api_sglang.rs, api_sglang_wire.rs}`, `helios-core/tests/api_sglang.rs`, `helios-cli/src/compat/sglang_shim.rs` (replaces 02's stub; no compat/mod.rs edit), `scripts/compat/sglang-shim/` (only if CD14=B). Create also: `docs/compat-sections/09-sglang-routes.json` (rendered by 02's generator). Modify: `docs/vllm-sglang-compat-matrix.md` (regenerate and commit after 08 merges), `helios-core/src/api/routes.rs` (conditional mount; after 03/04), `docs/decisions/0011-compat-flag-policy.md` (CD14 addendum). Delete: none.

## Implementation Steps
1. Get CD14 answer. 2. Wire types from fixtures (read via `env!("CARGO_MANIFEST_DIR")/../helios-cli/tests/compat-fixtures`). 3. `/generate` non-stream, then stream with verified chunk semantics, then batch. 4. Info/health/ready/flush routes. 5. Fixture schema-diff tests. 6. Run `sglang.bench_serving --backend sglang` against stub server (`[RUN on stub]`, needs pinned sglang installed in scratch venv, `[NEEDS-NETWORK]` for install). 7. Matrix rows: write `09-sglang-routes.json`, regenerate and commit `docs/vllm-sglang-compat-matrix.md`.

## Todo List
- [ ] CD14  - [ ] /generate  - [ ] stream  - [ ] info/health routes  - [ ] schema-diff  - [ ] bench_serving smoke  - [ ] shim (if B)

## Success Criteria
- Key-set equality vs SGLang fixtures for `/generate` (non-stream, stream chunk), `/get_model_info`, `/server_info`. `[RUN]`
- `sglang.bench_serving` (native backend) completes a small run on the stub server; numbers not compared. `[RUN on stub]`
- `launch_server --model-path X --tp 1 --mem-fraction-static 0.85 --context-length 4096 --host 0.0.0.0 --port 30000` accepted (02 golden) and serves. `[RUN on stub]`
- Real model: `[NEEDS-GPU]` phase 10.

## Test Strategy & Quality Gate
- G/W/T: G = SGLang-native clients work; W = key-set equality, bench_serving native run, accepted launch line; T = `api_sglang`, e2e `sglang_native_bench`.
- Spec: Input native JSON. Output native JSON/SSE. Edge: batch text list, `input_ids`, cumulative stream. Error: unsupported sampling param, not-ready 503, bad key.
- Pyramid: 60% unit, 35% integration (router + stub), e2e `sglang_native_bench`: `sglang.bench_serving --backend sglang` run against the stub server. Coverage diff >=90%/>=75% (`CMD-RUST-COV -p helios-core`, `CMD-DIFF`, plan.md).
- Evidence: `cargo test -p helios-core --test api_sglang`; `python -I -m sglang.bench_serving ...` (scratch venv).

## ADR Rationale
No ADR; CD14 recorded in the ADR 0011 addendum (listed under Modify). Rejected: always-on native routes (surface area + auth exposure for vLLM-style users).

## Risk Assessment
SGLang-native shapes change often (pin + fixtures). Shim name-squatting (opt-in only). Low user value if clients use OpenAI routes: keep optional, cut if schedule slips (stated, not silent: matrix marks native routes `not supported` and `/generate` returns 404 with message).

## Security Considerations
`/flush_cache` and `/server_info` expose internals: behind api-key like upstream; `/server_info` must not echo secrets (api-key, HF token); native routes off by default for vLLM style.

## Next Steps
Phase 10 runs native clients on pod.
