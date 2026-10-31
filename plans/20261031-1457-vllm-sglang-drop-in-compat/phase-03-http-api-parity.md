# Phase 03: HTTP API parity (schemas, auth, errors, models, tokenize)

## Context Links
- `helios-core/src/api.rs` (385 lines: `/health`, `POST /v1/chat/completions`; `ChatMessage.content: String`; fields model/messages/max_tokens/temperature/top_p/stream; 501 no tokenizer, 400 too long, 429 queue full), `scheduler.rs` `SamplingParams`, `helios-hal/src/sampling.rs` (temperature, top_p, seed), `chat_template.rs`
- phase-01 fixtures (`vllm-*/`, `sglang-*/`), `gap-analysis.md`; ADR 0007 (slot init carries sampling state)

## Overview
Priority P1. Status pending. Wave 2 (after 02 provides the complete `RuntimeConfig`). Single owner of: `SamplingParams` extension, `sampling_wire.rs`, core-side `SamplingCaps`, `ApiConfig`, `/version` content, error shapes, core `lib.rs` module declarations (metrics, api_sglang, api_sglang_wire stubs), the `AppState.metrics` field, `scripts/compat/openai_client_smoke.py`. Make the OpenAI-compatible surface match vLLM's (and where they differ, SGLang's OpenAI routes) in request fields, response/stream shape, usage, errors, auth, `/v1/models`, `/health`, `/ping`, `/version`, `/tokenize`, `/detokenize`. Sampling features the backend cannot do are parsed and REJECTED with 400 naming the field (CD8), never ignored.

## Key Insights
- api.rs is 385 lines (over the 200 rule): split into `api/` submodules first (pure move, tests green), then extend.
- vLLM tolerates unknown request keys; clients send `extra_body` extras (`chat_template_kwargs`, `top_k`, ...). Policy: unknown keys tolerated + one-time warn log; KNOWN keys that change output and are unsupported -> 400 (value-level: `top_k: -1`/`0`, `presence_penalty: 0`, `repetition_penalty: 1.0` are neutral = accepted).
- `SamplingParams` extension and `sampling_wire.rs` are owned HERE (additive `Option` fields: seed, top_k, min_p, penalties, stop, ignore_eos, logprobs, n, min_tokens, logit_bias, stop_token_ids; the last three parse-only and gated). `ignore_eos`, request `seed` and `stop` are host-side and need no trait change (EOS check `step_scheduler.rs:249`; seed today = request_id at `scheduler.rs:225`, `scheduler_plans.rs:46`; step 2 re-verifies, including the legacy non-StepPlan path: if ignore_eos cannot be honoured there, that path 400s naming planned(08), the gap row and phase 10 follow).
- `SamplingCaps` is defined HERE, core-side (struct in `sampling_wire.rs`), every gated field (top_k, min_p, penalties, logit_bias, min_tokens, stop_token_ids, logprobs, echo, structured) false; NO `ComputeBackend`/trait change. 03 also defines the hook `SamplingCaps::from_capabilities(&Capabilities)` (default all-false) and calls it in `serve_flow.rs`; phase 08 owns only that body, after the owner approves CD7.
- Helios-core touch (CD6) is API/HTTP only, hardware-agnostic (rules 1-2 hold).

## Requirements
Functional (each verified against phase-01 fixtures):
1. Chat request: `content` string OR parts array (`text` parts joined; `image_url`/audio -> 400 "multimodal not supported"), roles system/user/assistant/tool/developer, `max_tokens`+`max_completion_tokens`, `stop` (str|list, host-side match on detokenized stream, trimmed from output, `finish_reason:"stop"`, `stop_reason` set), `n` (fan-out at ingress into n sequences sharing the prefix; n>1 with `stream` per vLLM index semantics), `stream_options.include_usage`, `user` ignored, `ignore_eos` (S: skip the host EOS stop), `seed` (S: request seed -> `SamplingInit.seed`), `logit_bias`, `min_tokens`, `stop_token_ids` (parse-only, gated), `logprobs/top_logprobs`, `top_k`, `min_p`, penalties, `response_format` json_schema/json_object, `tools`/`tool_choice` -> gated 400 until 08 (neutral values accepted).
2. `/v1/completions`: prompt str | list[str] | list[int] | list[list[int]]; `max_tokens`, `stop`, `n`, `stream`; `echo`, `logprobs`, `best_of`, `suffix` gated (echo+logprobs needed by lm-eval loglikelihood: gap, ph08).
3. Responses: `id chatcmpl-/cmpl-`, `object`, `created`, `model` (echo requested name), `choices[].{index,message,finish_reason,stop_reason,logprobs:null}`, `usage{prompt_tokens,completion_tokens,total_tokens}` (+`prompt_tokens_details:null`), `system_fingerprint` per fixtures. Stream: first delta role chunk, content deltas, finish chunk, optional usage chunk with `choices:[]`, `data: [DONE]`; mid-stream error as `data: {"error":...}`.
4. Errors: JSON shape exactly as fixtures PER INVOCATION STYLE AND ROUTE (vLLM style, /v1/*: nested `error{message,type,param,code}`; SGLang style: shape from the phase-01 `sglang-*/error-*.json` fixtures, native routes per 09), selected by `ApiConfig.style`;  types `BadRequestError/NotFoundError/...`; 404 unknown model (model must equal a served name unless none configured), 400 context overflow containing "maximum context length", 401 body per fixture, 429, 503/Retry-After.
5. Auth: `--api-key`/`VLLM_API_KEY` list; `Authorization: Bearer` constant-time compare; `/v1/*` and `/tokenize` `/detokenize` protected, `/health /ping /version /metrics` open (per 01 result); no api-key => open; non-loopback bind without api-key logs a loud warning.
6. `GET /v1/models` (one entry per served name, `owned_by:"hyperion"`, `max_model_len`, `root`), `GET /version` (ONE definition, owned here: `{"version":"<helios-cli semver>"}` matching the fixture key set, plus additive keys `"engine":"hyperion"` and `"bundle":{"signed":bool,"origin":"prebuilt|autobuilt|local-path"}`; unsigned bundles show `signed:false`; values come from `ApiConfig.bundle_info`, set by phase 07 in `serve_flow.rs` after load, default `signed:false, origin:"local-path"`), `GET/POST /ping`, `/health` 200 empty body when Ready/Serving (today's 503 semantics unchanged: core never reports Building, that lives only in the CLI pre-engine router of 07), `POST /tokenize` `/detokenize` per fixtures; caps: text input <=1 MiB (1048576 bytes), detokenize <=262144 tokens, request body <=32 MiB, else 413/400 in error shape.
7. `ApiConfig` (core-side projection of 02's `RuntimeConfig`; mapping code in `helios-cli/src/serve_flow.rs`) = {version (`env!("CARGO_PKG_VERSION")`, set in `serve_flow.rs`; read by `/version`), api_keys, served_names, max_model_len, default_sampling (from manifest generation, parity checked vs vLLM `--generation-config auto`), style (`ApiStyle` vllm|sglang|legacy, defined in core), enable_metrics, enable_sglang_api, seed (default), bundle_info, max_body_bytes, tokenize_caps}. `AppState` gets `metrics: Option<Arc<Metrics>>` = None and `metrics/mod.rs` declares the empty stub `pub struct Metrics;`; 04 builds the real one from `enable_metrics`/`style`.
8. Out-of-scope routes (`/v1/embeddings`, `/v1/responses`, ...) -> 404 in vLLM error shape with message "not supported by hyperion".

## Architecture
```
axum Router -> auth layer (api_keys) -> model-name layer -> handler
handler: wire types (api/wire_chat.rs, wire_completion.rs) -> validate (api/validate.rs: tolerated | neutral | gated-reject using SamplingCaps; table = `docs/compat-sections/03-api-fields.json`, `include_str!`) -> SamplingParams -> ingress (existing channel) -> stream encoder (api/sse.rs) | collector
```

## Related Code Files
Create: `helios-core/src/api/{mod.rs, routes.rs, wire_chat.rs, wire_completion.rs, validate.rs, sse.rs, errors.rs, auth.rs, models.rs, version.rs, tokenize.rs}`, `helios-core/src/sampling_wire.rs` (wire fields + `SamplingCaps` + `from_capabilities` hook), `docs/compat-sections/03-api-fields.json`, empty stubs `helios-core/src/{metrics/mod.rs (`pub struct Metrics;`), api_sglang.rs, api_sglang_wire.rs}`, `scripts/compat/{openai_client_smoke.py, tests/test_openai_client_smoke.py}`, `helios-core/tests/{api_chat_schema.rs, api_completions.rs, api_auth.rs, api_errors_shape.rs, api_stream.rs, api_gated_fields.rs, api_sampling_fields.rs}`. Modify: `helios-core/src/{api.rs->api/mod.rs, scheduler.rs (SamplingParams additive, seed), scheduler_plans.rs (seed into SamplingInit), step_scheduler.rs (ignore_eos: skip EOS stop), error.rs, lib.rs (declares all core modules)}`, `helios-cli/src/serve_flow.rs` (ApiConfig mapping incl. `version`, `from_capabilities` call; created by 02, NOT main.rs), `docs/decisions/0011-compat-flag-policy.md` (API addendum), `gap-analysis.md` (step 2 result), `docs/unverified-ledger.md` (step 12), `docs/vllm-sglang-compat-matrix.md` (regenerated). Delete: `helios-core/src/api.rs` (moved).

## Implementation Steps
1. Pure refactor: split api.rs into `api/` modules; existing tests green. 2. Verify what reaches hal today: grep `SamplingParams` -> `StepPlan` slot init on BOTH scheduler paths; confirm seed/ignore_eos/stop host-side plumbing, implement it, and record the result in gap-analysis (rows already S; flip to planned(08) only for a path that cannot). 3. Add wire types with `#[serde(flatten)] extra` for unknown keys. 4. `validate.rs` neutral/gated table (one table = tested, mirrors phase-02 style; doc `docs/vllm-sglang-compat-matrix.md` gets an API-fields section generated from it). 5. Responses+usage; golden-compare field sets with fixtures (schema diff, not values). 6. Stop-string matcher with partial-match hold-back across stream chunks (UTF-8 boundary safe). 7. `n` fan-out. 8. Auth layer + warning. 9. models/version/ping/tokenize/detokenize. 10. Error layer unifies all paths including axum extractor errors (malformed JSON -> 400 in shape). 11. Add `ApiConfig` mapping in `serve_flow.rs` from 02's `RuntimeConfig`. 12. Update ledger rows (what was only type-checked). 13. Regenerate and commit `docs/vllm-sglang-compat-matrix.md` (via 02's generator) after writing `03-api-fields.json`. 14. Write `scripts/compat/openai_client_smoke.py` (+ pytest of its result summarizer); phases 04 and 10 reuse it, never edit it.

## Todo List
- [ ] refactor split  - [ ] wire types  - [ ] validate table  - [ ] responses+usage  - [ ] stop  - [ ] n  - [ ] auth  - [ ] models/tokenize  - [ ] errors shape  - [ ] fixture schema-diff tests  - [ ] ledger

## Success Criteria
- Recorded vLLM fixtures (read via `env!("CARGO_MANIFEST_DIR")/../helios-cli/tests/compat-fixtures`): our response JSON has identical key sets (recursively, values ignored) for chat non-stream, stream chunks, completions, models, tokenize, errors 400/401/404. `[RUN]`
- `openai` python client (`models.list`, `chat.completions.create` stream + include_usage, `completions.create`) works against `helios-cli` on the stub backend. `[RUN on stub]`
- Gated field matrix: each gated field returns 400 naming the field and `phase-08`; neutral values pass. `[RUN]`
- Auth: missing/wrong/right key table incl. open routes. `[RUN]`
- Seed/stop/ignore_eos on stub: same seed -> identical tokens over 10 runs, different seeds differ; `ignore_eos:true` finishes with `length` at exactly `max_tokens`; `stop` string trimmed and `finish_reason:"stop"`. `[RUN on stub]`
- Real model: greedy (temperature 0) first 32 tokens for 3 pinned prompts are identical across two runs and the first token's argmax equals the existing logit oracle (`scripts/gen-oracle-large.py`, `testdata/oracle/`); no phase-10 artifact needed, no token-equality swap test; logprobs parity deferred to 08. `[NEEDS-GPU]` blocked-by sm90 KV (phase 10).

## Test Strategy & Quality Gate
- G/W/T: G = OpenAI-surface parity, never silently ignored; W = key-set equality, gated matrix, auth table, openai client; T = `api_chat_schema`, `api_gated_fields`, `api_auth`, `api_sampling_fields`, e2e `sampling_fields_roundtrip`.
- Spec: Input JSON request (+headers). Output JSON/SSE + status. Edge: content parts, empty `messages`, `stop` split across chunks, `max_tokens` > remaining context, `n=3` stream. Error: malformed JSON, unknown model, bad key, unsupported field, queue full.
- Pyramid: 55% unit (wire parse, validate, stop matcher), 40% integration (axum `oneshot` with stub backend: real router, real scheduler), e2e `sampling_fields_roundtrip`: spawn `helios-cli` stub, run `openai_client_smoke.py` plus a seed/stop/ignore_eos round-trip (requests -> observed tokens/finish_reason).
- Coverage: diff >=90% line / >=75% branch (`CMD-RUST-COV -p helios-core`, `CMD-DIFF`, plan.md).
- Evidence: `cargo test -p helios-core --test api_chat_schema --test api_completions --test api_auth --test api_errors_shape --test api_stream --test api_gated_fields --test api_sampling_fields`; `python -I scripts/compat/openai_client_smoke.py --base-url http://127.0.0.1:PORT/v1`.

## ADR Rationale
No new ADR file; policy lives in ADR 0011 (reject vs tolerate rules, CD8) with an API addendum section (listed under Modify). Alternative "ignore unknown/unsupported silently" rejected (silent wrong output). CD6: touching helios-core is unavoidable because the HTTP layer lives there; no hardware coupling added.

## Risk Assessment
Schema drift vs pinned versions (fixtures + phase 10 re-capture). Stop-string/stream edge bugs (property test over random chunking). `n>1` memory use at small queue (respect 429 accounting per sequence). `ignore_eos` on the legacy non-StepPlan path may not be honourable (step 2): `vllm bench serve` needs it on the StepPlan path only; escalate to owner if the StepPlan path also fails.

## Security Considerations
Bearer compare constant-time; key never logged; request body size cap (vLLM default 32 MiB class, set explicit); `/tokenize` input length cap; error messages never echo the key; unknown-field warn log truncated (no prompt echo); no authentication on health/metrics is by design, documented.

## Next Steps
Unblocks 04, 07, 09. Phase 08 flips gated rows to supported. Matrix doc regenerated.
