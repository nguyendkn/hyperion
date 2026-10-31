# Phase 08: Sampling and decoding features needing backend / StepPlan changes (OWNER-GATED)

## Context Links
- `helios-hal/src/sampling.rs` (`SamplingConfig`: temperature, top_p, seed), ADR 0006 (merged trait; further change needs owner approval), ADR 0007 (slot init: temperature, top-p, top-k, seed, penalties, stop parts, max length; per-slot device state), `helios-hal/src/backends/hopper_sm90/` (forward done; paged-KV `submit_step` pending), phase 03 core-side `SamplingCaps` and validate table (03 owns `SamplingParams`, `sampling_wire.rs` and the host-side fields seed/ignore_eos/stop)
- Old plan phases 07, 10, 12, 13; `docs/decisions/0002`, `0007`

## Overview
Priority P2 (P1 if owner needs `vllm bench serve`/lm-eval parity). Status GATED: no work starts before the owner approves the StepPlan/trait impact (CD7). Wave 4, GPU. Phase 03 already parses and 400-rejects these fields behind its core-side `SamplingCaps` (no trait change); this phase owns ONLY backend/StepPlan behaviour for top_k, min_p, penalties, logit_bias, min_tokens, stop_token_ids, logprobs, echo, structured output, fills only the `SamplingCaps::from_capabilities` body from backend `capabilities` after owner approval (CD7), and flips rows.

## Key Insights
- ADR 0007 text lists top-k, seed, penalties in slot init, but hal `SamplingConfig` has only temperature/top_p/seed: first step is to diff ADR vs code and say what is truly missing (min_p, logprobs, logit_bias, guided masks likely need a StepPlan field or a new output channel).
- Logprobs change `take_output` payload (per token logprob + top-n) = trait/output contract change. Prompt logprobs (echo) need prefill logits retained: memory + kernel work; needed for lm-eval loglikelihood tasks.
- Penalties need per-slot token-count state on device. Structured output/tools need per-step logits mask from a grammar engine (CPU side, `llguidance`/`xgrammar`-class dependency or own): a separate gate (item 5).
- Client-visible risk: until this lands, requests with these fields get 400 (never silently ignored).

## Requirements
Functional (each gated item states backend work):
1. (seed, ignore_eos, stop are 03's host-side work; nothing here unless 03 step 2 flagged the legacy path as planned(08).) 2. `top_k`, `min_p`, `repetition/presence/frequency penalty`, `logit_bias`, `min_tokens`, `stop_token_ids` (device-side parts). 3. `logprobs`/`top_logprobs` (completion tokens). 4. `echo` + prompt logprobs (lm-eval). 5. Structured output (own owner gate, formerly "08b"): `response_format` json_schema/json_object, tools/tool_choice, `guided_*`/`structured_outputs` via logits masking; tool-call parser output shape. 6. Fill 03's `SamplingCaps` per backend from `capabilities` (stub implements host-side reference so Windows tests run: named test double, tagged `[RUN on stub]`; sm90 = real kernels `[NEEDS-GPU]` blocked-by sm90 KV). 7. After each item: flip row in phase-03 validate table + compat matrix.
Non-functional: determinism: same seed => same tokens (stub + sm90 each), TP-identical draws (ADR 0007 rank-identical state); no throughput regression >5% at default params (measure per runbook).

## Architecture
Extend `SamplingConfig` -> slot-init encoding in `StepPlan` (additive, versioned) -> sampling kernel reads per-slot params -> `take_output` optional logprob block. Backend declares `SamplingCaps` in `capabilities`. Grammar masks (item 5): scheduler computes a bitmask per constrained slot per step, ships as StepPlan delta or computed device-side (design in ADR 0014).

## Related Code Files
Modify (after owner OK): `helios-hal/src/{sampling.rs, step.rs (StepPlan lives here), traits.rs (output, capabilities)}`, `helios-hal/src/backends/{stub.rs, hopper_sm90/**}`, `helios-hal/kernels/*.cu` (+`bash scripts/build-kernels.sh` on GPU host; `test_kernel_binaries.py` fails without), `helios-core/src/{step_scheduler.rs, scheduler_plans.rs}` (plumb only), `docs/compat-sections/03-api-fields.json` (flip rows; file owned by 03, edited here in W4), regenerated `docs/vllm-sglang-compat-matrix.md` (08 regenerates before 09), Create: `helios-hal/tests/sampling_features_hw.rs`, `helios-core/tests/sampling_stub.rs`. `helios-core/src/sampling_wire.rs`: ONLY the body of `SamplingCaps::from_capabilities` (hook and its call in `serve_flow.rs` are 03's; no other edit). Create: `docs/decisions/0014-sampling-surface-stepplan.md`, `helios-hal/tests/sampling_features_hw.rs`, `helios-core/tests/sampling_stub.rs`. Delete: none.

## Implementation Steps
1. Gap diff: ADR 0007 vs `sampling.rs`/`StepPlan` encoding (RUN, Windows). 2. ADR 0014: list exact StepPlan/trait changes, versioning, alternatives (host-side sampling = slow; fixed-function subset), request owner approval (CD7); STOP until answered. 3. Stub reference implementation + tests per feature (RUN). 4. sm90 kernels per feature, cross-check against stub on same logits (GPU). 5. Logprobs output plumb to `/v1/*` shapes (03 wire types exist). 6. Echo/prompt logprobs; verify with `lm-eval` `local-completions` small task. 7. Item 5 structured output: owner chooses dependency (new dep = CD12). 8. Regenerate and commit `docs/vllm-sglang-compat-matrix.md` after flipping rows in 03's JSON. 9. Perf A/B per `docs/h100-test-runbook.md` (one CARGO_TARGET_DIR per build, sha256 printed).

## Todo List
- [ ] gap diff  - [ ] ADR 0014 + owner approval  - [ ] stub impl  - [ ] sm90 kernels  - [ ] logprobs  - [ ] echo  - [ ] flip matrix rows  - [ ] item 5 structured output (separate gate)  - [ ] perf A/B

## Success Criteria
- Owner approval recorded before any trait edit. `[RUN]` (doc check)
- Stub: same seed -> identical tokens over 10 runs; penalty math exact within 1e-6 (f32 abs) vs a closed-form reference; top_k/min_p empirical distribution vs analytic: chi-square p > 0.001 over 100k fixed-seed draws. `[RUN on stub]`
- sm90 vs stub on identical logits: max abs logit diff <= 1e-2 (bf16) and argmax equal on 100% of steps over 256 greedy tokens for 3 prompts. `[NEEDS-GPU]` blocked-by sm90 KV
- lm-eval `local-completions` loglikelihood (task limit 200) completes and accuracy is within 1.0 absolute point of the HF reference run on the same limit. `[NEEDS-GPU]` blocked-by sm90 KV
- Default-params decode time regression <= 5%: baseline = same commit built with the new sampling paths disabled, separate `CARGO_TARGET_DIR`, CUDA-event decode time at batch 1 and 8, median of 5 runs (runbook). `[NEEDS-GPU]` blocked-by sm90 KV

## Test Strategy & Quality Gate
- G/W/T: G = gated sampling features become real behind approved CD7; W = stub determinism/statistics, GPU cross-check, lm-eval, perf; T = `sampling_stub` (+ e2e `sampling_stub_e2e`), `sampling_features_hw`.
- Spec: Input request sampling params. Output tokens/logprobs deterministic per seed. Edge: temperature 0 + top_k, penalties with empty history, logprobs with stop, `n`>1 seeds. Error: unsupported combo -> 400 from caps.
- Pyramid: 60% unit/statistical, 35% integration (stub through scheduler), e2e `sampling_stub_e2e` (`helios-core/tests/sampling_stub.rs`, `[RUN on stub]`, after approval): openai-style requests with each of top_k, min_p, penalties through router + scheduler on the stub, asserting fields change/steer tokens as specified; the pod variant `[NEEDS-GPU]` blocked-by sm90 KV.
- Coverage: diff >=90% line / >=75% branch for Rust; kernel code: cross-check tests are the gate (coverage tool N/A for CUDA, stated in report).
- Evidence: `cargo test -p helios-core --test sampling_stub`; `cargo test -p helios-hal --features stub`; `CMD-RUST-COV`/`CMD-DIFF` (plan.md); pod: `cargo test -p helios-hal --features sm90 --test sampling_features_hw`.

## ADR Rationale
ADR 0014 required (high-risk: trait/StepPlan contract). Options: A per-slot device state + StepPlan slot-init extension (matches ADR 0007, rec); B host-side sampling on logits copied back (simple, breaks graph decode perf); C subset only (top_k, seed) leaving the rest rejected permanently. Owner picks; this plan does not.

## Risk Assessment
Trait change after "once" promise (ADR 0006): needs explicit approval. Kernel cost on decode path. Logprob memory for prompt logits. Grammar engine dependency weight. sm90 paged-KV/`submit_step` not yet implemented: this phase cannot run on GPU before old-plan phase 12/13 land that path.

## Security Considerations
Grammar/regex from untrusted clients: ReDoS and mask-compile CPU caps, timeouts; `logit_bias` bounds; seed does not leak across tenants (per-slot state cleared at slot reuse); logprobs reveal model internals only as upstream does.

## Next Steps
Phase 10 re-runs clients after flips. Without 08, matrix stays at 400-reject for these fields.
