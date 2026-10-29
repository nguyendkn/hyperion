# Phase 02: P1 core correctness

## Context Links
- `docs/project-overview-pdr.md` FR-9, FR-10 and section 9 item 2; research-02 idea 6 (HTTP semantics), research-03 ideas 2/6
- Files: `helios-core/src/{api,state,scheduler,tokenizer}.rs`, `helios-core/tests/encode_path_smoke.rs`

## Overview
Priority P1. Pending. Depends: 01. Wave: W2. Effort: 4d. Fix existing runtime defects before building features on them. Cache redesign is phase 07 (this phase only keeps cache API stable).

## Key Insights
- State machine (AtomicU8 + CAS, 6 states) is a decided design. Scheduler ALREADY holds `pub state: Arc<StateMachine>` (scheduler.rs:107; StateMachine clones over an inner Arc); only the constructor signature may change.
- Transition table and drain tests already exist in state.rs (`transition_table_is_exactly_the_documented_set`); add only new coverage.
- 501 no-tokenizer / 503 Uninitialized|Stopped decisions stay.

## Requirements
- Truncation of over-long prompt -> HTTP 400 (not silent cut); tokenization in `spawn_blocking`.
- In-flight request counting (guard type, decrement on drop) used for graceful shutdown drain.
- OpenAI-shaped JSON errors `{error:{message,type,param,code}}`, one error enum -> status mapping; additive machine-readable `fix` hint field (idea from research-02).
- New tests: api error mapping, truncation 400, in-flight guard, shutdown drain (state transition table tests already exist; extend only if a new state edge is added).

## Architecture
`ApiError` enum -> `IntoResponse`; `InflightGuard(Arc<AtomicUsize>)`; Scheduler keeps its existing `state: Arc<StateMachine>`. No hardware knowledge (rule 1).

## Related Code Files
- Modify: api.rs (after 01's `/health`), state.rs, scheduler.rs, tokenizer.rs, lib.rs, helios-cli/src/main.rs (shutdown drain wiring only, after 01)
- Create: helios-core/src/error.rs, tests/api_errors.rs, tests/shutdown_drain.rs
- Append: 02 named section in docs/unverified-ledger.md (append-only, after 08 landed; [NEEDS-GPU] real decode, etc.); no other section touched.

## Implementation Steps
1. Tests first: error JSON shape fixtures, 400 on truncation, drain waits for guards.
2. Introduce `error.rs`, route all handlers through it.
3. `spawn_blocking` around tokenizer; length check against runtime `--max-seq-len` (phase 01).
4. In-flight guard + drain in shutdown path.
5. Confirm scheduler/API share one `Arc<StateMachine>` (no duplicate); adjust constructor only if drain wiring needs it.

## Todo
- [ ] error.rs  [x] truncation 400 (done after phase 01, see reports/phase-01-report.md)  [ ] inflight + drain  [ ] tests

## Success Criteria
- [RUN] `cargo test -p helios-core` green; new tests listed by name: api_errors (>=5 cases), truncation_400, inflight_guard_drop, shutdown_drain.
- [RUN] shutdown test: request in flight delays Stopped until guard drops, with timeout.
- [NEEDS-GPU] real decode untested; recorded in ledger (no executable claim here).

## Test Strategy & Quality Gate
- Spec: Goal = defined, observable failure semantics. AC: Given 2x max prompt Then 400 `context_length_exceeded`; Given no tokenizer Then 501; Given state Stopped Then 503; Given N in-flight on shutdown Then drain then exit. I/O: OpenAI error JSON. Out of scope: SSE terminal errors, 429 (phase 07).
- Pyramid 70/20/10. E2E: axum test server (real, in-process) + stub backend, POST oversized prompt, assert 400 body.
- Coverage diff line >=90 / branch >=75.
- Evidence: `cargo test -p helios-core`; phase-01 templates `CMD-RUST-COV` (`-p helios-core` replaces `--workspace --features stub`; output `target/coverage/02/lcov.info`) and `CMD-DIFF` (Rust form); branch gate may be UNVERIFIED per template.

## ADR Rationale
Context: handlers return ad-hoc strings; scheduler duplicates state. Decision: single error enum + shared Arc state. Why over alternatives: per-handler status codes (inconsistent, untestable), Mutex around state (violates lock-free read decision).

## Risk Assessment
Behaviour change for clients that relied on silent truncation: documented 400. Low risk of CAS ordering bugs: loom-style test optional, only if time.

## Security Considerations
Error bodies must not echo prompt text or paths; bounded request body (limit set here, tuned in 07).

## Next Steps
Unblocks 07. Out of scope: cache index, scheduling policy.
