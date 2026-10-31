# Validation Round 2

**Lane:** high-risk | **Round type:** delta (r2)
**Lenses run:** A,B | **Carried:** C,D 'carried PASS (r1)' | **Scope:** bundle delta since r1

## Lens Results
### 4.A Structural — 2 BLOCKER / 4 MAJOR-MINOR
- BLOCKER A-B1: `scripts/compat/openai_client_smoke.py` used by phase-03:54,56 and phase-04:40 (W2/W3) but created by phase-10 (W5).
- BLOCKER A-B2: `helios-cli/src/lib.rs` `pub mod model_source;` / `autobuild;` declarations have no owner (02 owns lib.rs; 05 W1 parallel; 07 W3); `compat/sglang_shim.rs` (09) same.
- MAJOR A-M1: `scripts/compat/schema-allow.json` (phase-10 success criterion) missing from phase-10 Create and matrix.
- MAJOR A-M2: phase-10 `scripts/compat/*` glob overlaps phase-01 files.
- MAJOR A-M3: matrix row 04 omits scheduler.rs, lib.rs, metrics_bench.rs; row 09 omits lib.rs / compat/mod.rs declarations.
- MAJOR A-M4: phase table 05 Depends vs matrix vs phase-05:9 disagree.
### 4.B Consistency — 4 BLOCKER / 7 MINOR
- BLOCKER B-B1: `AppState::new` builds `Arc<Metrics>` in 03 (W2) but `Metrics` type is created by 04 (W3).
- BLOCKER B-B2: phase 05 (W1) depends on 02 (`[lib]`, RuntimeConfig, Cargo.toml) yet table says Depends 01 and runs parallel.
- BLOCKER B-B3: phase-02:52 'no flag Supported without a consumer' cannot hold at W1; reader map incomplete (tp_size, kv_fraction, prefix_cache, max_num_seqs, kv_dtype, host, port, log_level).
- BLOCKER B-B4: phase 08 fills `SamplingCaps` but `sampling_wire.rs` is 03's and 'read, not modified'.
- M1: `planned(phase)` runtime-only vs 02 report stage exit 2 (plan.md:15, phase-02 step 9, phase-07 req 8).
- M2: gap row `--chat-template` still P although implemented via cache key (05/06/07).
- M3: `ApiConfig` lacks `version` (used by `/version`).
- M4: reader map drift: 07 reads cache_dir, link_mode, chat_template; dtype/quant read only by cache key.
- M5: 03 wire fields lack min_tokens, logit_bias, stop_token_ids (parse-only).
- M6: gap-analysis.md row edits by 01, 03, 10: ordering not stated.
- M7: phase-04 ADR text says CD12 (should be CD6 + no-new-dep note); `[verify 01]` undefined tag (phase-02:21); cross-crate fixture path form unstated; Ignition recipe for stub arch id unchecked (phase 07 target_arch on Windows).
### 4.C Fidelity — carried PASS (r1)
### 4.D Testability — carried PASS (r1)

## Fixes Applied
Greps run after edits over plan.md, gap-analysis.md, phase-*.md: `openai_client_smoke`, `Arc<Metrics>`, `verify 01`, `CD12`, `parallel with 02`, `W1)`, `without a consumer`, `read, not modified`, `scripts/compat/\*`, `lib.rs` (stale hits fixed; remaining reviewed). plan.md = 68 lines.
1. A-B1 -> `openai_client_smoke.py` (+ `test_openai_client_smoke.py`) created by 03 (W2, new step 13, matrix row 03); 04 and 10 consume only; removed from 10's Create/matrix. Grep: `openai_client_smoke`.
2. A-B2 -> 02 creates `lib.rs` declaring ALL CLI modules + empty stubs (model_source/mod.rs, autobuild/mod.rs, compat/sglang_shim.rs); 05/07/09 replace stub bodies, no lib.rs edit; core `lib.rs` owned by 03 declaring metrics, api_sglang, api_sglang_wire stubs (04/09 never edit); stated in matrix + ordering notes. Grep: `lib.rs`.
3. A-M1 -> `schema-allow.json` in phase-10 Create + Architecture + matrix row 10. A-M2 -> row/Create 10 list explicit files, consumes (never edits) 01/03 scripts. Grep: `scripts/compat/\*`.
4. A-M3 -> row 04 adds api/mod.rs, scheduler.rs, step_scheduler.rs, tests incl. metrics_bench; row 09 stubs replaced, no declaration edits; row 03 completed (AppState, stubs, scripts, serve_flow). A-M4 -> table, matrix, phase-05 Overview all: Depends 01,02, W1b; phase 06 text and 01 Next Steps aligned. Grep: `W1b`, `parallel with 02`.
5. B-B1 (pick: 03 owns field) -> 03 adds `AppState.metrics: Option<Arc<Metrics>>` = None + empty `pub struct Metrics;` stub; 04 owns real Metrics and the `AppState::new` build line (api/mod.rs in row 04); plan.md, 02, 03, 04 updated. Grep: `Arc<Metrics>`.
6. B-B2 (pick: wave 1b) -> 05 Depends 01,02, W1b, after 02 merges; consistent in table, matrix, phase-05, phase-06, phase-01.
7. B-B3 -> 02 criterion now 'every Supported maps_to field has a named reader phase' with `compat/readers.rs` const table + test; reader map completed (tp_size, kv_fraction, prefix_cache, max_num_seqs, kv_dtype, host, port, log_level, plus M4 set). Grep: `without a consumer`.
8. B-B4 -> 03 defines `SamplingCaps::from_capabilities` hook (default all-false, called in serve_flow.rs); 08 edits only its body; sampling_wire.rs added to matrix row 08 and phase-08 Modify. Grep: `read, not modified`, `from_capabilities`.
9. M1 -> planned flags exit 2 at 02 report stage (plan.md:15, phase-02 req 7 + step 7, phase-07 req 8). M2 -> gap row `--chat-template` = S (ph05/06/07). M3 -> `ApiConfig.version` set in serve_flow.rs (03 req 7, Modify). M4 -> reader map: 07 reads cache_dir, link_mode, chat_template (+max_num_seqs); dtype/quant cache key only. M5 -> min_tokens, logit_bias, stop_token_ids parse-only gated in 03 (fields, SamplingCaps, plan.md note). M6 -> ordering note: gap-analysis row edits sequential 01, 03, 10.
10. M7 -> phase-04 ADR text CD6 + no-new-dep; phase-04:13 aligned; `[verify 01]` replaced by pins-doc row reference (no tag); cross-crate fixture path `env!("CARGO_MANIFEST_DIR")/../helios-cli/tests/compat-fixtures` in plan.md, 03, 04, 09; phase-01 step 7 checks Ignition recipe for stub arch id, phase 07 uses documented override otherwise.

**BLOCKERs this round:** 6 | **Round counts as clean:** no
**Exit state:** continue -> delta round 3 re-running lenses A, B
