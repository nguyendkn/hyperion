# Validation Round 1

**Lane:** high-risk | **Round type:** deep (r1)
**Lenses run:** A,B,C,D | **Scope:** whole bundle

## Lens Results
### 4.A Structural — 1 BLOCKER / 5 MINOR
- BLOCKER A1 (item 5): phases 02 and 05 (both W1) both modify `helios-cli/Cargo.toml`; matrix row 05 does not list it (plan.md:41-42,51; phase-05:28).
- MINOR A2: matrix "Owns" incomplete: phase-03 edits main.rs/scheduler.rs; 06 creates specification/fixtures/build/exit-codes.json; 07 touches docs/unverified-ledger.md, docs/serving-in-kubernetes.md, ADR 0013; 09 creates api_sglang_wire.rs + edits api/routes.rs; 04 edits matrix doc owned by 02.
- MINOR A3: phase-01 creates helios-cli/tests/compat-fixtures/** and scripts/compat/{capture-fixtures.sh,check-fixtures.py}; matrix says compat/fixtures/**, scripts/compat/** owned by 10.
- MINOR A4: unmeasurable criteria: phase-08:37 ("within tolerance"), 08:38 (baseline unnamed), 03:46, 10:40.
- MINOR A5: phase-06 ADR rationale deferred to ADR 0013 (W3).
- MINOR A6: phases 03/04/09 write ADR 0011 addenda but do not list it under Modify.
### 4.B Consistency — 5 BLOCKER / 6 MAJOR-MINOR
- B1: RuntimeConfig/ApiConfig lack fields read later (enable_metrics, style, enable-sglang-api, build_timeout, bundle/cache dir/link mode, chat_template, seed); phase 04 Arc<Metrics> wiring vs "04 never touches main.rs".
- B2: `SamplingCaps` used by phase 03 (W2) but defined by phase 08 (W4, owner-gated trait change).
- B3: cache key needs Ignition version but cache-hit path must be Python-free; no version source (propose Rust constant).
- B4: phase 07 vs 03 on `/version` extras and who adds Building; Building router covers only /health,/v1/* so SGLang pollers (/health_generate,/ready) 404 during build.
- B5: ignore_eos/seed/min_p and sampling_wire ownership contradictory (plan.md:51 vs phase-08:25; gap "S ph03" vs phase-03 conditional vs phase-08 gated); phase-10 needs ignore_eos.
- M1 matrix-doc generator ownership across 02/03/04/08/09; M2 ADR 0013 timing, CD4/CD12 assignment drift; M3 `planned` runtime behaviour undefined (Planned=Reject?), gpu-memory-utilization initial status, prefix-cache toggle owner; M4 research constraints overridden silently (--enforce-eager, --trust-remote-code rejected; single error shape vs per-style; metric labels engine/engine_type); M5 owns drift, target_arch input w/o GPU, /tokenize cap value; M6 "ph08b" undefined in table, effort sum, undefined tags ([RUN on stub] etc.).
### 4.C Fidelity — 0 BLOCKER / 3 non-blocking
- F1 phase-08 `helios-hal/src/step_plan*.rs` does not exist; StepPlan is in `step.rs`.
- F2 api.rs is 385 lines (plan says "near 200"); main.rs 493 lines already over limit.
- F3 helios-cli is bin-only (no `[lib]`): integration tests/`lib API` need `src/lib.rs` + `[lib]` declared.
### 4.D Testability — 0 BLOCKER / 6 MINOR
- M1 Spec line is Input/Output/Edge/Error, not G/W/T mapping; M2 e2e names vague (04,08,09), add stub e2e for 08; M3 use `--test <file>` filters not name substrings (04,05,07,09); M4 pytest/pytest-cov not installed on host: add `pip install -e "ignition[test]" diff-cover` step 0; M5 CMD-PY-COV for scripts/compat undefined; M6 CMD-RUST-COV/CMD-DIFF only referenced from old plan: inline; origin/main precondition. Tag risks: R1 fixture capture not fully [RUN]; R2 microbench needs --release; R4 Windows kill semantics for 07 tests; R6 phase-08 GPU items blocked by sm90 KV.

## Fixes Applied
Greps run after edits over plan.md, gap-analysis.md, phase-*.md: `08b`, `step_plan`, `main.rs`, `kv_memory_fraction`, `SamplingCaps`, `0013`, `CMD-`, `hw <sm`, `Building`, `compat-matrix`, `ignore_eos` (stale names fixed; remaining hits reviewed). plan.md = 67 lines.
1. A1 Cargo.toml double owner -> 02 sole W1 owner with CD12 list (YAML, sha2, fs4, libc/windows-sys, [lib]); 05 appends ONE HTTP-client line only after 02 merges; matrix rows 02/05/07 + ordering note. Grep: `Cargo.toml`.
2. A2 matrix Owns incomplete -> rewritten all 10 rows (serve_flow.rs, exit-codes.json, ADR 0011/0012/0013/0014, ledger, k8s doc, api_sglang_wire.rs, routes.rs, compat-sections). Grep: `Owns`/file names.
3. A3 fixtures path -> 01 owns tests/compat-fixtures + scripts/compat/{capture,check,test}; 10 owns the rest. A4 criteria -> concrete tolerances (08: logit diff 1e-2, chi-square p>0.001, 1.0 pt lm-eval, 5% perf with named baseline; 03: greedy 32 tokens vs oracle; 10: 200 prompts/0 failed, schema allow-list). A5 -> 06 writes ADR 0013 section 1 with real rationale (07 appends). A6 -> ADR 0011 addenda listed under Modify in 03/04/09.
4. B1 -> ONE complete `RuntimeConfig` in 02 (all later fields + reader map); `ApiConfig` projection in 03; `Arc<Metrics>` built in core `AppState::new`, so 04 never touches CLI; main.rs only dispatch, wiring in `serve_flow.rs`. Old plan.md:51 note replaced. Grep: `Arc<Metrics>`, `main.rs`.
5. B2 -> `SamplingCaps` core-side in 03 (gated false, no trait change); 08 fills after CD7. Grep: `SamplingCaps`.
6. B3 -> Rust const `EXPECTED_IGNITION_VERSION` in cache key (05), `hyperion-build --version-json` handshake (06), mismatch = build error (07); cache hit Python-free. [flagged CD15]
7. B4 -> `/version` content incl. `bundle{signed,origin}` + `engine` single-owned by 03; 07 Building router also answers /ping, /v1/models, /v1/* (503+Retry-After) and /health_generate, /ready when enable_sglang_api; Building only in CLI pre-engine router (CD5=A). Grep: `Building`.
8. B5 -> 03 owns SamplingParams/sampling_wire incl. ignore_eos/seed/stop; verified in code: EOS check host-side (`step_scheduler.rs:249`), seed=request_id (`scheduler.rs:225`) so S, no trait change; 08 owns only backend behaviour; gap, 03, 08, 10 aligned; 10 `--ignore-eos` conditional on gap row. Grep: `ignore_eos`, `seed`.
9. M1 -> one generator (02); phases add `docs/compat-sections/<NN>-<name>.json` (03, 04, 09; 08 edits 03's). M2 -> ADR 0013 stub rationale in 06; CD3/CD4/CD12 assignments aligned in plan.md and phases (CD4 decided 05, implemented 06). Grep: `CD4`, `CD12`.
10. M3 -> `planned(phase)` defined in plan.md (typed runtime reject); gpu-memory-utilization = planned(old 12); prefix-cache enable W / disable R, core untouched; owner-example golden updated (4 rejects). M4 -> enforce-eager and trust-remote-code W (gap rows reversed, plan.md states why); error JSON per style/route via 01 fixtures; metric labels per style. [flagged CD15]
11. M5 -> `--target-arch` input (probe or override, stub path for Windows) in 02/07; `--hw` always explicit in 06; /tokenize caps (1 MiB / 262144 tokens, body 32 MiB) in 03. M6 -> "ph08b" removed (item 5 of 08); effort sum fixed to ~40-44d; all evidence tags defined in plan.md. Grep: `08b`.
12. F1 step_plan*.rs -> `step.rs`. F2 honest counts (api.rs 385, main.rs 493) and main.rs growth moved to new `serve_flow.rs` (pure move first). F3 02 adds `src/lib.rs` + `[lib]`. Grep: `step_plan`, `lib.rs`.
13. Testability: G/W/T line in every phase; named e2e (metrics_scrape_during_smoke, sampling_fields_roundtrip, sglang_native_bench, sampling_stub_e2e [RUN on stub], serve_owner_example, serve_twice_no_python, ...); `--test <file>` filters everywhere; step 0 `pip install -e "ignition[test]" diff-cover`, CMD-RUST-COV/CMD-PY-COV/CMD-DIFF inlined in plan.md with origin/main precondition and scripts/compat python cov form; tag honesty: 01 capture [NEEDS-NETWORK]/[NEEDS-GPU], 04 microbench --release only, 07 Windows job-object [RUN] vs Linux pgroup [TYPE-CHECK], 08/10 GPU items blocked-by sm90 KV. Grep: `CMD-`.

**BLOCKERs this round:** 6 | **Round counts as clean:** no
**Exit state:** continue → delta round 2 re-running lenses A, B
