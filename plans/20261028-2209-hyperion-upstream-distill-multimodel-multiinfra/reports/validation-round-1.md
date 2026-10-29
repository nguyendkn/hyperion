# Validation Round 1

**Lane:** high-risk | **Round type:** deep (r1)
**Lenses run:** A,B,C,D | **Scope:** whole bundle (plan.md, phase-01..09, research/*)

## Lens Results
### 4.A Structural — 3 BLOCKER / 3 MINOR
- A1 BLOCKER phase-09: no Security Considerations; also missing SSOT sections (Key Insights, Requirements, Architecture, Related Code Files, Implementation Steps, Todo, Success Criteria). One-liners / "N/A, entry-gated" allowed but Success Criteria must be a verifiable entry criterion.
- A2 BLOCKER phase-08: no `## Architecture` section.
- A3 BLOCKER (4.A.5) W3 runs 04||05||06: phase-04:27 and phase-06:26 both modify `ignition/src/hyperion_ignition/cli.py`; phase-06:28 edits root Cargo.toml/Taskfile/helios-cli although plan.md:39,45 say 01 owns Taskfile/Cargo.toml; phase-05 modifies helios-cli load path (adapter) not in its file list while 06 changes the same load path. "serialize on conflict" is not an ownership rule.
- A4 MINOR cross-wave touches (02 vs 07: scheduler/state/api; 01 vs 05: stub/triage/backend_factory) lack declared hand-off note in matrix.
- A5 MINOR non-verifiable success bullets: phase-02 bullet 3 ("No GPU claims..."), phase-07:44 ("Throughput claims: none"), phase-04 bullet 3 ([NEEDS-GPU] comparison, no pass condition).
- A6 MINOR phase-05 Success Criteria is one run-on line, [NEEDS-GPU] part has no verifiable condition.
- Passed: 4.A.2 table<->files, 4.A.3 placeholders, 4.A.7 links; ADR Rationale + Security present in 01-08.

### 4.B Consistency — 3 BLOCKER / 12 MINOR
- B1 BLOCKER phase-07:16 consumes "manifest graph/cost-catalog field (from 03) keyed by target_arch"; phase-09:18 "spec reserves section (03)"; plan.md:53 and phase-05:19 also mention a cost catalog; phase-03 candidate fields 1-7 define no such field.
- B2 BLOCKER plan.md:45 says "05 only triage output" for helios-cli but phase-05:32 step 3 decouples the trait with an adapter in helios-cli while phase-06 modifies helios-cli load path (AuthorizedPackage) in the same wave W3.
- B3 BLOCKER cli.py clash 04 vs 06 (same as A3); W3 matrix row gives 04 only `compiler/**`, 06 no ignition glob.
- M1 matrix globs omit files phases touch (02: error.rs, lib.rs, helios-cli drain; 03: manifest_schema.py + tests; 05: triage.rs, backend_factory.rs, capabilities.rs, device_group.rs, tests; 06: root Cargo.toml, Taskfile, ignition signing.py; 07: state.rs, api.rs, lib.rs, helios-cli; 03/04/06/08 write docs/ which 08 owns).
- M2 phase-08:20,31 (L1 tests in `task test`, every Taskfile target exists) and phase-08:5 ("Taskfile via phase 01") conflict; 08 and 01 are both W1.
- M3 07 Depends lists 02,03 only but 05 and 06 say "Feeds 07"; W4 placement has no stated reason.
- M4 phase-04:42 plans a Rust `#[ignore]` oracle test but no phase lists a file for it.
- M5 phase files carry no effort/depends fields (only plan.md table).
- M6 01 defines `ProbedBackend{arch,formats}` + single `supported_formats(arch)` table; 05 defines per-backend `capabilities()` with formats. State that Capabilities supersedes.
- M7 contract-version "re-export" rejection split across 01 (context only), 03 item 7, 06 req 6; pick one owner.
- M8 pre-socket capacity error claimed by phase-01:29 and phase-07:20.
- M9 cache identity: phase-06:60 says payload id; 03:20 and 07:17 inputs omit it.
- M10 decision 8 (plan.md:65) says keep state-row hook in "cache/trait" but only 07 adds a cache hook.
- M11 research-02 ideas 10,12 (trust store, encrypted keygen) tagged adopt-later but phase-06 req 4-5 pulls them into W3 silently; state the promotion reason.
- M12 decisions 5 and 6 (publisher identity, safetensors vs framing) are asked in phase-06 step 1 after spec edits (phase 03) are done; gate phase 03 on 3,5,6,7 and do not re-ask 6 later.
- M13 phase-04 builds the Python oracle without citing open decision 2; assumes decision 1 (hedged at :21).
- M14 phase-09 lacks research-01 idea 10 (prompt-boundary endpoints), research-04 idea 9 (host-OS crate), idea 12 (per-arch fixture set).
- M15 research-03 idea 4 "state mode ordinary/spec" dropped from cache identity inputs; state that it is deliberate (speculation skipped).

### 4.C Fidelity — 3 BLOCKER / 5 MINOR (commands run: git ls-files, cat of Cargo/Taskfile/sources, greps, cargo check/test --no-run, tool probes)
- C1 BLOCKER phase-02 Test Strategy evidence `cargo test -p helios-core --features stub` fails: "package 'helios-core' does not contain this feature: stub" (helios-core/Cargo.toml has no [features]; `stub` is on helios-hal/Cargo.toml:17). Same command in phase-07:49. Use a command that works (e.g. `cargo test -p helios-core`, verify with `-- --list`) or `cargo test --workspace --features stub`.
- C2 BLOCKER phase-01 AC "load failure => state != Ready and `/health` 503": no `/health` route exists (api.rs:60-62 router() has only `/chat/completions`); phase 01 does not create it and api.rs is not in its file list.
- C3 BLOCKER phase-01 E2E "GET `/v1/models`": `/models` does not exist (only phase-07 adds a models endpoint later).
- C4 MINOR phase-02 "Scheduler must share `Arc<StateMachine>`...delete duplicate": already true (scheduler.rs:107 `pub state: Arc<StateMachine>`; StateMachine is Clone over an inner Arc). Only the constructor signature may change.
- C5 MINOR phase-02 "exhaustive transition table test" already exists (state.rs `transition_table_is_exactly_the_documented_set`, drain test). Reword as done; keep only new coverage.
- C6 MINOR citations: docs/project-overview-pdr.md has no P0..P3 list and no items 27-28 (its §9 has items 1-12, FR-1..16). Point phase-01/05 and plan.md at PDR §9 items 5/6 and FR-8/FR-15 or at CLAUDE.md/this plan; P-labels have no on-disk source.
- C7 MINOR phase-01 Step 5 "check, test, schema:check" Taskfile targets already exist; only `check:sm120` is new. `build:sm90`/`build:sm120` pass `--features` to helios-cli which has no [features] (phase 01 fixes this).
- C8 MINOR tooling not declared: pytest-cov absent from ignition `[test]` extra; torch/transformers/safetensors/numpy not declared for the phase-04 oracle (needs a justified optional extra); cargo-llvm-cov not installed; `task` not installed.
- Passed 4.C.1 (every Modify path exists), most 4.C.2/3 claims verified true (triage shape, main.rs facts, truncation, spawn_blocking absent, RadixCache shape, unused workspace deps, supported_formats, deepseek unsupported).

### 4.D Testability — 4 BLOCKER / 5 MINOR (commands run: cargo test --list variants, pytest --collect-only, tool probes)
- D1 BLOCKER = C1 (stub flag, phases 02 and 07).
- D2 BLOCKER cargo-llvm-cov is the only Rust coverage tool (phases 01:45-46, 02:46, 05:46, 06:48, 07:49), not installed, no install step declared (plan.md:72 lists it as unresolved only).
- D3 BLOCKER phases 03:46 and 04:47 use `pytest --cov=...`; pytest-cov not installed and not in `[test]` extra; no dependency change declared (CLAUDE.md needs a reason for new deps).
- D4 BLOCKER phase-08:37-39 gate on `task check`/`task test`; go-task not installed, no install declared.
- D5 MINOR phase-01:44 e2e uses a "stub bundle fixture" and a helios-cli e2e; helios-cli has no tests/ dir and the Create list omits them.
- D6 MINOR helios-hal has 0 tests today; success criteria must name tests or a minimum count (zero tests = false green).
- D7 MINOR phase-07 sim location ambiguous ("sim crate/dir or helios-core/tests/sim"); pick one so `-p` filter and coverage command are fixed.
- D8/D9 informational (type-check-only claims correctly scoped; phase-09 N/A justified).

## Fixes Applied
Hygiene greps run over whole plan dir after all edits: `cost.catalog|graph field` (only Rust-side-table mentions remain), `P0..P3|27-28|sched_counterexamples|sim crate|helios-core --features stub|task check|task test|P3|UB ` (only intentional mentions remain), `ProbedBackend|Feeds 07`, `decisions 5,6|Ask user`, `task ` (all secondary/optional). Verified: `cargo test -p helios-core -- --list` lists 7 tests; `cargo test --workspace --features stub --no-run` builds.

1. A1 -> phase-09 rewritten with Key Insights, Requirements, Architecture, Related Code Files, Implementation Steps, Todo, verifiable Success Criteria (entry criterion), Security Considerations -> grep of phase-09 headings.
2. A2 -> phase-08 `## Architecture` added -> grep.
3. A3/B3/M1 -> plan.md matrix rewritten: exclusive globs per wave incl. all touched files; cli.py owned by 04; 06 signing in new `signing_cli.py` (`hyperion-sign`), pyproject line handed to 04 (recorded plan.md rules + 06 Architecture); 06 owns helios-cli load path; Taskfile/root Cargo.toml owned by 01 (members glob; 06/07 append deps only) -> grep `cli.py|Taskfile|Cargo.toml` across phases.
4. B2 -> 05 trait adapter moved to helios-hal `backend_config.rs`, helios-cli explicitly untouched; `create_backend`/`Triage` additive-only -> grep `adapter in helios-cli` (none left).
5. A4 -> cross-wave hand-offs listed once in plan.md (01->02, 01->05, 02->07, 05->07, 06->07) -> grep `hand-off`.
6. A5/A6 -> reworded: 02 criterion 3, 04 criterion 3 (ledger-row pass condition), 05 split into 3 tagged bullets with named counts, 07 grep-checkable rule -> grep `No GPU claims|Throughput/latency claims: none` (none left).
7. B1 -> cost catalog removed from manifest everywhere; 07 owns Rust-side `helios-hal/src/cost_catalog.rs`; 05 recipe seed dropped (YAGNI, surfaced as design decision); 03 states no cost field; 09 and plan.md reworded -> grep `cost.catalog|graph field`.
8. M2 -> 08 no longer edits Taskfile; references 01's four fixed task names; L0 gate = cargo/pytest commands -> grep `Taskfile` in 08.
9. M3 -> 07 Depends 02,03,05 (+06 wave order); 06 no longer says "Feeds 07 (cache identity)"; reason for W4 stated in plan.md -> grep `Feeds 07`.
10. M4 -> Rust `#[ignore]` oracle test removed from 04 (no backend forward to call); added as phase 09 row + ledger row -> grep `ignore` in 04.
11. M5 -> Depends/Wave/Effort line in Overview of all 9 phases -> grep `Depends:`.
12. M6 -> 01 Architecture and 05 Overview: Capabilities supersedes ProbedBackend + table; 05 migrates table, matrix tests and cross-check test -> grep `ProbedBackend`.
13. M7 -> 03 item 7 owns contract-version rejection; 06 req 6 implements by reference; research-04 idea 5 dropped from 01 context -> grep `re-export|contract`.
14. M8 -> 01 owns pre-socket capacity error; 07 req 6 only SSE/429, models, metrics; 07 context link fixed -> grep `capacity`.
15. M9/M15 -> payload id added to cache identity in 03 req 5 and 07 req 3; idea 4 spec-mode dropped deliberately, stated in both -> grep `payload id`.
16. M10 -> decision 8 reworded: 07 adds opaque `Option<OpaqueStateRef>` hook, no trait change -> grep `state-row|OpaqueStateRef`.
17. M11 -> 06 req 5 promotion note (trust store + keygen) -> grep `Promotion`.
18. M12 -> 03 gated on 3,5,6,7 (asked once); 06 step 1 reads ADR, no re-ask; plan.md Open decisions gate line -> grep `3,5,6,7|do not re-ask`.
19. M13 -> 04 Key Insights cites decisions 1 and 2; hedge at old :21 reworded -> grep `decision 1`.
20. M14 -> 09 rows added: prompt-boundary endpoints (r01 idea 10), host-OS crate (r04 idea 9), per-arch fixture set (r04 idea 12), plus Helios-vs-oracle row.
21. C1/D1 -> `cargo test -p helios-core` (confirmed lists 7 tests) in 02 and 07; 01 notes the invalid flag; workspace form `cargo test --workspace --features stub` confirmed to build -> grep `helios-core --features stub` (only the 01 warning remains).
22. C2/C3 -> 01 creates root `GET /health`, api.rs added to 01 Modify list, e2e uses `/health` and `/v1/chat/completions`; `/v1/models` moved to 07 only -> grep `/v1/models`.
23. C4/C5 -> 02 Key Insights/Steps/Requirements/Architecture reworded (scheduler Arc and transition table already exist) -> grep `Arc<StateMachine>`.
24. C6 -> citations use PDR section 9 items / FR ids; P-labels explained once in plan.md -> grep `P0\.\.P3|27-28`.
25. C7 -> 01 Step 5/Modify: only `check:sm120` new; features forwarding fixes build:sm90/sm120 -> grep `check:sm120`.
26. C8/D2/D3/D4 -> 01 step 1 dev tooling (cargo-llvm-cov + llvm-tools-preview, pytest-cov in `[test]`, diff-cover pip tool, go-task optional); 04 `oracle` extra with reason + skipif + ledger; all evidence commands have cargo/pytest forms; 08 gates cargo/pytest; plan.md unresolved question reworded -> grep `llvm-cov|pytest-cov|oracle`.
27. D5 -> 01 Create list adds helios-cli/tests/e2e_stub.rs + fixtures/stub-manifest.json (CARGO_MANIFEST_DIR) -> grep `e2e_stub`.
28. D6 -> 01 criterion >=12 named triage cases, 02 >=5 api_errors etc., 05 >=6 tests.
29. D7 -> sim fixed at `helios-core/tests/sim/`, run `cargo test -p helios-core --test sim`; no new crate -> grep `sim crate|sched_counterexamples` (none).
30. Extras: plan.md trimmed to 78 lines; phase files normalized to LF.

Not fixed / open: none from the report. Surfaced design calls for user review: no recipe cost seed; Helios-vs-oracle Rust test deferred to 09; `/health` at root.

**BLOCKERs this round:** 13 distinct (A1-A3, B1-B3, C1-C3, D2-D4; D1=C1, A3=B3) | **Round counts as clean:** no
**Exit state:** continue -> delta round 2 re-running lenses A, B, C, D (all reported BLOCKERs)
