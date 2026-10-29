# Validation Round 2

**Lane:** high-risk | **Round type:** delta
**Lenses run:** A,B,C,D (all four reported BLOCKERs in r1) | **Scope:** fixes from r1 + touched artifacts

## Lens Results
### 4.A Structural — PASS (0 BLOCKER / 3 MINOR)
All 9 phases carry the SSOT sections incl. ADR + Security; plan.md 78 lines; matrix globs disjoint per wave; links resolve.
- A-m1 phase-07 Overview has no separate `Wave:` field (only parenthetical W4).
- A-m2 matrix W2 row for 03 omits `docs/spec-freeze-checklist.md` (phase-03 creates it).
- A-m3 phase-06 needs a Python Ed25519 dependency (cryptography/PyNaCl) in `ignition/pyproject.toml`, owned by 04 in W3; the 06->04 hand-off covers only the script line. `trybuild` dev-dep for the compile-fail test is also undeclared.

### 4.B Consistency — PASS (0 BLOCKER / 10 MINOR)
All r1 consistency fixes verified in place (cost catalog = Rust table, ownership of cli.py/pyproject/Taskfile/Cargo/helios-cli/docs, Capabilities vs ProbedBackend, contract-version owner, cache identity incl. payload id, decisions 5/6 gating, phase-09 rows, phase-07 depends); no stale references.
- B-F1 plan overrides research-03 (manifest cost catalog / seed in signed manifest) and research-01 idea 12 (reserve schema section) without annotating; only the cost-catalog override was surfaced as a design call. Add one-line rationale in plan.md.
- B-F2 phase-05:18 "create_backend/Triage keep public shape, additive only" vs phase-01/05 removing `ProbedBackend` and `triage(&Manifest,&Probed)`: say these are internal to helios-hal; `create_backend` is the only cli-facing API.
- B-F3 naming `ProbedBackend` (phase-01:21) vs `&Probed` (phase-01:45): pick one.
- B-F4 same as A-m3 (python dep + trybuild hand-off).
- B-F5 ownership gaps: per-crate Cargo.toml files in no glob (01 `cc` build-dep in helios-hal, 07 proptest dev-dep in helios-core); docs/spec-freeze-checklist.md missing from 03 glob; ledger rows of 04/06/07 not in their file lists; 01's Rust-table dump for the Python cross-check (cargo bin or committed JSON) missing from Create list.
- B-F6 phase-08 consumes 01's ledger rows while sharing W1 with 01: state an intra-W1 ordering note.
- B-F7 phase-08:17 says L0 is "what 01's `task test` runs" (cargo stub + pytest); real `task test` = `cargo test --workspace` + schema:check; pytest is `ignition:test`. Reword.
- B-F8 phase-07:8 and plan.md:30 justify W4/05-dependency by "reads 05's Capabilities" but no 07 step consumes it. Add consuming step or soften.
- B-F9 cost table centralised in `helios-hal/src/cost_catalog.rs` sits against "new arch = new dir": note it is a per-arch seed table or hang it off Capabilities.
- B-F10 phase-03:21 / phase-06:21 contract-version rejection has no AC/test and no source cite (research-04 idea 5); phase-05:4 cites research-04 idea 12 as context but phase-09:30 defers it; decision 1 shapes ModelConfig (03 req 1) but the 03 gate asks only 3,5,6,7.

### 4.C Fidelity — 1 BLOCKER / 1 MINOR (commands run: ls/grep over cited paths, cargo test -- --list x2, tool probes)
- C-B1 BLOCKER phase-01:17 "chat route stays `/v1/chat/completions` (existing nest)" and the phase-01:46 e2e POST `/v1/chat/completions`: FALSE today. helios-cli/src/main.rs:163 `api::router().with_state(app_state)` has no nest; api.rs:60-61 registers only `/chat/completions`. Fix: add `.nest("/v1", ...)` as an explicit phase-01 step (item 10) or change the e2e path.
- C-m2 api.rs:58-59 doc comment ("Mounted under /v1 by the CLI") is stale; fold into C-B1.
- Verified true: stub flag command, /health and /v1/models absent and declared new, Arc<StateMachine> already present, state tests exist, test counts (helios-core 5 lib + 7 smoke = 12; helios-hal 0), PDR section 9 / FR citations, Taskfile targets, workspace members, BackendConfig new.

### 4.D Testability — 1 BLOCKER / 5 MINOR (commands run: cargo test --list x2, pytest --collect-only, cargo llvm-cov --version, tool probes)
- D-B1 BLOCKER branch coverage >=75 stated but unmeasurable as written (phases 01:47-48, 02:46-47, 05:49, 06:48, 07:49): `cargo llvm-cov --lcov` gives line coverage only; branch data needs `--branch` (nightly-only to our knowledge, unverified because the tool is not installed; nightly toolchain is present on host). Python commands lack `--cov-branch` (01:48, 03:46) and `--cov-report=xml` (needed by diff-cover).
- D-m1 bare `diff-cover` with no args in 03:46, 05:49, 06:48, 07:49.
- D-m2 phase-01:28 adds pytest-cov to `[test]` but never says to re-run `pip install -e "ignition[test]"`.
- D-m3 `ignition build` in 04:48 and 06:47 is not a command (only `hyperion-compile`); declare it or use `hyperion-compile`.
- D-m4 phase-06:48 `diff-cover` has no coverage-file arg; `cargo llvm-cov -p helios-pkg` no output path.
- D-m5 phase-04:43 oracle conditional on `ignition[oracle]`: state whether the gate requires the extra or only reports the skip.
- D-m6 phase-05 evidence in prose without an exact command.

## Fixes Applied
1. C-B1/C-m2 -> phase-01: states route is `/chat/completions` at root today; explicit step 4 nests chat router under `/v1` in helios-cli main.rs, `/health` stays at root, fixes api.rs:58-59 comment; e2e path valid after nest. Hygiene grep `/v1/chat/completions`: only that e2e line remains.
2. D-B1 -> phase-01 SSOT templates `CMD-RUST-COV` (`cargo +nightly llvm-cov --branch --lcov --output-path lcov.info`, UNVERIFIED-on-host rule + ledger), `CMD-PY-COV` (`--cov-branch --cov-report=xml`), `CMD-DIFF` (coverage file + `--compare-branch=origin/main`, branch % read from tool tables since diff-cover is line-only); phases 02,03,04,05,06,07 cite by name. Grep `llvm-cov`/`diff-cover`: only template/plan-note lines remain.
3. D-m1, D-m4, D-m6 -> same templates; every phase evidence line now names templates and scope args (phase-05 prose replaced).
4. D-m2 -> phase-01 step 1 re-run `pip install -e "ignition[test]"`, nightly `llvm-tools-preview`.
5. D-m3 -> `ignition build` replaced by `hyperion-compile` (04, 06); `list`/`fit` declared NEW cli.py modes owned by 04; `ignition list` replaced. Grep `ignition build|ignition list`: 0 hits.
6. D-m5 -> phase-04 gate reports the skip; extra required only for the "oracle evidence" claim.
7. A-m3/B-F4 -> plan.md hand-off + phase-04 + phase-06: `cryptography` dep (Ed25519) and script line via 04; `trybuild` dev-dep in helios-pkg Cargo.toml owned by 06.
8. B-F1 -> plan.md "Deliberate research overrides" rule.
9. B-F2/F3 -> `ProbedBackend` everywhere, `&Probed` removed; internal to helios-hal, `create_backend` only cli-facing (plan.md, 01, 05). Grep `&Probed`: 0 hits.
10. B-F5 -> Cargo.toml files, spec-freeze-checklist (03 matrix glob), 04/05/06/07 ledger rows, `helios-hal/tests/data/format-table.json` Rust-table dump added to file lists and matrix globs; same-wave ledger append rule in plan.md.
11. B-F6 -> intra-W1 ordering note (plan.md, 01 step 7, 08 step 0). B-F7 -> 08 L0 reworded. B-F8 -> 07 step 3 consumes `Capabilities` (cost row by arch+formats, test). B-F9 -> note in 07. 
12. B-F10 -> 03 req 7 cites research-04 idea 5 with acceptance test + fixture; 06 test/criterion by reference; phase-05 idea 12 marked deferred (09 aligned); decision 1 added to 03 gate (1,3,5,6,7) in 03 and plan.md.
13. A-m1 -> phase-07 `Wave: W4` field. A-m2 -> 03 matrix glob adds docs/spec-freeze-checklist.md.
Left unfixed: none. Note: diff-cover lcov support and `cargo llvm-cov --branch` unverified (tools not installed); plan tells executor to check and falls back to UNVERIFIED.

**BLOCKERs this round:** 2 (C-B1, D-B1) | **Round counts as clean:** no
**Exit state:** continue -> delta round 3 re-running lenses C, D (reported BLOCKERs) + A (always); B carried PASS (r2)
