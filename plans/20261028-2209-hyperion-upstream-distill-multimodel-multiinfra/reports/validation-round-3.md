# Validation Round 3

**Lane:** high-risk | **Round type:** delta
**Lenses run:** A, C, D (B carried PASS (r2)) | **Scope:** fixes from r2 + touched artifacts

## Lens Results
### 4.A Structural — PASS (0 BLOCKER / 4 MINOR)
plan.md 80 lines; all 9 phases have the full SSOT section set incl. ADR + Security; table<->files 9/9; Wave fields match; no placeholders; matrix globs disjoint within each wave; CMD-* templates defined once (phase-01:49-51); /v1 nest step present.
- A-m1 matrix (plan.md:37-41) omits `docs/unverified-ledger.md`, which phases 04/05/06/07 list; plan.md:48 documents the append-only exception but the matrix does not show it.
- A-m2 phases 02 (phase-02:41) and 03 (phase-03:53) write ledger rows without listing the file; plan.md:48 allows only W3 (04,05,06). Hand rows to 08 or extend the exception to W2.
- A-m3 plan.md:30 "else 08 seeds and 01 appends" contradicts phase-01:25 (no docs/ edit in 01; hand-off only) and 08's W1 docs/** ownership. Drop "01 appends".
- A-m4 round-2 fix log wording drift (same root cause as A-m1).

### 4.B Consistency — carried PASS (r2)

### 4.C Fidelity — PASS (0 BLOCKER / 1 MINOR)
Verified: `/chat/completions` at root today (main.rs:163, api.rs:60-61), stale doc comment api.rs:58-59, helios-cli has no [features], Taskfile targets, quality-gate 90/75, `hyperion-compile`, build.rs, new files exempt.
- C-m1 "cc build-dep" (phase-01:24,32; plan.md:37) inaccurate: `cc` is already an optional regular dependency (helios-hal/Cargo.toml:46), [build-dependencies] holds only optional bindgen, and the `cc::Build` call in build.rs:44 is commented out. Reword: "move `cc` to [build-dependencies] if build.rs uses it".

### 4.D Testability — PASS (0 BLOCKER / 5 MINOR)
Verified: diff-cover accepts lcov (`coverage_files` = XML or lcov.info; `--compare-branch`, `--fail-under` exist), nightly 1.101 present without llvm-tools-preview (installed by phase-01 step 1), baselines (helios-core 7 integration tests listed; workspace 18; pytest 33), `origin/main` exists, oracle skip rule coherent.
- D-m1 phase-01 step 1 `--cobertura` fallback is moot: lcov is accepted (verified r3); mention `--branch-coverage` needs Cobertura for a real branch diff gate; the line-only caveat is lcov-specific.
- D-m2 CMD-RUST-COV bakes `--workspace --features stub` (phase-01:49) while phases 02/05/06/07 say "add -p X" and phase 03 uses `-p hyperion-specification --features schema` (`stub` invalid there). State: for scoped runs replace `--workspace --features stub` by `-p X [--features Y]`.
- D-m3 a branch-gate UNVERIFIED row during phase 01 has no destination until 08 creates docs/unverified-ledger.md; add to 01's hand-off list.
- D-m4 lcov.info/coverage.xml overwritten per run and phase; state per-phase output path and a gitignore entry.
- D-m5 Python branch % read via `coverage report --include=...`; say run `coverage report` after the same pytest run or read `branch-rate` from the XML.

## Fixes Applied
1. A-m1 -> `docs/unverified-ledger.md (append-only section)` added to W3 (04,05,06) and W4 (07) matrix rows in plan.md; rule restated in plan.md docs/ bullet. Grep: clean.
2. A-m2 -> phases 02 and 03 file lists gain "Hand-off to 08" line (never edit the ledger); plan.md:30 states 01/02/03 hand rows to 08. Grep: clean.
3. A-m3 -> plan.md:30 "else 08 seeds and 01 appends" removed; phase-08 docs paragraph updated (01/02/03 hand-offs, 08 creates, 04-07 append). Grep 'else 08 seeds' / '01 appends': 0 hits.
4. A-m4 -> same root cause as A-m1; covered by 1 (no separate wording to change).
5. C-m1 -> phase-01 Modify list + step 5 and plan.md matrix reworded (`cc` optional-dep -> [build-dependencies] only if build.rs calls it; cc::Build commented out today; sm90 wiring unchanged). Grep 'cc build-dep': 0 hits.
6. D-m1 -> phase-01 step 1: lcov accepted (verified); `--cobertura` only as route to a branch diff gate (`--branch-coverage`); line-only caveat marked lcov-specific in CMD-DIFF.
7. D-m2 -> CMD-RUST-COV: scoped runs replace `--workspace --features stub` by `-p <crate> [--features <f>]`, never combined; phases 02/05/06/07 cite the replacement. Grep: remaining `--workspace --features stub` hits are the unscoped template, plain test commands, or the explicit "replaces" wording.
8. D-m3 -> phase-01 step 7 hand-off now includes the Rust branch-gate UNVERIFIED row, recorded in 01's report (plan.md ledger rule covers pre-file case).
9. D-m4 -> outputs now `target/coverage/<phase>/lcov.info` and `coverage-<phase>.xml`; phase-01 owns .gitignore lines `coverage*.xml`, `.coverage` (Modify list + plan.md W1 row). Phases 02/06/07 paths updated.
10. D-m5 -> CMD-PY-COV: branch % = `coverage report` after same run or `branch-rate`; `--include` variant dropped. Grep 'coverage report --include': 0 hits.
Unfixed: none. plan.md still 80 lines.

**BLOCKERs this round:** 0 | **Round counts as clean:** yes (MINORs fixed mechanically in the fix pass)
**Exit state:** continue -> round 4 (delta: Structural only; no lens reported a BLOCKER in r3) — high-risk needs a SECOND consecutive clean round
