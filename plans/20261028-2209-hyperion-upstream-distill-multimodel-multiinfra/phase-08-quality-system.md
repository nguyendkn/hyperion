# Phase 08: Quality system

## Context Links
- research-04 sections 3-4 (ideas 1-5,11,12); CLAUDE.md "Working rules"; `docs/code-standards.md`
- Files: `CLAUDE.md`, `docs/`, `.github/workflows/`, root `LICENSE` (only after decision 11). Does NOT edit `Taskfile.yml` (owned by 01); references only 01's fixed task names `check`, `test`, `schema:check`, `check:sm120`.

## Overview
Priority P2. Pending. Depends: none. Wave: W1 (parallel with 01). Effort: 3d. Can start immediately (docs/CI only). Makes the no-GPU limits explicit and enforceable.

## Key Insights
- Process ceremony can exceed value for a small team: adopt rules and ledger, defer the staged-diff hash hook (09/backlog).
- Honesty ledger = the answer to "claims unverifiable without GPU".

## Requirements
1. CLAUDE.md additions: REJECTION criteria (scope expansion, rule 1/2 violation, weakened guarantee, speculative abstraction, unfocused validation); task -> required-doc table; "no per-arch/per-model boolean flags in helios-core" (differences = manifest data); "ideas only from upstream" rule IF decision 11 = yes.
2. `docs/unverified-ledger.md`: per backend/feature: executed here / type-checked only / needs GPU; updated by phases 02-07 (append-only named sections) and seeded by 08 from 01's hand-off.
3. Four-layer test model documented (not wired by editing Taskfile): L0 CPU (stub, schema, fixtures, oracle, sim) = `cargo test --workspace --features stub` + `pytest ignition/tests` (cargo/pytest form; `task test` = `cargo test --workspace` + schema:check, pytest is the separate `ignition:test` task); coverage commands = phase 01's templates `CMD-RUST-COV`/`CMD-PY-COV`/`CMD-DIFF`; L1 hardware-gated (`#[ignore]` + feature, env var only to enable tests); L2 manual qualification scripts; L3 bench. Names: `*_hw` suffix for L1.
4. CI: Linux + Windows L0 matrix (decision 9) incl. `cargo check -p helios-hal --features sm120`, schema drift, pytest. No GPU runners.
5. Evidence rules: perf only via real exported bundle + real server; disclose unrun checks.

## Architecture
Docs + CI only, no runtime code. `docs/` structure (this phase): rules in CLAUDE.md, `testing-layers.md` (L0-L3 defs, commands in cargo/pytest form first, `task` equivalents second), `unverified-ledger.md` (one section per phase, appended by that phase; 01's rows arrive as a hand-off in its report and 08 copies them (step 0); 02-07 append named sections), CI = one workflow, two OS matrix, L0 commands only. Later phases add new docs files or append their own named section; they do not restructure.

## Related Code Files
- Modify: CLAUDE.md, docs/code-standards.md, docs/system-architecture.md, docs/codebase-summary.md
- Create: docs/unverified-ledger.md, docs/testing-layers.md, .github/workflows/ci.yml, (optional, owned by 08) root LICENSE only after open decision 11 is answered

## Implementation Steps
0. Intra-W1: ledger seeding that consumes 01's rows runs after 01 step 7 lands; until then seed only sm120/sm90 type-check rows. 1. Ask user decisions 9, 11. 2. Write docs and ledger seeded from current state (sm120/sm90 type-check only). 3. CI workflow; run locally via `cargo`/`pytest` equivalents; CI itself unrunnable until pushed ([NEEDS-REMOTE]). 4. Document the L1 gating convention (`*_hw` suffix, `#[ignore]` + feature) used by phases 05/07; no code helper (YAGNI).

## Todo
- [ ] rules  [ ] ledger  [ ] testing-layers  [ ] CI  [ ] decision 9/11

## Success Criteria
- [RUN] markdown link check; every task name cited in docs is one of 01's four fixed names and exists in Taskfile.yml (grep, evaluated after 01 lands); L0 commands in docs run green here: `cargo test --workspace --features stub`, `pytest ignition/tests`, and list skipped L1 tests via `cargo test --workspace --features stub -- --ignored --list`.
- CI green on both OS: [NEEDS-REMOTE].

## Test Strategy & Quality Gate
- Spec: Goal = enforceable, honest process. AC: Given a PR adding sm120 code Then ledger row exists; Given the L0 test command on a no-GPU host Then it passes and lists skipped L1 tests. Out of scope: commit hash hook, GPU CI.
- Pyramid n/a (docs/CI); e2e = fresh clone runs `cargo check --workspace && cargo test --workspace --features stub && pytest ignition/tests` on Windows (`task check test` optional).
- Coverage: N/A for docs; CI config linted (actionlint if available).
- Evidence: `cargo check --workspace`, `cargo test --workspace --features stub`, `pytest ignition/tests`, `cargo test --workspace --features stub -- --ignored --list` (`task check`/`task test` optional, only if go-task installed).

## ADR Rationale
Context: many claims unverifiable here. Decision: ledger + layered tests + rejection criteria. Why over alternatives: hash-hook ceremony (bypassable, overhead), trusting PR text (violates verification gate).

## Risk Assessment
Docs rot: ledger update is part of each phase's Todo. CI minutes/cost: L0 only.

## Security Considerations
CI secrets none; no signing keys in CI.

## Next Steps
Runs first; referenced by all phases. Out of scope: GPU runners, release packaging.
