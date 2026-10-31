# Validation Round 3

**Lane:** high-risk | **Round type:** delta (r3)
**Lenses run:** A,B | **Carried:** C,D 'carried PASS (r1)' | **Scope:** bundle delta since r2

## Lens Results
### 4.A Structural — 1 BLOCKER / 3 MINOR
- BLOCKER A-B1: `build` and `cache` subcommands have no main.rs owner: phase-07 wires them but says main.rs untouched; 02 dispatch (<=30 lines) lists only serve, launch_server, legacy, compat-matrix.
- MINOR A-M1: gap-analysis.md edited by phases 01 and 03 but absent from their matrix Owns/Modify; `scripts/compat/sglang-shim/` (CD14=B) absent from row 09.
- MINOR A-M2: phase-03 success criterion 'phase-10 oracle token-for-token' references a nonexistent oracle and contradicts phase-10 'no token equality in swap test'.
- MINOR A-M3: phases 03, 04, 09 add compat-sections JSON but no step regenerates/commits `docs/vllm-sglang-compat-matrix.md`.
### 4.B Consistency — 0 BLOCKER / 4 MINOR
- MINOR B-M1: stale 'rejected at runtime' for planned flags (gap-analysis.md:12, phase-02 req 3); policy = exit 2 at 02 report stage before any download.
- MINOR B-M2: `docs/unverified-ledger.md` missing from phase-03 matrix Owns/Modify (step 12 updates it).
- MINOR B-M3: gap-analysis chat-fields row P(08) list lacks min_tokens, stop_token_ids; phase-08 Overview lacks stop_token_ids.
- MINOR B-M4: phase-04 Key Insights wording: hand-rolled metrics = no new dep; `prometheus` crate would be a new dep needing a CD12 entry; e2e `metrics_scrape_during_smoke` file unnamed in matrix row 04.
### 4.C Fidelity — carried PASS (r1)
### 4.D Testability — carried PASS (r1)

## Fixes Applied
Greps run after edits over plan.md, gap-analysis.md, phase-*.md: `cli_cmds`, `run_build`, `run_cache`, `rejected at runtime`, `oracle`, `compat-matrix.md`, `stop_token_ids`, `metrics_scrape_during_smoke`, `prometheus`. plan.md <=80 lines.
1. A-B1 -> 02 creates `helios-cli/src/cli_cmds.rs` (declared in lib.rs) with stub handlers `run_build`, `run_cache` returning typed PlannedFeature (exit 2, naming 07 / 05); main.rs dispatch (<=30 lines) routes serve, launch_server, legacy, compat-matrix, build, cache. 05 fills `run_cache` body (via gc.rs), 07 fills `run_build` body; neither edits main.rs/lib.rs. Updated phase-02 req 1/Create/step 10, phase-05 Modify + step 9, phase-07 Modify + step 6 + todo, plan.md matrix rows 02/05/07 and ordering notes.
2. A-M1 -> gap-analysis.md in matrix Owns + Modify of 01 and 03; `scripts/compat/sglang-shim/` (only if CD14=B) in row 09.
3. A-M2 -> phase-03 criterion rewritten: greedy 32 tokens x 3 pinned prompts identical across two runs and first-token argmax equals the existing logit oracle (`scripts/gen-oracle-large.py`, `testdata/oracle/`); no phase-10 artifact, no token-equality swap test.
4. A-M3 -> steps in 03, 04, 09 (and 08 when it flips rows): regenerate and commit `docs/vllm-sglang-compat-matrix.md`; matrix rows 03/04/09/08 list it.
5. B-M1 -> gap-analysis.md:12 and phase-02 req 3: 'rejected at the 02 report stage (exit 2, before any download)'; no other occurrences.
6. B-M2 -> `docs/unverified-ledger.md` in phase-03 Modify + matrix row 03.
7. B-M3 -> min_tokens, stop_token_ids added to gap-analysis P(08) list; stop_token_ids added to phase-08 Overview.
8. B-M4 -> phase-04 wording reworded (no new dep; `prometheus` crate = new dep needing CD12 entry, not chosen); matrix row 04 names `helios-core/tests/metrics_hooks.rs` as home of e2e `metrics_scrape_during_smoke`.

**BLOCKERs this round:** 1 | **Round counts as clean:** no
**Exit state:** continue -> delta round 4, lens A only re-run; B carried PASS (r3) (0 BLOCKER)
