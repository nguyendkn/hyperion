# Validation Round 4
**Lane:** high-risk | **Round type:** delta | **Lenses run:** A | **Scope:** fixes from r3 + touched artifacts
## Lens Results
### 4.A Structural — 1 BLOCKER / 3 MINOR
- BLOCKER: 02/03 (W2) hand ledger rows to 08 (W1) which is already finished; no routing path.
- MINOR: phase-08:16 "updated by every phase" contradicts append model.
- MINOR: phase-08 creates optional root LICENSE not covered by its W1 matrix row.
- MINOR: stale `.githooks/**` owned by 08 (plan.md:37, phase-08:5) though no hook is created.

## Fixes Applied
1. Ledger routing gap -> 02 and 03 now append their own named sections to docs/unverified-ledger.md (listed in their Related Code Files and in the W2 matrix rows); only 01 hands rows to 08 (08 step 0). plan.md line 30 and docs/ bullet rewritten to one rule; phase-08:22 and phase-01 hand-off wording aligned; phase-05 todo "ledger rows to 08" -> "ledger section appended". Grep for 'hand rows to 08', 'next phase touching', 'never edits docs', '01/02/03': 0 hits. 'Hand-off to 08' only in phase-01.
2. phase-08:16 -> "updated by phases 02-07 (append-only named sections) and seeded by 08 from 01's hand-off". Grep 'every phase': 0 hits.
3. LICENSE -> added to 08's W1 matrix row as "LICENSE (root, only after open decision 11 is answered)"; phase-08 Files and Create lines say owned by 08, only after decision 11. Grep LICENSE: consistent.
4. `.githooks` -> removed from plan.md W1 row and phase-08:5. Grep '.githooks': only phase-09 backlog row (legit, deferred item).

**BLOCKERs this round:** 1 | **Round counts as clean:** no
**Exit state:** continue -> round 5 (final allowed; Structural re-run)
