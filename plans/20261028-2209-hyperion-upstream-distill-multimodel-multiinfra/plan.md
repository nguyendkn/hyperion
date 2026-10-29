---
title: "Hyperion upstream distill: multi-model, multi-infra roadmap"
description: "Finish single-GPU path, make every contract rank-aware for multi-GPU and multi-node H100, then widen models, infra, package trust, serving and quality; ideas from upstream only, no copied code."
status: pending
priority: P1
effort: ~40-50d for 01-08 + 10 + 12; phase 13 ~4-6d design + ~15-25d implementation; phase 11 ~10-14d, entry-gated (13 phases, see table)
lane: high-risk
branch: main
tags: [roadmap, multi-model, multi-infra, spec-v1, helios-pkg, scheduler, quality, decode-perf, kv-tiers, multi-gpu, multi-node]
created: 2026-10-28
---

# Hyperion: distill upstream, widen models + infra

Scope (owner, 2026-10-29): Hyperion targets many H100s per node and many nodes; one GPU is the first milestone, not the scope. Contracts that are expensive to change later (manifest, backend trait, scheduler output, KV layout and hashing, process model) become rank-aware now (phase 13).

Rule: IDEAS ONLY. No upstream code/text copied; phases cite only our files + `research/` reports. No mocks on main. Host = Windows, no GPU: every success criterion is tagged `[RUN]` (executes here) or `[TYPE-CHECK]` / `[NEEDS-GPU]` (ledger, phase 08). Research: `research/researcher-01..04-*.md`. Gaps cite `docs/project-overview-pdr.md` section 9 items / FR ids; "P0..P3" in phase names are the earlier review's labels (no on-disk source).

## Phases
| # | Phase | Depends | Wave | Status | Effort |
|---|-------|---------|------|--------|--------|
| 01 | [Finish P0 + H100-survey fixes: factory/triage, main.rs, tokenizers, architectures check, tooling](phase-01-finish-p0-single-gpu-path.md) | - | W1 | done (3294d70; see reports/phase-01-report.md for the later factory `shutdown()` fix) | 4d |
| 10 | [First forward on sm90: logits match HF oracle on H100](phase-10-first-forward-sm90.md) | 01 (decisions 12,13 decided) | GPU | forward + oracle done on H100; leftovers in its Todo | ~8-10d |
| 12 | [sm90 decode performance + long context (graph, attention, kernels, dlopen, KV design, tiled prefill)](phase-12-sm90-decode-performance.md) | 10 | GPU | in progress (steps 0, 2, 3, 3a, 3b done; next pod: 3a+, 4, 5; then 7 before K) | ~8-12d left |
| 13 | [Multi-GPU + multi-node foundation: rank-aware contracts (D1-D6), TP on one node, then multi-node](phase-13-multi-gpu-multi-node.md) | D-items: 01 (beside 12); M1: D1-D3 + 12 step 4 | GPU + Windows | pending (D-items can start now) | ~4-6d design + ~15-25d impl |
| 02 | [P1 core correctness: scheduler/state/api errors/tests](phase-02-p1-core-correctness.md) | 01 | W2 | pending | 4d |
| 03 | [Spec pre-freeze additive fields (decision gate)](phase-03-spec-prefreeze-additive-fields.md) | 01 | W2 | pending | 3d |
| 04 | [Model breadth: declaration, mapping, logit oracle](phase-04-model-breadth.md) | 03 | W3 | pending | 6d |
| 05 | [Infra breadth: capabilities, DeviceGroup, sm120 gaps](phase-05-infra-breadth.md) | 01,03 | W3 | pending | 5d |
| 06 | [Package + signing: helios-pkg, type-state loader](phase-06-package-signing.md) | 03 | W3 | pending | 6d |
| 07 | [Serving: tree cache, scheduling shape, sim, HTTP/SSE](phase-07-serving-cache-scheduler.md) | 02,03,05 (+06 order) | W4 | pending | 8d |
| 08 | [Quality system: rules, ledger, 4-layer tests, CI](phase-08-quality-system.md) | - | W1 | pending | 3d |
| 09 | [LATER backlog with entry criteria](phase-09-later-backlog.md) | 04-07 + real GPU | W5 | pending | n/a |
| 11 | [Tiered KV store: GPU HBM, host RAM, NVMe](phase-11-tiered-kv-store.md) | 07, 12 (steps 6, 7, 8); entry criteria in file | W5 | gated (only sub-phase A may run, after 07 sim exists) | ~10-14d |

Phase 13 design items D1-D6 run on Windows beside phase 12 and land through their owners (D2 via 03, D3 via 05 together with 12 step 6 as ONE trait change, D4/D5 via 07); its milestones M1-M4 run on multi-GPU pods and multi-node clusters. GPU track (10, 12) runs on H100 pods beside the waves, never in parallel with phase 05 on `helios-hal/src/backends/**` (see hand-offs). 11 is W5 and entry-gated; its sub-phase A (CPU simulation) may start once 07's simulator exists.

07 is W4: consumes 05 `Capabilities` (picks the cost-table row), builds on 02 scheduler/state/api edits, edits helios-cli after 06 (file order only; payload id is a 03 field). Ledger: 08 (W1) creates `docs/unverified-ledger.md`; only 01 (same wave) hands rows to 08 via its report (08 step 0); 02..07 each append their own named section after 08 landed (append-only).

## Parallel Execution Matrix
01 -> {02, 03, 05, 10}; 10 -> 12; 12 (step 6 ADR) -> {05 trait change, 11}; 12 (steps 7, 8) -> 11; 07 -> 11; 12 -> 09; 13 D2 -> 03; {05, 12 step 6, 13 D3} -> one trait change; 13 D4/D5 -> 07; {13 D1-D3, 12 step 4} -> 13 M1 -> M2 -> {M3, M4}; 13 -> 11 (per-rank tiers); 13 -> 09 (EP, CP, disaggregation); 03 -> {04, 05, 06, 07}; 02 -> 07; 05 -> 07; 04..07 -> 09. Globs are exclusive within a wave.

| Wave | Phase: owned globs |
|------|--------------------|
| W1 | 01: helios-hal/{src/backend_factory.rs,src/triage.rs,src/backends/*/ (probe fn only),build.rs,tests/}, helios-cli/** , helios-core/src/api.rs (`/health` only), helios-hal/Cargo.toml (`cc` -> build-deps only if used), helios-hal/tests/data/format-table.json (committed dump, written by a hal test), Taskfile.yml, root Cargo.toml, .gitignore (coverage lines), ignition/pyproject.toml, ignition/tests/test_format_table_crosscheck.py. 08: CLAUDE.md, docs/**, .github/**, LICENSE (root, only after open decision 11 is answered) |
| W2 | 10: helios-hal/src/backends/hopper_sm90/**, helios-hal/tests/forward_hw.rs, docs/h100-test-runbook.md, docs/unverified-ledger.md (append-only section); shared-core extraction touching 05 files goes by hand-off. 02: helios-core/src/{state,scheduler,tokenizer,error,lib,api}.rs, helios-core/tests/{api_errors,shutdown_drain}.rs, helios-cli/src/main.rs (drain wiring only), docs/unverified-ledger.md (append-only section). 03: specification/**, ignition/.../manifest_schema.py, ignition/tests/test_manifest.py, docs/decisions/ ADR (new file), docs/spec-freeze-checklist.md (new), docs/unverified-ledger.md (append-only section) |
| W3 | 04: ignition/.../{compiler/**,cli.py}, ignition/pyproject.toml, its ignition/tests files, specification/fixtures/logit-trace/**, docs/add-a-model.md, docs/unverified-ledger.md (append-only section). 05: helios-hal/Cargo.toml, helios-hal/src/{lib,traits,triage,backend_factory,capabilities,device_group,backend_config}.rs, helios-hal/tests/data/format-table.json (migrated), helios-hal/src/backends/**, helios-hal/tests/**, ignition/.../hardware/**, ignition/tests/test_format_table_crosscheck.py, docs/unverified-ledger.md (append-only section). 06: helios/crates/helios-pkg/** (new; own Cargo.toml incl. `trybuild` dev-dep), helios-cli/**, root Cargo.toml `[workspace.dependencies]` append-only, ignition/.../{signing,signing_cli}.py, ignition/tests/test_signing_roundtrip.py, specification/fixtures/hostile/**, docs/trust-model.md, docs/unverified-ledger.md (append-only section) |
| W4 | 07: helios-core/Cargo.toml (proptest dev-dep), helios-core/src/{cache,scheduler,state,api,lib,shape,trace,admission}.rs, helios-core/tests/** (incl. `tests/sim/`), helios-hal/src/{cost_catalog.rs, lib.rs mod line}, helios-cli/src/main.rs, root Cargo.toml dep append (proptest), docs/unverified-ledger.md (append-only section) |
| GPU track | 10: see W2 row (done). 12: helios-hal/src/cuda/**, helios-hal/src/backends/hopper_sm90/**, blackwell_sm120/** (step 1 only, by hand-off with 05), kernel sources + build script, helios-hal/tests/*_hw.rs, docs/decisions/0002-kv-virtual-range.md (new), docs/unverified-ledger.md (append-only section) |
| Phase 13 | D-items: docs/decisions/ ADRs for D1, D3-D6 (new files) and the multi-rank stub tests under helios-hal/tests/ (new files); all code edits to specification, helios-hal traits/factory, helios-core scheduler/cache and helios-cli go through the owning phase by hand-off. M-items: helios-hal/src/comm/** (new, NCCL via dlopen), sharded load path in helios-pkg by hand-off with 06, docs/multi-gpu-runbook.md (new), docs/unverified-ledger.md (append-only section) |
| W5 | 09: none (entry-gated). 11: helios/crates/helios-kvstore/** (new), helios-core/tests/sim/tiers.rs (new), docs/decisions/0003-tiered-kv-store.md (new), docs/kv-tiers.md (new), docs/unverified-ledger.md (append-only section); edits to helios-hal traits/cuda, helios-core scheduler/cache and helios-cli flags only by hand-off after 05, 07, 12 |

Rules (ownership, stated once):
- **Cross-wave hand-offs (sequenced, never parallel):** 01 -> 02 (api.rs, helios-cli main.rs); 01 -> 05 (triage/factory/stub, cross-check test); 02 -> 07 (scheduler/state/api); 05 -> 07 (helios-hal lib.rs); 06 -> 07 (helios-cli main.rs, Cargo.toml); 13 -> {03, 05, 06, 07} (design items land through their owners); 12 <-> 05 (helios-hal/src/backends/** and cuda/**: whichever starts second waits for the other's commit; the KV trait change from 12's ADR 0002 is implemented once, in 05 or a follow-up); 05, 07, 12 -> 11 (traits/cuda copy primitives, scheduler admission + cache residency + sim, cli flags).
- **cli.py** (Ignition) belongs to 04. 06 adds sign/keygen in new `signing_cli.py` with its own console script `hyperion-sign`; 04 adds in `pyproject.toml` the `[project.scripts]` line AND the dependency `cryptography` (Ed25519; one audited wheel-distributed lib, avoids PyNaCl's libsodium build) (documented hand-off; inert until 06 lands). Recorded in 06 ADR. Per-crate Cargo.toml belongs to the phase owning the crate (06 adds `trybuild` dev-dep to helios-pkg).
- **05 never edits helios-cli:** trait decoupling adapter (`BackendConfig::from(&Manifest)`) lives in helios-hal; `create_backend` is the only cli-facing hal API and stays additive-only so helios-cli compiles untouched; `ProbedBackend`/`triage` are internal to helios-hal (01 introduces, 05 supersedes). 06 owns the helios-cli load path in W3.
- **Taskfile + root Cargo.toml:** 01 owns. Fixed task names declared by 01: `check`, `test`, `schema:check`, `check:sm120`; 08 only references them. 01 sets workspace `members` to include `helios/crates/*`, so new crates need no root edit; later phases only append deps.
- **docs/:** 08 owns structure + existing files. Other phases add NEW uniquely named docs files or append a section named for their phase after 08 landed; no phase edits another's section. `docs/unverified-ledger.md`: created by 08; 01 hands rows to 08 (report); 02-07 (W2-W4) append named sections only (the one allowed shared file, merge in phase order).
- **Evidence command templates (SSOT):** defined once in phase 01 (`CMD-RUST-COV`, `CMD-PY-COV`, `CMD-DIFF`); other phases cite them by name and may only add scope args.
- **Deliberate research overrides:** no seed/cost catalog in the signed manifest (research-03 idea 1) and no reserved schema section (research-01 idea 12): the catalog needs measurements that do not exist yet, so no blind reservation; revisit when 07 has measurements (an additive optional field is still possible before freeze).
- Spec edits in W3+ go through 03 (additive PR only).
- Every evidence command has a cargo/pytest form that runs without `task`; `task ...` is optional convenience.

## Where Hyperion beats upstream (and honest limits)
- Hopper SM90 + Blackwell SM120 behind one trait (upstream: no backend abstraction, SM103/120/121 only); Rust-owned schema + committed JSON Schema + two-sided round-trip tests.
- Offline-first pinned-key trust (air-gapped OK); default weight-hash scan with cached verified marker.
- No-GPU testability: stub backend, Python/CPU logit oracle, offline scheduler sim, hostile-package fixtures.
- Cost catalog (Rust-side table in helios-hal keyed by `target_arch`, NOT a manifest field) + cache identity keyed by `target_arch` + `tensor_format`; one scheduler serves two cost curves.
- Measured on H100 (phase 10/12): forward matches the HF oracle; greedy 16/16 equals HF generate; decode graph replay bit-identical to direct; split-K attention 6x faster at ~8k context.
- Tiered KV store (phase 11) designed around one location-agnostic index and cost-based restore-vs-recompute, gated on measured need.
- Graphs are captured at load, so one canonical package can serve TP 1/2/4/8 by sharding at load (upstream packages fix the rank count). NCCL (graph-capturable collectives, NVLink/NVSwitch/InfiniBand) instead of custom transports; a KV-event stream compatible with Dynamo's KV-aware router for the cluster layer.
- Honest risks: scope creep (upstream far broader, 11 recipes, multi-host); perf/numeric claims hold only for what ran on the H100 (single GPU, batch 1, Qwen2.5-0.5B); scheduling policy (07), tiers (11) and the 09 backlog stay gated on measurements.

## Open decisions (USER-CHALLENGE; not decided here)
Gate: phase 03 blocks on 1,3,5,6,7 (asked once there; 04/06 read the ADR). Phase 04 also states 2. Phase 08 asks 9,11. Phase 12 step 6 asks 17. Phase 11 asks 14,15,16,18 before sub-phase B. Phase 13 asks 19-22 (20 before phase 03 merges its parallelism fields).
1. Descriptor-driven generic decoder vs per-arch Python? Rec: per-arch Python, descriptor fields grown additively; revisit at 3rd arch. Consequence: descriptor-first = bigger Helios runtime, needs CPU oracle; phase 04 would be re-planned.
2. CPU reference backend/oracle in scope? Rec: Python oracle (HF) fixtures only now; Rust CPU backend deferred. Consequence: no Rust-side numeric check before GPU.
3. Multi-GPU TP: abstraction only vs real; do parallelism fields enter v1 now? UPDATED (owner scope 2026-10-29): real (phase 13); add the sharding plan, supported TP sizes and optional `deployment {tp_size, pp_size}` to v1 now (phase 03 req 4). Consequence: late fields after freeze force v2.
4. Second vendor for the trait? Rec: none; keep trait narrow. Consequence: generality unproven.
5. Publisher identity: key pinning vs repo-based vs CA? Rec: pinned keys + local trust store. Consequence: no revocation feed.
6. Weights as safetensors payload files vs custom framing? Rec: safetensors. Consequence: custom = more parser surface.
7. One deployment per bundle vs many (arch x TP)? Rec: one package per target arch; TP size chosen at load from canonical weights (decision 20), so no per-TP bundles. Consequence: many = larger signing groups.
8. Hybrid/recurrent models in scope? Rec: no; phase 07 cache keeps only an opaque state-row hook (no trait change). Consequence: yes => state checkpoints in 05/07.
9. Windows+Linux CI matrix? Rec: yes, L0 only. Consequence: CI cost, catches path bugs.
10. Cache chunk 256 vs 64? Rec (updated 2026-10-29): decouple them: hash blocks of 64 tokens for prefix matching, KV pages stay larger (256, configurable) for memory and transfers; measure in the 07 sim. Consequence: finer reuse without small pages; index holds 4x more entries.
11. Make "ideas only" a CLAUDE.md rule + add LICENSE? Rec: rule yes; LICENSE needs owner's choice.
12. DECIDED (user, 2026-10-28): phase 10 first model = Qwen2.5-0.5B (Qwen2 arch, no spec change). Qwen3 follows once forward numerics are proven (needs QK-norm in `ModelConfig` + a compiler).
13. DECIDED (user, 2026-10-28): sm90 = shared CUDA Driver-API core + per-arch kernels, cuBLAS first; no TensorRT-LLM shim. Consequence: architecture change vs original plan; more kernel work later.
14. Phase 11: L3 (NVMe) persistence across restarts? Rec: default off, opt-in via `--kv-disk-path`. Consequence: on = resume sessions after restart, needs identity + integrity checks at startup.
15. Phase 11: encryption at rest for L3? Rec: not now; files 0600 + purge command; revisit before multi-user deployments. Consequence: on-disk KV is readable by the service account.
16. Phase 11: explicit session hint in the API? Rec: only if sub-phase A shows implicit LRU + hit counts are not enough. Consequence: non-OpenAI request field.
17. Phase 12 step 6 / phase 11: page-major KV layout (all layers of a page contiguous)? Rec: yes, if attention cost is unchanged when measured. Consequence: no = per-layer batched copies for every tier move.
18. Phase 11: share one host KV cache across engine processes on a multi-GPU node? Rec: per-rank NUMA-local tiers inside one engine first (needed by phase 13 anyway); cross-process sharing only for several data-parallel engines on one node. Consequence: yes = out-of-process store, shared memory, cross-process identity.
19. Phase 13: cluster layer = NVIDIA Dynamo worker vs built-in router? Rec: Dynamo first (engine-agnostic router, planner, KVBM, NIXL; custom engines publish KV events). Consequence: built-in = we own routing and autoscaling.
20. Phase 13: shard at load from canonical weights (one package for all TP sizes) vs pre-sharded packages per TP size? Rec: shard at load. Consequence: pre-sharded = faster load, one package per TP size.
21. Phase 13: first 70B-class target model for M2 (8 KV heads for TP8)? Owner picks by workload.
22. Phase 13: collectives = NCCL only vs NCCL + custom small-message all-reduce? Rec: NCCL only until M2 measures the all-reduce share. Consequence: custom = latency gain at decode, more kernel code.

## Unresolved questions
- Coverage tools (cargo-llvm-cov + nightly llvm-tools, diff-cover) are installed by phase 01; if Rust branch data cannot be produced here the gate says 'branch gate UNVERIFIED on this host' and records it per the ledger rule.
- Upstream recipe sizes/tolerances unread; logprob K vs Hopper graphs deferred to 09.
- Decode batching has a GPU side (phase 12 step 8) and a scheduler side (phase 07) but no detailed plan yet; phase 11 needs both.
- Multi-GPU hardware access for phase 13: a 2xH100 pod (M1), a full 8xH100 node (M2), two nodes with InfiniBand (M3/M4). Not yet arranged.
- Phase 12: 8k-context per-token time not yet reliable (needs the 3a+ in-run timing); PTX grew 72 KB -> 683 KB (startup cost unmeasured; step 5).
- Phase 10 oracle tolerance is 3.0 max abs logit difference while the measured difference is 0.25; tightening it is an owner call (never loosen).

## Automatic Validation

**Rounds run:** 5 (deep r1 + 4 delta)
**Outcome:** CAP REACHED, no open BLOCKER. High-risk exit rule (2 consecutive clean rounds) NOT met: r3 clean, r4 BLOCKER (ledger routing, fixed), r5 clean. Not declared CLEAN.

### Round Log
| Round | Type | Structural | Consistency | Fidelity | Testability | BLOCKERs | Clean |
|---|---|---|---|---|---|---|---|
| 1 | deep | 3B/3m | 3B/12m | 3B/5m | 4B/5m | 13 | no |
| 2 | delta | PASS (3m) | PASS (10m) | 1B/1m | 1B/5m | 2 | no |
| 3 | delta | PASS (4m) | carried PASS (r2) | PASS (1m) | PASS (5m) | 0 | yes |
| 4 | delta | 1B/3m | carried | carried | carried | 1 | no |
| 5 | delta | PASS (1m, fixed) | carried | carried | carried | 0 | yes |

Full round artifacts: `reports/validation-round-*.md` (r5 recorded here only). Residual risk: B/C/D lenses last ran in r3; r4/r5 fixes touched only ledger/ownership wording.

Post-validation edits (2026-10-28, NOT covered by rounds 1-5): phase 10 added; phase 01 gained the H100-survey items (tokenizers bump, architectures check, backend-arch triage); open decisions 12, 13 added.
Post-validation edits (2026-10-29, NOT covered by rounds 1-5): phase 11 (tiered KV store) and phase 12 (decode performance, absorbing the optimization task steps 0-6 plus tiled prefill and batch buckets) added; GPU track defined; phase 07 gained the location-agnostic index requirement; phase 09 rows promoted to 11/12; decision 10 recommendation updated; decisions 14-18 added; statuses of 01, 10 updated.
Post-validation edits (2026-10-29, later, NOT covered by rounds 1-5): owner scope widened to multi-GPU and multi-node; phase 13 added; decisions 3, 7, 18 updated; 19-22 added; phases 03, 05, 06, 07, 09, 11, 12 updated (parallelism fields, one merged trait change, StepPlan, KV events, KVBM block lifecycle, per-path transfer queues, per-rank tiers, speculation and parallelism backlog rows, phase 12 progress 3a/3b).
