# Phase 05: Infra breadth (capabilities, DeviceGroup, sm120 gaps)

## Context Links
- research-04 sec 2, ideas 6,7 (idea 12 deferred to phase 09, context only here); research-03 "better" (discrete HBM vs unified tier rule); CLAUDE.md rules 1,2,5-7
- Files: `helios-hal/src/{traits,triage,backend_factory}.rs`, `backends/**`, `ignition/hardware/{base,registry,sm90_hopper,sm120_blackwell}.py`

## Overview
Priority P2. Pending. Depends: 01, 03. Wave: W3. Effort: 5d. Capabilities supersede phase 01's `ProbedBackend` and single `supported_formats(arch)` table: 05 migrates that table into each backend's `capabilities()` and updates the cross-check test. New vendor/arch = new directory + recipe, no core change. Includes sm120 kernel/dims gaps (PDR section 9 items 5, 6; FR-15) and trait decoupling from `v1::Manifest` (own item; not in PDR section 9).

## Key Insights
- Upstream rejects backend abstraction; we keep the trait NARROW (alloc, kv alloc, graph/stream launch, capability probe). Share contract, not kernels. No "interchangeable kernels" goal.
- Multi-GPU and multi-node are owner requirements (phase 13). The rank-aware pieces here are no longer speculative.
- ONE trait change only: this phase's decoupling from `v1::Manifest`, the page-index KV design (phase 12 step 6, ADR 0002) and phase 13 D3 (per-rank backends, `DeviceGroup`, graph-capturable `Communicator`) land together. Changing the trait twice is the costly mistake to avoid.

## Requirements
1. `ComputeBackend::capabilities() -> Capabilities` (arch, formats, memory kind discrete/unified, max graph captures, peer_access flag, collective transport kind). Data, not model allowlists.
2. Unsupported ops return typed `HalError::Unsupported(op)`; never panic.
3. `DeviceGroup { ranks: Vec<RankInfo> }` (topology, NUMA node per GPU) built by the factory, plus a `Communicator` capability (all-reduce, all-gather, reduce-scatter, send/recv) enqueued on the rank's stream and graph-capturable (phase 13 D3). Single-rank default; helios-core sees ranks only. NCCL is loaded via dlopen (phase 13 D6); this phase defines the contract and the multi-rank stub, phase 13 M1 wires NCCL.
4. Trait takes a small `BackendConfig` struct (`load_model_weights(&self, &Path, &BackendConfig)`); conversion `BackendConfig::from(&Manifest)` lives in helios-hal (`backend_config.rs`). helios-cli is not edited: `create_backend` is the only cli-facing hal API and keeps its shape (additive only); `ProbedBackend` and `triage` are internal to helios-hal, so replacing them does not touch helios-cli.
5. sm120: read ModelConfig instead of hard-coded dims; fix PDR section 9 items 5 and 6 (each fix separately cited).
6. Ignition recipe per target declares `supported_formats` (cross-checked against Capabilities). No cost-catalog seed in recipes (YAGNI; the catalog is a Rust-side table owned by phase 07).
7. Adding hardware demo: a `backends/example`-free proof by test that registering a stub variant with different capabilities requires no edit under helios-core (compile-level test).

## Architecture
`backend_factory` iterates compiled-in backends (cfg features), asks each to probe, picks first match in order sm120 -> sm90 -> stub. DeviceGroup built by factory; collective trait optional (`Option<&dyn CollectiveBackend>`).

## Related Code Files
- Modify: helios-hal/Cargo.toml (only if a dep is needed), helios-hal/tests/data/format-table.json (regenerate from `Capabilities`), helios-hal/src/{lib,traits,triage,backend_factory}.rs, backends/blackwell_sm120/{backend,kernels,device,graph}.rs, backends/hopper_sm90/backend.rs, backends/stub.rs, helios-hal/tests/triage_matrix.rs (migrate to Capabilities), ignition/hardware/*.py, ignition/tests/test_format_table_crosscheck.py (migrate)
- Create: 05 ledger section appended to docs/unverified-ledger.md (after 08), helios-hal/src/{capabilities,device_group,backend_config}.rs, tests/capabilities.rs, tests/no_core_change.rs
- Do NOT touch: helios-cli (owned by 06 in W3)

## Implementation Steps
1. Tests first for Capabilities-driven triage and typed Unsupported.
2. Add capability + DeviceGroup types (no collective impl); stub reports multi-rank variants for tests.
3. Decouple trait from Manifest (adapter `BackendConfig::from(&Manifest)` in helios-hal; keep `create_backend` signature).
4. sm120 dims from config + PDR section 9 items 5,6 fixes, type-check only.
5. Update Ignition recipes + cross-check test from phase 01.

## Todo
- [ ] capabilities  [ ] DeviceGroup  [ ] trait decouple  [ ] sm120 items 5,6  [ ] recipes  [ ] ledger section appended

## Success Criteria
- [RUN] `cargo test -p helios-hal` >=6 new passing tests (capabilities-driven refusal, 2-rank group via trait, typed Unsupported, no_core_change compile test) and migrated triage matrix still >=12 cases; `cargo check --workspace` with helios-cli unmodified.
- [TYPE-CHECK] `cargo check -p helios-hal --features sm120`.
- [NEEDS-GPU] (ledger only, no executable claim here) sm120 kernel fixes behaviour, peer access, NVLink/PCIe transport.

## Test Strategy & Quality Gate
- Spec: Goal = hardware growth without core edits. AC: Given stub with capabilities {fp8} and fp4 manifest Then refuse; Given 2-rank stub group Then core sees 2 ranks via trait only; Given unsupported op Then typed error. Out of scope: real TP, collectives, second vendor.
- Pyramid 80/15/5. E2E: real `helios-cli` binary (unmodified by this phase, using 01's e2e test) still passes on stub; 2-rank group exercised via helios-hal integration test.
- Coverage diff >=90/75 (cfg-gated sm120 code excluded and listed in ledger as type-check only). Evidence: `cargo test --workspace --features stub`; phase-01 templates `CMD-RUST-COV` (`-p helios-hal` replaces `--workspace --features stub`; `target/coverage/05/`) and `CMD-DIFF` (Rust form); `cargo check -p helios-hal --features sm120`.

## ADR Rationale
Context: want many infrastructures; upstream chose none. Decision: narrow trait + capability data + optional collective, additive. Why over alternatives: thick abstraction (wide, unprovable), no abstraction (blocks goal, edits core per vendor), implementing TP now (unverifiable here, user decision 3/4).

## Risk Assessment
Abstraction unproven beyond two NVIDIA archs; keep honest in plan "beats upstream" claims. Trait churn: keep behind additive default methods.

## Security Considerations
Probe must not execute untrusted code; device memory zeroed on alloc where applicable ([NEEDS-GPU]).

## Next Steps
Feeds 07 (reads `Capabilities`), 13 (M1 uses `DeviceGroup` + `Communicator`), 11 (copy primitives). Out of scope: multi-host, RDMA.
