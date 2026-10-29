# Researcher-04: upstream infra breadth + quality system (ideas only)

Upstream = NVIDIA local-inference clone (product name "Flare"). Paths relative to clone `repo/`. [V]=read, [I]=inferred.
Note: TODO is docs/TODO.txt. Upstream crates named Helios/Ignition like ours; its CLI is "flare".

## 1. Component map
- `ignition/` [V] Python build-time: recipes/ (per model), components/ (reusable projections/experts/collectives), preparation/, direct/ (Python debug exec), signing.py, planning.py.
- `helios/` [V] Rust runtime lib: scheduler/, cache/, offload/, runtime_package/, frontend/, server/, grammar, native/ (C RoCE proxy), proto/ (Protobuf prepared contract), manifest.md, model-contract.md.
- `flare/` [V] Rust CLI binary: run, agent, orbit, inspect, cache cmds, cluster/fabric/trust/identity/package_cache/hub, windows.rs.
- `heliopause/` [V] Rust crate = PyO3 extension: Python<->Helios bridge, preparation allocator (VMM arenas), embedding tables, export.
- `selene/` [V] host cache tiers for prefix checkpoints: DRAM slabs + disk record files; no CUDA dep; seams `PageCopier`/`HostRegistrar` implemented by the GPU owner; composite store = write-through/read-through; never blocks model worker thread (enqueue + poll).
- `eos/` [V] host-OS abstraction (unix / windows incl. IoRing): positional file reads etc.
- `qualify/` [V] Python evidence/qualification helpers (graph replay, checkpoint state/http, host cache, multihost cache, perf traces, MTP/speculative, text oracle, provenance, C PCIe peer probe).
- `scripts/` [V] benchmark_*, qualify_* (server, worker failure, startup validation, prefill capacity, context limit), profile_*, repro_*; `scheduling_sim/`.
- `tests/` [V] pytest: cli/, components/ (many `*_cuda.py`), direct/, frontend/, preparation/, recipes/, reference/, server/, scheduling oracle/sim/utility tests, design-review hook test. `helios/tests` has Rust tests plus Python package fixtures.
- `packaging/debian` [V] .deb + Dockerfile; `.github/workflows/native-serving.yml` [V] Linux+Windows matrix.
- `docs/` [V] ~30 design docs (core-design, package-format, runtime-package, scheduling-theory/oracle/utility, prefix-cache-offload, signing, windows).

## 2. Scaling and hardware abstraction
- Same-process TP [V same-process-tp.md]: one Helios process, one CUDA context + one exec thread per rank; each rank owns fixed device arena + mapped host region; one graph per rank. Peer memory via CUDA VMM (create/map/set-access, fd export), not legacy IPC handles.
- Fixed-address trick [V]: preparation reserves every rank's arena at a deterministic per-rank base (1 TiB stride), so captured pointer tables stay valid at serve time; loader reserves ranges first, fails startup on collision. Loader validates GPU count/arch/rank layout/peer deps before serving; rank loading is staged and parallel.
- Preparation = Python rank processes; serving = Rust rank threads [V].
- Multi-host [V multi-host.md]: one TP rank per host (DGX Spark), rank 0 serves HTTP; collectives over RoCEv2 via native C CPU proxy; control + package payload over multi-stream TCP; rank-control packets capped (64 MiB), all ranks ack before replay. Package fixes rank count + GPU arch. Three launch modes: manual, agent-assisted, Orbit.
- RoCE ring [V roce-ring.md]: experimental, 2..16 ranks, neighbor-only queue pairs, host CPU relays non-neighbor traffic along shorter direction; collective math stays in GPU programs, routing/progress native. NIC hairpin forwarding not implemented. Also chain mode.
- Orbit [V orbit.md]: GPU-less controller (any CPU arch) + per-node agent (paired, encrypted, signed grants); placement, saved groups, attach/stop/status, package push/prestage. Unbuilt proposals [V orbit-proposals.md]: mDNS discovery, fabric check, prepare-through-agents, reconcile.
- Hardware abstraction: NONE by design [V AGENTS.md, design-review #4]: "no backend or device abstractions"; non-NVIDIA/CPU/interchangeable backends are rejected scope. Portability exists only for host OS (eos, Windows build) and host cache tiers (selene seams).

## 3. Quality system and process
- AGENTS.md [V]: task -> required-doc table; explicit REJECTION criteria (scope expansion, architectural violation, weakened guarantees, speculative abstraction, unfocused validation). Every change must trace to maintainer request / defect / measured problem / doc error. Reviewers told to recommend rejection, not rescue with opt-ins.
- Commit design review [V design-review.md, .githooks/commit-msg]: 15 principles; commit trailer carries reviewer name + 4 hex of sha256 of the staged checklist; hook verifies hash so the checklist must have been read in its current form. Doc admits it is an attention check, bypassable. Hook itself is tested (tests/test_design_review_hook.py).
- Invariants [V]: no per-model flags (`supports_*`) in shared runtime; contract is data not policy; env-var behavior switches banned (only device mask); new graph-op family needs owner approval; delete superseded loaders (no back-compat unless asked); versioned contracts rejected pre-load with a "re-export" message.
- Evidence rules [V]: perf comparisons only on exported package through the real server; debug-mode timings diagnostic only; keep raw measurements; do not produce audit reports/receipts/hash inventories; disclose unrun checks.
- Test layering [V names, I interpretation]: L0 CPU pytest + cargo tests (eos, selene, helios, flare) on Linux+Windows CI; L1 GPU tests opt-in by env var (`FLARE_RUN_CUDA_TESTS`, `FLARE_RUN_TP_CUDA_TESTS`, skipif, `*_cuda.py`, TP2/TP4/TP16 variants); L2 `qualify/` + `scripts/qualify_*` manual GPU qualification vs real checkpoints/oracles (exact graph execution, worker failure, startup validation, context limit); L3 benchmarks/profiling. [I] CI never runs GPU tests.
- docs/TODO.txt [V]: honest ledger of unvalidated-on-GPU paths per model/hardware (Windows offload, SM103, image prefill, a seed-repeatability bug).

## 4. Ideas for Hyperion
1. [adopt now] Explicit rejection-criteria list + task->required-doc table in repo docs. Module: CLAUDE.md, docs/.
2. [adopt now] "Unverified ledger" (docs/UNVERIFIED or TODO): per backend/feature, what is only type-checked or stub-run. Fits Windows no-GPU host and our verification-gate rule. Module: docs/.
3. [adopt now] 4-layer test model: L0 stub+schema+fixtures (runs on Windows CI), L1 GPU tests opt-in (`#[ignore]`/feature + env used only to gate tests), L2 qualification scripts, L3 bench. Module: Taskfile, helios-hal tests, ignition/tests.
4. [adopt now] Review principle: no per-arch/per-model boolean flags in helios-core; differences carried as manifest data (ModelConfig, target_arch, tensor_format already do this). Module: specification, helios-core.
5. [adopt now] Contract-version rejection with clear "re-export" error before load; complements exact target_arch triage. Module: helios-cli, triage.
6. [adopt now, cheap] Other vendors without rewrite: keep ComputeBackend narrow (alloc, kv alloc, graph/stream launch, capability probe); new vendor = new dir under backends/ + new Ignition recipe; unsupported ops return typed errors; capabilities are data, not model allowlists. Kernels stay per-backend; share only contract. Do not chase "interchangeable kernels" (upstream's reason for rejecting). Module: helios-hal, ignition/hardware.
7. [adopt later] Multi-GPU TP: per-rank context/thread, fixed per-rank arenas for graph capture, peer memory mapping. In Hyperion put it behind HAL: `DeviceGroup`/rank + optional `CollectiveBackend` capability (peer access, transport kind reported by probe: NVLink on Hopper, PCIe on workstation Blackwell). helios-core sees ranks only. Add additive manifest fields (`tp_size`, rank layout) BEFORE v1 freezes. Module: helios-hal (new trait), specification, triage.
8. [adopt later] Split host/disk cache tier into CUDA-free crate with copier/registrar seams (selene-like), testable on Windows; matches planned LMCache-style crate. Module: new crate, helios-core cache.rs.
9. [adopt later] Host-OS crate (positional read, io_uring/IoRing) for bundle/weight loading; aligns with helios-pkg. Module: helios-pkg.
10. [adopt later] Multi-host skeleton: package fixes world size + arch; rank 0 serves API; `--rank/--rendezvous` shape; control plane (TCP) separate from data plane. Module: helios-cli.
11. [adopt later] Staged-diff design-review hash trailer + hook; worth it once multiple agents/contributors commit. Module: .githooks.
12. [adopt later] Per-target_arch fixture/qualification set in specification/fixtures (Hopper vs Blackwell expected formats), extending commit 2c40cf7's supported_formats checks.
13. [skip] Orbit controller/agent orchestration, mDNS, RoCE ring/chain CPU relay, ConnectX specifics, Debian/Docker packaging, signed agent grants: cluster orchestration, untestable hardware.
14. [skip] Upstream's "no backend abstraction" stance itself (our differentiator) — keep its discipline only.

## 5. Risks
- Scope creep: upstream rejects multi-backend, cluster orchestration, multi-model hosting; our "more infra" goal pulls the opposite way. Do seams and fields first, one real feature at a time, each tied to a request (matches our "ask before scope change").
- Unverifiable here [I]: VMM peer mapping, graph capture at fixed addresses, NVLink/PCIe collectives, sm90 recipes, RDMA, numerical/perf claims. Only type-check, stub, schema and fixtures run on this host.
- Fixing multi-GPU fields in spec v1 with no multi-GPU run risks wrong shape; use additive optional fields and log in ledger.
- Process ceremony (15 principles, hash hook) may exceed value for small team.
- Upstream is a self-described research prototype [V README]; docs claims unrun by us.
- HARD RULE: ideas only; no reuse of upstream code, hook, proto, allocator text.

## 6. Unresolved
- Is multi-GPU TP in Hyperion's near-term scope, or only the abstraction (decides 7, 10)?
- Add `world_size`/rank fields to spec before v1 freezes?
- Which second vendor, if any, is a real target for the trait, or stop at Hopper+Blackwell?
- Windows+Linux CI matrix mirroring L0?
- Not read: core-design.md, scheduling-*.md, runtime-package-design.md, qualify internals, helios/tests contents (names only).
