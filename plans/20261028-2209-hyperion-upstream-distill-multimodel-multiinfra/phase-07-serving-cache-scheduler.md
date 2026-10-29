# Phase 07: Serving (tree cache, scheduling shape, simulator, HTTP/SSE)

## Context Links
- research-03 sections 1-2,5,6; research-02 idea 6 (capacity-error text is owned by phase 01)
- Files: `helios-core/Cargo.toml`, `helios-core/src/{cache,scheduler,state,api}.rs`, new `helios-core/tests/sim/` (integration-test dir; no new crate), `helios-hal/src/cost_catalog.rs`, `helios-cli`

## Overview
Priority P2. Pending. Depends: 02, 03, 05; wave-order after 06 for helios-cli/Cargo.toml. Wave: W4 (consumes 05's `Capabilities` in step 3; edits scheduler/state/api after 02). Effort: 8d. Build the serving skeleton in types and CPU-testable logic. No scheduling POLICY tuned on guessed costs; policy value-scoring is phase 09.

## Key Insights
- Policy needs measured graph costs (GPU) - not available. Types, state, cache index, simulator and HTTP semantics are all CPU-testable now.
- Current RadixCache owns index + block ids with 256-token chunks (decision stands); chunk size becomes configurable and measured (decision 10).

## Requirements
1. `SchedulingShape` type (capacities, width, est cost) at the Scheduler boundary. Cost catalog = Rust-side table in new `helios-hal/src/cost_catalog.rs` keyed by `target_arch` (+ `tensor_format`), NOT a manifest field; entries are labelled synthetic seeds until real GPU measurements exist; injected into core as a value (core stays hardware-agnostic, rule 1). `cost_catalog.rs` is a per-arch seed table keyed by `target_arch`: new hardware adds rows, not directories (accepted deviation from rule 2's directory idiom; still no helios-core change).
2. Request phases add `Inbound`, `Restore`; two-step admission (plan without reserving, then admit); transfer tickets shared by N requests; cancel trivial.
3. RadixCache rewrite (index must be location-agnostic: the key never contains a device address; each node carries a residency field that today only holds `gpu`, so phase 11 adds tiers without redesigning the index; token ids stored per block and verified on hit): chained per-chunk hash of full prefix (exact endpoint), tree with parent link, pin counts, LRU tick; evict unpinned leaf first, else compact single-child ancestor. Cache identity = hash(model id, layout fingerprint, payload id, target_arch, tensor_format, domain tag); refuse cross-load. The research-03 idea 4 "state mode ordinary/spec" input is deliberately omitted (speculation skipped). State-row hook (decision 8): each node carries `Option<OpaqueStateRef>` the index never interprets; no trait change.
4. Decision trace + counters per scheduling decision (shape, estimated cost, reason); separate counters for exploration vs normal.
5. Offline scheduler simulator in `helios-core/tests/sim/` (`main.rs` + modules; run `cargo test -p helios-core --test sim`): workload generator + discrete-event sim with synthetic costs, cache-aware (restore cost, eviction); counterexample tests (starvation, head-of-line, prefill flood).
6. HTTP: terminal SSE error event after 200; no role chunk before first token; 429 on admission exhaustion; bounded body + content-encoding limits; NEW `GET /v1/models` advertising manifest capabilities; /metrics counters. (Pre-socket capacity error is phase 01's.)
7. Chunk-size measurement plan (below). Per decision 10 (updated recommendation, owner to confirm): measure 64-token hash blocks decoupled from the KV page size, not only equal sizes.
8. Scheduler output is a serializable `StepPlan` (phase 13 D4): graph/capacity id, metadata buffer contents, page-table deltas, sampling params. One rank (0) decides; every rank applies the same plan. With one GPU it is a local call.
9. Block lifecycle (idea from Dynamo KVBM): Reset -> Partial (private to its sequence) -> Complete -> Registered (shared, deduplicated by chained hash). Only Registered blocks are visible to other requests.
10. KV events for a cluster router (phase 13 D5): block stored/removed/cleared with monotonically increasing ids, chained hash with parent link, and the hash block size, so a KV-aware router (Dynamo) can track prefixes per engine. The prefix index is per engine, not per rank.

## Architecture
Scheduler (pure, deterministic given inputs) <- ComputeBackend only via trait; cache independent of device (HAL `allocate_kv_cache` unchanged). Sim reuses real Scheduler + RadixCache code with a virtual clock.

## Related Code Files
- Modify: cache.rs, scheduler.rs, state.rs, api.rs, lib.rs, helios-cli/src/main.rs (inject cost catalog; after 06), helios-hal/src/lib.rs (one `mod` line, after 05), helios-core/Cargo.toml (proptest dev-dep) + root Cargo.toml `[workspace.dependencies]` append, docs/unverified-ledger.md (07 section appended after 08)
- Create: helios-core/src/{shape,trace,admission}.rs, helios-hal/src/cost_catalog.rs, helios-core/tests/{cache_tree,cache_identity,sse_semantics,http_limits}.rs, helios-core/tests/sim/{main,workload,counterexamples}.rs

## Implementation Steps
1. Tests first for cache: hash-chain collision/ordering, pin/evict, identity refusal, property tests (proptest, justified dep).
2. Implement cache; keep old API shim for one commit only, then delete.
3. Shape + phases + admission with state-machine tests; cost-table row selected from `Capabilities` (arch + supported formats from 05) so unsupported (arch, format) pairs get no row and are refused (test: Capabilities without fp4 -> no fp4 row).
4. Simulator + 3 counterexample tests; trace/counters.
5. HTTP semantics against in-process server + stub token stream.
6. Chunk-size study: replay synthetic and, when available, recorded multi-turn traces through sim at 64/128/256; report hit rate vs index memory; default stays 256 until data says otherwise.

## Todo
- [ ] cache tree  [ ] identity  [ ] shape/phases  [ ] trace  [ ] sim  [ ] SSE/429/limits  [ ] chunk study

## Success Criteria
- [RUN] `cargo test -p helios-core` green incl. `--test sim` (3 named counterexample tests) and cache/HTTP tests on stub; deterministic sim output committed as test expectation.
- [RUN] chunk study over synthetic traces only; real-trace result [NEEDS-GPU/real traffic].
- [RUN] a grep-checkable rule: no phase doc or ledger row contains a throughput/latency number from this phase; sim numbers are labelled synthetic.

## Test Strategy & Quality Gate
- Spec: Goal = correct, observable serving semantics. AC: Given two requests sharing a 512-token prefix Then second hits 2 chunks; Given pinned node Then not evicted; Given cache id mismatch Then refuse; Given admission full Then 429; Given error after first byte Then terminal SSE `error` event. Out of scope: policy tuning, offload tiers, speculation.
- Pyramid 70/25/5. E2E: real HTTP server + stub backend: streaming request, forced mid-stream failure, assert event order and counters.
- Coverage diff >=90/75. Evidence: `cargo test -p helios-core`; phase-01 templates `CMD-RUST-COV` (`-p helios-core` replaces `--workspace --features stub`; `target/coverage/07/lcov.info`) and `CMD-DIFF` (Rust form); sim run `cargo test -p helios-core --test sim`.

## ADR Rationale
Context: current cache is a placeholder radix; no admission model. Decision: chained-hash tree + pin counts + typed phases now, policy later. Why over alternatives: keep radix of raw tokens (memory heavy, no identity), copy upstream scoring (needs measured costs; ideas-only), skip simulator (no proof on no-GPU host).

## Risk Assessment
Hash chaining collision: use 256-bit hash, verify chunk on hit. Sim drift from real timing: label synthetic. Scope creep into policy: forbidden before real costs.

## Security Considerations
Cache identity prevents cross-model/tampered reuse; no cross-tenant prefix sharing without identity equality; bounded body and encoded/decoded sizes (compression-bomb guard).

## Next Steps
Feeds 11 (tiered KV store: residency field, `Restore` phase and transfer tickets, simulator reused by its sub-phase A) and 13 (`StepPlan`, KV events).
Feeds 09 (speculation, value scoring). Out of scope: bearer auth (decide separately), Responses/Messages routes.
