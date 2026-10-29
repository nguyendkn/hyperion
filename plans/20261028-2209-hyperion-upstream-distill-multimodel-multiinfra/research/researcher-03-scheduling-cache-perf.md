# Researcher-03: scheduling, KV/prefix cache, speculation, perf (ideas only)

Source: NVIDIA local-inference clone (docs/*.md, scripts/scheduling_sim/). Own-words summary, no code copied. [V]=read, [I]=inferred. Read depth: heads of docs (~first 6-9 KB each) + targeted sections; not whole files.

## 1) Scheduling model

- [V] Unit of choice = a prepared, non-preemptible graph (a "quantum") with fixed prefill capacity, decode rows, verify width, duration. Scheduler picks graph + which requests fill lanes; cannot pick arbitrary time slice. (docs/scheduling-theory.md)
- [V] Baseline objective: expected useful work per unit time. Useful = consumed uncached prompt tokens + beta * expected output tokens; padding, inactive rows, rejected drafts, restored prefix = zero. Pure prefill, pure decode, mixed graphs all compete on one score; no hard chunk-size or duration ceiling. beta is explicit input, never inferred.
- [V] Expected output for spec width k = 1 + sum of cumulative conditional acceptance products. Capped by remaining output budget; EOS unknown.
- [V] Known failure modes of greedy value/time (they wrote executable counterexamples in tests/test_scheduling_sim.py): (a) picks 128-chunk then needs another graph for 129-token prompt; (b) backlogged decode starves prefill; (c) round-robin tie-break gives uneven token share (10,10,10,5,5). So pure throughput score is not fairness/latency.
- [V] Fix layer (docs/scheduling-utility.md): proportional-fair utility. Per request, log of token-rate (progress) and log of delivery-rate (visible bursts count once regardless of token count). Gradient gives dynamic "prices" per prefill token / output token / delivery. Rate history = exponential decay by elapsed time (not per launch, so splitting launches composes); normalisers rescale stored history so changing normaliser does not reprice service.
- [V] Decode-age continuation term: nonpreemptible graph delays every other decoder; cost of waiting grows quadratically with age of last visible output. Reference interval sigma derived from fastest batch-feasible delivery rate.
- [V] Fairness entity = request, not graph slot. CPU CFS/EEVDF used only as accounting analogy; doc explicitly warns batched cost is non-additive (2 decodes 15ms each, batch 17ms) so per-request cost split is convention.
- [V] Runtime (docs/scheduling-runtime-design.md): Rust files helios/src/scheduler/{policy,service,utility,cost}.rs. Timings are measured per complete program (proposal+verify+commit+collectives+stalls); GPU replay and serial host handoff estimated separately. Funded exploration: bounded budget for timing refresh and acceptance learning, ledgers never double-charge one execution, counters per execution kind. Unknown timings -> "discovery" semantics. Plan trace records physical shape, cost, observation age, selection reason, exploration credit.
- [V] Validation offline (all Python, scripts/scheduling_sim/: simulator, scheduler, utility, oracle, oracle_sweep, timing, learning*, workloads, analysis): discrete-event simulator up to ~16 requests; exact DP oracle over completion episodes for tiny workloads (enumerates cohorts, priors on output length, state-limit error, no approximate fallback); oracle compares objectives J = E[R - rho*T] incl. first-token cost, completion cost, age cost; policy-improvement prototype uses continuation value; timing-learning sim with hidden costs. Doc stresses Python results do not prove native or GPU behavior.
- [V] Admission/cache/restore/pipelined submission excluded from sim; handled in the runtime design.

## 2) KV and prefix cache (docs/prefix-cache-offload.md)

- [V] Checkpoint = immutable page chain + one "state row" (recurrent state, conv history, partial tail page, last hidden, consumed count) per rank. Model-agnostic: recipe declares bytes; storage moves bytes without interpreting. No physical ids inside; restore into any free ids; row self-validates consumed count.
- [V] Tiers behind one `PrefixStore` trait (lookup/store/fetch/poll/stats/close): memory (pinned), disk (direct I/O through bounded staging), composite (write-through, disk read warms memory), or none. Integrated-memory boxes (Spark): disk-only to avoid duplicating one physical pool. GPUDirect Storage deferred.
- [V] Index: tree keyed by exact-endpoint key (not a flat map); buckets keyed by digest of the aligned 64-token boundary at or below the endpoint. Lookup walks aligned boundaries longest-to-shortest, one small hash per probe, checks parent chain feasibility to GPU ancestor; probe count ~ new tokens / 64. Single-writer, lock-free (owned by worker thread). [V]
- [V] Chained/prefix hashing: keys come from a history object giving key at any consumed count (the hash covers the token prefix). Page content determined by prefix ending at page => host chain suffix composes with GPU ancestor pages. Missing parent => store as root record with full chain so descendants stay restorable.
- [V] Eviction: LRU unpinned leaf; if all pinned, compact LRU single-child ancestor (drop row, keep pages as page-only). Eviction rules differ per store.
- [V] Isolation / namespace: layout fingerprint (16-byte hash of rank count, page tokens, plane names+sizes) + 32-byte identity (model identity + continuation-state mode: ordinary / learned-spec / copy-spec) in a domain-separated hash. Ordinary and speculative records cannot cross-load even with equal layout. Corruption: payload checksums, tests reject corrupted payloads.
- [V] Scheduler integration: new `Inbound` request phase never scheduled; two-step admission (plan without reserving, then admit with optional inbound). Scheduler owns transfer tickets (not requests) -> cancel is trivial, N requests share one transfer, ticket pins GPU ancestor. `take_publications()` emits GPU-ready checkpoints once each after a completion fence.
- [V] Policy: offload every publication immediately (unless exists / no slabs); fetch only if estimated transfer time < half the estimated prefill time of tokens gained (rates start from constants, replaced by measured). Never offload live state or in-flight spec rounds.
- [V] Interference control: model stream never waits on offload; copy batches bounded (16 MiB); low-priority streams; transfer threads at batch scheduling class + positive nice; direct I/O (no page-cache pollution); all pools fixed at startup; all queues bounded, skip-and-count never block. Honest note: stream priority may not isolate copy engines; measure per-token latency.
- [V] Defaults stay off until correctness + latency thresholds pass (section 4). Section 2.14 ranks on separate hosts, 2.13 reusable CPU input records.

## 3) Speculation / MTP as state transaction (docs/mtp-design.md)

- [V] Round = Draft -> Verify (K+1 distributions kept, each row sees only its causal prefix) -> Accept (prefix + correction/bonus) -> Commit (clip at EOS/budget, select matching target state, advance histories once, repair draft history from ACTUAL target hidden states). Unaccepted rows private and unreachable.
- [V] Counts distinct: consumed, generated, visible. Emitting only d[0] consumes p, not d[0]. Terminal token stays pending. Device terminal flag suppresses later queued rounds.
- [V] Recurrent/linear-attention states need a per-input-row checkpoint (or correction records) then accepted-prefix select; conv/PLE histories keep all candidates until acceptance. Attention KV may physically hold rejected rows but logical length/tags must exclude them; rejected writes must never touch an immutable shared page; reserve private suffix pages (incl. boundary pages) before submission.
- [V] Prefix publication under speculation: checkpoint at P holds draft-cache entries for P-1 only (pair with x[P] not yet known); alignment of draft cache differs from target at 64-boundaries; publish only at actually accepted consumed counts. Draft-proposal state is approximate -> never reuse it as history.
- [V] Spec graph identified by 4 extents: request capacity B, depth D, per-request width W, total rows P. Host-allocated (exact query lengths) vs device-allocated (GPU confidence frontier within ceilings; reserve for ceilings). Width-one programs compete as ordinary decode and keep draft ctx. Acceptance estimates per source with priors, aging (half-life 128 tokens), optimistic bonus only for funded learning.
- [V] Copy drafting: ring of last 65,536 committed generated tokens across requests; 8-gram then 4-gram suffix match, up to 7 continuation tokens; only validated outputs enter ring. Default off.
- [V] Sampling correctness uses chain speculative sampling with explicit device RNG (draft probs [B,K,V], target [B,K+1,V]). Acceptance stays on device.
- [V] Defect case (docs/glm-batched-speculation-defect.md): batched spec on Spark corrupted answers because export zeroed a scratch tensor a kernel read; only bit batch>16 rows; reproducible by script across cached/uncached; fixed. Lesson: a packaging step zeroed "scratch" that was really input; tests must cover batched spec through the exported artifact, not same-process.

## 4) Performance techniques and measurement (docs/performance-work.md, direct-execution.md)

- [V] Baseline discipline: same checkpoint, GPU, precision, graph capacity, sampling, cache state, completed work. Correctness failure blocks perf claim. Separate GPU replay time / native scheduling+submit / HTTP latency.
- [V] Measure complete GPU graph duration (incl. vocab projection, collectives, sampling) across all ranks; kernel timelines at batch 1 and 4 with per-operator attribution; host launch traces alone cannot attribute. Spec rounds/s = delta rounds / delta GPU-ns counters.
- [V] Perf comparisons only via exported package + production runtime; debug/direct/eager path is fault bisect only (direct mode separates scheduler/transport/replay faults from model faults; eager vs replay vs no-compile ladder).
- [V] Techniques: fuse residual+norm; pack projections at load; preserve FP32 stats and rounding points; compile whole forward before capture; autotune per captured shape (P rows decides small-row vs grouped MoE kernel band); parallel-branch overlap with disjoint scratch (aux streams, only up to small row caps); poison reusable storage after last use to catch lifetime bugs; allocation stability under repeated replay; compare compiled vs uncompiled logits over 256 steps.
- [V] Never infer speedup from draft depth or component benchmark; measure every captured spec shape.

## 5) Ideas for Hyperion

Legend: tag | reason | prerequisites | module.

1. [adopt now] Model the unit of scheduling as a "prepared plan/step shape" (capacities, width, measured cost) in the Scheduler trait boundary, not as ad-hoc token budgets. Reason: shapes the skeleton before logic accretes. Prereq: none (types only). Module: helios-core/scheduler.rs + manifest ModelConfig/graph catalog field in specification.
2. [adopt now] Request-phase `Inbound`/`Restore` and two-step admission (plan without reserving, then admit); scheduler-owned transfer tickets so N requests share one fetch and cancel is trivial. Prereq: none for the state/phase types; transfer real only later. Module: helios-core scheduler.rs, cache.rs, state.rs.
3. [adopt now] Prefix key = exact-endpoint hash of the full token prefix chained per chunk, tree of nodes with parent link, pin count, LRU tick; evict unpinned leaf, else compact single-child ancestor keeping pages. Our 256-token chunks fit; their 64-token probe granularity is a finer bucket (we could keep a coarse bucket map at chunk boundaries and verify with one chunk hash). Prereq: none (CPU only, unit-testable). Module: helios-core/cache.rs RadixCache.
4. [adopt now] Cache namespace/identity = hash(model identity, layout fingerprint, state mode ordinary/spec, tensor_format, target_arch) with domain separation; refuse cross-load. Fits our "manifest records target_arch and tensor_format" rule. Prereq: manifest fields exist. Module: specification (ModelConfig/manifest), cache.rs key.
5. [adopt now] Checkpoint = pages + opaque state row (recipe-defined bytes), no model-specific code in storage; physical ids never stored. Needed for hybrid/recurrent models later; keep index agnostic. Prereq: HAL allocate_kv_cache. Module: helios-hal ComputeBackend trait (state-row API), cache.rs.
6. [adopt now] Counters and trace per scheduling decision (shape, estimated cost, reason) plus separate counters for exploration vs normal executions. Prereq: none. Module: helios-core scheduler.rs, api.rs metrics.
7. [adopt now] Offline scheduler simulator in Rust or Python in repo (workload generator + discrete-event sim + counterexample tests). Their three counterexamples are cheap regression tests for any policy we write. Prereq: none (timings synthetic). Module: new tests under helios-core, or ignition/tests-like sim dir.
8. [adopt later] Value/time score with explicit beta and fairness prices (PF utility, exp-decay rate history, decode-age term). Reason: only meaningful with measured graph costs. Prereq: real backend step timings (GPU), real prefill/decode graphs. Module: helios-core scheduler.rs + cost estimator fed by HAL timing hooks.
9. [adopt later] Tiered PrefixStore trait (memory/disk/composite), with fetch-only-if-transfer < half prefill time and measured rates. Prereq: working GPU cache + pinned host alloc in HAL + prefix hits measured. Module: new crate (matches our "LMCache-style sidecar later") + HAL.
10. [adopt later] Offload isolation rules (bounded queues, skip-and-count, never block model stream, low-priority copy stream, direct I/O). Prereq: GPU streams; SM90/SM120 copy engine test. Module: helios-hal backends.
11. [adopt later] Speculation as state transaction (draft/verify/accept/commit; counts consumed vs generated vs visible; private suffix pages reserved before submit; publish prefix only at accepted counts). Prereq: real logits, a draft model or MTP weights, backend verify op, sampler. Module: ComputeBackend trait (verify/commit), scheduler, cache.rs (private suffix pages).
12. [adopt later] Copy/history drafting (n-gram ring over committed output). No weights needed; cheap first speculation win. Prereq: real token output + verification path. Module: helios-core.
13. [adopt later] Per-shape cost measurement + qualify harness (whole-graph GPU time incl. collectives, per-rank interval, kernel timelines, counters to compute rounds/s). Prereq: GPU. Module: helios-cli bench / helios-hal.
14. [adopt later] Poisoning reusable scratch after last use and compiled vs eager logit comparison as correctness gate. Prereq: GPU, Ignition graphs. Module: Ignition + backends.
15. [skip] Device-side confidence allocation of target rows, funded acceptance learning with optimistic bonus, parallel branch overlap, per-layer fusion lists. Reason: depend on kernel stack and models we do not have; revisit after baseline decode works.
16. [skip] Exact DP oracle. Reason: research tool, high effort; use only if policy disputes arise.
17. [skip] Their b12x/FlashInfer-specific operator contracts, and NVFP4 recipe details (Hopper has no FP4 per our rule 7).

## 6) Where Hyperion can be better

- [I] Hopper path: they target SM120/121 (consumer/Spark/GB300); Hopper H100/H200 with FP8 + large HBM3 and NVLink is a different cost curve; a hardware-keyed cost catalog and recipes per target_arch (which we already have) lets one scheduler serve both. Their docs mention GB10 integrated memory special case; we can generalise as a tier-selection rule from probe (discrete HBM vs unified).
- [I] Their scheduling stack is single-graph-in-flight; we could design for pipelined submission (2 in flight) from the start with the cost model accounting overlap.
- [I] Their prefix index is single-writer, per-process; our Rust runtime can make read path lock-free with the same state-machine approach already decided, giving cheap multi-threaded API reads.
- [I] Signed bundle: we can put the cache namespace identity and cost catalog seed in the signed manifest so a cache cannot be loaded against a tampered model; they derive identity from package at load.
- [I] Schema contract (Rust source of truth + round-trip tests) already stronger than their protobuf+Python approach for graph catalog; extend it to the cost catalog and scheduling config.
- [I] Chunk size 256 reduces index memory and probes versus their 64; trade-off: lower hit granularity. Make chunk size a measured manifest/config parameter, evaluate hit-rate on real multi-turn traces.
- [I] Beat them on honesty-of-validation: their sim is Python-only and admits excluded cache/admission effects; add cache-aware simulation (restore cost, eviction) to our sim.

## 7) Unresolved questions

- Their core-design.md (cache ownership, graph lifetime, performance acceptance thresholds) was not read; exact acceptance numbers unknown.
- Sections 2.4-2.6 (record file format, slabs, checksums) and 2.14 (multi-host) of prefix-cache-offload not read in detail.
- Did not read learning*.py or the full oracle results; the quality/gain of the PF+age policy versus simple baselines in their sims is not verified by me.
- Whether their tier-rate constants (20/3/10 GB/s, 8k tok/s) are reasonable for H100 hosts: unknown.
- Does Hyperion keep recurrent/hybrid architectures in scope? Decides whether state-row checkpoints are needed now or only page chains (pure attention).
- Is multi-GPU/TP in scope for Helios? Their checkpoints span ranks; affects cache key and ticket design.
- License/upstream-ideas-only rule honoured: nothing copied; only file paths cited.
