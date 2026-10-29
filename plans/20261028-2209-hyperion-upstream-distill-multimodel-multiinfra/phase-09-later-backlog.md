# Phase 09: LATER backlog with entry criteria

## Context Links
- research-03 ideas 8-16; research-02 ideas 7-13; research-04 ideas 7-10, 9, 12; research-01 ideas 4,6,10,12

## Overview
Priority P3. Pending. Depends: 04-07 + real GPU. Wave: W5. Effort: n/a. NOTHING here starts before real logits and a real GPU path exist. Each item = its own future plan with its own gate.

## Key Insights
- Every item below was judged unverifiable or premature on a no-GPU host; entry criteria are what turns each into a plan.
- Whole-forward graph capture exports a section into the bundle; the manifest reserves nothing now and the cost catalog is a Rust-side table (phase 07), not a manifest field. Adding a section later is a spec decision at that time (v2 if v1 has frozen).

## Requirements
Each backlog row names: item, measurable entry criteria, touched modules. Rows are added, never silently dropped.

## Backlog (item | entry criteria | touch)
| Item | Entry criteria | Touch |
|------|----------------|-------|
| Tiered cache offload (host/disk, CUDA-free crate with copier/registrar seams) | PROMOTED to [phase 11](phase-11-tiered-kv-store.md); entry criteria live there | helios-kvstore (new), helios-core cache |
| Offload isolation rules (bounded queues, skip-and-count, low-prio copy stream) | MERGED into phase 11 (low-priority copy stream, measured decode-interference bound) | helios-hal |
| Speculative decoding as state transaction (draft/verify/accept/commit) | real logits, sampler, verify op in trait, draft source | trait, scheduler, cache |
| Copy/history (n-gram) drafting (FIRST speculation method: no training, fits agent workloads with repeated tool output and code) | verified decode path + batch buckets (phase 12 step 8) + device sampler | helios-core |
| DSpark-style block drafting + confidence-scheduled verification (ideas only; DeepSpec is MIT) | a published or trained drafter for the target model; n-gram drafting already measured | ignition, hal, scheduler |
| Value/time scoring with fairness prices | measured per-shape graph costs on both archs | scheduler |
| Helios-vs-oracle logit comparison (`*_hw` L1 test over phase 04 trace) | a backend forward exists to call | helios-hal tests |
| Per-layer residual comparison | working backend kernels + phase 04 trace | oracle, hal |
| Whole-forward graph capture per capacity bucket exported in bundle | kernels exist; spec decision on a manifest section (not reserved by 03). Note: load-time capture of the decode graph already runs on H100 (phase 12); this row is only about EXPORTING graphs in the bundle | ignition, spec |
| Real multi-GPU TP (per-rank arenas, peer mapping, collectives) | PROMOTED to [phase 13](phase-13-multi-gpu-multi-node.md) (M1, M2) | hal, pkg, core |
| Multi-host skeleton (rank 0 serves, control vs data plane; research-04 idea 10) | PROMOTED to [phase 13](phase-13-multi-gpu-multi-node.md) (D1, M3, M4) | helios-cli |
| Expert parallel (MoE) | an MoE target model; phase 13 M2 done | hal, ignition, scheduler |
| Context parallel (ring attention) for very long contexts | measured need beyond one GPU's attention at target lengths; phase 13 M2 done | hal, scheduler |
| Disaggregated prefill/decode with KV transfer (NIXL, via Dynamo) | phase 13 M3 done; measured prefill/decode interference | hal, kvstore, cluster layer |
| Peer-to-peer weight transfer between replicas (ModelExpress idea) | many replicas of one model; cold start measured as a problem | helios-pkg |
| Host-OS crate (positional read, io_uring/IoRing) for bundle/weight loading (research-04 idea 9) | helios-pkg load path measured as the bottleneck; reuse the I/O layer phase 11 builds for its NVMe tier if it exists | helios-pkg |
| Per-target_arch fixture/qualification set (Hopper vs Blackwell expected formats; research-04 idea 12, deferred here; phase 05 cites it as context only) | second recipe arch has real qualified checkpoints | specification/fixtures |
| Prompt-boundary endpoints (message-end marks for cache cuts; research-01 idea 10) | chat serving + prefix cache working; template engine in Rust | helios-core tokenizer/cache |
| Signature groups, CAS cache with hardlinks, chunk hashes pre-decompress | multiple deployments share weights / compressed blobs | helios-pkg |
| Logprobs (raw log-softmax semantics, K fixed at export) | real sampler; decide K vs graph capture | ignition, hal, api |
| Components layer (shared attention/MoE/norm/rope) | 3rd real architecture lands | ignition |
| Hybrid/recurrent state rows | decision 8 = yes | trait, cache |
| Staged-diff review hash hook | multiple committers | .githooks |
| Direct dlopen of CUDA driver instead of link-time toolkit | PROMOTED to [phase 12](phase-12-sm90-decode-performance.md) step 1 | hal build |

## Explicitly skipped (revisit only on request)
Anthropic/Responses/Ollama routes, Orbit-style cluster controller, RoCE relay, mDNS, S3 uploader, .deb/Docker packaging, FP4 compressed-scale layout (Hopper never FP4), DP oracle, per-layer fusion lists.

## Architecture
N/A: no code. Each item will carry its own Architecture section when promoted to a plan.

## Related Code Files
N/A: entry-gated; modules are listed per row in the Touch column.

## Implementation Steps
1. At each real-GPU milestone, re-read the table and re-check entry criteria with measured evidence.
2. Promote an item by creating its own plan (lane high-risk, full gate) citing the satisfied criteria.
3. Add newly distilled ideas as rows; never delete without a recorded reason.

## Todo
- [ ] review after phases 01-07 and first real-logit run  [ ] re-check each row's entry criteria

## Success Criteria
- Verifiable entry criterion: phase 09 may start only when (a) phases 01-07 are done per their criteria, (b) a real-GPU bundle has produced logits that pass the phase 04 oracle comparison, and (c) the unverified ledger shows no open [NEEDS-GPU] row for the item's prerequisites. Any item failing its own row criteria stays backlog.

## Test Strategy & Quality Gate
N/A until an item enters its own plan (each will carry full gate; lane high-risk). Entry-criteria review replaces tests here.

## ADR Rationale
Context: temptation to build ahead of evidence. Decision: gate each item on measurable entry criteria. Why over alternatives: building speculatively (YAGNI, unverifiable on this host), dropping items (loses distilled ideas).

## Risk Assessment
Backlog rots; revisit at each real-GPU milestone. Entry criteria themselves need review once GPU exists.

## Security Considerations
N/A for this phase (no code). Items touching trust or network (multi-host, CAS cache, host-OS I/O, offload to disk) must carry a Security section in their own plan before entry.

## Next Steps
Review after phases 01-07 complete and first real-logit run exists.
