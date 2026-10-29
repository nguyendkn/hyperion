# Phase 10: First forward on sm90 (critical path)

## Context Links
- `reports/gpu-survey-h100-qwen3-0.6b.md` (H100 facts), `phase-01-*.md`, `phase-04-*.md` (oracle), `phase-05-*.md`; CLAUDE.md rules 1, 2, 6, 7 and the "CUDA FFI" decision
- Files: `helios/crates/helios-hal/src/backends/{hopper_sm90,blackwell_sm120}/`, `helios-core/src/scheduler.rs`, `ignition/.../compiler/architectures/qwen.py`

## Overview
Priority P1 (critical path). Status: core goal done on H100 (assembled forward matches the oracle, greedy 16/16 equals HF, chat served end to end); leftovers in Todo. Performance work moved to phase 12. Depends: 01. Wave: W2. Effort: ~8-10d (estimate, not measured; decisions 12 and 13 are decided). Goal: Hyperion logits for a small model on a real H100 match the HF oracle within a declared tolerance. Nothing else in the roadmap proves the engine computes anything: today the scheduler is a no-op and the sm90 backend is a 60-line skeleton.

## Key Insights
- [V] The CUDA Driver API code of the sm120 backend links and probes on H100 (`sm_90`); the Driver API is arch-independent, only kernels are not.
- [V] Environment: driver 580.126.20, CUDA 13.0 runtime, nvcc 12.4 in the devel image; `cuda-driver-sys` 0.3 linked. [V] Graph capture/replay works on CUDA 13 via a hand-declared `cuGraphInstantiateWithFlags` (the crate's `cuGraphInstantiate` is the legacy 5-arg form); 40 decode steps bit-identical to direct (phase 12).
- The first run must isolate risk: validate forward numerics only, not a spec change at the same time.

## Requirements
- Func: weights from the package load to device memory; embedding -> N x (GQA attention with KV cache, MLP) -> final norm -> logits for one prompt, batch 1, bf16; greedy next token; logits compared with the Python/HF oracle (phase 04 fixture format) at a tolerance declared in the oracle artifact BEFORE the run.
- Decided (plan.md 12, 13): first model = Qwen2.5-0.5B (add it to `testdata/models.json` with `compiler: qwen2`; files stay gitignored); sm90 path = shared CUDA Driver-API core + per-arch kernels, cuBLAS first, no TensorRT-LLM shim.
- Non-func: hardware-gated test (`*_hw`, `#[ignore]`) runnable on the staging H100 pod flow from the survey report; no hard-coded model dims (read `ModelConfig`); typed errors, no panics on device errors.

## Architecture
Shared driver core (context, streams, memory, module/graph handles) with per-arch kernel sets behind the existing `ComputeBackend` trait; `helios-core` unchanged (rule 1). The oracle comparison is a Rust `*_hw` test reading `specification/fixtures/logit-trace/` (phase 04). Exact module split is fixed in the ADR.

## Related Code Files
- Modify: helios-hal/src/backends/hopper_sm90/**; blackwell_sm120/** only if the shared core is extracted (coordinate with 05 by hand-off); helios-core/src/scheduler.rs (minimal real step loop); helios-cli/src/main.rs (load weights).
- Create: helios-hal/tests/forward_hw.rs, docs/h100-test-runbook.md (pod flow, deadline, cleanup), an append-only section in docs/unverified-ledger.md.

## Implementation Steps
1. Write the ADR under docs/decisions/ recording decisions 12 and 13 and the module split.
2. Land oracle fixtures for the chosen model (phase 04 step) with the tolerance.
3. Device core + one kernel path per op; test each op against a torch reference on the pod first (matmul, rmsnorm, rope, attention, mlp).
4. Assemble the forward; compare logits; iterate until within tolerance; record the measured numbers.

## Todo
- [x] decisions 12/13  [x] ADR (docs/decisions/0001-sm90-shared-driver-core.md)  [x] Qwen2.5-0.5B in testdata/models.json, compiles for sm_90  [x] Ignition real layer map (strict mapping, ignition/.../compiler/weights.py)  [x] oracle fixture (Qwen2.5-0.5B, tolerance 3.0 + argmax + top-5; scripts/gen-oracle.py)  [x] shared CUDA core + weight upload verified on H100 (helios-hal/src/cuda, weights_hw.rs)  [x] GEMM (cuBLAS) verified on H100 vs CPU reference  [x] RMSNorm kernel verified on H100 (PTX via Driver API)  [x] embedding, add, SwiGLU, RoPE kernels verified on H100  [x] attention kernel (prefill+decode, GQA, causal) verified on H100  [x] ASSEMBLED FORWARD matches the HF oracle on H100 (max diff 0.25, argmax Paris, top-5 5/5)  [x] greedy generation with a K/V cache equals HF generate (16/16 tokens)  [x] chat answered end to end by helios-cli on the H100 (real scheduler, optional `generate` trait method)  [x] forward  [x] logits match  [ ] device-resident KV cache + activations  [ ] temperature/top-p sampling on device (greedy on device is done in phase 12 step 2)  [ ] streaming  [ ] chat template  [ ] runbook (docs/h100-test-runbook.md)

## Success Criteria
- [RUN on H100] `cargo test -p helios-hal --features sm90 --test forward_hw -- --ignored` passes: max abs logit difference vs the oracle <= the tolerance in the oracle artifact; the test prints the measured difference.
- [RUN] Without a GPU the same test is skipped with an explicit ledger row; `cargo test --workspace --features stub` stays green.
- [RUN] `grep -rn "HEAD_DIM\|NUM_HEADS\|VOCAB_SIZE" helios/crates/helios-hal/src/backends/hopper_sm90` returns nothing.

## Test Strategy & Quality Gate
- Spec: Goal = numerically correct first forward. AC: Given the packaged model and a fixed prompt When forward runs on H100 Then logits match the oracle within tolerance; Given a non-H100 host Then the hardware test is skipped, not failed. Out of scope: batching, graphs, FP8, throughput.
- Pyramid 60/30/10 (per-op unit tests, forward integration, e2e). E2E: `forward_hw`, plus `helios-cli` serving one request on the pod if the scheduler step lands.
- Coverage: diff line >=90 / branch >=75 via `CMD-RUST-COV` for host-runnable code; device paths reported as [RUN on H100] evidence with the branch gate UNVERIFIED (ledger).
- Evidence: commands above on the pod flow from `docs/h100-test-runbook.md`; `cargo test --workspace --features stub`.

## ADR Rationale
Context: real compute must exist before scheduler/cache work means anything. Decision (user, 13): shared driver core + per-arch kernels, cuBLAS first. Why over a TensorRT-LLM shim (rejected): the arch-independent Driver API is already verified on H100, a single Rust binary is kept, no large C++ runtime; cost: more kernel work later. Revisit if cuBLAS-first cannot reach acceptable speed.

## Risk Assessment
Numerics drift (bf16 accumulation) -> tolerance declared before the run. Staging cluster access/quotas -> runbook with deadline + cleanup. CUDA 13 vs 12.4 mismatch -> record the verified combination.

## Security Considerations
Pods run in the user's namespace with a deadline; no secrets in manifests; weights and tokenizers are public HF files (record the revision); no inbound ports exposed.

## Next Steps
Unblocks real use of 07 (scheduler policy), 12 (decode performance, graphs, long context) and 09 (speculation). Out of scope here: multi-GPU, FP8/FP4, graphs and performance (phase 12).
