# Phase 12: sm90 decode performance and long context

## Context Links
- Follows phase 10 (first forward matches the HF oracle on H100).
- Origin: the "optimize Hyperion's sm90 decode path" task (steps 0-6). This file is now the single source for that work.
- Upstream ideas only, no copied code (spark-local-inference): `docs/core-design.md` ("Runtime contract", "Capacities and memory"), `docs/runtime-package-design.md`, `helios/src/os.rs`, `helios/src/cuda.rs`, `helios/src/runtime_package/driver.rs`.
- CLAUDE.md rules 1, 2, 6, 7; ADR 0001 (shared driver core).

## Overview
Priority P1. Status: in progress (steps 0, 2, 3, 3a, 3b done). Depends: 10. Track: GPU track (runs on H100 pods; see plan.md). Remaining effort: ~8-12d (estimate, not measured).
Goal: make batch-1 decode fast at short and long context, keep graph replay bit-identical to direct execution, and remove the 8192-token limit.

## Measured so far (single H100, batch 1, Qwen2.5-0.5B, release build)
- **Old 9.51 / 10.27 ms numbers were measured with a debug build; discard them.**
- **Step 0 baseline (b2a6c2d):** direct 2.980 ms, graph 2.288 ms per token at context 5-133.
  - Direct is host-launch bound: ~533 launches per step.
  - Kernel time 1.66 ms per step: cuBLAS GEMV 46.5%, attention 14.1%, bias_cast 12.2%, rmsnorm 10.5%.
- **Step 2, tokens chained on GPU (79788ba):** direct 2.417 ms, graph 1.797 ms at short context. 40 tokens identical to host argmax in both modes.
- **Step 3, split-K attention (63c010a).** Graph ms/token:

  | Context | Before | After |
  |---|---|---|
  | 5-141 | 1.72 | 2.36 |
  | 1024 | 4.05 | 2.74 |
  | 4096 | 10.87 | 2.94 |
  | 7900 | 20.59 | 3.41 |

  - Short context regressed. Two graph measurements there differed by 19% (1.98 vs 2.36), so the measurement protocol must be fixed first (step 3a).
  - Attention at 8k still takes ~1.66 ms per step vs a ~0.03 ms bandwidth floor for ~97 MB of K/V.
- **Step 3a, interleaved measurement (6a43022):** 10 samples after warmup, median with min/max; SM clock steady at 1980 MHz. Cross-process noise is still large: differences under ~10% cannot be told apart, and some samples under 1 ms are artifacts of the (long run - short run) formula.
- **Register spills (ptxas -v):** hypothesis refuted. Old split-K kernel 66 registers, 0 spill; new kernel 0 spill (126-168 registers). Short-context slowdown came from the K loop reading 2-byte elements with a runtime trip count, so memory loads serialized.
- **Early exit for idle splits (b61318c):** no measurable change (B/A ~1.00 at 5 and 1024); kept because it removes work.
- **Step 3b, attention kernel tuning (d418851):** head_dim as a template, 16-byte K/V loads issued together, each V row read by 8-16 threads, accumulators in registers, combine weights computed once per block. Correctness unchanged: 0 error vs float64 for lengths 1-32768, 5 runs bit-identical, logits within 0.25 (tolerance 0.5), 18/18 hardware tests.

  Graph ms/token, median [min-max] of 8 interleaved samples:

  | Context | Split-K (step 3) | Tuned (3b) | One block per head (reference) |
  |---|---|---|---|
  | 5 | 2.36 | 1.75 | 1.72 |
  | 1024 | 2.57 | 1.81 | 4.20 |
  | 4096 | 2.96 | 1.93 [1.90-1.94] | - |
  | 7900 | 3.38 | 2.85 [0.70-4.54] | - |

  The short-context regression is gone; long context is 25-35% faster. The 4096 number is solid; the 1024 and 7900 ranges are too wide to trust.
- **New risk:** the PTX file grew from 72 KB to 683 KB, so driver JIT at load may slow startup (not measured yet).

## Requirements (remaining steps, in order)
- **3a+. Measurement protocol, second iteration (first iteration done).**
  - Within ONE run, after warmup, time 128 chained decode steps with cuEvents at start and end; divide by 128. No subtraction of two noisy runs, and prefill is excluded.
  - Compare A/B in the same process where possible.
  - Report median, p10 and p90 of 10 measurements; record SM and memory clocks.
  - Measure startup time once now, as the baseline for step 5.
- **3b. Attention tuning (DONE, d418851; kept here for reference).**
  - Check register spills without ncu (`nvcc -Xptxas -v` or `cuobjdump --dump-resource-usage`).
  - Splits beyond `ceil(len/128)` exit early; combine only over active splits; single-split writes the result directly.
  - Compile-time head_dim and query-heads-per-KV-head.
  - 16-byte loads; more warps per block.

  Correctness rules from step 3 still apply.
- **4. Graph hygiene.**
  - `cuGraphInstantiateWithFlags` + `cuGraphUpload` at load.
  - No synchronize after each replay.
  - Fixed cuBLAS workspace.
  - Enforce "after Ready: no allocation, compilation or capture".
- **K. Kernel-count reduction (same pod round as 4).**
  - Concatenate Q/K/V and gate/up weights at load (backend only, no spec change).
  - Bias in the cuBLASLt epilogue (removes bias_cast).
  - Fused residual add + rmsnorm.
  - A custom GEMV only if these are not enough.
- **1. Driver loader via dlopen/LoadLibrary** (hand-declared, no new crate):
  - remove `cuda-driver-sys`;
  - move sm120 onto the shared `cuda/` module;
  - builds for sm90/sm120 succeed on Windows without a toolkit.
- **5. Reproducible kernel binaries.** SASS for sm_90 (cubin or fatbin with SASS + PTX); build script records the nvcc version; committed binaries checked against sources. Instantiate templates only for the head_dim x query-group combinations real manifests need; others stay refused with a typed error. Needs nvcc, so it runs on the pod together with step 4.
- **6. KV memory design (DESIGN ONLY, then stop).**
  - `cuMemGetInfo` after weights load (closes the KV-vs-VRAM ledger row).
  - `docs/decisions/0002-kv-virtual-range.md`: one reserved virtual range, page-index addressing, reserve + driver margin, required trait change.
  - Must answer open decision 17 (page-major layout, input to phase 11).
  - The trait change itself is implemented ONCE, together with phase 05 decoupling and phase 13 D3 (per-rank backends, DeviceGroup, Communicator). Step 4 changes only backend internals, never the trait.
- **7. Tiled prefill beyond 8192 tokens**, so long contexts run end to end (today the 32768 test is kernel-level only and longer prompts are refused). Prefill chunks are fixed capacities (powers of two from 128 to 4096), one graph each, so prefill stays graph-capturable and matches `StepPlan` and multi-rank execution (phase 13).
- **8. Decode batch buckets 1/2/4 (GPU side), with inactive rows masked.** A graph per bucket; a missing bucket fails with a typed error. The scheduler side belongs to phase 07. Plan in detail after steps 4-7.

## Architecture
- Everything stays inside the shared CUDA core and the sm90 kernel set behind `ComputeBackend`; helios-core is unchanged (rule 1).
- Variable decode state (positions, lengths, next token) lives in fixed device buffers read by kernels; the host updates them before replay.
- Graph per capacity. Determinism: deterministic plans give identical logits; tolerances come only from arithmetic plus measured repeatability.

## Related Code Files
- Modify: `helios-hal/src/cuda/**`, `helios-hal/src/backends/hopper_sm90/**`, `blackwell_sm120/**` (step 1 only), kernel sources + build script, `helios-hal/tests/*_hw.rs`.
- Create: `docs/decisions/0002-kv-virtual-range.md`; an append-only section in `docs/unverified-ledger.md`.
- Hand-off: phase 05 owns `helios-hal/src/backends/**` in W3. Phase 12 and phase 05 never run in parallel on those files (plan.md rule).

## Implementation Steps
1. Done: 0, 2, 3, 3a, 3b.
2. Windows now (no GPU): step 1 (dlopen); 3a+ harness code.
3. Next H100 pod round: 3a+ measurements, step 4, step 5 (needs nvcc). One commit per item.
4. Step 7 (tiled prefill) right after step 4, BEFORE K (owner decision 2026-10-29: long context is the real target).
5. K (kernel-count reduction).
6. Step 6 design together with phase 13 D3; stop for the owner.
7. Plan step 8 (batch buckets).

## Todo
- [x] 0 profile
- [x] 2 tokens chained on GPU
- [x] 3 split-K attention
- [x] 3a protocol (first iteration)
- [ ] 3a+ in-run cuEvent timing
- [x] 3b attention tuning
- [ ] 4 graph hygiene
- [ ] K kernel-count reduction
- [ ] 1 dlopen
- [ ] 5 SASS
- [ ] 6 ADR 0002
- [ ] 7 tiled prefill
- [ ] 8 batch buckets

## Success Criteria
- [RUN on H100] Graph replay stays bit-identical to direct at every step. HF oracle and greedy 16/16 still pass.
- [RUN on H100] Per-token time table (direct and graph) at 5-141, 1024, 4096 and 8k, plus 32k after step 7, measured with the 3a protocol.
- [RUN] Windows build with sm90/sm120 features needs no CUDA toolkit (after step 1); `cargo test --workspace --features stub` green.
- [RUN on H100] After Ready, no allocation, compilation or capture (assertion or test).

Targets, not promises (estimates); latest measured: 1.75 ms at context 5, 1.93 ms at 4096, 8k not yet reliable:
- short context <= ~1.8 ms/token (graph);
- 8k ~2.0-2.5 ms/token;
- practical floor without deeper fusion ~0.8-1.0 ms/token.

## Test Strategy & Quality Gate
- Correctness first: bit-identity graph vs direct; split-K tolerance vs a float64 reference, declared before the run; 5 repeated runs bit-identical; long-context checks compare logits, not tokens.
- Evidence: release build, 3a protocol, nsys for kernel-time vs wall-time; ncu needs node profiling permission (ERR_NVGPUCTRPERM today; ledger).

## Risk Assessment
- **Measurement noise hides 10-20% changes.** Mitigation: 3a before any tuning claim.
- **Kernel work grows unbounded.** Mitigation: stop at the targets; deeper fusion or adopting FlashInfer kernels is an owner decision.
- **Trait churn between steps 6/8 and phase 05.** Mitigation: ADR first, a single trait change.

## Security Considerations
Pods in the user's namespace with a deadline and cleanup (`docs/h100-test-runbook.md`); no secrets; public HF weights with recorded revision.

## Next Steps
- Step 6's ADR is merged with phase 13 D3 and phase 05 into one trait change, and feeds phase 11 (decision 17; Dynamo KVBM uses the same page-major "fully contiguous" block layout).
- Step 7 is a prerequisite of phase 11.
- Step 8 feeds phase 07 scheduling.
