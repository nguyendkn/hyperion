# Decode performance log

Every number here is **single H100 80GB HBM3 (sm_90), driver 580.126.20, nvcc 12.4, batch 1, Qwen2.5-0.5B, context 5 to 133 tokens, greedy**, and nothing more general. Release builds only.

Protocol (from step 3b on): per-token time = GPU time between two CUDA events placed around a window of 128 decode steps inside one run (after three warm-up steps, so the prompt and the graph capture are outside), divided by 128; direct launches and graph replay alternate inside one process, 10 windows each, median with p10 and p90 (`HopperBackend::generate_timed`, `weights_hw.rs::measure_modes`). Host time over the same window is reported next to it. The figures of steps 0 to 3b below were taken as (136-token run minus 8-token run) / 128, which subtracts two noisy times and turned out to have a spread of tens of percent; they are kept as history, and the table in the last section replaces them. In-process GPU time per group of kernels comes from CUDA events (`weights_hw.rs::profile_decode_steps`, `HopperBackend::generate_profiled`); event marks add a little overhead, so group times are indicative. Kernel and gap numbers come from Nsight Systems (`nsys`, `--trace=cuda`, graph mode with `--cuda-graph-trace=node`).

| Step | Direct ms/token | Graph ms/token | Notes |
|---|---|---|---|
| (earlier, debug build: not a valid performance number) | 9.51 | 10.27 | `cargo test` without `--release`; host work in a debug build is far slower |
| 0. baseline, release | 2.980 | 2.288 (1.30x) | graph is faster than direct |
| 2. greedy token chain on the device (argmax kernel, 4-byte token ids to the host) | 2.417 | 1.797 (1.35x) | 15 of 15 hardware tests pass; tokens identical to the host argmax in both modes |
| 3. split-K decode attention | 3.103 | 1.979 (1.57x) | 17 of 17 hardware tests pass; slower than step 2 at short context (see below), 6x faster at 8k |
| 3b. vectorised split-K attention (see below) | 2.39 (context 5) | 1.75 (context 5) | 18 of 18 hardware tests pass; no short-context loss; graph 1.93 ms at 4096 and 2.85 at 7900 |

## Step 0 profile (release, 128 decode steps)

Direct launches issue about 533 kernels per step. Kernel-time sum is 1.66 ms per step in both modes.

| Where | Direct | Graph |
|---|---|---|
| Kernel-time sum per step (nsys) | 1.660 ms | 1.658 ms |
| GPU idle between consecutive kernels inside a step (nsys) | 1.993 ms over 584 gaps | 0.147 ms over 559 gaps |
| Host issue time per step (in-process) | 2.338 ms | 0.021 ms |
| Host blocked in synchronisation | 0.923 ms | 1.654 ms |
| Logits copy to host + conversion | 0.447 ms | 0.368 ms |
| Host argmax over 151936 logits | 0.118 ms | 0.109 ms |

Kernel time by kernel (direct run, share of the 1.66 ms): cuBLAS GEMV kernels (`gemvx` + `gemv2T`, used for the one-token linear layers) 46.5%, decode attention 14.1% (about 9.9 us per call on average at these context lengths), `bias_cast_bf16` 12.2% (168 launches per step), `rmsnorm_bf16` 10.5%, the lm_head GEMM 5.7% (94 us, about 0.27 GB of weights, close to the bandwidth bound), RoPE 3.9%, residual add 3.0%, SwiGLU 1.9%, KV append 1.5%.

Reading (inference from the numbers above): the direct path is bound by the host issuing about 533 launches per step; a graph removes those gaps and the step becomes about 1.66 ms of GPU work plus about 0.5 ms of logits copy and host argmax. The kernels are small and latency-bound (average about 3 us each): the cuBLAS GEMV calls move 1 to 2 MB each in 4 to 6 us, far below the memory-bandwidth bound. Attention is not the bottleneck at context 5 to 133 tokens; how it scales with long contexts has not been measured.

Not measured: occupancy and achieved bandwidth of any kernel (`ncu` was installed but the pod lacks permission to read GPU performance counters: `ERR_NVGPUCTRPERM`), HTTP latency, contexts beyond 133 tokens.

## Step 2: decode time grows with the context (release, same machine, batch 1, 136 tokens generated after the prompt)

Measured with `weights_hw.rs::decode_time_vs_context` on a prompt made of the oracle prompt repeated (the logits are meaningless; only timing counts). The attention share comes from the GPU-event profile of the logits-returning direct path, which runs the same attention math.

| Context (tokens) | Direct ms/token | Graph ms/token | Attention share of the GPU step |
|---|---|---|---|
| 5 to 141 | 2.414 | 1.718 | 8% |
| 1024 to 1160 | 4.740 | 4.049 | 47% |
| 4096 to 4232 | 11.606 | 10.870 | 77% |
| 7900 to 8036 | 19.818 | 20.594 | 86% |

Attention cost grows with the context and dominates from about 1k tokens: the untiled decode kernel runs one block per query head (14 blocks on 132 SMs) and each block walks the whole context. At 8k the graph is not faster than direct launches; the reason has not been investigated (one hypothesis, untested: the graph's attention kernel sizes shared memory for the longest sequence of the run, which lowers occupancy).

## Step 3: split-K decode attention (release, same machine, batch 1, 136 tokens generated after the prompt)

Measured with single-sample processes alternated between builds in one pod (`measure_decode_sample`, 8 to 10 rounds), SM clock 1980 MHz throughout, median [min to max], ms per token.

| Context (tokens) | Graph, split-K | Graph, one block per head (step 2) | Notes |
|---|---|---|---|
| 5 to 141 | 2.37 [1.83 to 3.16] | 1.72 [0.98 to 1.73] | split-K is slower at short context |
| 1024 to 1160 | 2.57 [2.41 to 2.97] | 4.04 [2.25 to 5.13] | split-K faster |
| 4096 to 4232 | 3.11 [2.78 to 5.20] | 10.87 (single earlier run) | |
| 7900 to 8036 | 3.51 [2.46 to 4.53] | 20.59 (single earlier run) | 6x faster |

At short context the split-K step is about 0.64 ms slower than before. Register spills were ruled out (`ptxas -v`: no spill, 66 registers). Letting idle splits exit early changed nothing measurable. Attention at 8k still takes about 1.7 ms per step for 97 MB of K and V (bandwidth bound about 0.03 ms), so the kernel is far from the memory system's limit. Correctness: see the ledger. The spread between single processes is large, so treat differences under about 10% as noise.

## Step 3b: vectorised split-K attention (same machine, 8 alternated single-sample processes per build, graph ms/token, median [min to max])

| Context | Split-K (3) | Vectorised split-K (3b) | One block per head (2) |
|---|---|---|---|
| 5 | 2.363 [2.095 to 2.573] | 1.752 [1.317 to 1.755] | 1.721 [1.419 to 1.832] |
| 1024 | 2.567 [2.276 to 2.646] | 1.806 [0.819 to 3.144] | 4.199 [2.734 to 6.040] |
| 4096 | 2.964 [2.940 to 6.014] | 1.928 [1.901 to 1.939] | not run |
| 7900 | 3.379 [1.461 to 5.066] | 2.850 [0.699 to 4.542] | not run |

The short-context loss of step 3 is gone and the long-context steps are 25% to 35% faster than step 3 by medians. Register spills were not the problem (none before or after); the change is the loop structure: head_dim known at compile time, 16-byte loads, independent loads issued back to back. The spread at 1024 and 7900 is too wide to read more into the medians; the 4096 figure is tight. Open: the PTX file grew about ten times, which makes the driver's load-time JIT slower (not measured; step 5 replaces PTX by a prebuilt binary).

## Event-based measurement (replaces the noisy figures above)

Single H100 80GB HBM3 (sm_90), driver 580.126.20, SM clock 1980 MHz throughout, batch 1, Qwen2.5-0.5B, vectorised split-K attention. GPU ms per token between two CUDA events around 128 decode steps, median [p10, p90] of 10 windows, direct and graph alternating inside one process; wall = host clock over the same window.

| Context (tokens) | Direct, GPU | Graph, GPU | Graph, wall | Attention share of the profiled direct step |
|---|---|---|---|---|
| 5 | 2.473 [2.472, 2.483] | 1.755 [1.753, 1.756] | 1.782 | not run |
| 1024 | 2.517 [2.514, 2.523] | 1.816 [1.809, 1.817] | 1.844 | 11% |
| 4096 | 2.657 [2.653, 2.672] | 1.929 [1.927, 1.929] | 1.959 | 10% |
| 7900 | 2.761 [2.753, 2.765] | 2.017 [2.013, 2.018] | 2.048 | 12% |

The spread inside a configuration is under 1%, against tens of percent with the earlier method, so the wide ranges seen before (for example 0.70 to 4.54 ms at 7900) were measurement noise. The step is GPU-bound: wall time exceeds GPU time by about 0.03 ms. Graph replay is 1.4x faster than direct launches at every context length. At 7900 tokens the graph step is 2.02 ms, inside the 2 to 2.5 ms target.

## Start-up cost (single H100, driver JIT cache disabled, 3 alternated runs of a fresh process)

The first generation of a process loads the kernel modules. With the kernels as PTX the driver compiled them at that moment: 867, 1000 and 852 ms. With fat binaries carrying sm_90 machine code (`scripts/build-kernels.sh`): 102, 157 and 150 ms. Later generations take about 21 ms either way. The remaining 100 to 160 ms include creating the cuBLAS handle. Weight upload (about 1 GB) took 0.8 to 2.8 s and varied between runs of the same build.

## Decode session at load, Qwen2.5-0.5B and Qwen2.5-7B (revision 6d1b976, tag pod-20261029-2241)

Single H100 80GB HBM3 (sm_90), driver 580.126.20 (CUDA 13.0), libcuda.so.1 and libcublas.so.12, SM clock 1980 MHz and memory clock 2619 MHz throughout (608 samples), temperature 33 to 48 C. Release build, batch 1, GPU ms per token between CUDA events around 128 chained steps after 3 warm-ups, median [p10, p90] of 10 windows. The session (buffers, K/V, workspace, kernel modules, both graphs) is built at load; a generation allocates and captures nothing. Graph replay is bit-identical to direct launches (0.5B, 40 steps).

| Context | 0.5B direct | 0.5B graph | 7B direct | 7B graph | 7B graph wall |
|---|---|---|---|---|---|
| 5 | 2.443 | 1.746 | 8.555 [8.554, 8.555] | 7.707 [7.706, 7.708] | 7.825 |
| 1024 | 2.533 | 1.808 | 8.620 [8.613, 8.621] | 7.786 [7.781, 7.789] | 7.906 |
| 4096 | 2.654 | 1.926 | 8.863 [8.860, 8.867] | 7.994 [7.986, 7.996] | 8.117 |
| 8000 | 2.758 | 2.017 | 9.459 [9.454, 9.462] | 8.548 [8.537, 8.549] | 8.680 |

0.5B against the earlier event-based figures (graph 1.755 / 1.816 / 1.929 / 2.017): unchanged within 0.5%, so moving the work to load time costs nothing per token. Graph is 1.10x to 1.11x faster than direct on 7B (1.40x on 0.5B): the 7B step is dominated by GEMV weight reads, so launch overhead is a smaller share. 7B at 8000 tokens: attention is 1.35 of 10.37 ms in the profiled direct step (13%), against 5% at context 5. Weight load 12.2 to 13.4 s for 14.19 GiB (4 shards); 58.27 GiB free after load. Weight bandwidth bound for reference: 14.19 GiB at 3.35 TB/s is about 4.5 ms, the graph step at context 5 is 7.7 ms, so decode runs at roughly 59% of the memory-bandwidth floor (inference from the datasheet bandwidth, not measured).

## Chunked prefill and the kernel-count series (pod session 2026-10-30)

Single H100 80GB HBM3 (sm_90), driver 580.126.20 (CUDA 13.0), nvcc 12.4.131, libcuda.so.1, libcublas.so.12, libcublasLt.so.12. SM clock median 1980 MHz in every window (a few seconds at 1845 to 1965 in the longest runs), memory 2619 MHz, 30 to 58 C. Release builds, one `CARGO_TARGET_DIR` per commit, every result starts with `BUILD revision=... sha256=...`. Decode: GPU ms/token between CUDA events around 128 chained steps after 3 warm-ups, median [p10, p90] of 10 windows, graph decode / direct launches. The same hardware-test code measured every commit (`scripts/package-variant.sh`). Between two processes the same build varies by up to about 1.5% (the final tip measured 6.495 ms where an earlier build of the same kernels measured 6.603); inside a run the spread is under 0.3%.

### Decode, Qwen2.5-7B, graph GPU ms/token [p10, p90] / direct

| build | ctx 5 | ctx 1024 | ctx 4096 | ctx 8000 | ctx 16000 |
|---|---|---|---|---|---|
| before the series (`9c5d380`) | 7.632 [7.629, 7.635] / 8.383 | 7.720 [7.719, 7.725] / 8.485 | 7.917 [7.916, 7.922] / 8.706 | 8.472 [8.457, 8.474] / 9.237 | 9.566 [9.563, 9.568] / 10.329 |
| + Q, K, V concatenated (`d797a6a`) | 7.311 [7.299, 7.317] / 7.941 | 7.384 [7.377, 7.385] / 8.033 | 7.609 [7.600, 7.613] / 8.257 | 8.147 [8.130, 8.150] / 8.780 | - |
| + gate, up concatenated (`35e639c`) | 7.044 [7.033, 7.045] / 7.635 | 7.133 [7.122, 7.133] / 7.736 | 7.351 [7.343, 7.352] / 7.968 | 7.892 [7.885, 7.897] / 8.507 | - |
| + cuBLASLt, bias in the epilogue (`82b8056`) | 6.909 [6.909, 6.911] / 7.385 | 6.995 [6.990, 6.996] / 7.472 | 7.212 [7.208, 7.212] / 7.706 | 7.752 [7.745, 7.754] / 8.223 | - |
| + residual fused into the norm (`eb518a1`) | 7.028 [7.016, 7.028] / 7.423 | 7.117 [7.109, 7.117] / 7.523 | 7.333 [7.329, 7.333] / 7.749 | 7.871 [7.856, 7.876] / 8.289 | - |
| + norm kernels with batched loads (`b888327`, experiment build) | 6.603 [6.601, 6.603] / 6.991 | 6.691 [6.683, 6.692] / 7.090 | 6.904 [6.898, 6.906] / 7.314 | 7.444 [7.431, 7.448] / 7.860 | - |
| **final tip `aaada2e`** | **6.495 [6.493, 6.500] / 6.893** | **6.589 [6.582, 6.590] / 7.001** | **6.793 [6.793, 6.801] / 7.226** | **7.346 [7.336, 7.346] / 7.767** | **8.437 [8.430, 8.439] / 8.859** |

Final against before: -14.9%, -14.6%, -14.2%, -13.3%, -11.8%. The fused residual+norm kernel with the old norm loop was 1.7% slower on 7B than the commit before it (the single block did the add too, one more dependent load per element); with batched loads (bit-identical results) fusing is 0.45% faster than the separate add on 7B (6.603 against 6.633 at context 5, 6.904 against 6.936 at 4096) and 5% faster on 0.5B.

### Decode, Qwen2.5-0.5B, graph GPU ms/token [p10, p90] / direct

| build | ctx 5 | ctx 1024 | ctx 4096 | ctx 8000 | ctx 16000 |
|---|---|---|---|---|---|
| before the series | 1.753 [1.751, 1.754] / 2.409 | 1.796 [1.795, 1.806] / 2.460 | 1.925 [1.924, 1.937] / 2.588 | 2.028 [2.011, 2.029] / 2.660 | 2.269 [2.268, 2.290] / 2.942 |
| + QKV | 1.580 / 2.133 | 1.631 / 2.187 | 1.756 / 2.318 | 1.836 / 2.385 | - |
| + gate/up | 1.500 / 1.984 | 1.539 / 2.035 | 1.687 / 2.168 | 1.762 / 2.250 | - |
| + cuBLASLt | 1.402 / 1.789 | 1.451 / 1.846 | 1.591 / 1.972 | 1.665 / 2.051 | - |
| + fused add+norm | 1.372 / 1.696 | 1.408 / 1.755 | 1.545 / 1.888 | 1.627 / 1.969 | - |
| **final tip** | **1.340 [1.340, 1.342] / 1.689** | **1.403 [1.394, 1.403] / 1.754** | **1.525 [1.523, 1.527] / 1.881** | **1.606 [1.599, 1.607] / 1.970** | **1.869 [1.868, 1.870] / 2.243** |

Final against before: -23.6%, -21.9%, -20.8%, -20.8%, -17.6%.

### Achieved bandwidth of each GEMM, 7B graph decode step (Nsight Systems over decode steps at contexts 5 to 133, `scripts/gemm-bandwidth.py`, H100 peak taken as 3350 GB/s)

| build | QKV (31.5 MiB) | o_proj (24.5) | gate+up (259) | down (129.5) | lm_head (1039.5) | all GEMMs |
|---|---|---|---|---|---|---|
| before (separate Q, K, V, gate, up) | q 49.3%, k 20.0%, v 21.1% | 47.5% | gate 82.1%, up 82.9% | 79.1% | 93.0% | 5720.6 us, 73.8% |
| + QKV concatenated | 17.7 us, 55.8% | 48.5% | 82.1%, 82.8% | 79.1% | 92.9% | 5473.7 us, 77.1% |
| + gate/up concatenated | 17.8 us, 55.5% | 48.6% | 90.8 us, 88.9% | 79.0% | 92.8% | 5279.7 us, 79.9% |
| + cuBLASLt (and every later build) | 17.9 us, 55.1% | 48.3% | 90.9 us, 89.1% | 79.8% | 92.8% | 5265.2 us, 80.2% |

The step is 84% GEMM time. What stays below 80%: the two small projections (QKV and o_proj, 55% and 48%: 17.9 and 15.9 us for 31.5 and 24.5 MiB, a latency-bound matrix-vector kernel) and `down` (79.8%). If every GEMM ran at 90% of peak they would take about 4.5 ms of the 6.3 ms of kernel time; the 0.46 ms between the two small projections and that ideal is where a custom matrix-vector kernel would pay, and was not attempted. Per step on the final build (Nsight, decode steps at contexts 5 to 133): 6298 us of kernels in 340 launches (the step takes 6.5 ms, so about 0.3 ms is gaps between launches); GEMMs 5265, attention split 445 and combine 70 (about 16 us per layer even at a context of 5), add_rmsnorm 306 (it was 716 us before the batched loads and the separate norm 542 us), rope 88, swiglu 73.
0.5B: GEMMs 942 MiB in 704 us (42% of peak: the matrices are too small to fill the device), `lm_head` 86.5%.

### Time to first token (prefill), chunked, graph = direct within 0.1%

| prompt tokens | Qwen2.5-7B | Qwen2.5-0.5B |
|---|---|---|
| 1024 | 76.2 ms (graph), 77.1 (direct) | 17.8 ms |
| 4096 | 840 ms | 216 ms |
| 8192 | 3099 ms | 820 ms |
| 16384 | 11 934 ms | 3171 ms |

The curve is quadratic: the chunk attention kernel uses no tensor cores and recomputes K for each query head (see the ledger), so at 16384 tokens the 7B prefill takes about 12 s, almost all of it attention. A fused tensor-core attention kernel is the next speed item for long prompts. Prefill at 16384 against 8192 grows 3.85x for 7B.

### Not byte-reproducible but equal

Rebuilding `elementwise.fatbin` from the committed source gives a different file each time (`cuobjdump -sass` is identical, the PTX differs only in nvcc's per-compile id in the names of shared variables); `rmsnorm.fatbin`, which has no such variable, rebuilds to the same bytes. Compare kernels by SASS, not by hash.

## Tensor-core prefill attention, cuBLASLt selection, page layouts (pod session 2026-10-30, tag `pod-20261030-1047`, build `8dd5bf1`)

Single H100 80GB HBM3, driver 580.126.20, same clocks as above. Release binary sha256 `1796ff35...` (`BUILD` line of each log).

### Time to first token (graph GPU ms, median [p10, p90] of 10 windows)

| prompt tokens | 7B, mma attention | 7B, previous kernel | 0.5B, mma | 0.5B, previous |
|---|---|---|---|---|
| 1024 | 31.8 | 76.4 | 6.2 | 17.9 |
| 4096 | 140.7 | 841.7 | 23.3 | 217.5 |
| 8192 | **332.3** [331.9, 332.7] | 3100.4 | 59.3 | 823.3 |
| 16384 | 886.7 | 11 940.7 | 175.3 | 3180.6 |

7B at 8192 tokens: 332 ms (target < 1 s), 9.3x faster; 16384 tokens 13.5x faster. Chunk graphs are bit-identical to direct launches at 1, 129, 1000, 4096, 4097, 9000, 16384 tokens on both models.

### Decode with the new build (7B graph GPU ms/token)

ctx 5: 6.475, 1024: 6.557, 4096: 6.787, 8000: 7.329, 16000: 8.425 (0.5B: 1.338, 1.396, 1.527, 1.608, 1.867). Within 0.3% of the previous tip at every context.

### cuBLASLt algorithm selection (timed top-8 candidates at load)

No change: the timed choice for the 1x4608x3584 and 1x3584x3584 shapes on 7B and the two 0.5B shapes is the algorithm the heuristic already ranks first (identical hex in the cache), decode ms/token equal within 0.1% with tune on and off, per-GEMM GB/s unchanged (7B QKV 1846, o_proj 1609-1612, gate+up 2987-2990, down 2710-2713, lm_head 3106-3109; 0.5B QKV 516, o_proj 469). QKV and o_proj stay at 55% and 48% of peak: no candidate beats the heuristic for a 1-row GEMM of 25 to 31 MiB. The tuner stays (it is cheap and persisted) but is not a speed item.

### Page layouts (decision 17), VMM

Decode attention sweep, 28/4/128: see ADR 0002 (page-major / layer-major 1.0152 at 8192, 1.0073 at 32768). VMM: granule 2 MiB; reserving 8 GiB of addresses uses 0 bytes; mapping 512 granules one by one 391 ms (764 us each), 4 GiB in one call 0.58 ms; a mapping beyond the free memory is refused (`cuMemCreate` out of memory); reserve and map are refused while serving.

## Gate after the merged trait and the cuBLASLt change (pod session 2026-10-31, tag `pod-20261031-1447`, build `e55e39a`)

Single H100 80GB HBM3, driver 580.126.20, SM clock 1980 MHz (1950 of 1953 samples at 1950 to 1980), memory 2619 MHz. Another pod than 2026-10-30: the same code decodes 7B about 1.5% slower than the table above on this GPU (graph ms/token at context 5: 6.579 against 6.475).

**A/B on this pod, alternating processes (3 rounds, old tree `8dd5bf1` with its own kernels against `e55e39a`, 7B graph decode ms/token median):** context 5: 6.578, 6.582, 6.581 old against 6.585, 6.580, 6.582 new; 1024: 6.670, 6.674, 6.674 against 6.674, 6.672, 6.674; 8000: 7.433, 7.441, 7.440 against 7.439, 7.439, 7.441. The two builds are equal within 0.1%, so the trait change (C3) and the removal of the timed cuBLASLt selection (A4) cost nothing; the 1.5% is the GPU unit.

| 7B graph ms/token | ctx 5 | 1024 | 4096 | 8000 | 16000 |
|---|---|---|---|---|---|
| this pod, `e55e39a` | 6.579 | 6.670 | 6.892 | 7.435 | 8.526 |
| 0.5B | 1.333 | 1.388 | 1.527 | 1.610 | 1.865 |

TTFT 7B graph ms: 1024 tokens 31.7, 4096 140.3, 8192 330.5, 16384 885.4; 0.5B: 6.2, 23.3, 59.4, 175.5.

cuBLASLt candidates timed by `scripts/bench-cublaslt.sh` (microseconds per call): on 7B the heuristic's first algorithm is the fastest for QKV (13.65 us) and o_proj (11.89 us); the third and second candidates are 7 to 34% slower; on 0.5B the first is within 2% of the best (QKV 4.66 against 4.66, o_proj 4.19 against 4.19).

## The sm90 session on the page-major KV pool, decode batching, serving (pod session 2026-11-01, tags `pod-20261101-0138` to `-0302`, final build `fae8255`)

Single H100 80GB HBM3, driver 580.126.20 (CUDA 13.0), nvcc 12.4.131, SM clock 1980 MHz, 34 to 35 C at the start of each run. The pod is a different unit from the 2026-10-31 one (the PVC could not be attached, so the cache was an emptyDir). **Every comparison below is an A/B of the previous tree (`7d8529c`: contiguous KV, own kernels, own target directory) against the new one, alternating processes on this pod**, because the absolute numbers move by about 1% between pods (7B graph ms/token at context 5: 6.653 here against 6.579 in the 2026-10-31 table).

### A5, decode (graph GPU ms/token, median of 10 windows of 128 steps; two processes of each build)

| 7B | ctx 5 | 1024 | 4096 | 8000 | 16000 |
|---|---|---|---|---|---|
| old `7d8529c` | 6.657, 6.653 | 6.743, 6.733 | 6.968, 6.963 | 7.512, 7.505 | 8.607, 8.601 |
| new, paged (build `d92a7a6`) | 6.651, 6.654 | 6.802, 6.802 | 7.081, 7.079 | 7.584, 7.587 | 8.605, 8.608 |
| new / old | -0.0% | +0.9% | **+1.6%** | +1.0% | -0.0% |
| budget | +1.5% | +1.5% | +1.5% | +1.5% | +3% |

The last kernel change (pages of a split loaded into shared memory once per split, `d92a7a6`) measured 6.648, 6.788, **7.069**, 7.568, 8.579 in one process: ctx 4096 is +1.5% of the old pair (7.069 / 6.9655), at the budget, not clearly under it. **The budget holds at 4 of 5 contexts and sits on the limit at 4096.** Before that change (`aa54664`): 6.802, 7.081 at 1024 and 4096.

| 0.5B | ctx 5 | 1024 | 4096 | 8000 | 16000 |
|---|---|---|---|---|---|
| old `7d8529c` | 1.343, 1.341 | 1.387, 1.387 | 1.527, 1.527 | 1.602, 1.604 | 1.863, 1.866 |
| new (`f225f2d`, before the shared-memory page copy) | 1.374, 1.375 | 1.485, 1.481 | 1.709, 1.718 | 1.799, 1.799 | 2.078, 2.075 |
| new (`d92a7a6`) | 1.350 | 1.446 | 1.683 | 1.775 | 2.022 |
| new / old (`d92a7a6`) | +0.6% | **+4.3%** | **+10.2%** | **+10.7%** | **+8.4%** |

**The 0.5B exceeds the budget of A5 (declared on the 7B table) from context 1024 up.** It is a small, launch- and attention-bound model (head_dim 64): the paged decode attention costs about 0.14 to 0.18 ms per step more once all splits are active. Investigated: the page lookup per K/V row (fixed, about 0.03 ms gained), not the cause of the rest; not investigated further (the cause is in `attention_split_rows_64_bf16` or its combine kernel; a profile by kernel is the next step). Owner decision (A5: report and stop that line).

### A5, TTFT (graph prefill GPU ms, 7B; two processes of each build)

| tokens | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| old | 31.97, 32.03 | 141.8, 142.3 | 333.2, 333.4 | 887.8, 886.4 |
| new (`d92a7a6`) | 32.31, 32.26 | 144.1, 143.9 | 340.6, 340.7 | 914.4, 914.2 |
| new / old (budget +5%) | +0.9% | +1.3% | +2.2% | +3.1% |

### Pool, graphs (log lines of the CLI)

7B, `--max-batch 32`: session buffers 1.02 GiB (fixed, at a 131072-token model maximum), free 60.7 GiB before the pool, capture growth 22 MiB, reserve 2070 MiB (growth + the 2 GiB default), granule 2 MiB, page 3.5 MiB, 17172 pages = 58.7 GiB mapped = 1.1 million tokens, 2.0 GiB free afterwards. 0.5B: 16384 pages of 0.75 MiB = 12 GiB. **Graph memory per bucket (1, 2, 4, 8, 16, 32): 2.0, 0.0, 2.0, 2.0, 0.0, 2.0 MiB on both models** (the driver rounds to 2 MiB; a few graphs measure 0).

### Decode step by bucket (GPU ms per step, 10 windows of 128 graph launches, every row at context 512; `batch_hw::decode_step_time_by_bucket`)

| bucket | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 7B ms/step | 6.852 | 6.862 | 6.901 | 7.363 | 7.909 | 8.881 |
| 7B tokens/s | 146 | 291 | 580 | 1087 | 2023 | 3603 |
| 0.5B ms/step | 1.502 | 1.747 | 1.771 | 1.806 | 1.918 | 2.219 |
| 0.5B tokens/s | 666 | 1145 | 2259 | 4431 | 8342 | 14423 |

Per-group GPU time of a direct 7B step (ms, share) at bucket 1 / 8 / 32: gate+up 2.81 / 2.87 / 2.88 (35 / 34 / 29%), down 1.64 / 1.67 / 1.63, attention 0.58 / 0.96 / 2.42 (7 / 11 / 24%), QKV 0.63 / 0.67 / 0.67, o_proj 0.56 / 0.56 / 0.52, lm_head 0.37 / 0.38 / 0.39. The MLP GEMMs read the weights once for any batch up to 32 (their time does not grow); what grows is attention, linearly with the rows. 0.5B at bucket 2: QKV 0.195 to about 0.37 ms (a different cuBLASLt algorithm, plus the QKV split kernel) is the jump from 1.50 to 1.75 ms. The profile is by CUDA event marks in a direct step, not nsys (nsys not installed on this pod).

### Serving, loadgen against the CLI (`--max-batch 32`, 16384-token limit, greedy, release build `fae8255`)

Every count is the server `usage`; the prompts differ at every request and every concurrency level (an earlier run of the same table used the same prompts at every level, so later levels were answered partly from the prefix cache; it was discarded). One server process per model and profile, requests per level = 4 x concurrency (at least 8). TTFT and ITL are p50 [p90] in ms; ITL of a request is (last chunk - first chunk) / (completion tokens - 1).

**7B, 512 in / 256 out**

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| TTFT | 32.7 [34.1] | 37.9 [68.4] | 38.2 [109] | 38.3 [186] | 225 [412] | 451 [874] |
| ITL | 6.78 [6.84] | 6.86 [7.47] | 13.0 [19.4] | 10.1 [31.8] | 31.8 [32.7] | 28.4 [33.8] |
| output tokens/s | 130 | 145 | 325 | 661 | 832 | 1149 |

**7B, 8192 in / 256 out**

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| TTFT | 355 [357] | 365 [504] | 708 [1203] | 1379 [2200] | 2218 [4401] | 5383 [8970] |
| ITL | 7.60 [7.65] | 7.80 [15.9] | 26.1 [83.7] | 147 [167] | 87.2 [169] | 166 [176] |
| output tokens/s | 78 | 106 | 103 | 86 | 172 | 152 |

**0.5B, 512 in / 256 out**

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| TTFT | 7.9 [8.6] | 13.4 [14.0] | 9.5 [25.3] | 9.6 [38.0] | 12.9 [76.1] | 16.7 [146] |
| ITL | 1.47 [1.47] | 1.71 [1.72] | 1.83 [1.91] | 2.21 [2.56] | 2.74 [3.47] | 4.02 [6.56] |
| output tokens/s | 668 | 1051 | 1890 | 3102 | 5144 | 6921 |

**0.5B, 8192 in / 256 out**

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| TTFT | 72.2 [72.5] | 100 [129] | 156 [259] | 201 [393] | 242 [772] | 341 [1746] |
| ITL | 1.84 [1.85] | 2.17 [2.27] | 3.00 [3.51] | 4.59 [5.68] | 8.63 [8.91] | 19.1 [20.4] |
| output tokens/s | 473 | 698 | 967 | 1377 | 1641 | 1532 |

Where scaling stops. On the GPU a decode step of the 7B costs 6.85 ms at 1 row and 8.88 ms at 32 rows (3603 tokens/s), so the decode kernels scale; the served throughput does not follow (1149 tokens/s at 512 in, 150 to 170 at 8192 in) because **policy v1 runs a whole prefill chunk (up to 4096 tokens, 165 ms on the 7B) between two decode steps while any request is prefilling**: every decoding request then sees an inter-token latency of the chunk time (ITL 87 to 176 ms at 8192 in), and prefill of the next requests keeps the device busy. The profile by group says the same at the kernel level (MLP GEMMs read the weights once whatever the batch, attention grows with the rows: 0.58 ms at 1 row to 2.42 ms at 32 rows). The scheduler itself is not the limit: the log lines `scheduler steps` show 9.4 ms per step in the backend against 9.5 ms per loop iteration at concurrency 8. Smaller prefill chunks, or mixed prefill+decode steps, are the owner decisions that follow (NOT done).

### Prefix cache (7B, two requests sharing an 8192-token prefix, the second sent after the first finished; `scripts/prefix-bench.py`, 3 rounds)

| | first request | second request | second / first | hit |
|---|---|---|---|---|
| 7B TTFT ms | 392.9, 390.1, 390.0 | 54.3, 51.8, 51.4 | 0.133 (median) | 8128 of 8219 prompt tokens (98.9%) |
| 0.5B TTFT ms | 83.8, 83.4, 82.4 | 32.2, 28.0, 31.3 | 0.380 | 8128 of 8219 |

(The last 91 tokens are computed: the last-token rule and the 64-token hash blocks.)
