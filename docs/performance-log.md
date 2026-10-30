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
