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
