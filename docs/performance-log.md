# Decode performance log

Every number here is **single H100 80GB HBM3 (sm_90), driver 580.126.20, nvcc 12.4, batch 1, Qwen2.5-0.5B, context 5 to 133 tokens, greedy**, and nothing more general. Release builds only.

Protocol: per-token time = (136-token run minus 8-token run) / 128, median of 3 runs, whole step including token upload, logits download and greedy sampling (`weights_hw.rs::graph_decode_is_bit_identical_to_direct_launches`). In-process GPU time per group of kernels comes from CUDA events (`weights_hw.rs::profile_decode_steps`, `HopperBackend::generate_profiled`); event marks add a little overhead, so group times are indicative. Kernel and gap numbers come from Nsight Systems (`nsys`, `--trace=cuda`, graph mode with `--cuda-graph-trace=node`).

| Step | Direct ms/token | Graph ms/token | Notes |
|---|---|---|---|
| (earlier, debug build: not a valid performance number) | 9.51 | 10.27 | `cargo test` without `--release`; host work in a debug build is far slower |
| 0. baseline, release | 2.980 | 2.288 (1.30x) | graph is faster than direct |

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
