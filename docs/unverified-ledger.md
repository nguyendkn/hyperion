# Verification ledger

What has been run for real, and what has not. Tags: `[RUN]` executed on the build host (Windows, no GPU); `[RUN on H100]` executed on a real H100 (staging cluster pod, sm_90, driver 580.126.20, CUDA 13.0, nvcc 12.4); `[TYPE-CHECK]` compiled only; `[NEEDS-GPU]` not verified anywhere. Phases append their own named section (append-only).

## Phase 01 (2026-10-28)

| Item | Tag | Evidence |
|---|---|---|
| sm120 backend links against the CUDA 12.4 toolkit (`helios-cli --features sm120`, `CUDA_PATH=/usr/local/cuda`, `LIBRARY_PATH=.../stubs`) | `[RUN on H100]` | build finished on the pod, no source edits needed |
| Real device probe (`cuInit`/`cuDeviceGetAttribute`) reports `sm_90` | `[RUN on H100]` | log `actual=sm_90` |
| Blackwell backend on an `sm_90` device is refused; `sm_90` and `sm_120` packages are refused with every backend's reason; exit 1, no panic | `[RUN on H100]` | pod run, `reports/gpu-survey-h100-qwen3-0.6b.md` + `reports/phase-01-report.md` |
| Graceful shutdown (SIGTERM) returns GPU memory to 0 MiB; idle CUDA context ~518 MiB | `[RUN on H100]` | `nvidia-smi` before/after |
| Rust suite (51 tests) on Linux with the H100 present | `[RUN on H100]` | `cargo test --workspace --features stub` on the pod |
| Prompt over the limit -> HTTP 400 `PromptTooLong` (no silent truncation): 60003-token prompt vs limit 40960 with the real Qwen3 tokenizer, over HTTP, on the H100 host (sm120 build; sm120 refused, stub served) | `[RUN on H100]` | pod run 2026-10-29 (commit 3294d70); also `[RUN]` locally: `prompt_limit.rs`, `model_tokenizers.rs` |
| Factory fall-through: sm120 refused on `sm_90`, stub accepted for an `sm_0` package; `/health` 200; short prompt -> 503 (no scheduler); KV/sequence 4 697 620 480 B for the Qwen3-0.6B shape (matches 28x8x128x2x2x40960) | `[RUN on H100]` | same pod run |
| Rust suite incl. real Qwen3 tokenizer tests with `HYPERION_REQUIRE_MODEL_FIXTURES=1` (no skips) on Linux | `[RUN on H100]` | pod run, sha256 of fixtures verified |
| DEFECT: a backend refused by triage kept its CUDA context (~527 MiB) until process exit while another backend served (SM120 backend has no `Drop`; teardown only via `shutdown()`) | found `[RUN on H100]` | fix = factory calls `shutdown()` on refused backends; re-run on the H100 (2026-10-29): GPU memory while the stub serves 4 MiB (was 527 MiB), 0 MiB after SIGTERM and after an all-backends-refused exit `[RUN on H100]` |
| Triage matrix, format table, `/health`, `/v1` route, `--max-seq-len`, loopback bind, Ignition `architectures` check, tokenizer under `tokenizers` 0.22 | `[RUN]` | Rust ~45 tests, pytest 43 + 2 skipped |
| Model-specific tests (real tokenizer/config) | `[RUN]` only with `python scripts/fetch-model-fixtures.py`; skipped otherwise | `HYPERION_REQUIRE_MODEL_FIXTURES=1` makes absence a failure |
| Rust branch coverage on the diff | UNVERIFIED | `cargo +nightly llvm-cov --branch` works (54 BRDA) but `diff-cover` on lcov reports lines only (diff 95%); whole-file branch for `main.rs`/`api.rs` is dominated by older code |
| sm90 backend: real probe (`sm_90`), factory picks `HopperSm90` on an H100 with triage `Exact` | `[RUN on H100]` | pod run 2026-10-29, log `triaged backend=HopperSm90 actual=sm_90 outcome=Exact` |
| `helios-cli --features sm90` builds and links on the H100 pod (shared CUDA core, no TensorRT-LLM) | `[RUN on H100]` | same run; also `[TYPE-CHECK]` on Windows |
| Any kernel, forward pass, decode, throughput; KV cache allocation on sm90 | `[NEEDS-GPU]` | scheduler is a no-op; sm90 `allocate_kv_cache`/`execute_*` return "not wired"; next phase-10 steps |
| KV-vs-VRAM check against a real device total | `[NEEDS-GPU]` | unit-tested with synthetic totals only; stub reports 0 and is skipped |
| HF reference oracle for Qwen2.5-0.5B (prompt "The capital of France is" -> top-1 " Paris"; bf16, eager vs sdpa differ by 0.75 max-abs; declared tolerance 3.0 + argmax + top-5 criteria) generated on the H100 (torch 2.5.1, transformers 4.51.3) | `[RUN on H100]` | `scripts/gen-oracle.py`; file in gitignored `testdata/oracle/` |
| Ignition maps the real Qwen2.5-0.5B safetensors header (290 BF16 tensors, header read by HTTP range at the pinned revision) to manifest layers: no missing, no unmapped, shapes checked against `ModelConfig`, byte ranges contiguous from 0 to 988065536; manifest validates against the Rust-generated schema | `[RUN]` | `ignition/tests/test_weights.py` (12), `test_model_fixtures.py`; header file via `scripts/fetch-model-fixtures.py` |
| Qwen2.5-0.5B weights uploaded through `HopperBackend::load_model_weights`: 290 tensors, 988 065 536 bytes read back byte-identical to the file (upload + readback + compare 1.92 s), clean error for a missing weights file, GPU memory 0 MiB afterwards | `[RUN on H100]` | `helios-hal/tests/weights_hw.rs` (`--ignored`, `HYPERION_WEIGHTS_PACKAGE`) |
| `helios-cli` with the real package: engine `Loaded` only after the load succeeded, `/health` `{"status":"Loaded"}`, 1603 MiB resident (942 MiB weights + context), SIGTERM -> 0 MiB, missing weights file -> non-zero exit and 0 MiB (no leak: `CudaDevice` now has `Drop`) | `[RUN on H100]` | pod run 2026-10-29 |
| Weights are not copied into a `.hyperion` package: the loader reads `model.safetensors` next to the package; `helios-pkg` does not exist | not implemented | later phase (06) |
| Hyperion logits vs this oracle | `[NEEDS-GPU]` | no forward pass exists yet (phase 10) |
| `cuda-driver-sys` 0.3 graph API at kernel time on CUDA 13 | `[NEEDS-GPU]` | links on CUDA 12.4 toolkit; graph calls never executed |
