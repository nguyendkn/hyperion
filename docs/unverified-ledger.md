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
| sm90 backend: no probe is wired, `initialize_device` returns an error | `[NEEDS-GPU]` | TensorRT-LLM/bindgen path never built |
| `--features sm90` | `[TYPE-CHECK]` not even type-checked in phase 01 | |
| Any kernel, forward pass, decode, throughput | `[NEEDS-GPU]` | scheduler is a no-op; first forward is phase 10 |
| KV-vs-VRAM check against a real device total | `[NEEDS-GPU]` | unit-tested with synthetic totals only; stub reports 0 and is skipped |
| `cuda-driver-sys` 0.3 graph API at kernel time on CUDA 13 | `[NEEDS-GPU]` | links on CUDA 12.4 toolkit; graph calls never executed |
