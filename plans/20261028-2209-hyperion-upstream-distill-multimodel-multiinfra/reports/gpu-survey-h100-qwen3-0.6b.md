# GPU survey: H100 + Qwen3-0.6B (2026-10-28)

Context `tf-nguyendk9-at-fpt-com@fke-ncp-modas-stg-qc8ifaxe`, namespace `nguyendk9`. Pod `nvidia/cuda:12.4.1-devel-ubuntu22.04`, 1x `nvidia.com/gpu`. Repo at commit 2c40cf7 (`git archive HEAD`), pod copy only; the repo was not modified. Pods deleted afterwards.

## Hardware [V]
NVIDIA H100 80GB HBM3, compute cap 9.0 (sm_90), driver 580.126.20, CUDA 13.0 (nvcc 12.4 in image), 128 CPU, ~2 TB RAM. Egress to rustup, crates.io, Hugging Face, PyPI works.

## Verified on real hardware / Linux
| Check | Result |
|---|---|
| `cargo test --workspace --features stub` (Linux) | exit 0; same counts as Windows (5 + 7 + 6 + spec) |
| `pytest ignition/tests` (Python 3.12, Linux) | 33 passed |
| `hyperion-compile --model qwen2 --hw sm_90` on real Qwen3-0.6B config.json | OK: head_dim 128 (explicit field; hidden/heads would give 64), 28 layers, 16 heads, 8 kv, gqa, vocab 151936, max_seq_len 40960, fp4 disabled |
| `helios-cli --features sm120` build+link against CUDA 12.4 | links (first time; previously type-check only); needed a feature forward added to helios-cli/Cargo.toml in the pod copy |
| Real device probe (cuInit/cuDeviceGetAttribute) on H100 | works: reports `sm_90` |
| SIGTERM graceful shutdown with real CUDA context | ordered log to `engine stopped`; GPU memory back to 0 MiB |
| Idle server GPU footprint | 518 MiB (one CUDA context) |

## Defects found (all [V])
1. **Wrong backend accepted on Hopper.** `triaged backend=BlackwellSm120 actual=sm_90 target=sm_90 outcome=Exact`: triage compares manifest vs probed cap, never vs the backend's own arch. -> phase 01 (ProbedBackend arch check) / 05 (Capabilities). `create_backend` also `return`s at the first candidate.
2. **Qwen3-0.6B tokenizer.json fails to load with `tokenizers` 0.19.1**: HTTP 422 `data did not match any variant of untagged enum ModelWrapper`. Bumping to 0.22 in the pod copy builds without code changes and the tokenizer loads. -> NEW plan item (phase 01 or 02): bump `tokenizers` (workspace Cargo.toml) + drop the onig workaround if no longer needed, with a Qwen3 tokenizer fixture test.
3. **Ignition accepts a Qwen3 config under `--model qwen2`** and writes `model_id: qwen2`; it never checks `architectures`/`model_type` (Qwen3 has QK-norm, so a runtime would be wrong). -> phase 04 (strict architecture check in compilers).
4. **Over-long prompt (~60k tokens > max_seq_len 40960) is accepted and silently truncated** (503 from the no-op scheduler, not 400). -> phase 02 (truncation => 400).
5. `/v1/chat/completions` and `/health` are 404; chat is at `/chat/completions`. -> phase 01 (already planned).
6. No generation: scheduler is a no-op ("scheduler closed stream before first event", 503) and the hopper_sm90 backend is a 60-line skeleton, so a real Qwen3-0.6B decode on H100 is NOT possible yet. Real e2e needs: sm90 backend (or CUDA-graph path) + weight loading + forward pass + sampling, none of which exist.

## Not tested
Weights (model.safetensors) were not downloaded or loaded; no kernel executed; no throughput/latency numbers; sm90 TensorRT-LLM path (bindgen/cc) not built; HTTP 503 during drain, scheduler-death branch, exit code (timeout wrapper hid it).

## Plan impact
Confirms phase 01 (items 8/9, /v1 nest, /health) and phase 02 (truncation) premises on real hardware. Adds: tokenizers bump (2), architecture check (3). Suggests a hardware-gated e2e test (`*_hw`, `#[ignore]`) that runs on this cluster: probe on H100 must refuse the sm120 backend.
