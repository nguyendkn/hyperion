# 0001: sm90 backend = shared CUDA Driver-API core + per-arch kernels

Status: accepted (user, 2026-10-28). Plan: phase 10 (`plans/.../phase-10-first-forward-sm90.md`), open decision 13.

## Context
Hopper (SM90) has no working backend: `hopper_sm90/backend.rs` is a skeleton that cannot probe or load. The original plan wrapped TensorRT-LLM in a C++ shim. A survey on a real H100 showed the SM120 backend's Driver-API code (`cuInit`, `cuDeviceGetAttribute`, context, streams) links against the CUDA 12.4 toolkit and probes the H100 correctly (`sm_90`); the Driver API does not depend on the GPU generation, only kernels do.

## Decision
- One shared CUDA Driver-API core (context, streams, memory, module/graph handles, typed errors) used by every NVIDIA generation.
- Each architecture owns only its kernel set behind the existing `ComputeBackend` trait; `helios-core` stays hardware-agnostic (rule 1); a new arch is still a new directory under `helios-hal/src/backends/` (rule 2).
- The first forward pass uses simple, correct kernels (cuBLAS for GEMM), then optimises against measurements.
- First verification model: Qwen2.5-0.5B (Qwen2 architecture, no spec change). Success = logits within a tolerance declared before the run, versus the Hugging Face oracle, on an H100.
- No TensorRT-LLM C++ shim.

## Why over the alternatives
- TensorRT-LLM shim: a large C++ runtime, `bindgen`/`cc` build coupling, and it works against the single-Rust-binary goal; the Driver API route is already proven to link and probe on H100.
- Per-arch copies of the driver code: duplicates context/memory handling and leaks (the SM120 backend already had a context-leak bug found on H100).

## Consequences
- More kernel work later; performance parity with vendor runtimes is not assumed.
- The SM120 device code is extracted into the shared core (coordinate with phase 05); it also needs a `Drop` so a refused/failed backend releases its context.
- `cuda-driver-sys` 0.3 graph APIs on CUDA 13 remain unverified until a kernel runs (see `docs/unverified-ledger.md`).
- Revisit if cuBLAS-first cannot reach acceptable speed.
