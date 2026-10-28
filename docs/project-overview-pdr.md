# Hyperion: Product Overview and PDR

Last updated: 2026-10-28 | Status: pre-alpha, skeleton | Version: 0.1.0 (workspace and package versions)
Sources: `repomix-output.xml` snapshot and `CLAUDE.md`. Code was read, not built or run in this pass.

## 1. Product overview

Hyperion is a local LLM inference engine. It is a fork re-scoped from NVIDIA `flare-local-inference` (Apache 2.0). Upstream design details are not verified here and are not used as fact.

Two stages, split by language and lifecycle:

| Stage | Language | When | Job |
|---|---|---|---|
| Ignition (`ignition/`) | Python | Build time | Read HF weights, apply hardware recipe, compile graph, package and sign a `.hyperion` bundle |
| Helios (`helios/`) | Rust | Run time | Load signed bundle, probe device, triage, manage KV cache, serve OpenAI-compatible API. No Python at runtime |

Target hardware: Blackwell SM120/SM121 (primary, CUDA Graph path) and Hopper SM90 (H100/H200, planned).
Target models: Llama-family, Qwen-family, DeepSeek-family (compilers registered as stubs).

## 2. Goals

- G1. Run locally on supported NVIDIA GPUs with an OpenAI-compatible `/v1/chat/completions` endpoint.
- G2. Support multiple GPU generations without changing `helios-core`. New hardware = new directory under `helios-hal/src/backends/`.
- G3. Support multiple model architectures without changing the runtime. New model = new file under `ignition/.../architectures/`.
- G4. Refuse incompatible packages at load time with a clear error. No panics, no segfaults.
- G5. No Python at runtime.

## 3. Non-goals

- Not a cloud or multi-node serving system. Single host, single device per backend instance.
- Not a training or fine-tuning tool.
- Not CPU or Apple/AMD inference (no backend exists; not planned here).
- No FP4 on Hopper (hardware does not support it).
- No mocks or simulated hardware on `main`. Hardware code is real FFI behind `cfg` features.
- Not a general-purpose Python ML toolkit; Ignition only compiles and packages.

## 4. Functional requirements

Status key: **Done** = implemented in code. **Partial** = some code, not end-to-end. **Stub** = signature or skeleton only. **Planned** = not in code.

| ID | Requirement | Status |
|---|---|---|
| FR-1 | Compile a model + hardware recipe to `.hyperion` package and `.manifest.json` sidecar | Partial: CLI and manifest JSON write work; `layers` always empty; weight blob not written |
| FR-2 | Sign weight blob (ed25519), verify at load; release loader rejects unsigned | Planned: `signature` field exists; no signing, no verification |
| FR-3 | Package codec (`helios-pkg` crate) reads `.hyperion` | Planned: CLI loads `<model>.manifest.json` sidecar as placeholder |
| FR-4 | Read manifest, parse `target_arch`, reject unknown `manifest_version` | Partial: enum tag parse exists; see §9 for Ignition mismatch |
| FR-5 | Probe real device compute capability via HAL factory | Partial: SM120 probe is real FFI, not run here; SM90 hardcodes `sm_90`; stub reports `sm_0` |
| FR-6 | Triage package target vs device (exact match) | Partial: code allows downgrade (see §6) |
| FR-7 | Load weights to device, capture decode graph | Stub: SM120 `load_model_weights` returns error; SM90 not wired |
| FR-8 | Allocate paged KV cache via HAL | Partial: SM120 `cuMemAlloc` real; sizes hard-coded (HEAD_DIM 128, 32 heads) |
| FR-9 | Tokenize OpenAI `messages` to token ids | Partial: HF `tokenizers` wrapper works against a test fixture; flatten is `role:content`, not a chat template |
| FR-10 | Continuous-batching scheduler (prefill + decode) | Stub: `Scheduler::run()` closes ingress and returns |
| FR-11 | Decode and return text in `/v1/chat/completions` | Stub: no tokens generated; see §9 |
| FR-12 | Streaming (SSE) responses | Planned: non-streaming only |
| FR-13 | Engine lifecycle state machine with safe transitions | Done (unit-level logic): 6 states, CAS transitions, exhaustive table |
| FR-14 | Map engine state to HTTP status (501 no tokenizer; 503 engine down; 504 first-event timeout) | Done in code; see §6 for the 503 rule |
| FR-15 | Model parameters (layers, KV heads, head_dim, vocab, max_seq_len) read from manifest | Planned: `ModelConfig` not in manifest; SM120 hard-codes dims |
| FR-16 | Hardware recipes per target (`sm_90` FP8/BF16, `sm_120` FP4/FP6/FP8/BF16 + graphs) | Done as data (Python dict); not yet consumed by runtime |

## 5. Non-functional requirements

| ID | Requirement | Target / Current |
|---|---|---|
| NFR-1 | Forward-compatible manifest | `v1` frozen after first released package. Until then may change. Breaking change = new `v2` |
| NFR-2 | Device-agnostic core | `helios-core` must not reference any backend. Enforced by convention, not yet by a CI check |
| NFR-3 | Build without CUDA | `default = ["stub"]`; `cargo build --workspace` must need no GPU or toolkit |
| NFR-4 | Hot-path lock-free status read | `AtomicU8` state; API reads do not take a mutex |
| NFR-5 | Bounded ingress | `INGRESS_CAPACITY = 32`; full queue returns 503 after 500 ms |
| NFR-6 | Code file size | Under 200 lines per code file (repo rule) |
| NFR-7 | Test data | Real fixtures; no mocks on `main`. Fixtures resolved via `CARGO_MANIFEST_DIR` |
| NFR-8 | Security | Ed25519 package signatures (planned, FR-2). No secrets in repo (`.gitignore` covers `.env`) |
| NFR-9 | Performance (decode) | Target: one `cudaGraphLaunch` per decode step on SM120. Not measured; no benchmark exists |
| NFR-10 | Release build | Fat LTO, `codegen-units = 1`, `panic = "abort"` in `[profile.release]` |

## 6. Architectural decisions

Authoritative source: `CLAUDE.md`. "Code state" is from reading the repomix snapshot.

| Decision | Code state |
|---|---|
| Triage: exact `target_arch` match only. Ignition compiles per target. PTX fallback, if ever, is an explicit manifest field | **Decided, code pending.** `triage.rs` still allows `Downgrade` when `target < actual`. Also `lib.rs` docs still describe forward-compat |
| Manifest records `target_arch` and `tensor_format` | `target_arch` yes. `tensor_format` absent from `v1::Manifest` |
| `ModelConfig` (num_layers, num_kv_heads, head_dim, vocab_size, max_seq_len) in manifest; backends read it | **Decided, code pending.** SM120 hard-codes dims |
| Rust structs are schema source of truth; JSON Schema via `schemars`; Ignition validates; Rust/Python round-trip test | **Decided, code pending.** No schema generated, no round-trip test |
| `specification/v1` immutable at first released package | Pre-release: still mutable |
| `.hyperion` codec is a dedicated crate `helios-pkg` | Not created. `.manifest.json` sidecar placeholder in use |
| HTTP: 501 when no tokenizer; 503 only for `Uninitialized`/`Stopped` engine states | Handler also returns 503 for `Draining`, queue-closed, queue-full, and no-scheduler. Decision text and code disagree on `Draining` and scheduler errors |
| Backend selection order: sm120 -> sm90 -> stub | Implemented in `backend_factory::create_backend` |
| Hopper (SM90) does not run FP4. sm90 recipe forces FP8/BF16 | Recipe data matches (`fp4: disabled`). Not enforced at runtime |
| RadixCache: in-process, 256-token chunks, index only; device memory via HAL | Index struct exists (`insert`/`lookup`); no eviction, no tiers, not used by scheduler |
| CUDA FFI: `cuda-driver-sys` 0.3, only under `sm120` | Matches. `sm90` path uses no CUDA FFI yet (see §8) |
| Tokenizer: HF `tokenizers` 0.19 with `onig` | Matches. Re-check on upgrade |

## 7. Success criteria

Phase-level. Each item is a checkable condition, not a date.

- SC-1. `cargo build --workspace` and `cargo test --workspace` pass on a no-CUDA Windows host (stub). Not verified in this pass.
- SC-2. `cargo check -p helios-hal --features sm120` type-checks on a no-CUDA host. Not verified in this pass.
- SC-3. `hyperion-compile --model llama3 --hw sm_120` writes a manifest with non-empty `layers` and a weight blob, round-tripping into Rust `v1::Manifest`.
- SC-4. Load a signed package; a tampered blob is rejected at load with a clear error.
- SC-5. On SM120 hardware: `/v1/chat/completions` returns generated text for a Llama-family model, end-to-end.
- SC-6. Triage rejects a mismatched package with a typed error and no panic, on both stub and real backends.
- SC-7. Hopper (SM90) builds, loads a recipe-compliant package, and generates text. Blocked on §8.
- SC-8. Measured decode latency and tokens/s published for SM120. No target number set yet.

## 8. Status and milestones (proposed, not agreed)

Ordering follows the critical path in the code. Dates are not set.

- **M0 (current): skeleton.** Workspace layout, state machine, HAL trait, factory, triage logic, HTTP route, tokenizer wrapper with test fixture, Ignition plugin registries, SM120 FFI scaffolding.
- **M1: package codec.** `helios-pkg` crate; real `.hyperion` read and write; signature verify. Unblocks FR-3, FR-2, SC-4.
- **M2: real Ignition compile.** Llama-family weight extraction, layer map with byte offsets, schema round-trip test. Unblocks SC-3.
- **M3: SM120 load and decode.** Weight upload, graph capture of real kernels, real argmax/sampling. Unblocks SC-5.
- **M4: scheduler and API.** Continuous batching, streaming, chat template from package. Unblocks FR-10 to FR-12.
- **M5: Hopper.** TensorRT-LLM C++ shim + `bindgen` per `build.rs` plan. Unblocks SC-7.
- **M6: release.** `specification/v1` freeze, signature enforcement in release loader.

Hardware note: no GPU or CUDA toolkit on the build host. SM120 can only be type-checked here. Hopper has no FFI yet. Runtime results need a GPU host.

## 9. Known gaps and mismatches

Each is a code-reading finding. None was reproduced by running the code.

1. **Default build cannot serve a real package.** Stub reports `sm_0`. A package with `target_arch` `sm_90`/`sm_120` fails triage (`UpgradeRequired`), so `create_backend` returns an error. Inference from code.
2. **Scheduler exits at once.** `Scheduler::run()` closes ingress and returns `Ok`. After that, `/v1/chat/completions` returns 503 "queue closed" for any request that gets past tokenization. No generation path exists.
3. **Manifest tag mismatch.** Ignition writes `"manifest_version": 1` (integer). `specification/src/lib.rs` uses `#[serde(tag = "manifest_version")]` with variant `V1`, which serde expects as the string `"V1"`. Inference from serde semantics; likely fails to parse Ignition output. No round-trip test exists.
4. **Signature type drift.** Python `signature: list[int]`. Rust `Option<Vec<u8>>`. Matches JSON shape, but no test verifies it.
5. **SM120 placeholder kernels.** `prefill_launch` captures an empty graph. `decode_replay` reads an uninitialised logits buffer. Result is not a real token. Do not treat SM120 decode as working.
6. **SM120 vocab and dims hard-coded** (`VOCAB_SIZE = 128_256`, HEAD_DIM 128, 32 heads). Conflicts with the `ModelConfig` decision.
7. **Doc drift in code comments.** `helios-hal/src/lib.rs` and `triage.rs` describe forward-compat downgrade as a design invariant. Conflicts with the decided exact-match rule.
8. **SM90 `build.rs` linkage is commented out.** `sm90` feature compiles, but every method returns "not wired". `Taskfile.yml` says "requires TensorRT-LLM + CUDA".
9. **SM121 unresolved.** Exact-match triage treats `sm_121` and `sm_120` as different. Whether a SM120 package may run on SM121 is not decided.
10. **Ignition writes no weights and no signature.** `cli.py` has TODOs. `layers` is always `[]`.
11. **`ModelConfig` and `tensor_format` absent** from manifest (see §6).
12. **Taskfile `build:sm120` comment** mentions NVRTC. No NVRTC code exists.

## 10. Requirement change history

| Date | Change | Source |
|---|---|---|
| 2026-10-28 | Initial PDR from repomix snapshot and CLAUDE.md. Triage decision recorded as exact-match (code pending). | docs-manager, wave 1 |

## Unresolved questions

1. Does SM120 package run on SM121 device under exact match, or does SM121 need its own build? (CLAUDE.md lists both; no rule.)
2. Is `tensor_format` a new manifest field in `v1` (pre-release, allowed) or does it wait for `v2`?
3. Which Ignition/Rust manifest tag format wins (integer `1` vs string `"V1"`)? Fix on Python side or Rust side?
4. Schema generation: `schemars` on Rust structs at build time, or a separate export step? Who owns the generated JSON Schema file?
5. Streaming (FR-12) in scope for first public release, or after M4?
6. Performance targets (NFR-9, SC-8): none set. Need a number and a reference model.
7. Chat template source: package asset (Jinja) or Ignition-generated? Current code uses `role:content` join.
8. Is `helios-core` device-agnostic check to be a CI lint (grep or cargo-deny rule), or convention only?
