# Codebase Summary

Last updated: 2026-10-28 · Workspace v0.1.0 · Branch main (no commits yet)

Sources: helios-core, helios-cli, helios-hal, specification and ignition read directly from source. `repomix-output.xml` not used for these (stale for helios-core). Tags: **[v]** verified by reading source or `cargo metadata`; **[i]** inference; **[x]** not executed.

## Layout

```
specification/           shared manifest schema (Rust = source of truth)
helios/crates/helios-core/   device-agnostic runtime: api, scheduler, state, tokenizer, cache
helios/crates/helios-hal/    ComputeBackend trait, factory, triage, `cuda/` (shared CUDA Driver-API core), `weights_plan`, backends (stub, hopper_sm90, blackwell_sm120)
helios/crates/helios-cli/    binary: manifest -> HAL probe -> triage -> engine state -> serve
ignition/                    Python build-time compiler (package hyperion_ignition)
```

## Module map

### specification (Rust)
| File | Purpose |
|---|---|
| `src/lib.rs` | `Manifest` enum, `#[serde(tag="manifest_version")]` (wire tag `"V1"`); `ManifestError`; `LATEST = 1`; with feature `schema`: `schema.rs` + `bin/gen-schema.rs` generate `schema/manifest.v1.json` |
| `src/v1/mod.rs` | v1 structs: `Manifest` (model_id, target_arch, tokenizer, layers, hardware_recipe, signature), `TokenizerConfig`, `LayerMapping`, `TensorMetadata` |

### helios-core (Rust)
| File | Purpose |
|---|---|
| `src/lib.rs` | Module list + re-exports (`StateMachine`, `Scheduler`, `StreamEvent`, ...) |
| `src/api.rs` | Axum `POST chat_completions`; `AppState {state, tokenizer, ingress_tx}`; non-streaming; 500 ms ingress timeout, 5 s first-event timeout |
| `src/scheduler.rs` | `Scheduler`, `InferenceRequest`, `StreamEvent`, `SamplingParams`, `FinishReason`, `ingress_channel()` (bounded, cap 32). `run()` is a no-op |
| `src/state.rs` | `EngineStatus` (6 states, `repr(u8)`), `EngineState` (`AtomicU8` + CAS loop, transition table), `StateMachine` wrapper |
| `src/tokenizer.rs` | `PromptTokenizer`: lazy `OnceCell` load of `tokenizer.json`, right-truncation (default 4096), `encode_messages`, `decode`; flatten = `role:content` joined by `\n` |
| `src/cache.rs` | `RadixCache` = `HashMap<ChunkHash, BlockEntry>`, `CHUNK_TOKENS = 256`; `lookup`/`insert` only |

### helios-hal (Rust)
| File | Purpose |
|---|---|
| `src/lib.rs` | Re-exports `create_backend`, `ComputeBackend`, `DynBackend`, `Triage` |
| `src/traits.rs` | `ComputeBackend` trait; `HardwareInfo`, `DevicePointer`, `TensorData`; `DynBackend = Box<dyn ComputeBackend>` |
| `src/triage.rs` | `triage(&Manifest, &ProbedBackend)`: backend arch == device arch, manifest `target_arch` == device arch, `tensor_format` in the arch's format table; `supported_formats(arch)` is the single table (committed dump `tests/data/format-table.json`) |
| `src/backend_factory.rs` | `create_backend(target)`; `BackendKind` (cfg-gated); `instantiate()` per kind |
| `src/backends/mod.rs` | cfg-gated module list |
| `src/backends/stub.rs` | `StubBackend`: reports `sm_0`, returns placeholder Ok values |
| `src/backends/hopper_sm90/` | `HopperBackend` on the shared `cuda/` core: real probe, real weight upload from `model.safetensors` beside the package; KV cache, kernels and forward pass return "not wired" errors |
| `src/cuda/{loader,sys,device}.rs` | run-time loading of the driver; `CudaDevice`: `probe`, `alloc`, `free`, `teardown_ref` |
| `.../blackwell_sm120/graph.rs` | `StaticCudaGraph`: begin/end capture, instantiate, launch, destroy |
| `.../blackwell_sm120/kernels.rs` | `prefill_launch` (empty-graph capture + logits alloc), `decode_replay` (argmax over raw u16 lanes), `VOCAB_SIZE = 128_256` |
| `.../blackwell_sm120/ffi.rs` | `eval_cuda`: `CUresult` -> `anyhow::Result` |
| `.../blackwell_sm120/backend.rs` | `BlackwellBackend` (Mutex-guarded device + graph); `load_model_weights` returns error |
| `build.rs` | only `rerun-if-env-changed` for CUDA_PATH / CUDA_LIB_DIR under the `cuda` feature; no bindgen/cc, no TensorRT-LLM shim (ADR 0001) |

### helios-cli (Rust)
| File | Purpose |
|---|---|
| `src/main.rs` | Boot: read `<model>.manifest.json` sidecar -> `create_backend(&manifest)` (probe + triage per compiled-in backend) -> resolve `--max-seq-len` (KV fit check) -> engine `Ready` -> spawn scheduler + supervisor -> resolve tokenizer (path-safe) -> `axum::serve` on `--bind` (default `127.0.0.1`) -> graceful shutdown |

### ignition (Python)
| File | Purpose |
|---|---|
| `cli.py` | click `hyperion-compile --model --hw --out --source`; validates the manifest against the schema, then writes `<out>.manifest.json` only |
| `compiler/base.py` | `CompiledManifest` (+ `to_json_dict`), `LayerMap`, `ModelCompiler` Protocol |
| `compiler/registry.py` | `register`, `build_compiler`; imports `architectures` at bottom |
| `compiler/architectures/{llama,qwen,deepseek}.py` | One compiler class each, ids `llama3`, `qwen2`, `deepseek`; `layers=[]` |
| `hardware/base.py` | `HardwareRecipe` Protocol (`target_arch`, `constraints()`) |
| `hardware/registry.py` | `register`, `build_recipe`; imports sm90/sm120 recipes at bottom |
| `hardware/sm90_hopper.py` | Recipe `sm_90`: fp4 off, fp8/bf16 on, cuda_graphs off |
| `hardware/sm120_blackwell.py` | Recipe `sm_120`: fp4/fp6/fp8/bf16 on, cuda_graphs on |

## Key contracts

- `ComputeBackend` (helios-hal/traits.rs): `initialize_device`, `load_model_weights`, `allocate_kv_cache`, `execute_prefill`, `execute_decode_step`, `shutdown` (default no-op). Must be `Send + Sync`.
- `InferenceRequest` + `StreamEvent` (scheduler.rs): API -> scheduler over bounded mpsc; scheduler streams `Tokens*` then exactly one `Done` or `Error`, then drops the sender.
- `EngineStatus` transitions (state.rs): Uninitialized -> Ready -> Loaded <-> Serving; Loaded/Serving -> Draining -> Stopped; Loaded/Ready -> Stopped.
- `Manifest` (specification): versioned by `manifest_version`; `target_arch` drives triage.

## Feature flags

| Crate | Feature | Enables |
|---|---|---|
| helios-hal | `default` = `stub` | stub backend only |
| helios-hal | `stub` | nothing extra |
| helios-hal | `cuda` / `sm90` / `sm120` | `cuda` = shared core (driver and cuBLAS loaded at run time, no dependency); `sm90` and `sm120` both enable it |
| helios-hal | `sm120` | enables `cuda`; `BlackwellSm120` candidate |
| helios-hal | `multi` | `sm90` + `sm120` |
| helios-core | none | |
| helios-cli | none [v] | no `[features]` table; does not forward HAL features |

Backend selection (backend_factory.rs): candidate list built from enabled features, order sm120, sm90, stub. The loop returns on the first candidate; a failing probe is an error, not a fallthrough. [v]

## Build and test commands

| Command | Source | Status |
|---|---|---|
| `task build` = `cargo build --workspace` | Taskfile.yml | [x] stub build, expected to work |
| `task build:sm90` = `cargo build -p helios-cli --no-default-features --features sm90` | Taskfile.yml | BROKEN [v]: helios-cli has no `sm90` feature |
| `task build:sm120` = same with `sm120` | Taskfile.yml | BROKEN [v]: same cause |
| `task check` = `cargo check --workspace --all-targets` | Taskfile.yml | [x] |
| `task test` = `cargo test --workspace` | Taskfile.yml | [x] |
| `cargo check -p helios-hal --features sm120` | CLAUDE.md | type-check only, no CUDA on host |
| `python -m compileall ignition/src` | CLAUDE.md | [x] |
| `pip install -e ignition` / `hyperion-compile ...` | Taskfile.yml | [v] run; 21 pytest pass |

Release profile: fat LTO, `codegen-units=1`, `panic="abort"` [v].

## Tests present

- helios-core unit (`src/tokenizer.rs`): 2 (`flatten_single_message`, `flatten_multiple_messages_join_with_newline`).
- helios-core integration (`tests/encode_path_smoke.rs`): 7 `#[tokio::test]` over `tests/fixtures/dummy_tokenizer.json` (WordLevel fixture, not a real model).
- Rust (`cargo test --workspace`, stub backend): helios-core 5 unit (2 tokenizer, 3 state) + 7 encode smoke; specification 6 (7 with `--features schema`, adds the schema drift check). Python: 21 pytest in `ignition/tests`. No tests yet for helios-hal, cache, scheduler, api, cli.

## Module status

Status: **implemented** (works as designed, reachable) / **partial** (some paths wired or working) / **skeleton** (seams only) / **not wired** (no callers) / **smoke** (FFI path only, unexecuted here).

| Module | Status | Evidence |
|---|---|---|
| specification v1 | implemented | Structs + tagged enum, `deny_unknown_fields`, `ModelConfig`, committed JSON Schema, fixtures shared with Ignition; 7 Rust tests [v] |
| core/api | partial | `/v1/chat/completions` + `GET /health` [v]; `router()` applies the `/v1` nest. End-to-end never returns 200 today (see scheduler) [v] |
| core/scheduler | skeleton | `run()` closes ingress and returns Ok; no batching, no backend calls [v] |
| core/state | implemented, partial wiring | CAS table works. `begin_serving`, `end_serving`, `mark_loaded` have no callers; CLI stays `Ready` [v] |
| core/tokenizer | implemented, wired | Built by CLI from manifest; 9 tests above [v] |
| core/cache | skeleton, not wired | `HashMap` index; no callers, no eviction; not a radix tree despite name/docs [v] |
| hal/traits | implemented | Contract only; no runtime caller invokes backend methods [v] |
| hal/triage | implemented | Exact-match triage with 14 matrix tests + 2 table tests [v] |
| hal/backend_factory | partial | First candidate only; no sm120 -> sm90 -> stub fallthrough [v] |
| hal/stub | implemented as fake | Reports `sm_0`, placeholder tensors. Conflicts with "no fakes on main" rule (see Q4) [v] |
| hal/hopper_sm90 | partial | Real probe + weight upload verified on an H100 (byte-identical, 290 tensors); no KV cache, kernels or forward pass [v] |
| hal/blackwell device + ffi | smoke | Real driver calls through the shared `cuda` core; never run on a GPU (no Blackwell host) [x] |
| hal/blackwell graph | smoke | Real capture/instantiate/launch calls; never run [x] |
| hal/blackwell kernels | smoke | Prefill captures empty graph; decode argmax compares raw bf16 bits [v] |
| hal/blackwell backend | partial | `load_model_weights` returns error, so `Loaded` never reached [v] |
| helios-cli | partial | Boot path complete for tokenizer/engine; no features; sm90/sm120 tasks broken [v] |
| ignition cli | skeleton | Writes sidecar only; weight blob and signature are TODO [v] |
| ignition registries | implemented | Import-side-effect registration; circular import fixed, verified by `pip install -e ignition` + pytest [v] |
| ignition architectures | skeleton | `layers=[]`, hardcoded tokenizer dict per arch [v] |
| ignition hardware recipes | implemented (declarative) | Emitted into `hardware_recipe`; no helios code reads it [v] |

Hard-coded model dims: `HEAD_DIM = 128`, `NUM_HEADS = 32` (blackwell backend.rs), `VOCAB_SIZE` (kernels.rs). Manifest v1 now carries `ModelConfig`, but the backend does not read it yet [v].

## Unresolved questions

1. Fix Taskfile `build:sm90` / `build:sm120` (add HAL feature forwarding to helios-cli) or remove them?
2. `/v1` prefix: nest the router in `main.rs`, or change api.rs docs and clients?
3. Factory: implement sm120 -> sm90 -> stub fallthrough, or document first-match-only?
4. `StubBackend` is a fake. Allowed in default builds, or move behind a test-only feature?
5. ~~Manifest tag~~ Resolved (P0b): both sides use `"V1"`; schema + shared fixtures guard it.
6. `ModelConfig` is in spec v1. Remaining: make the SM120 backend read it instead of hard-coded dims.
7. `panic = "abort"` in release makes main.rs supervisor panic path (JoinError) dev-only. Intended?
8. ~~Ignition registry circular import~~ Resolved: verified by `pip install -e ignition` and pytest.
9. Regenerate `repomix-output.xml` before next docs wave (helios-core stale).
