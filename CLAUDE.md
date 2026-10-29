# CLAUDE.md: Hyperion

## Overview

Hyperion is a local LLM inference engine targeting multiple hardware generations and multiple model architectures. Supported hardware: Blackwell (SM120/SM121) and Hopper (H100/H200, SM90).

**Scope (owner direction, 2026-10-29): many GPUs per node and many nodes.** One GPU is the first milestone, not the scope. The order is tensor parallelism inside one node (TP 2/4/8 over NVLink), then independent engines across nodes behind a KV-aware router (data parallel; integrating as an NVIDIA Dynamo worker is the recommended first path, an owner decision), then pipeline parallel only when a target model does not fit one node. Contracts that are expensive to change later (manifest, backend trait, scheduler output, KV layout and hashing, process model) are designed rank-aware now; the kernels stay per GPU. The design items D1 to D6 are in `plans/20261028-2209-hyperion-upstream-distill-multimodel-multiinfra/phase-13-multi-gpu-multi-node.md` (the `plans/` directory is gitignored) and are **proposals until each is accepted as an ADR**; nothing in it is decided yet.

Two-stage architecture:

- **Ignition** (Python, build-time): reads weights, compiles/captures graphs, packages and signs the model into a `.hyperion` bundle. It may be ported to Rust long-term, but stays Python initially to reuse the Hugging Face/PyTorch ecosystem.
- **Helios** (Rust, runtime): loads the signed bundle, manages the KV cache, and serves an OpenAI-compatible API. No Python at runtime.

## Environment

- Build host: Windows, no CUDA toolkit or GPU. Project directory: `C:\Users\nguyendk\Documents\Projects\hyperion`.
- GPU (verified 2026-10-28 on a staging H100 pod, sm_90): driver 580.126.20 / CUDA 13.0, `nvidia/cuda:12.4.1-devel-ubuntu22.04` image (nvcc 12.4). Builds need no CUDA toolkit: the driver (`libcuda.so.1` / `nvcuda.dll`) and cuBLAS (`libcublas.so.13`, then `.so.12`; `cublas64_13.dll`, then `cublas64_12.dll`) are loaded at run time, and the library actually loaded plus the driver version are logged; nvcc is only needed to rebuild the kernel binaries. Minimum driver and toolkit versions are not established. Stream capture + `cuGraphInstantiateWithFlags` + launch work on this driver (a decode step replays bit-identical to direct launches), and in a release build a graph is 1.30x faster than direct launches at batch 1 (docs/performance-log.md).
- Default builds use the `stub` feature. Model-specific test data: `python scripts/fetch-model-fixtures.py` (see `testdata/README.md`). The `sm90` and `sm120` features build and link on this host without the CUDA toolkit; with no NVIDIA driver the backend probe fails with a typed `LoadError` and the factory falls back to the stub.

## Repository layout

```
hyperion/
├── Cargo.toml              # Rust workspace, dependency inheritance, release: fat LTO
├── Taskfile.yml
├── specification/          # Shared schemas (schema/, fixtures/v1/ shared with Ignition tests). Rust is the schema source of truth; v1 freezes at first released package (Manifest enum tagged by manifest_version)
├── ignition/               # Python (pyproject.toml, src/hyperion_ignition/): cli.py, compiler/ (registry + architectures/), hardware/ (recipes)
└── helios/crates/
    ├── helios-core/        # api.rs, cache.rs (RadixCache), scheduler.rs, state.rs, tokenizer.rs
    ├── helios-hal/         # ComputeBackend trait, backend_factory, triage, backends/{stub.rs, hopper_sm90/, blackwell_sm120/}
    └── helios-cli/         # Binary: read manifest -> probe -> triage -> load -> serve
```

## Common commands

```bash
cargo check --workspace                          # default: stub
cargo build --workspace --features stub
cargo check -p helios-hal --features sm120       # type-check only when CUDA is absent
python -m compileall ignition/src                # syntax check for Ignition
pytest ignition/tests                            # Ignition tests (pip install -e "ignition[test]")
bash scripts/build-kernels.sh                    # rebuild helios-hal/kernels/*.fatbin (needs nvcc, i.e. a GPU host); ignition/tests/test_kernel_binaries.py fails when a source changed without it
cargo test -p hyperion-specification --features schema   # manifest contract + schema drift (task schema:check)
cargo run -p hyperion-specification --features schema --bin gen-schema   # regenerate schema (task schema:gen)
```

## Architecture rules (mandatory)

1. **`helios-core` must not know what GPU it runs on.** All hardware interaction goes through the `ComputeBackend` trait in `helios-hal`.
2. **Adding new hardware = adding a new directory under `helios-hal/src/backends/`.** Do not modify `helios-core`.
3. **Adding a new model = adding a new file under `ignition/.../architectures/`** and registering it via import side-effect in `architectures/__init__.py`. No decorator scanning or metaclasses.
4. **`specification/v1` is immutable from the first released package onward.** Until then it may change. After that, breaking changes require a new `v2`; `v1` must keep working.
5. **Backend selection order** when multiple features are enabled: sm120 -> sm90 -> stub.
6. **The manifest records `target_arch` and `tensor_format`.** The runtime probes the real hardware and refuses to run unless `target_arch` matches exactly (no cross-generation forward-compat). No panics or segfaults.
7. **Hopper (SM90) does not run FP4.** The sm90 recipe forces FP8/BF16.

## Decisions already made

- **State machine:** `AtomicU8` + CAS loop, 6 states, exhaustive transition table. The API read path is lock-free.
- **RadixCache:** in-process, 256-token chunks, owns only the index and block IDs. Device memory is allocated through the HAL `allocate_kv_cache`. An LMCache-style sidecar becomes a separate crate later.
- **sm90 backend (decided 2026-10-28):** shared CUDA Driver-API core with per-arch kernel sets, cuBLAS first for the initial forward pass; no TensorRT-LLM C++ shim. First verification model is Qwen2.5-0.5B (plan phase 10).
- **Backend trait changes happen once.** Every change to `ComputeBackend` is held for a single merged change (rank-aware backends and a device group, a graph-capturable collective capability, page-index KV allocation per ADR 0002). Until then work stays inside the backends; stop and ask before touching the trait.
- **CUDA FFI:** the CUDA Driver API and cuBLAS are loaded at run time by hand-written bindings (`helios-hal/src/cuda/{loader,sys,cublas}.rs`: `dlopen`/`LoadLibraryA`, no crate, no link-time dependency), behind the `cuda` feature (enabled by `sm90` and `sm120`). Functions are resolved under explicit versioned names (`cuMemAlloc_v2`, `cuStreamBeginCapture_v2`, `cuGraphInstantiateWithFlags`, ...), and a missing library or symbol is a typed `LoadError`, never a panic. `cuda-driver-sys` was dropped (its `cuGraphInstantiate` has the pre-CUDA-12 signature). Real FFI, no emulation.
- **Chat template:** the model's own Jinja `chat_template` from `tokenizer_config.json` next to the tokenizer, rendered with `minijinja` (same whitespace settings as Hugging Face); no prompt format is hard-coded per model. Checked against `apply_chat_template` for Qwen2.5-0.5B. Moving the template (and hidden size, EOS, sampling defaults) into the manifest is a pending spec decision.
- **Tokenizer:** Hugging Face `tokenizers` 0.22 with the pure-Rust `fancy-regex` backend (0.22 requires choosing one; `onig` was dropped after producing identical token ids on a varied multilingual corpus for two real tokenizers and passing every Hugging Face reference test). 0.19 could not parse the current Qwen3 `tokenizer.json`; the real files are test fixtures fetched by script. Encoding speed matched `onig` within noise on a 2.8 MB prompt (one machine, one text shape).
- **HTTP:** chat is `POST /v1/chat/completions`; `GET /health` (root) is 200 in `Ready`/`Loaded`/`Serving`, 503 otherwise. A prompt longer than the limit is rejected with 400 (never truncated); 501 when no tokenizer is loaded (capability not wired); 503 is reserved for engine states `Uninitialized`/`Draining`/`Stopped`. `helios-cli` binds `127.0.0.1` by default (`--bind`), logs at `info` by default, and takes `--max-seq-len` (<= the model's).
- **Triage:** exact `target_arch` match only, plus the backend's own arch must equal the probed device and the manifest's `tensor_format` must be in that arch's format table (`helios-hal/src/triage.rs`, one Rust table; Ignition recipes are cross-checked against its committed JSON dump). The factory tries sm120 -> sm90 -> stub and reports every refusal. A PTX fallback, if ever needed, is an explicit manifest field, never inferred from the arch number.
- **Model parameters:** `ModelConfig` (num_layers, attention kind, num_kv_heads, head_dim, vocab_size, max_seq_len) lives in the manifest. `attention` is an enum (`mha`/`gqa`); a new KV layout such as MLA is a new variant, never inferred from the dims. HF mapping: `head_dim` from config else hidden/heads; `max_seq_len` = `max_position_embeddings`; `rope_scaling` and sliding window are refused. Package-level `tensor_format` (`bf16`/`fp8`/`fp4`) records the format the packager actually wrote (bf16 until a quantizer exists), never what the recipe allows; recipes only list `supported_formats`. Backends read it; no hard-coded model dimensions.
- **Schema source of truth:** Rust structs in `specification`, manifest is JSON. `schemars` behind a `schema` feature (not pulled into `helios-cli`) generates `specification/schema/manifest.v1.json`, committed; CI regenerates and runs `git diff --exit-code`. `#[serde(deny_unknown_fields)]` on manifest types. `manifest_version` is the string `"V1"`. Ignition validates with `jsonschema` (a real runtime dependency) before writing/signing; Rust/Python round-trip tests both ways. Signatures cover the manifest's stored bytes, never a re-serialization. **Not Protobuf**: manifest is off the hot path and read once; revisit only for a runtime IPC protocol (e.g. KV sidecar, multi-node), as a separate decision.
- **`.hyperion` package codec:** will be a dedicated crate, `helios-pkg`. A `.manifest.json` placeholder is used for now.

## Working rules

- **No mocks or fakes** on the main branch. Hardware code must be real FFI gated by `cfg`, never simulated.
- **Verification gate:** before reporting work as done, run the relevant checks and paste the real output. State clearly which parts were only type-checked and not executed.
- Surface any design decision you make on your own in your report so the user can review it.
- Ask before changing project direction or scope; do not decide unilaterally.
- Test fixtures resolve paths via `env!("CARGO_MANIFEST_DIR")`, never the cwd.
- **H100 runs follow `docs/h100-runbook.md`**: one `CARGO_TARGET_DIR` per build in any A/B comparison, the revision and the binary's sha256 printed at the start of every result, decode time measured with CUDA events inside one run.
- **No per-model test files in the engine.** Real third-party model files (HF `config.json`, `tokenizer.json`) live in `testdata/models/<id>/`, which is gitignored; `testdata/models.json` (committed) pins repo, revision and sha256, and `python scripts/fetch-model-fixtures.py` downloads them. Tests are generic over that registry and name no model; missing files skip with a visible message, `HYPERION_REQUIRE_MODEL_FIXTURES=1` makes that a failure. (The small `llama3-hf`/`qwen2-hf` configs under `ignition/tests/fixtures` predate this rule and back the committed golden manifests.)
- Avoid adding new dependencies unless needed, and give the reason when you do.
- Reply in the language the user writes in; keep code, identifiers and commit messages in English.
