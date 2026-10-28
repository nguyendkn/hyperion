# CLAUDE.md: Hyperion

## Overview

Hyperion is a local LLM inference engine targeting multiple hardware generations and multiple model architectures. Supported hardware: Blackwell (SM120/SM121) and Hopper (H100/H200, SM90).

Two-stage architecture:

- **Ignition** (Python, build-time): reads weights, compiles/captures graphs, packages and signs the model into a `.hyperion` bundle. It may be ported to Rust long-term, but stays Python initially to reuse the Hugging Face/PyTorch ecosystem.
- **Helios** (Rust, runtime): loads the signed bundle, manages the KV cache, and serves an OpenAI-compatible API. No Python at runtime.

## Environment

- Build host: Windows, no CUDA toolkit or GPU. Project directory: `C:\Users\nguyendk\Documents\Projects\hyperion`.
- Default builds use the `stub` feature. The `sm120` cargo feature can only be type-checked on this host; the link step needs the CUDA toolkit.

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
- **CUDA FFI:** `cuda-driver-sys` v0.3 (Driver API, includes the graph API), wrapped in `#[cfg(feature = "sm120")]` (the only feature linking CUDA so far). Real FFI, no emulation.
- **Tokenizer:** Hugging Face `tokenizers` crate with the `onig` feature enabled. This is currently a workaround so the crate links with `tokenizers` 0.19; re-check against newer versions.
- **HTTP status:** 501 when no tokenizer is loaded (capability not wired); 503 is reserved for engine states `Uninitialized`/`Stopped`.
- **Triage:** exact `target_arch` match only. Ignition compiles per target. A PTX fallback, if ever needed, is an explicit manifest field, never inferred from the arch number.
- **Model parameters:** `ModelConfig` (num_layers, num_kv_heads, head_dim, vocab_size, max_seq_len, ...) lives in the manifest. Backends read it; no hard-coded model dimensions.
- **Schema source of truth:** Rust structs in `specification`, manifest is JSON. `schemars` behind a `schema` feature (not pulled into `helios-cli`) generates `specification/schema/manifest.v1.json`, committed; CI regenerates and runs `git diff --exit-code`. `#[serde(deny_unknown_fields)]` on manifest types. `manifest_version` is the string `"V1"`. Ignition validates with `jsonschema` (a real runtime dependency) before writing/signing; Rust/Python round-trip tests both ways. Signatures cover the manifest's stored bytes, never a re-serialization. **Not Protobuf**: manifest is off the hot path and read once; revisit only for a runtime IPC protocol (e.g. KV sidecar, multi-node), as a separate decision.
- **`.hyperion` package codec:** will be a dedicated crate, `helios-pkg`. A `.manifest.json` placeholder is used for now.

## Working rules

- **No mocks or fakes** on the main branch. Hardware code must be real FFI gated by `cfg`, never simulated.
- **Verification gate:** before reporting work as done, run the relevant checks and paste the real output. State clearly which parts were only type-checked and not executed.
- Surface any design decision you make on your own in your report so the user can review it.
- Ask before changing project direction or scope; do not decide unilaterally.
- Test fixtures resolve paths via `env!("CARGO_MANIFEST_DIR")`, never the cwd.
- Avoid adding new dependencies unless needed, and give the reason when you do.
- Reply in the language the user writes in; keep code, identifiers and commit messages in English.
