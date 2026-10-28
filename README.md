# Hyperion: local LLM inference engine in Rust for NVIDIA Blackwell and Hopper GPUs

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Rust](https://img.shields.io/badge/runtime-Rust-orange)
![Status](https://img.shields.io/badge/status-pre--alpha-red)

Hyperion is a local LLM inference engine with an OpenAI-compatible API, built for NVIDIA CUDA GPUs (Blackwell SM120, Hopper SM90). A model is compiled ahead of time into a `.hyperion` package for one target GPU generation, then served by a native Rust runtime through an OpenAI-compatible HTTP API. The runtime has no Python dependency. Targets: Blackwell SM120/SM121 (primary) and Hopper SM90 (H100/H200, planned).

> **Status: pre-alpha.** One path works end to end, verified on an H100: `helios-cli --features sm90` serves Qwen2.5-0.5B and answers `POST /v1/chat/completions` with text whose tokens equal Hugging Face's greedy output (the model's own chat template, non-streaming, one request at a time, greedy or temperature/top-p sampling; activations and the K/V cache stay on the GPU: ~12 ms/token on an H100 for this 0.5B model, measured in a hardware test). Everything else (other models, sm120, batching, device KV cache, sampling, streaming) is not implemented; see `docs/unverified-ledger.md` for exactly what was run and what was not.

## Architecture

```
HF checkpoint
     |
     v
Ignition (Python, build time)       reads weights, applies hardware recipe,
     |                              compiles graph, packages .hyperion
     v
<out>.hyperion + <out>.manifest.json   (manifest sidecar is the only file read today)
     |
     v
Helios (Rust, run time)             read manifest -> probe device via HAL -> triage
     |                              -> load weights -> serve /v1/chat/completions
     v
OpenAI-compatible HTTP API
```

- **Ignition** (`ignition/`, Python): build-time compiler. Stays in Python to reuse the Hugging Face and PyTorch ecosystem. Model architectures and hardware recipes are plugins registered by import.
- **Helios** (`helios/`, Rust workspace): runtime. `helios-core` is device-agnostic. Hardware sits behind the `ComputeBackend` trait in `helios-hal`, with one directory per GPU generation.
- **Specification** (`specification/`): the manifest schema. Rust structs are the source of truth. `v1` is not frozen until the first released package.

Triage is an exact `target_arch` match (`sm_120` package on `sm_120` device). The runtime refuses mismatches with an error, not a panic. Code still allows a downgrade path, which conflicts with this rule (see the docs below).

## Layout

```
hyperion/
├── specification/              # Manifest schema (Rust, v1 pending freeze)
├── ignition/                   # Build-time compiler (Python, package hyperion_ignition)
│   └── src/hyperion_ignition/  #   cli.py, compiler/ (architectures/), hardware/ (recipes)
└── helios/crates/
    ├── helios-core/            # API, scheduler, engine state, tokenizer, KV cache index
    ├── helios-hal/             # ComputeBackend trait, backend factory, triage, backends
    │   └── src/backends/
    │       ├── stub.rs             # placeholder backend, default build
    │       ├── blackwell_sm120/    # CUDA Driver FFI, behind feature `sm120`
    │       └── hopper_sm90/        # not wired, behind feature `sm90`
    └── helios-cli/             # Binary: reads manifest, triage, serve
```

## Quick start

Requirements: Rust (edition 2021, rust-version 1.75+). Python 3.11+ only for Ignition.

Rust, stub backend (no GPU or CUDA needed). These pass on the current tree:

```sh
cargo check --workspace
cargo build --workspace
cargo test --workspace             # 9 tests: tokenizer unit + tokenizer smoke fixture
```

Blackwell SM120 type-check only. Linking needs the CUDA toolkit, so this does not produce a runnable binary:

```sh
cargo check -p helios-hal --features sm120
```

Ignition (verified: installs and runs, 21 pytest pass; writes the manifest only, no weights):

```sh
pip install -e "ignition[test]"
hyperion-compile --model llama3 --hw sm_120 --source <hf-dir> --out out/model.hyperion
                                            # <hf-dir> must hold a Hugging Face config.json
pytest ignition/tests
```

The manifest schema is generated from the Rust structs and committed:

```sh
task schema:gen     # rewrite specification/schema/manifest.v1.json
task schema:check   # fail if it is stale
```

The `Taskfile.yml` wraps the Cargo commands as `task build`, `task check`, `task test`. `task build:sm90` and `task build:sm120` are broken: `helios-cli` declares no Cargo features, so the `--features` flag fails.

## Runtime

`helios-cli --model <path>.hyperion [--port 8080] [--bind 127.0.0.1] [--max-seq-len N]` reads `<path>.manifest.json`, probes the device, triages (exact arch + tensor_format), selects a backend, loads the weights (`model.safetensors` and `config.json` next to the package until `helios-pkg` exists) and serves `POST /v1/chat/completions` plus `GET /health`. With the stub backend (default build) generation answers HTTP 500 "this backend cannot generate yet"; with `--features sm90` on an H100 it generates.

## Documentation

- [docs/project-overview-pdr.md](docs/project-overview-pdr.md): goals, non-goals, requirements with status, milestones, known gaps.
- [docs/system-architecture.md](docs/system-architecture.md): two-stage flow, HAL contract, engine state machine, API, SM120 graph path.
- [docs/code-standards.md](docs/code-standards.md): architecture rules, Rust and Python conventions, naming, testing, recipes for adding hardware or models.
- [docs/codebase-summary.md](docs/codebase-summary.md): module map, feature flags, build commands, module status.
- [CLAUDE.md](CLAUDE.md): decisions and working rules for this repository.

## Contributing

- `helios-core` must not reference any GPU. New hardware goes in a new directory under `helios-hal/src/backends/`.
- New models go in a new file under `ignition/src/hyperion_ignition/compiler/architectures/`.
- No mocks or simulated hardware on `main`. Hardware code is real FFI behind a Cargo feature.
- Before reporting a change as done, run `cargo check --workspace` and `cargo test --workspace`.

## License

Apache-2.0. `Cargo.toml` and `ignition/pyproject.toml` declare it. The repository has no `LICENSE` file yet.
