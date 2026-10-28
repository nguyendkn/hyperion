# Hyperion

Hyperion is a local LLM inference engine. A model is compiled ahead of time into a `.hyperion` package for one target GPU generation, then served by a native Rust runtime through an OpenAI-compatible HTTP API. The runtime has no Python dependency. Targets: Blackwell SM120/SM121 (primary) and Hopper SM90 (H100/H200, planned).

> **Status: pre-alpha skeleton.** The workspace, engine state machine, HAL trait, manifest triage, HTTP route and tokenizer wrapper exist. **No end-to-end generation works yet.** The scheduler does not run, no weights are loaded to a device, and Ignition writes a manifest sidecar only (no weight blob, no signature). The SM120 path is type-checked on a host without CUDA and has never been executed. The SM90 backend is not wired.

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
     |                              -> load weights -> serve /chat/completions
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

Ignition (not verified yet):

```sh
pip install -e ignition                     # not run yet
hyperion-compile --model llama3 --hw sm_120 --source <hf-dir> --out out/model.hyperion
                                            # not run yet; writes out/model.manifest.json only
```

The `Taskfile.yml` wraps the Cargo commands as `task build`, `task check`, `task test`. `task build:sm90` and `task build:sm120` are broken: `helios-cli` declares no Cargo features, so the `--features` flag fails.

## Runtime

`helios-cli --model <path>.hyperion [--port 8080]` is meant to read `<path>.manifest.json`, probe the device, select a backend, and serve `POST /chat/completions`. Today the boot path runs, but generation does not: the scheduler is a no-op that drops each request, so requests that pass tokenization get 503.

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
