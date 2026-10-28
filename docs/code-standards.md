# Code Standards

Last updated: 2026-10-28. Sources: `CLAUDE.md`, `repomix-output.xml` snapshot (same day). Not re-verified by build.

Status tags: **[enforced]** code matches · **[pending]** decided, code not there yet · **[drift]** code contradicts decision (fix code or decision; flagged, not silently rewritten).

## 1. Architecture rules (mandatory, from CLAUDE.md)

1. **`helios-core` must not know what GPU it runs on.** All hardware interaction goes through the `ComputeBackend` trait in `helios-hal`. [enforced]
2. **Adding new hardware = adding a new directory under `helios-hal/src/backends/`.** Do not modify `helios-core`. [enforced for core; note: factory + Cargo + build.rs also change, see §10]
3. **Adding a new model = adding a new file under `ignition/.../architectures/`** and registering it via import side-effect in `architectures/__init__.py`. No decorator scanning or metaclasses. [enforced]
4. **`specification/v1` is immutable from the first released package onward.** Until then it may change. After that, breaking changes require a new `v2`; `v1` must keep working. [pending: no package released yet]
5. **Backend selection order** when multiple features are enabled: sm120 -> sm90 -> stub. [enforced in `backend_factory::create_backend`]
6. **The manifest records `target_arch` and `tensor_format`.** The runtime probes the real hardware and refuses to run unless `target_arch` matches exactly (no cross-generation forward-compat). No panics or segfaults. [drift: `triage.rs` allows `Downgrade`; v1 `Manifest` has no `tensor_format`]
7. **Hopper (SM90) does not run FP4.** The sm90 recipe forces FP8/BF16. [enforced in `ignition/.../hardware/sm90_hopper.py`: `fp4: disabled`]

## 2. Decisions already made

| Decision | Status |
|---|---|
| State machine: `AtomicU8` + CAS loop, 6 states, exhaustive transition table, lock-free read path | [enforced] `helios-core/src/state.rs` |
| RadixCache: in-process, 256-token chunks, owns index + block IDs only, memory via HAL `allocate_kv_cache` | [enforced] chunk const in `cache.rs`; ownership not audited |
| CUDA FFI: `cuda-driver-sys` v0.3, Driver API, behind `#[cfg(feature = "sm120")]`, real FFI | [enforced] `helios-hal/src/backends/blackwell_sm120/` |
| Tokenizer: HF `tokenizers` 0.19 with `onig` feature (workaround; re-check on upgrade) | [enforced] `helios-core/Cargo.toml` |
| HTTP status: 501 when no tokenizer; 503 only for `Uninitialized`/`Stopped` | [drift] `api.rs` also returns 503 for queue-full/send-failure; `scheduler.rs` docs mention 429 |
| Triage: exact `target_arch` match only; PTX fallback only as explicit manifest field | [drift] `triage.rs` implements `Downgrade`; `lib.rs` docs say forward-compat |
| Model parameters (`ModelConfig`: layers, kv heads, head_dim, vocab, max_seq_len) in manifest; no hard-coded dims | [partial] `ModelConfig` is in the manifest (P0b); `blackwell_sm120/backend.rs` still hard-codes `HEAD_DIM`, `NUM_HEADS`; `kernels.rs` hard-codes `VOCAB_SIZE` |
| Schema source of truth = Rust structs; JSON Schema via `schemars`; Rust/Python round-trip test | [done] `schemars` behind feature `schema`; `specification/schema/manifest.v1.json` committed, a test fails on drift; Ignition validates with `jsonschema`; shared fixtures in `specification/fixtures/v1/` |
| `.hyperion` codec = dedicated crate `helios-pkg`; `.manifest.json` sidecar until then | [pending] `helios-cli/src/main.rs` reads JSON sidecar |
| Ed25519 signature over weight blob, verified at load | [pending] field exists; `cli.py` has TODO; `ed25519-dalek` in workspace |

## 3. Working rules (from CLAUDE.md)

- **No mocks or fakes** on the main branch. Hardware code = real FFI gated by `cfg`, never simulated. [drift risk: default `stub` backend returns zeroed data, see §12]
- **Verification gate:** before reporting work done, run relevant checks and paste real output. State which parts were only type-checked and not executed.
- Surface any design decision made on your own in the report so the user can review it.
- Ask before changing project direction or scope; do not decide unilaterally.
- Test fixtures resolve paths via `env!("CARGO_MANIFEST_DIR")`, never the cwd. [drift: `encode_path_smoke.rs` uses relative `tests/fixtures/...`]
- Avoid new dependencies unless needed; give the reason when adding one.
- Reply in the user's language; code, identifiers and commit messages in English.

## 4. Rust conventions

- **Edition/versions:** `edition = "2021"`, `rust-version = "1.75"`, license Apache-2.0, all from `[workspace.package]` via `field.workspace = true`.
- **Dependencies:** declared once in root `[workspace.dependencies]`; crates use `dep = { workspace = true }`. Feature additions per crate are allowed (`tokenizers` + `onig`). [drift: `helios-cli` `clap`, `helios-hal` `bindgen`/`cc` pinned locally]
- **Errors:** `anyhow::Result` for backend/app paths; `thiserror` for typed errors that callers branch on (`TriageError`, `ManifestError`). Use `eval_cuda("<driver fn>", call)?` for every CUDA Driver call; it attaches the driver error string and the call label.
- **Logging:** `tracing` only (no `println!` in library code). Structured fields: `tracing::info!(model = %id, target = ?x, "msg")`. `EnvFilter::from_default_env()` in the CLI.
- **Hardware gating:** backend code under `#[cfg(feature = "...")]`; feature-off builds must compile with zero native linkage (`stub` default).
- **No panics at runtime:** return `Result`, map errors to HTTP status codes; never `unwrap()` on request/load paths. Existing exceptions to review (see §12): mutex-poison `expect`s in `blackwell_sm120/backend.rs`, `expect` in `state.rs`, signal-handler `expect`s in `main.rs`.
- **Release profile:** fat LTO, `codegen-units = 1`, `panic = "abort"`. A panic kills the process, so the no-panic rule is load-bearing.
- **unsafe / FFI:**
  - `unsafe` confined to `helios-hal/src/backends/<arch>/` (`device.rs`, `graph.rs`, `kernels.rs`, `ffi.rs`). `helios-core` has none.
  - Every `unsafe impl Send` carries a `// Safety:` comment stating the thread-ownership argument.
  - Real FFI only. No simulated driver calls, no fake pointers in sm120 code.
  - Driver API (not runtime API), because graph capture lives in the driver.
- **Docs:** `//!` module docs state purpose and invariants; `///` on public items. Keep files under 200 lines (global rule); split modules when they grow.
- **Stability:** `ComputeBackend` is ABI-like. New trait methods are `default`-implemented first; required only after every backend implements them.

## 5. Python conventions (Ignition)

- **Tooling:** `ruff` `line-length = 100`, `target-version = "py311"`; `requires-python = ">=3.11"`; build via hatchling; CLI via `click`.
- **Module style:** `from __future__ import annotations`; module docstring on every file; `dataclass` for data records; `typing.Protocol` for plugin interfaces.
- **Registry by import side-effect:** each architecture/recipe module calls `register(...)` at import. Aggregator `__init__.py` imports each module with `# noqa: F401`. Bottom-of-file imports in registries carry `# noqa: E402,F401`.
- **Forbidden:** decorator-based scanning, metaclasses, `importlib` directory walking.
- **Shape of output:** `CompiledManifest.to_json_dict()` emits snake_case keys matching the Rust `v1::Manifest` field names exactly.

## 6. Naming

| Item | Convention | Example |
|---|---|---|
| Rust crate (package) | kebab-case | `helios-hal`, `hyperion-specification` |
| Rust lib name | snake_case | `helios_hal` |
| Rust files/modules/fns | snake_case | `backend_factory.rs`, `eval_cuda` |
| Rust types / traits | PascalCase | `BlackwellBackend`, `ComputeBackend` |
| Rust consts | SCREAMING_SNAKE | `CHUNK_TOKENS`, `CU_STREAM_NON_BLOCKING` |
| Backend dirs | `<generation>_sm<NN>/` | `blackwell_sm120/`, `hopper_sm90/`; single-file `stub.rs` |
| Cargo features | short arch tag | `stub`, `sm90`, `sm120`, `multi` |
| `target_arch` strings | `sm_<NN>` with underscore | `sm_120`, `sm_90` |
| Python modules | snake_case | `sm120_blackwell.py`, `sm90_hopper.py` |
| Python classes | PascalCase + role suffix | `LlamaCompiler`, `Sm120BlackwellRecipe` |
| Model ids | lowercase, no dash | `llama3`, `qwen2`, `deepseek` |
| JSON / manifest fields | snake_case = Rust field names | `model_id`, `byte_offset` |
| Docs files | kebab-case | `code-standards.md` |

## 7. Testing

- **Real data only.** No mocks, fakes or stubbed responses in tests that claim to verify behavior. Hardware paths are tested only where the real device exists.
- **Layout:** Rust integration tests in `<crate>/tests/*.rs`; fixtures in `<crate>/tests/fixtures/`.
- **Fixture path:** `concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/<file>")` or `Path::new(env!("CARGO_MANIFEST_DIR")).join(...)`. Never the cwd.
- **Async tests:** `#[tokio::test]`.
- **Run:** `cargo test --workspace` (`task test`).
- **Fail loudly:** a fixture load error should fail the test with the loader's message, not a downstream assertion.

## 8. Verification gate

Before reporting work as done:
1. Run the relevant checks, in this order when applicable:
   ```bash
   cargo check --workspace
   cargo test --workspace
   cargo check -p helios-hal --features sm120   # type-check only; no CUDA here
   python -m compileall ignition/src
   ruff check ignition/src                      # not wired in Taskfile yet
   ```
2. Paste the real output into the report. No paraphrased results.
3. State which parts were **type-checked only** (sm120 path) vs **executed**.
4. List design decisions made on your own.

## 9. Dependency policy

- Add a dependency only when needed; state the reason in the report.
- Prefer `workspace.dependencies`; add to root `Cargo.toml` and reference with `workspace = true`.
- Build-only deps (`bindgen`, `cc`) stay behind the feature that needs them.
- Native CUDA deps must not be enabled on hosts without CUDA; those hosts build with `stub`.

## 10. Recipes

### 10a. Adding hardware (new GPU generation)

1. Create `helios/crates/helios-hal/src/backends/<generation>_sm<NN>/` with `mod.rs` and `backend.rs` implementing `ComputeBackend`. Put FFI in `device.rs`/`ffi.rs`/`kernels.rs` as needed. Use `eval_cuda` for every driver call.
2. `helios-hal/Cargo.toml`: add feature `sm<NN> = [...]` with its optional deps; add to `multi` if it should ship in distributable builds.
3. `helios-hal/src/backends/mod.rs`: `#[cfg(feature = "sm<NN>")] pub mod <dir>;`
4. `helios-hal/build.rs`: add `wire_sm<NN>()` if native linkage is needed; add the feature to the `rustc-check-cfg` value list (currently hard-coded `"sm90","sm120","multi"`).
5. `helios-hal/src/backend_factory.rs`: add a `BackendKind` variant with `#[cfg]`, a `target_arch` arm, a `candidates` entry at the correct precedence position (newest first), and an `instantiate` arm.
6. Optional: forward the feature in `helios-cli/Cargo.toml` (it currently has no `[features]`).
7. Ignition: add `ignition/src/hyperion_ignition/hardware/<name>.py` with a class that has `target_arch` and `constraints()`, and `register("sm_<NN>", Recipe())` at module bottom. Add `from . import <name>  # noqa: E402,F401` to the bottom of `hardware/registry.py`.
8. Do **not** edit `helios-core`.
9. Verify: `cargo check --workspace`, `cargo check -p helios-hal --features sm<NN>`, `python -m compileall ignition/src`.

### 10b. Adding a model architecture

1. Create `ignition/src/hyperion_ignition/compiler/architectures/<family>.py`:
   ```python
   class <Family>Compiler:
       model_id = "<id>"
       def compile(self, *, source: Path, recipe: HardwareRecipe) -> CompiledManifest: ...

   register("<id>", <Family>Compiler)
   ```
2. Add `from . import <family>  # noqa: F401` to `compiler/architectures/__init__.py`.
3. Do **not** edit existing architecture files or `compiler/base.py`.
4. Do not put model dimensions in Rust code; they belong in the manifest (`ModelConfig`; Rust backends are not wired to it yet).
5. Verify: `python -m compileall ignition/src`; `hyperion-compile --model <id> --hw sm_90 --source <dir> --out <file>`.

## 11. Commands

```bash
cargo build --workspace --features stub
cargo check -p helios-hal --features sm120     # type-check only without CUDA
pip install -e ignition                        # task ignition:install
hyperion-compile --model llama3 --hw sm_90 --source <dir> --out out/model.hyperion
```

## 12. Known drift and gaps

| # | Item | Where |
|---|---|---|
| D1 | Triage allows downgrade; decision says exact match | `helios-hal/src/triage.rs`, `lib.rs` docs |
| D2 | 503 used for queue-full; decision reserves 503 for engine states | `helios-core/src/api.rs` |
| D3 | Hard-coded model dims (`HEAD_DIM`, `NUM_HEADS`, `VOCAB_SIZE`) | `blackwell_sm120/backend.rs`, `kernels.rs` |
| D4 | Fixture path relative to cwd | `helios-core/tests/encode_path_smoke.rs` |
| D5 | `task build:sm90` / `build:sm120` pass `--features` to `helios-cli`, which has no `[features]` | `Taskfile.yml` |
| D6 | `backend_factory.rs` docs say "add file, don't edit"; adding hardware also edits factory, `build.rs`, `Cargo.toml` | `helios-hal/src/backends/mod.rs`, `backend_factory.rs` |
| D7 | `compiler/base.py` docstring says register in `registry.py`; real registration is `architectures/__init__.py` | `ignition/.../compiler/base.py` |
| D8 | Stub backend returns zeroed tensors and `sm_0`; default build depends on it | `helios-hal/src/backends/stub.rs` |
| D9 | Workspace-dep inheritance not used by `clap` (cli), `bindgen`/`cc` (hal) | crate `Cargo.toml` files |
| D10 | `api.rs` ~300 lines, over 200-line guideline | `helios-core/src/api.rs` |
| D11 | Taskfile `build:sm120` desc mentions NVRTC; no NVRTC use in code | `Taskfile.yml` |

## 13. Unresolved questions

1. Does the `stub` backend count as a "fake" under the no-mocks rule? If yes, what replaces it for the default build?
2. Are `expect()` on mutex poisoning, `state.rs` invariant, and signal-handler install acceptable on runtime paths, or must they become `Result`?
3. Triage: exact-match (CLAUDE.md) or forward-compat (`lib.rs`)? Pick one, then fix code or doc.
4. Backpressure status code: 503 or 429 for queue-full?
5. Coverage threshold and gate: not defined here (`workflows/quality-gate.md` §5 not read in this wave).
6. No `rustfmt.toml` / `clippy` config and no `ruff` task in `Taskfile.yml`. Add, or leave?
7. Is the 200-line file limit from the global CLAUDE.md meant to apply to this project?
