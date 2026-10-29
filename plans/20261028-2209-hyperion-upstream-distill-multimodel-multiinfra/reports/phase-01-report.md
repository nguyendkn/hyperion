# Phase 01 report (2026-10-28) — implemented, NOT committed

Verified here (Windows): `cargo test --workspace --features stub` 45+ tests green (hal 14 matrix + 2 table, core 5+7+4+3, cli 6 unit + 4 e2e, spec 6); `pytest ignition/tests` 43 passed; `cargo check --workspace --all-targets`; `cargo check -p helios-hal|helios-cli --features sm120`; `cargo test -p hyperion-specification --features schema`.
Verified on H100 pod (Linux, sm_90): same Rust suite green (51 tests); `helios-cli --features sm120` links with CUDA 12.4 and refuses `sm_90`/`sm_120` packages (Blackwell backend on an sm_90 device -> `BackendArchMismatch`, exit 1, no panic, GPU memory 0 MiB).

## Coverage (CMD-RUST-COV / CMD-PY-COV / CMD-DIFF)
- Rust diff line coverage 95% (diff-cover on lcov, `origin/main...HEAD` incl. unstaged; untracked new test files excluded). triage.rs 100% line + 100% branch; backend_factory.rs 93% line.
- Rust branch gate on the DIFF: UNVERIFIED. `cargo +nightly llvm-cov --branch` works here (54 BRDA records) but diff-cover on lcov is line-only; whole-file branch for main.rs 33% / api.rs 17% is dominated by pre-existing code (shutdown path, chat handler).
- Python: diff 100% lines; total 97% line+branch.

## Ledger rows (now written to docs/unverified-ledger.md, seeded early at the user's request; phase 08 still owns its structure)
- [RUN on H100] sm120 backend links against CUDA 12.4 toolkit (driver 580.126.20 / CUDA 13.0) via `helios-cli --features sm120`.
- [RUN on H100] real device probe reports `sm_90`; Blackwell backend refused on it; SIGTERM shutdown returns GPU memory to 0 MiB; idle context ~518 MiB.
- [NEEDS-GPU] sm90 backend: `initialize_device` now returns an error (no probe wired); TensorRT-LLM/bindgen path unbuilt.
- [NEEDS-GPU] any kernel / forward pass; KV VRAM check against a real device total (only unit-tested with synthetic totals).
- [TYPE-CHECK] `--features sm90` not type-checked in this phase.

## H100 re-run on committed code 3294d70 (2026-10-29)
Full Rust suite green on Linux with real fixtures required; HTTP 400 for a 60003-token prompt; fall-through sm120 -> stub; KV/seq matches hand calc. New defect: refused SM120 backend held a ~527 MiB CUDA context while the stub served (no `Drop`); fixed in `backend_factory.rs` (call `shutdown()` on refusal), verified on the H100: 4 MiB while serving (was 527), 0 MiB after exit; still uncommitted. Phase 05 should also add `Drop` for the SM120 device.

## Pulled forward from phase 02 (user goal item 4)
Prompt longer than the limit -> `PromptTooLong` -> HTTP 400 (was silent truncation + 503); tests `helios-core/tests/prompt_limit.rs` (3) + generic `model_tokenizers.rs`. Phase 02 keeps `spawn_blocking`, in-flight counting, OpenAI JSON errors, the remaining tests.

## Decisions I made (review)
1. `/v1` nest lives in `helios_core::api::router()` (not main.rs) so every caller gets one public path.
2. `create_backend(&Manifest)` now takes the manifest and returns `CreatedBackend{backend, device_arch, triage, hardware}` (needed for format check + VRAM); main.rs is the only caller.
3. sm90 backend no longer reports a hard-coded `sm_90`; it errors until a real probe exists (sm90 is refused, not faked).
4. Stub accepts every tensor_format; unknown archs have no formats (refused).
5. `TriageOutcome` keeps a single `Exact` variant; `Downgrade`/`UpgradeRequired` removed (exact match only). Added `NoGpuBackend` for stub-only builds.
6. `tokenizers` 0.22 keeps `onig` (0.22 requires a regex backend; `fancy-regex` not evaluated).
7. VRAM check = one sequence's KV must fit total VRAM (hard impossibility only); skipped when the backend reports 0 VRAM (stub). Not a policy for weights/headroom.
8. Real model files (Qwen3 config + 11 MB tokenizer) are NOT in git: `testdata/models/` is gitignored, pinned in `testdata/models.json`, fetched by `scripts/fetch-model-fixtures.py`; tests are generic over the registry and skip visibly when files are absent (`HYPERION_REQUIRE_MODEL_FIXTURES=1` makes that a failure). Rule recorded in CLAUDE.md. llama3/qwen2 configs stay committed (tiny; back the committed golden manifests).
9. Removed unused deps: hal `serde`, core `hyperion-specification` + `tracing`, workspace `bytes`; `sha2`/`ed25519-dalek` kept (phase 06). `cc` left optional ([build-dependencies] move not needed: `cc::Build` is commented out in build.rs).
10. I installed cargo-llvm-cov 0.9.1 and llvm-tools (stable + nightly) per step 1, and also cargo-machete 0.9.2 (not in plan) to find unused deps.
