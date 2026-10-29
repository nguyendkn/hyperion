# Spec: Hyperion upstream-distill roadmap (phases 01-08)

See `plan.md` (phase table, waves, open decisions) and `phase-*.md` (requirements, steps, test strategy). Phases 10-13 (GPU track, tiered KV store, multi-GPU/multi-node) are tracked in plan.md and are not part of this spec's success criteria. Ideas from upstream only: never copy its code/text.

## Success criteria (verifiable)
- `cargo test --workspace --features stub` and `cargo check --workspace` exit 0; `cargo test -p helios-hal` >= 12 tests (arch x format matrix); `pytest ignition/tests` green; `cargo test -p hyperion-specification --features schema` green (schema drift test).
- Triage refuses a backend whose own arch differs from the probed GPU and any manifest whose `target_arch` / `tensor_format` the backend does not support; helios-cli exits non-zero without panic.
- helios-cli: chat route at `/v1/chat/completions`, `GET /health` (503 unless Ready/Serving), bind 127.0.0.1 by default, `--max-seq-len` runtime limit; over-long prompt -> HTTP 400 (no silent truncation).
- Manifest v1 stays additive-only until the first released package; no model dimensions hard-coded in Rust backends; no per-arch flags in helios-core.
- Every success criterion not runnable on this Windows/no-GPU host is tagged `[TYPE-CHECK]`/`[NEEDS-GPU]` in `docs/unverified-ledger.md`.

## Constraints
- No mocks/fakes on main; hardware code is real FFI behind `cfg`.
- Open decisions in `plan.md` (1,3,5,6,7 gate phase 03) are the user's: ask, never decide.

## Verification
`cargo test --workspace --features stub && pytest ignition/tests && cargo test -p hyperion-specification --features schema`
