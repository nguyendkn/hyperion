# Phase 07: Auto-build orchestration + Building readiness

## Context Links
- Phases 02 (RuntimeConfig), 05 (resolver/cache/BuildJob), 06 (`hyperion-build` contract), 03 (router, `/health`, `/v1/models`)
- research-03 section 4 (subprocess risks, K8s probes); `helios-core/src/state.rs` (6 states, exhaustive table; `/health` 200 in Ready/Loaded/Serving, 503 otherwise); `helios-cli/src/serve_flow.rs` (created by 02; main.rs is dispatch only)
- Old plan phase 06 (signing/trust store), phase 07 (serving)

## Overview
Priority P1. Status pending. Wave 3. `hyperion serve <ref>`: resolve -> (cache hit: load) | (miss: download, run Ignition, atomically publish bundle, load). The HTTP port is bound FIRST so orchestrators see a live process while a 15 GB build runs.

## Key Insights
- Production path = prebuilt bundle (`hyperion build` then `serve --bundle`/cached); auto-build is dev/convenience. Both use the same code.
- Python discovery order: `$HYPERION_PYTHON`, `$VIRTUAL_ENV`, `./.venv`, `python3`, `python` (Windows `py -3`); probe `python -I -m hyperion_ignition.build_cli --version-json`; compare with the Rust-embedded `EXPECTED_IGNITION_VERSION` (05); mismatch = build error with `pip install hyperion-ignition==X` (cache-hit path never runs this).
- Building state (CD5): option A (rec) CLI-level wrapper router: before the engine exists, a pre-engine router (the ONLY place `Building` exists; core `/health` and `EngineStatus` are untouched) serves `/health` 503 `{"status":"Building","stage":..,"progress":..}` + `Retry-After`, `/ping` and `/v1/models` and all other `/v1/*` 503 same shape + `Retry-After`, and, when `RuntimeConfig.enable_sglang_api` (SGLang style or flag, same condition as 09's mount), `/health_generate` and `/ready` 503 too, so vLLM/SGLang pollers never see 404 during a build; `/live` 200 for liveness (K8s `startupProbe` guidance in docs); then swaps to the real router. No change to `EngineStatus` (exhaustive table, rule intact, helios-core untouched). Option B: new `EngineStatus::Building` (touches state.rs, tests, transition table, ADR) - needs owner OK.
- Signing (CD3): old phase 06 not done. Options: (a) unsigned + loud "unsigned" log until trust store exists (rec for interim), (b) per-machine ed25519 key under cache `keys/` (0600), self-trust on first use (UX good, proves nothing about origin: never presented as provenance), (c) `HYPERION_SIGNING_KEY` for CI. Never auto-trust downloaded third-party bundles (only locally built ones).

## Requirements
Functional: (1) orchestration state machine `Resolving -> Downloading -> Building -> Verifying -> Loading -> Ready` surfaced in logs and health body; (2) subprocess driver: explicit args (no shell), `-I`, cwd = temp dir, env `HF_HUB_OFFLINE=1`, process-group kill on SIGTERM/Ctrl-C and on `--build-timeout` (default generous, configurable), stderr tail (last 50 lines) into the error, JSON-lines progress parsing, exit-code table from `specification/fixtures/build/exit-codes.json` -> messages with remedy; (3) cache publish via 05 atomic rename + lock; second process waits with log lines; (4) bad bundle (`<key>.bad` marker) -> rebuild once, never loop; (5) `hyperion build <ref> [--out]` explicit subcommand (init-container use) and `hyperion cache ls|rm|gc`; (6) after load: print `The server is ready` line variants for vLLM ("Application startup complete.") and SGLang ("The server is fired up and ready to roll!") per phase-01 verification, so launch scripts that grep logs work; (7) SIGTERM during build exits cleanly, tmp dir cleaned or marked stale; (8) all `RuntimeConfig` rejects and planned flags already stopped earlier (02, exit 2), so no download for a doomed flag set; (9) reads from `RuntimeConfig` (02): bundle, cache_dir, link_mode, chat_template, offline, build_timeout, exit_on_build_failure, target_arch, max_num_seqs; after load it sets `ApiConfig.bundle_info` (03 defines `/version` content; this phase only supplies the values).
Non-functional: files <200 lines; no panics. `target_arch` input = `RuntimeConfig.target_arch` (`--target-arch` override) else the real device probe; with no driver and no override the build fails typed (`LoadError`) naming `--target-arch`. Windows/stub path: `--target-arch` set to the stub backend's arch id (read from `triage.rs` in step 1), then triage accepts it for the stub; passed to Ignition as `--hw`. If phase 01 found no Ignition hardware recipe for that arch id, the build uses the documented override recorded in the pins doc (never a silent fallback).

## Architecture
```
main -> compat(02) -> bind port + pre-engine router (Building) -> model_source(05) resolve
   hit -> load bundle (existing path) -> swap router -> Ready
   miss -> download(05) -> autobuild::run(06 subprocess) -> publish(05) -> load -> swap
errors: typed -> log + keep serving 503 with error body for N s -> exit non-zero (configurable: --exit-on-build-failure default true)
```

## Related Code Files
Create: `helios-cli/src/autobuild/{mod.rs (replaces 02's stub; no lib.rs edit), python.rs, subprocess.rs, progress.rs, exit_map.rs, building_router.rs, publish.rs}`, `helios-cli/tests/{autobuild_subprocess.rs, building_health.rs, autobuild_e2e.rs}`, `docs/serving-in-kubernetes.md` (probes). Modify: `helios-cli/src/cli_cmds.rs` (ONLY the `run_build` body), `helios-cli/src/serve_flow.rs` (autobuild flow, after 03; main.rs untouched; no Cargo.toml edit: deps from 02's list), `docs/decisions/0013-auto-build-and-signing-policy.md` (append signing + Building sections; file created by 06), `docs/unverified-ledger.md` (append-only named section). helios-core: none (CD5 option A); if owner picks B: `helios-core/src/state.rs` + tests. Delete: none.

## Implementation Steps
1. Append to ADR 0013 (created by 06) the owner answers CD3, CD5 (and Python-at-serve acceptance). 2. `python.rs` discovery + probe tests (fake interpreter scripts are real scripts, not mocks of logic). 3. `subprocess.rs` spawn/kill/timeout with a tiny real helper script emitting JSON lines + chosen exit codes. 4. `exit_map.rs` from the shared JSON (cross-check test with the Python side). 5. `building_router.rs` + test: `/health` 503 `Building` with progress while a slow helper runs; liveness 200. 6. Wire flow in `serve_flow.rs`; fill `run_build` in `cli_cmds.rs` (02's stub; `run_cache` is 05's; main.rs untouched). 7. Signing per CD3 decision (interim a). 8. e2e on stub: serve a fixture HF-style dir through the full path twice; second run must not spawn Python (assert via env marker/no interpreter on PATH). 9. Docs: probes, offline, prebuilt-bundle path.

## Todo List
- [ ] ADR 0013 appended + CD3/CD5 answers  - [ ] python discovery  - [ ] subprocess driver  - [ ] exit map  - [ ] Building router  - [ ] flow wired  - [ ] `build` subcommand (`run_build`)  - [ ] e2e twice  - [ ] k8s doc

## Success Criteria
- First run builds, second run starts with no interpreter available (PATH stripped) and identical served model. `[RUN on stub]`
- Kill during build leaves no half bundle; next run succeeds. `[RUN]` for the Windows job-object kill path (this host); the Linux process-group kill path is `[TYPE-CHECK]` here and `[NEEDS-TOOL]` (Linux CI) to run. Tests use `CTRL_BREAK`/`TerminateJobObject` semantics on Windows, not POSIX signals.
- Each Ignition exit code 10-15 yields the documented message. `[RUN]`
- Two simultaneous `serve` of the same ref: one builds, one waits, both serve. `[RUN]`
- 7B real model build + load on H100: `[NEEDS-GPU]` phase 10.

## Test Strategy & Quality Gate
- G/W/T: G = serve from an HF ref with auto-build, Python-free on cache hit; W = twice-run, kill-safe, exit-code messages, race, Building routes; T = `autobuild_subprocess`, `building_health`, `autobuild_e2e`, e2e `serve_twice_no_python`.
- Spec: Input RuntimeConfig+env. Output serving engine or typed failure. Edge: offline + miss, local path, bundle path, Python missing, disk full, SIGTERM mid-build. Error: timeout, bad exit code, corrupt bundle.
- Pyramid: 50% unit, 40% integration (real subprocess helpers, real local HTTP for Hub), e2e `serve_twice_no_python` (`autobuild_e2e.rs`): full serve twice via the binary.
- Coverage: diff >=90% line / >=75% branch (`CMD-RUST-COV -p helios-cli`, `CMD-DIFF`, plan.md). Evidence: `cargo test -p helios-cli --test autobuild_subprocess --test building_health --test autobuild_e2e`.

## ADR Rationale
ADR 0013 (sections 2-3 appended here; section 1 written by 06). Chosen: subprocess + JSON-lines contract; CLI-level Building (core untouched); interim unsigned-with-warning. Rejected: embedding Python (breaks "no Python at runtime"); new EngineStatus (blast radius on exhaustive table) unless owner prefers; silently signing with a throwaway key (false assurance).

## Risk Assessment
Serving image lacks Python (typed error + prebuilt path documented). Probe kills during long build (startupProbe doc, liveness 200). NFS lock behaviour UNVERIFIED. Windows job objects vs Linux process groups: see Success Criteria tags.

## Security Considerations
Untrusted repo contents fed to Ignition: safetensors only, no remote code, subprocess `-I`, restricted cwd/env (whitelist env), no network (`HF_HUB_OFFLINE=1`), output confined to cache. Local-key signatures are not provenance; unsigned state visible in the `/version` `bundle.signed:false` extra (defined by 03) and in logs. Token not passed to Ignition.

## Next Steps
Phase 10 measures build time on pod. Old phase 06 replaces interim signing; `contract_revision` in key invalidates on migration.
