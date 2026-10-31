# Phase 05: Model resolver, HF downloader, bundle cache

## Context Links
- `research/researcher-03-hf-fetch-build-cache.md` (sections 1-2); phase-01 verified HF facts
- ADR 0003 (payload table, bare file names), 0005 (TP at load, one package per arch), 0010 (`qualified_checkpoint`, `contract_revision`)
- `helios-cli/src/main.rs` (`load_manifest` = `<pkg>.manifest.json` sidecar, `resolve_in_package`)

## Overview
Priority P1. Status pending. Wave 1b (starts after 02 merges: needs 02's `[lib]`, `RuntimeConfig` and Cargo.toml; parallel with 06 only; no main.rs/lib.rs edit, replaces 02's empty `model_source/mod.rs` stub). Turns `model_ref` (hf-id | local dir | local bundle) into either a ready bundle dir (cache hit, zero Python) or a fully downloaded source snapshot + the cache key for a build (phase 07 runs the build).

## Key Insights
- Cache key needs the resolved commit sha, so sha resolution (one API call, or `refs/` offline) is Rust-side; "bundle exists -> serve" fast path must be Python-free.
- Key excludes `--max-model-len` and TP (ADRs 0005/0010). Includes: repo+commit (or local fingerprint), Ignition version (a Rust-embedded `EXPECTED_IGNITION_VERSION` constant in `cache_key.rs`, so the cache-hit path never starts Python; phase 07 checks it against `hyperion-build --version-json` before any build and a mismatch is a build error; test asserts the constant equals the `ignition/pyproject.toml` version), `contract_revision`, `manifest_version`, `target_arch` (`RuntimeConfig.target_arch`, resolved by 07: probe or `--target-arch`), `tensor_format`, dtype/quant request, chat-template override.
- Fixed bundle file name `bundle.hyperion` (avoid `with_suffix` mangling of dotted names like `qwen2.5-7b`, research-03 section 3).

## Requirements
Functional: (1) resolver classifies input: existing `*.hyperion` / dir with manifest -> Bundle; dir with `config.json`+safetensors -> LocalSource; `org/name[@rev]` -> Hf; else clear error listing what was tried. (2) Hf: pin rev->sha, list siblings (+sizes/sha256), select files (config, generation_config, tokenizer.json, tokenizer_config.json, safetensors or index+shards; skip bin/pt/gguf/onnx), disk check, resumable download (`.incomplete` + Range), sha256 verified while streaming (feeds CD4), progress events. (3) offline (`HF_HUB_OFFLINE=1` or `--offline`): never touch network; use `refs/` + snapshots. (4) HF env: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `$HF_HOME/token`, `HF_ENDPOINT`, `HF_HUB_CACHE`. (5) Reads an existing HF/vLLM cache layout zero-network. (6) Cache: `$HYPERION_CACHE/bundles/<org>--<name>-<key16>/`; atomic build dir rename; advisory lock; `BUILD.json` (inputs, versions, sizes, last_used); `cache ls|rm|gc --max-size --older-than`; never auto-delete at serve; stale `.tmp-*` >24h cleaned. (7) Gated repo error -> message "accept licence at huggingface.co/<id>, set HF_TOKEN". Non-functional: files <200 lines; Windows symlink fallback chain symlink->hardlink->copy.

## Architecture
```
model_ref -> Resolver -> Bundle(path) ------------------------------> serve (no Python)
                      -> LocalSource(dir) -> fingerprint(headers hash) -\
                      -> Hf(id,rev) -> Fetcher(trait: Http | Offline) -> snapshot dir (HF layout) -/-> key16 -> cache lookup -> hit: bundle dir | miss: BuildJob{source dir, key, out tmp dir} -> phase 07
```
`Fetcher` trait with `HttpFetcher` (real) and `OfflineFetcher`; tests use a REAL local HTTP server (hyper/axum on 127.0.0.1) serving fixture bytes with Range/redirect/403 behaviours: a named test double of the Hub protocol, results tagged `[RUN on local HTTP server]`, plus one `[NEEDS-NETWORK]` test against the real Hub.

## Related Code Files
Create: `helios-cli/src/model_source/{mod.rs (replaces 02's stub), resolve.rs, hf_api.rs, download.rs, cache_layout.rs, cache_key.rs, lock.rs, disk.rs, gc.rs, build_job.rs}`; `helios-cli/tests/{model_resolve.rs, hf_download_local_server.rs, cache_key.rs, cache_lock.rs}`; `docs/decisions/0012-model-source-and-bundle-cache.md`.
Modify: `helios-cli/src/cli_cmds.rs` (ONLY the `run_cache` body), `helios-cli/Cargo.toml` (ONLY one HTTP-client dependency line per CD1, appended after 02 has merged; `sha2`, `fs4`, `libc`/`windows-sys` are already in 02's CD12 list; reason in ADR 0012). Delete: none.

## Implementation Steps
1. ADR 0012 with owner answers: CD1 downloader (A: hand-written `reqwest` blocking+rustls behind `Fetcher`, rec; B: `hf-hub` crate; C: Python `snapshot_download` through Ignition, simplest but needs Python before the key is known), CD2 write layout (rec: write HF layout so vLLM/SGLang/Python tools share the cache; bundles in own dir), CD4 trust LFS sha256 (rec: hash during download, pass `--known-sha256` to Ignition: no re-hash, no blind trust). Block steps 4+ until answered. The Cargo.toml HTTP-client edit happens only after 02 is merged (ordering in plan.md matrix). 2. `resolve.rs` + tests with temp dirs. 3. `cache_key.rs`: canonical string -> sha256; golden vectors; test "changing max-seq-len/TP leaves key unchanged; changing any listed input changes it". 4. `hf_api.rs` revision/siblings parsing against recorded JSON from phase 01. 5. `download.rs`: HEAD, ranged resume, sha256 streaming, `Authorization` stripped on cross-host redirect (test with two local servers), retries with backoff, progress callback. 6. `disk.rs` (`fs4`/`statvfs`) exact-deficit error. 7. `lock.rs` (`std::fs::File::lock` if MSRV allows, else crate; else `create_dir` mutex + heartbeat); loser waits, logs every 10s, re-checks. 8. `cache_layout.rs` atomic rename (Windows rename-over-existing = someone else won). 9. `gc.rs` + fill `run_cache` in `cli_cmds.rs` (02's stub; list/gc/clear via gc.rs; no main.rs edit). 10. Real-Hub test with the pinned small model from `testdata/models.json`.

## Todo List
- [ ] ADR 0012 + CD1/2/4 answers  - [ ] resolver  - [ ] key + golden  - [ ] api parse  - [ ] download+resume  - [ ] disk  - [ ] lock (2-process test)  - [ ] gc  - [ ] real-Hub test

## Success Criteria
- Resolver classification table test passes for all 4 inputs + error. `[RUN]`
- Resume: kill mid-download (local server drops), rerun completes with correct sha256. `[RUN on local HTTP server]`
- Two processes race on one key: exactly one builds/downloads, other waits then hits. `[RUN]`
- Offline with populated HF cache: zero sockets opened (test binds nothing, asserts no connect). `[RUN]`
- Gated 403 -> exact message. Disk deficit message numbers correct. `[RUN on local HTTP server]`
- Real Hub download of pinned tiny model matches `testdata/models.json` sha256. `[NEEDS-NETWORK]`

## Test Strategy & Quality Gate
- G/W/T: G = deterministic resolve/download/cache; W = classification table, resume, lock race, offline zero-socket, gated/disk messages, key golden; T = `model_resolve`, `hf_download_local_server`, `cache_key`, `cache_lock`, e2e `cache_e2e`.
- Spec: Input model_ref+env+flags. Output Bundle|BuildJob|typed error. Edge: dotted names, rev as branch vs sha, sharded index, repo with only `.bin`. Error: 401/403/404, truncated body, disk full, lock holder died.
- Pyramid: 60% unit (key, parse, resolve), 35% integration (local server, locks across processes), e2e `cache_e2e` (in `model_resolve.rs`): resolves pinned model -> BuildJob; second call -> hit.
- Coverage: diff >=90% line / >=75% branch (`CMD-RUST-COV -p helios-cli`, `CMD-DIFF`, plan.md). No mocks returning canned results for logic under test (the local server is a real protocol peer).
- Evidence: `cargo test -p helios-cli --test model_resolve --test hf_download_local_server --test cache_key --test cache_lock`; coverage commands as above.

## ADR Rationale
ADR 0012. Alternatives per CD1 weighed in ADR; HF-layout write chosen for cross-tool reuse at moderate cost; key excludes runtime-only params to avoid rebuilding 15 GB per flag change.

## Risk Assessment
HF xet/CDN behaviour changes (phase 01 verifies; Fetcher isolates). NFS lock unreliability: mark UNVERIFIED in ledger. Hash trust (CD4). Windows symlink privilege: fallback chain, size check follows links (test).

## Security Considerations
Token never logged, never sent cross-host. Path traversal in repo file names (`rfilename` with `..`) rejected. Only `.safetensors`/json/tokenizer files fetched (no pickle). Downloaded tree treated as untrusted data; sizes capped by API-declared sizes.

## Next Steps
Feeds 07. `helios-pkg` (old 06) later replaces the sidecar placeholder; key includes `contract_revision` so migration invalidates cleanly.
