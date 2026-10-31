# Research 03: HF fetch + auto-build + cache for `hyperion serve <hf-id|path>`

Evidence tags: [LOCAL] read in repo today; [MEM] from model knowledge of HF Hub docs, NOT re-fetched this session (web tools not used, 5-call budget); [INFER] my inference.

## 1. HF Hub download from Rust (no Python)

### REST surface [MEM]
- File: `GET {endpoint}/{repo_id}/resolve/{revision}/{path}` (models: no `models/` prefix). `revision` = branch, tag or 40-hex commit; default `main`. 302 to CDN (xet/cas-bridge); response headers `X-Repo-Commit` (resolved sha), `X-Linked-Etag`/`ETag` (sha256 for LFS files, git blob sha1 for small), `X-Linked-Size`, `Content-Length`. HEAD gives these without body.
- Listing: `GET {endpoint}/api/models/{repo_id}/revision/{rev}` -> JSON `{sha, siblings:[{rfilename}]}` (add `?blobs=true` for sizes/lfs sha256 [MEM]). Use this to pin `rev` -> 40-hex sha once, then resolve every file against the sha (consistent snapshot).
- Sharded safetensors: fetch `model.safetensors.index.json`, read `weight_map` values (unique shard names), fetch each. Matches ADR 0003 (Ignition already reads index + shard headers) [LOCAL].
- Needed files for Ignition: `config.json`, `generation_config.json` (optional), `tokenizer.json`, `tokenizer_config.json` (chat_template; ADR 0010), `model.safetensors` or index+shards. Skip `*.bin`, `*.pt`, `*.gguf`, onnx, original/.
- Auth: `Authorization: Bearer $HF_TOKEN` (also `HUGGING_FACE_HUB_TOKEN`, token file `$HF_HOME/token`) [MEM]. Gated repo without access: 401/403 with `X-Error-Code: GatedRepo`; map to a clear "accept licence at huggingface.co/<id> and set HF_TOKEN". Do not forward the Bearer token to the CDN redirect host (cross-origin redirect: reqwest strips Authorization on cross-host redirects by default [MEM]; verify).
- Mirror: `HF_ENDPOINT` (default `https://huggingface.co`) [MEM]. Offline: `HF_HUB_OFFLINE=1` -> never touch network; resolve via `refs/` + snapshots only.
- Resume: `Range: bytes=N-` against a `.incomplete` blob; CDN supports it [MEM]. Verify sha256 (LFS etag) after finish; ETag for non-LFS is a git sha1, don't compare it with sha256.

### Cache layout to reuse an existing vLLM/HF cache [MEM]
```
$HF_HUB_CACHE (default $HF_HOME/hub, HF_HOME default ~/.cache/huggingface; legacy TRANSFORMERS_CACHE)
  models--{org}--{name}/
    blobs/<etag>                    (content-addressed; *.incomplete while downloading)
    refs/<branch-or-tag>            (text file: 40-hex commit)
    snapshots/<commit>/<path>       (symlinks -> ../../blobs/<etag>; real files on Windows without dev mode)
```
- Lookup order: explicit sha -> `snapshots/<sha>`; else `refs/<revision>` -> sha -> snapshot; if present and complete (all index shards exist) use it with zero network. vLLM/SGLang use the same `HF_HOME`, so a pod with a prepared volume works unmodified. Windows: symlink fallback = hardlink/copy; reading follows symlinks fine in Rust (`std::fs`).
- Writing in the same layout (blobs + refs + snapshot symlinks) keeps the cache shared with vLLM/Python tools. [INFER] cost is moderate; a flat own-layout cache saves effort but forfeits reuse in the reverse direction. Recommend: read the HF layout always; write in HF layout too.
- Local path input: `serve ./dir` with config.json + safetensors; no revision, key uses content fingerprint (see section 2).

### Options
| Option | Pros | Cons / cost |
|---|---|---|
| `hf-hub` crate (huggingface/hf-hub, sync+tokio features) [MEM] | Implements the cache layout, token, endpoint, resume, parallel chunked download; maintained by HF | New dep tree (ureq or reqwest+tokio, rustls/native-tls); API churn; check Windows symlink behavior and `HF_ENDPOINT` support; git-sha/refs handling coverage uncertain. Project rule: avoid new deps unless needed. |
| `reqwest` (blocking, rustls) by hand | Full control; ~300 lines (resolve, HEAD, Range, sha256, layout) | We own edge cases (redirect auth strip, xet, retries). helios-core already has tokio + HTTP server (axum/hyper?) [INFER, not checked]; reqwest adds client TLS stack. |
| Shell to `huggingface-cli download` / `hf download` | Zero Rust code; best compat (xet, hf_transfer) | Needs Python + huggingface_hub at serve time, but the build step needs Python anyway [INFER]; a second subprocess; progress parsing is brittle. |
| Shell to Ignition and let Python download (`huggingface_hub.snapshot_download`) | One subprocess does fetch+build; Python lib handles gated/xet/resume | Rust cannot compute the cache key (needs the sha) before spawn unless it asks Python; offline fast-path must still be reimplemented in Rust. |

Recommendation [INFER]: Rust resolves sha only via cheap `api/models/.../revision` (one HTTP call, or `refs/` offline), then either (a) hf-hub/reqwest download, or (b) phase 1: shell to Python `snapshot_download` with `allow_patterns`, and later replace with pure Rust. Whichever, the cache-key computation (needs sha) and the "bundle already built -> serve with no Python" fast path must be in Rust. Decision for owner (new dependency).

## 2. Built-bundle cache key and lifecycle

### Key inputs [INFER, grounded in LOCAL ADRs]
Key = sha256 over canonical string of:
1. Source identity: HF `repo_id` + resolved 40-hex commit (matches `qualified_checkpoint {repo, revision}` ADR 0010 [LOCAL]); for local path: sha256 over (relative path, size, mtime-or-header-hash) of config.json, tokenizer files, safetensors headers (hash of headers, not 15 GB; full hash would be slow).
2. Ignition version (`0.2.0` today [LOCAL]) and manifest `contract_revision` (1 [LOCAL]) and `manifest_version` ("V1"). Bump of either invalidates.
3. `target_arch` (probed device, e.g. `sm_90`; exact match rule 6 [LOCAL CLAUDE.md]).
4. `tensor_format` (bf16 now; fp8 later when a quantizer exists; records what the packager wrote).
5. Quantization/dtype request flags (`--dtype`, `--quantization`) once they exist.
6. Chat template: only if user overrides (`--chat-template`); otherwise it derives from source files already in key (1). The template is in the signed manifest [LOCAL].
Not in key:
- `max-model-len` / `--max-seq-len`: runtime-only (Cli `--max_seq_len` is checked against manifest at load [LOCAL main.rs]); do not rebuild per value.
- TP size / rank layout: ADR 0005 decision 7 and ADR 0010 #7/#20 say one package per target arch, TP chosen at load, sharded at load from canonical weights [LOCAL]. The manifest holds `supported_tp_sizes`, so a TP change must NOT rebuild. (Note vLLM drop-in `--tensor-parallel-size` maps to a load arg.)
- Signature/signing key: affects bundle bytes but not content; if signing key rotates, key it separately or re-sign only the manifest (weights are untouched).
Layout: `$HYPERION_CACHE (default $XDG_CACHE_HOME/hyperion or %LOCALAPPDATA%\hyperion)/bundles/<key16>/` containing `<name>.manifest.json`, tokenizer asset, payload files, `BUILD.json` (inputs echo, ignition version, timestamps, sizes). Include human-readable `<org>--<name>` prefix in dir name for debuggability.

### Weights: avoid duplicating 15 GB [INFER]
Per ADR 0003, payload `file` must be a bare filename resolved beside the manifest, with sha256 + byte_length in the signed table [LOCAL]. Options: symlink/hardlink the HF snapshot shard files (names like `model-00001-of-00004.safetensors` are bare names, fine) into the bundle dir, or copy. Hardlink is not possible across filesystems; symlink on Windows needs privilege. Fallback chain: symlink -> hardlink -> copy. Size check at runtime follows symlinks (uses `metadata`, not `symlink_metadata`: verify). Disk-space check applies only on copy fallback.

### Atomic build, lock, corruption, GC [INFER]
- Build into `bundles/.tmp-<key>-<pid>/`, write `BUILD.json` last, `fsync`, then `rename` to `bundles/<key>/` (atomic on same fs; on Windows rename onto an existing dir fails -> treat as "someone else won", discard tmp).
- Concurrent launches (multi-GPU pods: N ranks/processes, or K8s replicas on a shared PVC): advisory lock file `bundles/<key>.lock` (`fs2`/`fd-lock` crate, or `std::fs::File::lock` stabilized in Rust 1.89 [MEM]; check MSRV). Loser waits with periodic log lines, then re-checks for completed dir. Stale-lock safe because OS locks die with the process. NFS: flock reliability varies; mark unverified. Alternative with no deps: `create_dir` as mutex with pid+heartbeat file.
- Disk check before download and before build: required = sum(shard sizes from the API `siblings` sizes) x (1 if linking, 2 if copying) + margin (e.g. 5%); `fs4`/`sysinfo` or `statvfs`; refuse early with the exact deficit.
- Corruption: at serve, cheap verify = manifest parses, `contract_revision` ok (`Manifest::from_json_bytes` already refuses mismatch with a re-export hint [LOCAL]), each payload exists and size equals the table (existing runtime check [LOCAL]). Full sha256 is phase 06 and not implemented ("cache of already verified" noted in ADR 0003 [LOCAL]); the build step can write `verified.json` (mtime+size of each payload) after hashing once. Any failure -> mark `<key>.bad`, rebuild once, never loop.
- GC: `hyperion cache ls|rm|gc --max-size --older-than`; keep LRU via `BUILD.json` last_used (touch on serve, best effort). Never auto-delete at serve time (surprising on shared volumes); only clean stale `.tmp-*` older than e.g. 24 h with no live lock.

## 3. Ignition CLI today vs. a non-interactive "build to this dir" call

### Exists today [LOCAL cli.py, 54 lines]
`hyperion-compile --model <id: llama3|qwen2|deepseek> --hw <sm_90|sm_120> --source <dir> --out <file> [--hf-repo owner/name --hf-revision <40hex>]`.
- Flow: build_compiler(model_id) -> build_recipe(arch) -> compile(source, recipe, checkpoint) -> `validate_manifest` (jsonschema) -> write `<out>.manifest.json` via `with_suffix(".manifest.json")`, `indent=2`. Prints `compiling ...` and `wrote manifest -> path` to stdout.
- Errors: `click.ClickException` -> exit code 1 + `Error: msg` on stderr, for KeyError/ValueError/OSError/NotImplementedError. Unexpected exceptions -> traceback, exit 1 too. Usage errors exit 2 (click default).
- Runtime reads: `load_manifest(path)` = `path.with_extension("manifest.json")` sidecar [LOCAL main.rs:383]. Note `with_suffix` (Py) vs `with_extension` (Rust): `foo.hyperion` -> `foo.manifest.json` in both. A directory name with dots (e.g. `qwen2.5-7b`) would be mangled: `with_suffix` on `qwen2.5-7b` -> `qwen2.manifest.json`. Real bug risk for naming the out path after the model; use fixed `bundle.hyperion` inside the cache dir.
- Cli (Rust): `--model <PathBuf>` (package path, required), `--port 8080`, `--bind`, `--max-seq-len`. No subcommands, no `serve`, no HF id.
- Tokenizer asset is resolved relative to the package dir via `resolve_in_package` (rejects abs and `..`) [LOCAL]; so tokenizer.json must be copied/linked inside the bundle dir.

### Missing for Rust-driven build [INFER from the above]
1. **`--model` is a registry id** (`llama3|qwen2|deepseek`), not an HF id. Rust needs `config.json` `architectures[0]` -> registry id mapping (`Qwen2ForCausalLM` -> qwen2, `LlamaForCausalLM` -> llama3; Qwen3? not in the id list in help text; check registry). Add `--auto` that reads config.json, or let Rust do the mapping and pass `--model`; unknown arch -> a distinct exit code and message ("not supported; supported: ...").
2. **Payload files are not written by cli.py**: the `TODO` says the weight blob/signature is not written. Per ADR 0003 weights stay as the source safetensors, listed in `payloads` with sha256; so "build" = manifest + (link|copy) of shards + tokenizer/config assets into the dir. Whether `compile()` copies assets or only the CLI must (the CLI doesn't) is not verifiable from the files I read; need `compiler/` inspection. Missing: `--link-mode symlink|hardlink|copy`, copying `tokenizer.json`.
3. **Signing is absent** (TODO; "signature appended" not done; ADR 0010 #5: pinned keys + local trust store, phase 06). For auto-build, key handling options: (a) unsigned until phase 06 and the runtime logs "unsigned"; (b) per-machine local key under cache dir (`keys/ed25519`, 0600) signing local builds, with the same local trust store entry (self-signed = trust-on-first-use; reasonable for drop-in UX, weaker than publisher keys); (c) `HYPERION_SIGNING_KEY` env/file path for CI-built bundles. Never auto-trust downloaded third-party bundles. Decision needed.
4. **Exit codes**: only 0/1/2. Need stable: 0 ok; 10 unsupported architecture; 11 source missing/incomplete (config/tokenizer/shards); 12 unsupported feature (rope_scaling, sliding window refused per CLAUDE.md); 13 disk full; 14 schema validation fail; 1 other. Today all of 10-14 are lumped into exit 1 with a free-text message.
5. **Machine-readable progress**: stdout is human prose only. Add `--progress-json` (JSON lines on stdout: `{"event":"stage","name":"hash_payload","file":..,"done":N,"total":M}` and a final `{"event":"done","manifest":path}` / `{"event":"error","code":..,"message":..}`) so Rust can forward to logs and decide health state. sha256 of 15 GB is the long phase (Ignition computes size+sha256 for every payload [LOCAL ADR 0003]).
6. **`--out` semantic**: is a file path, sidecar derived. Want `--out-dir` writing into the given (temp) dir; Rust renames afterwards. Needs the hf-repo/revision pass-through (already exists) so `qualified_checkpoint` is recorded.
7. **Hash reuse**: the HF blob name is the sha256 for LFS files, so Ignition could skip hashing (take from blob etag) [INFER]; saves minutes. Needs `--known-sha256 file=hex` or a sidecar; trade-off: trusting cache layout vs re-hashing. Owner decision (security vs time).
8. **No `--no-network`/env hygiene**: Ignition itself must not touch the network (the downloader is separate); set `HF_HUB_OFFLINE=1` for the subprocess [INFER].
9. **Python deps**: Ignition imports click, jsonschema; whether `transformers`/`torch` are imported by compilers is unknown from cli.py (it imports none directly; ADR text says "reuse HF/PyTorch ecosystem" but it reads safetensors headers and JSON). If torch is not required for bf16 manifest build, the serve-time env can be minimal (python + click + jsonschema + safetensors?) [UNVERIFIED: check `compiler/` imports and `pyproject.toml` dependencies].

### Producing the cache dir [INFER]
`bundles/.tmp-K/` <- `bundle.manifest.json` (Ignition), `bundle.hyperion` (placeholder / empty until codec crate `helios-pkg` lands, per CLAUDE.md), symlinks `model-0000x-of-0000N.safetensors` -> snapshot blobs, `tokenizer.json` (+ `tokenizer_config.json`), `BUILD.json`. Rust then `helios-cli --model bundles/K/bundle.hyperion` equivalent in-process. Only the manifest sidecar is required today by `load_manifest`.

## 4. Risks of Python subprocess at serve time on a GPU pod (all [INFER], unverified)
- **Discovery**: which python? Order: `$HYPERION_PYTHON`, `./.venv`, `$VIRTUAL_ENV`, `python3`, `python` (Windows: `py -3`). Probe with `python -I -c "import hyperion_ignition; print(version)"`; mismatch of Ignition version vs Rust-embedded expected version -> fail with install hint (`pip install hyperion-ignition==X`). Runtime images like `vllm/vllm-openai` have Python+torch; a minimal Rust image may have none -> then only prebuilt bundles work; say so in error.
- **Pod image**: `nvidia/cuda:12.4.1-devel-ubuntu22.04` [LOCAL env] ships no pip by default; `pip install` at start requires network. Offer `--bundle <dir>` / prebuilt bundle as the production path; auto-build is the dev/convenience path.
- **torch not needed?** Probably not for manifest + safetensors headers + sha256 [UNVERIFIED, see 3.9]; if Ignition imports torch eagerly, import cost 5-15 s and 1+ GB of image; make it lazy.
- **Time budget**: dominated by download (7B bf16 about 15 GB: minutes) and sha256 of payloads (~1-2 GB/s per core: ~10-20 s warm cache, minutes cold from network volume); manifest compile itself is seconds. Second launch: cache hit, no Python.
- **Orchestrator probes**: K8s liveness/readiness may kill the pod during first build. Bind the HTTP port first (project already has `GET /health` 503 outside Ready/Loaded/Serving [LOCAL CLAUDE.md]) and report a `Building`/`Downloading` state with 503 + `Retry-After` and progress body; readiness stays false, liveness must be satisfied (200 on a separate `/live` or tolerate 503, since 503 on liveness would kill). Recommend documenting `startupProbe` with `failureThreshold x period >= 30 min`. This needs a new engine state (the state machine has 6 states, exhaustive table [LOCAL]): or keep the build state outside the engine (CLI-level) to avoid touching `helios-core` (rule 1/2). [INFER]
- **Concurrency**: N replicas on a RWX volume race on the same key -> lock (section 2); on node-local disk each replica builds separately (wasteful: suggest an init container running `hyperion build`).
- **Subprocess hygiene**: pass explicit args (no shell), `-I` isolation per repo rule, set `cwd` to the temp dir not the model dir (planted modules), kill process group on SIGTERM (pod termination during build), capture stderr tail (last 50 lines) into the error, enforce `--build-timeout`.
- **Reproducibility/security**: building from an arbitrary HF repo at serve time reads untrusted `config.json`/tokenizer; the safetensors-only rule (no pickle) and `trust_remote_code` never being honored should be stated. A local-key signature proves nothing about origin: don't present it as provenance.
- **Windows dev host**: no GPU build host [LOCAL]; symlinks and file-lock behaviour differ; build path has `with_suffix` pitfalls (3).

## Unresolved questions
1. Hand-written reqwest vs `hf-hub` crate vs Python `snapshot_download` for phase 1? (new-dependency decision; owner).
2. Write cache in HF layout (shared with vLLM) or own layout with read-only HF reuse?
3. Signing in auto-build: unsigned, per-machine TOFU key, or env-provided key? Interaction with phase 06 trust store.
4. Trust HF LFS sha256 (blob etag) instead of re-hashing 15 GB?
5. Does `compiler/` copy tokenizer/shards into the output dir, or only write the manifest? Which Python packages does Ignition import (torch?)? (Not read this session.)
6. Is Qwen3 / other archs registered under `--model` ids? Need config.json `architectures` -> id map; where does it live (Rust or Python)?
7. Rust MSRV and whether a file-lock crate or `File::lock` is acceptable.
8. Health state during build: new engine state or CLI-level wrapper (rule 1: keep core hardware-agnostic; state machine is exhaustive)?
9. Should key include `max-model-len`? Recommendation no (runtime-only); confirm no manifest field depends on it.
10. Verify [MEM] items: HF `X-Repo-Commit`/`api/models/.../revision` fields, `HF_HUB_CACHE` precedence, xet-based download compatibility with plain `resolve` + Range.
