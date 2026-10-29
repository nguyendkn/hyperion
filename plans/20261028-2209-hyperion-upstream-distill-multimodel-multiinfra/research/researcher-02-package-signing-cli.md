# Researcher-02: packages, signing, storage, CLI/API (NVIDIA local-inference "Flare/Ignition/Helios")
Ideas only, own words. Source root: nv/repo/docs. [V]=read in source, [I]=inferred. Not read in full: cli-design.md (sec 3.x, 5), api.md middle (tools/reasoning/Responses/metrics detail), runtime-package-design.md sec 4-8, package-format-design.md derivations/TP/download sections.

## 1) Package model
- Two canonical forms, same schema: single-file "packed" and "directory" (primary + readable companion files). Cache always uses directory form. Pack/unpack = placement change only, needs no key or GPU. [V] package-format-design.md
- Three-way separation: immutable payload bytes / signed execution metadata / UNSIGNED physical locations. Payload identity = hash of payload bytes only (excludes header, tensor name, filename). Moving bytes between files is keyless; changing payload partitioning or exec bindings needs re-sign. [V]
- Container = safetensors (framing/offsets) + Protobuf (typed metadata). Header metadata is an untrusted hint, checked against the authenticated version. Parser rejects duplicates, overlaps, overflow, oversize metadata BEFORE allocating. 128 MiB cap on metadata envelope. [V]
- Per-file schema: primary holds manifest (envelope + shared body + deployment bodies + rank bodies), locations, optional "derivations" (bounded copy/fill recipes to rebuild payloads), tokenizer/chat-template as embedded payloads. Per-rank files: kernels, graphs, initial-state. Weights under weights/, one file per layer. [V]
- Granularity: one deployment per (GPU arch, TP size); one package describes all exported TP deployments; loader selects one deployment/rank. [V]
- Versioning domains (distinct, each explicit): payload format version (v2 current), legacy envelope 6 (still readable/servable; convert/sign refuse it), execution ABI number, prepared-model-contract number, protobuf defaults for absent fields. Legacy reader does not pretend old signatures cover new metadata. [V]
- Stable weight extraction: weight identity depends only on checkpoint bytes + declared packing + TP partition; excludes pointers, warmup residue, capacity padding, tuning-dependent layouts. Never requantizes to force cross-TP sharing. Deterministic owner ordering per layer; a change to one layer never shifts another layer's boundaries; no size-based merging. Precise byte intervals kept (strided holes not owned); every retained live byte must be classified (stable / tuned / initial-state / zero), unclassified goes to initial-state, never dropped. Zero-block elision on fixed 64 KiB blocks relative to owner. [V]
- Cache: one Rust-owned content-addressed store: blobs/<payload-sha256>, models/<key>/<entry>/ with hardlinks to blobs, mutable refs select default entry, plus staging/, locks/, origins/. Model keys never trust user path components. Ref switches only after the full deployment closure verifies; failure leaves default untouched. [V]
- Executables: one zstd level-3 frame per cubin/graph resource, 4 MiB chunks with per-chunk SHA-256 signed; decoder caps window at 8 MiB, rejects truncation/extra frames/length mismatch. Chunk verified BEFORE feeding the decoder (no whole-payload hash first). Weights stay uncompressed (direct IO + dedup). [V]
- Load path: pinned files, contiguous slices through bounded pinned staging + async upload; staging bound is process-wide (more ranks do not multiply it); preflight open-file limit; open descriptors kept alive so a path swap cannot redirect reads. [V]
- Compressed-scale checkpoint (tp-checkpoint-layout.md): NVFP4/MXFP4 block scales stored in a TP-size-independent compressed encoding; GPU regroups per rank at load; no TP padding/directories in checkpoint; no prepared-scale disk cache. [V] Blackwell FP4 specific.

## 2) Signing / trust model
- Ed25519; key = encrypted PKCS#8 PEM (passphrase >=16 bytes, terminal-only prompt, refuses non-hidden input); keygen never overwrites; perms 0700/0600. Separate public "publisher.json" profile (name, namespace, 32-byte pubkey, optional development flag). Passphrase change = atomic temp-file replace, same fingerprint. [V] signing.md
- Signature groups: G1 = index (sans rank bundles), namespace, official-source URL, dev marker, license name+link, rank count, commitments to G2/G3; G2 = frontend assets (tokenizer etc.); G3 = shared embedding-table hashes; plus one signed block per rank (rank bundle, executable spans/chunk hashes, allocation hashes). Package stores key FINGERPRINT, not the key. Multi-host: a host verifies shared groups + only its rank. [V] runtime-package-design.md
- Verification is a type-state pipeline: PackageClaim::read (unverified claims, syntax checked) -> VerifiedPackage::open (against profiles) -> authorize (acceptance string == signed namespace) -> AuthorizedPackage, the ONLY type the loader accepts. Full weight-bytes scan is optional (--verify-weights); executable authentication mandatory. Doc states plainly: a signed digest without a byte scan is not evidence bytes were checked. [V]
- Identity != signature: identity comes from publisher.json at root of the official source repo (HF), fetched every online load and must list the key (removal = revocation). Absent file = unverified; first-use prompt shows publisher, source, license link, full fingerprint; typing the namespace = license agreement; trust stored locally; `trust revoke FINGERPRINT`. Unattended needs two flags (fingerprint AND namespace). Trust never transfers to a new key with the same namespace. Source builds embed no profiles; a release tool stamps profiles into an ELF section of a separately named binary. [V]
- Dev vs prod: development key; `sign` promotes a dev package to prod identity with fresh package ID, no GPU/recompile; re-verifies sources, re-encodes executables (decoded bytes exact, compressed bytes may change). Weight identity independent of executables. Existing destinations never overwritten. [V]
- Honest limits: signatures authenticate bytes, do not sandbox executable code; compression is not secrecy. [V]
- model-storage.md: private S3 bucket, S3-managed AES-256, threaded multipart uploader (16 threads, 64 MiB parts, 6 retries, no cross-run resume, incomplete uploads expire in 7 days). Encryption at rest explicitly separate from package protection. [V]

## 3) CLI and API surface
- CLI principles [V cli-design.md]: flags state intent only (GPUs, concurrency, listen); per-model knowledge lives in recipes; derive geometry (context, pages, graph capacities); exactly two precedence levels (flags > config file > recipe); NO env vars (sole exception CUDA_VISIBLE_DEVICES; the tool sets library env itself). Commands: ignition export/sign/convert/keygen/key-password/release; flare run/inspect/trust; snapshot/download of published packages.
- Capacity errors: fail before any load/socket with needed-vs-available arithmetic plus the exact settings that would fix it; same style for GPU selection and HTTP admission rejects. [V sec 6]
- Model resolution order: existing dir -> "org/repo[@rev]" -> alias; recipes pin revisions; unsupported architecture error lists supported ones; reads config.json + safetensors index without downloading tensors to size the estimate. [V sec 8]
- API routes [V api.md]: /v1/chat/completions, /v1/completions, /v1/messages (+count_tokens), /v1/responses, /v1/models(/{id}), /health, /metrics (Prometheus, flare_ prefix). One model per server; request "model" ignored for routing, any alias/checkpoint id resolves on GET. Bearer auth (Anthropic also x-api-key); /health and /metrics unauthenticated. Unknown model GET -> 404 model_not_found.
- Model card = vLLM-compatible fields plus a vendor object advertising prepared limits: max_top_logprobs, max_concurrency, prefill chunk, default_max_tokens, sampling defaults AND ranges, image limits, modalities. Capability discovery comes from the package. [V]
- Transport: bodies accepted as identity/gzip/deflate/br/zstd, both encoded and decoded size capped (32 MiB default); streaming returns 200 after validate+enqueue, later admission failures are terminal SSE errors; keepalive comments after 15 s waits; first chat chunk only after first token; failure before first token = error, no role chunk and no [DONE]; temporary admission exhaustion 429; disconnect cancels all child choices. [V]
- Logprobs [V api.md, logprobs-design.md]: raw full-vocab log-softmax in FP32, BEFORE temperature/penalties/grammar/top-k/p; spec-decoding uses target logits only; chosen token always present; ties by token id; K=20 fixed at export, request 0..K; `bytes` exact incl. partial UTF-8; one entry per generated token incl. reasoning/tool; EOS stop has no entry; return_tokens_as_token_ids; logprob_token_ids (<=128 explicit ids); prompt_logprobs needs export flag (rank = 1 + count strictly greater, non-finite -> -9999, streaming rejected); older packages advertise allow_logprobs:false and 400 on request. Implementation = persistent buffers inside the captured graph with a per-lane request flag; does not perturb RNG state. /v1/messages rejects logprobs; temperature 0 rejected.
- Ollama discovery route and local model management exist (not read). [I]

## 4) Windows / container / packaging
- Windows: runtime uses the NVIDIA driver (nvcuda.dll) only, no Python/CUDA toolkit/WSL; needs MSVC redist; build natively (MSVC, CMake, NASM, protoc) or cross from Linux via cargo-xwin. Package is copied across, arch must match. Ignition on Windows only with --no-compile. [V windows.md]
- Container: Debian-slim image with Rust binary only; loads host libcuda.so.1; no Ignition/toolkit/weights inside; package + public profile + pre-approved trust store mounted; amd64 and arm64 share a Dockerfile. [V container.md]
- Relevance: matches our model (Windows build host, no CUDA). Runtime dlopen of the driver library instead of link-time CUDA toolkit could ease our sm120 link problem [I].

## 5) Ideas for Hyperion
1. [adopt now] Separate payload identity (hash of bytes) from placement (unsigned locations). Touch: specification manifest + helios-pkg. Reason: v1 not frozen yet; cheap now, impossible after freeze. Fits "signature covers stored bytes".
2. [adopt now] Type-state verify chain Claim -> Verified -> Authorized; loader accepts only the last. Touch: helios-pkg, helios-cli. Reason: "no unverified load" becomes compile-time.
3. [adopt now] Parser hardening: reject duplicates/overlap/overflow/oversize before allocating; header version is hint checked against signed version. Touch: helios-pkg, spec tests (hostile fixtures).
4. [adopt now] Capacity-error format: needed vs available + fix options, before opening sockets. Touch: helios-cli, helios-hal triage; pairs with target_arch refusal (rule 6).
5. [adopt now] Model card advertises capabilities taken from manifest (max logprobs, limits, sampling defaults/ranges, modalities). Touch: helios-core api.rs, spec ModelConfig.
6. [adopt now] HTTP semantics: streaming errors as terminal SSE after 200, no role chunk before first token, 429 on admission exhaustion, bounded bodies + content-encoding. Touch: helios-core api.rs. Our 501/503 decisions stay.
7. [adopt later] Signature groups (index / frontend / per-rank) so a host verifies only its rank. Touch: spec, helios-pkg. Reason: only multi-GPU/host.
8. [adopt later] Content-addressed cache with hardlinks + refs + staging + locks. Touch: helios-pkg (cache module). Reason: pays off once deployments share weights.
9. [adopt later] 4 MiB chunk hashes verified before decompress. Touch: helios-pkg. Reason: only if cubin/graph blobs are compressed.
10. [adopt later] Local trust store, first-use prompt, revoke command. Touch: helios-cli. Their identity-via-HF-repo is vendor-specific; we need our own identity source. Keep fingerprint-in-package, key-outside-package.
11. [adopt later] Logprobs as in-graph reduction with K fixed at build and per-request K<=Kmax; raw-distribution semantics. Touch: ignition recipes, ComputeBackend sampling output, helios-core api.rs.
12. [adopt later] Encrypted key, keygen/key-password, dev-vs-prod identity with `sign` promotion without recompile. Touch: ignition cli.py.
13. [adopt later] Both directory and single-file forms. Touch: helios-pkg. Start with directory (matches our sibling .manifest.json).
14. [skip] Protobuf container: our manifest is JSON from Rust schema, already decided. Safetensors for weight payload files still worth evaluating (Q1).
15. [skip] Anthropic /v1/messages, Responses, Ollama routes: scope. Chat + Completions + models + health + metrics first.
16. [skip] Absolute no-env-vars rule: adopt the spirit (two precedence levels), not the ban.
17. [skip] Compressed FP4 scale layout: Blackwell NVFP4-only; revisit with FP4 work (Hopper never runs FP4).
18. [skip] S3 uploader/bucket doc: ops detail.

## 6) Where Hyperion can be better
- Identity there depends on control of one vendor's repo namespace, fetched online each load. Offline-first pinned-key identity (optionally transparency log) works air-gapped. [I]
- They target SM120/121 only. We add SM90, so a tested compat matrix (arch x tensor_format x bundle) with exact target_arch refusal is a first-class contract they do not document. [I]
- Their doc admits signed digest != verified bytes. We can hash weights by default on first load and cache a verified-marker keyed by file identity. [I]
- Committed JSON Schema + drift CI + Rust/Python round-trip, human-inspectable manifest, deny_unknown_fields: stricter and more auditable than envelope Protobuf.
- Lock-free state machine gives a clean 501/503 split; their /health is binary.
- Errors: add machine-readable codes plus a "fixing setting" hint field, one shape across CLI and HTTP.
- Windows: we already build on Windows; an explicit Windows CI type-check target is a cheap differentiator.

## 7) Unresolved questions
1. Weights as safetensors payload files (interop, hardlink dedup) vs custom framing in .hyperion? Decide before helios-pkg layout.
2. Source of publisher identity for Hyperion (key pinning, HF-like repo, CA)? Needs user decision.
3. Multiple deployments (arch x TP) per bundle in v1, or one per bundle first?
4. Are derivation recipes (rebuild payloads from copy/fill) worth it? Unclear value.
5. Unread sections (see header) may hold more error-code and TP materialization ideas.
6. Does fixed K=20 logprob capacity at export fit our Hopper graph-capture plan?
