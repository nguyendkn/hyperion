# Phase 06: Package + signing (helios-pkg)

## Context Links
- research-02 sections 1,2,5,6; CLAUDE.md schema/signature decisions (".hyperion codec will be crate helios-pkg")
- Files: new `helios/crates/helios-pkg/`, `helios-cli/src/main.rs`, new `ignition/src/hyperion_ignition/{signing,signing_cli}.py`, `specification/` (read-only here)

## Overview
Priority P2. Pending. Depends: 03. Wave: W3. Effort: 6d. Real `.hyperion` codec replacing the `.manifest.json` placeholder, with trust that cannot be skipped at compile time.

## Key Insights
- Separate payload identity (hash of bytes) from placement (unsigned locations): moving bytes needs no key.
- Signed digest without a byte scan is not evidence; we hash weights on first load and cache a verified-marker keyed by file identity.
- Signatures authenticate bytes, they do not sandbox code (state in docs).
- Shard at load (phase 13 D2): the loader reads rank-local slices of canonical tensors. Row splits are contiguous reads; column splits need strided reads or a device-side gather. Verification hashes cover whole payload files, so slicing never weakens trust.
- Load path idea (from ModelExpress, ideas only): register one large VMM arena per rank instead of per tensor, and pick the fastest available path (host-staged POSIX I/O now; GPUDirect Storage later, if available). Peer-to-peer weight transfer between replicas stays in phase 09.

## Requirements
1. Container: directory form first (matches sibling manifest); weights as safetensors payload files if decision 6 says yes, else custom framing; same schema either way.
2. Type-state loader: `PackageClaim::read` -> `VerifiedPackage::open(trust_store)` -> `AuthorizedPackage`; only the last is accepted by the load path (compile-time enforced; trybuild-style compile-fail test).
3. Parser hardening: reject duplicates, overlaps, offset overflow, oversize metadata BEFORE allocating; header version is hint, checked against signed version; metadata cap.
4. Ed25519 signing in Ignition (encrypted key, keygen never overwrites), signature over manifest stored bytes; fingerprint in package, key outside it.
5. Offline-first trust: local trust store of pinned publisher keys, first-use prompt (non-interactive needs explicit fingerprint flag), revoke command. Identity source = decision 5 (answered in phase 03 ADR). Promotion note: research-02 ideas 10 (trust store) and 12 (encrypted keygen) were tagged adopt-later; they are promoted into this phase because the type-state chain (req 2) cannot exist without a trust store and signed packages need keygen.
6. Contract-version mismatch -> typed "re-export" error before load (research-04 idea 5): requirement owned by phase 03 item 7; implemented here in helios-pkg `verify` by reference, with a helios-pkg test reusing 03's unsupported-version fixture asserting the error text contains "re-export".

## Architecture
helios-pkg: `claim`, `verify`, `authorize`, `payload` (hash, bounded reader), `trust` modules; no CUDA dep; Windows+Linux path-safe (no user path components trusted). Ignition `signing.py` produces; Rust verifies; cross fixtures both ways. Ignition CLI: sign/keygen live in new `signing_cli.py` (own console script `hyperion-sign`), NOT in cli.py (owned by 04); 04 adds the `[project.scripts]` line AND the Python `cryptography` dependency (Ed25519) to pyproject.toml as documented hand-offs (inert until this phase lands); `signing.py` imports `cryptography`.

## Related Code Files
- Create: 06 ledger section appended to docs/unverified-ledger.md (after 08), helios-pkg/{Cargo.toml (owns `trybuild` dev-dep for the compile-fail test; deps ed25519-dalek, sha2),src/*.rs,tests/*.rs}, specification/fixtures/hostile/*, ignition/.../{signing,signing_cli}.py, ignition/tests/test_signing_roundtrip.py, tests/compile_fail/*.rs, docs/trust-model.md (new file)
- Modify: helios-cli (load via AuthorizedPackage; this phase owns the load path), root Cargo.toml `[workspace.dependencies]` append-only (members already a glob from 01; no Taskfile edit)

## Implementation Steps
1. Read the phase 03 ADR for decisions 5 and 6 (do not re-ask); if the ADR is missing, stop: layout is blocked.
2. Tests first: hostile fixtures (dup, overlap, overflow, oversize, truncated, tampered byte, wrong key, version lie).
3. Implement claim/verify/authorize; compile-fail test that loader rejects `VerifiedPackage`.
4. Ignition sign + Rust verify round-trip; tampered-after-sign negative test.
5. Trust store + CLI commands; document honest limits.
6. New deps (ed25519-dalek, sha2, safetensors?, trybuild dev-dep, Python `cryptography` via 04 hand-off) each justified in report.

## Todo
- [ ] read ADR 5,6  [ ] hostile fixtures  [ ] type-state  [ ] signing both sides  [ ] trust store  [ ] docs

## Success Criteria
- [RUN] all hostile fixtures rejected with typed errors, no panic, no large allocation (test with allocation cap); round-trip sign/verify; compile-fail test.
- [RUN] unsupported contract version -> typed error containing "re-export" before any load.
- [RUN] on Windows host. [NEEDS-GPU] none for this phase except load of real weights to device.

## Test Strategy & Quality Gate
- Spec: Goal = no unverified bytes reach the loader. AC: Given tampered weight byte Then hash scan fails; Given unknown key Then `Untrusted`; Given oversize metadata Then rejected pre-alloc; Given moved payload file Then still verifies. Out of scope: compression, signature groups per rank, CA, key transparency.
- Pyramid 70/20/10. E2E: `hyperion-compile` then `hyperion-sign` (04 owns cli.py, 06 owns signing_cli.py) -> helios-cli load succeeds; flip one byte -> refuses.
- Coverage diff >=90/75 incl. error branches (parser). Evidence: `cargo test -p helios-pkg`; phase-01 templates `CMD-RUST-COV` (`-p helios-pkg` replaces `--workspace --features stub`; `target/coverage/06/lcov.info`), `CMD-PY-COV`, `CMD-DIFF` (both forms).

## ADR Rationale
Context: placeholder manifest, no trust. Decision: dedicated crate, type-state chain, offline pinned keys, identity != placement. Why over alternatives: runtime check flags (skippable), vendor-repo identity (online-only, vendor-specific), Protobuf container (rejected in CLAUDE.md).

## Risk Assessment
Layout locked at first release: gate on spec freeze checklist. Key handling bugs are high impact: external review before release. Passphrase prompt on Windows terminals needs care.

## Security Considerations
Constant-time compare where relevant; never log keys/passphrases; no overwrite of existing keys/destinations; file descriptors kept open after verify to block path swap (TOCTOU) in loader.

## Next Steps
Payload id used by 07's cache identity is a manifest field from 03; 07 needs no 06 code. Feeds 09 (signature groups, CAS cache, chunk hashes). Out of scope: publishing infra, S3.
