# 0008: KV hashing and events a cluster router can use

Status: **accepted** (owner review, 2026-10-30). Design only: nothing is implemented. Plan: phase 13 D5, phase 07 (prefix cache), phase 11 (tiers); owner decisions 10 (hash block size) and 19 (Dynamo worker versus own router). Related: ADR 0002, 0007, 0010.

## Context
A prefix cache lets a request reuse the KV pages of an earlier request with the same beginning. A router in front of several engines (phase 13 M3) can send a request to the engine that already holds its prefix, if it knows which prefixes each engine holds. NVIDIA Dynamo's router expects engines to publish block events with a stored/removed/cleared vocabulary and monotonically increasing ids.

## Decision
- **Chained prefix hashes, seeded by the model's identity, with token verification.** A block's hash covers its tokens and its parent's hash. **The chain starts from `cache_identity()`** (ADR 0010: payload hashes, layout, target arch, tensor format, model configuration): the parent of the first block is that 32-byte digest, with a domain tag. Equal hashes therefore mean equal tokens *and* the same weights run the same way on the same hardware, and prefixes of two models, two quantisations or two architectures can never alias, in one engine or in a router that sees several. A stored block keeps its tokens so a hash collision is detected, never trusted. The hash block size is its own parameter, **64 tokens by default**, independent of the page size (ADR 0002 also defaults to 64).
- **Events carry token ids.** `BlockStored { id, hash, parent_hash, block_size, tier, tokens }` (the block's token ids), `BlockRemoved { id, hash }` and `Cleared { id }`, with a per-engine monotonically increasing `id` and the hash block size and `cache_identity` in the stream header, so a consumer can detect a lost event, ask for a resync (a snapshot of the current index) and verify the model it is talking to. A consumer that needs another hash (Dynamo computes its own over token ids) recomputes it from the token ids, so no engine-specific hash is binding on a router. **A Dynamo adapter** (a separate module behind a feature, written when decision 19 is acted on) translates this vocabulary into Dynamo's event schema and hash; the engine's own event format is not bent to a vendor's.
- **Under tensor parallelism the index is per engine, not per rank**: it lives with the scheduler on rank 0 and refers to page indices valid on every rank (ADR 0002).
- **Transport-agnostic.** Events are plain data with a stable binary encoding (same rules as ADR 0007); whether they go to an in-process subscriber, a socket or Dynamo's event plane is a deployment choice (decision 19).
- **No router in this change.** A router is a separate decision; this only keeps the engine's contract compatible with one.

## Why over the alternatives
- Hashes without parent links: equal block hashes at different depths would alias different prefixes.
- A chain that starts from zero: the same tokens under two models would collide in a shared router or tier store.
- Events with hashes only: a consumer cannot verify them or recompute them in another scheme.
- Per-rank indexes: ranks hold identical pages; N indexes add cost and can disagree.
- Events as an afterthought: a router written later forces a change of the identifiers the cache already uses.

## Consequences
- The prefix cache of phase 07 is built with these identifiers from the start; the radix cache in `helios-core/cache.rs` (256-token chunks) is replaced or adapted by the chunk study.
- Tier moves (phase 11) emit stored/removed with the tier, so a router can value a prefix held on device above one held on host.
- Event size grows by the token ids (64 tokens, 256 bytes per block): negligible next to the page.

## Open question left to the router decision
Whether tier information is part of the event (as above) or a separate stream, decided with decision 19.

## Verification plan
Simulator tests (phase 07): event ids are gap-free and monotonic, a resync snapshot equals the replayed events, parent links form a forest rooted at the identity digest, token verification rejects a forged collision, two manifests with different `cache_identity()` never share a hash for equal tokens; a consumer test that rebuilds the index from events alone.
