# 0008: KV hashing and events a cluster router can use

Status: **proposed**, needs owner sign-off (not accepted, nothing implemented). Plan: phase 13 D5, phase 07 (prefix cache), phase 11 (tiers); owner decisions 10 (hash block size) and 19 (Dynamo worker versus own router). Related: ADR 0002, 0007.

## Context
A prefix cache lets a request reuse the KV pages of an earlier request with the same beginning. A router in front of several engines (phase 13 M3) can send a request to the engine that already holds its prefix, if it knows which prefixes each engine holds. NVIDIA Dynamo's router expects engines to publish block events with a stored/removed/cleared vocabulary and monotonically increasing ids.

## Decision (proposed)
- **Chained prefix hashes with token verification** (phase 07): a block's hash covers its tokens and its parent's hash, so equal hashes mean equal prefixes, and a stored block keeps its tokens so a hash collision is detected, never trusted. Block size for hashing is its own parameter, independent of the page size (decision 10: 64 tokens is the candidate; to be measured with the chunk study).
- **Events.** The engine emits `BlockStored { id, hash, parent_hash, block_size, tier }`, `BlockRemoved { id, hash }` and `Cleared { id }` with a per-engine monotonically increasing `id` and the hash block size in the stream header, so a consumer can detect a lost event and ask for a resync (a snapshot of the current index).
- **Under tensor parallelism the index is per engine, not per rank**: it lives with the scheduler on rank 0 and refers to page indices valid on every rank (ADR 0002).
- **Transport-agnostic.** Events are plain data with a stable encoding; whether they go to an in-process subscriber, a socket or Dynamo's event plane is a deployment choice (decision 19).
- **No router in this change.** A router is a separate decision; this only keeps the engine's contract compatible with one.

## Why over the alternatives
- Hashes without parent links: equal block hashes at different depths would alias different prefixes.
- Per-rank indexes: ranks hold identical pages; N indexes add cost and can disagree.
- Events as an afterthought: a router written later forces a change of the identifiers the cache already uses.

## Consequences
- The prefix cache of phase 07 is built with these identifiers from the start; the radix cache in `helios-core/cache.rs` (256-token chunks) is replaced or adapted by the chunk study.
- Tier moves (phase 11) emit stored/removed with the tier, so a router can value a prefix held on device above one held on host.

## Open questions for the owner
- Decision 10 (hash block 64 tokens versus the page size) and decision 19.
- Whether tier information is part of the event or a separate stream.

## Verification plan
Simulator tests (phase 07): event ids are gap-free and monotonic, a resync snapshot equals the replayed events, parent links form a forest, token verification rejects a forged collision; a consumer test that rebuilds the index from events alone.
