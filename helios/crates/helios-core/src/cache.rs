//! Paged KV cache + Radix tree. The structure mirrors LMCache's out-of-process
//! posture only at the algorithm level (here, in-process for now): chunks of
//! 256 tokens form the cache unit, so prefix reuse is non-linear and keyed by
//! hash rather than by sequence position.
//!
//! The hierarchy in the architecture doc (GPU HBM ↔ CPU pin ↔ NVMe via
//! O_DIRECT) is a tier-residency problem; this module owns the *index* and
//! the block IDs. Actual device memory for blocks is allocated through the
//! HAL `allocate_kv_cache`. The out-of-process sidecar is a later crate; this
//! is the in-engine index.

use std::collections::HashMap;

/// Fixed chunk size used both for hashing (CacheBlend) and as the KV cache
/// block token size. 256 matches the architecture doc.
pub const CHUNK_TOKENS: usize = 256;

/// Content hash of one chunk. Cheap 64-bit hash is enough for a cache key —
/// collisions across real prefixes are vanishingly rare and a miss just costs
/// a recompute, not a correctness bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkHash(pub u64);

/// Maps a chunk hash to the device block(s) holding its KV. Multiple sequences
/// can share a block (prefix reuse) — the scheduler refcounts before freeing.
#[derive(Default)]
pub struct RadixCache {
    blocks: HashMap<ChunkHash, BlockEntry>,
}

#[derive(Debug, Clone)]
struct BlockEntry {
    /// HAL-allocated device pointer for this block. Index into the
    /// `Vec<DevicePointer>` returned by `allocate_kv_cache`.
    block_id: u32,
    /// How many live sequences reference this chunk. Drops to zero → evictable.
    refcount: u32,
}

impl RadixCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a chunk. Returns the block id if present, None on miss — the
    /// caller then runs a prefill for that chunk via the HAL. TODO: eviction
    /// policy (LRU per tier) lives here.
    pub fn lookup(&self, h: ChunkHash) -> Option<u32> {
        self.blocks.get(&h).map(|e| e.block_id)
    }

    /// Record a freshly computed chunk → block mapping. Increments refcount if
    /// the chunk is already known (shared prefix).
    pub fn insert(&mut self, h: ChunkHash, block_id: u32) {
        self.blocks
            .entry(h)
            .and_modify(|e| e.refcount += 1)
            .or_insert(BlockEntry { block_id, refcount: 1 });
    }
}
