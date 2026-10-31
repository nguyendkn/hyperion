//! The KV bookkeeping of the scheduler: chained prefix hashes, the page allocator and prefix index, and the events a router can follow
//! (ADR 0002, 0008). Indices and hashes only: no address and no device type.

pub mod events;
pub mod hash;
pub mod manager;

#[cfg(test)]
mod tests;

pub use events::{EventLog, KvEvent, Mirror, MirrorError, Snapshot, StoredBlock, StreamHeader, Tier};
pub use hash::{block_hash, chain, BlockHash};
pub use manager::{Admission, BlockState, KvConfig, KvError, KvManager, KvStats, PageIndex, SeqId};
