//! KV events a router or a tier store can follow (ADR 0008): `BlockStored` (with the block's token ids, so a consumer can verify the hash or
//! compute its own), `BlockRemoved` and `Cleared`, with a per-engine id that rises by exactly one per event. The stream header names the hash
//! block size and the cache identity. A consumer that sees a gap asks for a [`Snapshot`] and continues from its `last_id`. In-process
//! subscribers only: how the events travel is a deployment choice (decision 19).

use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};

use super::hash::BlockHash;

/// Where a block's KV lives. Only the device for now; a tier move (phase 11) re-announces a block with another tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Gpu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvEvent {
    BlockStored { id: u64, hash: BlockHash, parent_hash: BlockHash, block_size: usize, tier: Tier, tokens: Vec<u32> },
    BlockRemoved { id: u64, hash: BlockHash },
    Cleared { id: u64 },
}

impl KvEvent {
    pub fn id(&self) -> u64 {
        match self {
            Self::BlockStored { id, .. } | Self::BlockRemoved { id, .. } | Self::Cleared { id } => *id,
        }
    }
}

/// What a consumer must know before it reads events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamHeader {
    pub block_size: usize,
    pub identity: BlockHash,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredBlock {
    pub hash: BlockHash,
    pub parent_hash: BlockHash,
    pub tier: Tier,
    pub tokens: Vec<u32>,
}

/// The index at one moment, in the order the blocks were stored (parents first).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub header: StreamHeader,
    pub last_id: u64,
    pub blocks: Vec<StoredBlock>,
}

/// The producer side: numbers the events and hands them to the subscribers.
pub struct EventLog {
    header: StreamHeader,
    last_id: u64,
    subscribers: Vec<Sender<KvEvent>>,
}

impl EventLog {
    pub fn new(header: StreamHeader) -> Self {
        Self { header, last_id: 0, subscribers: Vec::new() }
    }

    pub fn header(&self) -> &StreamHeader {
        &self.header
    }

    pub fn last_id(&self) -> u64 {
        self.last_id
    }

    /// A receiver of every event from now on.
    pub fn subscribe(&mut self) -> Receiver<KvEvent> {
        let (tx, rx) = channel();
        self.subscribers.push(tx);
        rx
    }

    /// Number the event `make` builds and deliver it; a subscriber that went away is dropped.
    pub fn emit(&mut self, make: impl FnOnce(u64) -> KvEvent) -> KvEvent {
        self.last_id += 1;
        let event = make(self.last_id);
        self.subscribers.retain(|s| s.send(event.clone()).is_ok());
        event
    }
}

/// A consumer's copy of the index, built from events alone (or from a snapshot and the events after it).
pub struct Mirror {
    pub header: StreamHeader,
    pub last_id: u64,
    pub blocks: HashMap<BlockHash, StoredBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MirrorError {
    /// An event arrived whose id is not `last_id + 1`: events were lost, ask for a snapshot.
    Gap { expected: u64, got: u64 },
    /// A `BlockStored` whose hash does not match its tokens and parent (the consumer verifies, it never trusts).
    BadHash(BlockHash),
}

impl Mirror {
    pub fn new(header: StreamHeader) -> Self {
        Self { header, last_id: 0, blocks: HashMap::new() }
    }

    pub fn from_snapshot(snapshot: Snapshot) -> Self {
        let blocks = snapshot.blocks.into_iter().map(|b| (b.hash, b)).collect();
        Self { header: snapshot.header, last_id: snapshot.last_id, blocks }
    }

    pub fn apply(&mut self, event: &KvEvent) -> Result<(), MirrorError> {
        if event.id() != self.last_id + 1 {
            return Err(MirrorError::Gap { expected: self.last_id + 1, got: event.id() });
        }
        match event {
            KvEvent::BlockStored { hash, parent_hash, tier, tokens, .. } => {
                if super::hash::block_hash(parent_hash, self.header.block_size, tokens) != *hash {
                    return Err(MirrorError::BadHash(*hash));
                }
                self.blocks.insert(*hash, StoredBlock { hash: *hash, parent_hash: *parent_hash, tier: *tier, tokens: tokens.clone() });
            }
            KvEvent::BlockRemoved { hash, .. } => {
                self.blocks.remove(hash);
            }
            KvEvent::Cleared { .. } => self.blocks.clear(),
        }
        self.last_id = event.id();
        Ok(())
    }
}
