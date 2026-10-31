//! The KV page allocator and the prefix index (ADR 0002, 0008). It works on page indices only: it never sees an address, and `helios-core` stays
//! hardware-agnostic. A page holds `page_tokens` tokens; a hash block is `hash_block_tokens` tokens, a multiple of the page size (a block is
//! `k` whole pages), and the unit of sharing.
//!
//! A block lives through Reset (no pages) -> Partial (pages allocated, tokens still arriving) -> Complete (all its tokens are known, its KV is
//! not computed yet) -> Registered (its KV is computed and it is in the index: the only state a second request may share). A sequence pins
//! the blocks it uses; only Registered blocks nobody pins, and that no other Registered block builds on (leaves), are evicted, least
//! recently used first. The index is location-agnostic: every entry has a residency (`Tier::Gpu` now).

use std::collections::{HashMap, VecDeque};

use thiserror::Error;

use super::events::{EventLog, KvEvent, Snapshot, StoredBlock, StreamHeader, Tier};
use super::hash::{block_hash, chain, BlockHash};

pub type PageIndex = u32;
pub type SeqId = u64;
type BlockId = usize;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum KvError {
    #[error("the KV pool is full: {needed} pages needed, {free} free after evicting every unpinned cached block")]
    PoolFull { needed: usize, free: usize },
    #[error("sequence {0} is not known to the KV manager")]
    UnknownSequence(SeqId),
    #[error("the hash block size {block} is not a multiple of the page size {page}")]
    BlockNotPageMultiple { block: usize, page: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockState {
    Reset,
    Partial,
    Complete,
    Registered,
}

struct Block {
    state: BlockState,
    hash: Option<BlockHash>,
    pages: Vec<PageIndex>,
    tokens: Vec<u32>,
    pins: u32,
    last_use: u64,
    /// Registered blocks whose parent is this one.
    children: u32,
    parent: Option<BlockId>,
    parent_hash: Option<BlockHash>,
    registered_at: u64,
}

impl Block {
    fn reset() -> Self {
        Self { state: BlockState::Reset, hash: None, pages: Vec::new(), tokens: Vec::new(), pins: 0, last_use: 0, children: 0, parent: None, parent_hash: None, registered_at: 0 }
    }
}

struct Sequence {
    tokens: Vec<u32>,
    blocks: Vec<BlockId>,
    table: Vec<PageIndex>,
    computed: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct KvConfig {
    pub pages: usize,
    pub page_tokens: usize,
    pub hash_block_tokens: usize,
    pub identity: BlockHash,
}

/// What `admit` decided for a prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admission {
    /// Prompt tokens whose KV is already in the pool (a whole number of blocks, never the whole prompt: the last token is always computed).
    pub cached_tokens: usize,
    /// The sequence's page table so far (shared prefix pages first).
    pub table: Vec<PageIndex>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct KvStats {
    pub admissions: u64,
    pub prompt_tokens: u64,
    pub hit_tokens: u64,
    pub evicted_blocks: u64,
    pub registered_blocks: u64,
    pub collisions_rejected: u64,
}

impl KvStats {
    pub fn hit_rate(&self) -> f64 {
        if self.prompt_tokens == 0 { 0.0 } else { self.hit_tokens as f64 / self.prompt_tokens as f64 }
    }
}

pub struct KvManager {
    cfg: KvConfig,
    blocks: Vec<Block>,
    unused_blocks: Vec<BlockId>,
    index: HashMap<BlockHash, BlockId>,
    free_pages: VecDeque<PageIndex>,
    sequences: HashMap<SeqId, Sequence>,
    clock: u64,
    registrations: u64,
    events: EventLog,
    stats: KvStats,
}

impl KvManager {
    pub fn new(cfg: KvConfig) -> Result<Self, KvError> {
        if cfg.page_tokens == 0 || cfg.hash_block_tokens % cfg.page_tokens != 0 || cfg.hash_block_tokens == 0 {
            return Err(KvError::BlockNotPageMultiple { block: cfg.hash_block_tokens, page: cfg.page_tokens });
        }
        let events = EventLog::new(StreamHeader { block_size: cfg.hash_block_tokens, identity: cfg.identity });
        Ok(Self { cfg, blocks: Vec::new(), unused_blocks: Vec::new(), index: HashMap::new(), free_pages: (0..cfg.pages as PageIndex).collect(), sequences: HashMap::new(), clock: 0, registrations: 0, events, stats: KvStats::default() })
    }

    pub fn config(&self) -> &KvConfig {
        &self.cfg
    }
    pub fn pages_free(&self) -> usize {
        self.free_pages.len()
    }
    pub fn pages_used(&self) -> usize {
        self.cfg.pages - self.free_pages.len()
    }
    /// Pages held by a running sequence or shared by one, each counted once however many sequences share it.
    pub fn pages_pinned(&self) -> usize {
        self.blocks.iter().filter(|b| b.pins > 0).map(|b| b.pages.len()).sum()
    }

    /// Of the pages the cache would share with `prompt`, how many a running sequence already holds (no side effects). Same chain and same
    /// last-token rule as [`admit`](Self::admit).
    pub fn pinned_pages_shared_with(&self, prompt: &[u32]) -> usize {
        let b = self.cfg.hash_block_tokens;
        let mut hits: Vec<BlockId> = Vec::new();
        for (i, hash) in chain(&self.cfg.identity, b, prompt).iter().enumerate() {
            match self.index.get(hash) {
                Some(&id) if self.blocks[id].tokens == prompt[i * b..(i + 1) * b] => hits.push(id),
                _ => break,
            }
        }
        while !hits.is_empty() && hits.len() * b >= prompt.len() {
            hits.pop();
        }
        hits.iter().filter(|&&id| self.blocks[id].pins > 0).map(|&id| self.blocks[id].pages.len()).sum()
    }

    pub fn stats(&self) -> KvStats {
        self.stats
    }
    pub fn events(&mut self) -> &mut EventLog {
        &mut self.events
    }
    pub fn registered_blocks(&self) -> usize {
        self.index.len()
    }
    pub fn table(&self, seq: SeqId) -> Option<&[PageIndex]> {
        self.sequences.get(&seq).map(|s| s.table.as_slice())
    }
    pub fn state_of_blocks(&self, seq: SeqId) -> Vec<BlockState> {
        self.sequences.get(&seq).map_or_else(Vec::new, |s| s.blocks.iter().map(|&b| self.blocks[b].state).collect())
    }

    fn pages_per_block(&self) -> usize {
        self.cfg.hash_block_tokens / self.cfg.page_tokens
    }

    fn new_block(&mut self) -> BlockId {
        match self.unused_blocks.pop() {
            Some(id) => id,
            None => {
                self.blocks.push(Block::reset());
                self.blocks.len() - 1
            }
        }
    }

    /// Admit a prompt: share the longest chain of Registered blocks it begins with (each verified token by token), allocate pages for the rest.
    pub fn admit(&mut self, seq: SeqId, prompt: &[u32]) -> Result<Admission, KvError> {
        let b = self.cfg.hash_block_tokens;
        let hashes = chain(&self.cfg.identity, b, prompt);
        let mut hits: Vec<BlockId> = Vec::new();
        for (i, hash) in hashes.iter().enumerate() {
            let Some(&id) = self.index.get(hash) else { break };
            // The tokens are compared, never trusted to the hash alone.
            if self.blocks[id].tokens != prompt[i * b..(i + 1) * b] {
                self.stats.collisions_rejected += 1;
                break;
            }
            hits.push(id);
        }
        // The last prompt token is always computed (its logits start the generation): a prompt that is entirely cached keeps one block back.
        while !hits.is_empty() && hits.len() * b >= prompt.len() {
            hits.pop();
        }
        let cached = hits.len() * b;
        let pages_total = prompt.len().div_ceil(self.cfg.page_tokens);
        let needed = pages_total - cached / self.cfg.page_tokens;
        // Pin the hits first so that making room cannot evict what this prompt is about to use.
        self.clock += 1;
        for &id in &hits {
            self.blocks[id].pins += 1;
            self.blocks[id].last_use = self.clock;
        }
        if let Err(e) = self.make_room(needed) {
            for &id in &hits {
                self.blocks[id].pins -= 1;
            }
            return Err(e);
        }
        self.stats.admissions += 1;
        self.stats.prompt_tokens += prompt.len() as u64;
        self.stats.hit_tokens += cached as u64;
        let table: Vec<PageIndex> = hits.iter().flat_map(|&id| self.blocks[id].pages.clone()).collect();
        self.sequences.insert(seq, Sequence { tokens: prompt.to_vec(), blocks: hits, table, computed: cached });
        self.grow(seq, prompt.len())?;
        let table = self.sequences[&seq].table.clone();
        Ok(Admission { cached_tokens: cached, table })
    }

    /// Make sure the sequence's table covers `len` tokens; returns the new `(table index, page)` entries (the deltas of the plan).
    pub fn reserve(&mut self, seq: SeqId, len: usize) -> Result<Vec<(usize, PageIndex)>, KvError> {
        let before = self.sequences.get(&seq).ok_or(KvError::UnknownSequence(seq))?.table.len();
        let needed = len.div_ceil(self.cfg.page_tokens).saturating_sub(before);
        self.make_room(needed)?;
        self.grow(seq, len)?;
        let table = &self.sequences[&seq].table;
        Ok((before..table.len()).map(|i| (i, table[i])).collect())
    }

    fn grow(&mut self, seq: SeqId, len: usize) -> Result<(), KvError> {
        let (k, p) = (self.pages_per_block(), self.cfg.page_tokens);
        let target = len.div_ceil(p);
        while self.sequences[&seq].table.len() < target {
            let page = self.free_pages.pop_front().ok_or(KvError::PoolFull { needed: 1, free: 0 })?;
            let owner = {
                let s = &self.sequences[&seq];
                s.blocks.get(s.table.len() / k).copied()
            };
            let id = match owner {
                Some(id) => id,
                None => {
                    let id = self.new_block();
                    let block = &mut self.blocks[id];
                    *block = Block::reset();
                    block.state = BlockState::Partial;
                    block.pins = 1;
                    self.sequences.get_mut(&seq).expect("sequence").blocks.push(id);
                    id
                }
            };
            self.blocks[id].pages.push(page);
            self.sequences.get_mut(&seq).expect("sequence").table.push(page);
        }
        self.refresh(seq);
        Ok(())
    }

    /// Mark the blocks whose tokens are all known as Complete.
    fn refresh(&mut self, seq: SeqId) {
        let b = self.cfg.hash_block_tokens;
        let s = &self.sequences[&seq];
        let known = s.tokens.len() / b;
        for (i, &id) in s.blocks.iter().enumerate().take(known) {
            if self.blocks[id].state == BlockState::Partial {
                self.blocks[id].state = BlockState::Complete;
                self.blocks[id].tokens = s.tokens[i * b..(i + 1) * b].to_vec();
            }
        }
    }

    /// Record a generated token (its page must have been reserved).
    pub fn push_token(&mut self, seq: SeqId, token: u32) -> Result<(), KvError> {
        self.sequences.get_mut(&seq).ok_or(KvError::UnknownSequence(seq))?.tokens.push(token);
        self.refresh(seq);
        Ok(())
    }

    /// The KV of the sequence's tokens `[0, upto)` is computed: register every block that is now fully computed, emitting `BlockStored`.
    pub fn mark_computed(&mut self, seq: SeqId, upto: usize) -> Result<(), KvError> {
        let b = self.cfg.hash_block_tokens;
        let s = self.sequences.get_mut(&seq).ok_or(KvError::UnknownSequence(seq))?;
        s.computed = s.computed.max(upto);
        let (computed, blocks) = (s.computed, s.blocks.clone());
        for (i, &id) in blocks.iter().enumerate() {
            if (i + 1) * b > computed {
                break;
            }
            if self.blocks[id].state != BlockState::Complete {
                continue;
            }
            let parent_hash = if i == 0 { self.cfg.identity } else { self.blocks[blocks[i - 1]].hash.unwrap_or(self.cfg.identity) };
            let hash = block_hash(&parent_hash, b, &self.blocks[id].tokens);
            self.blocks[id].hash = Some(hash);
            if self.index.contains_key(&hash) {
                // Another sequence registered the same prefix first: this copy stays private.
                continue;
            }
            self.registrations += 1;
            self.clock += 1;
            let parent = (i > 0).then(|| self.index.get(&parent_hash).copied()).flatten();
            if let Some(p) = parent {
                self.blocks[p].children += 1;
            }
            let block = &mut self.blocks[id];
            block.state = BlockState::Registered;
            block.parent = parent;
            block.parent_hash = Some(parent_hash);
            block.last_use = self.clock;
            block.registered_at = self.registrations;
            let tokens = block.tokens.clone();
            self.index.insert(hash, id);
            self.stats.registered_blocks += 1;
            self.events.emit(|event_id| KvEvent::BlockStored { id: event_id, hash, parent_hash, block_size: b, tier: Tier::Gpu, tokens });
        }
        Ok(())
    }

    /// The sequence is finished or cancelled: private blocks free their pages now, Registered blocks stay cached (unpinned, evictable).
    pub fn release(&mut self, seq: SeqId) -> Result<(), KvError> {
        let s = self.sequences.remove(&seq).ok_or(KvError::UnknownSequence(seq))?;
        self.clock += 1;
        for id in s.blocks {
            let block = &mut self.blocks[id];
            block.pins -= 1;
            block.last_use = self.clock;
            if block.pins == 0 && block.state != BlockState::Registered {
                let pages = std::mem::take(&mut block.pages);
                *block = Block::reset();
                self.free_pages.extend(pages);
                self.unused_blocks.push(id);
            }
        }
        Ok(())
    }

    /// Free pages until `needed` are available by evicting unpinned leaf blocks, least recently used first.
    fn make_room(&mut self, needed: usize) -> Result<(), KvError> {
        // Decide before evicting anything: a request that cannot fit must not empty the cache on its way to failing. Every unpinned
        // Registered block can go (a pinned child pins its parent), so the pool can offer the free pages plus all of those.
        let evictable: usize = self.index.values().filter(|&&id| self.blocks[id].pins == 0).map(|&id| self.blocks[id].pages.len()).sum();
        if self.free_pages.len() + evictable < needed {
            return Err(KvError::PoolFull { needed, free: self.free_pages.len() });
        }
        while self.free_pages.len() < needed {
            let victim = self.index.values().copied().filter(|&id| self.blocks[id].pins == 0 && self.blocks[id].children == 0).min_by_key(|&id| (self.blocks[id].last_use, self.blocks[id].registered_at));
            let Some(id) = victim else { return Err(KvError::PoolFull { needed, free: self.free_pages.len() }) };
            self.evict(id);
        }
        Ok(())
    }

    fn evict(&mut self, id: BlockId) {
        let hash = self.blocks[id].hash.expect("a registered block has a hash");
        self.index.remove(&hash);
        if let Some(p) = self.blocks[id].parent {
            self.blocks[p].children -= 1;
        }
        let pages = std::mem::take(&mut self.blocks[id].pages);
        self.blocks[id] = Block::reset();
        self.free_pages.extend(pages);
        self.unused_blocks.push(id);
        self.stats.evicted_blocks += 1;
        self.events.emit(|event_id| KvEvent::BlockRemoved { id: event_id, hash });
    }

    /// Empty the index (one `Cleared` event). Blocks a running sequence still pins stay usable by it and free their pages when it ends.
    pub fn clear(&mut self) {
        for id in self.index.values().copied().collect::<Vec<_>>() {
            let block = &mut self.blocks[id];
            block.parent = None;
            block.children = 0;
            if block.pins == 0 {
                let pages = std::mem::take(&mut block.pages);
                *block = Block::reset();
                self.free_pages.extend(pages);
                self.unused_blocks.push(id);
            } else {
                block.state = BlockState::Complete;
            }
        }
        self.index.clear();
        self.events.emit(|event_id| KvEvent::Cleared { id: event_id });
    }

    /// Test hook: overwrite the tokens stored under `hash`, to show that a hit is verified against the tokens and not trusted to the hash.
    #[cfg(test)]
    pub(crate) fn forge_tokens(&mut self, hash: &BlockHash, tokens: Vec<u32>) {
        let id = self.index[hash];
        self.blocks[id].tokens = tokens;
    }

    /// The index now, parents before children, with the id of the last event it includes (a consumer resumes from there).
    pub fn snapshot(&self) -> Snapshot {
        let mut ids: Vec<BlockId> = self.index.values().copied().collect();
        ids.sort_by_key(|&id| self.blocks[id].registered_at);
        let blocks = ids
            .into_iter()
            .map(|id| {
                let block = &self.blocks[id];
                let hash = block.hash.expect("registered");
                let parent_hash = block.parent_hash.unwrap_or(self.cfg.identity);
                StoredBlock { hash, parent_hash, tier: Tier::Gpu, tokens: block.tokens.clone() }
            })
            .collect();
        Snapshot { header: self.events.header().clone(), last_id: self.events.last_id(), blocks }
    }
}
