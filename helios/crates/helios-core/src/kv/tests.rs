//! The prefix cache and its events (ADR 0008 verification plan): lifecycle, sharing, verification, pins and eviction, event ids, snapshots.

use std::collections::HashSet;

use super::*;

fn identity(byte: u8) -> BlockHash {
    BlockHash([byte; 32])
}

fn manager(pages: usize, page_tokens: usize, block: usize) -> KvManager {
    KvManager::new(KvConfig { pages, page_tokens, hash_block_tokens: block, identity: identity(7) }).unwrap()
}

fn prompt(from: u32, len: usize) -> Vec<u32> {
    (from..from + len as u32).collect()
}

/// Admit, compute the whole prompt, and (for `finish`) release.
fn run(m: &mut KvManager, seq: SeqId, tokens: &[u32], finish: bool) -> Admission {
    let admission = m.admit(seq, tokens).unwrap();
    m.mark_computed(seq, tokens.len()).unwrap();
    if finish {
        m.release(seq).unwrap();
    }
    admission
}

#[test]
fn a_block_goes_from_partial_to_complete_to_registered_and_only_registered_blocks_are_shared() {
    let mut m = manager(16, 4, 8);
    let tokens = prompt(0, 20);
    m.admit(1, &tokens).unwrap();
    use BlockState::*;
    assert_eq!(m.state_of_blocks(1), vec![Complete, Complete, Partial], "two full blocks known, the third still partial");
    assert_eq!(m.registered_blocks(), 0, "nothing is shared before its KV is computed");
    // A second request now finds nothing to share.
    assert_eq!(m.admit(2, &tokens).unwrap().cached_tokens, 0);
    m.release(2).unwrap();
    m.mark_computed(1, 20).unwrap();
    assert_eq!(m.state_of_blocks(1), vec![Registered, Registered, Partial]);
    assert_eq!(m.registered_blocks(), 2);
}

#[test]
fn a_second_request_shares_the_registered_prefix_and_always_computes_its_last_token() {
    let mut m = manager(16, 4, 8);
    let tokens = prompt(0, 20);
    let first = run(&mut m, 1, &tokens, false);
    assert_eq!(first.cached_tokens, 0);
    let second = m.admit(2, &tokens).unwrap();
    assert_eq!(second.cached_tokens, 16, "two blocks of 8");
    assert_eq!(second.table[..4], first.table[..4], "the shared prefix is the same pages");
    assert_ne!(second.table[4], first.table[4], "the partial block is private");
    // A prompt that is exactly the cached blocks keeps the last block back so there is a token to compute.
    let exact = m.admit(3, &tokens[..16]).unwrap();
    assert_eq!(exact.cached_tokens, 8);
    let stats = m.stats();
    assert_eq!((stats.admissions, stats.prompt_tokens, stats.hit_tokens), (3, 56, 24));
    assert!((stats.hit_rate() - 24.0 / 56.0).abs() < 1e-9);
}

#[test]
fn a_forged_collision_is_rejected_by_the_token_check() {
    let mut m = manager(16, 4, 8);
    let tokens = prompt(0, 20);
    run(&mut m, 1, &tokens, true);
    let first_hash = chain(&identity(7), 8, &tokens)[0];
    // Someone arranges for the stored block to hold other tokens than the hash was made from.
    m.forge_tokens(&first_hash, prompt(1000, 8));
    let admission = m.admit(2, &tokens).unwrap();
    assert_eq!(admission.cached_tokens, 0, "the hash matched but the tokens did not: not a hit");
    assert_eq!(m.stats().collisions_rejected, 1);
}

#[test]
fn pages_are_freed_when_a_sequence_is_cancelled_before_its_kv_exists() {
    let mut m = manager(16, 4, 8);
    m.admit(1, &prompt(0, 20)).unwrap();
    assert_eq!(m.pages_used(), 5);
    m.release(1).unwrap();
    assert_eq!(m.pages_free(), 16, "no block was registered, every page is back");
    assert_eq!(m.registered_blocks(), 0);
    // A finished sequence keeps its registered blocks cached and frees the rest.
    run(&mut m, 2, &prompt(0, 20), true);
    assert_eq!(m.registered_blocks(), 2);
    assert_eq!(m.pages_used(), 4, "two cached blocks of two pages; the partial third is freed");
}

#[test]
fn eviction_takes_unpinned_leaf_blocks_least_recently_used_first_and_never_a_pinned_one() {
    // 8 pages, one page per block of 4 tokens.
    let mut m = manager(8, 4, 4);
    let rx = m.events().subscribe();
    let a = prompt(0, 16);
    run(&mut m, 1, &a, true);
    assert_eq!((m.pages_used(), m.registered_blocks()), (4, 4));
    let chain_a = chain(&identity(7), 4, &a);
    // Another prompt needs 6 pages: two cached blocks must go, the deepest first (a parent is never evicted before its child).
    run(&mut m, 2, &prompt(100, 24), true);
    let removed: Vec<BlockHash> = rx.try_iter().filter_map(|e| if let KvEvent::BlockRemoved { hash, .. } = e { Some(hash) } else { None }).collect();
    assert_eq!(removed, vec![chain_a[3], chain_a[2]], "leaves first, deepest first");
    // The prefix of A that survives is still shared; pin it with a new request, then fill the pool: pinned blocks are not evicted.
    let pinned = m.admit(3, &a[..9]).unwrap();
    assert_eq!(pinned.cached_tokens, 8);
    let error = m.admit(4, &prompt(500, 32)).unwrap_err();
    assert!(matches!(error, KvError::PoolFull { .. }), "{error}");
    // Request 3 evicted one of request 2's blocks to get its page (7 left); the failed request 4 evicted nothing on its way to failing.
    assert_eq!(m.registered_blocks(), 2 + 5);
}

#[test]
fn event_ids_are_gap_free_and_monotonic_and_a_consumer_rebuilds_the_index_from_events_alone() {
    let mut m = manager(8, 4, 4);
    let rx = m.events().subscribe();
    let header = m.events().header().clone();
    run(&mut m, 1, &prompt(0, 16), true);
    run(&mut m, 2, &prompt(100, 24), true); // evicts
    run(&mut m, 3, &prompt(0, 12), true);
    m.clear();
    run(&mut m, 4, &prompt(0, 9), true);
    let events: Vec<KvEvent> = rx.try_iter().collect();
    assert!(events.len() > 10);
    for (i, event) in events.iter().enumerate() {
        assert_eq!(event.id(), i as u64 + 1, "ids rise by exactly one");
    }
    assert!(events.iter().any(|e| matches!(e, KvEvent::BlockRemoved { .. })) && events.iter().any(|e| matches!(e, KvEvent::Cleared { .. })));
    let mut mirror = Mirror::new(header);
    for event in &events {
        mirror.apply(event).expect("every event applies");
    }
    let snapshot = m.snapshot();
    assert_eq!(mirror.last_id, snapshot.last_id);
    let rebuilt: HashSet<BlockHash> = mirror.blocks.keys().copied().collect();
    let live: HashSet<BlockHash> = snapshot.blocks.iter().map(|b| b.hash).collect();
    assert_eq!(rebuilt, live, "the index rebuilt from events alone equals the index");
}

#[test]
fn a_lost_event_is_a_gap_and_a_snapshot_resumes_the_stream() {
    let mut m = manager(8, 4, 4);
    let rx = m.events().subscribe();
    run(&mut m, 1, &prompt(0, 13), true);
    let events: Vec<KvEvent> = rx.try_iter().collect();
    assert_eq!(events.len(), 3);
    let mut mirror = Mirror::new(m.events().header().clone());
    mirror.apply(&events[0]).unwrap();
    assert_eq!(mirror.apply(&events[2]), Err(MirrorError::Gap { expected: 2, got: 3 }), "event 2 was lost");
    // Resync: snapshot, then the events after it.
    let mut resumed = Mirror::from_snapshot(m.snapshot());
    run(&mut m, 2, &prompt(200, 13), true);
    let from = resumed.last_id;
    for event in rx.try_iter().filter(move |e| e.id() > from) {
        resumed.apply(&event).unwrap();
    }
    let live: HashSet<BlockHash> = m.snapshot().blocks.iter().map(|b| b.hash).collect();
    assert_eq!(resumed.blocks.keys().copied().collect::<HashSet<_>>(), live);
}

#[test]
fn parent_links_form_a_forest_rooted_at_the_identity() {
    let mut m = manager(32, 4, 4);
    run(&mut m, 1, &prompt(0, 17), true);
    let mut shared_then_diverging = prompt(0, 8);
    shared_then_diverging.extend(prompt(900, 9));
    run(&mut m, 2, &shared_then_diverging, true);
    let snapshot = m.snapshot();
    let mut seen: HashSet<BlockHash> = HashSet::new();
    let mut roots = 0;
    for block in &snapshot.blocks {
        if block.parent_hash == identity(7) {
            roots += 1;
        } else {
            assert!(seen.contains(&block.parent_hash), "a block's parent was stored before it");
        }
        assert!(seen.insert(block.hash), "no hash twice");
    }
    assert_eq!(roots, 1, "both prompts start with the same first block, so one root");
    assert_eq!(snapshot.blocks.len(), 4 + 2, "four blocks of the first prompt, two new ones of the second (it shares the first two)");
}

#[test]
fn a_mirror_refuses_a_stored_event_whose_hash_does_not_match_its_tokens() {
    let mut m = manager(8, 4, 4);
    let header = m.events().header().clone();
    let forged = KvEvent::BlockStored { id: 1, hash: identity(1), parent_hash: header.identity, block_size: 4, tier: Tier::Gpu, tokens: vec![1, 2, 3, 4] };
    assert_eq!(Mirror::new(header).apply(&forged), Err(MirrorError::BadHash(identity(1))));
}

#[test]
fn the_same_tokens_under_another_identity_share_nothing() {
    let tokens = prompt(0, 20);
    let mut a = manager(16, 4, 8);
    run(&mut a, 1, &tokens, true);
    let first: HashSet<BlockHash> = a.snapshot().blocks.iter().map(|b| b.hash).collect();
    let mut b = KvManager::new(KvConfig { pages: 16, page_tokens: 4, hash_block_tokens: 8, identity: identity(8) }).unwrap();
    run(&mut b, 1, &tokens, true);
    let second: HashSet<BlockHash> = b.snapshot().blocks.iter().map(|b| b.hash).collect();
    assert!(first.is_disjoint(&second));
}

#[test]
fn the_hash_block_must_be_a_whole_number_of_pages() {
    assert!(KvManager::new(KvConfig { pages: 4, page_tokens: 64, hash_block_tokens: 96, identity: identity(1) }).is_err());
    assert!(KvManager::new(KvConfig { pages: 4, page_tokens: 64, hash_block_tokens: 128, identity: identity(1) }).is_ok());
}

#[test]
fn growing_a_sequence_adds_pages_and_blocks_and_registers_generated_blocks_too() {
    let mut m = manager(16, 4, 8);
    m.admit(1, &prompt(0, 7)).unwrap();
    m.mark_computed(1, 7).unwrap();
    let mut length = 7;
    for token in 1000..1010 {
        let deltas = m.reserve(1, length + 1).unwrap();
        m.push_token(1, token).unwrap();
        length += 1;
        m.mark_computed(1, length).unwrap();
        let crossing = length % 4 == 1;
        assert_eq!(deltas.len(), usize::from(crossing), "a page is added only when the length crosses a page boundary (length {length})");
    }
    assert_eq!(m.table(1).unwrap().len(), 5, "17 tokens in pages of 4");
    assert_eq!(m.registered_blocks(), 2, "two full blocks of 8, the second made of generated tokens");
}
