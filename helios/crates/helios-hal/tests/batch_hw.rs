//! Hardware tests (H100) of the batched decode path through `submit_step`: a row is isolated from its neighbours (bit-identical logits in the same
//! bucket whoever shares it), a fixed composition repeats bit for bit, a masked row writes nothing, and the difference of a row between buckets is
//! reported. The tolerance of the first three is exact equality (declared before the first run: any differing bit fails); the fourth is
//! information only (GEMM reduction order depends on M, so bucket N and bucket 1 may differ). Each test drives the plans by hand with fixed pages,
//! the way a scheduler would, so the pages and slots are the same in every scenario.
//!
//! ```text
//! HYPERION_WEIGHTS_PACKAGE=/path/model.hyperion HYPERION_COMPACT_ORACLE=/path/<id>.compact.json \
//!   cargo test --release -p helios-hal --features sm90 --test batch_hw -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "sm90")]

use std::path::PathBuf;

use helios_hal::step::{Capacity, Input, PageDelta, Row, SamplingInit, SlotInit, StepPlan, Want};
use helios_hal::{ComputeBackend, HopperBackend, KvPool};
use hyperion_specification::Manifest;

const PAGE: usize = 64;

fn package() -> PathBuf {
    PathBuf::from(std::env::var("HYPERION_WEIGHTS_PACKAGE").expect("set HYPERION_WEIGHTS_PACKAGE to the .hyperion path"))
}

fn seed_prompt() -> Vec<u32> {
    let path = std::env::var_os("HYPERION_COMPACT_ORACLE").expect("set HYPERION_COMPACT_ORACLE");
    helios_hal::oracle::CompactOracle::from_json(&std::fs::read(path).unwrap()).expect("compact oracle").prompts[0].ids.clone()
}

/// The backend with the model loaded and a pool of `pages` pages and decode buckets up to `max_batch`.
fn open(max_batch: usize, pages: usize) -> (HopperBackend, KvPool) {
    println!("BUILD revision={}", std::env::var("HYPERION_REVISION").unwrap_or_else(|_| "unknown".into()));
    let package = package();
    let Manifest::V1(manifest) = Manifest::from_json_bytes(&std::fs::read(package.with_extension("manifest.json")).unwrap()).unwrap();
    let backend = HopperBackend::new().expect("an H100 must be present");
    helios_hal::load_package(&backend, &package, &manifest).expect("upload");
    let pool = backend.open_kv(max_batch, pages * PAGE).expect("the KV pool");
    println!("POOL {:?}", pool.info());
    (backend, pool)
}

/// One sequence of a scenario: its slot, its prompt and the physical pages of its table (enough for the prompt and the steps).
#[derive(Clone)]
struct Seq {
    slot: u32,
    prompt: Vec<u32>,
    pages: Vec<u32>,
}

/// Different prompts built from the seed prompt: sequence `i` has `20 + 13 i` tokens, read from the seed at a different stride.
fn sequences(count: usize) -> Vec<Seq> {
    let seed = seed_prompt();
    (0..count)
        .map(|i| Seq {
            slot: i as u32,
            prompt: (0..20 + 13 * i).map(|j| seed[(j * (i + 1) + i * 5) % seed.len()]).collect(),
            pages: vec![(2 * i) as u32, (2 * i + 1) as u32],
        })
        .collect()
}

fn chunk_capacity(tokens: usize) -> u32 {
    tokens.next_power_of_two().max(128) as u32
}

/// Prefill every sequence (one plan each), then `steps` decode steps of `bucket` rows holding all of them, asking for the logits of every row.
/// Returns, per sequence, the logits after the prompt's first sampled token was fed and after each later step.
fn run(backend: &HopperBackend, pool: &KvPool, seqs: &[Seq], bucket: u16, steps: usize) -> Vec<Vec<Vec<f32>>> {
    let mut step = 0u64;
    let mut take = |plan: &StepPlan| {
        let ticket = backend.submit_step(pool, plan).expect("submit_step");
        backend.take_output(ticket).expect("take_output")
    };
    for s in seqs {
        step += 1;
        let deltas = s.pages.iter().enumerate().map(|(i, &page)| PageDelta { slot: s.slot, index: i as u32, page }).collect();
        let plan = StepPlan {
            step,
            capacity: Capacity::Prefill { rows: chunk_capacity(s.prompt.len()) },
            rows: vec![Row { slot: s.slot, input: Input::Tokens(s.prompt.clone()), n_tokens: s.prompt.len() as u32, position: 0, table_len: s.prompt.len().div_ceil(PAGE) as u32, want: Want::Token }],
            deltas,
            inits: vec![SlotInit { slot: s.slot, sampling: SamplingInit::GREEDY, max_new_tokens: steps as u32 + 1 }],
        };
        let out = take(&plan);
        assert_eq!(out.rows[0].tokens.len(), 1, "the prefill sampled a token");
    }
    let mut logits = vec![Vec::new(); seqs.len()];
    let mut ordered: Vec<(usize, &Seq)> = seqs.iter().enumerate().collect();
    ordered.sort_by_key(|(_, s)| s.slot);
    for k in 0..steps {
        step += 1;
        let rows = ordered
            .iter()
            .map(|(_, s)| Row { slot: s.slot, input: Input::FromDevice, n_tokens: 1, position: (s.prompt.len() + k) as u32, table_len: (s.prompt.len() + k + 1).div_ceil(PAGE) as u32, want: Want::Logits })
            .collect();
        let plan = StepPlan { step, capacity: Capacity::Decode { bucket }, rows, deltas: Vec::new(), inits: Vec::new() };
        let out = take(&plan);
        for ((i, _), row) in ordered.iter().zip(&out.rows) {
            logits[*i].push(row.logits.clone().expect("logits were asked for"));
        }
    }
    logits
}

fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter().map(|r| r.iter().map(|v| v.to_bits()).collect()).collect()
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter().zip(b).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs())).fold(0.0, f32::max)
}

#[test]
#[ignore = "needs an H100, a compiled package and the compact oracle (see module docs)"]
fn a_row_is_isolated_from_its_neighbours_and_a_fixed_batch_repeats_bit_for_bit() {
    let (backend, pool) = open(8, 40);
    let seqs = sequences(8);
    let steps = 8;
    // The same bucket (8) and the same row 0; the neighbours differ: none, three, seven, three others.
    let alone = run(&backend, &pool, &seqs[..1], 8, steps);
    let three = run(&backend, &pool, &seqs[..4], 8, steps);
    let seven = run(&backend, &pool, &seqs[..8], 8, steps);
    let mut others: Vec<Seq> = vec![seqs[0].clone()];
    others.extend(seqs[4..7].iter().cloned().enumerate().map(|(k, s)| Seq { slot: k as u32 + 1, pages: vec![(2 * (k + 1)) as u32, (2 * (k + 1) + 1) as u32], ..s }));
    let other_three = run(&backend, &pool, &others, 8, steps);
    for (name, run) in [("three neighbours", &three), ("seven neighbours", &seven), ("three other neighbours", &other_three)] {
        assert!(bits(&alone[0]) == bits(&run[0]), "row 0 differs with {name}: max |difference| {}", max_diff(&alone[0], &run[0]));
        println!("row 0 alone == row 0 with {name}: bit-identical over {steps} steps");
    }
    // Every row of a fixed composition repeats.
    let again = run(&backend, &pool, &seqs[..8], 8, steps);
    for r in 0..8 {
        assert!(bits(&seven[r]) == bits(&again[r]), "row {r} differs between two runs of the same batch");
    }
    println!("a fixed batch of 8 rows repeats bit for bit over {steps} steps");
    backend.shutdown().unwrap();
}

#[test]
#[ignore = "needs an H100, a compiled package and the compact oracle (see module docs)"]
fn masked_rows_write_nothing_and_a_row_between_buckets_is_reported() {
    let (backend, pool) = open(8, 40);
    let page_bytes = pool.info().bytes_per_page;
    let seqs = sequences(3);
    // Pages 6.. belong to nobody: a step of bucket 8 with three live rows must leave them as they were (and the unused rows of the owned pages).
    let before = backend.read_pool_bytes((6 * page_bytes) as u64, 34 * page_bytes).unwrap();
    let in_8 = run(&backend, &pool, &seqs, 8, 6);
    let after = backend.read_pool_bytes((6 * page_bytes) as u64, 34 * page_bytes).unwrap();
    assert!(before == after, "a page nobody owns changed: a masked row or a stray row wrote KV");
    println!("34 pages owned by no sequence are unchanged after prefill and 6 steps of bucket 8 with 3 live rows");
    // Information: the same row in bucket 1 (alone) and in bucket 8.
    let in_1 = run(&backend, &pool, &seqs[..1], 1, 6);
    let diff = max_diff(&in_1[0], &in_8[0]);
    println!("INFO row 0, bucket 1 vs bucket 8: max |logit difference| {diff:.4} over 6 steps (not a gate: the GEMM reduction depends on M)");
    backend.shutdown().unwrap();
}
