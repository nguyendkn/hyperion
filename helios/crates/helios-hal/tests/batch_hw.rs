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
    let take = |plan: &StepPlan| {
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

/// Greedy tokens of `prompt` through the plan path alone: one prefill plan (the pages of slot 0 are 0..), then `count - 1` decode steps.
fn plan_greedy(backend: &HopperBackend, pool: &KvPool, prompt: &[u32], count: usize) -> Vec<u32> {
    let pages: Vec<u32> = (0..(prompt.len() + count).div_ceil(PAGE) as u32).collect();
    let take = |plan: &StepPlan| {
        let ticket = backend.submit_step(pool, plan).expect("submit_step");
        backend.take_output(ticket).expect("take_output")
    };
    let mut tokens = Vec::new();
    // The prompt in chunks of the largest capacity (4096), as the scheduler does.
    let mut start = 0;
    let mut step = 0;
    while start < prompt.len() {
        let end = (start + 4096).min(prompt.len());
        let last = end == prompt.len();
        step += 1;
        let plan = StepPlan {
            step,
            capacity: Capacity::Prefill { rows: chunk_capacity(end - start) },
            rows: vec![Row { slot: 0, input: Input::Tokens(prompt[start..end].to_vec()), n_tokens: (end - start) as u32, position: start as u32, table_len: end.div_ceil(PAGE) as u32, want: if last { Want::Token } else { Want::Nothing } }],
            deltas: if start == 0 { deltas_clone(&pages) } else { Vec::new() },
            inits: if start == 0 { vec![SlotInit { slot: 0, sampling: SamplingInit::GREEDY, max_new_tokens: count as u32 }] } else { Vec::new() },
        };
        let out = take(&plan);
        if last {
            tokens.push(out.rows[0].tokens[0]);
        }
        start = end;
    }
    for k in 1..count {
        step += 1;
        let position = (prompt.len() + k - 1) as u32;
        let plan = StepPlan { step, capacity: Capacity::Decode { bucket: 1 }, rows: vec![Row { slot: 0, input: Input::FromDevice, n_tokens: 1, position, table_len: (position as usize + 1).div_ceil(PAGE) as u32, want: Want::Token }], deltas: Vec::new(), inits: Vec::new() };
        tokens.push(take(&plan).rows[0].tokens[0]);
    }
    tokens
}

fn deltas_clone(pages: &[u32]) -> Vec<PageDelta> {
    pages.iter().enumerate().map(|(i, &page)| PageDelta { slot: 0, index: i as u32, page }).collect()
}

/// G2: the plan path gives the same greedy tokens as `generate` (the path every oracle was verified through) on each prompt of the compact
/// oracle that fits 4 chunks, and for the 0.5B the 16 tokens Hugging Face generated. Exact equality of the token ids.
#[test]
#[ignore = "needs an H100, a compiled package and the compact oracle (see module docs)"]
fn the_plan_path_gives_the_greedy_tokens_of_generate_and_of_hugging_face() {
    let path = std::env::var_os("HYPERION_COMPACT_ORACLE").expect("set HYPERION_COMPACT_ORACLE");
    let oracle = helios_hal::oracle::CompactOracle::from_json(&std::fs::read(path).unwrap()).expect("compact oracle");
    let (backend, pool) = open(1, 16384 / PAGE + 8);
    let count = 24;
    for prompt in oracle.prompts.iter().filter(|p| p.ids.len() + count <= 16384) {
        let via_plan = plan_greedy(&backend, &pool, &prompt.ids, count);
        let via_generate = {
            let mut ids = Vec::new();
            let package = package();
            let Manifest::V1(manifest) = Manifest::from_json_bytes(&std::fs::read(package.with_extension("manifest.json")).unwrap()).unwrap();
            let dims = helios_hal::Qwen2Dims::from_manifest(&manifest.model).unwrap();
            backend.generate_streaming(&dims, &prompt.ids, count, &[], &helios_hal::SamplingConfig::greedy(), &mut |t| ids.push(t)).expect("generate");
            ids
        };
        assert_eq!(via_plan, via_generate, "prompt of {} tokens: the plan path and generate disagree", prompt.ids.len());
        println!("prompt of {:>5} tokens: plan path == generate for {count} greedy tokens: {:?}", prompt.ids.len(), &via_plan[..8]);
    }
    if let Some(path) = std::env::var_os("HYPERION_ORACLE").filter(|p| std::path::Path::new(p).exists()) {
        let json: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        if let (Some(prompt), Some(want)) = (json["prompt_ids"].as_array(), json["generated_ids"].as_array()) {
            let prompt: Vec<u32> = prompt.iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let want: Vec<u32> = want.iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let got = plan_greedy(&backend, &pool, &prompt, want.len());
            assert_eq!(got, want, "the plan path differs from the Hugging Face generation");
            println!("plan path == Hugging Face generate for {} tokens", want.len());
        }
    }
    backend.shutdown().unwrap();
}

fn median_p10_p90(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[v.len() / 10], v[(v.len() * 9) / 10])
}

/// P5: GPU milliseconds per decode step of each bucket (10 windows of 128 chained graph launches, median [p10, p90]) with every row at context
/// 512, and the GPU time by group of kernels at the buckets named by `HYPERION_PROFILE_BUCKETS` (default 1, 8, 32).
#[test]
#[ignore = "needs an H100, a compiled package and the compact oracle (see module docs)"]
fn decode_step_time_by_bucket() {
    let (backend, pool) = open(32, 32 * 16);
    let seed = seed_prompt();
    let seqs: Vec<Seq> = (0..32usize)
        .map(|i| Seq { slot: i as u32, prompt: (0..512).map(|j| seed[(j + i) % seed.len()]).collect(), pages: ((16 * i) as u32..(16 * i + 16) as u32).collect() })
        .collect();
    // Prefill every slot (the decode steps only need their pages and a token in `slot_token`).
    let _ = run(&backend, &pool, &seqs, 32, 1);
    println!("context 512, every row; GPU ms per step, median [p10, p90] of 10 windows of 128 steps");
    for bucket in [1usize, 2, 4, 8, 16, 32] {
        let rows: Vec<(u32, u32)> = (0..bucket as u32).map(|slot| (slot, 513)).collect();
        let (m, lo, hi) = median_p10_p90(backend.time_decode_bucket(&rows, bucket, 128, 10).expect("timing"));
        println!("BUCKET {bucket:>2}: {m:.3} [{lo:.3}, {hi:.3}] ms/step = {:.1} tokens/s", bucket as f64 * 1000.0 / m);
    }
    let wanted: Vec<usize> = std::env::var("HYPERION_PROFILE_BUCKETS").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_else(|| vec![1, 8, 32]);
    for bucket in wanted {
        let rows: Vec<(u32, u32)> = (0..bucket as u32).map(|slot| (slot, 513)).collect();
        let repeats = 20;
        let mut groups = backend.profile_decode_bucket(&rows, bucket, repeats).expect("profile");
        let total: f64 = groups.iter().map(|g| g.1).sum();
        groups.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("PROFILE bucket {bucket}: {:.3} ms/step direct, by group (ms/step, share):", total / repeats as f64);
        for (group, ms) in groups {
            println!("   {group:<18} {:.3} {:>5.1}%", ms / repeats as f64, 100.0 * ms / total);
        }
    }
    backend.shutdown().unwrap();
}
