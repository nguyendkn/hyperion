//! Hardware tests of tensor parallelism at TP=2 on two GPUs of one node (gate K3, steps T2-T4): each rank holds its slices of the weights, the layers
//! all-reduce twice, the vocab-parallel head gathers, and the result is compared with the single-GPU run and with the float32 oracle.
//!
//! **Tolerances, declared before the first run** (never loosened):
//! * the TP=2 teacher-forced run passes the compact oracle under the same rules and the same tolerance as TP=1 (0.5B 0.3851, 7B 2.5574): binding;
//! * TP=2 against TP=1: every logit within the Hugging Face bfloat16-vs-float32 deviation the tolerance was derived from (`bf16_deviation`, the maximum over the
//!   compared positions: 0.5B 0.2568, 7B 1.7049): bf16 partial sums over two ranks differ from one rank by the same kind of rounding;
//! * two TP=2 runs of the same plans, and graph against direct launches at TP=2: bit-identical.
//!
//! ```text
//! HYPERION_WEIGHTS_PACKAGE=/path/model.hyperion HYPERION_COMPACT_ORACLE=/path/<id>.compact.json \
//!   cargo test --release -p helios-hal --features sm90 --test tp_hw -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "sm90")]

use std::path::PathBuf;

use helios_hal::oracle::{compare_at, CompactOracle};
use helios_hal::rank_group::ExpectedVersions;
use helios_hal::sharding::rank_slice;
use helios_hal::step::{Capacity, Input, PageDelta, Row, SamplingInit, SlotInit, StepOutput, StepPlan, Want};
use helios_hal::{BackendConfig, ComputeBackend, DecodeMode, Deployment, HopperBackend, TpBackend};
use hyperion_specification::Manifest;

const PAGE: usize = 64;

fn package() -> PathBuf {
    PathBuf::from(std::env::var("HYPERION_WEIGHTS_PACKAGE").expect("set HYPERION_WEIGHTS_PACKAGE to the .hyperion path"))
}

fn manifest() -> hyperion_specification::v1::Manifest {
    let Manifest::V1(m) = Manifest::from_json_bytes(&std::fs::read(package().with_extension("manifest.json")).unwrap()).unwrap();
    m
}

fn oracle() -> CompactOracle {
    let path = std::env::var_os("HYPERION_COMPACT_ORACLE").expect("set HYPERION_COMPACT_ORACLE");
    CompactOracle::from_json(&std::fs::read(path).unwrap()).expect("compact oracle")
}

fn chunk_capacity(tokens: usize) -> u32 {
    tokens.next_power_of_two().max(128) as u32
}

/// Teacher-forced run of `rows` oracle prompts through `submit` (a `StepPlan` in, the output out): every prompt prefilled in chunks of 4096 with the logits
/// of its last token asked for, then the oracle's own tokens fed one per step to all rows together. Returns, per row, the logits after the prompt and after
/// each forced token but the last.
fn forced(submit: &dyn Fn(&StepPlan) -> StepOutput, oracle: &CompactOracle, rows: usize, n: usize) -> Vec<Vec<Vec<f32>>> {
    let prompts = &oracle.prompts;
    let mut next_page = 0u32;
    let mut step = 0u64;
    let mut seqs = Vec::new();
    let mut traces = vec![Vec::new(); rows];
    for r in 0..rows {
        let prompt = &prompts[r % prompts.len()];
        let pages: Vec<u32> = (next_page..next_page + (prompt.ids.len() + n).div_ceil(PAGE) as u32).collect();
        next_page += pages.len() as u32;
        let mut start = 0;
        while start < prompt.ids.len() {
            let end = (start + 4096).min(prompt.ids.len());
            let last = end == prompt.ids.len();
            step += 1;
            let out = submit(&StepPlan {
                step,
                capacity: Capacity::Prefill { rows: chunk_capacity(end - start) },
                rows: vec![Row { slot: r as u32, input: Input::Tokens(prompt.ids[start..end].to_vec()), n_tokens: (end - start) as u32, position: start as u32, table_len: end.div_ceil(PAGE) as u32, want: if last { Want::Logits } else { Want::Nothing } }],
                deltas: if start == 0 { pages.iter().enumerate().map(|(i, &page)| PageDelta { slot: r as u32, index: i as u32, page }).collect() } else { Vec::new() },
                inits: if start == 0 { vec![SlotInit { slot: r as u32, sampling: SamplingInit::GREEDY, max_new_tokens: n as u32 }] } else { Vec::new() },
            });
            if last {
                traces[r].push(out.rows[0].logits.clone().expect("logits"));
            }
            start = end;
        }
        seqs.push(prompt);
    }
    let bucket = rows.next_power_of_two() as u16;
    for k in 1..n {
        step += 1;
        let plan_rows = (0..rows)
            .map(|r| {
                let p = seqs[r];
                Row { slot: r as u32, input: Input::Tokens(vec![p.generated_ids[k - 1]]), n_tokens: 1, position: (p.ids.len() + k - 1) as u32, table_len: (p.ids.len() + k).div_ceil(PAGE) as u32, want: Want::Logits }
            })
            .collect();
        let out = submit(&StepPlan { step, capacity: Capacity::Decode { bucket }, rows: plan_rows, deltas: Vec::new(), inits: Vec::new() });
        for (r, row) in out.rows.iter().enumerate() {
            traces[r].push(row.logits.clone().expect("logits"));
        }
    }
    traces
}

fn scenario(oracle: &CompactOracle) -> (usize, usize, usize) {
    let n = oracle.generated_tokens.min(oracle.prompts.iter().map(|p| p.generated_ids.len().min(p.reference.len())).min().unwrap());
    let rows = oracle.prompts.len().min(8);
    let pages: usize = (0..rows).map(|r| (oracle.prompts[r].ids.len() + n).div_ceil(PAGE)).sum();
    (rows, n, pages + 8)
}

fn bits(traces: &[Vec<Vec<f32>>]) -> Vec<u32> {
    traces.iter().flatten().flatten().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "needs two H100s on one node, libnccl.so.2, a compiled package and the compact oracle"]
fn each_rank_uploads_exactly_its_slices_of_the_weights() {
    println!("BUILD revision={}", std::env::var("HYPERION_REVISION").unwrap_or_else(|_| "unknown".into()));
    let manifest = manifest();
    let single = HopperBackend::new().expect("a GPU");
    helios_hal::load_package(&single, &package(), &manifest).expect("TP=1 load");
    let whole = single.weight_bytes();
    single.shutdown().unwrap();
    drop(single);
    let group = TpBackend::start(&[0, 1], &ExpectedVersions::default()).expect("a TP=2 group");
    let started = std::time::Instant::now();
    let bytes = group.load_package(&package(), &manifest).expect("sharded load");
    println!("TP=2 load of {:.1} s", started.elapsed().as_secs_f64());
    for rank in 0..2u32 {
        let config = BackendConfig::for_deployment(&manifest, Deployment { tp_size: 2, pp_size: 1, tp_rank: rank, pp_stage: 0 }).unwrap();
        let expected: usize = config
            .tensors
            .iter()
            .map(|t| {
                let full: usize = t.shape.iter().product::<usize>() * 2;
                match rank_slice(&config, t).unwrap() {
                    Some(_) if t.name == "embed_tokens.weight" => full,
                    Some(s) => full / t.shape[s.axis] * (s.end - s.start),
                    None => full,
                }
            })
            .sum();
        println!("rank {rank}: {} bytes uploaded ({:.1} MiB), the sharding plan says {expected}; one GPU holds {whole} ({:.1} MiB)", bytes[rank as usize], bytes[rank as usize] as f64 / 1048576.0, whole as f64 / 1048576.0);
        assert_eq!(bytes[rank as usize], expected, "rank {rank} must hold exactly its slices");
        assert!(bytes[rank as usize] < whole);
    }
}

/// The TP=1 teacher-forced logits (a plain backend on one GPU) and the TP=2 ones (a group), for the same plans.
fn both(oracle: &CompactOracle, rows: usize, n: usize, pages: usize) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>, TpBackend) {
    let manifest = manifest();
    let single = HopperBackend::new().expect("a GPU");
    helios_hal::load_package(&single, &package(), &manifest).expect("TP=1 load");
    let pool = single.open_kv_serving(8, pages * PAGE).expect("TP=1 pool");
    let one = forced(&|plan| single.take_output(single.submit_step_mode(&pool, plan, DecodeMode::Graph).expect("submit")).expect("take"), oracle, rows, n);
    drop(pool);
    single.shutdown().unwrap();
    drop(single);
    let group = TpBackend::start(&[0, 1], &ExpectedVersions::default()).expect("a TP=2 group");
    group.load_package(&package(), &manifest).expect("sharded load");
    let pool_pages = group.open_kv(8, pages * PAGE).expect("pools");
    println!("TP=2 pool: {pool_pages:?} pages per rank (same on every rank)");
    let two = forced(&|plan| group.submit_step(plan, DecodeMode::Graph).expect("tp2 step"), oracle, rows, n);
    (one, two, group)
}

#[test]
#[ignore = "needs two H100s on one node, libnccl.so.2, a compiled package and the compact oracle"]
fn tp2_passes_the_oracle_and_is_bit_identical_run_to_run_and_graph_to_direct() {
    println!("BUILD revision={}", std::env::var("HYPERION_REVISION").unwrap_or_else(|_| "unknown".into()));
    let oracle = oracle();
    let (rows, n, pages) = scenario(&oracle);
    let (_one, two, group) = both(&oracle, rows, n, pages);
    let (mut overall, mut at, mut worst_ratio) = (0f32, String::new(), 0f32);
    let mut failures = Vec::new();
    for (r, trace) in two.iter().enumerate() {
        let prompt = &oracle.prompts[r % oracle.prompts.len()];
        for (i, (logits, record)) in trace.iter().zip(&prompt.reference).enumerate() {
            let c = compare_at(logits, record, &oracle.tolerance, prompt.bf16_deviation.as_ref().map(|d| d[i].0)).expect("comparison");
            worst_ratio = worst_ratio.max(c.max_abs_top_logits / oracle.tolerance.max_abs_top_logits);
            if c.max_abs_top_logits > overall {
                overall = c.max_abs_top_logits;
                at = format!("row {r} ({}, {} prompt tokens), step {i}", prompt.label, prompt.ids.len());
            }
            if !c.within(&oracle.tolerance) {
                failures.push(format!("row {r} ({}): step {i}: top-k diff {:.4} (tolerance {}), logsumexp diff {:.4}", prompt.label, c.max_abs_top_logits, oracle.tolerance.max_abs_top_logits, c.abs_logsumexp));
            }
        }
    }
    println!("TP=2 ORACLE SUMMARY: max top-k logit diff {overall:.4} at {at}; tolerance {}; ratio to the tolerance {worst_ratio:.2}", oracle.tolerance.max_abs_top_logits);
    assert!(failures.is_empty(), "TP=2 fails the oracle:
{}", failures.join("
"));
    let again = forced(&|plan| group.submit_step(plan, DecodeMode::Graph).expect("tp2 step"), &oracle, rows, n);
    assert!(bits(&two) == bits(&again), "two TP=2 runs of the same plans differ");
    println!("TP=2 run to run: {} logits bit-identical", bits(&two).len());
    let direct = forced(&|plan| group.submit_step(plan, DecodeMode::Direct).expect("tp2 direct step"), &oracle, rows, n);
    assert!(bits(&two) == bits(&direct), "TP=2 graph and direct launches differ");
    println!("TP=2 graph == direct: {} logits bit-identical", bits(&two).len());
}

/// The declared bound of this file's header (TP=2 against TP=1 within the Hugging Face bf16 deviation). **It failed on 2026-11-05** (bf16 all-reduce: 0.5019,
/// ratio 1.95; f32 all-reduce: 0.3818, ratio 1.49) while TP=2 itself is within the oracle tolerance and closer to float32 than TP=1: the bound compares two
/// implementations that each deviate from float32, so by the triangle inequality it can only hold with twice the deviation. The assertion is kept as declared;
/// the test stays red until the owner decides the bound.
#[test]
#[ignore = "needs two H100s on one node, libnccl.so.2, a compiled package and the compact oracle; red by the declared bound, see the doc comment"]
fn tp2_matches_tp1_within_the_declared_bound() {
    let oracle = oracle();
    let (rows, n, pages) = scenario(&oracle);
    let (one, two, _group) = both(&oracle, rows, n, pages);
    let bf16_max = oracle.prompts.iter().flat_map(|p| p.bf16_deviation.iter().flatten().map(|d| d.0)).fold(0f32, f32::max);
    let mut worst = (0f32, String::new());
    for (r, (a, b)) in one.iter().zip(&two).enumerate() {
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let d = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
            if d > worst.0 {
                worst = (d, format!("row {r}, step {i}"));
            }
        }
    }
    println!("TP=2 vs TP=1: max |logit difference| {:.4} at {}; declared bound (the HF bf16 deviation) {bf16_max:.4}; ratio {:.2}", worst.0, worst.1, worst.0 / bf16_max);
    assert!(worst.0 <= bf16_max, "TP=2 differs from TP=1 by {} at {} (bound {bf16_max})", worst.0, worst.1);
}
