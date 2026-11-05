//! Performance of tensor parallelism (gate K3, T5): GPU milliseconds per decode step at TP=1 and TP=2 for buckets 1, 8, 32 and contexts 5, 4096, 16000
//! (10 windows of 128 chained graph launches after 3 warm-up steps, median with p10/p90, measured with CUDA events inside one run), and the wall time to the
//! first token of a prompt of 1024, 8192 and 16384 tokens (median of 3). A 7B model fits one H100, so TP=2 here measures the cost of communication and the
//! correctness of the split, not a speed-up the model needs.
//!
//! ```text
//! HYPERION_WEIGHTS_PACKAGE=... HYPERION_COMPACT_ORACLE=... cargo test --release -p helios-hal --features sm90 --test tp_perf_hw -- --ignored --nocapture
//! ```
#![cfg(feature = "sm90")]

use std::path::PathBuf;
use std::time::Instant;

use helios_hal::rank_group::ExpectedVersions;
use helios_hal::step::{Capacity, Input, PageDelta, Row, SamplingInit, SlotInit, StepPlan, Want};
use helios_hal::{ComputeBackend, DecodeMode, HopperBackend, TpBackend};
use hyperion_specification::Manifest;

const PAGE: usize = 64;
const BUCKETS: [usize; 3] = [1, 8, 32];
const CONTEXTS: [usize; 3] = [5, 4096, 16000];
const PROMPTS: [usize; 3] = [1024, 8192, 16384];

fn package() -> PathBuf {
    PathBuf::from(std::env::var("HYPERION_WEIGHTS_PACKAGE").expect("set HYPERION_WEIGHTS_PACKAGE to the .hyperion path"))
}

fn manifest() -> hyperion_specification::v1::Manifest {
    let Manifest::V1(m) = Manifest::from_json_bytes(&std::fs::read(package().with_extension("manifest.json")).unwrap()).unwrap();
    m
}

fn seed_tokens() -> Vec<u32> {
    let path = std::env::var_os("HYPERION_COMPACT_ORACLE").expect("set HYPERION_COMPACT_ORACLE");
    helios_hal::oracle::CompactOracle::from_json(&std::fs::read(path).unwrap()).expect("compact oracle").prompts[0].ids.clone()
}

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[v.len() / 10], v[v.len() * 9 / 10])
}

/// A backend, one way or the other: `submit` runs a plan, `time` measures a decode bucket.
trait Runner {
    fn submit(&self, plan: &StepPlan);
    fn time(&self, rows: &[(u32, u32)], bucket: usize) -> Vec<f64>;
}

struct One {
    backend: HopperBackend,
    pool: helios_hal::KvPool,
}

impl Runner for One {
    fn submit(&self, plan: &StepPlan) {
        self.backend.take_output(self.backend.submit_step_mode(&self.pool, plan, DecodeMode::Graph).expect("submit")).expect("take");
    }
    fn time(&self, rows: &[(u32, u32)], bucket: usize) -> Vec<f64> {
        self.backend.time_decode_bucket(rows, bucket, 128, 10).expect("timing")
    }
}

struct Two(TpBackend);

impl Runner for Two {
    fn submit(&self, plan: &StepPlan) {
        self.0.submit_step(plan, DecodeMode::Graph).expect("tp2 step");
    }
    fn time(&self, rows: &[(u32, u32)], bucket: usize) -> Vec<f64> {
        let rows = rows.to_vec();
        let answers = self.0.run_all(move |rank| rank.backend.time_decode_bucket(&rows, bucket, 128, 10));
        answers.into_iter().next().unwrap().expect("timing")
    }
}

fn measure(name: &str, runner: &dyn Runner) {
    let tokens = seed_tokens();
    println!("=== {name}");
    // Decode: map the pages of 32 slots to the longest context and time each (context, bucket).
    let slot_pages = (CONTEXTS[2] + 256).div_ceil(PAGE);
    let deltas: Vec<PageDelta> = (0..32u32).flat_map(|s| (0..slot_pages as u32).map(move |i| PageDelta { slot: s, index: i, page: s * slot_pages as u32 + i })).collect();
    // A step holds at most one table's worth of page-table deltas (plus a little): map the pages in batches.
    for (i, batch) in deltas.chunks(256).enumerate() {
        runner.submit(&StepPlan { step: 1 + i as u64, capacity: Capacity::Decode { bucket: 1 }, rows: Vec::new(), deltas: batch.to_vec(), inits: Vec::new() });
    }
    for context in CONTEXTS {
        for bucket in BUCKETS {
            let rows: Vec<(u32, u32)> = (0..bucket as u32).map(|slot| (slot, context as u32)).collect();
            let (median, p10, p90) = stats(runner.time(&rows, bucket));
            println!("DECODE {name} context {context:>5} bucket {bucket:>2}: {median:.3} ms/step [p10 {p10:.3}, p90 {p90:.3}]");
        }
    }
    // First token: a prompt of n tokens (distinct tokens, prefilled in chunks of 4096 through the plan path), wall time to the last chunk's token.
    for n in PROMPTS {
        let prompt: Vec<u32> = (0..n).map(|j| tokens[(j * 7 + 3) % tokens.len()]).collect();
        let pages: Vec<u32> = (0..n.div_ceil(PAGE) as u32).collect();
        let mut walls = Vec::new();
        for attempt in 0..4 {
            let started = Instant::now();
            let mut start = 0;
            let mut step = 10_000 + attempt * 100;
            while start < n {
                let end = (start + 4096).min(n);
                let last = end == n;
                step += 1;
                runner.submit(&StepPlan {
                    step: step as u64,
                    capacity: Capacity::Prefill { rows: (end - start).next_power_of_two().max(128) as u32 },
                    rows: vec![Row { slot: 0, input: Input::Tokens(prompt[start..end].to_vec()), n_tokens: (end - start) as u32, position: start as u32, table_len: end.div_ceil(PAGE) as u32, want: if last { Want::Token } else { Want::Nothing } }],
                    deltas: if start == 0 { pages.iter().enumerate().map(|(i, &page)| PageDelta { slot: 0, index: i as u32, page }).collect() } else { Vec::new() },
                    inits: if start == 0 { vec![SlotInit { slot: 0, sampling: SamplingInit::GREEDY, max_new_tokens: 1 }] } else { Vec::new() },
                });
                start = end;
            }
            if attempt > 0 {
                walls.push(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let (median, p10, p90) = stats(walls);
        println!("TTFT {name} prompt {n:>5}: {median:.1} ms [p10 {p10:.1}, p90 {p90:.1}]");
    }
}

#[test]
#[ignore = "needs two H100s on one node, libnccl.so.2, a compiled package and the compact oracle"]
fn decode_and_first_token_at_tp1_and_tp2() {
    println!("BUILD revision={}", std::env::var("HYPERION_REVISION").unwrap_or_else(|_| "unknown".into()));
    let manifest = manifest();
    let max_tokens = 32 * (CONTEXTS[2] + 256) + 4096;
    {
        let backend = HopperBackend::new().expect("a GPU");
        helios_hal::load_package(&backend, &package(), &manifest).expect("TP=1 load");
        let pool = backend.open_kv_serving(32, max_tokens).expect("TP=1 pool");
        println!("TP=1 pool: {} pages", pool.info().pages);
        let one = One { backend, pool };
        measure("TP=1", &one);
        let One { backend, pool } = one;
        drop(pool);
        backend.shutdown().unwrap();
    }
    let group = TpBackend::start(&[0, 1], &ExpectedVersions::default()).expect("a TP=2 group");
    group.load_package(&package(), &manifest).expect("sharded load");
    let pages = group.open_kv(32, max_tokens).expect("pools");
    println!("TP=2 pool: {pages:?} pages per rank");
    measure("TP=2", &Two(group));
}
