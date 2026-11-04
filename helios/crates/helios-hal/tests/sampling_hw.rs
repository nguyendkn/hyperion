//! Hardware tests (H100) of the device sampler (sampler v2, exact for every top_k and top_p, `sampler_reference.rs`), driven through `submit_step` the way a scheduler would.
//!
//! * **Exact match** (declared before the first run: any differing token fails): the logits a step produced are captured by repeating the step
//!   teacher-forced with the logits asked for, and the token the device drew must equal the host reference's on those logits with the same seed and
//!   draw counter, for every row of every bucket (1 to 32), in all sampler modes (full vocabulary, top-k, top-p, both, with and without a penalty).
//! * **Distribution** (significance level 0.001 per row, declared before the first run): one step is repeated with the draw counter advancing, so the
//!   logits are fixed and the draws differ; the counts of a row must fit the distribution the sampler defines (chi-square, bins with an expected
//!   count of at least 5, the rest pooled).
//! * **Overhead**: the GPU time of a decode step at each bucket with every slot greedy and with every slot sampling; budget (owner, 2026-11-04, in absolute
//!   time) at most 150 microseconds per step at every bucket for temperature 0.8, top_p 0.95, top_k 50; the other paths are reported.
//!
//! ```text
//! HYPERION_WEIGHTS_PACKAGE=/path/model.hyperion HYPERION_COMPACT_ORACLE=/path/<id>.compact.json \
//!   cargo test --release -p helios-hal --features sm90 --test sampling_hw -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "sm90")]

use std::path::PathBuf;

use helios_hal::sampler_reference::{self, Params};
use helios_hal::step::{Capacity, Input, PageDelta, Row, SamplingInit, SlotInit, StepPlan, Want};
use helios_hal::{ComputeBackend, HopperBackend, KvPool};
use hyperion_specification::Manifest;

const PAGE: usize = 64;
const PAGES_PER_SEQ: usize = 4;

fn package() -> PathBuf {
    PathBuf::from(std::env::var("HYPERION_WEIGHTS_PACKAGE").expect("set HYPERION_WEIGHTS_PACKAGE to the .hyperion path"))
}

fn seed_prompt() -> Vec<u32> {
    let path = std::env::var_os("HYPERION_COMPACT_ORACLE").expect("set HYPERION_COMPACT_ORACLE");
    helios_hal::oracle::CompactOracle::from_json(&std::fs::read(path).unwrap()).expect("compact oracle").prompts[0].ids.clone()
}

fn open(max_batch: usize, pages: usize) -> (HopperBackend, KvPool) {
    println!("BUILD revision={}", std::env::var("HYPERION_REVISION").unwrap_or_else(|_| "unknown".into()));
    let package = package();
    let Manifest::V1(manifest) = Manifest::from_json_bytes(&std::fs::read(package.with_extension("manifest.json")).unwrap()).unwrap();
    let backend = HopperBackend::new().expect("an H100 must be present");
    helios_hal::load_package(&backend, &package, &manifest).expect("upload");
    // The pool also holds the private area of `generate`: ask for a third more than the pages the scenario addresses.
    let pool = backend.open_kv(max_batch, (pages * 4 / 3 + 1) * PAGE).expect("the KV pool");
    assert!(pool.info().pages >= pages);
    (backend, pool)
}

/// The sampling of sequence `i`: ten modes in turn, a repetition penalty on the odd ones, a seed of its own. Modes 0-3 are the bounded path (the whole
/// vocabulary, top-k, top-p); modes 4-6 and 8 force the full per-row sort (a nucleus over a flat distribution wider than 1024 tokens, a top_k above 1024
/// that is not the vocabulary); mode 7 sits on the boundary (top_k 1024); modes 8 and 9 use a top_k larger than the vocabulary.
const MODES: usize = 10;
fn sampling_of(i: usize, penalties: bool) -> SamplingInit {
    let (temperature, top_p, top_k) =
        [(1.0, 1.0, 0), (0.8, 0.95, 50), (0.7, 1.0, 40), (1.3, 0.9, 0), (60.0, 0.9, 0), (1.0, 1.0, 2000), (1.0, 0.95, 1025), (200.0, 0.5, 1024), (60.0, 0.95, u32::MAX), (1.0, 1.0, u32::MAX)][i % MODES];
    SamplingInit { temperature, top_p, top_k, repetition_penalty: if penalties && i % 2 == 1 { 1.1 } else { 1.0 }, seed: 1000 + i as u64 }
}

fn params_of(s: &SamplingInit) -> Params {
    Params { temperature: s.temperature, top_p: s.top_p, top_k: s.top_k, repetition_penalty: s.repetition_penalty }
}

struct Seq {
    slot: u32,
    prompt: Vec<u32>,
    pages: Vec<u32>,
    sampling: SamplingInit,
}

fn sequences(count: usize, prompt_tokens: impl Fn(usize) -> usize, penalties: bool) -> Vec<Seq> {
    let seed = seed_prompt();
    (0..count)
        .map(|i| Seq {
            slot: i as u32,
            prompt: (0..prompt_tokens(i)).map(|j| seed[(j * (i + 1) + i * 5) % seed.len()]).collect(),
            pages: ((PAGES_PER_SEQ * i) as u32..(PAGES_PER_SEQ * (i + 1)) as u32).collect(),
            sampling: sampling_of(i, penalties),
        })
        .collect()
}

struct Driver<'a> {
    backend: &'a HopperBackend,
    pool: &'a KvPool,
    step: u64,
}

impl Driver<'_> {
    fn take(&mut self, capacity: Capacity, rows: Vec<Row>, deltas: Vec<PageDelta>, inits: Vec<SlotInit>) -> helios_hal::StepOutput {
        self.step += 1;
        let plan = StepPlan { step: self.step, capacity, rows, deltas, inits };
        let ticket = self.backend.submit_step(self.pool, &plan).expect("submit_step");
        self.backend.take_output(ticket).expect("take_output")
    }

    /// Prefill `s` (one chunk) with a fresh sampling state; `want` is what the row asks for.
    fn prefill(&mut self, s: &Seq, want: Want) -> helios_hal::RowOutput {
        let deltas = s.pages.iter().enumerate().map(|(i, &page)| PageDelta { slot: s.slot, index: i as u32, page }).collect();
        let capacity = Capacity::Prefill { rows: s.prompt.len().next_power_of_two().max(128) as u32 };
        let row = Row { slot: s.slot, input: Input::Tokens(s.prompt.clone()), n_tokens: s.prompt.len() as u32, position: 0, table_len: s.prompt.len().div_ceil(PAGE) as u32, want };
        let inits = vec![SlotInit { slot: s.slot, sampling: s.sampling, max_new_tokens: 64 }];
        self.take(capacity, vec![row], deltas, inits).rows.remove(0)
    }

    /// One decode step of `bucket` rows: `(sequence, position, input token or None for the token on the device)`.
    fn decode(&mut self, seqs: &[Seq], bucket: u16, positions: &[usize], inputs: Option<&[u32]>, want: Want) -> Vec<helios_hal::RowOutput> {
        let rows = seqs
            .iter()
            .enumerate()
            .map(|(i, s)| Row {
                slot: s.slot,
                input: inputs.map_or(Input::FromDevice, |t| Input::Tokens(vec![t[i]])),
                n_tokens: 1,
                position: positions[i] as u32,
                table_len: (positions[i] + 1).div_ceil(PAGE) as u32,
                want,
            })
            .collect();
        self.take(Capacity::Decode { bucket }, rows, Vec::new(), Vec::new()).rows
    }
}

#[test]
#[ignore = "needs an H100, a compiled package and the compact oracle (see module docs)"]
fn the_device_draws_exactly_the_tokens_of_the_host_reference_in_every_bucket() {
    let steps = 12;
    let (backend, pool) = open(32, 32 * PAGES_PER_SEQ);
    let mut matched = 0usize;
    // `HYPERION_SAMPLING_BUCKETS=8` narrows the run (a sanitizer run, which is slow).
    let buckets: Vec<usize> = std::env::var("HYPERION_SAMPLING_BUCKETS").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32]);
    for bucket in buckets {
        let seqs = sequences(bucket, |i| 24 + 5 * i, true);
        let mut d = Driver { backend: &backend, pool: &pool, step: 0 };
        // Pass 1: the device samples and chains its own tokens.
        let mut drawn: Vec<Vec<u32>> = vec![Vec::new(); bucket];
        for (i, s) in seqs.iter().enumerate() {
            drawn[i].push(d.prefill(s, Want::Token).tokens[0]);
        }
        for k in 0..steps {
            let positions: Vec<usize> = seqs.iter().map(|s| s.prompt.len() + k).collect();
            for (i, out) in d.decode(&seqs, bucket as u16, &positions, None, Want::Token).into_iter().enumerate() {
                drawn[i].push(out.tokens[0]);
            }
        }
        // Pass 2: the same tokens fed by the host with the logits asked for.
        let mut logits: Vec<Vec<Vec<f32>>> = vec![Vec::new(); bucket];
        for (i, s) in seqs.iter().enumerate() {
            logits[i].push(d.prefill(s, Want::Logits).logits.expect("logits"));
        }
        for k in 0..steps {
            let positions: Vec<usize> = seqs.iter().map(|s| s.prompt.len() + k).collect();
            let inputs: Vec<u32> = drawn.iter().map(|t| t[k]).collect();
            for (i, out) in d.decode(&seqs, bucket as u16, &positions, Some(&inputs), Want::Logits).into_iter().enumerate() {
                logits[i].push(out.logits.expect("logits"));
            }
        }
        for (i, s) in seqs.iter().enumerate() {
            let vocab = logits[i][0].len();
            for k in 0..=steps {
                let mut seen = vec![false; vocab];
                for &t in &drawn[i][..k] {
                    seen[t as usize] = true;
                }
                let want = sampler_reference::sample(&logits[i][k], &seen, &params_of(&s.sampling), s.sampling.seed, k as u32).expect("the reference samples");
                assert_eq!(drawn[i][k], want, "bucket {bucket}, row {i} (sampling {:?}), draw {k}: the device drew {} and the host reference {want}", s.sampling, drawn[i][k]);
                matched += 1;
            }
        }
        println!("bucket {bucket:>2}: {} rows x {} draws equal the host reference", bucket, steps + 1);
    }
    println!("EXACT MATCH: {matched} draws, 0 differences");
    backend.shutdown().unwrap();
}

/// The probabilities sampler v2 assigns, in f64.
fn expected(logits: &[f32], s: &SamplingInit) -> Vec<(u32, f64)> {
    let z: Vec<f64> = logits.iter().map(|&v| f64::from(v) / f64::from(s.temperature)).collect();
    let mut order: Vec<usize> = (0..z.len()).collect();
    let mut keep = z.len();
    if s.top_k != 0 || s.top_p < 1.0 {
        order.sort_by(|&a, &b| z[b].partial_cmp(&z[a]).unwrap().then(a.cmp(&b)));
        keep = if s.top_k == 0 { z.len() } else { (s.top_k as usize).min(z.len()) };
        order.truncate(keep);
    }
    let max = order.iter().map(|&i| z[i]).fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = order.iter().map(|&i| (z[i] - max).exp()).collect();
    let total: f64 = w.iter().sum();
    if s.top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut n = keep;
        for (j, x) in w.iter().enumerate() {
            cumulative += x / total;
            if cumulative >= f64::from(s.top_p) {
                n = j + 1;
                break;
            }
        }
        let total: f64 = w[..n].iter().sum();
        return order[..n].iter().zip(&w[..n]).map(|(&i, x)| (i as u32, x / total)).collect();
    }
    order.iter().zip(&w).map(|(&i, x)| (i as u32, x / total)).collect()
}

/// Critical value of chi-square with `df` degrees of freedom at significance 0.001 (Wilson-Hilferty).
fn chi_square_critical(df: f64) -> f64 {
    let z = 3.0902;
    df * (1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt()).powi(3)
}

#[test]
#[ignore = "needs an H100, a compiled package and the compact oracle (see module docs)"]
fn the_draws_of_a_row_follow_the_distribution_the_sampler_defines() {
    let draws = 4000usize;
    let rows = 8usize;
    let (backend, pool) = open(8, 8 * PAGES_PER_SEQ);
    // Rows 4-7 are modes 4, 5, 6, 9 of `sampling_of` replaced by moderate temperatures: a flat row cannot fill bins of expected count 5 in 4000 draws, the
    // exact-match test covers the flat modes; here the wide set (top_k 2000, top_k 1025 with a nucleus, a wide nucleus) is tested with distributions that
    // still have bins.
    let mut seqs = sequences(rows, |i| 24 + 5 * i, false);
    for (i, (temperature, top_p, top_k)) in [(1.0f32, 0.9999f32, 0u32), (1.6, 1.0, 2000), (1.2, 0.97, 1025), (1.0, 1.0, u32::MAX)].into_iter().enumerate() {
        seqs[4 + i].sampling = SamplingInit { temperature, top_p, top_k, repetition_penalty: 1.0, seed: 2000 + i as u64 };
    }
    let mut d = Driver { backend: &backend, pool: &pool, step: 0 };
    for s in &seqs {
        d.prefill(s, Want::Nothing);
    }
    let positions: Vec<usize> = seqs.iter().map(|s| s.prompt.len()).collect();
    let inputs: Vec<u32> = (0..rows as u32).map(|i| 100 + i).collect();
    // The same step again and again: the same logits, a new draw each time (the counter advances).
    let logits: Vec<Vec<f32>> = d.decode(&seqs, rows as u16, &positions, Some(&inputs), Want::Logits).into_iter().map(|o| o.logits.expect("logits")).collect();
    let mut counts: Vec<std::collections::BTreeMap<u32, u32>> = vec![Default::default(); rows];
    for _ in 0..draws {
        for (i, out) in d.decode(&seqs, rows as u16, &positions, Some(&inputs), Want::Token).into_iter().enumerate() {
            *counts[i].entry(out.tokens[0]).or_default() += 1;
        }
    }
    for (i, s) in seqs.iter().enumerate() {
        let probs = expected(&logits[i], &s.sampling);
        let support: std::collections::BTreeSet<u32> = probs.iter().map(|p| p.0).collect();
        assert!(counts[i].keys().all(|t| support.contains(t)), "row {i}: a token outside the sampler's support was drawn");
        let (mut statistic, mut bins, mut pooled_expected, mut pooled_seen) = (0.0f64, 0usize, 0.0f64, 0.0f64);
        for (token, p) in &probs {
            let e = p * draws as f64;
            let o = f64::from(counts[i].get(token).copied().unwrap_or(0));
            if e >= 5.0 {
                statistic += (o - e) * (o - e) / e;
                bins += 1;
            } else {
                pooled_expected += e;
                pooled_seen += o;
            }
        }
        if pooled_expected >= 5.0 {
            statistic += (pooled_seen - pooled_expected).powi(2) / pooled_expected;
            bins += 1;
        }
        assert!(bins >= 2, "row {i}: the distribution is too concentrated to test ({} candidates)", probs.len());
        let critical = chi_square_critical((bins - 1) as f64);
        println!("row {i} ({:?}): chi-square {statistic:.1} with {} degrees of freedom, critical (0.001) {critical:.1}", s.sampling, bins - 1);
        assert!(statistic < critical, "row {i}: the draws do not follow the distribution (chi-square {statistic:.1} >= {critical:.1})");
    }
    backend.shutdown().unwrap();
}

fn median_p10_p90(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[v.len() / 10], v[v.len() * 9 / 10])
}

#[test]
#[ignore = "needs an H100, a compiled package and the compact oracle (see module docs)"]
fn sampling_costs_at_most_the_declared_share_of_a_decode_step() {
    let (backend, pool) = open(32, 32 * 16);
    let seed = seed_prompt();
    let prompts: Vec<Vec<u32>> = (0..32usize).map(|i| (0..512).map(|j| seed[(j + i) % seed.len()]).collect()).collect();
    let median_of = |sampling: SamplingInit, bucket: usize| -> f64 {
        let mut d = Driver { backend: &backend, pool: &pool, step: 0 };
        for i in 0..32usize {
            let s = Seq { slot: i as u32, prompt: prompts[i].clone(), pages: ((16 * i) as u32..(16 * i + 16) as u32).collect(), sampling: SamplingInit { seed: 7 + i as u64, ..sampling } };
            d.prefill(&s, Want::Nothing);
        }
        let rows: Vec<(u32, u32)> = (0..bucket as u32).map(|slot| (slot, 513)).collect();
        median_p10_p90(backend.time_decode_bucket(&rows, bucket, 128, 10).expect("timing")).0
    };
    let sampled = SamplingInit { temperature: 0.8, top_p: 0.95, top_k: 50, repetition_penalty: 1.0, seed: 0 };
    let nucleus_only = SamplingInit { temperature: 0.8, top_p: 0.95, top_k: 0, repetition_penalty: 1.0, seed: 0 };
    let full = SamplingInit { temperature: 1.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0, seed: 0 };
    let wide = SamplingInit { temperature: 60.0, top_p: 0.9, top_k: 0, repetition_penalty: 1.0, seed: 0 };
    let mut misses = Vec::new();
    println!("context 512; GPU ms per step, median of 10 windows of 128 steps; extra cost against greedy in microseconds");
    println!("budget (owner, 2026-11-04): the bounded path (temperature 0.8, top_p 0.95, top_k 50) costs at most 150 us per step at every bucket; the nucleus-only and full-vocabulary rows and the wide path (full sort) are reported without a budget");
    for bucket in [1usize, 2, 4, 8, 16, 32] {
        let greedy = median_of(SamplingInit::GREEDY, bucket);
        let us = |sampling: SamplingInit| (median_of(sampling, bucket) - greedy) * 1000.0;
        let (bounded, nucleus, vocabulary, wide_us) = (us(sampled), us(nucleus_only), us(full), us(wide));
        println!("BUCKET {bucket:>2}: greedy {greedy:.3} ms  extra us: top_k 50 + top_p 0.95 {bounded:.0}  top_p 0.95 only {nucleus:.0}  full vocabulary {vocabulary:.0}  wide nucleus (full sort) {wide_us:.0}");
        if bounded > 150.0 {
            misses.push(format!("bucket {bucket}: the bounded path costs {bounded:.0} us against a budget of 150 us"));
        }
    }
    assert!(misses.is_empty(), "{misses:?}");
    backend.shutdown().unwrap();
}
