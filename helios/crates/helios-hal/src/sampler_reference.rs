//! The host reference of the device sampler (H4): the same algorithm, the same Philox stream and the same f32 operations in the same order as the
//! `sample_rows_bf16` kernel, so the device must choose exactly the token this code chooses (tests only; the serving path never calls it).
//!
//! **Sampler v1** (greedy, temperature 0, is the argmax kernel and is not described here). For the logits `z` of one step of one sequence:
//!
//! 1. *Penalty.* A token the sequence has already generated (the prompt does not count: a prefix-cache hit must not change the result) has its
//!    logit divided by `repetition_penalty` if positive, multiplied by it if negative.
//! 2. *Temperature.* `z' = z / temperature` (IEEE division).
//! 3. *Candidates.* With no truncation (`top_k == 0` and `top_p >= 1`) every token is a candidate (mode A). Otherwise the candidates are the first
//!    `K = min(top_k or 1024, 1024, vocab)` tokens ordered by `z'` descending, ties by lower index (mode B). **`top_k` above 1024, and a nucleus
//!    wider than 1024 tokens, are capped at 1024 candidates** (a documented limit of the device sort).
//! 4. *Probabilities.* `w = exp(z' - max z')` with [`det_exp`] (a polynomial built only from IEEE-exact operations and fused multiply-adds, so the
//!    host and the device agree to the bit; relative error below 2e-6).
//! 5. *Top-p* (mode B): with prefix sums `P_j` of `w` over the sorted candidates (sequential f32 additions), keep the first `n` candidates where `n - 1`
//!    is the first `j` with `P_j >= top_p * P_{K-1}` (all `K` when `top_p >= 1`). Top-k is applied before top-p, so the nucleus is taken over the
//!    renormalised top-k.
//! 6. *Draw.* `u = (philox4x32_10(counter = [step, 0, 0, 0], key = [seed_lo, seed_hi])[0] >> 8) * 2^-24`, `step` the number of tokens the sequence
//!    sampled before. Mode B: `target = u * P_{n-1}`, the token is the first kept candidate with `P_j > target` (the last kept one if none). Mode A: the
//!    vocabulary is cut into 1024 chunks of `ceil(vocab / 1024)` consecutive tokens, chunk sums are sequential f32 sums, cumulative sums `C_c` are
//!    sequential over the chunks, `target = u * C_last`, the chunk is the first with `C_c > target` (the first maximal token if none) and the token is
//!    the first one in it whose running sum from `C_{c-1}` exceeds `target` (the chunk's last token if none).
//!
//! Non-finite logits are an error here; the device does not check.

/// Candidates the device sorts for top-k / top-p.
pub const MAX_CANDIDATES: usize = 1024;
/// Chunks of the full-vocabulary draw (one per thread of the kernel block).
pub const CHUNKS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SampleError {
    #[error("logit {index} is not finite")]
    NotFinite { index: usize },
    #[error("temperature must be positive for sampling (0 is greedy)")]
    Temperature,
    #[error("no logits")]
    Empty,
}

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

/// Philox 4x32 with 10 rounds (Salmon et al., SC11).
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (mut c, mut k) = (counter, key);
    for round in 0..10 {
        let p0 = u64::from(PHILOX_M0) * u64::from(c[0]);
        let p1 = u64::from(PHILOX_M1) * u64::from(c[2]);
        c = [((p1 >> 32) as u32) ^ c[1] ^ k[0], p1 as u32, ((p0 >> 32) as u32) ^ c[3] ^ k[1], p0 as u32];
        if round < 9 {
            k = [k[0].wrapping_add(PHILOX_W0), k[1].wrapping_add(PHILOX_W1)];
        }
    }
    c
}

/// The uniform draw in `[0, 1)` of step `step` of a sequence with `seed`.
pub fn uniform(seed: u64, step: u32) -> f32 {
    let word = philox4x32_10([step, 0, 0, 0], [seed as u32, (seed >> 32) as u32])[0];
    (word >> 8) as f32 * (1.0 / 16_777_216.0)
}

const LOG2E: f32 = 1.442_695_04;
const LN2_HI: f32 = 0.693_145_751_953_125;
const LN2_LO: f32 = 1.428_606_77e-6;
const POLY: [f32; 6] = [0.001_388_888_9, 0.008_333_333_8, 0.041_666_668, 0.166_666_67, 0.5, 1.0];

/// `exp(x)` for `x <= 0` from IEEE-exact operations only (see the module docs); 0 below -87.
pub fn det_exp(x: f32) -> f32 {
    if x < -87.0 {
        return 0.0;
    }
    let n = (x * LOG2E).round_ties_even();
    let r = (-n).mul_add(LN2_HI, x);
    let r = (-n).mul_add(LN2_LO, r);
    let mut p = POLY[0];
    for c in &POLY[1..] {
        p = p.mul_add(r, *c);
    }
    p = p.mul_add(r, 1.0);
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// Steps 1 and 2: the logits after the penalty and the temperature. `seen[i]` says whether token `i` was generated before.
pub fn adjust(logits: &[f32], seen: &[bool], params: &Params) -> Result<Vec<f32>, SampleError> {
    if logits.is_empty() {
        return Err(SampleError::Empty);
    }
    if params.temperature.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(SampleError::Temperature);
    }
    logits
        .iter()
        .enumerate()
        .map(|(i, &z)| {
            if !z.is_finite() {
                return Err(SampleError::NotFinite { index: i });
            }
            let z = if seen.get(i).copied().unwrap_or(false) && params.repetition_penalty != 1.0 {
                if z > 0.0 {
                    z / params.repetition_penalty
                } else {
                    z * params.repetition_penalty
                }
            } else {
                z
            };
            Ok(z / params.temperature)
        })
        .collect()
}

/// The token sampler v1 chooses for `logits` with `seed` at `step`.
pub fn sample(logits: &[f32], seen: &[bool], params: &Params, seed: u64, step: u32) -> Result<u32, SampleError> {
    let z = adjust(logits, seen, params)?;
    let u = uniform(seed, step);
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if params.top_k == 0 && params.top_p >= 1.0 {
        return Ok(draw_full(&z, max, u));
    }
    let k = if params.top_k == 0 { MAX_CANDIDATES } else { (params.top_k as usize).min(MAX_CANDIDATES) }.min(z.len());
    let mut order: Vec<usize> = (0..z.len()).collect();
    order.sort_by(|&a, &b| z[b].partial_cmp(&z[a]).expect("finite").then(a.cmp(&b)));
    order.truncate(k);
    let mut prefix = Vec::with_capacity(k);
    let mut run = 0.0f32;
    for &i in &order {
        run += det_exp(z[i] - max);
        prefix.push(run);
    }
    let mut n = k;
    if params.top_p < 1.0 {
        let threshold = params.top_p * prefix[k - 1];
        n = prefix.iter().position(|&p| p >= threshold).map_or(k, |j| j + 1);
    }
    let target = u * prefix[n - 1];
    let j = prefix[..n].iter().position(|&p| p > target).unwrap_or(n - 1);
    Ok(order[j] as u32)
}

fn draw_full(z: &[f32], max: f32, u: f32) -> u32 {
    let chunk = z.len().div_ceil(CHUNKS);
    let w: Vec<f32> = z.iter().map(|&v| det_exp(v - max)).collect();
    let mut cumulative = Vec::with_capacity(CHUNKS);
    let mut run = 0.0f32;
    for c in 0..CHUNKS {
        let (lo, hi) = ((c * chunk).min(z.len()), ((c + 1) * chunk).min(z.len()));
        let mut sum = 0.0f32;
        for &x in &w[lo..hi] {
            sum += x;
        }
        run += sum;
        cumulative.push(run);
    }
    let target = u * run;
    let Some(c) = cumulative.iter().position(|&x| x > target) else {
        return z.iter().position(|&v| v == max).expect("the maximum is in the logits") as u32;
    };
    let (lo, hi) = ((c * chunk).min(z.len()), ((c + 1) * chunk).min(z.len()));
    let mut running = if c == 0 { 0.0 } else { cumulative[c - 1] };
    for i in lo..hi {
        running += w[i];
        if running > target {
            return i as u32;
        }
    }
    (hi - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: Params = Params { temperature: 1.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0 };

    /// Known-answer vectors of Random123 (`kat_vectors`, philox4x32 with 10 rounds).
    #[test]
    fn philox_matches_the_published_test_vectors() {
        assert_eq!(philox4x32_10([0, 0, 0, 0], [0, 0]), [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]);
        assert_eq!(philox4x32_10([u32::MAX; 4], [u32::MAX; 2]), [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]);
        assert_eq!(philox4x32_10([0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344], [0xa409_3822, 0x299f_31d0]), [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]);
    }

    #[test]
    fn uniform_draws_are_in_range_reproducible_and_differ_by_seed_and_step() {
        let draws: Vec<f32> = (0..1000).map(|s| uniform(7, s)).collect();
        assert!(draws.iter().all(|u| (0.0..1.0).contains(u)));
        assert_eq!(draws, (0..1000).map(|s| uniform(7, s)).collect::<Vec<_>>());
        assert_ne!(uniform(7, 0), uniform(8, 0));
        assert_ne!(uniform(7, 0), uniform(7, 1));
        let mean = draws.iter().sum::<f32>() / 1000.0;
        assert!((mean - 0.5).abs() < 0.05, "mean {mean}");
    }

    #[test]
    fn det_exp_is_accurate_and_monotonic_on_the_range_the_sampler_uses() {
        let mut worst = 0.0f64;
        let mut previous = 0.0f32;
        let mut x = -87.0f32;
        while x <= 0.0 {
            let got = det_exp(x);
            let want = f64::from(x).exp();
            worst = worst.max(((f64::from(got) - want) / want).abs());
            assert!(got >= previous, "not monotonic at {x}");
            previous = got;
            x += 0.0137;
        }
        assert!(worst < 2e-6, "worst relative error {worst}");
        assert_eq!(det_exp(0.0), 1.0);
        assert_eq!(det_exp(f32::NEG_INFINITY), 0.0);
        assert_eq!(det_exp(-100.0), 0.0);
    }

    fn logits(n: usize, seed: u64) -> Vec<f32> {
        (0..n).map(|i| (uniform(seed, i as u32) - 0.5) * 12.0).collect()
    }

    #[test]
    fn the_same_seed_gives_the_same_token_and_other_steps_other_draws() {
        let z = logits(5000, 1);
        let params = Params { temperature: 0.8, top_p: 0.95, top_k: 50, repetition_penalty: 1.0 };
        let a: Vec<u32> = (0..64).map(|s| sample(&z, &[], &params, 42, s).unwrap()).collect();
        let b: Vec<u32> = (0..64).map(|s| sample(&z, &[], &params, 42, s).unwrap()).collect();
        assert_eq!(a, b);
        assert!(a.iter().collect::<std::collections::BTreeSet<_>>().len() > 3, "the draws vary with the step");
        let c: Vec<u32> = (0..64).map(|s| sample(&z, &[], &params, 43, s).unwrap()).collect();
        assert_ne!(a, c);
    }

    #[test]
    fn top_k_one_and_a_tiny_top_p_pick_the_maximum_and_ties_go_to_the_lower_index() {
        let mut z = logits(300, 2);
        z[17] = 50.0;
        z[200] = 50.0; // a tie for the maximum
        for s in 0..20 {
            assert_eq!(sample(&z, &[], &Params { top_k: 1, ..P }, 5, s).unwrap(), 17);
            assert_eq!(sample(&z, &[], &Params { top_p: 1e-6, ..P }, 5, s).unwrap(), 17);
            assert_eq!(sample(&z, &[], &Params { top_p: 0.0, ..P }, 5, s).unwrap(), 17);
        }
    }

    #[test]
    fn all_equal_logits_are_uniform_over_the_vocabulary_and_over_the_first_candidates() {
        let z = vec![1.0f32; 4096];
        let mut hit = std::collections::BTreeSet::new();
        for s in 0..400 {
            hit.insert(sample(&z, &[], &P, 9, s).unwrap());
        }
        assert!(hit.len() > 300 && hit.iter().all(|&t| t < 4096), "mode A spreads over the vocabulary: {}", hit.len());
        // With a nucleus below 1 the candidates are the first 1024 by index (ties), and the draw stays among them.
        for s in 0..200 {
            assert!(sample(&z, &[], &Params { top_p: 0.5, ..P }, 9, s).unwrap() < 1024);
        }
        for s in 0..200 {
            assert!(sample(&z, &[], &Params { top_k: 5, ..P }, 9, s).unwrap() < 5);
        }
    }

    #[test]
    fn top_k_never_leaves_the_k_best_and_a_wide_k_is_capped_at_1024() {
        let z = logits(3000, 3);
        let mut order: Vec<usize> = (0..z.len()).collect();
        order.sort_by(|&a, &b| z[b].partial_cmp(&z[a]).unwrap().then(a.cmp(&b)));
        for s in 0..300 {
            let t = sample(&z, &[], &Params { temperature: 1.5, top_k: 7, ..P }, 11, s).unwrap() as usize;
            assert!(order[..7].contains(&t), "step {s}: {t}");
            let t = sample(&z, &[], &Params { temperature: 3.0, top_k: 5000, ..P }, 11, s).unwrap() as usize;
            assert!(order[..1024].contains(&t));
        }
    }

    #[test]
    fn sampled_frequencies_follow_the_softmax() {
        let z = vec![2.0f32, 1.0, 0.0, -1.0];
        let n = 20_000u32;
        let mut counts = [0u32; 4];
        for s in 0..n {
            counts[sample(&z, &[], &P, 77, s).unwrap() as usize] += 1;
        }
        let total: f64 = z.iter().map(|&v| f64::from(v).exp()).sum();
        for (i, &c) in counts.iter().enumerate() {
            let want = f64::from(z[i]).exp() / total;
            let got = f64::from(c) / f64::from(n);
            assert!((got - want).abs() < 0.015, "token {i}: {got:.4} vs {want:.4}");
        }
    }

    #[test]
    fn the_repetition_penalty_moves_a_seen_token_down_whatever_its_sign() {
        let z = vec![3.0f32, 2.9, -1.0];
        let seen = [true, false, false];
        let adjusted = adjust(&z, &seen, &Params { repetition_penalty: 2.0, ..P }).unwrap();
        assert_eq!(adjusted, vec![1.5, 2.9, -1.0]);
        let seen = [false, false, true];
        assert_eq!(adjust(&z, &seen, &Params { repetition_penalty: 2.0, ..P }).unwrap(), vec![3.0, 2.9, -2.0]);
        // With a strong penalty the seen leader loses a near-greedy choice.
        let near_greedy = Params { temperature: 0.005, repetition_penalty: 2.0, ..P };
        assert_eq!(sample(&z, &[true, false, false], &near_greedy, 1, 0).unwrap(), 1);
        assert_eq!(sample(&z, &[], &near_greedy, 1, 0).unwrap(), 0);
    }

    #[test]
    fn nan_inf_and_a_zero_temperature_are_refused() {
        assert_eq!(sample(&[1.0, f32::NAN], &[], &P, 1, 0), Err(SampleError::NotFinite { index: 1 }));
        assert_eq!(sample(&[f32::INFINITY, 1.0], &[], &P, 1, 0), Err(SampleError::NotFinite { index: 0 }));
        assert_eq!(sample(&[1.0, 2.0], &[], &Params { temperature: 0.0, ..P }, 1, 0), Err(SampleError::Temperature));
        assert_eq!(sample(&[], &[], &P, 1, 0), Err(SampleError::Empty));
    }
}
