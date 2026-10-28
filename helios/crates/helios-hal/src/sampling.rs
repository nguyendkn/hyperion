//! Choosing the next token from the logits: greedy, or temperature plus nucleus (top-p)
//! sampling. Pure host code with its own tiny deterministic random generator (no dependency),
//! so a request with a given seed always produces the same text and the behaviour is
//! unit-tested without any GPU.

/// How the next token is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    /// `None` or `<= 0`: greedy (always the highest logit). Otherwise logits are divided by it.
    pub temperature: Option<f32>,
    /// Keep the smallest set of most likely tokens whose probability sums to at least this
    /// (`None` or `>= 1`: keep all). Applied after the temperature.
    pub top_p: Option<f32>,
    /// Seed of the random generator; the same seed gives the same sequence.
    pub seed: u64,
}

impl SamplingConfig {
    pub const fn greedy() -> Self {
        Self { temperature: None, top_p: None, seed: 0 }
    }

    fn is_greedy(&self) -> bool {
        self.temperature.is_none_or(|t| t <= 0.0)
    }
}

/// xorshift64*: small, fast and good enough for sampling; not for anything security related.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // A zero state would stay zero; mix the seed so nearby seeds give unrelated streams.
        let mixed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        Self(if mixed == 0 { 0x9E37_79B9_7F4A_7C15 } else { mixed })
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11; // 53 random bits
        bits as f64 / (1u64 << 53) as f64
    }
}

/// Index of the largest value; the first one wins ties.
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// Pick a token id from `logits`.
pub fn sample(logits: &[f32], config: &SamplingConfig, rng: &mut Rng) -> u32 {
    assert!(!logits.is_empty(), "no logits to sample from");
    if config.is_greedy() {
        return argmax(logits) as u32;
    }
    let temperature = config.temperature.unwrap_or(1.0);

    // Softmax of logits / temperature, in f64 and shifted by the maximum for stability.
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut candidates: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, f64::from((l - max) / temperature).exp()))
        .collect();
    let total: f64 = candidates.iter().map(|c| c.1).sum();
    for c in &mut candidates {
        c.1 /= total;
    }

    // Nucleus: most likely first (ties by index so the order is deterministic), keep the
    // shortest prefix reaching top_p.
    if let Some(top_p) = config.top_p.filter(|p| *p < 1.0) {
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let mut cumulative = 0.0;
        let mut keep = candidates.len();
        for (n, c) in candidates.iter().enumerate() {
            cumulative += c.1;
            if cumulative >= f64::from(top_p.max(f32::MIN_POSITIVE)) {
                keep = n + 1;
                break;
            }
        }
        candidates.truncate(keep);
        let kept: f64 = candidates.iter().map(|c| c.1).sum();
        for c in &mut candidates {
            c.1 /= kept;
        }
    }

    let draw = rng.next_f64();
    let mut cumulative = 0.0;
    for (index, probability) in &candidates {
        cumulative += probability;
        if draw < cumulative {
            return *index as u32;
        }
    }
    // Rounding can leave the cumulative sum a hair under 1.
    candidates.last().map_or(0, |c| c.0 as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(temperature: f32, top_p: Option<f32>, seed: u64) -> SamplingConfig {
        SamplingConfig { temperature: Some(temperature), top_p, seed }
    }

    #[test]
    fn greedy_is_the_argmax_and_ignores_the_generator() {
        let logits = [0.1, 2.0, -1.0, 2.0];
        let mut rng = Rng::new(1);
        assert_eq!(sample(&logits, &SamplingConfig::greedy(), &mut rng), 1, "first maximum wins");
        assert_eq!(sample(&logits, &config(0.0, None, 1), &mut rng), 1, "temperature 0 is greedy");
        assert_eq!(sample(&logits, &config(-1.0, None, 1), &mut rng), 1, "negative temperature is greedy");
    }

    #[test]
    fn the_same_seed_gives_the_same_sequence_and_another_seed_does_not() {
        let logits: Vec<f32> = (0..50).map(|i| (i % 7) as f32 * 0.3).collect();
        let run = |seed| {
            let mut rng = Rng::new(seed);
            (0..40).map(|_| sample(&logits, &config(1.0, None, seed), &mut rng)).collect::<Vec<_>>()
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }

    #[test]
    fn sampling_frequencies_follow_the_softmax() {
        // logits [0, ln 3] -> probabilities 0.25 / 0.75 at temperature 1.
        let logits = [0.0, 3f32.ln()];
        let mut rng = Rng::new(42);
        let n = 20_000;
        let ones = (0..n).filter(|_| sample(&logits, &config(1.0, None, 42), &mut rng) == 1).count();
        let share = ones as f64 / n as f64;
        assert!((share - 0.75).abs() < 0.02, "got {share}");
    }

    #[test]
    fn a_low_temperature_concentrates_on_the_best_token() {
        let logits = [1.0, 1.5, 0.5];
        let mut rng = Rng::new(3);
        let hits = (0..2000).filter(|_| sample(&logits, &config(0.05, None, 3), &mut rng) == 1).count();
        assert!(hits > 1990, "only {hits} of 2000 picked the best token");
    }

    #[test]
    fn a_high_temperature_spreads_over_all_tokens() {
        let logits = [1.0, 1.5, 0.5];
        let mut rng = Rng::new(3);
        let mut seen = [false; 3];
        for _ in 0..2000 {
            seen[sample(&logits, &config(10.0, None, 3), &mut rng) as usize] = true;
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn top_p_keeps_only_the_nucleus() {
        // probabilities (temperature 1): ~[0.62, 0.23, 0.14, 0.01]; top_p 0.8 keeps the first two.
        let logits = [2.0f32, 1.0, 0.5, -2.5];
        let mut rng = Rng::new(9);
        for _ in 0..3000 {
            let t = sample(&logits, &config(1.0, Some(0.8), 9), &mut rng);
            assert!(t == 0 || t == 1, "token {t} is outside the nucleus");
        }
    }

    #[test]
    fn a_tiny_top_p_is_the_most_likely_token() {
        let logits = [0.0, 0.2, 0.1];
        let mut rng = Rng::new(5);
        for _ in 0..200 {
            assert_eq!(sample(&logits, &config(1.0, Some(1e-6), 5), &mut rng), 1);
        }
    }

    #[test]
    fn top_p_one_or_more_keeps_everything() {
        let logits = [0.0, 0.0, 0.0];
        let mut rng = Rng::new(11);
        let mut seen = [false; 3];
        for _ in 0..500 {
            seen[sample(&logits, &config(1.0, Some(1.0), 11), &mut rng) as usize] = true;
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn extreme_logits_do_not_overflow() {
        let logits = [1.0e4, 0.0, -1.0e4];
        let mut rng = Rng::new(1);
        assert_eq!(sample(&logits, &config(1.0, None, 1), &mut rng), 0);
    }

    #[test]
    fn the_generator_is_uniform_in_the_unit_interval() {
        let mut rng = Rng::new(123);
        let draws: Vec<f64> = (0..20_000).map(|_| rng.next_f64()).collect();
        assert!(draws.iter().all(|d| (0.0..1.0).contains(d)));
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }
}
