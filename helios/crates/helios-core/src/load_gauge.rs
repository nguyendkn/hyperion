//! What the API needs to know about the load to refuse a request that cannot get its first token in time (`--first-token-timeout`): how many prompt
//! tokens wait for their prefill, and how long the scheduler recently took per prompt token.
//!
//! The estimate of the time to the first token of a new request is
//!
//! ```text
//! (prompt tokens queued or being prefilled + the new prompt's tokens) * recent wall nanoseconds per prefilled token
//! ```
//!
//! The per-token time is an exponential moving average (weight 1/4) of `(the prefill step + the decode step that ran since the previous prefill
//! step) / tokens of the step`, so the decode steps the policy interleaves between chunks are counted. It is a plain average of recent steps, not a
//! prediction: it ignores prefix-cache hits (a hit makes the real wait shorter, so the estimate errs towards refusing a request that would have
//! been served) and the decode that runs after the prefill. With no sample yet the estimate is unknown and nothing is refused on it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Debug, Default)]
pub struct LoadGauge {
    queued_tokens: AtomicU64,
    ns_per_token: AtomicU64,
}

impl LoadGauge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimated time until the first token of a request with `own_tokens` prompt tokens, or `None` before the first prefill step was measured.
    pub fn estimated_first_token(&self, own_tokens: usize) -> Option<Duration> {
        let per_token = self.ns_per_token.load(Ordering::Relaxed);
        if per_token == 0 {
            return None;
        }
        let tokens = self.queued_tokens.load(Ordering::Relaxed).saturating_add(own_tokens as u64);
        Some(Duration::from_nanos(tokens.saturating_mul(per_token)))
    }

    /// A request with `tokens` prompt tokens was handed to the scheduler.
    pub fn add(&self, tokens: usize) {
        self.queued_tokens.fetch_add(tokens as u64, Ordering::Relaxed);
    }

    /// Its prompt is done (first token produced), or the request ended or was refused before.
    pub fn settle(&self, tokens: usize) {
        let mut current = self.queued_tokens.load(Ordering::Relaxed);
        while let Err(seen) = self.queued_tokens.compare_exchange_weak(current, current.saturating_sub(tokens as u64), Ordering::Relaxed, Ordering::Relaxed) {
            current = seen;
        }
    }

    /// One prefill step of `tokens` tokens took `step` of wall time, with `decode` of decode-step time interleaved before it.
    pub fn record_prefill(&self, tokens: usize, step: Duration, decode: Duration) {
        if tokens == 0 {
            return;
        }
        let sample = ((step + decode).as_nanos() / tokens as u128).clamp(1, u64::MAX as u128) as u64;
        let old = self.ns_per_token.load(Ordering::Relaxed);
        let next = if old == 0 { sample } else { (old * 3 + sample) / 4 };
        self.ns_per_token.store(next.max(1), Ordering::Relaxed);
    }

    pub fn queued_tokens(&self) -> u64 {
        self.queued_tokens.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_estimated_before_a_prefill_step_was_measured() {
        let gauge = LoadGauge::new();
        gauge.add(10_000);
        assert_eq!(gauge.estimated_first_token(8192), None);
    }

    #[test]
    fn the_estimate_is_queued_plus_own_tokens_times_the_measured_time_per_token() {
        let gauge = LoadGauge::new();
        gauge.record_prefill(4096, Duration::from_millis(160), Duration::from_millis(8));
        let per_token = Duration::from_millis(168).as_nanos() / 4096;
        gauge.add(8192);
        let estimate = gauge.estimated_first_token(8192).unwrap();
        assert_eq!(estimate.as_nanos(), 16384 * per_token);
        gauge.settle(8192);
        assert_eq!(gauge.estimated_first_token(8192).unwrap().as_nanos(), 8192 * per_token);
    }

    #[test]
    fn the_average_follows_the_recent_steps_and_settling_never_goes_below_zero() {
        let gauge = LoadGauge::new();
        gauge.record_prefill(1000, Duration::from_micros(1000), Duration::ZERO); // 1000 ns per token
        gauge.record_prefill(1000, Duration::from_micros(2000), Duration::ZERO); // sample 2000: (3 * 1000 + 2000) / 4
        assert_eq!(gauge.estimated_first_token(1).unwrap(), Duration::from_nanos(1250));
        gauge.settle(5);
        assert_eq!(gauge.queued_tokens(), 0);
        gauge.record_prefill(0, Duration::from_secs(1), Duration::ZERO);
        assert_eq!(gauge.estimated_first_token(1).unwrap(), Duration::from_nanos(1250), "a step of no tokens is not a sample");
    }
}
