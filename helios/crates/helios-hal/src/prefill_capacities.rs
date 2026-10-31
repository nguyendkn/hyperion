//! The prefill chunk capacities of a loaded model: one captured graph per capacity, from `MIN_CHUNK` to the largest. Pure arithmetic shared by the
//! backends (which capture the graphs) and the scheduler (which picks the smallest graph that holds a chunk).

/// Smallest chunk capacity.
pub const MIN_CHUNK: usize = 128;
/// Largest chunk capacity: bounds the activation buffers, so it does not grow with the context.
pub const MAX_CHUNK: usize = 4096;

/// The largest chunk capacity for a session that holds `context` tokens: [`MAX_CHUNK`], or the smallest power of
/// two that covers the context when that is smaller (never below [`MIN_CHUNK`]).
pub fn largest_chunk(context: usize) -> usize {
    context.max(1).next_power_of_two().clamp(MIN_CHUNK, MAX_CHUNK)
}

/// Every chunk capacity of a session whose largest is `largest`: 128, 256, ... `largest`.
pub fn capacities(largest: usize) -> Vec<usize> {
    std::iter::successors(Some(MIN_CHUNK), |c| (*c < largest).then_some(c * 2)).collect()
}

/// The capacities of a model loaded for `context` tokens, ascending.
pub fn for_context(context: usize) -> Vec<usize> {
    capacities(largest_chunk(context))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacities_double_from_128_up_to_the_largest() {
        assert_eq!(for_context(1), vec![128]);
        assert_eq!(for_context(4096), vec![128, 256, 512, 1024, 2048, 4096]);
        assert_eq!(for_context(131_072), for_context(4096), "the largest chunk does not grow with the context");
        assert_eq!(for_context(300), vec![128, 256, 512]);
    }
}
