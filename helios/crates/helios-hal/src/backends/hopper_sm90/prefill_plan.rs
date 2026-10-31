//! How a prompt is cut into prefill chunks. Pure arithmetic, so the boundary cases are tested without a GPU.
//!
//! A chunk has a *capacity* (a power of two from [`MIN_CHUNK`] to the session's largest chunk; one CUDA graph
//! is captured per capacity when the model is loaded) and holds `valid <= capacity` real prompt tokens. A
//! prompt is cut into chunks of the largest capacity, and the remainder goes in the smallest capacity that
//! holds it; the remainder's unused rows are padding that the kernels never write to the cache.

use anyhow::{anyhow, Result};

/// Why a prompt cannot be prefilled. Typed so a caller can tell a request that is too long from a failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PrefillError {
    #[error("prompt + max_new = {requested} exceeds the {capacity} tokens of context this model was loaded with")]
    ContextTooLong { requested: usize, capacity: usize },
    #[error("no prefill graph of {capacity} rows was captured when the model was loaded")]
    NoGraph { capacity: usize },
    #[error("no prompt tokens")]
    EmptyPrompt,
}

pub(super) use crate::prefill_capacities::{capacities, largest_chunk, MAX_CHUNK, MIN_CHUNK};

/// One chunk of a prompt: `valid` tokens at positions `start..start + valid`, run in a graph of `capacity` rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Chunk {
    pub start: usize,
    pub valid: usize,
    pub capacity: usize,
}

/// Cut a prompt of `tokens` tokens into chunks. An empty prompt is an error: there is nothing to run.
pub(super) fn plan_chunks(tokens: usize, largest: usize) -> Result<Vec<Chunk>> {
    if tokens == 0 {
        return Err(PrefillError::EmptyPrompt.into());
    }
    if !largest.is_power_of_two() || !(MIN_CHUNK..=MAX_CHUNK).contains(&largest) {
        return Err(anyhow!("the largest chunk capacity must be a power of two between {MIN_CHUNK} and {MAX_CHUNK}, got {largest}"));
    }
    let mut chunks = Vec::with_capacity(tokens.div_ceil(largest));
    let mut start = 0;
    while start < tokens {
        let valid = (tokens - start).min(largest);
        let capacity = if valid == largest { largest } else { valid.next_power_of_two().max(MIN_CHUNK) };
        chunks.push(Chunk { start, valid, capacity });
        start += valid;
    }
    Ok(chunks)
}

/// Virtual rows per block of the tensor-core prefill attention kernel (4 warps of 16).
pub(super) const MMA_BLOCK_ROWS: usize = 64;

/// The launch grid of the tensor-core prefill attention for a chunk of `rows` rows: one block per 64 "virtual rows" (a virtual row is a
/// (chunk row, query head of the group) pair) and one grid row per KV head. `None` when heads is not a multiple of kv_heads.
pub(super) fn mma_grid(rows: usize, heads: usize, kv_heads: usize) -> Option<(usize, usize)> {
    if kv_heads == 0 || heads % kv_heads != 0 || rows == 0 {
        return None;
    }
    Some(((rows * (heads / kv_heads)).div_ceil(MMA_BLOCK_ROWS), kv_heads))
}

/// The (chunk row, query head) a virtual row stands for, as the kernel computes it.
#[cfg(test)]
pub(super) fn virtual_row(v: usize, heads: usize, kv_heads: usize, kv_head: usize) -> (usize, usize) {
    let group = heads / kv_heads;
    (v / group, kv_head * group + v % group)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(tokens: usize) -> Vec<(usize, usize, usize)> {
        plan_chunks(tokens, MAX_CHUNK).unwrap().iter().map(|c| (c.start, c.valid, c.capacity)).collect()
    }

    #[test]
    fn short_prompts_use_the_smallest_capacity_that_holds_them() {
        assert_eq!(plan(1), vec![(0, 1, 128)]);
        assert_eq!(plan(127), vec![(0, 127, 128)]);
        assert_eq!(plan(128), vec![(0, 128, 128)]);
        assert_eq!(plan(129), vec![(0, 129, 256)]);
        assert_eq!(plan(4095), vec![(0, 4095, 4096)]);
        assert_eq!(plan(4096), vec![(0, 4096, 4096)]);
    }

    #[test]
    fn long_prompts_are_cut_into_full_chunks_and_a_remainder() {
        assert_eq!(plan(8191), vec![(0, 4096, 4096), (4096, 4095, 4096)]);
        assert_eq!(plan(8192), vec![(0, 4096, 4096), (4096, 4096, 4096)]);
        assert_eq!(plan(8193), vec![(0, 4096, 4096), (4096, 4096, 4096), (8192, 1, 128)]);
        let long = plan(32768);
        assert_eq!(long.len(), 8);
        assert!(long.iter().all(|&(_, valid, capacity)| valid == 4096 && capacity == 4096));
    }

    #[test]
    fn chunks_tile_the_prompt_exactly_for_every_length_up_to_a_few_chunks() {
        for tokens in 1..=3 * MAX_CHUNK + 5 {
            let chunks = plan_chunks(tokens, MAX_CHUNK).unwrap();
            let mut next = 0;
            for c in &chunks {
                assert_eq!(c.start, next, "{tokens}: chunks are contiguous");
                assert!(c.valid >= 1 && c.valid <= c.capacity, "{tokens}: {c:?}");
                assert!(c.capacity.is_power_of_two() && (MIN_CHUNK..=MAX_CHUNK).contains(&c.capacity), "{tokens}: {c:?}");
                next += c.valid;
            }
            assert_eq!(next, tokens, "{tokens}: every token is in exactly one chunk");
            // Only the last chunk may be partial, and it is padded to the smallest capacity that holds it.
            let last = chunks.last().unwrap();
            assert!(chunks[..chunks.len() - 1].iter().all(|c| c.valid == MAX_CHUNK), "{tokens}");
            assert!(last.capacity == MAX_CHUNK || last.capacity / 2 < last.valid || last.capacity == MIN_CHUNK, "{tokens}: {last:?}");
        }
    }

    #[test]
    fn an_empty_prompt_and_an_invalid_largest_capacity_are_refused() {
        assert_eq!(plan_chunks(0, MAX_CHUNK).unwrap_err().downcast_ref::<PrefillError>(), Some(&PrefillError::EmptyPrompt));
        for bad in [0, 64, 127, 129, 3000, 8192] {
            assert!(plan_chunks(10, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_smaller_session_uses_a_smaller_largest_chunk() {
        assert_eq!(largest_chunk(16), 128);
        assert_eq!(largest_chunk(128), 128);
        assert_eq!(largest_chunk(129), 256);
        assert_eq!(largest_chunk(5000), 4096);
        assert_eq!(largest_chunk(1_000_000), 4096);
        assert_eq!(plan_chunks(300, 256).unwrap().iter().map(|c| c.capacity).collect::<Vec<_>>(), vec![256, 128]);
    }

    #[test]
    fn every_chunk_row_and_query_head_is_covered_by_exactly_one_virtual_row() {
        // 7B (28 heads / 4 kv heads, group 7), 0.5B (14/2, group 7), MHA (group 1), group 2 and 8; chunk capacities and ragged sizes.
        for (heads, kv_heads) in [(28usize, 4usize), (14, 2), (16, 16), (16, 8), (64, 8)] {
            for rows in [1usize, 2, 9, 10, 63, 64, 65, 127, 128, 129, 4096] {
                let (blocks, kv_rows) = mma_grid(rows, heads, kv_heads).unwrap();
                assert_eq!(kv_rows, kv_heads);
                let group = heads / kv_heads;
                let mut seen = vec![0u8; rows * heads];
                for kv_head in 0..kv_heads {
                    for v in 0..blocks * MMA_BLOCK_ROWS {
                        if v < rows * group {
                            let (row, head) = virtual_row(v, heads, kv_heads, kv_head);
                            assert!(row < rows && head / group == kv_head, "({heads},{kv_heads}) rows {rows} v {v}");
                            seen[row * heads + head] += 1;
                        }
                    }
                }
                assert!(seen.iter().all(|&n| n == 1), "({heads},{kv_heads}) rows {rows}: every (row, head) exactly once");
                assert!((blocks - 1) * MMA_BLOCK_ROWS < rows * group, "no block is empty");
            }
        }
        assert_eq!(mma_grid(4096, 28, 4), Some((448, 4)));
        assert!(mma_grid(10, 14, 4).is_none() && mma_grid(0, 8, 8).is_none());
    }

    #[test]
    fn the_graph_capacities_are_the_powers_of_two_up_to_the_largest() {
        assert_eq!(capacities(4096), vec![128, 256, 512, 1024, 2048, 4096]);
        assert_eq!(capacities(128), vec![128]);
        assert_eq!(capacities(512), vec![128, 256, 512]);
    }
}
