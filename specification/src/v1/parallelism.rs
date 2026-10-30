//! How a model may be split across GPUs (ADR 0005): the sharding class of each tensor and the tensor-parallel sizes the model supports.
//!
//! Weights stay canonical (unsharded safetensors); each rank reads its own slices at load. The compiler declares the class of every
//! tensor and the supported sizes, so a loader computes a rank's slice from the shape and the class alone, and a size outside the list
//! fails with a typed error before any CUDA call. Both fields are optional with defaults that mean "single GPU, no split".

use serde::{Deserialize, Serialize};

/// The largest tensor-parallel size recorded: one node of eight GPUs. Larger groups are a pipeline or data-parallel matter (ADR 0004).
pub const MAX_TP_SIZE: u32 = 8;

/// How the tensor of a layer entry may be split between tensor-parallel ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum ShardClass {
    /// Split along the output dimension (rows of a `[out, in]` matrix, and a bias with them): Q, K, V, gate, up.
    Column,
    /// Split along the input dimension (columns): the attention output projection and the MLP down projection. A bias of a row-split
    /// projection is added once, after the all-reduce, never per rank.
    Row,
    /// Every rank holds the whole tensor: norms.
    Replicated,
    /// Split along the vocabulary rows, which the loader pads to a multiple of `tp * 64`: the embedding and the output head.
    Vocab,
}

/// The tensor-parallel sizes a model supports, written by the compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Parallelism {
    /// Ascending, starts with 1. A size `t` is listed only when [`tp_size_supported`] holds. Where the KV heads do not divide by `t`
    /// but `t` is a multiple of them, each KV head is replicated on `t / num_kv_heads` ranks.
    pub supported_tp_sizes: Vec<u32>,
}

/// Whether tensor-parallel size `tp` fits a model: `tp` divides the attention heads and the MLP width, and the KV heads either divide
/// by `tp` or `tp` is a multiple of them (each KV head is then replicated on `tp / num_kv_heads` ranks).
pub fn tp_size_supported(tp: u32, num_attention_heads: u32, num_kv_heads: u32, intermediate_size: u32) -> bool {
    tp >= 1
        && tp <= MAX_TP_SIZE
        && num_attention_heads % tp == 0
        && intermediate_size % tp == 0
        && (num_kv_heads % tp == 0 || (num_kv_heads > 0 && tp % num_kv_heads == 0))
}

/// Every supported size from 1 to [`MAX_TP_SIZE`], ascending: what the compiler writes.
pub fn supported_tp_sizes(num_attention_heads: u32, num_kv_heads: u32, intermediate_size: u32) -> Vec<u32> {
    (1..=MAX_TP_SIZE).filter(|&t| tp_size_supported(t, num_attention_heads, num_kv_heads, intermediate_size)).collect()
}

/// The vocabulary rows after padding for a vocab split: a multiple of `tp * 64`. The padded rows are zero and their logits are masked to
/// `-inf` before any softmax, argmax or sampling, so a padded id is never produced.
pub fn padded_vocab(vocab_size: usize, tp: usize) -> usize {
    let unit = tp.max(1) * 64;
    vocab_size.div_ceil(unit) * unit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_supported_sizes_of_the_models_in_the_registry() {
        // Qwen2.5-7B: 28 heads, 4 KV heads, intermediate 18944.
        assert_eq!(supported_tp_sizes(28, 4, 18944), vec![1, 2, 4]);
        // Qwen2.5-0.5B: 14 heads, 2 KV heads, intermediate 4864.
        assert_eq!(supported_tp_sizes(14, 2, 4864), vec![1, 2]);
        // Qwen3-0.6B: 16 heads, 8 KV heads, intermediate 3072.
        assert_eq!(supported_tp_sizes(16, 8, 3072), vec![1, 2, 4, 8]);
    }

    #[test]
    fn a_size_that_does_not_divide_the_heads_is_refused() {
        assert!(!tp_size_supported(8, 28, 4, 18944), "28 heads do not divide by 8");
        assert!(!tp_size_supported(3, 28, 4, 18944));
        assert!(!tp_size_supported(0, 28, 4, 18944));
    }

    #[test]
    fn a_kv_head_is_replicated_when_the_size_is_a_multiple_of_the_kv_heads() {
        // 8 heads, 2 KV heads: TP 4 puts each KV head on two ranks.
        assert!(tp_size_supported(4, 8, 2, 1024));
        // 8 heads, 3 KV heads is not a model; TP 2 divides neither way.
        assert!(!tp_size_supported(2, 8, 3, 1024));
    }

    #[test]
    fn the_vocabulary_pads_to_a_multiple_of_tp_times_64() {
        assert_eq!(padded_vocab(151_936, 1), 151_936);
        assert_eq!(padded_vocab(151_936, 2), 151_936, "151936 = 1187 * 128");
        assert_eq!(padded_vocab(151_936, 4), 152_064, "151936 is not a multiple of 256");
        assert_eq!(padded_vocab(152_064, 8), 152_064);
        assert_eq!(padded_vocab(100, 2), 128);
        assert_eq!(padded_vocab(129, 2), 256);
    }
}
