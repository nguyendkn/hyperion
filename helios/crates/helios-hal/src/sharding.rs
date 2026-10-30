//! A rank's share of each tensor under tensor and pipeline parallelism (ADR 0005). Pure host logic, no CUDA: the loader reads each rank's
//! slices from the canonical safetensors at load, so one package serves every supported tensor-parallel size.
//!
//! Classes come from the manifest (declared by Ignition). Column: output rows (Q, K, V, gate, up and the biases of Q, K, V). Row: input
//! columns (attention output, MLP down). Vocab: embedding and head rows, padded to a multiple of `tp * 64` (padded rows are zero, their
//! logits masked to `-inf`). Replicated: whole. K and V rows follow the KV heads: split when they divide by `tp`, otherwise each KV
//! head is read by `tp / num_kv_heads` ranks (its q heads stay on those ranks).

use std::ops::Range;

use hyperion_specification::v1::{padded_vocab, ShardClass};

use crate::contract::{BackendConfig, HalError, TensorSlice, TensorSpec};

/// How the KV heads sit on the ranks of a tensor-parallel group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPlacement {
    /// Each rank holds `per_rank` whole KV heads, none shared.
    Split { per_rank: usize },
    /// Every KV head is held by `ranks_per_head` ranks; each rank holds one.
    Replicated { ranks_per_head: usize },
}

pub fn kv_placement(num_kv_heads: usize, tp: usize) -> KvPlacement {
    if num_kv_heads % tp == 0 {
        KvPlacement::Split { per_rank: num_kv_heads / tp }
    } else {
        KvPlacement::Replicated { ranks_per_head: tp / num_kv_heads }
    }
}

/// What one rank computes: its attention heads, its KV heads, its MLP width and its vocabulary rows (the padded count and the real ones).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankShape {
    pub heads: usize,
    pub kv_heads: usize,
    pub intermediate: usize,
    pub padded_vocab: usize,
    /// The real vocabulary rows of this rank's slice (the last ranks may hold fewer, or none, when the vocabulary is padded).
    pub vocab_rows: Range<usize>,
}

pub fn rank_shape(config: &BackendConfig) -> RankShape {
    let (tp, rank) = (config.deployment.tp_size as usize, config.deployment.tp_rank as usize);
    let m = &config.model;
    let kv_heads = match kv_placement(m.num_kv_heads as usize, tp) {
        KvPlacement::Split { per_rank } => per_rank,
        KvPlacement::Replicated { .. } => 1,
    };
    let padded = padded_vocab(m.vocab_size as usize, tp);
    let per = padded / tp;
    let real = |at: usize| at.min(m.vocab_size as usize);
    RankShape { heads: m.num_attention_heads as usize / tp, kv_heads, intermediate: m.intermediate_size as usize / tp, padded_vocab: padded, vocab_rows: real(rank * per)..real((rank + 1) * per) }
}

/// The slice of `tensor` this rank reads, or `None` when it reads the whole tensor.
pub fn rank_slice(config: &BackendConfig, tensor: &TensorSpec) -> Result<Option<TensorSlice>, HalError> {
    let (tp, rank) = (config.deployment.tp_size as usize, config.deployment.tp_rank as usize);
    let bad = |why: String| HalError::Invalid(format!("{}: {why}", tensor.name));
    let dim = |axis: usize| tensor.shape.get(axis).copied().ok_or_else(|| bad(format!("has no axis {axis} (shape {:?})", tensor.shape)));
    if tp == 1 {
        return Ok(None);
    }
    match tensor.sharding.unwrap_or(ShardClass::Replicated) {
        ShardClass::Replicated => Ok(None),
        ShardClass::Column => {
            let rows = dim(0)?;
            let is_kv = tensor.name.contains(".k_proj.") || tensor.name.contains(".v_proj.");
            let kv_heads = config.model.num_kv_heads as usize;
            if is_kv {
                if let KvPlacement::Replicated { ranks_per_head } = kv_placement(kv_heads, tp) {
                    let head_rows = rows / kv_heads;
                    let head = rank / ranks_per_head;
                    return Ok(Some(TensorSlice { axis: 0, start: head * head_rows, end: (head + 1) * head_rows }));
                }
            }
            if rows % tp != 0 {
                return Err(bad(format!("{rows} rows do not divide by tp {tp}")));
            }
            Ok(Some(TensorSlice { axis: 0, start: rank * rows / tp, end: (rank + 1) * rows / tp }))
        }
        ShardClass::Row => {
            let cols = dim(1)?;
            if cols % tp != 0 {
                return Err(bad(format!("{cols} columns do not divide by tp {tp}")));
            }
            Ok(Some(TensorSlice { axis: 1, start: rank * cols / tp, end: (rank + 1) * cols / tp }))
        }
        ShardClass::Vocab => {
            let rows = dim(0)?;
            let per = padded_vocab(rows, tp) / tp;
            Ok(Some(TensorSlice { axis: 0, start: (rank * per).min(rows), end: ((rank + 1) * per).min(rows) }))
        }
    }
}

/// The slices of every split tensor for this rank: the `WeightSource::slices` of a tensor-parallel load.
pub fn rank_slices(config: &BackendConfig) -> Result<Vec<(String, TensorSlice)>, HalError> {
    let mut out = Vec::new();
    for tensor in &config.tensors {
        if let Some(slice) = rank_slice(config, tensor)? {
            out.push((tensor.name.clone(), slice));
        }
    }
    Ok(out)
}

/// The layers of pipeline stage `stage`: balanced contiguous groups (the first `layers % pp` stages hold one more). The embedding sits
/// on stage 0, the final norm and the head on the last stage; none of that is stored in the manifest.
pub fn pipeline_layers(num_layers: usize, pp_size: usize, stage: usize) -> Range<usize> {
    let (base, extra) = (num_layers / pp_size, num_layers % pp_size);
    let start = stage * base + stage.min(extra);
    start..start + base + usize::from(stage < extra)
}

/// Copy `slice` out of a row-major tensor of `shape` with `elem` bytes per element.
pub fn extract(bytes: &[u8], shape: &[usize], elem: usize, slice: &TensorSlice) -> Vec<u8> {
    let outer: usize = shape[..slice.axis].iter().product();
    let inner: usize = shape[slice.axis + 1..].iter().product::<usize>() * elem;
    let along = shape[slice.axis];
    let mut out = Vec::with_capacity(outer * (slice.end - slice.start) * inner);
    for o in 0..outer {
        let base = o * along * inner;
        out.extend_from_slice(&bytes[base + slice.start * inner..base + slice.end * inner]);
    }
    out
}

/// Write `part` (a slice taken with [`extract`]) back into `whole`.
pub fn place(whole: &mut [u8], shape: &[usize], elem: usize, slice: &TensorSlice, part: &[u8]) {
    let outer: usize = shape[..slice.axis].iter().product();
    let inner: usize = shape[slice.axis + 1..].iter().product::<usize>() * elem;
    let along = shape[slice.axis];
    let chunk = (slice.end - slice.start) * inner;
    for o in 0..outer {
        let base = o * along * inner + slice.start * inner;
        whole[base..base + chunk].copy_from_slice(&part[o * chunk..(o + 1) * chunk]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_stages_are_balanced_contiguous_and_cover_every_layer_once() {
        for (layers, pp) in [(28, 1), (28, 2), (28, 3), (28, 4), (5, 5), (7, 3)] {
            let mut next = 0;
            for stage in 0..pp {
                let range = pipeline_layers(layers, pp, stage);
                assert_eq!(range.start, next, "{layers} layers, pp {pp}, stage {stage}");
                assert!(range.len() >= layers / pp && range.len() <= layers / pp + 1);
                next = range.end;
            }
            assert_eq!(next, layers);
        }
    }

    #[test]
    fn extract_and_place_round_trip_on_both_axes() {
        let shape = [4usize, 6];
        let whole: Vec<u8> = (0..24u8).flat_map(|v| [v, v]).collect();
        for slice in [TensorSlice { axis: 0, start: 1, end: 3 }, TensorSlice { axis: 1, start: 2, end: 5 }] {
            let part = extract(&whole, &shape, 2, &slice);
            let mut rebuilt = vec![0xEE; whole.len()];
            place(&mut rebuilt, &shape, 2, &slice, &part);
            assert_eq!(extract(&rebuilt, &shape, 2, &slice), part);
        }
        // Rows 1..3 of a 4x6 matrix are elements 6..18 (12 bytes per row, so bytes 12..36).
        assert_eq!(extract(&whole, &shape, 2, &TensorSlice { axis: 0, start: 1, end: 3 }), whole[12..36]);
        // Columns 2..5 of row 0 are elements 2, 3, 4 (bytes 4..10).
        assert_eq!(&extract(&whole, &shape, 2, &TensorSlice { axis: 1, start: 2, end: 5 })[..6], &whole[4..10]);
    }

    #[test]
    fn kv_heads_split_or_replicate_by_the_size() {
        assert_eq!(kv_placement(4, 2), KvPlacement::Split { per_rank: 2 });
        assert_eq!(kv_placement(4, 4), KvPlacement::Split { per_rank: 1 });
        assert_eq!(kv_placement(2, 4), KvPlacement::Replicated { ranks_per_head: 2 });
    }
}
