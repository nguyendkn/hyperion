//! Where each manifest tensor lives in a safetensors file. Pure host logic with no
//! CUDA dependency, so it is unit-tested everywhere; the device upload that
//! consumes the plan lives in `cuda::weights`.
//!
//! A safetensors file is an 8-byte little-endian header length, the JSON header,
//! then the tensor data. The manifest's `byte_offset` indexes the data section
//! (see `ignition/.../compiler/weights.py`), so a tensor's file position is
//! `8 + header_len + byte_offset`.

use std::collections::HashSet;
use std::io::Read;

use anyhow::{anyhow, bail, Result};
use hyperion_specification::v1::Manifest;

/// Header-length prefixes above this are corrupt, not real checkpoints.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// One tensor to copy: where it starts in the file and how long it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPlan {
    pub name: String,
    pub file_offset: u64,
    pub len: u64,
}

/// Read the 8-byte prefix and return the byte offset where tensor data begins.
pub fn data_start(file: &mut impl Read) -> Result<u64> {
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix).map_err(|e| anyhow!("safetensors prefix unreadable: {e}"))?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > MAX_HEADER_BYTES {
        bail!("safetensors header length {header_len} exceeds {MAX_HEADER_BYTES}");
    }
    Ok(8 + header_len)
}

/// Turn the manifest's layer map into file positions, refusing a map that cannot
/// describe real weights: empty, duplicate names, or ranges that overflow or
/// run past the end of the file.
pub fn plan(manifest: &Manifest, data_start: u64, file_len: u64) -> Result<Vec<TensorPlan>> {
    if manifest.layers.is_empty() {
        bail!("manifest has no layers: nothing to load (was it compiled with a weights header?)");
    }
    let mut seen = HashSet::new();
    let mut plans = Vec::with_capacity(manifest.layers.len());
    for layer in &manifest.layers {
        if !seen.insert(layer.name.as_str()) {
            bail!("duplicate tensor name in manifest: {}", layer.name);
        }
        let file_offset = data_start
            .checked_add(layer.byte_offset)
            .ok_or_else(|| anyhow!("{}: offset overflows", layer.name))?;
        let end = file_offset
            .checked_add(layer.byte_length)
            .ok_or_else(|| anyhow!("{}: range overflows", layer.name))?;
        if end > file_len {
            bail!("{}: range ends at {end} but the file has {file_len} bytes", layer.name);
        }
        plans.push(TensorPlan { name: layer.name.clone(), file_offset, len: layer.byte_length });
    }
    Ok(plans)
}

#[cfg(test)]
mod tests {
    use hyperion_specification::v1::{
        AttentionKind, LayerMapping, Manifest, ModelConfig, TensorFormat, TensorMetadata, TokenizerConfig,
    };

    use super::*;

    fn manifest(layers: &[(&str, u64, u64)]) -> Manifest {
        Manifest {
            model_id: "m".into(),
            target_arch: "sm_90".into(),
            tensor_format: TensorFormat::Bf16,
            model: ModelConfig {
                num_layers: 1,
                attention: AttentionKind::Mha,
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: 1,
                vocab_size: 1,
                max_seq_len: 1,
            },
            tokenizer: TokenizerConfig::default(),
            layers: layers
                .iter()
                .enumerate()
                .map(|(i, (name, off, len))| LayerMapping {
                    index: i as u32,
                    name: (*name).into(),
                    byte_offset: *off,
                    byte_length: *len,
                    tensor: TensorMetadata { dtype: "bf16".into(), shape: vec![*len as usize / 2] },
                })
                .collect(),
            hardware_recipe: Default::default(),
            signature: None,
        }
    }

    #[test]
    fn data_starts_after_the_prefix_and_header() {
        let mut bytes = 100u64.to_le_bytes().to_vec();
        bytes.extend([0u8; 4]);
        assert_eq!(data_start(&mut bytes.as_slice()).unwrap(), 108);
    }

    #[test]
    fn a_huge_header_length_is_refused() {
        let bytes = (1u64 << 40).to_le_bytes();
        assert!(data_start(&mut bytes.as_slice()).unwrap_err().to_string().contains("exceeds"));
    }

    #[test]
    fn a_short_prefix_is_refused() {
        assert!(data_start(&mut [1u8, 2, 3].as_slice()).is_err());
    }

    #[test]
    fn offsets_are_shifted_by_the_data_start() {
        let plans = plan(&manifest(&[("a", 0, 8), ("b", 8, 4)]), 108, 120).unwrap();
        assert_eq!(plans[0], TensorPlan { name: "a".into(), file_offset: 108, len: 8 });
        assert_eq!(plans[1], TensorPlan { name: "b".into(), file_offset: 116, len: 4 });
    }

    #[test]
    fn a_range_past_the_end_of_the_file_is_refused() {
        let err = plan(&manifest(&[("a", 0, 8)]), 108, 112).unwrap_err().to_string();
        assert!(err.contains("a:") && err.contains("112"), "{err}");
    }

    #[test]
    fn overflowing_offsets_are_refused() {
        assert!(plan(&manifest(&[("a", u64::MAX, 8)]), 108, u64::MAX).is_err());
        assert!(plan(&manifest(&[("a", 0, u64::MAX)]), 108, u64::MAX).is_err());
    }

    #[test]
    fn duplicates_and_empty_manifests_are_refused() {
        assert!(plan(&manifest(&[("a", 0, 2), ("a", 2, 2)]), 8, 100).unwrap_err().to_string().contains("duplicate"));
        assert!(plan(&manifest(&[]), 8, 100).unwrap_err().to_string().contains("no layers"));
    }
}
