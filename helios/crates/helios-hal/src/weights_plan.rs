//! Where each manifest tensor lives in the package's safetensors files. Pure host logic with no
//! CUDA dependency, so it is unit-tested everywhere; the device upload that
//! consumes the plan lives in `cuda::weights`.
//!
//! A safetensors file is an 8-byte little-endian header length, the JSON header,
//! then the tensor data. The manifest's `byte_offset` indexes the data section of the
//! tensor's payload file (see `ignition/.../compiler/weights.py`), so a tensor's file
//! position is `8 + header_len + byte_offset` in that file. A package without a payload
//! table has one file, `model.safetensors`; a sharded checkpoint lists one payload per shard.

use std::collections::HashSet;
use std::io::Read;

use anyhow::{anyhow, bail, Result};
use crate::contract::BackendConfig;

/// Header-length prefixes above this are corrupt, not real checkpoints.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// What the file itself says about one payload: where its tensor data starts and how long it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadFile {
    pub data_start: u64,
    pub file_len: u64,
}

/// One tensor to copy: which payload file holds it, where it starts in that file, and how long it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPlan {
    pub name: String,
    /// Index into the manifest's payload list (see `Manifest::payload_files`).
    pub payload: usize,
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

/// Turn the manifest's layer map into file positions, refusing a map that cannot describe real
/// weights: empty, duplicate names, a payload that does not exist, a file whose size disagrees
/// with the payload table, or ranges that overflow or run past the end of their file. `files`
/// holds one entry per payload, in the manifest's payload order.
pub fn plan(config: &BackendConfig, files: &[PayloadFile]) -> Result<Vec<TensorPlan>> {
    if config.tensors.is_empty() {
        bail!("manifest has no layers: nothing to load (was it compiled with a weights header?)");
    }
    let names = &config.payload_files;
    if files.len() != names.len() {
        bail!("the manifest lists {} payload file(s) but {} were given", names.len(), files.len());
    }
    for (index, declared) in config.payload_lengths.iter().enumerate() {
        if let Some(declared) = declared {
            if *declared != files[index].file_len {
                bail!("payload `{}` has {} bytes but the manifest says {}", names[index], files[index].file_len, declared);
            }
        }
    }
    let mut seen = HashSet::new();
    let mut plans = Vec::with_capacity(config.tensors.len());
    for layer in &config.tensors {
        if !seen.insert(layer.name.as_str()) {
            bail!("duplicate tensor name in manifest: {}", layer.name);
        }
        let payload = layer.payload;
        let Some(file) = files.get(payload) else {
            bail!("{}: refers to payload {payload} but the manifest has {}", layer.name, files.len());
        };
        let file_offset = file
            .data_start
            .checked_add(layer.byte_offset)
            .ok_or_else(|| anyhow!("{}: offset overflows", layer.name))?;
        let end = file_offset
            .checked_add(layer.byte_length)
            .ok_or_else(|| anyhow!("{}: range overflows", layer.name))?;
        if end > file.file_len {
            bail!("{}: range ends at {end} but `{}` has {} bytes", layer.name, names[payload], file.file_len);
        }
        plans.push(TensorPlan { name: layer.name.clone(), payload, file_offset, len: layer.byte_length });
    }
    Ok(plans)
}

#[cfg(test)]
mod tests {
    use hyperion_specification::v1::{
        Activation, AttentionKind, GenerationConfig, LayerMapping, Manifest, ModelConfig, Payload, RopeStyle, TensorFormat, TensorMetadata,
        TokenizerConfig, CONTRACT_REVISION,
    };

    use super::*;

    fn manifest(layers: &[(&str, u32, u64, u64)]) -> Manifest {
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
                hidden_size: 16,
                intermediate_size: 32,
                rope_theta: 10000.0,
                rope_style: RopeStyle::RotateHalf,
                rms_norm_eps: 1e-6,
                tie_word_embeddings: false,
                qkv_bias: true,
                qk_norm: false,
                activation: Activation::Silu,
                attention_scale: None,
            },
            contract_revision: CONTRACT_REVISION,
            generation: GenerationConfig { eos_token_ids: vec![0], sampling_defaults: None },
            chat_template: None,
            qualified_checkpoint: None,
            tokenizer: TokenizerConfig::default(),
            layers: layers
                .iter()
                .enumerate()
                .map(|(i, (name, payload, off, len))| LayerMapping {
                    index: i as u32,
                    name: (*name).into(),
                    payload: *payload,
                    byte_offset: *off,
                    byte_length: *len,
                    tensor: TensorMetadata { dtype: "bf16".into(), shape: vec![*len as usize / 2] },
                })
                .collect(),
            payloads: Vec::new(),
            hardware_recipe: Default::default(),
            signature: None,
        }
    }

    fn with_payloads(mut m: Manifest, files: &[(&str, u64)]) -> Manifest {
        m.payloads = files.iter().map(|(file, len)| Payload { file: (*file).into(), byte_length: *len, sha256: "0".repeat(64) }).collect();
        m
    }

    /// The plan of a test manifest, through the config a backend is given.
    fn plan(m: &Manifest, files: &[PayloadFile]) -> Result<Vec<TensorPlan>> {
        super::plan(&BackendConfig::from_manifest(m)?, files)
    }

    fn file(data_start: u64, file_len: u64) -> PayloadFile {
        PayloadFile { data_start, file_len }
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
        let plans = plan(&manifest(&[("a", 0, 0, 8), ("b", 0, 8, 4)]), &[file(108, 120)]).unwrap();
        assert_eq!(plans[0], TensorPlan { name: "a".into(), payload: 0, file_offset: 108, len: 8 });
        assert_eq!(plans[1], TensorPlan { name: "b".into(), payload: 0, file_offset: 116, len: 4 });
    }

    #[test]
    fn a_range_past_the_end_of_the_file_is_refused() {
        let err = plan(&manifest(&[("a", 0, 0, 8)]), &[file(108, 112)]).unwrap_err().to_string();
        assert!(err.contains("a:") && err.contains("112"), "{err}");
    }

    #[test]
    fn overflowing_offsets_are_refused() {
        assert!(plan(&manifest(&[("a", 0, u64::MAX, 8)]), &[file(108, u64::MAX)]).is_err());
        assert!(plan(&manifest(&[("a", 0, 0, u64::MAX)]), &[file(108, u64::MAX)]).is_err());
    }

    #[test]
    fn duplicates_and_empty_manifests_are_refused() {
        assert!(plan(&manifest(&[("a", 0, 0, 2), ("a", 0, 2, 2)]), &[file(8, 100)]).unwrap_err().to_string().contains("duplicate"));
        assert!(plan(&manifest(&[]), &[file(8, 100)]).unwrap_err().to_string().contains("no layers"));
    }

    #[test]
    fn tensors_in_several_files_use_the_data_start_and_length_of_their_own_file() {
        // Shard 0: header 100 bytes, 40 bytes of data. Shard 1: header 20 bytes, 60 bytes of data.
        let m = with_payloads(manifest(&[("a", 0, 0, 40), ("b", 1, 0, 24), ("c", 1, 24, 36)]), &[("shard-0.safetensors", 148), ("shard-1.safetensors", 88)]);
        let plans = plan(&m, &[file(108, 148), file(28, 88)]).unwrap();
        assert_eq!(plans[0], TensorPlan { name: "a".into(), payload: 0, file_offset: 108, len: 40 });
        assert_eq!(plans[1], TensorPlan { name: "b".into(), payload: 1, file_offset: 28, len: 24 });
        assert_eq!(plans[2], TensorPlan { name: "c".into(), payload: 1, file_offset: 52, len: 36 });
    }

    #[test]
    fn a_range_is_checked_against_its_own_file_not_a_longer_one() {
        // "b" would fit in shard 0 (148 bytes) but is declared in shard 1 (88 bytes).
        let m = with_payloads(manifest(&[("a", 0, 0, 8), ("b", 1, 40, 40)]), &[("s0.safetensors", 148), ("s1.safetensors", 88)]);
        let err = plan(&m, &[file(108, 148), file(28, 88)]).unwrap_err().to_string();
        assert!(err.contains("b:") && err.contains("s1.safetensors"), "{err}");
    }

    #[test]
    fn a_layer_pointing_at_a_missing_payload_is_refused() {
        let err = plan(&manifest(&[("a", 1, 0, 8)]), &[file(8, 100)]).unwrap_err().to_string();
        assert!(err.contains("payload 1"), "{err}");
    }

    #[test]
    fn the_number_of_files_must_match_the_payload_table() {
        let m = with_payloads(manifest(&[("a", 0, 0, 8)]), &[("s0.safetensors", 100), ("s1.safetensors", 100)]);
        assert!(plan(&m, &[file(8, 100)]).unwrap_err().to_string().contains("2 payload file(s) but 1"));
    }

    #[test]
    fn a_file_whose_size_disagrees_with_the_payload_table_is_refused() {
        let m = with_payloads(manifest(&[("a", 0, 0, 8)]), &[("s0.safetensors", 100)]);
        let err = plan(&m, &[file(8, 99)]).unwrap_err().to_string();
        assert!(err.contains("s0.safetensors") && err.contains("99") && err.contains("100"), "{err}");
    }
}
