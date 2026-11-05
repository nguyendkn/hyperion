//! Family envelope contract (ADR 0012): LLM manifests parse unchanged through `parse_any`, speech manifests are strict.
use hyperion_specification::family::{parse_any, AnyManifest, FamilyManifest, ModelFamily, FAMILY_CONTRACT_REVISION};
use hyperion_specification::family_config::*;
use hyperion_specification::v1::*;
use hyperion_specification::ManifestError;
use serde_json::{json, Value};
use std::path::PathBuf;

fn spec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn decoder() -> ModelConfig {
    ModelConfig {
        attention: AttentionKind::Gqa,
        num_layers: 28,
        num_attention_heads: 16,
        num_kv_heads: 8,
        head_dim: 128,
        vocab_size: 151936,
        max_seq_len: 8192,
        hidden_size: 2048,
        intermediate_size: 6144,
        rope_theta: 1e6,
        rope_style: RopeStyle::RotateHalf,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        qkv_bias: false,
        qk_norm: true,
        activation: Activation::Silu,
        attention_scale: None,
    }
}

fn payloads() -> Vec<Payload> {
    vec![Payload { file: "model-00001-of-00002.safetensors".into(), byte_length: 10, sha256: "0".repeat(64) }]
}

fn tensors() -> Vec<LayerMapping> {
    vec![LayerMapping { index: 0, name: "encoder.layers.0.self_attn.q_proj".into(), payload: 0, byte_offset: 0, byte_length: 8, tensor: TensorMetadata { dtype: "bf16".into(), shape: vec![2, 2] }, sharding: None }]
}

fn stt() -> FamilyManifest {
    FamilyManifest {
        family: ModelFamily::SttQwen3Asr,
        contract_revision: FAMILY_CONTRACT_REVISION,
        model_id: "stt-test".into(),
        target_arch: "sm_90".into(),
        tensor_format: TensorFormat::Bf16,
        license_spdx: "Apache-2.0".into(),
        production_approved: false,
        qualified_checkpoint: None,
        payloads: payloads(),
        tensors: tensors(),
        stt: Some(SttConfig {
            architecture: SttArchitecture::Qwen3Asr,
            frontend: AudioFrontend { sample_rate: 16000, n_fft: 400, hop: 160, n_mels: 128, f_min: 0.0, f_max: 8000.0, whisper_norm: true, chunk_seconds: 30 },
            encoder: AudioEncoderConfig { num_layers: 24, hidden_size: 1024, num_heads: 16, intermediate_size: 4096, conv_channels: 480, subsample: 8, window_frames_min: 13, window_frames_max: 100 },
            decoder: decoder(),
            audio_token_id: 151676,
        }),
        tts: None,
        signature: None,
    }
}

fn tts() -> FamilyManifest {
    let mut m = stt();
    m.family = ModelFamily::TtsVieneuV3;
    m.stt = None;
    m.tts = Some(TtsConfig {
        architecture: TtsArchitecture::VieneuV3,
        backbone: decoder(),
        depth_decoder: DepthDecoderConfig { num_layers: 2, hidden_size: 768, num_heads: 8, intermediate_size: 2048 },
        codec: CodecConfig { sample_rate: 48000, frame_rate_milli_hz: 12500, num_codebooks: 16, codebook_size: 1024 },
        voices: vec!["preset-1".into()],
    });
    m
}

fn json_of(m: &FamilyManifest) -> Value {
    serde_json::to_value(m).unwrap()
}

#[test]
fn llm_fixtures_parse_unchanged_through_parse_any() {
    let dir = spec_dir().join("fixtures/v1");
    let mut n = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let bytes = std::fs::read(entry.unwrap().path()).unwrap();
        match parse_any(&bytes) {
            Ok(AnyManifest::Llm(_)) => n += 1,
            // a fixture written under an older contract revision is refused by the unchanged LLM parser, never routed to the family parser
            Err(ManifestError::UnsupportedContractVersion { .. }) => n += 1,
            other => panic!("fixture went to the wrong parser: {other:?}"),
        }
    }
    assert!(n >= 3);
}

#[test]
fn speech_manifests_roundtrip_and_dispatch() {
    for m in [stt(), tts()] {
        let bytes = serde_json::to_vec(&m).unwrap();
        match parse_any(&bytes).unwrap() {
            AnyManifest::Family(f) => assert_eq!(f.family, m.family),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn section_must_match_family() {
    let mut j = json_of(&stt());
    j["family"] = json!("tts_vieneu_v3");
    assert!(matches!(FamilyManifest::from_json_bytes(&serde_json::to_vec(&j).unwrap()), Err(ManifestError::Invalid(_))));
}

#[test]
fn strictness_revision_and_payload_rules() {
    let base = json_of(&stt());
    let mut j = base.clone();
    j["surprise"] = json!(1);
    assert!(FamilyManifest::from_json_bytes(&serde_json::to_vec(&j).unwrap()).is_err(), "unknown field");
    let mut j = base.clone();
    j["contract_revision"] = json!(99);
    assert!(matches!(FamilyManifest::from_json_bytes(&serde_json::to_vec(&j).unwrap()), Err(ManifestError::UnsupportedContractVersion { .. })));
    let mut j = base.clone();
    j["tensors"][0]["payload"] = json!(5);
    assert!(matches!(FamilyManifest::from_json_bytes(&serde_json::to_vec(&j).unwrap()), Err(ManifestError::Invalid(_))));
    let mut j = base.clone();
    j["payloads"][0]["file"] = json!("../evil.safetensors");
    assert!(FamilyManifest::from_json_bytes(&serde_json::to_vec(&j).unwrap()).is_err(), "path traversal");
    let mut j = base;
    j["family"] = json!("decoder_lm");
    assert!(matches!(parse_any(&serde_json::to_vec(&j).unwrap()), Err(ManifestError::Invalid(_))));
}

#[test]
fn production_gate_defaults_to_false() {
    let mut j = json_of(&tts());
    j.as_object_mut().unwrap().remove("production_approved");
    let m = FamilyManifest::from_json_bytes(&serde_json::to_vec(&j).unwrap()).unwrap();
    assert!(!m.production_approved);
}

#[test]
fn tts_needs_a_voice() {
    let mut j = json_of(&tts());
    j["tts"]["voices"] = json!([]);
    assert!(FamilyManifest::from_json_bytes(&serde_json::to_vec(&j).unwrap()).is_err());
}

#[test]
#[cfg(feature = "schema")]
fn committed_family_schema_is_up_to_date() {
    let committed = std::fs::read_to_string(spec_dir().join("schema/family-manifest.v1.json")).expect("schema file exists").replace("\r\n", "\n");
    assert_eq!(committed, hyperion_specification::family_schema::family_schema_json(), "family-manifest.v1.json is stale: run gen-family-schema");
}
