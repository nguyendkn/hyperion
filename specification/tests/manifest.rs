//! Manifest contract tests: wire format, strictness, and fixtures shared with
//! Ignition (Python). `specification/fixtures/v1/*.json` are parsed by both
//! sides; `rust-sample.json` is written by this test, the Python-produced ones
//! by `ignition/tests`. Set `UPDATE_FIXTURES=1` to rewrite the Rust-owned files.

use std::collections::HashMap;
use std::path::PathBuf;

use hyperion_specification::v1::{
    is_bare_file_name, Activation, AttentionKind, ChatTemplate, GenerationConfig, LayerMapping, Manifest as ManifestV1, ModelConfig,
    Payload, QualifiedCheckpoint, RopeStyle, SamplingDefaults, TensorFormat, TensorMetadata, TokenizerConfig, CONTRACT_REVISION,
    DEFAULT_PAYLOAD_FILE,
};
use hyperion_specification::{Manifest, ManifestError};
use serde_json::{json, Value};

fn spec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixtures() -> PathBuf {
    spec_dir().join("fixtures/v1")
}

fn sample() -> Manifest {
    Manifest::V1(ManifestV1 {
        model_id: "llama3".into(),
        target_arch: "sm_120".into(),
        tensor_format: TensorFormat::Fp4,
        model: ModelConfig {
            attention: AttentionKind::Gqa,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 128256,
            max_seq_len: 8192,
            hidden_size: 4096,
            intermediate_size: 14336,
            rope_theta: 500000.0,
            rope_style: RopeStyle::RotateHalf,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            qkv_bias: false,
            qk_norm: false,
            activation: Activation::Silu,
            attention_scale: None,
        },
        contract_revision: CONTRACT_REVISION,
        generation: GenerationConfig {
            eos_token_ids: vec![128001, 128009],
            sampling_defaults: Some(SamplingDefaults { temperature: Some(0.6), top_p: Some(0.9), top_k: None, repetition_penalty: None }),
        },
        chat_template: Some(ChatTemplate { template: "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}".into(), bos_token: "<|begin_of_text|>".into(), eos_token: "<|eot_id|>".into() }),
        qualified_checkpoint: Some(QualifiedCheckpoint { repo: "meta-llama/Meta-Llama-3-8B".into(), revision: "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920".into() }),
        tokenizer: TokenizerConfig { family: "bpe".into(), asset_path: "tokenizer.json".into() },
        layers: vec![LayerMapping {
            index: 0,
            name: "attention.q_proj".into(),
            payload: 0,
            byte_offset: 0,
            byte_length: 33_554_432,
            tensor: TensorMetadata { dtype: "bf16".into(), shape: vec![4096, 4096] },
        }],
        payloads: Vec::new(),
        hardware_recipe: HashMap::from([("fp4".to_string(), "enabled".to_string())]),
        signature: None,
    })
}

fn valid_json() -> Value {
    serde_json::to_value(sample()).unwrap()
}

#[test]
fn tag_is_the_variant_name() {
    assert_eq!(valid_json()["manifest_version"], json!("V1"));
}

#[test]
fn rust_sample_matches_committed_fixture() {
    let path = fixtures().join("rust-sample.json");
    let actual = valid_json();
    if std::env::var_os("UPDATE_FIXTURES").is_some() {
        std::fs::write(&path, serde_json::to_string_pretty(&actual).unwrap() + "\n").unwrap();
    }
    let committed: Value = serde_json::from_slice(&std::fs::read(&path).expect("fixture exists")).unwrap();
    assert_eq!(actual, committed, "run with UPDATE_FIXTURES=1 to refresh {}", path.display());
}

#[test]
fn every_fixture_parses_and_round_trips() {
    let mut n = 0;
    for entry in std::fs::read_dir(fixtures()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let parsed: Manifest = serde_json::from_value(value.clone())
            .unwrap_or_else(|e| panic!("{} must parse: {e}", path.display()));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), value, "{} round trip", path.display());
        n += 1;
    }
    assert!(n >= 1, "no fixtures found");
}

#[test]
fn integer_tag_is_rejected() {
    let mut v = valid_json();
    v["manifest_version"] = json!(1);
    assert!(serde_json::from_value::<Manifest>(v).is_err());
}

#[test]
fn unknown_fields_are_rejected_at_every_level() {
    for path in [vec!["extra"], vec!["model", "extra"], vec!["tokenizer", "extra"], vec!["layers", "0", "extra"]] {
        let mut v = valid_json();
        let mut node = &mut v;
        for key in &path[..path.len() - 1] {
            node = match key.parse::<usize>() {
                Ok(i) => &mut node[i],
                Err(_) => &mut node[*key],
            };
        }
        node[path[path.len() - 1]] = json!(1);
        assert!(serde_json::from_value::<Manifest>(v).is_err(), "{path:?} should be rejected");
    }
}

#[test]
fn missing_model_config_is_rejected() {
    let mut v = valid_json();
    v.as_object_mut().unwrap().remove("model");
    assert!(serde_json::from_value::<Manifest>(v).is_err());
}

fn v1(manifest: Manifest) -> ManifestV1 {
    let Manifest::V1(m) = manifest;
    m
}

fn with_payloads(files: &[(&str, &str)]) -> ManifestV1 {
    let mut m = v1(sample());
    m.payloads = files.iter().map(|(file, sha256)| Payload { file: (*file).into(), byte_length: 100, sha256: (*sha256).into() }).collect();
    m
}

const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn a_manifest_without_a_payload_table_has_the_one_default_file_and_writes_no_table() {
    let m = v1(sample());
    assert_eq!(m.payload_files(), vec![DEFAULT_PAYLOAD_FILE]);
    assert!(m.validate_payloads().is_ok());
    let json = valid_json();
    assert!(json.get("payloads").is_none(), "an empty table is not written");
    assert!(json["layers"][0].get("payload").is_none(), "payload 0 is not written");
}

#[test]
fn a_payload_table_round_trips_and_names_the_files_in_order() {
    let mut m = with_payloads(&[("model-00001-of-00002.safetensors", SHA), ("model-00002-of-00002.safetensors", SHA)]);
    m.layers[0].payload = 1;
    assert!(m.validate_payloads().is_ok());
    assert_eq!(m.payload_files(), vec!["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]);
    let value = serde_json::to_value(Manifest::V1(m.clone())).unwrap();
    assert_eq!(value["layers"][0]["payload"], json!(1));
    let back = v1(serde_json::from_value::<Manifest>(value).unwrap());
    assert_eq!(back.payloads, m.payloads);
    assert_eq!(back.layers[0].payload, 1);
}

#[test]
fn only_bare_file_names_are_payload_names() {
    // The same examples are read by the Ignition test (fixtures/bare-file-names.json), so the two
    // implementations of the rule cannot drift apart.
    let cases: Value = serde_json::from_slice(&std::fs::read(spec_dir().join("fixtures/bare-file-names.json")).unwrap()).unwrap();
    for good in cases["good"].as_array().unwrap() {
        assert!(is_bare_file_name(good.as_str().unwrap()), "{good}");
    }
    for bad in cases["bad"].as_array().unwrap() {
        assert!(!is_bare_file_name(bad.as_str().unwrap()), "{bad} must be refused");
    }
    assert!(!is_bare_file_name(&"a".repeat(256)), "256 bytes is too long");
    assert!(is_bare_file_name(&"a".repeat(255)));
}

#[test]
fn a_payload_table_with_a_path_a_duplicate_or_a_bad_hash_is_refused() {
    for (files, why) in [
        (vec![("../model.safetensors", SHA)], "path"),
        (vec![("a/b.safetensors", SHA)], "separator"),
        (vec![("a.safetensors", SHA), ("a.safetensors", SHA)], "duplicate"),
        (vec![("a.safetensors", "ABCDEF")], "short hash"),
        (vec![("a.safetensors", &SHA.to_uppercase())], "uppercase hash"),
    ] {
        let err = with_payloads(&files).validate_payloads().expect_err(why);
        assert!(err.to_string().contains("payload"), "{why}: {err}");
    }
}

#[test]
fn a_layer_pointing_past_the_payload_table_is_refused() {
    let mut m = with_payloads(&[("a.safetensors", SHA)]);
    m.layers[0].payload = 1;
    assert!(m.validate_payloads().unwrap_err().to_string().contains("refers to payload 1"));
    // Without a table there is exactly one payload.
    let mut m = v1(sample());
    m.layers[0].payload = 1;
    assert!(m.validate_payloads().is_err());
}

#[test]
fn the_compiled_sharded_manifest_parses_validates_and_round_trips() {
    // Written by Ignition from a Hugging Face `save_pretrained` checkpoint of five shards.
    let path = spec_dir().join("../testdata/synthetic/qwen2-tiny-sharded/model.manifest.json");
    let value: Value = serde_json::from_slice(&std::fs::read(&path).expect("fixture exists")).unwrap();
    let parsed: Manifest = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), value, "round trip");
    let manifest = v1(parsed);
    assert!(manifest.payloads.len() >= 3);
    manifest.validate_payloads().expect("the compiled table is valid");
}

#[test]
fn unknown_fields_in_a_payload_are_rejected() {
    let mut v = serde_json::to_value(Manifest::V1(with_payloads(&[("a.safetensors", SHA)]))).unwrap();
    v["payloads"][0]["extra"] = json!(1);
    assert!(serde_json::from_value::<Manifest>(v).is_err());
}

#[cfg(feature = "schema")]
#[test]
fn committed_schema_is_up_to_date() {
    let committed = std::fs::read_to_string(spec_dir().join("schema/manifest.v1.json"))
        .expect("schema file exists")
        .replace("\r\n", "\n");
    assert_eq!(
        committed,
        hyperion_specification::schema::manifest_schema_json(),
        "schema/manifest.v1.json is stale: run `task schema:gen`"
    );
}

fn bytes_of(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}

#[test]
fn from_json_bytes_reads_a_current_manifest_and_validates_it() {
    let parsed = Manifest::from_json_bytes(&bytes_of(&valid_json())).expect("a current manifest parses");
    let Manifest::V1(m) = parsed;
    assert_eq!(m.contract_revision, CONTRACT_REVISION);
    assert_eq!(m.generation.eos_token_ids, vec![128001, 128009]);
}

#[test]
fn an_old_manifest_without_the_new_fields_is_refused_with_the_reexport_hint() {
    // A manifest as written before the contract revision existed: no contract_revision, no constants.
    let mut old = valid_json();
    let obj = old.as_object_mut().unwrap();
    for key in ["contract_revision", "generation", "chat_template", "qualified_checkpoint"] {
        obj.remove(key);
    }
    for key in ["hidden_size", "intermediate_size", "rope_theta", "rope_style", "rms_norm_eps", "tie_word_embeddings", "qkv_bias", "qk_norm", "activation"] {
        old["model"].as_object_mut().unwrap().remove(key);
    }
    let err = Manifest::from_json_bytes(&bytes_of(&old)).unwrap_err();
    assert!(matches!(err, ManifestError::UnsupportedContractVersion { .. }), "{err}");
    let text = err.to_string();
    assert!(text.contains("re-export") && text.contains("Ignition >="), "{text}");
    assert!(text.contains("contract_revision missing"), "{text}");
}

#[test]
fn an_unknown_manifest_version_or_contract_revision_is_refused_before_any_field_is_read() {
    for (field, value) in [("manifest_version", json!("V2")), ("manifest_version", json!(1)), ("contract_revision", json!(0)), ("contract_revision", json!(99))] {
        let mut v = valid_json();
        v[field] = value.clone();
        // Even with the model object gone, the version complaint comes first.
        v.as_object_mut().unwrap().remove("model");
        let err = Manifest::from_json_bytes(&bytes_of(&v)).unwrap_err();
        assert!(matches!(err, ManifestError::UnsupportedContractVersion { .. }), "{field}={value}: {err}");
        assert!(err.to_string().contains("re-export"), "{err}");
    }
}

#[test]
fn constants_are_required_there_is_no_default() {
    for key in ["hidden_size", "intermediate_size", "rope_theta", "rope_style", "rms_norm_eps", "tie_word_embeddings", "qkv_bias", "qk_norm", "activation"] {
        let mut v = valid_json();
        v["model"].as_object_mut().unwrap().remove(key);
        let err = Manifest::from_json_bytes(&bytes_of(&v)).unwrap_err();
        assert!(err.to_string().contains(key), "{key}: {err}");
    }
    let mut v = valid_json();
    v.as_object_mut().unwrap().remove("generation");
    assert!(Manifest::from_json_bytes(&bytes_of(&v)).is_err());
}

#[test]
fn values_the_runtime_does_not_implement_are_not_representable() {
    for (key, value) in [("activation", json!("gelu")), ("rope_style", json!("interleaved"))] {
        let mut v = valid_json();
        v["model"][key] = value;
        assert!(Manifest::from_json_bytes(&bytes_of(&v)).is_err(), "{key}");
    }
}

#[test]
fn validation_refuses_an_empty_eos_list_a_zero_dimension_and_a_bad_checkpoint_revision() {
    for (what, mutate) in [
        ("empty eos", Box::new(|v: &mut Value| v["generation"]["eos_token_ids"] = json!([])) as Box<dyn Fn(&mut Value)>),
        ("zero hidden", Box::new(|v: &mut Value| v["model"]["hidden_size"] = json!(0))),
        ("negative rope", Box::new(|v: &mut Value| v["model"]["rope_theta"] = json!(-1.0))),
        ("kv heads", Box::new(|v: &mut Value| v["model"]["num_kv_heads"] = json!(5))),
        ("short revision", Box::new(|v: &mut Value| v["qualified_checkpoint"]["revision"] = json!("main"))),
        ("scale", Box::new(|v: &mut Value| v["model"]["attention_scale"] = json!(0.0))),
    ] {
        let mut v = valid_json();
        mutate(&mut v);
        let err = Manifest::from_json_bytes(&bytes_of(&v)).expect_err(what);
        assert!(matches!(err, ManifestError::Invalid(_)), "{what}: {err}");
    }
}

#[test]
fn the_attention_scale_defaults_to_one_over_sqrt_head_dim() {
    let m = v1(sample());
    assert!((m.model.effective_attention_scale() - 1.0 / 128f64.sqrt()).abs() < 1e-15);
    let mut with = m.model.clone();
    with.attention_scale = Some(0.25);
    assert_eq!(with.effective_attention_scale(), 0.25);
}

#[test]
fn cache_identity_is_computed_stable_and_sensitive_to_everything_it_covers() {
    let base = with_payloads(&[("a.safetensors", SHA)]);
    let id = base.cache_identity();
    assert_eq!(id.len(), 64);
    assert_eq!(id, base.cache_identity(), "deterministic");
    let hash = |m: &ManifestV1| m.cache_identity();
    let mut changed = base.clone();
    changed.payloads[0].sha256 = "f".repeat(64);
    assert_ne!(hash(&changed), id, "payload hash");
    let mut changed = base.clone();
    changed.target_arch = "sm_90".into();
    assert_ne!(hash(&changed), id, "target arch");
    let mut changed = base.clone();
    changed.tensor_format = TensorFormat::Fp8;
    assert_ne!(hash(&changed), id, "tensor format");
    let mut changed = base.clone();
    changed.layers[0].byte_offset += 1;
    assert_ne!(hash(&changed), id, "layout");
    let mut changed = base.clone();
    changed.model.rope_theta = 10000.0;
    assert_ne!(hash(&changed), id, "model config");
    // What does not change the computation does not change the identity.
    let mut same = base.clone();
    same.signature = Some(vec![1, 2, 3]);
    same.chat_template = None;
    same.generation.eos_token_ids = vec![1];
    assert_eq!(hash(&same), id, "signature, template and eos are outside the identity");
}

#[test]
fn sha256_matches_python_hashlib_for_a_serialised_manifest() {
    // The hash `cache_identity` rests on is the crate's own SHA-256; hashlib's answer for the same bytes is in the fixture.
    let expected: Value = serde_json::from_slice(&std::fs::read(fixtures().join("../sha256-vectors.json")).unwrap()).unwrap();
    for case in expected["cases"].as_array().unwrap() {
        let data = case["hex"].as_str().unwrap();
        let bytes: Vec<u8> = (0..data.len()).step_by(2).map(|i| u8::from_str_radix(&data[i..i + 2], 16).unwrap()).collect();
        assert_eq!(hyperion_specification::sha256::sha256_hex(&bytes), case["sha256"].as_str().unwrap());
    }
}
