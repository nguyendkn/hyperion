//! Manifest contract tests: wire format, strictness, and fixtures shared with
//! Ignition (Python). `specification/fixtures/v1/*.json` are parsed by both
//! sides; `rust-sample.json` is written by this test, the Python-produced ones
//! by `ignition/tests`. Set `UPDATE_FIXTURES=1` to rewrite the Rust-owned files.

use std::collections::HashMap;
use std::path::PathBuf;

use hyperion_specification::v1::{
    is_bare_file_name, AttentionKind, LayerMapping, Manifest as ManifestV1, ModelConfig, Payload, TensorFormat, TensorMetadata,
    TokenizerConfig, DEFAULT_PAYLOAD_FILE,
};
use hyperion_specification::Manifest;
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
        },
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
    for good in ["model.safetensors", "model-00001-of-00004.safetensors", "weights.v2.bin"] {
        assert!(is_bare_file_name(good), "{good}");
    }
    let too_long = "a".repeat(256);
    for bad in ["", ".", "..", "../x", "a/b", "a\\b", "/abs", "C:evil", "stream:name", "x\0y", too_long.as_str()] {
        assert!(!is_bare_file_name(bad), "{bad:?} must be refused");
    }
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
