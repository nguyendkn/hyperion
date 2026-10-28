//! Manifest contract tests: wire format, strictness, and fixtures shared with
//! Ignition (Python). `specification/fixtures/v1/*.json` are parsed by both
//! sides; `rust-sample.json` is written by this test, the Python-produced ones
//! by `ignition/tests`. Set `UPDATE_FIXTURES=1` to rewrite the Rust-owned files.

use std::collections::HashMap;
use std::path::PathBuf;

use hyperion_specification::v1::{LayerMapping, Manifest as ManifestV1, ModelConfig, TensorMetadata, TokenizerConfig};
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
        model: ModelConfig {
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
            byte_offset: 0,
            byte_length: 33_554_432,
            tensor: TensorMetadata { dtype: "bf16".into(), shape: vec![4096, 4096] },
        }],
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
