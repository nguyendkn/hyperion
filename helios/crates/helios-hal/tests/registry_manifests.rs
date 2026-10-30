#![cfg(feature = "sm90")]
//! The runtime learns a model's constants from its manifest alone. For every real model in the registry
//! (`testdata/models.json`), Ignition's test `test_model_constants.py` writes `constants.manifest.json` beside the
//! downloaded `config.json`; this test reads that manifest the way the loader does and checks that
//! (1) it parses under the current contract, (2) the dimensions the forward pass would use equal the model's
//! `config.json`, (3) a model the sm90 forward pass does not implement (QK-norm: Qwen3) is a typed refusal.
//! Without the downloaded files it skips visibly; `HYPERION_REQUIRE_MODEL_FIXTURES=1` makes that a failure.

use std::path::PathBuf;

use helios_hal::{Qwen2Dims, UnsupportedModel};
use hyperion_specification::Manifest;
use serde_json::Value;

fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../testdata")
}

#[test]
fn dimensions_come_from_the_manifest_and_equal_the_models_config() {
    let registry: Value = serde_json::from_slice(&std::fs::read(testdata().join("models.json")).unwrap()).unwrap();
    let (mut checked, mut refused) = (0, 0);
    for id in registry["models"].as_object().unwrap().keys() {
        let dir = testdata().join("models").join(id);
        let (manifest_path, config_path) = (dir.join("constants.manifest.json"), dir.join("config.json"));
        if !manifest_path.is_file() || !config_path.is_file() {
            let message = format!("{id}: run scripts/fetch-model-fixtures.py {id}, then pytest ignition/tests/test_model_constants.py");
            assert!(std::env::var_os("HYPERION_REQUIRE_MODEL_FIXTURES").is_none(), "{message}");
            eprintln!("SKIP {message}");
            continue;
        }
        let Manifest::V1(manifest) = Manifest::from_json_bytes(&std::fs::read(&manifest_path).unwrap()).unwrap_or_else(|e| panic!("{id}: {e}"));
        let config: Value = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        let int = |k: &str| config[k].as_u64().unwrap_or_else(|| panic!("{id}: config.json {k}")) as usize;
        if manifest.model.qk_norm {
            assert_eq!(Qwen2Dims::from_manifest(&manifest.model), Err(UnsupportedModel::QkNorm), "{id} must be refused");
            refused += 1;
            continue;
        }
        let dims = Qwen2Dims::from_manifest(&manifest.model).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!((dims.layers, dims.hidden, dims.intermediate), (int("num_hidden_layers"), int("hidden_size"), int("intermediate_size")), "{id}");
        assert_eq!((dims.heads, dims.kv_heads, dims.vocab), (int("num_attention_heads"), int("num_key_value_heads"), int("vocab_size")), "{id}");
        assert_eq!(dims.rope_theta, config["rope_theta"].as_f64().unwrap(), "{id}");
        assert_eq!(dims.rms_eps, config["rms_norm_eps"].as_f64().unwrap() as f32, "{id}");
        assert_eq!(dims.tied_head, config["tie_word_embeddings"] == true, "{id}");
        let eos = config["eos_token_id"].as_u64().unwrap() as u32;
        assert!(manifest.generation.eos_token_ids.contains(&eos), "{id}: the manifest's eos list holds config.json's");
        checked += 1;
    }
    eprintln!("registry manifests: {checked} accepted, {refused} refused");
}
