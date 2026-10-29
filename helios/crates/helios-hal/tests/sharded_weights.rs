//! Reading the tensors of a checkpoint split over several safetensors shards, with no GPU: the plan
//! must send every tensor to its own shard at the right offset. The fixture is a tiny random Qwen2
//! saved by Hugging Face's `save_pretrained(max_shard_size=...)` (`scripts/gen-sharded-fixture.py`),
//! its manifest is what Ignition compiled from it, and `expected.json` holds each tensor's first and
//! last 16 bytes taken from the tensors themselves, not from any parser of ours.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use helios_hal::weights_plan::{self, PayloadFile};
use hyperion_specification::Manifest;
use serde_json::Value;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../testdata/synthetic/qwen2-tiny-sharded")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn load() -> (hyperion_specification::v1::Manifest, Vec<PayloadFile>, Value) {
    let Manifest::V1(manifest) = serde_json::from_slice(&std::fs::read(fixture().join("model.manifest.json")).unwrap()).unwrap();
    manifest.validate_payloads().expect("the compiled payload table is valid");
    let facts = manifest
        .payload_files()
        .into_iter()
        .map(|name| {
            let mut file = File::open(fixture().join(name)).unwrap();
            let file_len = file.metadata().unwrap().len();
            PayloadFile { data_start: weights_plan::data_start(&mut file).unwrap(), file_len }
        })
        .collect();
    let expected = serde_json::from_slice(&std::fs::read(fixture().join("expected.json")).unwrap()).unwrap();
    (manifest, facts, expected)
}

#[test]
fn every_tensor_of_a_real_sharded_checkpoint_is_read_from_its_own_shard() {
    let (manifest, facts, expected) = load();
    assert!(manifest.payloads.len() >= 3, "the fixture must really be sharded");
    let plans = weights_plan::plan(&manifest, &facts).expect("the plan accepts the real files");
    assert_eq!(plans.len(), expected.as_object().unwrap().len());
    let files = manifest.payload_files();
    let mut payloads_used = std::collections::HashSet::new();
    for plan in &plans {
        // The manifest drops the `model.` prefix; `lm_head.weight` never had one.
        let hf_name = if plan.name.starts_with("lm_head") { plan.name.clone() } else { format!("model.{}", plan.name) };
        let want = &expected[&hf_name];
        assert_eq!(files[plan.payload], want["shard"].as_str().unwrap(), "{}: wrong shard", plan.name);
        assert_eq!(plan.len, want["length"].as_u64().unwrap(), "{}: wrong length", plan.name);
        let mut file = File::open(fixture().join(files[plan.payload])).unwrap();
        let mut data = vec![0u8; plan.len as usize];
        file.seek(SeekFrom::Start(plan.file_offset)).unwrap();
        file.read_exact(&mut data).unwrap();
        assert_eq!(hex(&data[..16]), want["first16"].as_str().unwrap(), "{}: first bytes", plan.name);
        assert_eq!(hex(&data[data.len() - 16..]), want["last16"].as_str().unwrap(), "{}: last bytes", plan.name);
        payloads_used.insert(plan.payload);
    }
    assert_eq!(payloads_used.len(), manifest.payloads.len(), "every shard holds at least one tensor");
}

#[test]
fn a_shard_whose_size_differs_from_the_payload_table_is_refused() {
    let (manifest, mut facts, _) = load();
    facts[2].file_len -= 1;
    let error = weights_plan::plan(&manifest, &facts).unwrap_err().to_string();
    assert!(error.contains(&manifest.payloads[2].file), "{error}");
}

#[test]
fn a_payload_name_that_could_leave_the_package_is_refused_before_any_path_is_built() {
    let (mut manifest, _, _) = load();
    manifest.payloads[1].file = "../../etc/passwd".into();
    assert!(manifest.validate_payloads().unwrap_err().to_string().contains("not a bare file name"));
}
