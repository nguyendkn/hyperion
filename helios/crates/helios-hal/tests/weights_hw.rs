//! Hardware test (H100): upload a real model's weights through the sm90 backend and
//! prove the bytes on the device equal the bytes in the checkpoint file. Needs a
//! GPU and a package directory, so it is `#[ignore]` and gated on the `sm90`
//! feature:
//!
//! ```text
//! HYPERION_WEIGHTS_PACKAGE=/path/model.hyperion \
//!   cargo test -p helios-hal --features sm90 --test weights_hw -- --ignored --nocapture
//! ```
//!
//! `<package>.manifest.json` (from `hyperion-compile`) and `model.safetensors`
//! must sit next to the package path.
#![cfg(feature = "sm90")]

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use helios_hal::{weights_plan, ComputeBackend, HopperBackend};
use hyperion_specification::Manifest;

fn package() -> PathBuf {
    PathBuf::from(std::env::var("HYPERION_WEIGHTS_PACKAGE").expect("set HYPERION_WEIGHTS_PACKAGE to the .hyperion path"))
}

#[test]
#[ignore = "needs an H100 and a compiled package (see module docs)"]
fn device_weights_equal_file_weights() {
    let package = package();
    let Manifest::V1(manifest) =
        serde_json::from_slice(&std::fs::read(package.with_extension("manifest.json")).unwrap()).unwrap();

    let backend = HopperBackend::new().expect("an H100 must be present");
    let info = backend.initialize_device().unwrap();
    assert_eq!(info.compute_capability, "sm_90", "this test is for Hopper");

    backend.load_model_weights(&package, &manifest).expect("upload");
    let (tensors, bytes) = backend.loaded_weights();
    assert_eq!(tensors, manifest.layers.len());

    // Compare every tensor, byte for byte, against the file.
    let file = package.with_file_name("model.safetensors");
    let mut source = std::fs::File::open(&file).unwrap();
    let file_len = source.metadata().unwrap().len();
    let data_start = weights_plan::data_start(&mut source).unwrap();
    let plans = weights_plan::plan(&manifest, data_start, file_len).unwrap();
    for plan in &plans {
        let mut expected = vec![0u8; plan.len as usize];
        source.seek(SeekFrom::Start(plan.file_offset)).unwrap();
        source.read_exact(&mut expected).unwrap();
        let actual = backend.read_back_weight(&plan.name).unwrap();
        assert!(actual == expected, "{}: device bytes differ from the file", plan.name);
    }
    println!("verified {tensors} tensors, {bytes} bytes on {} ({} MiB VRAM)", info.compute_capability, info.total_vram_bytes >> 20);

    backend.shutdown().unwrap();
}

#[test]
#[ignore = "needs an H100"]
fn a_missing_weights_file_is_a_clean_error_not_a_panic() {
    let Manifest::V1(manifest) = serde_json::from_slice(
        &std::fs::read(package().with_extension("manifest.json")).unwrap(),
    )
    .unwrap();
    let backend = HopperBackend::new().expect("an H100 must be present");
    let nowhere = std::env::temp_dir().join("hyperion-no-such-dir").join("model.hyperion");
    let err = backend.load_model_weights(&nowhere, &manifest).unwrap_err();
    assert!(err.to_string().contains("open weights"), "{err}");
    backend.shutdown().unwrap();
}
