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

/// First real op: a cuBLAS bf16 matrix multiply on the real `q_proj` weight of layer 0
/// (`[out, in]` from the manifest), checked against the f64 CPU reference.
#[test]
#[ignore = "needs an H100 and a compiled package (see module docs)"]
fn linear_on_a_real_weight_matches_the_cpu_reference() {
    use helios_hal::reference::{bf16_to_f32, f32_to_bf16, linear};

    const NAME: &str = "layers.0.self_attn.q_proj.weight";
    let package = package();
    let Manifest::V1(manifest) =
        serde_json::from_slice(&std::fs::read(package.with_extension("manifest.json")).unwrap()).unwrap();
    let layer = manifest.layers.iter().find(|l| l.name == NAME).expect("q_proj of layer 0 in the manifest");
    let (out_features, in_features) = (layer.tensor.shape[0], layer.tensor.shape[1]);

    // The weight as the file holds it, converted for the reference.
    let file = package.with_file_name("model.safetensors");
    let mut source = std::fs::File::open(&file).unwrap();
    let data_start = weights_plan::data_start(&mut source).unwrap();
    let mut raw = vec![0u8; layer.byte_length as usize];
    source.seek(SeekFrom::Start(data_start + layer.byte_offset)).unwrap();
    source.read_exact(&mut raw).unwrap();
    let w: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    assert!(w.iter().any(|&b| bf16_to_f32(b) != 0.0), "weight must not be all zeros");

    // Deterministic inputs that are exact in bf16 (small integers / 64).
    let tokens = 3;
    let x: Vec<u16> = (0..tokens * in_features)
        .map(|i| f32_to_bf16(((i * 37 % 101) as f32 - 50.0) / 64.0))
        .collect();

    let backend = HopperBackend::new().expect("an H100 must be present");
    backend.load_model_weights(&package, &manifest).expect("upload");
    let got = backend.linear_bf16_f32(NAME, &x, tokens, out_features, in_features).expect("gemm");
    let want = linear(&w, &x, tokens, out_features, in_features);

    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let max_err = got.iter().zip(&want).fold(0.0f32, |m, (g, r)| m.max((g - r).abs()));
    println!("linear [{tokens}x{in_features}] x [{out_features}x{in_features}]^T: max abs err {max_err:e}, output scale {scale:.4}, relative {:e}", max_err / scale);
    assert!(max_err / scale < 1e-4, "GPU result deviates from the CPU reference: {max_err} vs scale {scale}");

    // A wrongly sized request is a clean error, not a device fault.
    let err = backend.linear_bf16_f32(NAME, &x, tokens, out_features + 1, in_features).unwrap_err();
    assert!(err.to_string().contains("bf16 needs"), "{err}");
    backend.shutdown().unwrap();
}

/// First custom kernel: RMSNorm (`kernels/rmsnorm.cu`, loaded as PTX through the Driver API)
/// on the real `input_layernorm` weight of layer 0, compared with the CPU reference. The two
/// reductions add in different orders, so a result may differ by one bf16 step; anything
/// further apart is a bug.
#[test]
#[ignore = "needs an H100 and a compiled package (see module docs)"]
fn rmsnorm_kernel_matches_the_cpu_reference() {
    use helios_hal::reference::{bf16_to_f32, f32_to_bf16, rmsnorm};

    const NAME: &str = "layers.0.input_layernorm.weight";
    const EPS: f32 = 1e-6; // rms_norm_eps in the Qwen2.5-0.5B config
    let package = package();
    let Manifest::V1(manifest) =
        serde_json::from_slice(&std::fs::read(package.with_extension("manifest.json")).unwrap()).unwrap();
    let layer = manifest.layers.iter().find(|l| l.name == NAME).expect("input_layernorm of layer 0");
    let hidden = layer.tensor.shape[0];

    let file = package.with_file_name("model.safetensors");
    let mut source = std::fs::File::open(&file).unwrap();
    let data_start = weights_plan::data_start(&mut source).unwrap();
    let mut raw = vec![0u8; layer.byte_length as usize];
    source.seek(SeekFrom::Start(data_start + layer.byte_offset)).unwrap();
    source.read_exact(&mut raw).unwrap();
    let w: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // Three rows at very different scales: the normalisation must make them comparable.
    let tokens = 3;
    let scales = [1.0f32, 50.0, 0.02];
    let x: Vec<u16> = (0..tokens * hidden)
        .map(|i| f32_to_bf16(((i * 37 % 101) as f32 - 50.0) / 8.0 * scales[i / hidden]))
        .collect();

    let backend = HopperBackend::new().expect("an H100 must be present");
    backend.load_model_weights(&package, &manifest).expect("upload");
    let got = backend.rmsnorm_bf16(NAME, &x, tokens, hidden, EPS).expect("rmsnorm kernel");
    let want = rmsnorm(&x, &w, tokens, hidden, EPS);

    assert_eq!(got.len(), want.len());
    let mut exact = 0;
    let mut worst = 0.0f32;
    for (g, r) in got.iter().zip(&want) {
        let (g, r) = (bf16_to_f32(*g), bf16_to_f32(*r));
        if g == r {
            exact += 1;
        }
        // One bf16 step is 2^-7 of the value (plus a tiny absolute floor for values near zero).
        let allowed = r.abs() / 128.0 + 1e-6;
        worst = worst.max((g - r).abs() / allowed);
        assert!((g - r).abs() <= 2.0 * allowed, "kernel {g} vs reference {r}");
    }
    let fraction = exact as f64 / got.len() as f64;
    println!("rmsnorm [{tokens}x{hidden}]: {:.2}% bit-identical to the CPU reference, worst error {worst:.2}x one bf16 step", fraction * 100.0);
    assert!(fraction > 0.97, "only {fraction} of elements are bit-identical");

    // Wrong size is a clean error.
    let err = backend.rmsnorm_bf16(NAME, &x, tokens, hidden + 1, EPS).unwrap_err();
    assert!(err.to_string().contains("expected tokens*hidden"), "{err}");
    backend.shutdown().unwrap();
}
