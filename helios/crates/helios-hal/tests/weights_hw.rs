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

/// The model dimensions, taken from the compiled package's manifest, the only place the runtime reads them.
fn qwen2_dims(package: &std::path::Path) -> helios_hal::Qwen2Dims {
    let Manifest::V1(manifest) = Manifest::from_json_bytes(&std::fs::read(package.with_extension("manifest.json")).unwrap())
        .unwrap_or_else(|e| panic!("manifest of {}: {e}", package.display()));
    helios_hal::Qwen2Dims::from_manifest_allowing(&manifest.model, allow_unverified_qk_norm()).expect("a model the sm90 forward pass implements")
}

/// `HYPERION_ALLOW_UNVERIFIED_QK_NORM=1` lets a run load a package that needs QK-norm (Qwen3) before that path has passed an
/// oracle comparison; it is the run that does the verifying.
fn allow_unverified_qk_norm() -> bool {
    std::env::var_os("HYPERION_ALLOW_UNVERIFIED_QK_NORM").is_some()
}

/// The prompt ids the timing tests repeat: the first prompt of the compact oracle when
/// `HYPERION_COMPACT_ORACLE` is set, otherwise the single prompt of the older `HYPERION_ORACLE`.
fn seed_prompt_ids() -> Vec<u32> {
    if let Some(path) = std::env::var_os("HYPERION_COMPACT_ORACLE") {
        let oracle = helios_hal::oracle::CompactOracle::from_json(&std::fs::read(path).unwrap()).expect("compact oracle");
        return oracle.prompts[0].ids.clone();
    }
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("HYPERION_ORACLE").expect("set HYPERION_ORACLE or HYPERION_COMPACT_ORACLE")).unwrap()).unwrap();
    oracle["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
}

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
    let plans = weights_plan::plan(&manifest, &[weights_plan::PayloadFile { data_start, file_len }]).unwrap();
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

// ---- single ops from kernels/elementwise.cu, each against the CPU reference ----

fn loaded_backend() -> (PathBuf, hyperion_specification::v1::Manifest, HopperBackend) {
    // The revision the pod script packaged (scripts/pod-lib.sh sets it), so a result names its build.
    println!("BUILD revision={}", std::env::var("HYPERION_REVISION").unwrap_or_else(|_| "unknown".into()));
    let package = package();
    let Manifest::V1(manifest) = Manifest::from_json_bytes(&std::fs::read(package.with_extension("manifest.json")).unwrap())
        .unwrap_or_else(|e| panic!("manifest of {}: {e}", package.display()));
    let backend = HopperBackend::new().expect("an H100 must be present");
    if allow_unverified_qk_norm() {
        backend.allow_unverified_qk_norm();
    }
    let started = std::time::Instant::now();
    backend.load_model_weights(&package, &manifest).expect("upload");
    println!(
        "LOADED in {:.2?}; {}; decode session capacity {:?} tokens",
        started.elapsed(),
        helios_hal::cuda::loaded_libraries(),
        backend.context_capacity()
    );
    (package, manifest, backend)
}

/// Deterministic bf16 test data in roughly [-range, range], exact in bf16.
fn pattern(n: usize, range: f32) -> Vec<u16> {
    use helios_hal::reference::f32_to_bf16;
    (0..n).map(|i| f32_to_bf16(((i * 37 % 101) as f32 - 50.0) / 50.0 * range)).collect()
}

/// Share of identical elements and the worst error in bf16 steps (1.0 = one step).
fn compare(got: &[u16], want: &[u16]) -> (f64, f32) {
    use helios_hal::reference::bf16_to_f32;
    assert_eq!(got.len(), want.len());
    let mut exact = 0;
    let mut worst = 0.0f32;
    for (g, r) in got.iter().zip(want) {
        let (g, r) = (bf16_to_f32(*g), bf16_to_f32(*r));
        if g == r {
            exact += 1;
        }
        worst = worst.max((g - r).abs() / (r.abs() / 128.0 + 1e-6));
    }
    (exact as f64 / got.len() as f64, worst)
}

#[test]
#[ignore = "needs an H100 and a compiled package (see module docs)"]
fn embedding_gather_equals_the_rows_of_the_file() {
    const NAME: &str = "embed_tokens.weight";
    let (package, manifest, backend) = loaded_backend();
    let layer = manifest.layers.iter().find(|l| l.name == NAME).unwrap();
    let hidden = layer.tensor.shape[1];
    // The prompt of the HF oracle ("The capital of France is") plus the first and last ids.
    let ids: Vec<u32> = vec![785, 6722, 315, 9625, 374, 0, layer.tensor.shape[0] as u32 - 1];
    let got = backend.embedding_bf16(NAME, &ids, hidden).expect("gather");

    let mut source = std::fs::File::open(package.with_file_name("model.safetensors")).unwrap();
    let data_start = weights_plan::data_start(&mut source).unwrap();
    for (t, &id) in ids.iter().enumerate() {
        let mut row = vec![0u8; hidden * 2];
        source.seek(SeekFrom::Start(data_start + layer.byte_offset + id as u64 * hidden as u64 * 2)).unwrap();
        source.read_exact(&mut row).unwrap();
        let want: Vec<u16> = row.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        assert!(got[t * hidden..(t + 1) * hidden] == want[..], "row for token {id} differs from the file");
    }
    let err = backend.embedding_bf16(NAME, &[layer.tensor.shape[0] as u32], hidden).unwrap_err();
    assert!(err.to_string().contains("outside the vocabulary"), "{err}");
    println!("embedding: {} rows bit-identical to the file", ids.len());
    backend.shutdown().unwrap();
}

#[test]
#[ignore = "needs an H100 and a compiled package (see module docs)"]
fn add_and_swiglu_match_the_cpu_reference() {
    use helios_hal::reference::{add, swiglu};
    let (_, _, backend) = loaded_backend();
    let n = 4864; // intermediate_size of Qwen2.5-0.5B
    let (a, b) = (pattern(n, 8.0), pattern(n + 0, 3.0).into_iter().rev().collect::<Vec<_>>());

    let (exact, worst) = compare(&backend.add_bf16(&a, &b).unwrap(), &add(&a, &b));
    println!("add [{n}]: {:.2}% bit-identical, worst {worst:.2} bf16 steps", exact * 100.0);
    assert!(exact == 1.0, "bf16 addition is exactly rounded; the kernel must match bit for bit");

    // SwiGLU uses expf on the device vs exp on the host: allow rare one-step differences.
    let (exact, worst) = compare(&backend.swiglu_bf16(&a, &b).unwrap(), &swiglu(&a, &b));
    println!("swiglu [{n}]: {:.2}% bit-identical, worst {worst:.2} bf16 steps", exact * 100.0);
    assert!(exact > 0.97 && worst <= 2.0, "swiglu deviates: {exact} identical, worst {worst}");

    assert!(backend.add_bf16(&a, &b[..n - 1]).unwrap_err().to_string().contains("differ in length"));
    backend.shutdown().unwrap();
}

#[test]
#[ignore = "needs an H100 and a compiled package (see module docs)"]
fn rope_matches_the_cpu_reference() {
    use helios_hal::reference::{rope, rope_tables};
    let (_, _, backend) = loaded_backend();
    // Qwen2.5-0.5B: 14 query heads of 64, rope_theta 1e6.
    let (heads, head_dim, theta) = (14, 64, 1_000_000.0);
    let (cos, sin) = rope_tables(128, head_dim, theta);
    let positions = [0u32, 5, 100];
    let x = pattern(positions.len() * heads * head_dim, 4.0);

    let got = backend.rope_bf16(&x, &positions, heads, head_dim, &cos, &sin).unwrap();
    let want = rope(&x, &positions, heads, head_dim, &cos, &sin);
    let (exact, worst) = compare(&got, &want);
    println!("rope [{}x{heads}x{head_dim}]: {:.2}% bit-identical, worst {worst:.2} bf16 steps", positions.len(), exact * 100.0);
    assert!(exact == 1.0, "the kernel uses the same bf16 arithmetic and tables as the reference");
    // position 0 is the identity
    assert!(got[..heads * head_dim] == x[..heads * head_dim]);

    let err = backend.rope_bf16(&x, &[0, 5, 128], heads, head_dim, &cos, &sin).unwrap_err();
    assert!(err.to_string().contains("outside the table"), "{err}");
    backend.shutdown().unwrap();
}

#[test]
#[ignore = "needs an H100 and a compiled package (see module docs)"]
fn attention_matches_the_cpu_reference_for_prefill_and_decode() {
    use helios_hal::reference::{attention, f32_to_bf16};
    let (_, _, backend) = loaded_backend();
    // Qwen2.5-0.5B: 14 query heads share 2 KV heads of 64 dims.
    let (heads, kv_heads, d) = (14usize, 2usize, 64usize);
    let data = |n: usize, mul: usize, range: f32| -> Vec<u16> {
        (0..n).map(|i| f32_to_bf16(((i * mul % 101) as f32 - 50.0) / 50.0 * range)).collect()
    };
    let check = |name: &str, tokens: usize, seq: usize, q_start: usize| {
        let q = data(tokens * heads * d, 37, 2.0);
        let k = data(seq * kv_heads * d, 53, 2.0);
        let v = data(seq * kv_heads * d, 29, 3.0);
        let got = backend.attention_bf16(&q, &k, &v, tokens, seq, heads, kv_heads, d, q_start).expect(name);
        let want = attention(&q, &k, &v, tokens, seq, heads, kv_heads, d, q_start);
        let (exact, worst) = compare(&got, &want);
        println!(
            "attention {name} [tokens {tokens}, seq {seq}, q_start {q_start}]: {:.2}% bit-identical, worst {worst:.2} bf16 steps",
            exact * 100.0
        );
        assert!(exact > 0.97 && worst <= 2.0, "{name}: {exact} identical, worst {worst}");
    };

    check("prefill", 5, 5, 0);
    check("decode", 1, 6, 5);
    check("long", 3, 300, 297);

    // Decode of the last token must equal the last row of the prefill of the same cache.
    let (tokens, seq) = (6, 6);
    let q = data(tokens * heads * d, 37, 2.0);
    let k = data(seq * kv_heads * d, 53, 2.0);
    let v = data(seq * kv_heads * d, 29, 3.0);
    let full = backend.attention_bf16(&q, &k, &v, tokens, seq, heads, kv_heads, d, 0).unwrap();
    let last_q = &q[(tokens - 1) * heads * d..];
    let one = backend.attention_bf16(last_q, &k, &v, 1, seq, heads, kv_heads, d, seq - 1).unwrap();
    assert!(full[(tokens - 1) * heads * d..] == one[..], "decode row differs from the prefill row");

    let err = backend.attention_bf16(&q, &k, &v, tokens, seq, heads, kv_heads, d, 1).unwrap_err();
    assert!(err.to_string().contains("outside the cache"), "{err}");
    let (big_k, big_v) = (data(9000 * kv_heads * d, 53, 2.0), data(9000 * kv_heads * d, 29, 3.0));
    let err = backend.attention_bf16(&q[..heads * d], &big_k, &big_v, 1, 9000, heads, kv_heads, d, 8999).unwrap_err();
    assert!(err.to_string().contains("limit"), "{err}");
    backend.shutdown().unwrap();
}

/// The decisive test: the whole model on the H100 against the Hugging Face reference. The
/// criteria and tolerance are the ones declared in the oracle file BEFORE any Hyperion run
/// (`scripts/gen-oracle.py`): max abs logit difference within `tolerance_max_abs`, the same
/// argmax, and at least 4 of the oracle's top-5 in ours.
///
/// Needs `HYPERION_ORACLE=/path/qwen2.5-0.5b.json` next to the usual package variable.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn first_forward_matches_the_hf_oracle() {

    let package = package();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("HYPERION_ORACLE").expect("set HYPERION_ORACLE")).unwrap()).unwrap();
    let dims = qwen2_dims(&package);

    let ids: Vec<u32> = oracle["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let want: Vec<f32> = oracle["logits"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let tolerance = oracle["tolerance_max_abs"].as_f64().unwrap() as f32;
    let want_top5: Vec<usize> = oracle["next_token_top5"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();

    let (_, _, backend) = loaded_backend();
    let started = std::time::Instant::now();
    let got = backend.forward_prefill_logits(&dims, &ids).expect("forward pass");
    let elapsed = started.elapsed();
    assert_eq!(got.len(), want.len());

    let max_abs = got.iter().zip(&want).fold(0.0f32, |m, (g, r)| m.max((g - r).abs()));
    let top = |v: &[f32]| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx.truncate(5);
        idx
    };
    let got_top5 = top(&got);
    let overlap = want_top5.iter().filter(|t| got_top5.contains(t)).count();
    println!(
        "forward {} layers over {} tokens in {:.2?}: max abs logit diff {max_abs:.4} (tolerance {tolerance}), argmax ours {} vs oracle {}, top-5 overlap {overlap}/5, ours {got_top5:?}, oracle {want_top5:?}",
        dims.layers,
        ids.len(),
        elapsed,
        got_top5[0],
        want_top5[0],
    );
    assert_eq!(got_top5[0], want_top5[0], "argmax must equal the oracle's");
    assert!(overlap >= 4, "at least 4 of the oracle's top-5 must be in ours");
    assert!(max_abs <= tolerance, "max abs logit difference {max_abs} exceeds the declared tolerance {tolerance}");
    backend.shutdown().unwrap();
}

/// Decode path: greedy generation with the K/V cache against Hugging Face's `generate`. The
/// criterion was declared in the oracle file before any Hyperion run: the first 8 generated ids
/// must equal ours; later ones may diverge when two bf16 logits are nearly tied.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn greedy_generation_matches_hf_generate() {

    let package = package();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("HYPERION_ORACLE").expect("set HYPERION_ORACLE")).unwrap()).unwrap();
    let dims = qwen2_dims(&package);
    let ids = |key: &str| -> Vec<u32> { oracle[key].as_array().expect(key).iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let (prompt, want) = (ids("prompt_ids"), ids("generated_ids"));

    let (_, _, backend) = loaded_backend();
    let started = std::time::Instant::now();
    let got = backend.generate_greedy(&dims, &prompt, want.len(), &[]).expect("generation");
    let elapsed = started.elapsed();
    let prefix = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
    println!(
        "generated {} tokens in {:.2?} ({:.0} ms/token incl. prefill): first {prefix}/{} equal to HF; ours {got:?}; HF {want:?}; HF text {:?}",
        got.len(),
        elapsed,
        elapsed.as_millis() as f64 / got.len() as f64,
        want.len(),
        oracle["generated_text"].as_str().unwrap_or(""),
    );
    assert!(prefix >= 8, "the first 8 generated tokens must equal HF's, only {prefix} do");
    backend.shutdown().unwrap();
}

/// The device-resident pipeline must compute exactly what the host-sliced reference path
/// (verified against Hugging Face) computes: same kernels, same rounding, so the prefill logits
/// are bit-identical, and so is a short greedy continuation.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn device_pipeline_is_bit_identical_to_the_host_path() {
    use helios_hal::SamplingConfig;

    let package = package();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("HYPERION_ORACLE").expect("set HYPERION_ORACLE")).unwrap()).unwrap();
    let dims = qwen2_dims(&package);
    let prompt: Vec<u32> = oracle["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let want: Vec<u32> = oracle["generated_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();

    let (_, _, backend) = loaded_backend();
    let host = backend.forward_prefill_logits(&dims, &prompt).expect("host path");
    let device = backend.forward_prefill_logits_device(&dims, &prompt).expect("device path");
    assert_eq!(host.len(), device.len());
    let differing = host.iter().zip(&device).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    println!("prefill logits: {differing} of {} differ between the host and the device pipeline", host.len());
    assert_eq!(differing, 0, "device pipeline must be bit-identical to the host path");

    // Speed and generation equality, device vs host vs Hugging Face.
    let started = std::time::Instant::now();
    let mut device_tokens = Vec::new();
    backend
        .generate_streaming(&dims, &prompt, want.len(), &[], &SamplingConfig::greedy(), &mut |t| device_tokens.push(t))
        .expect("device generation");
    let device_time = started.elapsed();
    let started = std::time::Instant::now();
    let host_tokens = backend.generate_greedy_host(&dims, &prompt, want.len(), &[]).expect("host generation");
    let host_time = started.elapsed();
    println!(
        "{} tokens: device {:.2?} ({:.1} ms/token incl. prefill) vs host path {:.2?} ({:.1} ms/token); equal to HF: device {}, host {}",
        want.len(),
        device_time,
        device_time.as_secs_f64() * 1000.0 / want.len() as f64,
        host_time,
        host_time.as_secs_f64() * 1000.0 / want.len() as f64,
        device_tokens == want,
        host_tokens == want,
    );
    assert_eq!(device_tokens, host_tokens, "device and host paths must generate the same tokens");
    assert_eq!(device_tokens, want, "must equal Hugging Face's greedy continuation");
    backend.shutdown().unwrap();
}


/// A decode step replayed from a CUDA graph must produce the same logits, bit for bit, as the same
/// step launched kernel by kernel (and as the path verified against Hugging Face), at every
/// position of a generation. Also measures time per token for both modes: one machine, batch 1,
/// the whole step including the host round trip (token upload, logits download).
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn graph_decode_is_bit_identical_to_direct_launches() {
    use helios_hal::{DecodeMode, SamplingConfig};

    let package = package();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("HYPERION_ORACLE").expect("set HYPERION_ORACLE")).unwrap()).unwrap();
    let dims = qwen2_dims(&package);
    let prompt: Vec<u32> = oracle["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let (_, _, backend) = loaded_backend();

    // Bit-identity over 40 steps (greedy; the same tokens feed both runs while they agree).
    let steps = 40;
    let (mut direct, mut graph) = (Vec::new(), Vec::new());
    backend
        .generate_traced(&dims, &prompt, steps, &[], &SamplingConfig::greedy(), DecodeMode::Direct, Some(&mut direct))
        .expect("direct generation");
    backend
        .generate_traced(&dims, &prompt, steps, &[], &SamplingConfig::greedy(), DecodeMode::Graph, Some(&mut graph))
        .expect("graph generation");
    assert_eq!(direct.len(), steps);
    assert_eq!(direct.len(), graph.len());
    for (i, (a, b)) in direct.iter().zip(&graph).enumerate() {
        let differing = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert_eq!(differing, 0, "step {i}: {differing} of {} logits differ between direct launches and the graph", a.len());
    }
    println!("graph vs direct: all {steps} steps bit-identical ({} logits each)", direct[0].len());

    let (direct, graph) = measure_modes(&backend, &dims, &prompt, SAMPLES);
    println!(
        "decode per token (batch 1, one machine; {SAMPLES} interleaved windows of {WINDOW_STEPS} steps, ms, median [p10, p90]): direct launches gpu {} wall {}; CUDA graph gpu {} wall {} ({:.2}x by gpu medians)",
        direct.gpu,
        direct.wall,
        graph.gpu,
        graph.wall,
        direct.gpu.median / graph.gpu.median
    );
    backend.shutdown().unwrap();
}

/// Where a decode step's time goes (single GPU, batch 1): GPU milliseconds per group of kernels
/// from CUDA events, and host milliseconds per phase. `HYPERION_PROFILE_ONLY=direct|graph` runs
/// one mode only, so an external profiler (nsys) sees a single loop of decode steps.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn profile_decode_steps() {
    use helios_hal::DecodeMode;

    let package = package();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("HYPERION_ORACLE").expect("set HYPERION_ORACLE")).unwrap()).unwrap();
    let dims = qwen2_dims(&package);
    let prompt: Vec<u32> = oracle["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let (_, _, backend) = loaded_backend();
    let only = std::env::var("HYPERION_PROFILE_ONLY").ok();
    let steps = 128usize;
    for (name, mode) in [("direct", DecodeMode::Direct), ("graph", DecodeMode::Graph)] {
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        backend.generate_profiled(&dims, &prompt, 16, mode).expect("warm-up");
        let started = std::time::Instant::now();
        let profile = backend.generate_profiled(&dims, &prompt, steps + 1, mode).expect("profiled generation");
        let wall = started.elapsed().as_secs_f64() * 1000.0;
        let n = profile.steps as f64;
        println!("== {name}: {} profiled decode steps, context {}..{} tokens", profile.steps, prompt.len(), prompt.len() + steps);
        let mut gpu_total = 0.0;
        for (group, ms) in &profile.gpu_ms_by_group {
            println!("   gpu  {group:<12} {:>8.3} ms/step", ms / n);
            gpu_total += ms / n;
        }
        if !profile.gpu_ms_by_group.is_empty() {
            println!("   gpu  {:<12} {gpu_total:>8.3} ms/step (sum of marked intervals)", "TOTAL");
        }
        let host = |label: &str, ms: f64| println!("   host {label:<12} {:>8.3} ms/step", ms / n);
        host("upload", profile.host_upload_ms);
        host("issue", profile.host_issue_ms);
        host("wait", profile.host_wait_ms);
        host("read logits", profile.host_read_ms);
        host("sample", profile.host_sample_ms);
        println!("   wall (incl. prompt) {:.3} ms for {} tokens", wall, steps + 1);
    }
    backend.shutdown().unwrap();
}

fn host_argmax(values: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best as u32
}

/// Greedy decoding with the token chain on the device (argmax kernel, 4-byte token ids to the
/// host) must pick exactly the tokens the host picks from the downloaded logits, in both decode
/// modes, and must stop at an end-of-sequence token with that token emitted last.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn device_token_chain_matches_host_argmax() {
    use helios_hal::{DecodeMode, SamplingConfig};

    let package = package();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("HYPERION_ORACLE").expect("set HYPERION_ORACLE")).unwrap()).unwrap();
    let dims = qwen2_dims(&package);
    let prompt: Vec<u32> = oracle["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let (_, _, backend) = loaded_backend();
    let greedy = SamplingConfig::greedy();

    // The reference: the host picks each token from the logits of the logits-returning path.
    let steps = 40;
    let mut trace = Vec::new();
    backend.generate_traced(&dims, &prompt, steps, &[], &greedy, DecodeMode::Direct, Some(&mut trace)).expect("reference");
    let want: Vec<u32> = trace.iter().map(|l| host_argmax(l)).collect();
    assert_eq!(want.len(), steps);

    for mode in [DecodeMode::Direct, DecodeMode::Graph] {
        let mut got = Vec::new();
        let stop = backend.generate_mode(&dims, &prompt, steps, &[], &greedy, mode, &mut |t| got.push(t)).expect("device chain");
        assert_eq!(got, want, "{mode:?}: tokens differ from the host argmax");
        assert_eq!(stop, helios_hal::traits::StopReason::MaxTokens);
        println!("{mode:?}: {steps} tokens identical to the host argmax");

        // End of sequence: stop at the 11th token, which is emitted last.
        let eos = want[10];
        let first = want.iter().position(|&t| t == eos).unwrap();
        let mut got = Vec::new();
        let stop = backend.generate_mode(&dims, &prompt, steps, &[eos], &greedy, mode, &mut |t| got.push(t)).expect("device chain with eos");
        assert_eq!(got, want[..=first].to_vec(), "{mode:?}: must stop right after the end-of-sequence token");
        assert_eq!(stop, helios_hal::traits::StopReason::EndOfSequence);
    }
    // One token, and none.
    let mut one = Vec::new();
    backend.generate_mode(&dims, &prompt, 1, &[], &greedy, DecodeMode::Graph, &mut |t| one.push(t)).unwrap();
    assert_eq!(one, want[..1].to_vec());
    backend.generate_mode(&dims, &prompt, 0, &[], &greedy, DecodeMode::Graph, &mut |_| panic!("no token expected")).unwrap();
    backend.shutdown().unwrap();
}

/// How decode time and the attention share grow with the context length (single GPU, batch 1).
/// The prompt is the oracle prompt repeated, so the logits are meaningless; only timing counts.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn decode_time_vs_context() {
    use helios_hal::DecodeMode;

    let package = package();
    let dims = qwen2_dims(&package);
    let seed: Vec<u32> = seed_prompt_ids();
    let (_, _, backend) = loaded_backend();
    for context in contexts_from_env(&[5, 1024, 4096, 8000, 16000]) {
        let prompt: Vec<u32> = (0..context).map(|i| seed[i % seed.len()]).collect();
        let (direct, graph) = measure_modes(&backend, &dims, &prompt, SAMPLES);
        // GPU time per group over 32 steps at this context (direct, logits path).
        let profile = backend.generate_profiled(&dims, &prompt, 33, DecodeMode::Direct).expect("profile");
        let n = profile.steps as f64;
        let total: f64 = profile.gpu_ms_by_group.iter().map(|(_, ms)| ms / n).sum();
        let attention = profile.gpu_ms_by_group.iter().find(|(g, _)| *g == "attention").map(|(_, ms)| ms / n).unwrap_or(0.0);
        println!(
            "context {:>5}: direct gpu {} wall {}; graph gpu {} wall {} (ms/token, median [p10, p90] of {SAMPLES} windows of {WINDOW_STEPS} steps); attention {attention:.3} of {total:.3} ms/step ({:.0}%) in the profiled direct path",
            prompt.len(),
            direct.gpu,
            direct.wall,
            graph.gpu,
            graph.wall,
            100.0 * attention / total
        );
    }
    backend.shutdown().unwrap();
}

/// The context lengths a timing test uses: `HYPERION_CONTEXTS=5,1024,4096` overrides the defaults (a smaller table for an A/B
/// run of several builds; the measurement protocol itself, windows and warm-up, is not affected).
fn contexts_from_env(defaults: &[usize]) -> Vec<usize> {
    match std::env::var("HYPERION_CONTEXTS") {
        Ok(list) => list.split(',').map(|v| v.trim().parse().unwrap_or_else(|_| panic!("HYPERION_CONTEXTS: `{v}` is not a number"))).collect(),
        Err(_) => defaults.to_vec(),
    }
}

/// Decode steps inside each timed window, and runs per configuration.
const WINDOW_STEPS: usize = 128;
const SAMPLES: usize = 10;

/// Spread of repeated measurements, in milliseconds: median with the 10th and 90th percentile.
struct Spread {
    median: f64,
    p10: f64,
    p90: f64,
}

impl std::fmt::Display for Spread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3} [p10 {:.3}, p90 {:.3}]", self.median, self.p10, self.p90)
    }
}

fn spread(mut values: Vec<f64>) -> Spread {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    Spread { median: at(0.5), p10: at(0.1), p90: at(0.9) }
}

/// Per-token time of one mode: GPU time between two events and host time over the same window.
struct Measured {
    gpu: Spread,
    wall: Spread,
}

/// Direct launches and graph replay measured alternately inside this one process, `samples` runs
/// each after a warm-up run each. Every run times a window of `WINDOW_STEPS` decode steps with CUDA
/// events (no subtraction of two runs; the prompt and the graph capture are outside the window), so
/// drift of the GPU clock or of neighbours on the node touches both modes alike.
fn measure_modes(backend: &helios_hal::HopperBackend, dims: &helios_hal::Qwen2Dims, prompt: &[u32], samples: usize) -> (Measured, Measured) {
    use helios_hal::DecodeMode::{Direct, Graph};
    let run = |mode| backend.generate_timed(dims, prompt, WINDOW_STEPS, mode).expect("timed generation");
    let _ = (run(Direct), run(Graph));
    let (mut direct, mut graph) = (Vec::new(), Vec::new());
    for _ in 0..samples {
        direct.push(run(Direct));
        graph.push(run(Graph));
    }
    let measured = |runs: Vec<helios_hal::DecodeTiming>| Measured {
        gpu: spread(runs.iter().map(|t| t.gpu_ms_per_token()).collect()),
        wall: spread(runs.iter().map(|t| t.wall_ms_per_token()).collect()),
    };
    (measured(direct), measured(graph))
}

/// One timed window per mode at `HYPERION_MEASURE_CONTEXT` tokens of context (default 5), printed
/// as `SAMPLE context=N direct=x graph=y` (GPU ms per token). A script runs it many times,
/// alternating between two builds, to compare builds under the same node conditions.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn measure_decode_sample() {
    use helios_hal::DecodeMode::{Direct, Graph};

    let package = package();
    let dims = qwen2_dims(&package);
    let seed: Vec<u32> = seed_prompt_ids();
    let context: usize = std::env::var("HYPERION_MEASURE_CONTEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let prompt: Vec<u32> = (0..context).map(|i| seed[i % seed.len()]).collect();
    let (_, _, backend) = loaded_backend();
    let run = |mode| backend.generate_timed(&dims, &prompt, WINDOW_STEPS, mode).expect("timed generation");
    let _ = (run(Direct), run(Graph));
    let (direct, graph) = (run(Direct), run(Graph));
    println!(
        "SAMPLE context={context} direct={:.4} graph={:.4} direct_wall={:.4} graph_wall={:.4}",
        direct.gpu_ms_per_token(),
        graph.gpu_ms_per_token(),
        direct.wall_ms_per_token(),
        graph.wall_ms_per_token()
    );
    backend.shutdown().unwrap();
}

/// Start-up cost of a fresh process: weight upload, then the first generation (which loads the
/// kernel modules, so the driver compiles PTX unless it finds a cached binary, and creates the
/// cuBLAS handle) against a second one. Run with `CUDA_CACHE_DISABLE=1` for the case of an empty
/// driver cache, which is what an ephemeral pod always has.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn cold_start_timing() {
    use helios_hal::{DecodeMode, SamplingConfig};

    let package = package();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("HYPERION_ORACLE").expect("set HYPERION_ORACLE")).unwrap()).unwrap();
    let dims = qwen2_dims(&package);
    let prompt: Vec<u32> = oracle["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let started = std::time::Instant::now();
    let (_, _, backend) = loaded_backend();
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let generate = |mode| {
        let started = std::time::Instant::now();
        backend.generate_mode(&dims, &prompt, 8, &[], &SamplingConfig::greedy(), mode, &mut |_| {}).expect("generation");
        started.elapsed().as_secs_f64() * 1000.0
    };
    let (first_direct, second_direct) = (generate(DecodeMode::Direct), generate(DecodeMode::Direct));
    let (first_graph, second_graph) = (generate(DecodeMode::Graph), generate(DecodeMode::Graph));
    println!(
        "COLD START weights {load_ms:.0} ms; first generation {first_direct:.0} ms then {second_direct:.0} ms (direct); first graph generation {first_graph:.0} ms then {second_graph:.0} ms; CUDA_CACHE_DISABLE={}",
        std::env::var("CUDA_CACHE_DISABLE").unwrap_or_else(|_| "unset".into())
    );
    backend.shutdown().unwrap();
}

/// Deterministic pseudo-random bf16 data in `[-range, range]` (a 32-bit LCG, so no two caches
/// share a periodic pattern).
fn noise(n: usize, seed: u32, range: f32) -> Vec<u16> {
    use helios_hal::reference::f32_to_bf16;
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32_to_bf16(((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * range)
        })
        .collect()
}

/// The split-K decode attention against a float64 reference, with the old untiled kernel as the
/// quality yardstick, at context lengths from 1 to 32768 (the old kernel stops at 8192).
///
/// Criteria, declared before the first run: let `ulp` be one bf16 step at the largest reference
/// output (`max|ref| / 128`; the outputs are rounded to bf16, so a 1-ulp flip is ordinary rounding
/// noise). The old kernel's worst error over all outputs is `old_err`. The new kernel passes at a
/// length when `new_err <= old_err + ulp`, and where the old kernel cannot run when
/// `new_err <= 2 * ulp`. Independently, five launches on the same input must give identical
/// bits: the merge order is fixed and nothing uses atomics.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn split_attention_matches_reference_and_is_repeatable() {
    use helios_hal::reference::{attention, bf16_to_f32};
    let (_, _, backend) = loaded_backend();
    // Qwen2.5-0.5B: 14 query heads share 2 KV heads of 64 dims.
    let (heads, kv_heads, d) = (14usize, 2usize, 64usize);
    let worst = |a: &[u16], b: &[u16]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((bf16_to_f32(*x) - bf16_to_f32(*y)).abs()));
    for len in [1usize, 5, 129, 1024, 4096, 8000, 32768] {
        let q = noise(heads * d, len as u32 + 1, 2.0);
        let k = noise(len * kv_heads * d, len as u32 + 2, 2.0);
        let v = noise(len * kv_heads * d, len as u32 + 3, 3.0);
        let reference = attention(&q, &k, &v, 1, len, heads, kv_heads, d, len - 1);
        let ulp = reference.iter().map(|x| bf16_to_f32(*x).abs()).fold(0.0f32, f32::max) / 128.0;

        let new = backend.attention_decode_split_bf16(&q, &k, &v, len, heads, kv_heads, d).expect("split kernel");
        let new_err = worst(&new, &reference);
        for run in 1..5 {
            let again = backend.attention_decode_split_bf16(&q, &k, &v, len, heads, kv_heads, d).unwrap();
            assert!(again == new, "len {len}: launch {run} differs from the first, bit for bit");
        }
        if len <= 8192 {
            let old = backend.attention_bf16(&q, &k, &v, 1, len, heads, kv_heads, d, len - 1).expect("old kernel");
            let old_err = worst(&old, &reference);
            println!("len {len:>5}: new err {new_err:.5}, old err {old_err:.5}, 1 ulp {ulp:.5}; 5 launches identical");
            assert!(new_err <= old_err + ulp, "len {len}: new {new_err} vs old {old_err} + ulp {ulp}");
        } else {
            println!("len {len:>5}: new err {new_err:.5}, 2 ulp {:.5}; 5 launches identical (old kernel cannot run here)", 2.0 * ulp);
            assert!(new_err <= 2.0 * ulp, "len {len}: new {new_err} exceeds 2 ulp {}", 2.0 * ulp);
        }
    }
    backend.shutdown().unwrap();
}

/// End to end at long context: the logits of a decode step (split-K attention, position from the
/// device) against the logits of the same token run through prefill (the untiled kernel), over
/// the same tokens. Bound declared before the first run: max absolute logit difference <= 0.5.
/// The prompt is the oracle prompt repeated, so only the difference matters, not the text.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn long_context_decode_logits_match_prefill_logits() {
    use helios_hal::{DecodeMode, SamplingConfig};

    let package = package();
    let dims = qwen2_dims(&package);
    let seed: Vec<u32> = seed_prompt_ids();
    let (_, _, backend) = loaded_backend();
    let bound = 0.5f32;
    for context in [5usize, 1024, 4096, 8000, 16000] {
        let prompt: Vec<u32> = (0..context).map(|i| seed[i % seed.len()]).collect();
        // trace[0]: prefill logits of the prompt; trace[1]: logits of one decode step that feeds the greedy token.
        let mut trace = Vec::new();
        backend.generate_traced(&dims, &prompt, 2, &[], &SamplingConfig::greedy(), DecodeMode::Direct, Some(&mut trace)).unwrap();
        let first = trace[0].iter().enumerate().fold(0, |b, (i, v)| if *v > trace[0][b] { i } else { b }) as u32;
        let mut extended = prompt.clone();
        extended.push(first);
        let prefill = backend.forward_prefill_logits_device(&dims, &extended).expect("prefill of the extended prompt");
        let diff = trace[1].iter().zip(&prefill).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let argmax = |v: &[f32]| v.iter().enumerate().fold(0, |b, (i, x)| if *x > v[b] { i } else { b });
        println!("context {context:>5}: max |decode - prefill| logit diff {diff:.4} (bound {bound}); argmax decode {} prefill {}", argmax(&trace[1]), argmax(&prefill));
        assert!(diff <= bound, "context {context}: logit difference {diff} exceeds the declared bound {bound}");
    }
    backend.shutdown().unwrap();
}

/// The chunked prefill attention kernel against a float64 reference at context lengths from 1 to 32768, which the
/// untiled kernel cannot reach (it stops at 8192). Same criteria as the split-K decode kernel, declared before the first
/// run: `ulp` is one bf16 step at the largest reference output (`max|ref| / 128`), and the kernel passes at a length when
/// its worst error against the reference is at most `2 * ulp`; five launches on the same input must give identical bits.
/// The query rows are the last 8 positions of the context (fewer when it is shorter), plus a padding row repeating the
/// last position, which must equal the row it repeats.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn chunk_attention_matches_reference_and_is_repeatable() {
    use helios_hal::reference::{attention, bf16_to_f32};
    let (_, _, backend) = loaded_backend();
    let worst = |a: &[u16], b: &[u16]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((bf16_to_f32(*x) - bf16_to_f32(*y)).abs()));
    // Qwen2.5-0.5B (14 heads over 2 KV heads of 64 dims) and Qwen2.5-7B (28 over 4 of 128).
    for (heads, kv_heads, d) in [(14usize, 2usize, 64usize), (28, 4, 128)] {
        for len in [1usize, 5, 129, 1024, 4096, 8000, 8193, 32768] {
            let rows = len.min(8);
            let q_start = len - rows;
            let q = noise(rows * heads * d, len as u32 + 1, 2.0);
            let k = noise(len * kv_heads * d, len as u32 + 2, 2.0);
            let v = noise(len * kv_heads * d, len as u32 + 3, 3.0);
            let reference = attention(&q, &k, &v, rows, len, heads, kv_heads, d, q_start);
            let ulp = reference.iter().map(|x| bf16_to_f32(*x).abs()).fold(0.0f32, f32::max) / 128.0;
            let positions: Vec<u32> = (q_start..len).map(|p| p as u32).collect();
            let out = backend.attention_chunk_bf16(&q, &k, &v, &positions, len, heads, kv_heads, d).expect("chunk kernel");
            let err = worst(&out, &reference);
            for run in 1..5 {
                let again = backend.attention_chunk_bf16(&q, &k, &v, &positions, len, heads, kv_heads, d).unwrap();
                assert!(again == out, "{heads}x{kv_heads}x{d} len {len}: launch {run} differs from the first, bit for bit");
            }
            // A padding row: the last row's query again at the same position gives the same output.
            let mut padded_q = q.clone();
            padded_q.extend_from_slice(&q[(rows - 1) * heads * d..]);
            let mut padded_positions = positions.clone();
            padded_positions.push(*positions.last().unwrap());
            let padded = backend.attention_chunk_bf16(&padded_q, &k, &v, &padded_positions, len, heads, kv_heads, d).unwrap();
            assert!(padded[..rows * heads * d] == out[..] && padded[rows * heads * d..] == out[(rows - 1) * heads * d..], "padding row differs");
            println!("heads {heads}/{kv_heads} d {d} len {len:>5}: err {err:.5}, 2 ulp {:.5}; 5 launches identical", 2.0 * ulp);
            assert!(err <= 2.0 * ulp, "{heads}x{kv_heads}x{d} len {len}: error {err} exceeds 2 ulp {}", 2.0 * ulp);
        }
    }
    backend.shutdown().unwrap();
}

/// The chunked prefill against the reference prefill (the whole prompt in one launch with the untiled kernel), on the
/// loaded model, for prompts that fit one chunk: 1 and 127 and 129 tokens (the padding cases), 1000 and 4096. Both run
/// the same kernels but the attention; the bound is the one declared for decode against prefill in this file, before the
/// first run: max absolute logit difference <= 0.5. The argmax of both is printed.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn chunked_prefill_matches_the_untiled_reference() {
    let package = package();
    let dims = qwen2_dims(&package);
    let seed: Vec<u32> = seed_prompt_ids();
    let (_, _, backend) = loaded_backend();
    let bound = 0.5f32;
    let argmax = |v: &[f32]| v.iter().enumerate().fold(0, |b, (i, x)| if *x > v[b] { i } else { b });
    for len in [1usize, 127, 129, 1000, 4096] {
        let prompt: Vec<u32> = (0..len).map(|i| seed[i % seed.len()]).collect();
        let chunked = backend.forward_prefill_logits_device(&dims, &prompt).expect("chunked prefill");
        let reference = backend.forward_prefill_logits_untiled_reference(&dims, &prompt).expect("untiled prefill");
        let diff = chunked.iter().zip(&reference).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("prompt {len:>5}: max |chunked - untiled| logit diff {diff:.4} (bound {bound}); argmax chunked {} untiled {}", argmax(&chunked), argmax(&reference));
        assert!(diff <= bound, "prompt {len}: {diff} exceeds the declared bound {bound}");
    }
    backend.shutdown().unwrap();
}

/// The graph of each chunk capacity replays bit for bit what direct launches of the same chunk compute: the logits of
/// prompts that use one chunk of each capacity (1, 129, 1000), a full chunk and a partial one (4097), several (9000) and
/// the longest this model can hold (up to 16384 and the session capacity).
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn chunked_prefill_graph_equals_direct_bit_for_bit() {
    use helios_hal::{DecodeMode, SamplingConfig};
    let package = package();
    let dims = qwen2_dims(&package);
    let seed: Vec<u32> = seed_prompt_ids();
    let (_, _, backend) = loaded_backend();
    let capacity = backend.context_capacity().expect("a session");
    for len in [1usize, 129, 1000, 4096, 4097, 9000, 16384] {
        if len + 1 > capacity {
            println!("prompt {len}: skipped, the session holds {capacity} tokens");
            continue;
        }
        let prompt: Vec<u32> = (0..len).map(|i| seed[i % seed.len()]).collect();
        let mut direct = Vec::new();
        let mut graph = Vec::new();
        backend.generate_traced(&dims, &prompt, 1, &[], &SamplingConfig::greedy(), DecodeMode::Direct, Some(&mut direct)).unwrap();
        backend.generate_traced(&dims, &prompt, 1, &[], &SamplingConfig::greedy(), DecodeMode::Graph, Some(&mut graph)).unwrap();
        assert!(direct == graph, "prompt {len}: the graph's logits differ from direct launches");
        println!("prompt {len:>5}: chunk graphs equal direct launches bit for bit");
    }
    backend.shutdown().unwrap();
}

/// A prompt longer than the old 8192-token limit runs end to end: prefill in chunks, then greedy decoding, with the graph
/// and the direct path producing the same tokens, and the time to first token printed.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn a_prompt_beyond_8192_tokens_runs_end_to_end() {
    use helios_hal::{DecodeMode, SamplingConfig};
    let package = package();
    let dims = qwen2_dims(&package);
    let seed: Vec<u32> = seed_prompt_ids();
    let (_, _, backend) = loaded_backend();
    let capacity = backend.context_capacity().expect("a session");
    assert!(capacity > 8192, "the session holds {capacity} tokens");
    let len = 20_000usize.min(capacity - 16);
    let prompt: Vec<u32> = (0..len).map(|i| seed[i % seed.len()]).collect();
    let mut tokens = Vec::new();
    for mode in [DecodeMode::Direct, DecodeMode::Graph] {
        let mut got = Vec::new();
        backend.generate_mode(&dims, &prompt, 8, &[], &SamplingConfig::greedy(), mode, &mut |t| got.push(t)).expect("generation beyond 8192");
        assert_eq!(got.len(), 8);
        tokens.push(got);
    }
    println!("prompt {len}: tokens {:?}", tokens[0]);
    assert_eq!(tokens[0], tokens[1], "graph and direct decode after a long chunked prefill");
    backend.shutdown().unwrap();
}

/// Time to prefill a prompt (ms) by length, for direct launches and the chunk graphs: GPU milliseconds between two events
/// around the whole prefill and the wall time (what a client waits before the first token), median [p10, p90] of 5 runs
/// after one warm-up.
#[test]
#[ignore = "needs an H100, a compiled package and the oracle file (see module docs)"]
fn prefill_time_vs_context() {
    use helios_hal::DecodeMode;
    let package = package();
    let dims = qwen2_dims(&package);
    let seed: Vec<u32> = seed_prompt_ids();
    let (_, _, backend) = loaded_backend();
    let capacity = backend.context_capacity().expect("a session");
    for len in [1024usize, 4096, 8192, 16384] {
        if len > capacity {
            println!("prefill {len}: skipped, the session holds {capacity} tokens");
            continue;
        }
        let prompt: Vec<u32> = (0..len).map(|i| seed[i % seed.len()]).collect();
        let mut line = format!("prefill {len:>5} tokens:");
        for (name, mode) in [("direct", DecodeMode::Direct), ("graph", DecodeMode::Graph)] {
            backend.time_prefill(&dims, &prompt, mode).expect("warm-up");
            let runs: Vec<(f64, f64)> = (0..5).map(|_| backend.time_prefill(&dims, &prompt, mode).expect("timed prefill")).collect();
            let (gpu, wall) = (spread(runs.iter().map(|r| r.0).collect()), spread(runs.iter().map(|r| r.1).collect()));
            line += &format!(" {name} gpu {gpu} wall {wall};");
        }
        println!("{line} (ms)");
    }
    backend.shutdown().unwrap();
}

/// A Hyperion forward against the compact oracle (`scripts/gen-oracle-large.py`): for each prompt, the
/// logits of the prefill and of each greedy decode step are compared with the float32 reference under the
/// tolerance the oracle file declared before this run (see `helios_hal::oracle` for the rule). Positions are
/// compared teacher-forced (the reference's own tokens are fed, so a near-tie flip cannot make later steps incomparable); free greedy decoding is a secondary check that may leave the reference's chain only at a near tie.
/// Also prints the device memory left after the weights loaded (`cuMemGetInfo`), for the KV sizing.
///
/// Needs `HYPERION_COMPACT_ORACLE=/path/<id>.compact.json` next to the usual package variable.
#[test]
#[ignore = "needs an H100, a compiled package and the compact oracle (see module docs)"]
fn forward_matches_the_compact_oracle() {
    use helios_hal::oracle::{compare, CompactOracle};
    use helios_hal::sampling::argmax;
    use helios_hal::{DecodeMode, SamplingConfig};

    let package = package();
    let oracle = CompactOracle::from_json(&std::fs::read(std::env::var("HYPERION_COMPACT_ORACLE").expect("set HYPERION_COMPACT_ORACLE")).unwrap()).expect("compact oracle");
    let dims = qwen2_dims(&package);
    let (_, _, backend) = loaded_backend();
    let (tensors, bytes) = backend.loaded_weights();
    let (free, total) = backend.device_memory().expect("cuMemGetInfo");
    println!(
        "VRAM after loading {tensors} tensors ({:.2} GiB): {:.2} GiB free of {:.2} GiB",
        bytes as f64 / (1u64 << 30) as f64,
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64
    );
    println!("tolerance declared in the oracle: top-k logits {} , logsumexp {}", oracle.tolerance.max_abs_top_logits, oracle.tolerance.max_abs_logsumexp);
    if let Some(d) = &oracle.bf16_drift {
        println!(
            "ORACLE HF bf16 vs fp32 drift: top-k logits {:.4}, logsumexp {:.4}; the top-k limit is set by the {} (margin x drift, or the floor)",
            d.max_abs_top_logits, d.max_abs_logsumexp, d.binding_top_logits
        );
    }
    // Failures are collected, not asserted at once, so the summary below (ratio and worst position) is
    // always printed. A failure is never answered by loosening the tolerance: see the runbook.
    let (mut failures, mut overall_top, mut overall_lse, mut worst_at) = (Vec::new(), 0f32, 0f64, String::new());
    // HF bf16's deviation over exactly the positions Hyperion was compared on (the like-for-like denominator).
    let mut bf16_same_set = Some(0f32);
    for prompt in &oracle.prompts {
        let n = oracle.generated_tokens.min(prompt.generated_ids.len()).min(prompt.reference.len());

        // 1. The check: teacher-forced. The reference's own tokens are fed, so every one of the n positions
        //    is compared against the declared tolerance, whatever the model would have picked.
        let mut trace = Vec::new();
        backend
            .generate_forced(&dims, &prompt.ids, &prompt.generated_ids[..n], DecodeMode::Direct, &mut trace)
            .expect("teacher-forced generation");
        let (mut compared, mut worst_top, mut worst_lse) = (0usize, 0f32, 0f64);
        for (i, (logits, record)) in trace.iter().zip(&prompt.reference).enumerate() {
            let c = compare(logits, record, &oracle.tolerance).expect("comparison");
            if std::env::var_os("HYPERION_ORACLE_VERBOSE").is_some() {
                // Per position: Hyperion's differences next to Hugging Face bf16's, for telling a systematic error from one noisy region.
                println!("   step {i:>2}: top-k {:.4} lse {:.4} | HF bf16 vs fp32 (top-k, lse): {:?}", c.max_abs_top_logits, c.abs_logsumexp, prompt.bf16_deviation.as_ref().map(|d| d[i]));
            }
            worst_top = worst_top.max(c.max_abs_top_logits);
            worst_lse = worst_lse.max(c.abs_logsumexp);
            if c.max_abs_top_logits > overall_top {
                overall_top = c.max_abs_top_logits;
                worst_at = format!("{} ({} prompt tokens), position {} (step {i})", prompt.label, prompt.ids.len(), record.position);
            }
            overall_lse = overall_lse.max(c.abs_logsumexp);
            bf16_same_set = match (bf16_same_set, &prompt.bf16_deviation) {
                (Some(m), Some(dev)) => Some(m.max(dev[i].0)),
                _ => None,
            };
            if !c.within(&oracle.tolerance) {
                failures.push(format!(
                    "{}: position {} (step {i}): top-k diff {:.4} (tolerance {}), logsumexp diff {:.4} (tolerance {}), argmax {} vs {} (ok: {})",
                    prompt.label, record.position, c.max_abs_top_logits, oracle.tolerance.max_abs_top_logits, c.abs_logsumexp, oracle.tolerance.max_abs_logsumexp, c.argmax, record.argmax, c.argmax_ok
                ));
            }
            compared += 1;
        }
        println!(
            "ORACLE {} (teacher-forced): {} prompt tokens, compared {compared} of {n} positions; max top-k logit diff {worst_top:.4}, max logsumexp diff {worst_lse:.4}",
            prompt.label,
            prompt.ids.len()
        );
        if compared < n {
            failures.push(format!("{}: only {compared} of {n} positions were produced", prompt.label));
        }

        // 2. Secondary: free greedy decoding, where Hyperion's own tokens feed the next step. It may leave the
        //    reference's chain only at a near tie: the fp32 gap between the two best logits within the
        //    tolerance and our token one of those two. The token we chose and its logit are always printed.
        let mut free = Vec::new();
        backend
            .generate_traced(&dims, &prompt.ids, n, &[], &SamplingConfig::greedy(), DecodeMode::Direct, Some(&mut free))
            .expect("free greedy generation");
        match (0..free.len().min(n)).find(|&j| argmax(&free[j]) as u32 != prompt.generated_ids[j]) {
            None => println!("ORACLE {} (free greedy): all {n} tokens equal the reference's", prompt.label),
            Some(j) => {
                let chosen = argmax(&free[j]) as u32;
                let reference = &prompt.reference[j];
                let c = compare(&free[j], reference, &oracle.tolerance).expect("comparison");
                println!(
                    "ORACLE {} (free greedy): left the reference chain at step {j}: ours id {chosen} (logit {:.4}), reference id {} (our logit {:.4}); fp32 top two {:?} = {:.4}, {:.4}, gap {:.4}; legitimate near tie: {}",
                    prompt.label,
                    free[j][chosen as usize],
                    reference.argmax,
                    free[j][reference.argmax as usize],
                    &reference.top_ids[..2],
                    reference.top_logits[0],
                    reference.top_logits[1],
                    reference.top_logits[0] - reference.top_logits[1],
                    c.argmax_ok
                );
                if !c.argmax_ok {
                    failures.push(format!("{}: free greedy left the reference chain at step {j} with token {chosen}, which is not within a near tie of the fp32 top two", prompt.label));
                }
            }
        }
    }
    let ratio = |denominator: Option<f32>| denominator.filter(|d| *d > 0.0).map_or("n/a".to_string(), |d| format!("{:.2}", overall_top / d));
    println!(
        "ORACLE SUMMARY: Hyperion max top-k logit diff {overall_top:.4} at {worst_at}; max logsumexp diff {overall_lse:.4}; tolerance {}",
        oracle.tolerance.max_abs_top_logits
    );
    println!(
        "ORACLE SUMMARY ratio, same positions (HF bf16 max over the compared positions: {}): {}; ratio, all positions (HF bf16 max over every position, the basis of the tolerance: {}): {}",
        bf16_same_set.map_or("n/a".to_string(), |v| format!("{v:.4}")),
        ratio(bf16_same_set),
        oracle.bf16_drift.as_ref().map_or("n/a".to_string(), |d| format!("{:.4}", d.max_abs_top_logits)),
        ratio(oracle.bf16_drift.as_ref().map(|d| d.max_abs_top_logits))
    );
    backend.shutdown().unwrap();
    assert!(
        failures.is_empty(),
        "{} failure(s); do not loosen the tolerance and rerun (docs/h100-test-runbook.md, `If Hyperion exceeds the tolerance`):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The rule "after the model is loaded, nothing is built while serving": a device that is told to forbid
/// new GPU work refuses a cuBLAS handle, a graph capture, a kernel module and an allocation, and allows
/// them again afterwards. (The generation path itself runs under that restriction, so every hardware test
/// that generates after loading also shows that a generation needs none of these.)
#[test]
#[ignore = "needs an H100 (no weights)"]
fn new_gpu_work_is_refused_while_forbidden_and_allowed_after() {
    use helios_hal::cuda::{Cublas, CudaDevice, GraphExec, KernelModule};

    let mut dev = CudaDevice::probe().expect("an H100 must be present");
    {
        let _serving = dev.forbid_new_gpu_work();
        let refused = |what: &str, error: anyhow::Error| {
            let text = error.to_string();
            assert!(text.contains(what) && text.contains("not allowed while serving"), "{what}: {text}");
        };
        refused("cuBLAS handle creation", Cublas::new(&dev).err().expect("must be refused"));
        refused("graph capture", GraphExec::capture(&dev, || Ok(())).err().expect("must be refused"));
        refused("kernel module load", KernelModule::load(&dev, &[0u8; 16]).err().expect("must be refused"));
        refused("device memory allocation", dev.alloc(16).err().expect("must be refused"));
    }
    let pointer = dev.alloc(1024).expect("allocation is allowed again once the guard is dropped");
    dev.free(pointer).unwrap();
    let _handle = Cublas::new(&dev).expect("so is a cuBLAS handle");
    dev.teardown_ref().unwrap();
}
