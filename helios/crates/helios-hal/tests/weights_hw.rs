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

// ---- single ops from kernels/elementwise.cu, each against the CPU reference ----

fn loaded_backend() -> (PathBuf, hyperion_specification::v1::Manifest, HopperBackend) {
    let package = package();
    let Manifest::V1(manifest) =
        serde_json::from_slice(&std::fs::read(package.with_extension("manifest.json")).unwrap()).unwrap();
    let backend = HopperBackend::new().expect("an H100 must be present");
    backend.load_model_weights(&package, &manifest).expect("upload");
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
    use helios_hal::Qwen2Dims;

    let package = package();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("HYPERION_ORACLE").expect("set HYPERION_ORACLE")).unwrap()).unwrap();
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(package.with_file_name("config.json")).unwrap()).unwrap();
    let int = |k: &str| config[k].as_u64().unwrap_or_else(|| panic!("config.json {k}")) as usize;
    let heads = int("num_attention_heads");
    let dims = Qwen2Dims {
        layers: int("num_hidden_layers"),
        hidden: int("hidden_size"),
        intermediate: int("intermediate_size"),
        heads,
        kv_heads: int("num_key_value_heads"),
        head_dim: int("hidden_size") / heads,
        vocab: int("vocab_size"),
        rope_theta: config["rope_theta"].as_f64().expect("rope_theta"),
        rms_eps: config["rms_norm_eps"].as_f64().expect("rms_norm_eps") as f32,
    };
    assert_eq!(config["tie_word_embeddings"], true, "the forward pass assumes a tied output head");

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
    use helios_hal::Qwen2Dims;

    let package = package();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("HYPERION_ORACLE").expect("set HYPERION_ORACLE")).unwrap()).unwrap();
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(package.with_file_name("config.json")).unwrap()).unwrap();
    let int = |k: &str| config[k].as_u64().unwrap_or_else(|| panic!("config.json {k}")) as usize;
    let heads = int("num_attention_heads");
    let dims = Qwen2Dims {
        layers: int("num_hidden_layers"),
        hidden: int("hidden_size"),
        intermediate: int("intermediate_size"),
        heads,
        kv_heads: int("num_key_value_heads"),
        head_dim: int("hidden_size") / heads,
        vocab: int("vocab_size"),
        rope_theta: config["rope_theta"].as_f64().unwrap(),
        rms_eps: config["rms_norm_eps"].as_f64().unwrap() as f32,
    };
    let ids = |key: &str| -> Vec<u32> { oracle[key].as_array().expect(key).iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let (prompt, want) = (ids("prompt_ids"), ids("generated_ids"));

    let (_, _, backend) = loaded_backend();
    let started = std::time::Instant::now();
    let got = backend.generate_greedy(&dims, &prompt, want.len(), None).expect("generation");
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
