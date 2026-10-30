//! Tensor-parallel slices (ADR 0005, W1.5): for every supported size, the slices of all ranks tile each split tensor exactly once
//! (a replicated KV head is read by exactly `tp / num_kv_heads` ranks), a size outside the list is refused before any device is touched,
//! and on the real bytes of the committed sharded fixture the slices put back together are the canonical tensor.
//!
//! Two inputs: the tiny sharded Qwen2 fixture (real safetensors bytes, manifest compiled by Ignition), and the tensor list of the real
//! Qwen2.5-7B (names, shapes and classes read from its four shard headers by `scripts/gen-shard-plan-fixture.py`; no weights).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use helios_hal::sharding::{extract, kv_placement, pipeline_layers, place, rank_shape, rank_slice, rank_slices, KvPlacement};
use helios_hal::weights_plan::{self, PayloadFile};
use helios_hal::{BackendConfig, Deployment, HalError, TensorSlice, TensorSpec};
use hyperion_specification::v1::{Activation, AttentionKind, ModelConfig, RopeStyle, ShardClass, TensorFormat};
use hyperion_specification::Manifest;
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../testdata/synthetic")
}

fn deployment(tp: u32, rank: u32) -> Deployment {
    Deployment { tp_size: tp, pp_size: 1, tp_rank: rank, pp_stage: 0 }
}

/// For one tensor, every rank's slice along the split axis, as `(start, end)` intervals in rank order.
fn intervals(config: &BackendConfig, tensor: &TensorSpec, tp: u32) -> Vec<Option<TensorSlice>> {
    (0..tp)
        .map(|rank| {
            let mut c = config.clone();
            c.deployment = deployment(tp, rank);
            rank_slice(&c, tensor).unwrap()
        })
        .collect()
}

/// The slices of a split tensor cover `[0, dim)` exactly once; a KV tensor under replication covers it `ranks_per_head` times.
fn assert_tiles(config: &BackendConfig, tensor: &TensorSpec, tp: u32) {
    let slices = intervals(config, tensor, tp);
    let copies = match tensor.sharding.unwrap() {
        ShardClass::Replicated => {
            assert!(slices.iter().all(Option::is_none), "{}: a replicated tensor is read whole", tensor.name);
            return;
        }
        ShardClass::Column if (tensor.name.contains(".k_proj.") || tensor.name.contains(".v_proj.")) => match kv_placement(config.model.num_kv_heads as usize, tp as usize) {
            KvPlacement::Replicated { ranks_per_head } => ranks_per_head,
            KvPlacement::Split { .. } => 1,
        },
        _ => 1,
    };
    let axis = slices[0].as_ref().unwrap().axis;
    let dim = tensor.shape[axis];
    let mut count = vec![0usize; dim];
    for slice in slices.iter().flatten() {
        assert_eq!(slice.axis, axis, "{}", tensor.name);
        assert!(slice.start <= slice.end && slice.end <= dim, "{}: {slice:?} outside {dim}", tensor.name);
        for c in &mut count[slice.start..slice.end] {
            *c += 1;
        }
    }
    assert!(count.iter().all(|&c| c == copies), "{} at tp {tp}: some index is read {:?} times, expected {copies}", tensor.name, count.iter().find(|&&c| c != copies));
}

fn model_config(heads: u32, kv: u32, head_dim: u32, hidden: u32, inter: u32, vocab: u32, layers: u32) -> ModelConfig {
    ModelConfig {
        num_layers: layers,
        attention: AttentionKind::Gqa,
        num_attention_heads: heads,
        num_kv_heads: kv,
        head_dim,
        vocab_size: vocab,
        max_seq_len: 4096,
        hidden_size: hidden,
        intermediate_size: inter,
        rope_theta: 1e6,
        rope_style: RopeStyle::RotateHalf,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        qkv_bias: true,
        qk_norm: false,
        activation: Activation::Silu,
        attention_scale: None,
    }
}

fn config_of(model: ModelConfig, tensors: Vec<TensorSpec>, supported: Vec<u32>) -> BackendConfig {
    BackendConfig {
        model_id: "m".into(),
        model,
        tensor_format: TensorFormat::Bf16,
        target_arch: "sm_90".into(),
        eos_token_ids: vec![0],
        payload_files: vec!["model.safetensors".into()],
        payload_lengths: vec![None],
        tensors,
        supported_tp_sizes: supported,
        deployment: Deployment::SINGLE,
    }
}

fn class(name: &str) -> ShardClass {
    match name {
        "vocab" => ShardClass::Vocab,
        "column" => ShardClass::Column,
        "row" => ShardClass::Row,
        _ => ShardClass::Replicated,
    }
}

fn qwen7b() -> BackendConfig {
    let path = root().join("qwen2.5-7b-tensors.json");
    let v: Value = serde_json::from_slice(&std::fs::read(path).expect("the committed 7B tensor list")).unwrap();
    let n = |key: &str| v[key].as_u64().unwrap() as u32;
    let tensors = v["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| TensorSpec {
            name: t["name"].as_str().unwrap().into(),
            payload: 0,
            byte_offset: 0,
            byte_length: 0,
            dtype: "bf16".into(),
            shape: t["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect(),
            sharding: Some(class(t["sharding"].as_str().unwrap())),
        })
        .collect();
    let supported = v["supported_tp_sizes"].as_array().unwrap().iter().map(|s| s.as_u64().unwrap() as u32).collect();
    config_of(model_config(n("num_attention_heads"), n("num_kv_heads"), n("head_dim"), n("hidden_size"), n("intermediate_size"), n("vocab_size"), n("num_layers")), tensors, supported)
}

#[test]
fn on_the_real_7b_tensor_list_the_slices_tile_every_tensor_for_tp_1_2_and_4() {
    let config = qwen7b();
    assert_eq!(config.tensors.len(), 339, "28 layers x 12 tensors + embedding, final norm, head");
    assert_eq!(config.supported_tp_sizes, vec![1, 2, 4]);
    for tp in [1u32, 2, 4] {
        for tensor in &config.tensors {
            if tp == 1 {
                assert!(intervals(&config, tensor, 1)[0].is_none());
            } else {
                assert_tiles(&config, tensor, tp);
            }
        }
    }
}

#[test]
fn on_the_7b_each_kv_head_stays_with_its_query_heads() {
    // 28 query heads, 4 KV heads: query head h uses KV head h / 7. At tp 4 rank r holds query heads 7r..7r+7 and KV head r.
    let config = qwen7b();
    let k = config.tensors.iter().find(|t| t.name == "layers.0.self_attn.k_proj.weight").unwrap();
    let q = config.tensors.iter().find(|t| t.name == "layers.0.self_attn.q_proj.weight").unwrap();
    for rank in 0..4usize {
        let mut c = config.clone();
        c.deployment = deployment(4, rank as u32);
        let (qs, ks) = (rank_slice(&c, q).unwrap().unwrap(), rank_slice(&c, k).unwrap().unwrap());
        let (first_q_head, kv_head) = (qs.start / 128, ks.start / 128);
        assert_eq!(first_q_head / 7, kv_head, "rank {rank}");
        assert_eq!((qs.end - qs.start) / 128, 7);
        assert_eq!(ks.end - ks.start, 128, "one KV head per rank");
    }
}

#[test]
fn a_size_the_model_does_not_allow_is_refused_before_any_device_work() {
    let config = qwen7b();
    for tp in [3u32, 5, 8] {
        let mut c = config.clone();
        c.deployment = deployment(tp, 0);
        assert_eq!(c.check_deployment(), Err(HalError::UnsupportedTp { tp, supported: vec![1, 2, 4] }), "tp {tp}");
    }
    let mut c = config.clone();
    c.deployment = deployment(2, 2);
    assert!(matches!(c.check_deployment(), Err(HalError::Invalid(_))), "rank 2 of a group of 2");
    c.deployment = Deployment { tp_size: 1, pp_size: 29, tp_rank: 0, pp_stage: 0 };
    assert!(c.check_deployment().is_err(), "29 stages for 28 layers");
}

#[test]
fn the_rank_shape_of_the_7b_and_the_padded_vocabulary() {
    let mut c = qwen7b();
    c.deployment = deployment(4, 3);
    let shape = rank_shape(&c);
    assert_eq!((shape.heads, shape.kv_heads, shape.intermediate), (7, 1, 18944 / 4));
    assert_eq!(shape.padded_vocab, 152_064, "152064 is a multiple of 4 * 64");
    assert_eq!(shape.vocab_rows, 114_048..152_064);
    // 0.5B-like vocabulary 151936 at tp 4 pads to 152064: the last rank holds 128 padded rows it never reads from the file.
    let mut small = config_of(model_config(14, 2, 64, 896, 4864, 151_936, 24), Vec::new(), vec![1, 2]);
    small.supported_tp_sizes = vec![1, 2, 4];
    small.deployment = deployment(4, 3);
    let shape = rank_shape(&small);
    assert_eq!(shape.padded_vocab, 152_064);
    assert_eq!(shape.vocab_rows.end, 151_936, "the real rows stop at the vocabulary size");
}

#[test]
fn pipeline_stages_of_the_7b() {
    assert_eq!(pipeline_layers(28, 4, 0), 0..7);
    assert_eq!(pipeline_layers(28, 3, 0), 0..10);
    assert_eq!(pipeline_layers(28, 3, 2), 19..28);
}

// ---- the tiny sharded fixture: real bytes -------------------------------------------------------------------------------------

fn fixture() -> PathBuf {
    root().join("qwen2-tiny-sharded")
}

#[test]
fn on_real_bytes_the_rank_slices_put_back_together_are_the_canonical_tensors() {
    let Manifest::V1(manifest) = serde_json::from_slice(&std::fs::read(fixture().join("model.manifest.json")).unwrap()).unwrap();
    let supported = manifest.parallelism.as_ref().expect("Ignition writes the supported sizes").supported_tp_sizes.clone();
    assert!(supported.len() >= 2, "the fixture must allow some tensor parallelism: {supported:?}");
    let base = BackendConfig::from_manifest(&manifest).unwrap();
    let files: Vec<PayloadFile> = base
        .payload_files
        .iter()
        .map(|name| {
            let mut file = File::open(fixture().join(name)).unwrap();
            PayloadFile { data_start: weights_plan::data_start(&mut file).unwrap(), file_len: file.metadata().unwrap().len() }
        })
        .collect();
    let plans = weights_plan::plan(&base, &files).unwrap();
    let read = |tensor: &TensorSpec| -> Vec<u8> {
        let plan = plans.iter().find(|p| p.name == tensor.name).unwrap();
        let mut file = File::open(fixture().join(&base.payload_files[plan.payload])).unwrap();
        file.seek(SeekFrom::Start(plan.file_offset)).unwrap();
        let mut bytes = vec![0u8; plan.len as usize];
        file.read_exact(&mut bytes).unwrap();
        bytes
    };
    for &tp in supported.iter().filter(|&&t| t > 1) {
        for tensor in &base.tensors {
            assert_tiles(&base, tensor, tp);
            let whole = read(tensor);
            let elem = 2; // bf16
            let mut rebuilt = vec![0xEEu8; whole.len()];
            let mut covered = false;
            for rank in 0..tp {
                let mut c = base.clone();
                c.deployment = deployment(tp, rank);
                match rank_slice(&c, tensor).unwrap() {
                    None => {
                        rebuilt.copy_from_slice(&whole);
                        covered = true;
                    }
                    Some(slice) => {
                        let part = extract(&whole, &tensor.shape, elem, &slice);
                        assert_eq!(part.len(), (slice.end - slice.start) * tensor.shape.iter().enumerate().filter(|(i, _)| *i != slice.axis).map(|(_, d)| d).product::<usize>() * elem);
                        place(&mut rebuilt, &tensor.shape, elem, &slice, &part);
                        covered = true;
                    }
                }
            }
            assert!(covered);
            assert_eq!(rebuilt, whole, "{} at tp {tp}: the slices of all ranks do not reassemble to the tensor", tensor.name);
        }
        // The slices of one rank are what a load would ask the weight source for.
        let mut c = base.clone();
        c.deployment = deployment(tp, 0);
        assert!(!rank_slices(&c).unwrap().is_empty());
    }
}

#[test]
fn a_manifest_deployment_outside_the_supported_sizes_is_refused_by_the_config() {
    let Manifest::V1(manifest) = serde_json::from_slice(&std::fs::read(fixture().join("model.manifest.json")).unwrap()).unwrap();
    let supported = manifest.parallelism.as_ref().unwrap().supported_tp_sizes.clone();
    let unsupported = (1..=16).find(|t| !supported.contains(t)).unwrap();
    let error = BackendConfig::for_deployment(&manifest, deployment(unsupported, 0)).unwrap_err();
    assert!(matches!(error, HalError::UnsupportedTp { tp, .. } if tp == unsupported), "{error}");
    assert!(BackendConfig::for_deployment(&manifest, deployment(supported[supported.len() - 1], 0)).is_ok());
}

#[test]
fn a_kv_head_shared_by_several_ranks_is_read_by_exactly_those_ranks() {
    // 8 query heads of 4 dims, 2 KV heads, MLP 64: tp 4 puts each KV head on two ranks (ranks 0, 1 and ranks 2, 3).
    let spec = |name: &str, shape: &[usize], sharding| TensorSpec { name: name.into(), payload: 0, byte_offset: 0, byte_length: 0, dtype: "bf16".into(), shape: shape.to_vec(), sharding: Some(sharding) };
    let tensors = vec![
        spec("layers.0.self_attn.q_proj.weight", &[32, 16], ShardClass::Column),
        spec("layers.0.self_attn.k_proj.weight", &[8, 16], ShardClass::Column),
        spec("layers.0.self_attn.k_proj.bias", &[8], ShardClass::Column),
        spec("layers.0.self_attn.v_proj.weight", &[8, 16], ShardClass::Column),
        spec("layers.0.self_attn.o_proj.weight", &[16, 32], ShardClass::Row),
        spec("layers.0.input_layernorm.weight", &[16], ShardClass::Replicated),
    ];
    let config = config_of(model_config(8, 2, 4, 16, 64, 100, 1), tensors.clone(), vec![1, 2, 4]);
    for tensor in &tensors {
        assert_tiles(&config, tensor, 4);
    }
    let k = &tensors[1];
    let rows: Vec<(usize, usize)> = intervals(&config, k, 4).into_iter().map(|s| s.unwrap()).map(|s| (s.start, s.end)).collect();
    assert_eq!(rows, vec![(0, 4), (0, 4), (4, 8), (4, 8)], "KV head 0 on ranks 0 and 1, KV head 1 on ranks 2 and 3");
    // Query heads of rank r are 2r and 2r+1; they use KV head (2r) / 4 = r / 2.
    let q = &tensors[0];
    for (rank, slice) in intervals(&config, q, 4).into_iter().enumerate() {
        let first_q_head = slice.unwrap().start / 4;
        assert_eq!(first_q_head / 4, rows[rank].0 / 4, "rank {rank}");
    }
}
