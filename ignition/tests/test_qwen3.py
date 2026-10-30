"""The Qwen3 compiler: the Qwen2 decoder without Q/K/V biases and with per-head Q/K norm weights. The model's real
config and safetensors header come from the registry (skipped visibly when not downloaded); the strictness of the
mapping is checked on a synthetic header."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hyperion_ignition.compiler.model_config import ModelConfig
from hyperion_ignition.compiler.registry import build_compiler
from hyperion_ignition.compiler.weights import TensorEntry, map_qwen2_layers, read_safetensors_header
from hyperion_ignition.hardware.registry import build_recipe
from hyperion_ignition.manifest_schema import validate_manifest

TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
MODEL = TESTDATA / "models" / "qwen3-0.6b"


def need_files() -> Path:
    if not (MODEL / "config.json").is_file() or not (MODEL / "model.safetensors.header").is_file():
        message = "qwen3-0.6b: run `python scripts/fetch-model-fixtures.py qwen3-0.6b`"
        if os.environ.get("HYPERION_REQUIRE_MODEL_FIXTURES"):
            pytest.fail(message)
        pytest.skip(message)
    return MODEL


def test_the_real_qwen3_06b_compiles_with_qk_norm_and_no_qkv_bias():
    source = need_files()
    manifest = build_compiler("qwen3").compile(source=source, recipe=build_recipe("sm_90"))
    model = manifest.model
    assert (model.qk_norm, model.qkv_bias, model.tie_word_embeddings) == (True, False, True)
    assert (model.num_layers, model.head_dim, model.num_attention_heads, model.num_kv_heads) == (28, 128, 16, 8)
    names = {layer.name for layer in manifest.layers}
    assert "layers.0.self_attn.q_norm.weight" in names and "layers.27.self_attn.k_norm.weight" in names
    assert not any(name.endswith(".bias") for name in names), "Qwen3 has no Q/K/V bias"
    # The checkpoint carries a second copy of the output head although the embeddings are tied; it is mapped, not refused.
    assert "lm_head.weight" in names
    document = manifest.to_json_dict()
    validate_manifest(document)
    assert document["model"]["qk_norm"] is True


def test_the_other_compilers_refuse_a_qwen3_config_and_qwen3_refuses_a_qwen2_one():
    source = need_files()
    for other in ("qwen2", "llama3"):
        with pytest.raises(ValueError, match="not supported by this compiler"):
            build_compiler(other).compile(source=source, recipe=build_recipe("sm_90"))
    qwen2 = Path(__file__).resolve().parent / "fixtures" / "qwen2-hf"
    with pytest.raises(ValueError, match="not supported by this compiler"):
        build_compiler("qwen3").compile(source=qwen2, recipe=build_recipe("sm_90"))


CFG = ModelConfig(num_layers=1, attention="gqa", num_attention_heads=4, num_kv_heads=2, head_dim=4, vocab_size=8, max_seq_len=16,
                  hidden_size=16, intermediate_size=32, rope_theta=1e6, rms_norm_eps=1e-6, tie_word_embeddings=True, qkv_bias=False, qk_norm=True)


def entries(names_shapes: dict[str, tuple[int, ...]]) -> dict[str, TensorEntry]:
    out, at = {}, 0
    for name, shape in names_shapes.items():
        size = 2
        for dim in shape:
            size *= dim
        out[name] = TensorEntry(name, "bf16", shape, at, at + size)
        at += size
    return out


QWEN3_LAYER = {
    "model.layers.0.input_layernorm.weight": (16,), "model.layers.0.post_attention_layernorm.weight": (16,),
    "model.layers.0.self_attn.q_proj.weight": (16, 16), "model.layers.0.self_attn.k_proj.weight": (8, 16),
    "model.layers.0.self_attn.v_proj.weight": (8, 16), "model.layers.0.self_attn.o_proj.weight": (16, 16),
    "model.layers.0.self_attn.q_norm.weight": (4,), "model.layers.0.self_attn.k_norm.weight": (4,),
    "model.layers.0.mlp.gate_proj.weight": (32, 16), "model.layers.0.mlp.up_proj.weight": (32, 16), "model.layers.0.mlp.down_proj.weight": (16, 32),
}
BASE = {"model.embed_tokens.weight": (8, 16), "model.norm.weight": (16,), **QWEN3_LAYER}


def test_the_mapping_needs_the_qk_norm_weights_and_refuses_biases_when_the_config_has_none():
    assert len(map_qwen2_layers(entries(BASE), CFG, tie_word_embeddings=True)) == len(BASE)
    without_norm = {k: v for k, v in BASE.items() if "q_norm" not in k}
    with pytest.raises(ValueError, match="missing.*q_norm"):
        map_qwen2_layers(entries(without_norm), CFG, tie_word_embeddings=True)
    with_bias = {**BASE, "model.layers.0.self_attn.q_proj.bias": (16,)}
    with pytest.raises(ValueError, match="unmapped.*q_proj.bias"):
        map_qwen2_layers(entries(with_bias), CFG, tie_word_embeddings=True)
    wrong_norm = {**BASE, "model.layers.0.self_attn.k_norm.weight": (8,)}
    with pytest.raises(ValueError, match="k_norm.*shape"):
        map_qwen2_layers(entries(wrong_norm), CFG, tie_word_embeddings=True)


def test_a_tied_checkpoint_may_carry_its_head_but_an_untied_one_must():
    tied_with_head = {**BASE, "lm_head.weight": (8, 16)}
    assert len(map_qwen2_layers(entries(tied_with_head), CFG, tie_word_embeddings=True)) == len(BASE) + 1
    with pytest.raises(ValueError, match="missing.*lm_head"):
        map_qwen2_layers(entries(BASE), CFG, tie_word_embeddings=False)
