"""Weight mapping: read a safetensors header and map every tensor to a manifest
layer, failing on missing AND unmapped tensors, bad ranges and unsupported dtypes.

Synthetic cases build real safetensors bytes (8-byte little-endian header length,
JSON header, data), so the reader is exercised on the actual format. The real
Qwen2.5-0.5B header is checked through the model registry (generic, no model
named here); see testdata/README.md.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from hyperion_ignition.compiler.model_config import ModelConfig
from hyperion_ignition.compiler.weights import map_qwen2_layers, read_safetensors_header

TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
REGISTRY = json.loads((TESTDATA / "models.json").read_text(encoding="utf-8"))["models"]

# 2 layers, 4 heads (head_dim 4), 2 kv heads, hidden 16, vocab 8
CFG = ModelConfig(num_layers=2, attention="gqa", num_attention_heads=4, num_kv_heads=2, head_dim=4,
                  vocab_size=8, max_seq_len=16, hidden_size=16, intermediate_size=32, rope_theta=10000.0,
                  rms_norm_eps=1e-06, tie_word_embeddings=False, qkv_bias=True)
HIDDEN = 16


def expected_shapes(tied: bool) -> dict[str, list[int]]:
    shapes = {"model.embed_tokens.weight": [CFG.vocab_size, HIDDEN], "model.norm.weight": [HIDDEN]}
    for i in range(CFG.num_layers):
        p = f"model.layers.{i}."
        shapes.update({
            p + "input_layernorm.weight": [HIDDEN], p + "post_attention_layernorm.weight": [HIDDEN],
            p + "self_attn.q_proj.weight": [16, HIDDEN], p + "self_attn.q_proj.bias": [16],
            p + "self_attn.k_proj.weight": [8, HIDDEN], p + "self_attn.k_proj.bias": [8],
            p + "self_attn.v_proj.weight": [8, HIDDEN], p + "self_attn.v_proj.bias": [8],
            p + "self_attn.o_proj.weight": [HIDDEN, 16],
            p + "mlp.gate_proj.weight": [32, HIDDEN], p + "mlp.up_proj.weight": [32, HIDDEN],
            p + "mlp.down_proj.weight": [HIDDEN, 32],
        })
    if not tied:
        shapes["lm_head.weight"] = [CFG.vocab_size, HIDDEN]
    return shapes


def write_safetensors(path: Path, shapes: dict[str, list[int]], dtype="BF16", mutate=None) -> Path:
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, shape in shapes.items():
        size = 2
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    if mutate:
        mutate(header)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return path


def test_reader_returns_entries_with_normalised_dtype(tmp_path):
    entries = read_safetensors_header(write_safetensors(tmp_path / "w.safetensors", expected_shapes(True)))
    e = entries["model.layers.0.self_attn.q_proj.weight"]
    assert (e.dtype, e.shape, e.end - e.begin) == ("bf16", (16, HIDDEN), 16 * HIDDEN * 2)


def test_mapping_covers_every_tensor_exactly_once_and_contiguously(tmp_path):
    entries = read_safetensors_header(write_safetensors(tmp_path / "w.safetensors", expected_shapes(True)))
    layers = map_qwen2_layers(entries, CFG, tie_word_embeddings=True)
    assert len(layers) == len(entries)
    ranges = sorted((l.byte_offset, l.byte_offset + l.byte_length) for l in layers)
    assert ranges[0][0] == 0 and all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert [l.name for l in layers][:2] == ["embed_tokens.weight", "layers.0.input_layernorm.weight"]
    assert layers[-1].name == "norm.weight" and layers[-1].index == CFG.num_layers + 1


def test_untied_model_requires_lm_head(tmp_path):
    entries = read_safetensors_header(write_safetensors(tmp_path / "w.safetensors", expected_shapes(True)))
    with pytest.raises(ValueError, match="lm_head.weight"):
        map_qwen2_layers(entries, CFG, tie_word_embeddings=False)


def test_missing_tensor_is_reported(tmp_path):
    shapes = expected_shapes(True)
    del shapes["model.layers.1.mlp.up_proj.weight"]
    entries = read_safetensors_header(write_safetensors(tmp_path / "w.safetensors", shapes))
    with pytest.raises(ValueError, match="missing.*model.layers.1.mlp.up_proj.weight"):
        map_qwen2_layers(entries, CFG, tie_word_embeddings=True)


def test_unmapped_tensor_is_reported(tmp_path):
    shapes = expected_shapes(True)
    shapes["model.layers.0.self_attn.rotary_emb.inv_freq"] = [2]
    entries = read_safetensors_header(write_safetensors(tmp_path / "w.safetensors", shapes))
    with pytest.raises(ValueError, match="unmapped.*rotary_emb.inv_freq"):
        map_qwen2_layers(entries, CFG, tie_word_embeddings=True)


def test_wrong_shape_is_reported(tmp_path):
    shapes = expected_shapes(True)
    shapes["model.layers.0.self_attn.k_proj.weight"] = [16, HIDDEN]  # kv rows must be kv_heads*head_dim = 8
    entries = read_safetensors_header(write_safetensors(tmp_path / "w.safetensors", shapes))
    with pytest.raises(ValueError, match="k_proj.weight.*shape"):
        map_qwen2_layers(entries, CFG, tie_word_embeddings=True)


def test_unsupported_dtype_is_refused(tmp_path):
    path = write_safetensors(tmp_path / "w.safetensors", expected_shapes(True), dtype="F32")
    with pytest.raises(ValueError, match="F32"):
        read_safetensors_header(path)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda h: h["model.norm.weight"].update(data_offsets=[0, 32]), "overlap"),
        (lambda h: h["model.norm.weight"].update(data_offsets=[10, 5]), "range"),
        (lambda h: h["model.norm.weight"].update(data_offsets=[10, 11]), "bytes"),
    ],
)
def test_corrupt_headers_are_rejected(tmp_path, mutate, message):
    path = write_safetensors(tmp_path / "w.safetensors", expected_shapes(True), mutate=mutate)
    with pytest.raises(ValueError, match=message):
        read_safetensors_header(path)


def test_oversized_header_length_is_rejected_before_reading(tmp_path):
    path = tmp_path / "w.safetensors"
    path.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(ValueError, match="header"):
        read_safetensors_header(path)


def model_header(model_id: str) -> Path:
    path = TESTDATA / "models" / model_id / "model.safetensors.header"
    if not path.is_file():
        message = f"{model_id}: run `python scripts/fetch-model-fixtures.py {model_id}`"
        if os.environ.get("HYPERION_REQUIRE_MODEL_FIXTURES"):
            pytest.fail(message)
        pytest.skip(message)
    return path


@pytest.mark.parametrize("model_id", [m for m, s in REGISTRY.items() if "model.safetensors.header" in s["files"]])
def test_real_header_maps_every_tensor(model_id):
    from hyperion_ignition.compiler.model_config import model_config_from_hf

    cfg_dir = TESTDATA / "models" / model_id
    # The fixture check comes first: without the downloaded files this test skips visibly, it does not fail.
    header = model_header(model_id)
    qwen3 = REGISTRY[model_id]["compiler"] == "qwen3"  # no QKV bias, per-head Q/K norm weights
    cfg = model_config_from_hf(cfg_dir, qkv_bias=not qwen3, qk_norm=qwen3)
    entries = read_safetensors_header(header)
    layers = map_qwen2_layers(entries, cfg, tie_word_embeddings=cfg.tie_word_embeddings)
    assert len(layers) == len(entries)
    ranges = sorted((l.byte_offset, l.byte_offset + l.byte_length) for l in layers)
    assert ranges[0][0] == 0 and all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
