"""A sharded checkpoint must be refused, never compiled into a package that has no layers."""

import json
from pathlib import Path

import pytest

from hyperion_ignition.compiler.registry import build_compiler
from hyperion_ignition.hardware.registry import build_recipe


def _qwen2_config() -> dict:
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 1000,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
    }


def test_a_sharded_checkpoint_is_refused_with_a_clear_message(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(_qwen2_config()), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="sharded checkpoints"):
        build_compiler("qwen2").compile(source=tmp_path, recipe=build_recipe("sm_90"))
