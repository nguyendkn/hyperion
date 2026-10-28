"""A compiler must refuse an HF config whose `architectures` it does not handle.

Found on a real run: Ignition compiled a Qwen3 config under `--model qwen2`
and wrote `model_id: qwen2`, though Qwen3 adds QK-norm. The checks against real
third-party configs are generic and live in test_model_fixtures.py; this file
covers the committed fixtures and synthetic configs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperion_ignition.compiler.registry import build_compiler
from hyperion_ignition.hardware.registry import build_recipe

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("model,source", [("llama3", "llama3-hf"), ("qwen2", "qwen2-hf")])
def test_real_supported_configs_are_still_accepted(model, source):
    manifest = build_compiler(model).compile(source=FIXTURES / source, recipe=build_recipe("sm_90"))
    assert manifest.model_id == model


def test_config_without_architectures_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                    "vocab_size": 10, "max_position_embeddings": 16}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="architectures"):
        build_compiler("qwen2").compile(source=tmp_path, recipe=build_recipe("sm_90"))


def test_config_naming_another_family_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["SomethingElseForCausalLM"], "hidden_size": 64,
                    "num_hidden_layers": 2, "num_attention_heads": 4, "vocab_size": 10,
                    "max_position_embeddings": 16}),
        encoding="utf-8",
    )
    for model in ("llama3", "qwen2"):
        with pytest.raises(ValueError, match="SomethingElseForCausalLM"):
            build_compiler(model).compile(source=tmp_path, recipe=build_recipe("sm_90"))
