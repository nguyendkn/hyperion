"""Sharding classes and supported tensor-parallel sizes written into the manifest (ADR 0005).

The numbers asserted here are asserted again in `specification/src/v1/parallelism.rs`: the two sides compute the sizes
independently and must agree. The registry models skip visibly when their `config.json` is not downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperion_ignition.compiler.base import supported_tp_sizes
from hyperion_ignition.compiler.model_config import model_config_from_hf
from hyperion_ignition.compiler.registry import build_compiler
from hyperion_ignition.hardware.registry import build_recipe
from hyperion_ignition.manifest_schema import validate_manifest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "testdata" / "synthetic" / "qwen2-tiny-sharded"


def _compile() -> dict:
    return build_compiler("qwen2").compile(source=FIXTURE, recipe=build_recipe("sm_90")).to_json_dict()


@pytest.mark.parametrize(
    "model, architectures, qk_norm, expected",
    [("qwen2.5-7b", ("Qwen2ForCausalLM",), False, [1, 2, 4]), ("qwen2.5-0.5b", ("Qwen2ForCausalLM",), False, [1, 2]), ("qwen3-0.6b", ("Qwen3ForCausalLM",), True, [1, 2, 4, 8])],
)
def test_supported_sizes_of_the_registry_models(model, architectures, qk_norm, expected):
    source = ROOT / "testdata" / "models" / model
    if not (source / "config.json").exists():
        pytest.skip(f"{model} config not downloaded (python scripts/fetch-model-fixtures.py {model})")
    config = model_config_from_hf(source, architectures, qkv_bias=not qk_norm, qk_norm=qk_norm)
    assert supported_tp_sizes(config) == expected


def test_every_tensor_of_the_fixture_has_the_class_its_role_needs():
    manifest = _compile()
    classes = {layer["name"]: layer["sharding"] for layer in manifest["layers"]}
    assert all(layer.get("sharding") for layer in manifest["layers"]), "every tensor is classified"
    for name, cls in classes.items():
        if name in ("embed_tokens.weight", "lm_head.weight"):
            assert cls == "vocab", name
        elif any(f".{p}." in name for p in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
            assert cls == "column", name
        elif ".o_proj." in name or ".down_proj." in name:
            assert cls == "row", name
        else:
            assert cls == "replicated", name


def test_the_manifest_lists_its_supported_sizes_and_satisfies_the_schema():
    manifest = _compile()
    validate_manifest(manifest)
    assert manifest["parallelism"]["supported_tp_sizes"][0] == 1


def test_a_manifest_with_an_unknown_class_is_refused_by_the_schema():
    manifest = _compile()
    manifest["layers"][0]["sharding"] = "diagonal"
    with pytest.raises(Exception):
        validate_manifest(manifest)
    manifest = json.loads(json.dumps(_compile()))
    manifest["parallelism"]["pipeline"] = 2
    with pytest.raises(Exception):
        validate_manifest(manifest)
