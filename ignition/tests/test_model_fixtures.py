"""Generic checks over every real model listed in testdata/models.json.

Nothing here names a model: the registry says which Ignition compiler must
accept a config (`compiler`), or `null` when none supports it yet, in which case
every compiler must refuse it (a compiler must not emit a manifest for a model
family it does not implement). Model files are not committed; fetch them with
`python scripts/fetch-model-fixtures.py`. Missing files skip with a visible
reason unless HYPERION_REQUIRE_MODEL_FIXTURES=1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from hyperion_ignition.cli import main
from hyperion_ignition.compiler.model_config import model_config_from_hf
from hyperion_ignition.compiler.registry import _ARCH_MAP, build_compiler
from hyperion_ignition.compiler.weights import SHARD_INDEX, map_qwen2_layers, read_checkpoint
from hyperion_ignition.hardware.registry import build_recipe
from hyperion_ignition.manifest_schema import validate_manifest

TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
REGISTRY = json.loads((TESTDATA / "models.json").read_text(encoding="utf-8"))["models"]


def model_dir(model_id: str) -> Path:
    path = TESTDATA / "models" / model_id
    if not (path / "config.json").is_file():
        message = f"{model_id}: run `python scripts/fetch-model-fixtures.py {model_id}`"
        if os.environ.get("HYPERION_REQUIRE_MODEL_FIXTURES"):
            pytest.fail(message)
        pytest.skip(message)
    return path


def model_ids(supported: bool):
    return [m for m, spec in REGISTRY.items() if (spec["compiler"] is not None) == supported]


@pytest.mark.parametrize("model_id", model_ids(supported=True))
def test_supported_model_compiles_with_its_compiler(model_id):
    compiler_id = REGISTRY[model_id]["compiler"]
    if _sharded_headers_only(model_dir(model_id)):
        # Only the shard headers were fetched. A package must record each shard's size and sha256,
        # so the compiler has to refuse rather than write a manifest it cannot describe; the mapping
        # of tensors to shards is checked by the next test.
        with pytest.raises(ValueError, match="missing"):
            build_compiler(compiler_id).compile(source=model_dir(model_id), recipe=build_recipe("sm_90"))
        return
    manifest = build_compiler(compiler_id).compile(source=model_dir(model_id), recipe=build_recipe("sm_90"))
    assert manifest.model_id == compiler_id
    # With a weights header present the manifest carries the layer map, and it must
    # satisfy the Rust-generated schema.
    if (model_dir(model_id) / "model.safetensors.header").is_file():
        assert manifest.layers, "weights present but no layers mapped"
        validate_manifest(manifest.to_json_dict())


def _sharded_headers_only(source: Path) -> bool:
    return (source / SHARD_INDEX).is_file() and not list(source.glob("*.safetensors"))


def _sharded_models():
    return [m for m in model_ids(supported=True) if SHARD_INDEX in REGISTRY[m]["files"]]


@pytest.mark.parametrize("model_id", _sharded_models())
def test_every_tensor_of_a_sharded_model_maps_to_the_shard_its_index_names(model_id):
    """Uses only the fetched shard headers: the layer map must cover every tensor of the index, each one
    pointing at the shard the index says, with the number of payloads equal to the number of shards."""
    source = model_dir(model_id)
    compiler_id = REGISTRY[model_id]["compiler"]
    assert compiler_id == "qwen2", "only the qwen2 weight mapper exists so far"
    checkpoint = read_checkpoint(source, hash_payloads=False)
    assert checkpoint is not None and checkpoint.payloads == []
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    layers = map_qwen2_layers(checkpoint.entries, model_config_from_hf(source, ("Qwen2ForCausalLM",)), tie_word_embeddings=bool(config.get("tie_word_embeddings", False)))
    weight_map = json.loads((source / SHARD_INDEX).read_text(encoding="utf-8"))["weight_map"]
    shards = sorted(set(weight_map.values()))
    assert len(layers) == len(weight_map), "every tensor of the index is mapped"
    for layer in layers:
        hf_name = layer.name if layer.name.startswith("lm_head") else "model." + layer.name
        assert shards[layer.payload] == weight_map[hf_name], layer.name
    assert {layer.payload for layer in layers} == set(range(len(shards)))


@pytest.mark.parametrize("compiler_id", sorted(_ARCH_MAP))
@pytest.mark.parametrize("model_id", model_ids(supported=False))
def test_unsupported_model_is_refused_by_every_compiler(model_id, compiler_id):
    if compiler_id == "deepseek":
        pytest.skip("deepseek refuses everything (MLA unsupported)")
    with pytest.raises(ValueError, match="not supported by this compiler"):
        build_compiler(compiler_id).compile(source=model_dir(model_id), recipe=build_recipe("sm_90"))


@pytest.mark.parametrize("model_id", model_ids(supported=False))
def test_cli_refuses_an_unsupported_model_and_writes_nothing(model_id, tmp_path):
    out = tmp_path / "model.hyperion"
    result = CliRunner().invoke(
        main,
        ["--model", "qwen2", "--hw", "sm_90", "--out", str(out), "--source", str(model_dir(model_id))],
    )
    assert result.exit_code != 0
    assert "supported:" in result.output
    assert not out.with_suffix(".manifest.json").exists()
