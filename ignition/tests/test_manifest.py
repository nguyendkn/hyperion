"""Manifest contract tests: Ignition output must satisfy the schema generated
from the Rust structs, and the Rust-owned fixture must satisfy it too.

Fixtures in `specification/fixtures/v1/` are shared with `cargo test -p
hyperion-specification`: Python writes `python-*.json`, Rust writes
`rust-sample.json`. Set `UPDATE_FIXTURES=1` to rewrite the Python-owned ones.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from hyperion_ignition.cli import main
from hyperion_ignition.compiler.base import CompiledManifest, LayerMap
from hyperion_ignition.compiler.model_config import ModelConfig, model_config_from_hf
from hyperion_ignition.compiler.registry import build_compiler
from hyperion_ignition.hardware.registry import build_recipe
from hyperion_ignition.manifest_schema import ManifestValidationError, validate_manifest

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
SHARED = HERE.parents[1] / "specification" / "fixtures" / "v1"


# Meta-Llama-3-8B and Qwen2.5-7B as their published config.json files state them.
LLAMA3_8B = ModelConfig(32, "gqa", 32, 8, 128, 128256, 8192, hidden_size=4096, intermediate_size=14336, rope_theta=500000.0,
                        rms_norm_eps=1e-05, tie_word_embeddings=False, qkv_bias=False)
QWEN2_7B = ModelConfig(28, "gqa", 28, 4, 128, 152064, 131072, hidden_size=3584, intermediate_size=18944, rope_theta=1000000.0,
                       rms_norm_eps=1e-06, tie_word_embeddings=False, qkv_bias=True)


def compile_manifest(model: str, arch: str, source: str) -> dict:
    manifest = build_compiler(model).compile(source=FIXTURES / source, recipe=build_recipe(arch))
    return manifest.to_json_dict()


def check_golden(name: str, document: dict) -> None:
    path = SHARED / name
    if os.environ.get("UPDATE_FIXTURES"):
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == document, (
        f"{name} is stale: run with UPDATE_FIXTURES=1 and re-run cargo test"
    )


@pytest.mark.parametrize("arch", ["sm_90", "sm_120"])
@pytest.mark.parametrize("model,source", [("llama3", "llama3-hf"), ("qwen2", "qwen2-hf")])
def test_compiler_output_satisfies_schema(model, source, arch):
    validate_manifest(compile_manifest(model, arch, source))


def test_golden_fixtures_shared_with_rust():
    check_golden("python-llama3-sm_90.json", compile_manifest("llama3", "sm_90", "llama3-hf"))
    with_layer = CompiledManifest(
        model_id="llama3",
        target_arch="sm_120",
        tensor_format="fp4",
        model=LLAMA3_8B,
        tokenizer={"family": "bpe", "asset_path": "tokenizer.json"},
        generation={"eos_token_ids": [128001]},
        layers=[LayerMap(0, "attention.q_proj", 0, 33554432, "bf16", (4096, 4096))],
        hardware_recipe={"fp4": "enabled"},
        signature=[1, 2, 3],
    ).to_json_dict()
    validate_manifest(with_layer)
    check_golden("python-with-layer-and-signature.json", with_layer)


def test_rust_written_fixture_satisfies_schema():
    validate_manifest(json.loads((SHARED / "rust-sample.json").read_text(encoding="utf-8")))


def test_tag_is_the_string_v1():
    assert compile_manifest("llama3", "sm_90", "llama3-hf")["manifest_version"] == "V1"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(manifest_version=1),
        lambda m: m.update(extra=1),
        lambda m: m["model"].update(extra=1),
        lambda m: m["tokenizer"].update(max_seq_len=8192),
        lambda m: m.pop("model"),
        lambda m: m["model"].pop("vocab_size"),
        lambda m: m["model"].update(num_layers=-1),
        lambda m: m.update(tensor_format="int3"),
        lambda m: m.pop("tensor_format"),
        lambda m: m["model"].update(attention="mla"),
    ],
)
def test_schema_rejects_invalid_manifests(mutate):
    document = copy.deepcopy(compile_manifest("llama3", "sm_90", "llama3-hf"))
    mutate(document)
    with pytest.raises(ManifestValidationError):
        validate_manifest(document)


def test_model_config_derived_from_real_hf_configs():
    # Values from the published config.json files (Meta-Llama-3-8B, Qwen2-7B).
    assert model_config_from_hf(FIXTURES / "llama3-hf", qkv_bias=False) == LLAMA3_8B
    assert model_config_from_hf(FIXTURES / "qwen2-hf", qkv_bias=True) == QWEN2_7B


@pytest.mark.parametrize("arch", ["sm_90", "sm_120"])
def test_tensor_format_describes_what_is_packaged(arch):
    # No quantizer exists: the label is bf16 on every target, and the recipe
    # only says which formats the device could run.
    assert compile_manifest("llama3", arch, "llama3-hf")["tensor_format"] == "bf16"


def test_hopper_recipe_never_allows_fp4():
    assert "fp4" not in build_recipe("sm_90").supported_formats
    assert "fp4" in build_recipe("sm_120").supported_formats


BASE_HF_CONFIG = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
                  "vocab_size": 10, "max_position_embeddings": 16, "rope_theta": 10000.0, "rms_norm_eps": 1e-06,
                  "tie_word_embeddings": False, "hidden_act": "silu"}


def hf_config(tmp_path, **overrides):
    cfg = {**BASE_HF_CONFIG, **overrides}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return tmp_path


def test_null_and_false_scaling_fields_are_accepted(tmp_path):
    # Checked by value, not presence: Llama-3 ships `rope_scaling: null`,
    # Qwen2 ships `use_sliding_window: false`.
    cfg = model_config_from_hf(hf_config(tmp_path, rope_scaling=None, use_sliding_window=False), qkv_bias=True)
    assert cfg.max_seq_len == 16


def test_explicit_head_dim_wins_over_hidden_size_division(tmp_path):
    assert model_config_from_hf(hf_config(tmp_path, head_dim=32), qkv_bias=True).head_dim == 32


@pytest.mark.parametrize(
    "overrides",
    [
        {"rope_scaling": {"type": "linear", "factor": 4.0}},
        {"use_sliding_window": True},
        {"num_key_value_heads": 3},  # does not divide 4 heads
        {"num_key_value_heads": 8},  # more kv heads than heads
        {"hidden_act": "gelu"},
        {"partial_rotary_factor": 0.5},
        {"rope_parameters": {"rope_type": "yarn", "rope_theta": 10000.0}, "rope_theta": None},
    ],
)
def test_unsupported_hf_configs_are_refused(tmp_path, overrides):
    with pytest.raises(ValueError):
        model_config_from_hf(hf_config(tmp_path, **overrides), qkv_bias=True)


def test_model_config_defaults_and_errors(tmp_path):
    cfg = model_config_from_hf(hf_config(tmp_path), qkv_bias=True)
    assert (cfg.attention, cfg.num_kv_heads, cfg.head_dim) == ("mha", 4, 16)  # hidden // heads

    (tmp_path / "config.json").write_text(json.dumps({"num_attention_heads": 4}), encoding="utf-8")
    with pytest.raises(ValueError):
        model_config_from_hf(tmp_path, qkv_bias=True)
    with pytest.raises(FileNotFoundError):
        model_config_from_hf(tmp_path / "missing", qkv_bias=True)


def test_deepseek_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError):
        build_compiler("deepseek").compile(source=FIXTURES, recipe=build_recipe("sm_90"))


def test_cli_writes_validated_manifest(tmp_path):
    out = tmp_path / "out" / "model.hyperion"
    result = CliRunner().invoke(
        main,
        ["--model", "llama3", "--hw", "sm_120", "--out", str(out),
         "--source", str(FIXTURES / "llama3-hf")],
    )
    assert result.exit_code == 0, result.output
    document = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    validate_manifest(document)
    assert document["target_arch"] == "sm_120"


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "llama3-8b", "--hw", "sm_90"],  # unknown model id
        ["--model", "llama3", "--hw", "sm_0"],  # unknown hardware recipe
        ["--model", "deepseek", "--hw", "sm_90"],  # unsupported architecture
    ],
)
def test_cli_fails_cleanly_and_writes_nothing(tmp_path, args):
    out = tmp_path / "model.hyperion"
    result = CliRunner().invoke(
        main, [*args, "--out", str(out), "--source", str(FIXTURES / "llama3-hf")]
    )
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert not out.with_suffix(".manifest.json").exists()
