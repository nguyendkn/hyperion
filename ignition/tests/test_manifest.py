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
        model=ModelConfig(32, 32, 8, 128, 128256, 8192),
        tokenizer={"family": "bpe", "asset_path": "tokenizer.json"},
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
    ],
)
def test_schema_rejects_invalid_manifests(mutate):
    document = copy.deepcopy(compile_manifest("llama3", "sm_90", "llama3-hf"))
    mutate(document)
    with pytest.raises(ManifestValidationError):
        validate_manifest(document)


def test_model_config_derived_from_hf_config():
    cfg = model_config_from_hf(FIXTURES / "llama3-hf")
    assert cfg == ModelConfig(32, 32, 8, 128, 128256, 8192)


def test_model_config_defaults_and_errors(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                    "vocab_size": 10, "max_position_embeddings": 16}),
        encoding="utf-8",
    )
    cfg = model_config_from_hf(tmp_path)
    assert (cfg.num_kv_heads, cfg.head_dim) == (4, 16)  # MHA default, hidden // heads

    (tmp_path / "config.json").write_text(json.dumps({"num_attention_heads": 4}), encoding="utf-8")
    with pytest.raises(ValueError):
        model_config_from_hf(tmp_path)
    with pytest.raises(FileNotFoundError):
        model_config_from_hf(tmp_path / "missing")


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
