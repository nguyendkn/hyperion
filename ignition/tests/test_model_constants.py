"""The manifest carries every constant the runtime uses (model shape, RoPE, norm epsilon, tied head, the variants of
each operation), the tokens that end generation, the chat template and the checkpoint's provenance. A value the
source files do not state is an error here, never a default, and a variant the runtime does not implement is
refused at export. Real Hugging Face files from the registry are used when downloaded; without them those tests
skip visibly (`HYPERION_REQUIRE_MODEL_FIXTURES=1` turns the skip into a failure)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from hyperion_ignition.cli import main
from hyperion_ignition.compiler.base import CompiledManifest
from hyperion_ignition.compiler.generation import QualifiedCheckpoint, read_chat_template, read_generation
from hyperion_ignition.compiler.model_config import model_config_from_hf
from hyperion_ignition.compiler.registry import build_compiler
from hyperion_ignition.hardware.registry import build_recipe
from hyperion_ignition.manifest_schema import validate_manifest

TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
REGISTRY = json.loads((TESTDATA / "models.json").read_text(encoding="utf-8"))["models"]
REVISION = "0123456789abcdef0123456789abcdef01234567"


def model_dir(model_id: str) -> Path:
    path = TESTDATA / "models" / model_id
    needed = ("config.json", "generation_config.json")
    if not all((path / name).is_file() for name in needed):
        message = f"{model_id}: run `python scripts/fetch-model-fixtures.py {model_id}`"
        if os.environ.get("HYPERION_REQUIRE_MODEL_FIXTURES"):
            pytest.fail(message)
        pytest.skip(message)
    return path


def hf(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


ARCHITECTURES = {"qwen2": (("Qwen2ForCausalLM",), True, False), "qwen3": (("Qwen3ForCausalLM",), False, True)}


def compile_registry_model(model_id: str) -> dict:
    """The manifest a registry model's source yields for everything but the weights (which need the downloaded shards)."""
    source = model_dir(model_id)
    architectures, qkv_bias, qk_norm = ARCHITECTURES["qwen3" if model_id.startswith("qwen3") else "qwen2"]
    document = CompiledManifest(
        model_id=model_id, target_arch="sm_90", tensor_format="bf16",
        model=model_config_from_hf(source, architectures, qkv_bias=qkv_bias, qk_norm=qk_norm),
        tokenizer={"family": "bpe", "asset_path": "tokenizer.json"},
        generation=read_generation(source), chat_template=read_chat_template(source),
    ).to_json_dict()
    validate_manifest(document)
    return document


def test_the_qwen2_compiler_writes_the_same_constants(tmp_path):
    source = model_dir("qwen2.5-0.5b")
    if not (source / "model.safetensors.header").is_file():
        pytest.skip("qwen2.5-0.5b header not downloaded")
    compiled = build_compiler("qwen2").compile(source=source, recipe=build_recipe("sm_90")).to_json_dict()
    assert compiled["model"] == compile_registry_model("qwen2.5-0.5b")["model"]
    assert compiled["generation"] == compile_registry_model("qwen2.5-0.5b")["generation"]


@pytest.mark.parametrize("model_id", sorted(REGISTRY))
def test_every_constant_in_the_manifest_equals_the_models_config_json(model_id):
    document = compile_registry_model(model_id)
    config = hf(model_dir(model_id) / "config.json")
    model = document["model"]
    assert model["hidden_size"] == config["hidden_size"]
    assert model["intermediate_size"] == config["intermediate_size"]
    assert model["rope_theta"] == float(config["rope_theta"])
    assert model["rms_norm_eps"] == config["rms_norm_eps"]
    assert model["tie_word_embeddings"] is config["tie_word_embeddings"]
    assert (model["activation"], model["rope_style"]) == ("silu", "rotate_half")
    assert document["contract_revision"] == 1
    # The one place the runtime may not look at config.json again is the manifest: write it where the Rust test reads it.
    (model_dir(model_id) / "constants.manifest.json").write_text(json.dumps(document, indent=2), encoding="utf-8")


def test_qwen2_models_have_qkv_bias_and_no_qk_norm_and_qwen3_the_opposite():
    for model_id in REGISTRY:
        model = compile_registry_model(model_id)["model"]
        if model_id.startswith("qwen3"):
            assert (model["qk_norm"], model["qkv_bias"]) == (True, False), model_id
        else:
            assert (model["qk_norm"], model["qkv_bias"]) == (False, True), model_id


def test_eos_is_the_deduplicated_union_of_config_and_generation_config():
    config = hf(model_dir("qwen3-0.6b") / "config.json")
    generation = hf(model_dir("qwen3-0.6b") / "generation_config.json")
    assert config["eos_token_id"] == 151645 and generation["eos_token_id"] == [151645, 151643]
    document = compile_registry_model("qwen3-0.6b")
    assert document["generation"]["eos_token_ids"] == [151645, 151643]
    assert document["generation"]["sampling_defaults"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    # Qwen2.5 base: one eos, `do_sample: false` and no sampling recommendation.
    base = compile_registry_model("qwen2.5-0.5b")["generation"]
    assert base == {"eos_token_ids": [151643]}


def test_the_chat_template_text_is_in_the_manifest_not_a_path():
    source = model_dir("qwen2.5-0.5b")
    if not (source / "tokenizer_config.json").is_file():
        pytest.skip("qwen2.5-0.5b tokenizer_config.json not downloaded")
    template = compile_registry_model("qwen2.5-0.5b")["chat_template"]
    tokenizer_config = hf(source / "tokenizer_config.json")
    assert template["template"] == tokenizer_config["chat_template"]
    assert (template["bos_token"], template["eos_token"]) == ("", tokenizer_config["eos_token"])
    assert "\n" in template["template"] and not template["template"].endswith(".jinja")


BASE = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "vocab_size": 10,
        "max_position_embeddings": 16, "rope_theta": 10000.0, "rms_norm_eps": 1e-06, "tie_word_embeddings": False,
        "hidden_act": "silu", "eos_token_id": 3}


def write(tmp_path: Path, name: str, document: dict) -> None:
    (tmp_path / name).write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize("missing", ["hidden_size", "intermediate_size", "rope_theta", "rms_norm_eps", "tie_word_embeddings", "hidden_act"])
def test_a_constant_the_config_does_not_state_is_an_error_not_a_default(tmp_path, missing):
    write(tmp_path, "config.json", {k: v for k, v in BASE.items() if k != missing})
    with pytest.raises(ValueError):
        model_config_from_hf(tmp_path, qkv_bias=True)


def test_transformers_5_rope_parameters_are_read_and_a_scaled_type_refused(tmp_path):
    config = {k: v for k, v in BASE.items() if k != "rope_theta"}
    write(tmp_path, "config.json", {**config, "rope_parameters": {"rope_theta": 5000.0, "rope_type": "default"}})
    assert model_config_from_hf(tmp_path, qkv_bias=True).rope_theta == 5000.0
    write(tmp_path, "config.json", {**config, "rope_parameters": {"rope_theta": 5000.0, "rope_type": "yarn"}})
    with pytest.raises(ValueError, match="rope_type"):
        model_config_from_hf(tmp_path, qkv_bias=True)


def test_query_pre_attn_scalar_becomes_the_attention_scale(tmp_path):
    write(tmp_path, "config.json", {**BASE, "query_pre_attn_scalar": 256})
    assert model_config_from_hf(tmp_path, qkv_bias=True).attention_scale == pytest.approx(1 / 16)
    write(tmp_path, "config.json", BASE)
    assert model_config_from_hf(tmp_path, qkv_bias=True).attention_scale is None


def test_a_model_with_no_eos_anywhere_is_refused(tmp_path):
    write(tmp_path, "config.json", {k: v for k, v in BASE.items() if k != "eos_token_id"})
    with pytest.raises(ValueError, match="eos_token_id"):
        read_generation(tmp_path)
    write(tmp_path, "generation_config.json", {"eos_token_id": [7, 3]})
    write(tmp_path, "config.json", BASE)
    assert read_generation(tmp_path)["eos_token_ids"] == [3, 7]


def test_a_template_list_is_refused_and_no_template_is_none(tmp_path):
    assert read_chat_template(tmp_path) is None
    write(tmp_path, "tokenizer_config.json", {"eos_token": "<e>"})
    assert read_chat_template(tmp_path) is None
    write(tmp_path, "tokenizer_config.json", {"chat_template": [{"name": "default", "template": "x"}]})
    with pytest.raises(ValueError, match="one non-empty string"):
        read_chat_template(tmp_path)
    write(tmp_path, "tokenizer_config.json", {"chat_template": "{{ bos_token }}x", "bos_token": {"content": "<s>"}, "eos_token": "</s>"})
    assert read_chat_template(tmp_path) == {"template": "{{ bos_token }}x", "bos_token": "<s>", "eos_token": "</s>"}


def test_the_checkpoint_provenance_is_validated_and_written():
    assert QualifiedCheckpoint("Qwen/Qwen2.5-7B", REVISION).to_json_dict() == {"repo": "Qwen/Qwen2.5-7B", "revision": REVISION}
    for repo, revision in [("noslash", REVISION), ("a/b/c", REVISION), ("Qwen/Qwen2.5-7B", "main"), ("Qwen/Qwen2.5-7B", REVISION.upper())]:
        with pytest.raises(ValueError):
            QualifiedCheckpoint(repo, revision)


def test_the_cli_takes_repo_and_revision_together(tmp_path):
    source = Path(__file__).resolve().parent / "fixtures" / "qwen2-hf"
    out = tmp_path / "model.hyperion"
    runner = CliRunner()
    base = ["--model", "qwen2", "--hw", "sm_90", "--source", str(source), "--out", str(out)]
    assert runner.invoke(main, base + ["--hf-repo", "Qwen/Qwen2-7B"]).exit_code != 0
    result = runner.invoke(main, base + ["--hf-repo", "Qwen/Qwen2-7B", "--hf-revision", REVISION])
    assert result.exit_code == 0, result.output
    document = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert document["qualified_checkpoint"] == {"repo": "Qwen/Qwen2-7B", "revision": REVISION}
    result = runner.invoke(main, base)
    assert result.exit_code == 0
    assert "qualified_checkpoint" not in json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
