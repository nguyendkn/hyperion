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
from hyperion_ignition.compiler.registry import _ARCH_MAP, build_compiler
from hyperion_ignition.hardware.registry import build_recipe

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
    manifest = build_compiler(compiler_id).compile(source=model_dir(model_id), recipe=build_recipe("sm_90"))
    assert manifest.model_id == compiler_id


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
