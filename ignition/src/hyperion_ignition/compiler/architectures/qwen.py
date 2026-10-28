"""Qwen-family compiler. Skeleton symmetric to LlamaCompiler."""

from __future__ import annotations

from pathlib import Path

from ..base import CompiledManifest, HardwareRecipe
from ..model_config import model_config_from_hf
from ..registry import register


class QwenCompiler:
    model_id = "qwen2"

    def compile(self, *, source: Path, recipe: HardwareRecipe) -> CompiledManifest:
        return CompiledManifest(
            model_id=self.model_id,
            target_arch=recipe.target_arch,
            tensor_format=recipe.tensor_format,
            model=model_config_from_hf(source),
            tokenizer={"family": "bpe", "asset_path": "tokenizer.json"},
            layers=[],
            hardware_recipe=recipe.constraints(),
        )


register("qwen2", QwenCompiler)
