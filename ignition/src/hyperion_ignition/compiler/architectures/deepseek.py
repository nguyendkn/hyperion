"""DeepSeek-family compiler. Skeleton symmetric to LlamaCompiler."""

from __future__ import annotations

from pathlib import Path

from ..base import CompiledManifest, HardwareRecipe
from ..registry import register


class DeepSeekCompiler:
    model_id = "deepseek"

    def compile(self, *, source: Path, recipe: HardwareRecipe) -> CompiledManifest:
        return CompiledManifest(
            model_id=self.model_id,
            target_arch=recipe.target_arch,
            tokenizer={"family": "bpe", "asset_path": "tokenizer.json", "max_seq_len": 65536},
            layers=[],
            hardware_recipe=recipe.constraints(),
        )


register("deepseek", DeepSeekCompiler)
