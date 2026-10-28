"""Llama-family compiler. Skeleton: the real pipeline (graph extraction,
weight remap, TensorRT plan build under `sm90` / CUDA Graph capture under
`sm120`) lands once the hardware recipes drive dtype selection."""

from __future__ import annotations

from pathlib import Path

from ..base import CompiledManifest, HardwareRecipe
from ..registry import register


class LlamaCompiler:
    model_id = "llama3"

    def compile(self, *, source: Path, recipe: HardwareRecipe) -> CompiledManifest:  # noqa: D401
        # TODO: load the HF checkpoint, remap tensors to the Helios layer scheme,
        #       write the weight blob, and emit one LayerMap per tensor using
        #       accumulating byte offsets. Recipe kicks in to choose dtype.
        return CompiledManifest(
            model_id=self.model_id,
            target_arch=recipe.target_arch,
            tokenizer={"family": "bpe", "asset_path": "tokenizer.json", "max_seq_len": 8192},
            layers=[],
            hardware_recipe=recipe.constraints(),
        )


register("llama3", LlamaCompiler)
