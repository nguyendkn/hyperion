"""Llama-family compiler. Skeleton: the real pipeline (graph extraction,
weight remap, TensorRT plan build under `sm90` / CUDA Graph capture under
`sm120`) lands once the hardware recipes drive dtype selection."""

from __future__ import annotations

from pathlib import Path

from ..base import CompiledManifest, HardwareRecipe, packaged_format
from ..generation import QualifiedCheckpoint, read_chat_template, read_generation
from ..model_config import model_config_from_hf
from ..registry import register


class LlamaCompiler:
    model_id = "llama3"
    architectures = ("LlamaForCausalLM",)

    def compile(self, *, source: Path, recipe: HardwareRecipe, checkpoint: QualifiedCheckpoint | None = None) -> CompiledManifest:  # noqa: D401
        # TODO: load the HF checkpoint, remap tensors to the Helios layer scheme,
        #       write the weight blob, and emit one LayerMap per tensor using
        #       accumulating byte offsets. Recipe kicks in to choose dtype.
        return CompiledManifest(
            model_id=self.model_id,
            target_arch=recipe.target_arch,
            tensor_format=packaged_format(recipe),
            model=model_config_from_hf(source, self.architectures, qkv_bias=False),
            tokenizer={"family": "bpe", "asset_path": "tokenizer.json"},
            generation=read_generation(source),
            chat_template=read_chat_template(source),
            qualified_checkpoint=checkpoint,
            layers=[],
            hardware_recipe=recipe.constraints(),
        )


register("llama3", LlamaCompiler)
