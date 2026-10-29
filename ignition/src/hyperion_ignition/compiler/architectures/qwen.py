"""Qwen-family compiler. Skeleton symmetric to LlamaCompiler."""

from __future__ import annotations

import json
from pathlib import Path

from ..base import CompiledManifest, HardwareRecipe, packaged_format
from ..model_config import model_config_from_hf
from ..registry import register
from ..weights import map_qwen2_layers, read_safetensors_header

# A full checkpoint or a header-only file; only the header is read either way.
WEIGHT_FILES = ("model.safetensors", "model.safetensors.header")
SHARD_INDEX = "model.safetensors.index.json"


class QwenCompiler:
    model_id = "qwen2"
    architectures = ("Qwen2ForCausalLM",)

    def compile(self, *, source: Path, recipe: HardwareRecipe) -> CompiledManifest:
        config = model_config_from_hf(source, self.architectures)
        weights = next((source / n for n in WEIGHT_FILES if (source / n).is_file()), None)
        if weights is None and (source / SHARD_INDEX).is_file():
            # Refuse rather than write a package with no layers: a sharded checkpoint (several
            # model-0000N-of-0000M.safetensors files plus an index) is not supported yet, because
            # the manifest records offsets inside a single weights file.
            raise ValueError(f"{source.name}: sharded checkpoints ({SHARD_INDEX}) are not supported yet; the manifest has no way to say which file holds a tensor")
        layers = []
        if weights is not None:
            tied = bool(json.loads((source / "config.json").read_text(encoding="utf-8")).get("tie_word_embeddings", False))
            layers = map_qwen2_layers(read_safetensors_header(weights), config, tie_word_embeddings=tied)
        return CompiledManifest(
            model_id=self.model_id,
            target_arch=recipe.target_arch,
            tensor_format=packaged_format(recipe),
            model=config,
            tokenizer={"family": "bpe", "asset_path": "tokenizer.json"},
            layers=layers,
            hardware_recipe=recipe.constraints(),
        )


register("qwen2", QwenCompiler)
