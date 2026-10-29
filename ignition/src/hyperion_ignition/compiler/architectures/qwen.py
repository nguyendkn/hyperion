"""Qwen-family compiler. Skeleton symmetric to LlamaCompiler."""

from __future__ import annotations

import json
from pathlib import Path

from ..base import CompiledManifest, HardwareRecipe, packaged_format
from ..model_config import model_config_from_hf
from ..registry import register
from ..weights import map_qwen2_layers, read_checkpoint



class QwenCompiler:
    model_id = "qwen2"
    architectures = ("Qwen2ForCausalLM",)

    def compile(self, *, source: Path, recipe: HardwareRecipe) -> CompiledManifest:
        config = model_config_from_hf(source, self.architectures)
        # A single checkpoint file, a header-only file, or shards listed by an index; with real files the
        # package records each weights file's name, size and sha256.
        checkpoint = read_checkpoint(source)
        layers = []
        payloads = []
        if checkpoint is not None:
            tied = bool(json.loads((source / "config.json").read_text(encoding="utf-8")).get("tie_word_embeddings", False))
            layers = map_qwen2_layers(checkpoint.entries, config, tie_word_embeddings=tied)
            payloads = checkpoint.payloads
        return CompiledManifest(
            model_id=self.model_id,
            target_arch=recipe.target_arch,
            tensor_format=packaged_format(recipe),
            model=config,
            tokenizer={"family": "bpe", "asset_path": "tokenizer.json"},
            layers=layers,
            payloads=payloads,
            hardware_recipe=recipe.constraints(),
        )


register("qwen2", QwenCompiler)
