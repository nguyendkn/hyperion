"""Qwen-family compiler. Skeleton symmetric to LlamaCompiler."""

from __future__ import annotations

from pathlib import Path

from ..base import CompiledManifest, HardwareRecipe, packaged_format
from ..generation import QualifiedCheckpoint, read_chat_template, read_generation
from ..model_config import model_config_from_hf
from ..registry import register
from ..weights import map_qwen2_layers, read_checkpoint



class QwenCompiler:
    model_id = "qwen2"
    architectures = ("Qwen2ForCausalLM",)

    def compile(self, *, source: Path, recipe: HardwareRecipe, checkpoint: QualifiedCheckpoint | None = None) -> CompiledManifest:
        # Qwen2 puts a bias on Q, K and V and has no QK-norm; both are properties of the architecture.
        config = model_config_from_hf(source, self.architectures, qkv_bias=True, qk_norm=False)
        # A single checkpoint file, a header-only file, or shards listed by an index; with real files the
        # package records each weights file's name, size and sha256.
        weights = read_checkpoint(source)
        qualified = checkpoint
        layers = []
        payloads = []
        if weights is not None:
            layers = map_qwen2_layers(weights.entries, config, tie_word_embeddings=config.tie_word_embeddings)
            payloads = weights.payloads
        return CompiledManifest(
            model_id=self.model_id,
            target_arch=recipe.target_arch,
            tensor_format=packaged_format(recipe),
            model=config,
            tokenizer={"family": "bpe", "asset_path": "tokenizer.json"},
            generation=read_generation(source),
            chat_template=read_chat_template(source),
            qualified_checkpoint=qualified,
            layers=layers,
            payloads=payloads,
            hardware_recipe=recipe.constraints(),
        )


register("qwen2", QwenCompiler)
