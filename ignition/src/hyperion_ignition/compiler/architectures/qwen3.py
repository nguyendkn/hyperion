"""Qwen3-family compiler: the Qwen2 decoder without the Q/K/V biases and with a RMSNorm over each head of Q and K
(`qk_norm`). The manifest says so (`qkv_bias=false`, `qk_norm=true`); a runtime that does not implement the per-head
norm refuses the package with a typed error (the sm90 forward pass does not yet)."""

from __future__ import annotations

import json
from pathlib import Path

from ..base import CompiledManifest, HardwareRecipe, packaged_format
from ..generation import QualifiedCheckpoint, read_chat_template, read_generation
from ..model_config import model_config_from_hf
from ..registry import register
from ..weights import map_qwen2_layers, read_checkpoint


class Qwen3Compiler:
    model_id = "qwen3"
    architectures = ("Qwen3ForCausalLM",)

    def compile(self, *, source: Path, recipe: HardwareRecipe, checkpoint: QualifiedCheckpoint | None = None) -> CompiledManifest:
        # Q, K and V have a bias only when the config says `attention_bias` (Qwen3 ships false); Q and K are always normalised.
        raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
        attention_bias = raw.get("attention_bias", False)
        if not isinstance(attention_bias, bool):
            raise ValueError(f"config.json: attention_bias must be true or false, got {attention_bias!r}")
        config = model_config_from_hf(source, self.architectures, qkv_bias=attention_bias, qk_norm=True)
        weights = read_checkpoint(source)
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
            qualified_checkpoint=checkpoint,
            layers=layers,
            payloads=payloads,
            hardware_recipe=recipe.constraints(),
        )


register("qwen3", Qwen3Compiler)
