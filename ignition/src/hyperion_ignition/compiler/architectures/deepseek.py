"""DeepSeek-family compiler. Not implemented.

DeepSeek V2/V3 use multi-head latent attention (MLA): the KV cache holds a
compressed latent, not `num_kv_heads x head_dim` tensors. Manifest v1's
`ModelConfig` cannot describe that, and guessing head dimensions would size the
KV cache wrongly. Supporting it needs a schema decision first.
"""

from __future__ import annotations

from pathlib import Path

from ..base import CompiledManifest, HardwareRecipe
from ..registry import register


class DeepSeekCompiler:
    model_id = "deepseek"

    def compile(self, *, source: Path, recipe: HardwareRecipe) -> CompiledManifest:
        raise NotImplementedError(
            "deepseek: MLA KV geometry is not representable by manifest v1 ModelConfig"
        )


register("deepseek", DeepSeekCompiler)
