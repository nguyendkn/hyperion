"""Blackwell SM120 recipe. Enables FP4/FP6 and CUDA Graph capture of the
static decode graph — the defining capabilities of the generation."""

from __future__ import annotations

from .registry import register


class Sm120BlackwellRecipe:
    target_arch = "sm_120"

    def constraints(self) -> dict[str, str]:
        return {
            "fp4": "enabled",
            "fp6": "enabled",
            "fp8": "enabled",
            "bf16": "enabled",
            "cuda_graphs": "enabled",  # native static graph capture on SM120
        }


register("sm_120", Sm120BlackwellRecipe())
