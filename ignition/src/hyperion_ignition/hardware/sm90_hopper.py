"""Hopper SM90 recipe. Disables FP4 (no native support on Hopper), allows
FP8/BF16, forbids static CUDA Graph capture of FP4/FP6 shapes."""

from __future__ import annotations

from .registry import register


class Sm90HopperRecipe:
    target_arch = "sm_90"
    tensor_format = "fp8"  # Hopper has no FP4

    def constraints(self) -> dict[str, str]:
        return {
            "fp4": "disabled",
            "fp8": "enabled",
            "bf16": "enabled",
            "cuda_graphs": "disabled",  # FP4/FP6 graph capture isn't valid on SM90
        }


register("sm_90", Sm90HopperRecipe())
