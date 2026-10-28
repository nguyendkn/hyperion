"""Hardware recipe base. A recipe is the compile-time constraint set for one
device class — dtype, whether CUDA Graph capture is allowed, FP4 enable, etc.
Adding a recipe = one file + register line in `registry.py`."""

from __future__ import annotations

from typing import Protocol


class HardwareRecipe(Protocol):
    target_arch: str
    supported_formats: tuple[str, ...]  # formats this device can run; not what a package holds

    def constraints(self) -> dict[str, str]:
        """k=v flags the backend later interprets (dtype, graph capture, ...)."""
        ...
