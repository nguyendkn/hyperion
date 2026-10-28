"""Recipe registry mirror of the compiler registry: import the recipe
modules to trigger `register(...)`, then look up by `target_arch`."""

from __future__ import annotations

from .base import HardwareRecipe


_RECIPES: dict[str, HardwareRecipe] = {}


def register(target_arch: str, recipe: HardwareRecipe) -> None:
    _RECIPES[target_arch] = recipe


def build_recipe(target_arch: str) -> HardwareRecipe:
    try:
        return _RECIPES[target_arch]
    except KeyError as exc:
        raise KeyError(f"no hardware recipe for {target_arch}") from exc


# Imported last: each recipe module calls `register` above on import.
from . import sm90_hopper  # noqa: E402,F401
from . import sm120_blackwell  # noqa: E402,F401
