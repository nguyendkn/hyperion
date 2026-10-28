"""Plugin registry for model compilers.

Adding a new architecture:
  1. create `architectures/<name>.py` that calls `register(model_id, cls)`,
  2. import it in `architectures/__init__.py`.

No edits to existing architectures or to base.py.
"""

from __future__ import annotations

from .base import ModelCompiler


_ARCH_MAP: dict[str, type] = {}


def register(model_id: str, compiler_cls: type) -> None:
    _ARCH_MAP[model_id] = compiler_cls


def build_compiler(model_id: str) -> ModelCompiler:
    try:
        cls = _ARCH_MAP[model_id]
    except KeyError as exc:
        raise KeyError(f"no compiler registered for model id: {model_id}") from exc
    return cls()


# Imported last: each architecture module calls `register` above on import.
from . import architectures  # noqa: E402,F401
