"""Speech family compilers (ADR 0012/0013/0014). Each takes a source directory of a real checkpoint and returns a family manifest dict
(`specification/schema/family-manifest.v1.json`). They are separate from the LLM compilers and the LLM registry: a family manifest is not a
`CompiledManifest`, and the LLM tests that iterate every LLM compiler never see these."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

SpeechCompiler = Callable[..., dict[str, Any]]

_COMPILERS: dict[str, SpeechCompiler] = {}


def register(compiler_id: str, fn: SpeechCompiler) -> None:
    if compiler_id in _COMPILERS:
        raise ValueError(f"speech compiler {compiler_id!r} is registered twice")
    _COMPILERS[compiler_id] = fn


def compile_speech(compiler_id: str, source: Path, **kwargs: Any) -> dict[str, Any]:
    if compiler_id not in _COMPILERS:
        raise KeyError(f"no speech compiler {compiler_id!r}; known: {', '.join(sorted(_COMPILERS)) or 'none'}")
    return _COMPILERS[compiler_id](source, **kwargs)


from . import qwen3_asr  # noqa: E402,F401  (registration by import side effect, as the LLM architectures do)
