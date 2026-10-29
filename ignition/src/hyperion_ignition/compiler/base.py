"""Base class for every model-architecture compiler.

A compiler owns the build-time pipeline for one architecture (Llama, Qwen,
DeepSeek…). Adding an architecture = adding one file under `architectures/`
plus registering its id in `registry.py`. Existing files are not edited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..hardware.base import HardwareRecipe
from .model_config import ModelConfig


# No quantizer exists yet, so every package holds bf16 weights. The manifest
# label must describe what was written, never what the recipe would allow:
# change this only together with code that really produces the new format.
PACKAGED_FORMAT = "bf16"


def packaged_format(recipe: HardwareRecipe) -> str:
    """The format of the weights this compiler writes, checked against the device."""
    if PACKAGED_FORMAT not in recipe.supported_formats:
        raise ValueError(f"{recipe.target_arch} cannot run {PACKAGED_FORMAT} weights")
    return PACKAGED_FORMAT


@dataclass
class LayerMap:
    """One tensor's place in the output package. Mirrors the Rust
    `hyperion_specification::v1::LayerMapping`."""

    index: int
    name: str
    byte_offset: int
    byte_length: int
    dtype: str
    shape: tuple[int, ...]
    # Index into the manifest's payload table of the file holding the tensor; the offset counts
    # from the start of that file's data section.
    payload: int = 0


@dataclass
class PayloadFile:
    """One weights file of the package; mirrors `hyperion_specification::v1::Payload`."""

    file: str
    byte_length: int
    sha256: str


@dataclass
class CompiledManifest:
    """The in-memory form of what becomes the package's manifest.json."""

    model_id: str
    target_arch: str
    tensor_format: str
    model: ModelConfig
    tokenizer: dict[str, Any]
    layers: list[LayerMap] = field(default_factory=list)
    payloads: list[PayloadFile] = field(default_factory=list)
    hardware_recipe: dict[str, str] = field(default_factory=dict)
    signature: list[int] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to the on-disk JSON form the runtime deserializes."""
        manifest = {
            "manifest_version": "V1",
            "model_id": self.model_id,
            "target_arch": self.target_arch,
            "tensor_format": self.tensor_format,
            "model": self.model.to_json_dict(),
            "tokenizer": self.tokenizer,
            "layers": [
                {
                    "index": l.index,
                    "name": l.name,
                    # payload 0 is the default and is not written, like an absent payload table.
                    **({"payload": l.payload} if l.payload else {}),
                    "byte_offset": l.byte_offset,
                    "byte_length": l.byte_length,
                    "tensor": {"dtype": l.dtype, "shape": list(l.shape)},
                }
                for l in self.layers
            ],
            "hardware_recipe": self.hardware_recipe,
            "signature": self.signature,
        }
        if self.payloads:
            manifest["payloads"] = [{"file": p.file, "byte_length": p.byte_length, "sha256": p.sha256} for p in self.payloads]
        return manifest


class ModelCompiler(Protocol):
    """Per-architecture compile pipeline."""

    model_id: str

    def compile(self, *, source: Path, recipe: HardwareRecipe) -> CompiledManifest:
        """Extract weights + emit the manifest. Implementation is per-arch."""
        ...

