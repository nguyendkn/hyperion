"""Read a safetensors header and map its tensors to manifest layers.

Only the header is read (8-byte little-endian length + JSON), so a header-only
file works as well as a full checkpoint. Byte offsets are relative to the start
of the tensor data section, i.e. into the weight payload, which is how the
manifest's `byte_offset` is meant to be used by the loader.

Mapping is strict: a tensor the architecture expects but the checkpoint lacks
("missing"), and a tensor the checkpoint has but the architecture does not
account for ("unmapped"), are both errors. Guessing would load a different
model than the manifest describes.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from .base import LayerMap
from .model_config import ModelConfig

# A header is a few tens of KB for a model with hundreds of tensors; this bound
# stops a corrupt length field from driving a huge read.
MAX_HEADER_BYTES = 64 * 1024 * 1024

# safetensors dtype -> (manifest dtype, bytes per element). Only what the runtime
# can load today; anything else is refused rather than silently converted.
DTYPES = {"BF16": ("bf16", 2)}


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


def read_safetensors_header(path: Path) -> dict[str, TensorEntry]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path.name}: too short to be a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        if length > MAX_HEADER_BYTES:
            raise ValueError(f"{path.name}: header length {length} exceeds {MAX_HEADER_BYTES}")
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError(f"{path.name}: header truncated ({len(raw)} of {length} bytes)")
    header = json.loads(raw)
    header.pop("__metadata__", None)

    entries: dict[str, TensorEntry] = {}
    for name, spec in header.items():
        dtype, size = DTYPES.get(spec["dtype"], (None, 0))
        if dtype is None:
            raise ValueError(f"{name}: dtype {spec['dtype']} is not supported (supported: {', '.join(DTYPES)})")
        begin, end = spec["data_offsets"]
        if not 0 <= begin <= end:
            raise ValueError(f"{name}: invalid data range [{begin}, {end}]")
        shape = tuple(spec["shape"])
        elements = 1
        for dim in shape:
            elements *= dim
        if end - begin != elements * size:
            raise ValueError(f"{name}: range holds {end - begin} bytes, shape {shape} needs {elements * size}")
        entries[name] = TensorEntry(name, dtype, shape, begin, end)

    ordered = sorted(entries.values(), key=lambda e: e.begin)
    for before, after in zip(ordered, ordered[1:]):
        if after.begin < before.end:
            raise ValueError(f"tensor ranges overlap: {before.name} and {after.name}")
    return entries


def _qwen2_expected(cfg: ModelConfig, tie_word_embeddings: bool) -> dict[str, tuple[int | None, ...]]:
    """Expected HF tensor names -> shape with None for dims that depend on hidden size."""
    q_rows = cfg.num_attention_heads * cfg.head_dim
    kv_rows = cfg.num_kv_heads * cfg.head_dim
    expected: dict[str, tuple[int | None, ...]] = {
        "model.embed_tokens.weight": (cfg.vocab_size, None),
        "model.norm.weight": (None,),
    }
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        expected.update({
            p + "input_layernorm.weight": (None,),
            p + "post_attention_layernorm.weight": (None,),
            p + "self_attn.q_proj.weight": (q_rows, None),
            p + "self_attn.q_proj.bias": (q_rows,),
            p + "self_attn.k_proj.weight": (kv_rows, None),
            p + "self_attn.k_proj.bias": (kv_rows,),
            p + "self_attn.v_proj.weight": (kv_rows, None),
            p + "self_attn.v_proj.bias": (kv_rows,),
            p + "self_attn.o_proj.weight": (None, q_rows),
            p + "mlp.gate_proj.weight": (None, None),
            p + "mlp.up_proj.weight": (None, None),
            p + "mlp.down_proj.weight": (None, None),
        })
    if not tie_word_embeddings:
        expected["lm_head.weight"] = (cfg.vocab_size, None)
    return expected


def _graph_index(name: str, num_layers: int) -> int:
    """Position in the model graph: embedding 0, layer i -> i + 1, final norm / head after."""
    if name.startswith("model.layers."):
        return int(name.split(".")[2]) + 1
    return 0 if name == "model.embed_tokens.weight" else num_layers + 1


def map_qwen2_layers(entries: dict[str, TensorEntry], cfg: ModelConfig, *, tie_word_embeddings: bool) -> list[LayerMap]:
    expected = _qwen2_expected(cfg, tie_word_embeddings)
    missing = sorted(set(expected) - set(entries))
    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} expected tensor(s): {', '.join(missing[:5])}")
    unmapped = sorted(set(entries) - set(expected))
    if unmapped:
        raise ValueError(f"checkpoint has {len(unmapped)} unmapped tensor(s): {', '.join(unmapped[:5])}")
    for name, want in expected.items():
        got = entries[name].shape
        if len(got) != len(want) or any(w is not None and w != g for w, g in zip(want, got)):
            raise ValueError(f"{name}: shape {got} does not match the model config (expected {want}, None = hidden size)")

    ordered = sorted(expected, key=lambda n: (_graph_index(n, cfg.num_layers), n))
    return [
        LayerMap(
            index=_graph_index(name, cfg.num_layers),
            name=name.removeprefix("model."),
            byte_offset=entries[name].begin,
            byte_length=entries[name].end - entries[name].begin,
            dtype=entries[name].dtype,
            shape=entries[name].shape,
        )
        for name in ordered
    ]
