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

import hashlib
import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path

from .base import LayerMap, PayloadFile
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
    # Which weights file holds the tensor (index into the payload table); 0 for a single file.
    payload: int = 0


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


SINGLE_FILE = "model.safetensors"
SHARD_INDEX = "model.safetensors.index.json"
HEADER_SUFFIX = ".header"


def _is_windows_device_name(name: str) -> bool:
    stem = name.split(".")[0].rstrip(" ").upper()
    if stem in ("CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"):
        return True
    return len(stem) == 4 and stem[:3] in ("COM", "LPT") and stem[3] in "0123456789¹²³"


def is_bare_file_name(name: str) -> bool:
    """A plain file name that cannot leave the package directory and opens an ordinary file on every
    platform (the rule of the Rust spec crate, `hyperion_specification::v1::is_bare_file_name`): not
    empty, at most 255 bytes, not `.` or `..`, no slash, backslash, colon or NUL, not ending in a dot
    or a space, and not a Windows device name (`NUL.safetensors` is one)."""
    return (
        bool(name)
        and len(name.encode()) <= 255
        and name not in (".", "..")
        and not name.endswith((".", " "))
        and not any(c in name for c in ("/", "\\", ":", "\0"))
        and not _is_windows_device_name(name)
    )


@dataclass(frozen=True)
class Checkpoint:
    """The tensors of a source checkpoint and the weights files they live in."""

    entries: dict[str, TensorEntry]
    # Empty for a header-only source: the package then has the default single file and nothing is
    # recorded about it, because there are no bytes to size and hash.
    payloads: list[PayloadFile]


def _describe(path: Path) -> PayloadFile:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return PayloadFile(file=path.name, byte_length=path.stat().st_size, sha256=digest.hexdigest())


def read_checkpoint(source: Path, *, hash_payloads: bool = True) -> Checkpoint | None:
    """Read the tensors of the checkpoint in `source`: a single `model.safetensors` (or a header-only
    `model.safetensors.header`), or shards listed by `model.safetensors.index.json`. Returns None when
    there are no weights at all. With real files the payload table (name, size, sha256) is filled in;
    `hash_payloads=False` skips the hashing and also accepts header-only shards (`<shard>.header`),
    for checking how tensors map onto shards without the weights."""
    single = source / SINGLE_FILE
    header_only = source / (SINGLE_FILE + HEADER_SUFFIX)
    index = source / SHARD_INDEX
    if index.is_file() and (single.is_file() or header_only.is_file()):
        raise ValueError(f"{source.name}: both {SHARD_INDEX} and a single weights file are present; keep one")
    if index.is_file():
        return _read_sharded(source, index, hash_payloads)
    if single.is_file():
        return Checkpoint(read_safetensors_header(single), [_describe(single)] if hash_payloads else [])
    if header_only.is_file():
        return Checkpoint(read_safetensors_header(header_only), [])
    return None


def _read_sharded(source: Path, index: Path, hash_payloads: bool) -> Checkpoint:
    document = json.loads(index.read_text(encoding="utf-8"))
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map or not all(isinstance(v, str) for v in weight_map.values()):
        raise ValueError(f"{index.name}: `weight_map` must map tensor names to shard file names")
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        if not is_bare_file_name(shard):
            raise ValueError(f"{index.name}: shard name {shard!r} is not a bare file name")
    entries: dict[str, TensorEntry] = {}
    payloads: list[PayloadFile] = []
    for number, shard in enumerate(shards):
        path = source / shard
        if path.is_file():
            header = read_safetensors_header(path)
            if hash_payloads:
                payloads.append(_describe(path))
        elif not hash_payloads and (source / (shard + HEADER_SUFFIX)).is_file():
            header = read_safetensors_header(source / (shard + HEADER_SUFFIX))
        else:
            raise ValueError(f"{source.name}: shard {shard} listed in {index.name} is missing")
        listed = {name for name, file in weight_map.items() if file == shard}
        if set(header) != listed:
            extra, missing = sorted(set(header) - listed), sorted(listed - set(header))
            raise ValueError(f"{shard}: its header and {index.name} disagree (in the file only: {extra[:3]}; in the index only: {missing[:3]})")
        for name, entry in header.items():
            entries[name] = replace(entry, payload=number)
    return Checkpoint(entries, payloads)


def _qwen2_expected(cfg: ModelConfig, tie_word_embeddings: bool) -> dict[str, tuple[int | None, ...]]:
    """Expected HF tensor names -> shape with None for dims that depend on hidden size. The Qwen2 and Qwen3 decoders
    differ in two optional groups, both stated by the manifest's `ModelConfig`: Q/K/V biases (`qkv_bias`) and the
    per-head Q/K norm weights (`qk_norm`)."""
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
            p + "self_attn.k_proj.weight": (kv_rows, None),
            p + "self_attn.v_proj.weight": (kv_rows, None),
            p + "self_attn.o_proj.weight": (None, q_rows),
            p + "mlp.gate_proj.weight": (None, None),
            p + "mlp.up_proj.weight": (None, None),
            p + "mlp.down_proj.weight": (None, None),
        })
        if cfg.qkv_bias:
            expected.update({p + "self_attn.q_proj.bias": (q_rows,), p + "self_attn.k_proj.bias": (kv_rows,), p + "self_attn.v_proj.bias": (kv_rows,)})
        if cfg.qk_norm:
            # Qwen3 normalises Q and K per head with one weight vector of head_dim each.
            expected.update({p + "self_attn.q_norm.weight": (cfg.head_dim,), p + "self_attn.k_norm.weight": (cfg.head_dim,)})
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
    if tie_word_embeddings and "lm_head.weight" in entries:
        # A checkpoint with tied embeddings may still carry the output head as a second copy of the embedding matrix
        # (Qwen3-0.6B does). It is mapped, its shape checked, and a runtime that ties the head simply does not read it.
        expected["lm_head.weight"] = (cfg.vocab_size, None)
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
            payload=entries[name].payload,
        )
        for name in ordered
    ]
