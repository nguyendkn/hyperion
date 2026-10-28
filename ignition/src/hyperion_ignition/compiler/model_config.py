"""Model dimensions the runtime needs (manifest `model` object).

Derived from the Hugging Face `config.json` in the model source directory so
the numbers come from the checkpoint, never from constants in a compiler.

Mapping rules: `head_dim` = the config's own `head_dim` when present, else
`hidden_size / num_attention_heads`; `num_kv_heads` defaults to the head count
(MHA); `max_seq_len` = `max_position_embeddings` (see `model_config_from_hf`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Mirrors the Rust `hyperion_specification::v1::ModelConfig`."""

    num_layers: int
    attention: str  # "mha" | "gqa"
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    max_seq_len: int

    def to_json_dict(self) -> dict[str, int | str]:
        return {
            "num_layers": self.num_layers,
            "attention": self.attention,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
        }


def _positive_int(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"config.json: {key!r} must be a positive integer, got {value!r}")
    return value


def model_config_from_hf(source: Path) -> ModelConfig:
    """Read `<source>/config.json` (Hugging Face format) into a `ModelConfig`."""
    path = source / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"no config.json in model source directory: {source}")
    cfg = json.loads(path.read_text(encoding="utf-8"))

    heads = _positive_int(cfg, "num_attention_heads")
    if "head_dim" in cfg:
        head_dim = _positive_int(cfg, "head_dim")
    else:
        hidden = _positive_int(cfg, "hidden_size")
        if hidden % heads:
            raise ValueError(f"config.json: hidden_size {hidden} not divisible by heads {heads}")
        head_dim = hidden // heads
    kv_heads = _positive_int(cfg, "num_key_value_heads") if "num_key_value_heads" in cfg else heads
    if kv_heads > heads or heads % kv_heads:
        raise ValueError(f"config.json: {kv_heads} kv heads do not divide {heads} attention heads")
    # max_seq_len is `max_position_embeddings`, taken as-is. It is only the real
    # context length when no RoPE scaling is configured: with `rope_scaling` the
    # usable length depends on the scaling type, so refuse rather than guess.
    if cfg.get("rope_scaling"):
        raise ValueError("config.json: rope_scaling is not supported; cannot derive max_seq_len")
    if cfg.get("use_sliding_window"):
        raise ValueError("config.json: sliding-window attention is not supported (KV cache is bounded differently)")
    return ModelConfig(
        num_layers=_positive_int(cfg, "num_hidden_layers"),
        attention="mha" if kv_heads == heads else "gqa",
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=_positive_int(cfg, "vocab_size"),
        max_seq_len=_positive_int(cfg, "max_position_embeddings"),
    )
