"""Model dimensions the runtime needs (manifest `model` object).

Derived from the Hugging Face `config.json` in the model source directory so
the numbers come from the checkpoint, never from constants in a compiler.
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
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    max_seq_len: int

    def to_json_dict(self) -> dict[str, int]:
        return {
            "num_layers": self.num_layers,
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
    return ModelConfig(
        num_layers=_positive_int(cfg, "num_hidden_layers"),
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=_positive_int(cfg, "vocab_size"),
        max_seq_len=_positive_int(cfg, "max_position_embeddings"),
    )
