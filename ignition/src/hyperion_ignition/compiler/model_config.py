"""Model dimensions the runtime needs (manifest `model` object).

Derived from the Hugging Face `config.json` in the model source directory so
the numbers come from the checkpoint, never from constants in a compiler.

Mapping rules: `head_dim` = the config's own `head_dim` when present, else
`hidden_size / num_attention_heads`; `num_kv_heads` defaults to the head count
(MHA); `max_seq_len` = `max_position_embeddings` (see `model_config_from_hf`).

Everything the runtime needs is read here and written to the signed manifest; a value the config does
not state (RoPE base, norm epsilon, tied head, activation) is an error, never a default, and a
variant the runtime does not implement (another activation, scaled or partial RoPE, sliding window) is
refused at export instead of being packaged and failing at load.
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
    hidden_size: int
    intermediate_size: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    qkv_bias: bool
    qk_norm: bool = False
    activation: str = "silu"
    rope_style: str = "rotate_half"
    attention_scale: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "num_layers": self.num_layers,
            "attention": self.attention,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "rope_theta": self.rope_theta,
            "rope_style": self.rope_style,
            "rms_norm_eps": self.rms_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "qkv_bias": self.qkv_bias,
            "qk_norm": self.qk_norm,
            "activation": self.activation,
        }
        if self.attention_scale is not None:
            document["attention_scale"] = self.attention_scale
        return document


def _positive_int(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"config.json: {key!r} must be a positive integer, got {value!r}")
    return value


def _positive_number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0 or value != value or value == float("inf"):
        raise ValueError(f"config.json: {what} must be a finite positive number, got {value!r}")
    return float(value)


def _rope_theta(cfg: dict[str, Any]) -> float:
    """The RoPE base, from the classic `rope_theta` or from transformers 5's `rope_parameters`."""
    if "rope_theta" in cfg:
        return _positive_number(cfg["rope_theta"], "rope_theta")
    parameters = cfg.get("rope_parameters")
    if isinstance(parameters, dict):
        if parameters.get("rope_type", "default") != "default":
            raise ValueError(f"config.json: rope_parameters.rope_type {parameters.get('rope_type')!r} is not supported (only the default)")
        if "rope_theta" in parameters:
            return _positive_number(parameters["rope_theta"], "rope_parameters.rope_theta")
    raise ValueError("config.json: no rope_theta (neither at the top level nor in rope_parameters)")


def model_config_from_hf(
    source: Path,
    accepted_architectures: tuple[str, ...] | None = None,
    *,
    qkv_bias: bool,
    qk_norm: bool = False,
) -> ModelConfig:
    """Read `<source>/config.json` (Hugging Face format) into a `ModelConfig`.

    `qkv_bias` and `qk_norm` are properties of the architecture, stated by its compiler (Qwen2 has
    biases on Q, K and V; Qwen3 normalises Q and K per head); the config does not carry them reliably.

    When `accepted_architectures` is given, the config's `architectures` list must
    contain one of them: a compiler must not emit a manifest for a model family
    it does not implement (e.g. Qwen3's QK-norm under the Qwen2 compiler).
    """
    path = source / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"no config.json in model source directory: {source}")
    cfg = json.loads(path.read_text(encoding="utf-8"))

    if accepted_architectures is not None:
        found = cfg.get("architectures")
        if not isinstance(found, list) or not set(found) & set(accepted_architectures):
            raise ValueError(
                f"config.json: architectures {found!r} not supported by this compiler; "
                f"supported: {', '.join(accepted_architectures)}"
            )

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
    if cfg.get("hidden_act") != "silu":
        raise ValueError(f"config.json: hidden_act {cfg.get('hidden_act')!r} is not supported (only silu)")
    if cfg.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("config.json: partial rotary embedding is not supported (the whole head is rotated)")
    tied = cfg.get("tie_word_embeddings")
    if not isinstance(tied, bool):
        raise ValueError(f"config.json: tie_word_embeddings must be stated as true or false, got {tied!r}")
    scale = None
    if "query_pre_attn_scalar" in cfg:
        scale = _positive_number(cfg["query_pre_attn_scalar"], "query_pre_attn_scalar") ** -0.5
    return ModelConfig(
        hidden_size=_positive_int(cfg, "hidden_size"),
        intermediate_size=_positive_int(cfg, "intermediate_size"),
        rope_theta=_rope_theta(cfg),
        rms_norm_eps=_positive_number(cfg.get("rms_norm_eps"), "rms_norm_eps"),
        tie_word_embeddings=tied,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        attention_scale=scale,
        num_layers=_positive_int(cfg, "num_hidden_layers"),
        attention="mha" if kv_heads == heads else "gqa",
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=_positive_int(cfg, "vocab_size"),
        max_seq_len=_positive_int(cfg, "max_position_embeddings"),
    )
