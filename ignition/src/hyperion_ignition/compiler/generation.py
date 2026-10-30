"""What the manifest says about generation: which tokens end it, the recommended sampling, and the chat
template. Read from the Hugging Face files beside the weights (`config.json`, `generation_config.json`,
`tokenizer_config.json`); a required value that none of them states is an error, not a default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty")


@dataclass(frozen=True)
class QualifiedCheckpoint:
    """The public checkpoint the weights came from: `repo` at the git commit `revision`."""

    repo: str
    revision: str

    def __post_init__(self) -> None:
        if not self.repo or self.repo.count("/") != 1 or any(c.isspace() for c in self.repo):
            raise ValueError(f"checkpoint repo must look like owner/name, got {self.repo!r}")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError(f"checkpoint revision must be a 40-digit lowercase hex commit, got {self.revision!r}")

    def to_json_dict(self) -> dict[str, str]:
        return {"repo": self.repo, "revision": self.revision}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value


def _token_ids(value: Any, where: str) -> list[int]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in items):
        raise ValueError(f"{where}: eos_token_id must be a non-negative integer or a list of them, got {value!r}")
    return list(items)


def read_generation(source: Path) -> dict[str, Any]:
    """`{"eos_token_ids": [...], "sampling_defaults": {...}?}` from `config.json` and `generation_config.json`."""
    config = _read_json(source / "config.json")
    if config is None:
        raise FileNotFoundError(f"no config.json in model source directory: {source}")
    generation = _read_json(source / "generation_config.json")
    eos: list[int] = []
    for ids in (_token_ids(config.get("eos_token_id"), "config.json"), _token_ids((generation or {}).get("eos_token_id"), "generation_config.json")):
        eos += [i for i in ids if i not in eos]
    if not eos:
        raise ValueError("no eos_token_id in config.json or generation_config.json: generation would never end")
    document: dict[str, Any] = {"eos_token_ids": eos}
    sampling = {k: generation[k] for k in SAMPLING_KEYS if generation and generation.get(k) is not None}
    for key, value in sampling.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"generation_config.json: {key} must be a non-negative number, got {value!r}")
    if "top_k" in sampling:
        sampling["top_k"] = int(sampling["top_k"])
    if sampling:
        document["sampling_defaults"] = sampling
    return document


def _token_text(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if isinstance(value, dict):
        value = value.get("content")
    return value if isinstance(value, str) else ""


def read_chat_template(source: Path) -> dict[str, str] | None:
    """The model's chat template text and its special-token texts, or None when it has no template."""
    config = _read_json(source / "tokenizer_config.json")
    if config is None:
        return None
    template = config.get("chat_template")
    if template is None:
        return None
    if not isinstance(template, str) or not template.strip():
        # A list of named templates (`default`, `tool_use`) is a different mechanism; do not pick one silently.
        raise ValueError("tokenizer_config.json: chat_template must be one non-empty string")
    return {"template": template, "bos_token": _token_text(config, "bos_token"), "eos_token": _token_text(config, "eos_token")}
