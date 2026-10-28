#!/usr/bin/env python3
"""Download the real model files listed in testdata/models.json.

Files land in testdata/models/<id>/ (gitignored) at the pinned Hugging Face
revision and are verified against the recorded sha256, so every machine tests
against byte-identical inputs. Standard library only.

    python scripts/fetch-model-fixtures.py            # all models
    python scripts/fetch-model-fixtures.py qwen3-0.6b # one model
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import urllib.request
from pathlib import Path


def _ipv4_first(*args, **kwargs):
    """Try IPv4 addresses before IPv6. On networks without working IPv6 (common on
    corporate and dev networks) urllib otherwise waits out the full timeout on
    every IPv6 address before falling back; curl hides this with happy eyeballs."""
    return sorted(_getaddrinfo(*args, **kwargs), key=lambda info: info[0] != socket.AF_INET)


_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = _ipv4_first

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "testdata" / "models.json"
DEST = ROOT / "testdata" / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(model_id: str, spec: dict) -> None:
    target = DEST / model_id
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in spec["files"].items():
        path = target / name
        if path.is_file() and sha256(path) == expected:
            print(f"ok       {model_id}/{name}")
            continue
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}"
        print(f"download {model_id}/{name}")
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
        actual = sha256(path)
        if actual != expected:
            path.unlink()
            sys.exit(f"{model_id}/{name}: sha256 {actual} != pinned {expected}")


def main() -> None:
    models = json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
    wanted = sys.argv[1:] or list(models)
    unknown = [m for m in wanted if m not in models]
    if unknown:
        sys.exit(f"unknown model id(s): {', '.join(unknown)}; known: {', '.join(models)}")
    for model_id in wanted:
        fetch(model_id, models[model_id])


if __name__ == "__main__":
    main()
