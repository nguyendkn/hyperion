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
import struct
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


def download(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    request = urllib.request.Request(url)
    if byte_range:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch(model_id: str, spec: dict) -> None:
    target = DEST / model_id
    target.mkdir(parents=True, exist_ok=True)
    for name, entry in spec["files"].items():
        # A plain string is the sha256 of a whole file; an object may say the file
        # is only the safetensors header (8-byte length + JSON) of a large file,
        # which is all the weight-mapping tests need (32 KB instead of ~1 GB).
        expected = entry if isinstance(entry, str) else entry["sha256"]
        header_of = None if isinstance(entry, str) else entry.get("header_of")
        path = target / name
        if path.is_file() and sha256(path) == expected:
            print(f"ok       {model_id}/{name}")
            continue
        print(f"download {model_id}/{name}")
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{header_of or name}"
        if header_of:
            (length,) = struct.unpack("<Q", download(url, (0, 7)))
            path.write_bytes(download(url, (0, 8 + length - 1)))
        else:
            path.write_bytes(download(url))
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
