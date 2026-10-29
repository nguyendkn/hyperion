#!/usr/bin/env python3
"""Download the real model files listed in testdata/models.json.

Files land in testdata/models/<id>/ (gitignored) at the pinned Hugging Face
revision and are verified against the recorded sha256, so every machine tests
against byte-identical inputs. Standard library only.

    python scripts/fetch-model-fixtures.py            # all models
    python scripts/fetch-model-fixtures.py qwen3-0.6b # one model
    python scripts/fetch-model-fixtures.py qwen2.5-7b --weights   # also the full weights (about 15 GB)

`--weights` downloads the files listed under `weights` in the registry (name, size, sha256, taken from the
pinned revision's LFS metadata) into the same directory. Before downloading it prints the git revision
(the `REVISION` file or `HYPERION_REVISION`), the sha256 of this script and of each expected file, and
refuses to start when the disk has less free space than the files still missing need plus a margin.
It uses `hf_transfer` (many parallel connections) when that package is installed, otherwise one
resumable stream; an interrupted download continues from its `.part` file; every file is checked for
size and sha256 and deleted when it does not match. `HYPERION_HF_ENDPOINT` overrides the server.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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
ENDPOINT = os.environ.get("HYPERION_HF_ENDPOINT", "https://huggingface.co")
GIB = 1 << 30
# Free space kept beyond the files still to download: a margin for the oracle outputs, caches and the
# package Ignition writes, and 10 percent on top of the download itself.
DISK_MARGIN_BYTES = 5 * GIB


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


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def git_revision() -> str:
    explicit = os.environ.get("HYPERION_REVISION")
    if explicit:
        return explicit
    for candidate in (Path.cwd() / "REVISION", ROOT / "REVISION"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return "unknown"


def _download_stream(url: str, part: Path, total: int) -> None:
    """One connection, resuming from the bytes already in `part`."""
    start = part.stat().st_size if part.is_file() else 0
    if start > total:
        part.unlink()
        start = 0
    if start == total:
        return
    request = urllib.request.Request(url)
    if start:
        request.add_header("Range", f"bytes={start}-")
    done, next_report = start, start
    with urllib.request.urlopen(request, timeout=120) as response, part.open("ab" if start else "wb") as out:
        if start and response.status != 206:
            sys.exit(f"{url}: the server ignored the range request; delete {part} and retry")
        while chunk := response.read(8 * 1024 * 1024):
            out.write(chunk)
            done += len(chunk)
            if done >= next_report + total // 20:
                print(f"         {part.name}: {100 * done // total}%", flush=True)
                next_report = done


def _download(url: str, part: Path, total: int) -> None:
    try:
        import hf_transfer  # type: ignore[import-not-found]
    except ImportError:
        hf_transfer = None
    if hf_transfer is not None and not (part.is_file() and part.stat().st_size):
        try:
            hf_transfer.download(url=url, filename=str(part), max_files=16, chunk_size=10 * 1024 * 1024, parallel_failures=3, max_retries=5)
            return
        except Exception as error:  # fall back to the plain stream rather than fail a long download
            print(f"hf_transfer failed ({error}); falling back to one stream")
            part.unlink(missing_ok=True)
    _download_stream(url, part, total)


def fetch_weights(model_id: str, spec: dict, dest: Path = DEST, endpoint: str = ENDPOINT) -> None:
    weights = spec.get("weights")
    if not weights:
        sys.exit(f"{model_id}: the registry lists no `weights` to download")
    target = dest / model_id
    target.mkdir(parents=True, exist_ok=True)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(f"FETCH git_revision={git_revision()} script_sha256={script_hash}")
    print(f"FETCH model={model_id} repo={spec['repo']} revision={spec['revision']} endpoint={endpoint}")
    for name, info in weights.items():
        print(f"FETCH expect {name} bytes={info['bytes']} sha256={info['sha256']}")

    missing = [(n, i) for n, i in weights.items() if not (target / n).is_file() or (target / n).stat().st_size != i["bytes"]]
    already = sum((target / (n + ".part")).stat().st_size for n, _ in missing if (target / (n + ".part")).is_file())
    need = sum(i["bytes"] for _, i in missing) - already
    required = need + need // 10 + DISK_MARGIN_BYTES
    free = free_bytes(target)
    print(f"FETCH disk: {need / GIB:.1f} GiB still to download, {required / GIB:.1f} GiB required, {free / GIB:.1f} GiB free")
    if missing and free < required:
        sys.exit(f"not enough disk space in {target}: {free / GIB:.1f} GiB free, {required / GIB:.1f} GiB required")

    for name, info in weights.items():
        path, part = target / name, target / (name + ".part")
        if path.is_file() and path.stat().st_size == info["bytes"] and sha256(path) == info["sha256"]:
            print(f"ok       {model_id}/{name}")
            continue
        path.unlink(missing_ok=True)
        print(f"download {model_id}/{name} ({info['bytes'] / GIB:.2f} GiB)")
        _download(f"{endpoint}/{spec['repo']}/resolve/{spec['revision']}/{name}", part, info["bytes"])
        if part.stat().st_size != info["bytes"]:
            sys.exit(f"{model_id}/{name}: {part.stat().st_size} bytes downloaded, expected {info['bytes']}; run again to resume")
        actual = sha256(part)
        if actual != info["sha256"]:
            part.unlink()
            sys.exit(f"{model_id}/{name}: sha256 {actual} != pinned {info['sha256']}")
        part.replace(path)
        print(f"verified {model_id}/{name}")


def main() -> None:
    models = json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
    arguments = [a for a in sys.argv[1:] if a != "--weights"]
    with_weights = "--weights" in sys.argv[1:]
    wanted = arguments or list(models)
    unknown = [m for m in wanted if m not in models]
    if unknown:
        sys.exit(f"unknown model id(s): {', '.join(unknown)}; known: {', '.join(models)}")
    for model_id in wanted:
        fetch(model_id, models[model_id])
        if with_weights:
            fetch_weights(model_id, models[model_id])


if __name__ == "__main__":
    main()
