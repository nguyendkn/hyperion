"""The committed kernel binaries must match the CUDA sources they were built from.

`scripts/build-kernels.sh` (needs nvcc) writes `<name>.fatbin` and a sidecar `<name>.fatbin.json` with
the sha256 of the source and the nvcc version. Windows builds cannot run nvcc, so this check is what
stops a source edit from shipping with a stale binary.
"""

import hashlib
import json
from pathlib import Path

KERNELS = Path(__file__).resolve().parents[2] / "helios" / "crates" / "helios-hal" / "kernels"
FATBIN_MAGIC = bytes.fromhex("50ed55ba")  # 0xBA55ED50, little endian


def _sources():
    return sorted(KERNELS.glob("*.cu"))


def test_there_are_kernel_sources():
    assert _sources(), f"no .cu files under {KERNELS}"


def test_every_source_has_a_binary_built_from_it():
    for source in _sources():
        binary = source.with_suffix(".fatbin")
        sidecar = source.with_suffix(".fatbin.json")
        assert binary.is_file(), f"{binary.name} is missing: run scripts/build-kernels.sh"
        assert sidecar.is_file(), f"{sidecar.name} is missing: run scripts/build-kernels.sh"
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        digest = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert meta["source"] == source.name
        assert meta["source_sha256"] == digest, f"{source.name} changed since {binary.name} was built: run scripts/build-kernels.sh"
        assert meta["nvcc"].startswith("release "), meta["nvcc"]
        assert "sm_90" in meta["flags"], "the binary must carry machine code for sm_90"
        assert binary.read_bytes()[:4] == FATBIN_MAGIC, f"{binary.name} is not a fat binary"
