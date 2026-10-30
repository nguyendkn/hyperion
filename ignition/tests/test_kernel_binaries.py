"""The committed kernel binaries must match the CUDA sources they were built from.

`scripts/build-kernels.sh` (needs nvcc) writes `<name>.fatbin` and a sidecar `<name>.fatbin.json` with
the sha256 of the source, the sha256 of the machine code (`cuobjdump -sass`: the binary's bytes are not
reproducible, its machine code is) and the nvcc version. Windows builds cannot run nvcc, so this check is what
stops a source edit from shipping with a stale binary.
"""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

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


def test_every_sidecar_records_the_hash_of_the_machine_code():
    for source in _sources():
        meta = json.loads(source.with_suffix(".fatbin.json").read_text(encoding="utf-8"))
        assert re.fullmatch(r"[0-9a-f]{64}", meta.get("sass_sha256", "")), f"{source.name}: no sass_sha256 in the sidecar: run scripts/build-kernels.sh"


def test_the_committed_binary_has_the_machine_code_its_sidecar_records():
    if shutil.which("cuobjdump") is None:
        pytest.skip("cuobjdump is not installed here (a pod or a GPU host has it); scripts/check-kernels.sh does the full rebuild check")
    for source in _sources():
        meta = json.loads(source.with_suffix(".fatbin.json").read_text(encoding="utf-8"))
        sass = subprocess.run(["cuobjdump", "-sass", str(source.with_suffix(".fatbin"))], capture_output=True, check=True).stdout
        assert hashlib.sha256(sass).hexdigest() == meta["sass_sha256"], f"{source.name}: the binary's machine code is not what the sidecar says"
