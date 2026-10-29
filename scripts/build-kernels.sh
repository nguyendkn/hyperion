#!/usr/bin/env bash
# Build the kernel binaries from the CUDA sources in helios-hal/kernels/. Each one is a fat binary:
# machine code for sm_90 (loading on a Hopper GPU compiles nothing) plus PTX for compute_80 (the
# driver JIT-compiles it on any other generation). Needs nvcc. A sidecar <name>.fatbin.json records
# the sha256 of the source (line endings normalised) and the nvcc version, so a test can tell when
# a committed binary no longer matches its source.
set -euo pipefail
cd "$(dirname "$0")/../helios/crates/helios-hal/kernels"
NVCC_VERSION=$(nvcc --version | grep -o "release [0-9.]*, V[0-9.]*")
FLAGS="-fatbin -O3 -gencode arch=compute_80,code=compute_80 -gencode arch=compute_90,code=sm_90"
for src in *.cu; do
  name="${src%.cu}"
  # shellcheck disable=SC2086
  nvcc $FLAGS "$src" -o "$name.fatbin"
  sha=$(tr -d '\r' < "$src" | sha256sum | cut -d' ' -f1)
  printf '{\n  "source": "%s",\n  "source_sha256": "%s",\n  "nvcc": "%s",\n  "flags": "%s"\n}\n' "$src" "$sha" "$NVCC_VERSION" "$FLAGS" > "$name.fatbin.json"
  echo "built $name.fatbin ($(wc -c < "$name.fatbin") bytes, nvcc $NVCC_VERSION)"
done
