#!/usr/bin/env bash
# Regenerate the committed PTX from the CUDA sources in helios-hal/kernels/.
# Needs nvcc (CUDA toolkit). PTX is forward-compatible: the driver JIT-compiles it for
# the installed GPU, so one file serves Hopper and Blackwell.
set -euo pipefail
cd "$(dirname "$0")/../helios/crates/helios-hal/kernels"
for src in *.cu; do
  nvcc -ptx -arch=compute_80 -O3 "$src" -o "${src%.cu}.ptx"
  echo "built ${src%.cu}.ptx ($(wc -c < "${src%.cu}.ptx") bytes)"
done
