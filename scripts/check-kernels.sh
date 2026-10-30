#!/usr/bin/env bash
# Rebuild every kernel from its committed source and check that the machine code equals the committed binary's: the sha256 of
# `cuobjdump -sass` must match the sidecar's `sass_sha256`. (The fat binary itself is not byte-reproducible, see
# docs/h100-test-runbook.md.) Needs nvcc and cuobjdump, so it runs on a GPU host or a pod.
set -euo pipefail
cd "$(dirname "$0")/../helios/crates/helios-hal/kernels"
FLAGS="-fatbin -O3 -gencode arch=compute_80,code=compute_80 -gencode arch=compute_90,code=sm_90"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
status=0
for src in *.cu; do
  name="${src%.cu}"
  # shellcheck disable=SC2086
  nvcc $FLAGS "$src" -o "$work/$name.fatbin"
  rebuilt=$(cuobjdump -sass "$work/$name.fatbin" | sha256sum | cut -d' ' -f1)
  committed=$(cuobjdump -sass "$name.fatbin" | sha256sum | cut -d' ' -f1)
  recorded=$(grep -o '"sass_sha256": "[0-9a-f]*"' "$name.fatbin.json" | grep -o '[0-9a-f]\{64\}')
  if [ "$rebuilt" = "$committed" ] && [ "$committed" = "$recorded" ]; then
    echo "ok       $name: machine code of the committed binary = a rebuild from the source = the sidecar ($recorded)"
  else
    echo "DIFFERENT $name: rebuilt $rebuilt, committed binary $committed, sidecar $recorded" >&2
    status=1
  fi
done
exit $status
