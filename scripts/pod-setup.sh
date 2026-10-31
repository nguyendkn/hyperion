#!/usr/bin/env bash
# First step of a GPU session, on the pod: unpack the tree that scripts/package-for-pod.sh made, install the toolchain into the persistent cache
# (docs/h100-test-runbook.md, "Persistent cache"), rebuild nothing yet. Safe to run again: everything already in /cache is reused.
#   bash pod-setup.sh /tmp/tree.tgz
# Leaves /work as the tree, `. /work/scripts/pod-lib.sh; use_cache; export PATH=$CARGO_HOME/bin:/cache/uv:$PATH` as the environment.
set -euo pipefail
tree=${1:?usage: pod-setup.sh <tree.tgz>}
rm -rf /work && mkdir -p /work && tar -xzf "$tree" -C /work
cd /work
export DEBIAN_FRONTEND=noninteractive
if ! command -v cc > /dev/null || ! command -v git > /dev/null; then
  apt-get update -qq
  apt-get install -y -qq curl git build-essential pkg-config ca-certificates python3-venv > /dev/null
fi
. scripts/pod-lib.sh
use_cache
export PATH="$CARGO_HOME/bin:/cache/uv:$PATH"
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null
fi
if [ ! -x /cache/uv/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/cache/uv UV_NO_MODIFY_PATH=1 sh > /dev/null
fi
# Model files and packages live on the cache, whatever the tree.
mkdir -p /cache/testdata-models
rm -rf testdata/models && ln -s /cache/testdata-models testdata/models
echo "SETUP revision=$(hyperion_revision) cargo=$(cargo --version) uv=$(uv --version) nvcc=$(nvcc --version | tail -1)"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm --format=csv,noheader
