#!/usr/bin/env bash
# On the pod, after pod-setup.sh: fetch a model of testdata/models.json (sha256-checked, resumable, kept on /cache) and compile its package with
# Ignition into testdata/models/<id>/model.hyperion (plus the manifest beside it). Skips what already exists.
#   bash scripts/pod-package-model.sh qwen2.5-0.5b qwen2
set -euo pipefail
id=${1:?usage: pod-package-model.sh <registry id> <ignition model>}
compiler=${2:?ignition model name (qwen2, qwen3)}
cd "$(dirname "$0")/.."
export PATH="${CARGO_HOME:-/cache/cargo}/bin:/cache/uv:$PATH"
if [ ! -x /cache/venv/bin/python ]; then
  uv venv --python 3.12 /cache/venv
  uv pip install --python /cache/venv/bin/python -e "ignition[test]" > /dev/null
fi
. /cache/venv/bin/activate
python scripts/fetch-model-fixtures.py "$id" --weights
out=testdata/models/$id/model.hyperion
if [ ! -e "$out" ]; then
  hyperion-compile --model "$compiler" --hw sm_90 --source "testdata/models/$id" --out "$out"
fi
ls -la "testdata/models/$id" | head -20
