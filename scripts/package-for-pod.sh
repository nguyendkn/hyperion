#!/usr/bin/env bash
# Package the working tree (tracked and untracked files that are not ignored) for a GPU pod,
# with a REVISION file holding `git rev-parse HEAD`, plus "-dirty" when the tree has changes.
# Usage: scripts/package-for-pod.sh <out.tgz>
# On the pod, `. scripts/pod-lib.sh` prints that revision and each binary's sha256 before a run
# (docs/h100-test-runbook.md, rule 2).
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:?usage: scripts/package-for-pod.sh <out.tgz>}
revision=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then revision="$revision-dirty"; fi
printf '%s\n' "$revision" > REVISION
trap 'rm -f REVISION' EXIT
flags=""
if tar --force-local --version > /dev/null 2>&1; then flags="--force-local"; fi  # a "C:" in the path is not a remote host
{
  git ls-files -co --exclude-standard | while IFS= read -r file; do [ -e "$file" ] && printf '%s\n' "$file"; done
  echo REVISION
} | tar $flags -czf "$out" -T -
echo "packaged revision $revision -> $out"
