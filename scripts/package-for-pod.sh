#!/usr/bin/env bash
# Package the committed tree for a GPU pod and tag it, so every result from the pod names a revision
# that exists in git. Refuses to run when the working tree has uncommitted or untracked changes.
#   - REVISION inside the tarball holds `<commit> <tag>`.
#   - The tag is `pod-YYYYMMDD-HHMM`, created locally on HEAD; this script never pushes it.
# Usage: scripts/package-for-pod.sh <out.tgz>
# On the pod, `. scripts/pod-lib.sh` prints that revision and each binary's sha256 before a run
# (docs/h100-test-runbook.md, rule 2).
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:?usage: scripts/package-for-pod.sh <out.tgz>}
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing to package a dirty tree: commit (or remove) these first, so the pod run names a real revision:" >&2
  git status --short >&2
  exit 1
fi
commit=$(git rev-parse HEAD)
tag="pod-$(date +%Y%m%d-%H%M)"
if git rev-parse -q --verify "refs/tags/$tag" > /dev/null; then
  echo "tag $tag already exists (one package per minute); wait a minute and run again" >&2
  exit 1
fi
printf '%s %s\n' "$commit" "$tag" > REVISION
trap 'rm -f REVISION' EXIT
flags=""
if tar --force-local --version > /dev/null 2>&1; then flags="--force-local"; fi  # a "C:" in the path is not a remote host
{
  git ls-files | while IFS= read -r file; do [ -e "$file" ] && printf '%s\n' "$file"; done
  echo REVISION
} | tar $flags -czf "$out" -T -
git tag "$tag" "$commit"
echo "packaged $commit as $tag -> $out (the tag is local; push it only with the owner's approval)"
