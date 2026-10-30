#!/usr/bin/env bash
# Package an older commit for an A/B build on a pod: the commit's tree, with the hardware tests and tools of HEAD laid over
# it, so every variant is measured by the same test code. REVISION holds `<commit> tests-from <HEAD>`, which `scripts/pod-lib.sh`
# prints before every result. No tag is created.
# Usage: scripts/package-variant.sh <commit> <out.tgz>
set -euo pipefail
cd "$(dirname "$0")/.."
commit=${1:?usage: scripts/package-variant.sh <commit> <out.tgz>}
out=${2:?usage: scripts/package-variant.sh <commit> <out.tgz>}
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing to package with a dirty tree: the overlay must be what HEAD says" >&2
  exit 1
fi
head=$(git rev-parse HEAD)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git archive "$commit" | tar -x -C "$work"
for path in helios/crates/helios-hal/tests scripts; do
  rm -rf "${work:?}/$path"
  mkdir -p "$work/$path"
  git archive "$head" "$path" | tar -x -C "$work"
done
printf '%s tests-from %s\n' "$(git rev-parse "$commit")" "$head" > "$work/REVISION"
flags=""
if tar --force-local --version > /dev/null 2>&1; then flags="--force-local"; fi
tar $flags -czf "$out" -C "$work" .
echo "packaged $commit (tests and scripts from $head) -> $out"
