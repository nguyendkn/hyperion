#!/usr/bin/env bash
# Cache BUILT packages and oracles between sessions in an internal object store (PACKAGE_STORE, an http(s) URL that accepts GET and PUT; `none` or
# unset = nothing happens, each session rebuilds). Every object is a tar of one directory plus its sha256, verified after the download.
#   PACKAGE_STORE=https://store.example/hyperion bash scripts/pod-package-store.sh pull /cache/packages/qwen2.5-7b
#   PACKAGE_STORE=... bash scripts/pod-package-store.sh push /cache/packages/qwen2.5-7b
# `pull` exits 0 and prints MISS when the store has nothing (the caller builds, then pushes); a checksum mismatch is an error, never a silent rebuild.
set -euo pipefail
action=${1:?usage: pod-package-store.sh <pull|push> <directory>}
dir=${2:?directory}
store=${PACKAGE_STORE:-none}
if [ "$store" = none ] || [ -z "$store" ]; then
  echo "PACKAGE_STORE=none: skipped $action of $dir"
  exit 0
fi
name=$(basename "$dir")
tmp=$(mktemp -d /tmp/hyperion-scratch.XXXXXX)
trap 'rm -rf "$tmp"' EXIT
case "$action" in
  pull)
    if ! curl -fsS -o "$tmp/$name.tgz" "$store/$name.tgz" 2> /dev/null || ! curl -fsS -o "$tmp/$name.sha256" "$store/$name.sha256" 2> /dev/null; then
      echo "MISS $name"
      exit 0
    fi
    want=$(cut -d' ' -f1 "$tmp/$name.sha256")
    got=$(sha256sum "$tmp/$name.tgz" | cut -d' ' -f1)
    [ "$want" = "$got" ] || { echo "ERROR: $name.tgz sha256 $got, the store says $want" >&2; exit 1; }
    mkdir -p "$(dirname "$dir")" && rm -rf "$dir" && tar -xzf "$tmp/$name.tgz" -C "$(dirname "$dir")"
    echo "HIT $name sha256=$got"
    ;;
  push)
    [ -d "$dir" ] || { echo "ERROR: $dir is not a directory" >&2; exit 1; }
    tar -czf "$tmp/$name.tgz" -C "$(dirname "$dir")" "$name"
    sha256sum "$tmp/$name.tgz" | cut -d' ' -f1 > "$tmp/$name.sha256"
    curl -fsS -T "$tmp/$name.tgz" "$store/$name.tgz" && curl -fsS -T "$tmp/$name.sha256" "$store/$name.sha256"
    echo "PUSHED $name sha256=$(cat "$tmp/$name.sha256")"
    ;;
  *) echo "usage: pod-package-store.sh <pull|push> <directory>" >&2; exit 2 ;;
esac
