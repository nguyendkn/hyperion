# Source on a GPU pod from the tree root: `. scripts/pod-lib.sh`.
#   announce <exe...>      print "BUILD revision=... exe=... sha256=..." for each binary
#   run_test <exe> <args>  announce the binary, then run it with HYPERION_REVISION exported
# A wrong build (for example a stale binary from a shared target directory) shows up in the
# output instead of being noticed by luck.

hyperion_revision() {
  cat "${HYPERION_TREE:-.}/REVISION" 2> /dev/null || echo unknown
}

announce() {
  for exe in "$@"; do
    echo "BUILD revision=$(hyperion_revision) exe=$(basename "$exe") sha256=$(sha256sum "$exe" | cut -d' ' -f1)"
  done
}

run_test() {
  local exe=$1
  shift
  announce "$exe"
  HYPERION_REVISION=$(hyperion_revision) "$exe" "$@"
}

# Persistent cache (docs/h100-test-runbook.md, "Persistent cache"): with the PVC mounted at /cache, one cargo target directory per revision,
# the Hugging Face cache and the built packages survive the pod. `use_cache` is a no-op without /cache.
use_cache() {
  [ -d /cache ] || { echo "no /cache mount: building in the pod's ephemeral disk"; return 0; }
  local rev
  rev=$(hyperion_revision | cut -d' ' -f1 | cut -c1-12)
  export CARGO_TARGET_DIR=/cache/target/${rev:-unknown}
  export HF_HOME=/cache/hf
  export HYPERION_PACKAGES=/cache/packages
  export CARGO_HOME=/cache/cargo
  export RUSTUP_HOME=/cache/rustup
  mkdir -p "$CARGO_TARGET_DIR" "$HF_HOME" "$HYPERION_PACKAGES" "$CARGO_HOME" "$RUSTUP_HOME"
  echo "CACHE target=$CARGO_TARGET_DIR hf=$HF_HOME packages=$HYPERION_PACKAGES"
}
