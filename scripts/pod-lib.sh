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
