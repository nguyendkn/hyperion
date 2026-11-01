#!/usr/bin/env bash
# compute-sanitizer restricted to the kernels of this repository (cuBLAS/cutlass kernels are not instrumented, so their reports, which earlier runs
# drowned the output in, do not appear). One tool at a time; they are slow and exclusive.
#   bash scripts/pod-sanitize.sh <memcheck|initcheck|racecheck|synccheck> <test-exe> <test-name> [extra test args]
# Records nothing: copy the tool, the test name and the report count into the ledger row.
set -euo pipefail
tool=${1:?usage: pod-sanitize.sh <tool> <test-exe> <test-name> [args]}
exe=${2:?test executable}
name=${3:?test name}
shift 3
# Kernels defined under helios-hal/kernels/*.cu (names as in `cuobjdump -sass` / the module).
repo_kernels='regex=^(attention_|kv_append|qkv_split|swiglu|argmax_|sample_rows|apply_page_deltas|gather_inputs|rmsnorm|rope|embedding|add_|elementwise|decode_|prefill_|paged_)'
compute-sanitizer --tool "$tool" --kernel-name "$repo_kernels" --print-limit 50 --error-exitcode 1 "$exe" "$name" --ignored --nocapture "$@"
