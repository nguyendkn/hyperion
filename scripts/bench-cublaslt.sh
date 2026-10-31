#!/usr/bin/env bash
# Times the cuBLASLt heuristic's candidate algorithms for the decode projections (needs a GPU host and the sm90 release test binary).
# The engine never uses the result: it runs the heuristic's first algorithm and logs its id per GEMM at start-up.
set -euo pipefail
cd "$(dirname "$0")/.."
HYPERION_REVISION="$(git rev-parse HEAD 2>/dev/null || echo unknown)" \
  cargo test --release --features sm90 -p helios-hal --test weights_hw cublaslt_algorithm_benchmark -- --ignored --nocapture --test-threads=1
