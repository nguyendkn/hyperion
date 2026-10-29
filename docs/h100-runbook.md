# H100 runbook

How hardware runs are done and what went wrong before. Every number recorded from a run goes into `docs/unverified-ledger.md` with the tag `[RUN on H100]` and the conditions (GPU, driver, nvcc, batch, model, context).

## Pod

- A pod from a CUDA devel image (`nvidia/cuda:12.4.1-devel-ubuntu22.04` was used: nvcc 12.4, driver 580.126.20), one GPU, `activeDeadlineSeconds: 7200`, `command: sleep infinity`. The image is ephemeral: nothing survives, and the driver's JIT cache starts empty. A pod can disappear mid-run (seen once, exit code 137), so copy results out as soon as a step finishes.
- Delete the pod and give the GPU back when done.
- `apt-get install curl git build-essential pkg-config` takes several minutes. Then rustup (`--profile minimal`), `uv` with Python 3.12 (the image's Python is older than Ignition needs).
- Fixtures: `python scripts/fetch-model-fixtures.py <id>` (sha256-pinned), `hyperion-compile --model qwen2 --hw sm_90 --source <dir> --out <dir>/model.hyperion`.
- Build with `--release --features sm90`; environment `CUDA_PATH=/usr/local/cuda`, `LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs`. Hardware tests are `#[ignore]`d and need `HYPERION_WEIGHTS_PACKAGE` and `HYPERION_ORACLE`.
- Kernel binaries are built on the pod with `bash scripts/build-kernels.sh` (Windows has no nvcc) and copied back with `kubectl cp`.
- Profilers: install `nsight-systems-20xx.y.z` (the `-target` package has no CLI); use `--cuda-graph-trace=node` to see kernels inside graphs. `ncu` needs GPU counter access, which the pod does not grant (`ERR_NVGPUCTRPERM`).
- From Windows: set `MSYS_NO_PATHCONV=1`; `kubectl cp` reads a `C:` path as a remote spec, so run it from the file's directory with a relative name.

## Rules learned from incidents

1. **One `CARGO_TARGET_DIR` per build in an A/B comparison.** With one shared target directory cargo saw the second tree's sources as up to date and did not rebuild, so "B" was the A binary and a whole run was void. It was noticed only because the output format was the old one.
2. **Print the revision and the sha256 of every executable at the start of each result.** `scripts/package-for-pod.sh <out.tgz>` packages the tree with a `REVISION` file (`git rev-parse HEAD`, plus `-dirty` if the tree is modified); on the pod `. scripts/pod-lib.sh` gives `announce <exe>` and `run_test <exe> <args>`, which print `BUILD revision=... exe=... sha256=...` before the run, and the hardware tests print `BUILD revision=` themselves. A wrong build is visible in the output instead of being noticed by luck.
3. **Measure decode time with CUDA events inside one run** (a window of 128 decode steps after warm-up), alternating the configurations being compared in one process; report median with p10 and p90. A difference of two whole-run times has a spread of tens of percent.
4. For cross-build comparisons that need two processes, alternate the processes (A, B, A, B, ...) and record the SM clock (`nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv -l 1`).
5. Say in the ledger what was only type-checked, and what ran on which topology.
