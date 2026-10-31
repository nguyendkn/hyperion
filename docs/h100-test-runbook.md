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

## Persistent cache (PVC `hyperion-cache`, 200 GiB)

- Create once with `bash scripts/pod-spec.sh pvc | kubectl apply -f -`; reuse it after that. It is mounted at `/cache` by `scripts/pod-spec.sh pod <gpus> <hours>`. **Never delete it without the owner**; every report lists its state (`kubectl get pvc hyperion-cache`, and `du -sh /cache/*` from a pod).
- After `. scripts/pod-lib.sh`, `use_cache` sets `CARGO_TARGET_DIR=/cache/target/<rev12>` (one directory per revision, so A/B builds never share one), `HF_HOME=/cache/hf`, `HYPERION_PACKAGES=/cache/packages`, and `CARGO_HOME`/`RUSTUP_HOME` under `/cache` so a second session skips the toolchain download and the dependency build. It prints a `CACHE target=...` line; keep it with the `BUILD revision=... sha256=...` line.
- Weights and oracles are fetched into `/cache` only on a miss (`python scripts/fetch-model-fixtures.py <id> --weights` verifies sha256 and resumes); a session never re-downloads what is there. A package built into `/cache/packages` is keyed by the revision and the model id.
- Pod deadline and cleanup: `activeDeadlineSeconds` is set by `pod-spec.sh`, and the caller sets `trap 'kubectl delete pod hyperion-work --ignore-not-found' EXIT` before applying. At the end of a session delete the pod, check `kubectl get pods` is empty, and leave the PVC.

## Kernel bring-up with compute-sanitizer

`compute-sanitizer` ships with the CUDA devel image (`/usr/local/cuda/bin`). Run a new kernel on the smallest case first, one tool at a time (they are slow and exclusive):

```
compute-sanitizer --tool memcheck  --error-exitcode 1 <test-exe> <test-name> --ignored --nocapture
compute-sanitizer --tool racecheck --error-exitcode 1 <test-exe> <test-name> --ignored --nocapture
compute-sanitizer --tool initcheck --error-exitcode 1 <test-exe> <test-name> --ignored --nocapture
```

- Use a debug-info build of the kernels for readable reports (`nvcc -lineinfo`); the committed fatbin is not rebuilt for this.
- Run the paged-kernel unit tests (lengths 1, 63, 64, 65, 129) under all three before any end-to-end run; then the same for bucket 8.
- A graph-captured launch is checked through its direct-launch twin: sanitizer does not step inside a replayed graph.
- Record the tool, the test name and "0 errors" (or the report) in the ledger row with the tag `[RUN on H100]`.

## Qwen2.5-7B session (planned; nothing here has run on a GPU yet)

Local CPU venv for dry-running `gen-oracle-large.py` and other oracle scripts without a GPU: `%LOCALAPPDATA%\hyperion\venvs\oracle` (Python 3.12, CPU torch, transformers, safetensors). Not a scratch venv — keep it across sessions instead of recreating it in `%TEMP%\hyperion-scratch\`.

Needs about 40 GB of free disk (15.2 GB of weights, the package, caches), an 80 GB GPU for the 8192-token oracle prompt (the float32 weights alone are about 30 GB, plus the activations and attention of 8192 tokens; the oracle prints the peak GPU memory after each prompt, so the real need is measured; the oracle process exits before Hyperion starts, so the two never share the GPU). On a 40 GB card lower `--natural-contexts` to 4096 and say so in the report, and `uv pip install hf_transfer accelerate transformers` plus a CUDA build of torch. In order:

1. Step 1 check (about 10 minutes): all hardware tests with the run-time-loaded driver and cuBLAS; read the `CUDA driver library loaded` and `cuBLAS library loaded` log lines.
2. Step 4 (graph hygiene).
3. First fetch one small file of the model alone to see which download path works (`hf_transfer` has never run against the real huggingface.co): `python scripts/fetch-model-fixtures.py qwen2.5-7b` (config and tokenizer, a few MB) and read the `FETCH download path ...: hf_transfer | single stream` line. If `hf_transfer` fails, the single stream is acceptable (a download path, not a compute path); write which one was used in the log. Then `python scripts/fetch-model-fixtures.py qwen2.5-7b --weights` (prints the revision and the pinned hashes, checks the disk, verifies every shard).
4. `python scripts/gen-oracle-large.py qwen2.5-7b --contexts 1024`: float32 pass, bfloat16 pass, writes `testdata/oracle/qwen2.5-7b.compact.json` with the tolerance derived from the measured drift, then exits. Prompts: short, paragraph, a 1024-token repeated prompt, and 8192 tokens of non-repeating repository prose (`--natural-contexts`, default 8192; the long prompt is what exercises split-K attention at head_dim 128 over a varied cache; the passes run one after the other; the file keeps the token ids and `source` (the docs files and the sha256 of the joined text), so the oracle does not depend on `docs/` as it is later; the script exits with an error, never repeats or pads, if `docs/` holds fewer tokens than asked). Copy the file out of the pod. Read the last `ORACLE wrote` line: it says whether each limit is set by the margin (drift x 1.5) or by the 0.02 floor.
5. `hyperion-compile --model qwen2 --hw sm_90 --source testdata/models/qwen2.5-7b --out testdata/models/qwen2.5-7b/model.hyperion` (hashes the 4 shards into the payload table).
6. Hyperion: `forward_matches_the_compact_oracle` (also prints the free device memory after the weights loaded), `decode_time_vs_context` for ms per token at several contexts (CUDA events inside one run), then the hardware suite.

### If Hyperion exceeds the tolerance

The tolerance is fixed in the oracle file before Hyperion runs. A failure is never answered by loosening it and rerunning. Hyperion and Hugging Face bf16 sum in different orders, so the 1.5 margin may simply be tight; that is a finding, not a reason to move the bound. Instead:

1. Report the measured ratio (Hyperion's max top-k logit diff divided by the HF bf16 deviation, printed in `ORACLE SUMMARY`) and where the largest difference is (prompt, position). Two ratios are printed: "same positions" (both max over the positions Hyperion was compared on; this is the like-for-like one) and "all positions" (the denominator the tolerance was derived from, never smaller, so it flatters Hyperion). Quote both, labelled.
2. Investigate the cause first (which layer, which context length, short vs the 8192-token prompt).
3. Changing the margin or the floor is the owner's decision, written down (here and in the ledger) before the next run.

State in the report which term set the limit: the margin (drift x 1.5) or the 0.02 floor. When the floor binds (the fixture drift was 0.0031, so the limit was about 6 times the drift) the check is looser than "1.5 x bf16" suggests.

It closes these ledger rows at once: the device upload of a sharded model, an untied `lm_head` on a GPU, head_dim 128 on a real model, the free-memory number for KV sizing, and the first ms per token of a real-size model.

## A/B of several commits on one pod (2026-10-30)

- Package the tip with `scripts/package-for-pod.sh` and every older commit to compare with `scripts/package-variant.sh <commit> <out.tgz>`: the older tree with HEAD's hardware tests and scripts laid over it, so every build is measured by the same test code (`REVISION` reads `<commit> tests-from <HEAD>`).
- Unpack each tree into its own directory, run `bash scripts/build-kernels.sh` in each (nvcc builds in seconds), then `CARGO_TARGET_DIR=/work/target-<tree> cargo test --release --features sm90 -p helios-hal --no-run` for all trees in parallel (about a minute on a 128-core pod). One target directory per tree, always (rule 1 below).
- Data (models, oracles) lives in one tree and is shared through absolute `HYPERION_WEIGHTS_PACKAGE` and `HYPERION_COMPACT_ORACLE`.
- `HYPERION_CONTEXTS=5,1024,4096,8000` narrows the context lengths of `decode_time_vs_context`; the windows and warm-up are not affected.
- A new tolerance file (a regenerated oracle) is committed and pushed before the run that uses it.
- `scripts/gemm-bandwidth.py` turns an Nsight Systems profile into achieved GB/s per GEMM call site: `nsys profile --cuda-graph-trace=node -o decode <test>`, `nsys export --type sqlite decode.nsys-rep`, `python scripts/gemm-bandwidth.py decode.sqlite --model 7b`.
- Nsight Systems: `apt-get install nsight-systems-2025.5.2` (the CUDA image's apt repo has it); the binary is `/opt/nvidia/nsight-systems/2025.5.2/bin/nsys`.
- `kubectl exec ... nohup ... &` keeps the exec attached and times out: start long jobs with `(setsid nohup bash script.sh > log 2>&1 < /dev/null &)` and poll the log.

- Kernel binaries are not byte-reproducible: `elementwise.fatbin` differs on every rebuild (nvcc puts a per-compile id into the names of shared variables), its SASS (`cuobjdump -sass`) does not. Compare SASS, never hashes.
- An fp32 oracle at 16384 tokens does not fit next to the weights on an 80 GB GPU (the attention score matrix is 28 GiB); 12288 tokens peaked at 67 GiB.
- A model's package is compiled on the pod with the Ignition of the tree under test (`uv pip install -e <tree>/ignition`), with `--hf-repo` and `--hf-revision` from `testdata/models.json`.

## Rules learned from incidents

1. **One `CARGO_TARGET_DIR` per build in an A/B comparison.** With one shared target directory cargo saw the second tree's sources as up to date and did not rebuild, so "B" was the A binary and a whole run was void. It was noticed only because the output format was the old one.
2. **Print the revision and the sha256 of every executable at the start of each result.** `scripts/package-for-pod.sh <out.tgz>` packages the tree with a `REVISION` file (`git rev-parse HEAD`, plus `-dirty` if the tree is modified); on the pod `. scripts/pod-lib.sh` gives `announce <exe>` and `run_test <exe> <args>`, which print `BUILD revision=... exe=... sha256=...` before the run, and the hardware tests print `BUILD revision=` themselves. A wrong build is visible in the output instead of being noticed by luck.
3. **Measure decode time with CUDA events inside one run** (a window of 128 decode steps after warm-up), alternating the configurations being compared in one process; report median with p10 and p90. A difference of two whole-run times has a spread of tens of percent.
4. For cross-build comparisons that need two processes, alternate the processes (A, B, A, B, ...) and record the SM clock (`nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv -l 1`).
5. Say in the ledger what was only type-checked, and what ran on which topology.

## Notes from the 2026-10-31 gate run

- A tree older than the manifest fields (`sharding`, `parallelism`) rejects a current package: for an old-versus-new A/B give the old binary a copy of the package directory (symlinks to the safetensors, a manifest without those two fields).
- An old tree needs `bash scripts/build-kernels.sh` on the pod before its test binary is built: its committed fatbin does not carry the symbols of the current test code.
- The pod `kubectl exec` with `setsid nohup ... &` and a Monitor on the `done` line of the log works for runs of 5 to 20 minutes; Windows needs `MSYS_NO_PATHCONV=1`.
