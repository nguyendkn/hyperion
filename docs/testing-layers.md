# Testing layers

Hyperion is developed on a host with no GPU and verified on hardware that exists only in short-lived pods, so every test belongs to a layer that says where it can run, and every claim in a report carries the tag of the layer it ran in (`docs/unverified-ledger.md`).

| Layer | Runs on | What belongs here | Ledger tag |
|---|---|---|---|
| **L0** | Any host: Windows or Linux, no GPU, no CUDA toolkit | Pure host logic and everything that needs only files: manifest contract and schema drift, Ignition (compiler, shard reading, hashing), weight plans, tokenizer and chat template against Hugging Face reference files, sampling, the oracle comparison rule, the decode-session budget, the loaders for the CUDA libraries (typed errors), the stub backend end to end through the real `helios-cli` binary, later the simulator and the multi-rank stub | `[RUN]` |
| **L1** | One GPU (the sm90 H100 pod; sm120 when Blackwell hardware exists) | Everything that launches a kernel or touches the driver: device probe, weight upload, each op against the CPU reference, the forward pass against the Hugging Face oracle, bit-identity of graph replay and direct launches, decode timing | `[RUN on H100]` |
| **L2** | One node, 2 to 8 GPUs | Tensor parallelism: sharded loading, communicator, all-reduce in graphs, TP versus single-GPU logits within a declared tolerance, run-to-run bit identity | `[RUN on NxH100]` |
| **L3** | Several nodes | Data parallel behind a router, pipeline parallel, rendezvous and failure handling across hosts | `[RUN on N nodes]` |

Tags are exact: a test that was only compiled is `[TYPE-CHECK]`; a check that needs hardware not yet available is `[NEEDS-GPU]`. Never write `[RUN]` for something that ran on a different layer.

## Conventions
- **L0 commands** (run them before every commit that touches the layer): `cargo test --workspace` (default `stub` feature; also run `cargo test -p helios-hal --features sm90` and `--features sm120` for the CUDA-free parts of those backends), `cargo test -p hyperion-specification --features schema` (schema drift), `pytest ignition/tests`.
- **Hardware tests (L1 and above)** live in files named `*_hw.rs` (today `helios-hal/tests/weights_hw.rs`), are marked `#[ignore = "needs ..."]` with the reason, and are run on a pod by name with `-- --ignored` (see `docs/h100-test-runbook.md`). A plain `cargo test` lists them as ignored and never runs them; `cargo test --workspace -- --ignored --list` shows which L1 tests exist.
- **Real files, not fakes.** Tests that need model files read the registry (`testdata/models.json`) and skip with a visible message when the files are missing; `HYPERION_REQUIRE_MODEL_FIXTURES=1` turns the skip into a failure (use it after fetching, and in CI once there is one). No test names a model.
- **Tolerances are declared before a run**, from arithmetic or from measured repeatability or drift; a tolerance is never loosened to make a test pass.
- **Measurements** (L1 and above): release build, CUDA events inside one run, ten windows, median with p10 and p90, GPU clocks recorded, one `CARGO_TARGET_DIR` per build in a comparison, and the revision and binary sha256 printed first.

## What does not exist yet
There is no CI. A workflow for L0 on Linux and Windows is waiting on the owner's decision 9 (the CI matrix); no GPU runner is planned, so L1 and above stay manual on pods.
