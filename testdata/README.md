# testdata

Real third-party model files used by tests (Hugging Face `config.json`, `tokenizer.json`). They are large and model-specific, so they are **not committed**: `testdata/models/` is gitignored.

- `models.json` (committed): which models, pinned revision, sha256 per file, and which Ignition compiler must accept the config (`null` = none yet, every compiler must refuse it).
- `models/<id>/` (gitignored): fetched files. Get them with `python scripts/fetch-model-fixtures.py [<id>]`.

Tests are generic and iterate over `models.json`; nothing in the code names a model. When the files are absent a test is skipped with a visible message. Set `HYPERION_REQUIRE_MODEL_FIXTURES=1` to turn a missing file into a failure (use it after fetching, and in any CI that has the files).

`oracle/` (committed, small and deterministic for the pinned revision): Hugging Face reference outputs. `<id>.json` holds reference logits plus the `tolerance_max_abs` declared before any Hyperion run (`python scripts/gen-oracle.py <id>`, needs torch + transformers, not project dependencies); `chat-<id>.json` holds rendered chat text and token ids, plus the ids of `tokenizer-corpus.json` (`python scripts/gen-chat-reference.py <id>`). With these committed, a fresh clone only needs `python scripts/fetch-model-fixtures.py` (sha256-pinned) to run the tokenizer and chat-template tests. Regenerating them with another torch/transformers version can change the numbers, so review the diff.

`synthetic/qwen2-tiny-sharded/` (committed, about 200 KB): a tiny random Qwen2 saved by Hugging Face's `save_pretrained(max_shard_size=...)` into 5 shards with a real `model.safetensors.index.json`, its `expected.json` (each tensor's shard, length and first and last 16 bytes) and the `model.manifest.json` Ignition compiled from it. Written by `scripts/gen-sharded-fixture.py` (needs torch and transformers); the Ignition and Helios tests of sharded checkpoints use it. The `qwen2.5-7b` entry of `models.json` fetches only its config, tokenizer files, shard index and the four shard headers (no weights).

`python scripts/fetch-model-fixtures.py <id> --weights` downloads the full weight files listed under `weights` in `models.json` (size and sha256 from the pinned revision's LFS metadata): it prints the git revision and each expected hash first, refuses to start without the free disk it needs (the missing bytes plus 10 percent plus 5 GiB), uses `hf_transfer` when installed, resumes from a `.part` file, and deletes a file whose size or sha256 does not match. `python scripts/gen-oracle-large.py <id>` writes `oracle/<id>.compact.json`: for a few prompts, the float32 Hugging Face reference's argmax, top-20 logits and logsumexp at each generated position, the same from a bfloat16 run, and the tolerance derived from the measured bfloat16-vs-float32 drift (declared in the file before any Hyperion run). `oracle/qwen2-tiny-sharded.compact.json` is that script's output on the tiny sharded fixture (CPU), used to test the Rust reader.

Add a model: add an entry to `models.json` (with sha256 from the pinned revision), run the fetch script. No test code changes.
