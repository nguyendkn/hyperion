# testdata

Real third-party model files used by tests (Hugging Face `config.json`, `tokenizer.json`). They are large and model-specific, so they are **not committed**: `testdata/models/` is gitignored.

- `models.json` (committed): which models, pinned revision, sha256 per file, and which Ignition compiler must accept the config (`null` = none yet, every compiler must refuse it).
- `models/<id>/` (gitignored): fetched files. Get them with `python scripts/fetch-model-fixtures.py [<id>]`.

Tests are generic and iterate over `models.json`; nothing in the code names a model. When the files are absent a test is skipped with a visible message. Set `HYPERION_REQUIRE_MODEL_FIXTURES=1` to turn a missing file into a failure (use it after fetching, and in any CI that has the files).

`oracle/` (committed, small and deterministic for the pinned revision): Hugging Face reference outputs. `<id>.json` holds reference logits plus the `tolerance_max_abs` declared before any Hyperion run (`python scripts/gen-oracle.py <id>`, needs torch + transformers, not project dependencies); `chat-<id>.json` holds rendered chat text and token ids, plus the ids of `tokenizer-corpus.json` (`python scripts/gen-chat-reference.py <id>`). With these committed, a fresh clone only needs `python scripts/fetch-model-fixtures.py` (sha256-pinned) to run the tokenizer and chat-template tests. Regenerating them with another torch/transformers version can change the numbers, so review the diff.

Add a model: add an entry to `models.json` (with sha256 from the pinned revision), run the fetch script. No test code changes.
