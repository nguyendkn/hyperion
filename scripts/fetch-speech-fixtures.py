#!/usr/bin/env python3
"""Download the real model files listed in testdata/models-speech.json (speech track registry).

Same mechanism as scripts/fetch-model-fixtures.py (pinned revision, sha256 per file, `header_of` entries are the 8-byte-length + JSON
header of a large safetensors file, `--weights` downloads the full files with size and sha256 checks); a separate registry file keeps
the speech models out of the generic LLM tests that iterate testdata/models.json. Files land in testdata/models/<id>/ (gitignored).

    python scripts/fetch-speech-fixtures.py [<id>] [--weights]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "testdata" / "models-speech.json"


def load_base():
    spec = importlib.util.spec_from_file_location("fetch_model_fixtures", Path(__file__).with_name("fetch-model-fixtures.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = load_base()
    models = json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
    arguments = [a for a in sys.argv[1:] if a != "--weights"]
    with_weights = "--weights" in sys.argv[1:]
    wanted = arguments or list(models)
    unknown = [m for m in wanted if m not in models]
    if unknown:
        sys.exit(f"unknown model id(s): {', '.join(unknown)}; known: {', '.join(models)}")
    for model_id in wanted:
        base.fetch(model_id, models[model_id])
        if with_weights:
            base.fetch_weights(model_id, models[model_id])


if __name__ == "__main__":
    main()
