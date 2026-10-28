#!/usr/bin/env python3
"""Reference chat-template output from Hugging Face for a model in testdata/models.json.

Renders fixed conversations with `apply_chat_template` (generation prompt on) and records
the text and token ids to `testdata/oracle/chat-<id>.json`; the Rust tests compare Helios's
own rendering and tokenisation with it. Needs `transformers` and `jinja2` (not project
dependencies; no GPU or torch required).

    python scripts/gen-chat-reference.py qwen2.5-0.5b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "testdata" / "models.json").read_text(encoding="utf-8"))["models"]

CONVERSATIONS = [
    [{"role": "user", "content": "The capital of France is"}],
    [
        {"role": "system", "content": "You answer in one word."},
        {"role": "user", "content": "Capital of Japan?"},
    ],
    [
        {"role": "user", "content": "Xin chào, 你好!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "What is 2 + 2?\nAnswer briefly."},
    ],
]


def main() -> None:
    from transformers import AutoTokenizer

    if len(sys.argv) != 2 or sys.argv[1] not in REGISTRY:
        sys.exit(f"usage: gen-chat-reference.py <{'|'.join(REGISTRY)}>")
    model_id = sys.argv[1]
    directory = ROOT / "testdata" / "models" / model_id
    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    cases = []
    for messages in CONVERSATIONS:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False).input_ids
        cases.append({"messages": messages, "text": text, "ids": ids})
    out = ROOT / "testdata" / "oracle" / f"chat-{model_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model_id, "revision": REGISTRY[model_id]["revision"], "cases": cases}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(cases)} conversations")
    print(repr(cases[0]["text"]))


if __name__ == "__main__":
    main()
