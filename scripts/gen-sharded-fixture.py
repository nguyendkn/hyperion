#!/usr/bin/env python3
"""Write a tiny, randomly initialised Qwen2 checkpoint split over several safetensors shards, saved by
Hugging Face's own `save_pretrained(max_shard_size=...)`, so it has the real shard layout and a real
`model.safetensors.index.json`. Used by the Ignition and Helios tests of sharded checkpoints.

    python scripts/gen-sharded-fixture.py          # rewrites testdata/synthetic/qwen2-tiny-sharded/

The output is committed (about 200 KB); regenerate only when the layout needs to change, and review
the diff: another torch or transformers version can write different bytes. Needs torch and
transformers (not project dependencies; CPU torch is enough). It also writes `expected.json`: for each
tensor the shard that holds it, its byte length in the file and its first and last 16 bytes, computed
from the tensors themselves, so a loader can be checked without trusting the loader's own parser.
Run `hyperion-compile --model qwen2 --hw sm_90 --source <dir> --out <dir>/model.hyperion` afterwards to
refresh `model.manifest.json` (the Ignition test checks that file against a fresh compile).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "testdata" / "synthetic" / "qwen2-tiny-sharded"


def main() -> None:
    import torch
    from safetensors import safe_open
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    model = Qwen2ForCausalLM(config).to(torch.bfloat16)
    if OUT.exists():
        shutil.rmtree(OUT)
    model.save_pretrained(OUT, max_shard_size="48KB", safe_serialization=True)
    for stray in ("generation_config.json",):
        (OUT / stray).unlink(missing_ok=True)

    index = json.loads((OUT / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) < 3:
        sys.exit(f"expected at least 3 shards, got {shards}")

    expected = {}
    for shard in shards:
        with safe_open(OUT / shard, framework="pt") as handle:
            for name in handle.keys():
                raw = handle.get_tensor(name).contiguous().view(torch.uint8).flatten().tolist()
                data = bytes(raw)
                expected[name] = {"shard": shard, "length": len(data), "first16": data[:16].hex(), "last16": data[-16:].hex()}
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"wrote {OUT}: {len(shards)} shards, {len(expected)} tensors, {total} bytes")


if __name__ == "__main__":
    main()
