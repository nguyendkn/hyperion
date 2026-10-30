"""Write `testdata/synthetic/qwen2.5-7b-tensors.json`: the tensor names, shapes and sharding classes of the real Qwen2.5-7B
checkpoint, read from its four safetensors shard headers (`python scripts/fetch-model-fixtures.py qwen2.5-7b` downloads them; the
weights are not needed). Ignition's own classifier assigns the classes, so the Rust test that slices every tensor for each supported
tensor-parallel size runs on the real tensor list without the 15 GB of weights. Regenerate only when the registry revision changes.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ignition" / "src"))

from hyperion_ignition.compiler.base import supported_tp_sizes  # noqa: E402
from hyperion_ignition.compiler.model_config import model_config_from_hf  # noqa: E402
from hyperion_ignition.compiler.weights import _qwen2_sharding  # noqa: E402

SOURCE = ROOT / "testdata" / "models" / "qwen2.5-7b"
OUT = ROOT / "testdata" / "synthetic" / "qwen2.5-7b-tensors.json"


def main() -> None:
    config = model_config_from_hf(SOURCE, ("Qwen2ForCausalLM",), qkv_bias=True)
    tensors = []
    for header in sorted(SOURCE.glob("*.safetensors.header")):
        raw = header.read_bytes()
        (length,) = struct.unpack("<Q", raw[:8])
        for name, entry in json.loads(raw[8 : 8 + length]).items():
            if name == "__metadata__":
                continue
            tensors.append({"name": name.removeprefix("model."), "shape": entry["shape"], "sharding": _qwen2_sharding(name)})
    tensors.sort(key=lambda t: t["name"])
    OUT.write_text(
        json.dumps(
            {
                "model": "qwen2.5-7b",
                "num_attention_heads": config.num_attention_heads,
                "num_kv_heads": config.num_kv_heads,
                "head_dim": config.head_dim,
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "vocab_size": config.vocab_size,
                "num_layers": config.num_layers,
                "supported_tp_sizes": supported_tp_sizes(config),
                "tensors": tensors,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
