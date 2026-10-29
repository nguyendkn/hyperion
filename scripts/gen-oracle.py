#!/usr/bin/env python3
"""Generate the logit oracle for a model in testdata/models.json.

Runs the Hugging Face reference implementation at the pinned revision on a
fixed prompt and writes `testdata/oracle/<id>.json`: the prompt ids, the
next-token logits (float32, full vocabulary) and a tolerance declared BEFORE any
Hyperion run. Needs `torch` and `transformers` (not project dependencies; run it
where they exist, e.g. a GPU pod).

Tolerance rule: run the same forward twice with two independent HF attention
implementations (eager and sdpa). `observed` is their max absolute logit
difference; the declared tolerance is `max(4 * observed, 0.05)`. A Hyperion
forward must stay within it, and must also match the argmax and at least 4 of the
top-5 (the max-abs bound alone is loose: bf16 logits reach ~18). The rule is fixed here so the number is not chosen
after seeing Hyperion's result.

    python scripts/gen-oracle.py qwen2.5-0.5b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "testdata" / "models.json").read_text(encoding="utf-8"))["models"]
PROMPT = "The capital of France is"
GENERATE_TOKENS = 16


def main() -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if len(sys.argv) != 2 or sys.argv[1] not in REGISTRY:
        sys.exit(f"usage: gen-oracle.py <{'|'.join(REGISTRY)}>")
    model_id = sys.argv[1]
    spec = REGISTRY[model_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

    def last_logits(attn: str) -> torch.Tensor:
        model = AutoModelForCausalLM.from_pretrained(
            spec["repo"], revision=spec["revision"], torch_dtype=torch.bfloat16, attn_implementation=attn
        ).to(device).eval()
        with torch.no_grad():
            out = model(ids).logits[0, -1].float().cpu()
        del model
        return out

    sdpa = last_logits("sdpa")
    eager = last_logits("eager")

    # Greedy continuation by the reference implementation (decode path check).
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["revision"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    with torch.no_grad():
        generated = model.generate(ids, max_new_tokens=GENERATE_TOKENS, do_sample=False)[0, ids.shape[1]:].tolist()
    del model
    observed = float((sdpa - eager).abs().max())
    tolerance = max(4 * observed, 0.05)

    oracle = {
        "model": model_id,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "prompt": PROMPT,
        "prompt_ids": ids[0].tolist(),
        "dtype": "bfloat16",
        "reference": {"device": device, "torch": torch.__version__, "transformers": transformers.__version__},
        "observed_eager_vs_sdpa_max_abs": observed,
        "tolerance_max_abs": tolerance,
        "criteria": [
            "max_abs(logits - oracle) <= tolerance_max_abs",
            "argmax equals oracle argmax",
            "at least 4 of the oracle's top-5 are in the candidate's top-5",
        ],
        "next_token_top5": [int(i) for i in sdpa.topk(5).indices],
        "generated_ids": generated,
        "generated_text": tokenizer.decode(generated),
        "generation_criterion": "the first 8 generated ids must equal ours (greedy); later ids may diverge on near-tied bf16 logits",
        "logits": [float(x) for x in sdpa],
    }
    out = ROOT / "testdata" / "oracle" / f"{model_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(oracle) + "\n", encoding="utf-8", newline="\n")
    top = oracle["next_token_top5"]
    print(f"wrote {out}: {len(ids[0])} prompt tokens, vocab {len(oracle['logits'])}, top5 {top}")
    print(f"greedy continuation: {oracle['generated_text']!r}")
    print(f"decoded top1: {tokenizer.decode([top[0]])!r}; observed {observed:.4f}; tolerance {tolerance:.4f}")


if __name__ == "__main__":
    main()
