#!/usr/bin/env python3
"""Compact logit oracle for a model too large to store full logits for (Qwen2.5-7B and up).

For a few prompts it runs the Hugging Face reference twice, one after the other (the GPU is freed in
between and the process exits before any Hyperion run):

  1. in float32, the reference: greedy-generates a short continuation, then one forward over
     prompt + continuation records, at every position that predicts a token (the last prompt position
     and each generated one), the argmax, the top-k logits with their ids, and the logsumexp of the
     full vocabulary row. Nothing else is kept, so the file is a few tens of KB, not 152 064 logits
     per position.
  2. in bfloat16, on the same sequences: the same records, to measure how far bf16 alone drifts from
     float32 on this model.

The tolerance a Hyperion forward must stay within is that measured drift times a margin, written to the
file before Hyperion runs (so it cannot be chosen after seeing a result):

    tolerance_max_abs_top_logits = max(margin * observed bf16-vs-fp32 max |logit diff| over the
                                       fp32 top-k ids, floor)
    tolerance_max_abs_logsumexp  = the same for |logsumexp diff|

Argmax rule: equal to the fp32 argmax, or, when the fp32 gap between its two best logits is not larger
than the logit tolerance, either of those two (a near tie may flip legitimately).

    python scripts/gen-oracle-large.py qwen2.5-7b                       # weights in testdata/models/qwen2.5-7b
    python scripts/gen-oracle-large.py tiny --source testdata/synthetic/qwen2-tiny-sharded --device cpu

Needs torch and transformers (not project dependencies). Prints the git revision, this script's sha256
and the pinned weight files first, so the output names its inputs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "testdata" / "models.json").read_text(encoding="utf-8"))["models"]
SHORT_PROMPT = "The capital of France is"
LONG_PROMPT = (
    "Inference engines turn a trained language model into a service. A request arrives as text, is split into "
    "tokens, and the model reads all of them at once in a step called prefill, which fills a cache of keys and "
    "values for every layer. After that the engine generates one token at a time, each step reading the cache and "
    "the weights again, which is why decoding is limited by memory bandwidth rather than arithmetic. A scheduler "
    "decides which requests share a step, and the cache must be sized so that many conversations fit in memory."
)
DEFAULT_MARGIN = 1.5
TOLERANCE_FLOOR = 0.02


def git_revision() -> str:
    explicit = os.environ.get("HYPERION_REVISION")
    if explicit:
        return explicit
    for candidate in (Path.cwd() / "REVISION", ROOT / "REVISION"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return "unknown"


def records_at(logits, positions, top_k):
    """argmax, top-k (ids and logits) and logsumexp of the float32 logits row of each position."""
    import torch

    out = []
    for position, row in zip(positions, logits):
        row = row.float()
        top = row.topk(top_k)
        out.append(
            {
                "position": position,
                "argmax": int(top.indices[0]),
                "top_ids": [int(i) for i in top.indices],
                "top_logits": [float(v) for v in top.values],
                "logsumexp": float(torch.logsumexp(row.double(), dim=0)),
            }
        )
    return out


def load_model(source: Path, dtype_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    kwargs = {"dtype": dtype, "attn_implementation": "sdpa"}
    on_gpu = device.startswith("cuda")
    if on_gpu:
        # Straight onto the GPU (needs `accelerate`): a float32 7B model is about 30 GB and should not first
        # sit in host memory.
        kwargs["device_map"] = {"": device}
    try:
        model = AutoModelForCausalLM.from_pretrained(str(source), **kwargs)
    except TypeError:  # older transformers call the argument torch_dtype
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(str(source), **kwargs)
    return (model if on_gpu else model.to(device)).eval()


def release(device: str) -> None:
    gc.collect()
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"ORACLE gpu memory after freeing the model: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free", flush=True)


def build_prompts(source: Path, vocab: int, contexts: list[int]):
    """(label, kind, ids) for each prompt. Natural text needs the tokenizer; without one (a synthetic
    model) fixed pseudo-random ids stand in for text."""
    tokenizer = None
    if (source / "tokenizer.json").is_file() or (source / "tokenizer_config.json").is_file():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(source))
    if tokenizer is not None:
        short = tokenizer(SHORT_PROMPT, add_special_tokens=False).input_ids
        paragraph = tokenizer(LONG_PROMPT, add_special_tokens=False).input_ids
    else:
        rng = random.Random(7)
        short = [rng.randrange(vocab) for _ in range(5)]
        paragraph = [rng.randrange(vocab) for _ in range(60)]
    prompts = [("short text", "text", short), ("paragraph", "text", paragraph)]
    for n in contexts:
        prompts.append((f"{n} tokens, the short prompt repeated", "cycled", [short[i % len(short)] for i in range(n)]))
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="registry id, or any name with --source")
    parser.add_argument("--source", type=Path, help="directory with config and weights (default: testdata/models/<model>)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--generate", type=int, default=16, help="greedy tokens recorded after each prompt")
    parser.add_argument("--contexts", default="1024", help="comma-separated lengths of the repeated-prompt contexts")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch
    import transformers

    spec = REGISTRY.get(args.model, {})
    source = args.source or (ROOT / "testdata" / "models" / args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or (ROOT / "testdata" / "oracle" / f"{args.model}.compact.json")
    if not (source / "config.json").is_file():
        sys.exit(f"{source}: no config.json; fetch the model first (python scripts/fetch-model-fixtures.py {args.model} --weights)")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))

    print(f"ORACLE git_revision={git_revision()} script_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}", flush=True)
    print(f"ORACLE model={args.model} repo={spec.get('repo', 'local')} revision={spec.get('revision', 'local')} source={source}", flush=True)
    print(f"ORACLE torch={torch.__version__} transformers={transformers.__version__} device={device}", flush=True)
    for name, info in spec.get("weights", {}).items():
        have = (source / name).stat().st_size if (source / name).is_file() else None
        print(f"ORACLE weight {name} bytes={info['bytes']} sha256={info['sha256']} on_disk={have}", flush=True)
        if have != info["bytes"]:
            sys.exit(f"{name}: {have} bytes on disk, expected {info['bytes']}: run fetch-model-fixtures.py --weights")

    prompts = build_prompts(source, int(config["vocab_size"]), [int(c) for c in args.contexts.split(",") if c])
    generated = args.generate

    # Pass 1: float32 reference.
    sequences, reference = [], []
    model = load_model(source, "float32", device)
    for label, kind, ids in prompts:
        tensor = torch.tensor([ids], device=device)
        with torch.no_grad():
            new = model.generate(tensor, max_new_tokens=generated, do_sample=False, pad_token_id=0)[0, len(ids):].tolist()
            sequence = torch.tensor([ids + new], device=device)
            rows = model(sequence, logits_to_keep=generated + 1).logits[0, :generated]
        positions = list(range(len(ids) - 1, len(ids) - 1 + generated))
        sequences.append((sequence, positions, new))
        reference.append(records_at(rows, positions, args.top_k))
        print(f"ORACLE fp32 {label}: {len(ids)} prompt tokens, generated {new[:6]}...", flush=True)
    del model
    release(device)

    # Pass 2: bfloat16 on the same sequences; keep only the deviation from float32.
    bf16_records, worst = [], {"top_logits": 0.0, "logsumexp": 0.0, "argmax_mismatches": 0, "near_tie_mismatches": 0, "positions": 0}
    model = load_model(source, "bfloat16", device)
    for (label, kind, ids), (sequence, positions, new), ref in zip(prompts, sequences, reference):
        with torch.no_grad():
            rows = model(sequence, logits_to_keep=generated + 1).logits[0, :generated].float()
        records = records_at(rows, positions, args.top_k)
        bf16_records.append(records)
        for row, ours, theirs in zip(rows, records, ref):
            ids_top = torch.tensor(theirs["top_ids"], device=row.device)
            diff = (row[ids_top].cpu() - torch.tensor(theirs["top_logits"])).abs().max().item()
            worst["top_logits"] = max(worst["top_logits"], diff)
            worst["logsumexp"] = max(worst["logsumexp"], abs(ours["logsumexp"] - theirs["logsumexp"]))
            worst["positions"] += 1
            if ours["argmax"] != theirs["argmax"]:
                worst["argmax_mismatches"] += 1
                if theirs["top_logits"][0] - theirs["top_logits"][1] <= 4 * diff:
                    worst["near_tie_mismatches"] += 1
    del model
    release(device)

    tolerance = {
        "margin": args.margin,
        "floor": TOLERANCE_FLOOR,
        "max_abs_top_logits": max(args.margin * worst["top_logits"], TOLERANCE_FLOOR),
        "max_abs_logsumexp": max(args.margin * worst["logsumexp"], TOLERANCE_FLOOR),
        "rule": (
            "Each tolerance is the largest difference seen between the bfloat16 and float32 Hugging Face runs "
            "(over the fp32 top-k ids, and over the logsumexp) times the margin, with a floor. A Hyperion bf16 "
            "forward sums in a different order than Hugging Face's, so its error against float32 is of the same "
            "size as, not identical to, bfloat16's; the margin covers that. Argmax: equal to the float32 argmax, "
            "or, when the float32 gap between its two best logits is within max_abs_top_logits, either of those two."
        ),
    }
    oracle = {
        "format": "compact-oracle-v1",
        "model": args.model,
        "repo": spec.get("repo", "local"),
        "revision": spec.get("revision", "local"),
        "generated_by": {
            "git_revision": git_revision(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
        },
        "top_k": args.top_k,
        "generated_tokens": generated,
        "bf16_vs_fp32": worst,
        "tolerance": tolerance,
        "criteria": [
            "for every compared position: max |our logit - reference logit| over the reference top-k ids <= tolerance.max_abs_top_logits",
            "for every compared position: |our logsumexp - reference logsumexp| <= tolerance.max_abs_logsumexp",
            "argmax: see tolerance.rule",
            "positions are compared while our greedy tokens equal the reference's; the first 8 generated tokens must agree",
        ],
        "prompts": [
            {"label": label, "kind": kind, "ids": ids, "generated_ids": new, "reference_fp32": ref, "hf_bf16": bf}
            for (label, kind, ids), (_, _, new), ref, bf in zip(prompts, sequences, reference, bf16_records)
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(oracle) + "\n", encoding="utf-8")
    print(
        f"ORACLE wrote {out} ({out.stat().st_size} bytes): bf16 vs fp32 over {worst['positions']} positions: "
        f"max top-k logit diff {worst['top_logits']:.4f}, max logsumexp diff {worst['logsumexp']:.4f}, "
        f"argmax mismatches {worst['argmax_mismatches']} (near ties {worst['near_tie_mismatches']}); "
        f"declared tolerance {tolerance['max_abs_top_logits']:.4f} / {tolerance['max_abs_logsumexp']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
