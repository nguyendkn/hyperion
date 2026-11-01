#!/usr/bin/env python3
"""Reproducibility of seeded sampling across server runs (standard library only).

Sends a fixed set of non-streaming requests with explicit seeds, temperature and top-p/top-k and prints one JSON object {request label: text}. Run it
against two server processes started one after the other and compare the two outputs byte for byte: the same request and seed must give the same text.

  python scripts/sampling-check.py --url http://127.0.0.1:8080 > run1.json   # restart the server
  python scripts/sampling-check.py --url http://127.0.0.1:8080 > run2.json && cmp run1.json run2.json
"""
import argparse
import json
import urllib.request

CASES = [
    {"label": "t1.0 seed 1", "temperature": 1.0, "seed": 1},
    {"label": "t0.8 p0.95 k50 seed 2", "temperature": 0.8, "top_p": 0.95, "top_k": 50, "seed": 2},
    {"label": "t0.7 k40 seed 3", "temperature": 0.7, "top_k": 40, "seed": 3},
    {"label": "t1.3 p0.9 seed 4", "temperature": 1.3, "top_p": 0.9, "seed": 4},
    {"label": "t0.8 rep1.1 seed 5", "temperature": 0.8, "repetition_penalty": 1.1, "seed": 5},
    {"label": "greedy", "temperature": 0},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="hyperion")
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()
    out = {}
    for case in CASES:
        body = {"model": args.model, "messages": [{"role": "user", "content": "Write one sentence about the sea."}], "max_tokens": args.max_tokens}
        body.update({k: v for k, v in case.items() if k != "label"})
        request = urllib.request.Request(args.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            out[case["label"]] = json.load(response)["choices"][0]["message"]["content"]
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
