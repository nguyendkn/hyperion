#!/usr/bin/env python3
"""Prefix-cache benchmark (standard library only): two requests that share a long prefix, the second sent after the first finished.

For each round a fresh prefix (drawn from --seed and the round number, so no earlier round shares it) of about --prefix-tokens tokens is built, the
word count calibrated by the server's own `prompt_tokens` (see loadgen.py). Request A is the prefix plus a first question, request B the same prefix
plus a different question. TTFT of each is the time to the first streamed chunk. With the prefix cache the second TTFT covers only the tokens after
the shared blocks; the server log line `request admitted` carries `cached_tokens` for each request (the hit count). Output is JSON on stdout.

  python scripts/prefix-bench.py --url http://127.0.0.1:8080 --prefix-tokens 8192 --rounds 3
"""
import argparse
import json
import os
import random
import statistics
import sys
import threading
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loadgen  # noqa: E402


def request_body(model, text, max_tokens):
    return json.dumps(
        {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "stream": True, "temperature": 0, "stream_options": {"include_usage": True}}
    ).encode()


def send(host, port, path, headers, body):
    results = []
    loadgen.one_request(host, port, path, body, headers, results, threading.Lock())
    r = results[0]
    if r["error"] or not r["stamps"]:
        raise SystemExit(f"request failed: {r['error'] or 'no token streamed'}")
    return {"ttft_ms": round((r["stamps"][0] - r["start"]) * 1000, 2), "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="hyperion")
    ap.add_argument("--prefix-tokens", type=int, default=8192)
    ap.add_argument("--suffix-words", type=int, default=24)
    ap.add_argument("--output-tokens", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tolerance", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    url = urllib.parse.urlparse(args.url)
    host, port = url.hostname, url.port or 80
    path = (url.path.rstrip("/") or "") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    words = args.prefix_tokens
    for _ in range(4):  # calibrate the word count of the prefix on the server's tokenizer
        probe = send(host, port, path, headers, request_body(args.model, loadgen.prompt_for(random.Random(args.seed - 1), words), 1))
        if abs(probe["prompt_tokens"] - args.prefix_tokens) <= args.tolerance:
            break
        words = max(1, round(words * args.prefix_tokens / probe["prompt_tokens"]))
    rounds = []
    for r in range(args.rounds):
        rng = random.Random(args.seed * 1000 + r)
        prefix = loadgen.prompt_for(rng, words)
        a = send(host, port, path, headers, request_body(args.model, prefix + " Question one: " + loadgen.prompt_for(rng, args.suffix_words), args.output_tokens))
        b = send(host, port, path, headers, request_body(args.model, prefix + " Question two: " + loadgen.prompt_for(rng, args.suffix_words), args.output_tokens))
        rounds.append({"round": r, "first": a, "second": b, "ratio_second_over_first": round(b["ttft_ms"] / a["ttft_ms"], 4)})
        print(json.dumps(rounds[-1]), flush=True)
    summary = {
        "prefix_words": words,
        "first_ttft_ms_median": statistics.median(x["first"]["ttft_ms"] for x in rounds),
        "second_ttft_ms_median": statistics.median(x["second"]["ttft_ms"] for x in rounds),
        "ratio_median": statistics.median(x["ratio_second_over_first"] for x in rounds),
        "rounds": rounds,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
