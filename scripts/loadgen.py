#!/usr/bin/env python3
"""Load generator for `POST /v1/chat/completions` (standard library only).

Runs `--requests` streaming requests per concurrency level with a fixed prompt size and `max_tokens`, and reports TTFT p50/p90, inter-token latency
p50/p90, output tokens/s (total and per request) and errors, as JSON. The prompt is `--input-tokens` words drawn by `--seed` from a fixed list of
common one-token English words, so the prompt length in tokens is close to, not exactly, the word count (the server's `prompt_tokens` is reported when
the stream carries a usage record). One streamed chunk is counted as one token: the scheduler streams one token per event.

  python scripts/loadgen.py --url http://127.0.0.1:8080 --input-tokens 512 --output-tokens 256 --concurrency 1,2,4,8,16,32 --out loadgen.json
"""
import argparse
import http.client
import json
import random
import statistics
import sys
import threading
import time
import urllib.parse

WORDS = (
    "the of and to in is that it was for on are as with his they at be this from have or by one had not but what all were when we there can an your "
    "which their said if do will each about how up out them then she many some so these would other into has more her two like him see time could no "
    "make than first been its who now people my made over did down only way find use may water long little very after words called just where most "
    "know get through back much before go good new write our used me man too any day same right look think also around another came come work three "
    "word must because does part even place well such here take why things help put years different away again off went old number great tell men say "
    "small every found still between name should home big give air line set own under read last never us left end along while might next sound below "
    "saw something thought both few those always looked show large often together asked house don world going want school important until form food "
    "keep children feet land side without boy once animals life enough took sometimes four head above kind began almost live page got earth need far "
    "hand high year mother light country father let night picture being study second soon story since white ever paper hard near sentence better best "
    "across during today however sure knew try told young sun thing whole hear example heard several change answer room sea against top turned learn "
    "point city play toward five using himself usually"
).split()


def prompt_for(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def one_request(host, port, path, body, headers, results, lock):
    start = time.perf_counter()
    stamps = []
    error = None
    prompt_tokens = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=600)
        conn.request("POST", path, body=body, headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            error = f"HTTP {response.status}: {response.read(300).decode('utf-8', 'replace')}"
        else:
            while True:
                line = response.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    prompt_tokens = event["usage"].get("prompt_tokens")
                for choice in event.get("choices", []):
                    if choice.get("delta", {}).get("content"):
                        stamps.append(time.perf_counter())
        conn.close()
    except Exception as exc:  # a failed request is a result, not a crash
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    with lock:
        results.append({"start": start, "end": end, "stamps": stamps, "error": error, "prompt_tokens": prompt_tokens})


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def run_level(args, host, port, path, headers, concurrency):
    rng = random.Random(args.seed)
    bodies = []
    for _ in range(args.requests):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt_for(rng, args.input_tokens)}],
            "max_tokens": args.output_tokens,
            "stream": True,
            "temperature": 0,
        }
        if args.stream_usage:
            payload["stream_options"] = {"include_usage": True}
        bodies.append(json.dumps(payload).encode())
    results, lock = [], threading.Lock()
    queue, qlock = list(range(len(bodies))), threading.Lock()

    def worker():
        while True:
            with qlock:
                if not queue:
                    return
                index = queue.pop(0)
            one_request(host, port, path, bodies[index], headers, results, lock)

    began = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - began
    good = [r for r in results if r["error"] is None and r["stamps"]]
    ttft = [r["stamps"][0] - r["start"] for r in good]
    itl = [b - a for r in good for a, b in zip(r["stamps"], r["stamps"][1:])]
    tokens = sum(len(r["stamps"]) for r in good)
    per_request = [len(r["stamps"]) / (r["stamps"][-1] - r["start"]) for r in good if r["stamps"][-1] > r["start"]]
    ms = lambda v: None if v is None else round(v * 1000, 3)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": [r["error"] for r in results if r["error"]][:5],
        "error_count": sum(1 for r in results if r["error"]),
        "output_tokens": tokens,
        "wall_s": round(wall, 3),
        "ttft_ms": {"p50": ms(percentile(ttft, 0.5)), "p90": ms(percentile(ttft, 0.9))},
        "itl_ms": {"p50": ms(percentile(itl, 0.5)), "p90": ms(percentile(itl, 0.9))},
        "tokens_per_s_total": round(tokens / wall, 2) if wall > 0 else None,
        "tokens_per_s_per_request": round(statistics.median(per_request), 2) if per_request else None,
        "prompt_tokens": next((r["prompt_tokens"] for r in good if r["prompt_tokens"] is not None), None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="hyperion")
    ap.add_argument("--input-tokens", type=int, default=512)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--concurrency", default="1,2,4,8,16,32", help="comma-separated levels")
    ap.add_argument("--requests", type=int, default=0, help="requests per level (default: 4 x the concurrency, at least 8)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream-usage", action="store_true", help="ask for a usage record in the stream (stream_options.include_usage)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default=None, help="write the JSON report here (default: stdout only)")
    args = ap.parse_args()
    url = urllib.parse.urlparse(args.url)
    host, port = url.hostname, url.port or 80
    path = (url.path.rstrip("/") or "") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    levels = [int(x) for x in args.concurrency.split(",") if x]
    report = {"url": args.url, "input_words": args.input_tokens, "output_tokens": args.output_tokens, "seed": args.seed, "levels": []}
    base_requests = args.requests
    for level in levels:
        args.requests = base_requests or max(8, 4 * level)
        result = run_level(args, host, port, path, headers, level)
        report["levels"].append(result)
        print(json.dumps(result), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 1 if any(r["error_count"] for r in report["levels"]) else 0


if __name__ == "__main__":
    sys.exit(main())
