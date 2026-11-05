#!/usr/bin/env python3
"""CPU speed / RAM / quality of the Qwen3-ASR reference under different settings (dev-machine path; not the Helios engine).

One configuration per process, so the peak RSS is that configuration's own. Each run appends one JSON line to --out with load time, wall
time, real-time factor, peak RSS (sampled every 50 ms from process start, model load included) and the transcripts. `--compare` reads the
lines and prints, per configuration, speed-up and memory against the fp32 batch-1 baseline and the WER of each configuration against the
fp32 text and against the dataset reference text, per language.

    <venv python> -I scripts/bench-stt-cpu.py --dtype bf16 --quant none --batch 4 --threads 8 --n 6 --out RESULTS.jsonl
    <venv python> -I scripts/bench-stt-cpu.py --compare RESULTS.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.path.expandvars("%LOCALAPPDATA%")) / "hyperion" / "data" / "speech"
MODEL_DIR = ROOT / "testdata" / "models" / "qwen3-asr-1.7b"
CLIPS = ROOT / "testdata" / "oracle" / "qwen3-asr-1.7b.stt-clips.json"


def norm(t: str) -> list[str]:
    t = unicodedata.normalize("NFC", t.lower())
    return re.sub(r"[^\w\s]", " ", t).replace("_", " ").split()


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)], len(ref)


def run(a) -> None:
    import psutil
    import soundfile as sf
    import torch

    proc, peak, stop = psutil.Process(), [0], threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.05)

    threading.Thread(target=sample, daemon=True).start()
    if a.threads:
        torch.set_num_threads(a.threads)
    from qwen_asr import Qwen3ASRModel

    clips = json.loads(CLIPS.read_text(encoding="utf-8"))
    items = [(lang, c) for lang in ("en", "vi") for c in clips["clips"][lang][: a.n]]
    t0 = time.time()
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    m = Qwen3ASRModel.from_pretrained(str(MODEL_DIR), dtype=dtype, device_map="cpu", max_new_tokens=300, max_inference_batch_size=a.batch)
    if a.quant == "int8":
        th = m.model.thinker
        for sub in (th.audio_tower, th.model):  # lm_head stays fp32 (tied to the embedding)
            torch.ao.quantization.quantize_dynamic(sub, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    load = time.time() - t0
    audio = []
    for lang, c in items:
        wav, _ = sf.read(io.BytesIO((DATA / "clips" / c["file"]).read_bytes()), dtype="float32")
        audio.append((wav, 16000))
    secs = sum(len(w) for w, _ in audio) / 16000
    langs = [clips["datasets"][lang]["force_language"] for lang, _ in items]
    m.transcribe(audio=audio[:1], language=langs[:1])  # warm-up (allocator, kernels), not timed
    t1 = time.time()
    res = m.transcribe(audio=audio, language=langs)
    wall = time.time() - t1
    stop.set()
    line = {"dtype": a.dtype, "quant": a.quant, "batch": a.batch, "threads": a.threads or torch.get_num_threads(), "clips": len(items), "audio_s": round(secs, 1),
            "load_s": round(load, 1), "wall_s": round(wall, 1), "rtf": round(wall / secs, 3), "peak_rss_gb": round(peak[0] / 2**30, 2),
            "items": [{"lang": lang, "id": c["id"], "ref": c["reference_text"], "text": r.text} for (lang, c), r in zip(items, res)]}
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print({k: v for k, v in line.items() if k != "items"}, flush=True)


def compare(path: str) -> None:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]
    base = next(r for r in rows if (r["dtype"], r["quant"], r["batch"]) == ("fp32", "none", 1))
    print(f"{'config':32} {'wall s':>7} {'speedup':>8} {'RTF':>6} {'RSS GB':>7} {'load s':>7} | WER vs fp32 en/vi | WER vs ref en/vi")
    for r in rows:
        w = {}
        for lang in ("en", "vi"):
            for key, src in (("fp32", lambda i: base["items"][i]["text"]), ("ref", lambda i: r["items"][i]["ref"])):
                e = n = 0
                for i, it in enumerate(r["items"]):
                    if it["lang"] == lang:
                        a, b = wer(norm(src(i)), norm(it["text"]))
                        e, n = e + a, n + b
                w[(key, lang)] = 100 * e / max(n, 1)
        name = f"{r['dtype']} {r['quant']} b{r['batch']} t{r['threads']}"
        print(f"{name:32} {r['wall_s']:7.1f} {base['wall_s'] / r['wall_s']:7.2f}x {r['rtf']:6.2f} {r['peak_rss_gb']:7.2f} {r['load_s']:7.1f} | "
              f"{w[('fp32', 'en')]:5.2f} {w[('fp32', 'vi')]:5.2f}      | {w[('ref', 'en')]:5.2f} {w[('ref', 'vi')]:5.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--quant", choices=("none", "int8"), default="none")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--n", type=int, default=6, help="clips per language")
    ap.add_argument("--out", default="bench-stt-cpu.jsonl")
    ap.add_argument("--compare")
    a = ap.parse_args()
    compare(a.compare) if a.compare else run(a)
