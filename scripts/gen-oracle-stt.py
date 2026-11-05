#!/usr/bin/env python3
"""STT oracle for Qwen3-ASR (ADR 0013, speech track W4): what the reference computes on the fixed clips, in float32 and bfloat16, and the
tolerances a Helios run must stay within, written to the file BEFORE any Helios run.

Per clip (clips: `testdata/oracle/<model>.stt-clips.json`, audio under %LOCALAPPDATA%/hyperion/data/speech/clips):
  features   log-mel as the reference feature extractor returns it: shape, per-band mean, per-frame mean, extremes, fixed slices; full dump of two clips
  encoder    after `conv_out`, after each of the 24 layers, after `ln_post`, after the projector (the embeddings spliced into the decoder):
             mean, std, max |x|, three fixed rows x eight fixed columns
  decoder    greedy generation of the transcript (forced language, so the prompt ends in `language X<asr_text>`), then one teacher-forced forward over
             prompt + generated ids: at each predicting position the argmax, top-20 logits with ids and the logsumexp of the whole row
  text       the greedy text

Tolerances (declared method, same as scripts/gen-oracle-large.py): bf16-vs-fp32 drift measured here, times 1.5, never below a floor.
  encoder tap   relative to that tap's fp32 max |x| of the clip: max(1.5 * worst observed relative drift over clips, 0.01)
  decoder       absolute on the top-k logits and on the logsumexp: max(1.5 * worst observed drift, 0.02); argmax rule as the LLM oracles
  features      absolute 1e-3 on the normalised features (both sides are float32 pipelines; declared, not measured)

    PYTHONIOENCODING=utf-8 <oracle-qwen3asr venv python> -I scripts/gen-oracle-stt.py [--limit N]

Resumable: per-clip results are appended to a scratch JSONL. Prints the git revision, this script's sha256 and the pinned weight hashes first.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen3-asr-1.7b"
REG = json.loads((ROOT / "testdata" / "models-speech.json").read_text(encoding="utf-8"))["models"][MODEL_ID]
MODEL_DIR = ROOT / "testdata" / "models" / MODEL_ID
DATA = Path(os.path.expandvars("%LOCALAPPDATA%")) / "hyperion" / "data" / "speech"
MARGIN, ENC_FLOOR, LOGIT_FLOOR, MEL_TOL, TOPK, MAX_NEW = 1.5, 0.01, 0.02, 1e-3, 20, 300
SLICE_ROWS, SLICE_COLS = (0.0, 0.5, 1.0), 8
FULL_DUMP = {"en": 0, "vi": 0}  # index of the clip whose full features are dumped


def r6(x):
    return float(f"{x:.6g}")


def summ(t: torch.Tensor) -> dict:
    x = t.detach().float().reshape(-1, t.shape[-1]) if t.dim() > 1 else t.detach().float().reshape(1, -1)
    n = x.shape[0]
    rows = [min(n - 1, int(round(f * (n - 1)))) for f in SLICE_ROWS]
    return {"shape": list(t.shape), "mean": r6(x.mean()), "std": r6(x.std()), "max_abs": r6(x.abs().max()),
            "rows": rows, "slice": [[r6(v) for v in x[r, :SLICE_COLS].tolist()] for r in rows]}


def feat_summary(f: torch.Tensor, mask_len: int) -> dict:
    x = f[0, :, :mask_len].float()
    sl = [[r6(v) for v in x[b, c:c + 8].tolist()] for b in (0, 31, 63, 95, 127) for c in (0, mask_len // 2)]
    return {"shape": list(x.shape), "max": r6(x.max()), "min": r6(x.min()), "band_mean": [r6(v) for v in x.mean(1).tolist()],
            "frame_mean": [round(v, 5) for v in x.mean(0).tolist()], "slices_bands_0_31_63_95_127_at_frames_0_mid": sl}


class Tap:
    def __init__(self, tower):
        self.out: dict[str, torch.Tensor] = {}
        self.handles = [tower.conv_out.register_forward_hook(self._h("conv_out")), tower.ln_post.register_forward_hook(self._h("ln_post")),
                        tower.proj2.register_forward_hook(self._h("embeds"))]
        self.handles += [l.register_forward_hook(self._h(f"layer_{i:02d}")) for i, l in enumerate(tower.layers)]

    def _h(self, name):
        def h(_m, _i, o):
            self.out[name] = (o[0] if isinstance(o, tuple) else o).detach().clone()
        return h

    def take(self) -> dict:
        o, self.out = self.out, {}
        return o


def decoder_records(logits: torch.Tensor) -> list[dict]:
    lg = logits.float()
    lse = torch.logsumexp(lg, dim=-1)
    top = lg.topk(TOPK, dim=-1)
    return [{"argmax": int(top.indices[i, 0]), "top_ids": top.indices[i].tolist(), "top_logits": [r6(v) for v in top.values[i].tolist()], "logsumexp": r6(lse[i])} for i in range(lg.shape[0])]


def run_clip(m, clip: dict, lang: str, meta: dict, taps: Tap, dtype: torch.dtype, forced_ids: list[int] | None) -> dict:
    raw = (DATA / "clips" / clip["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != clip["audio_sha256"]:
        sys.exit(f"{clip['file']}: audio sha256 differs from the clips file; run scripts/gen-oracle-stt-clips.py")
    wav, _ = sf.read(io.BytesIO(raw), dtype="float32")
    prompt = m._build_text_prompt("", meta["force_language"])
    inp = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
    mask_len = int(inp["feature_attention_mask"].sum())
    inp = inp.to(m.model.device).to(dtype)
    model, p_len = m.model, inp["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=MAX_NEW).sequences[0, p_len:].tolist()
        ids = forced_ids if forced_ids is not None else gen
        full = torch.tensor([inp["input_ids"][0].tolist() + ids])
        taps.take()
        out = model(input_ids=full, attention_mask=torch.ones_like(full), input_features=inp["input_features"], feature_attention_mask=inp["feature_attention_mask"])
        enc = {k: summ(v) for k, v in taps.take().items()}
        recs = decoder_records(out.logits[0, p_len - 1:p_len - 1 + len(ids)])
    rec = {"id": clip["id"], "lang": lang, "prompt_ids": inp["input_ids"][0].tolist(), "generated_ids": gen, "forced_ids": ids,
           "text": m.processor.batch_decode([gen], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0],
           "features": feat_summary(inp["input_features"].float(), mask_len) if dtype == torch.float32 else None, "encoder": enc, "decoder": recs}
    if dtype == torch.float32 and clip["_index"] == FULL_DUMP[lang]:
        b = inp["input_features"][0, :, :mask_len].float().numpy().astype("<f4").tobytes()
        dump = ROOT / "testdata" / "oracle" / f"{MODEL_ID}.stt-mel-{lang}-{clip['id']}.f32"
        dump.write_bytes(b)
        rec["features"]["full_dump"] = {"file": dump.name, "layout": "[128 bands][frames] little-endian f32, row major", "sha256": hashlib.sha256(b).hexdigest()}
    return rec


def run_all(dtype: torch.dtype, clips: dict, fp32: dict | None, scratch: Path, limit: int | None) -> dict:
    from qwen_asr import Qwen3ASRModel
    done = {}
    if scratch.exists():
        for line in scratch.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["lang"], r["id"])] = r
    m = Qwen3ASRModel.from_pretrained(str(MODEL_DIR), dtype=dtype, device_map="cpu", max_new_tokens=MAX_NEW)
    taps = Tap(m.model.thinker.audio_tower)
    t0 = time.time()
    with scratch.open("a", encoding="utf-8") as f:
        for lang, lst in clips["clips"].items():
            for i, c in enumerate(lst[:limit]):
                if (lang, c["id"]) in done:
                    continue
                c["_index"] = i
                forced = fp32[(lang, c["id"])]["generated_ids"] if fp32 else None
                r = run_clip(m, c, lang, clips["datasets"][lang], taps, dtype, forced)
                done[(lang, c["id"])] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{str(dtype)[6:]} {lang} {c['id']} {time.time() - t0:.0f}s {r['text'][:60]!r}", flush=True)
    del m, taps
    gc.collect()
    return done


def tolerances(f32: dict, b16: dict) -> dict:
    enc_drift: dict[str, float] = {}
    lg_drift = lse_drift = 0.0
    for key, a in f32.items():
        b = b16[key]
        for tap, sa in a["encoder"].items():
            sb = b["encoder"][tap]
            scale = max(sa["max_abs"], 1e-9)
            d = max(abs(sa["mean"] - sb["mean"]), abs(sa["std"] - sb["std"]), abs(sa["max_abs"] - sb["max_abs"]),
                    max(abs(x - y) for ra, rb in zip(sa["slice"], sb["slice"]) for x, y in zip(ra, rb))) / scale
            enc_drift[tap] = max(enc_drift.get(tap, 0.0), d)
        for ra, rb in zip(a["decoder"], b["decoder"]):
            fb = dict(zip(rb["top_ids"], rb["top_logits"]))
            lg_drift = max(lg_drift, max(abs(l - fb[i]) for i, l in zip(ra["top_ids"], ra["top_logits"]) if i in fb))
            lse_drift = max(lse_drift, abs(ra["logsumexp"] - rb["logsumexp"]))
    return {"method": "bf16-vs-fp32 drift measured on these clips, times margin, never below the floor; declared before any Helios run",
            "margin": MARGIN, "encoder_floor_rel": ENC_FLOOR, "logit_floor_abs": LOGIT_FLOOR, "mel_abs": MEL_TOL,
            "encoder_rel_to_tap_max_abs": {t: r6(max(MARGIN * d, ENC_FLOOR)) for t, d in sorted(enc_drift.items())},
            "observed_encoder_drift_rel": {t: r6(d) for t, d in sorted(enc_drift.items())},
            "top_logits_abs": r6(max(MARGIN * lg_drift, LOGIT_FLOOR)), "logsumexp_abs": r6(max(MARGIN * lse_drift, LOGIT_FLOOR)),
            "observed_top_logits_drift": r6(lg_drift), "observed_logsumexp_drift": r6(lse_drift),
            "argmax_rule": "equal to the fp32 argmax, or one of the recorded top-k whose fp32 logit is within top_logits_abs of the best (a near tie may flip)",
            "text_rule": "free greedy decoding must equal the fp32 text, or be within the WER budget of ADR 0013 on the fixed subsets"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print(f"ORACLE git_revision={rev} script_sha256={hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()}")
    for n, w in REG["weights"].items():
        print(f"ORACLE weights {n} sha256={w['sha256']}")
    clips = json.loads((ROOT / "testdata" / "oracle" / f"{MODEL_ID}.stt-clips.json").read_text(encoding="utf-8"))
    scratch = Path(os.path.expandvars("%TEMP%")) / "hyperion-scratch-speech"
    scratch.mkdir(parents=True, exist_ok=True)
    f32 = run_all(torch.float32, clips, None, scratch / "oracle-fp32.jsonl", a.limit)
    b16 = run_all(torch.bfloat16, clips, f32, scratch / "oracle-bf16.jsonl", a.limit)
    keys = sorted(f32, key=lambda k: (k[0], k[1]))
    out = {"model": MODEL_ID, "revision": REG["revision"], "git_revision": rev, "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
           "generation": {"forced_language": True, "greedy": True, "max_new_tokens": MAX_NEW}, "tolerance": tolerances(f32, b16),
           "clips": [{"lang": k[0], "id": k[1], "fp32": f32[k], "bf16": {x: y for x, y in b16[k].items() if x not in ("features", "prompt_ids")}} for k in keys]}
    dest = ROOT / "testdata" / "oracle" / f"{MODEL_ID}.stt.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print("wrote", dest, dest.stat().st_size, "bytes; tolerance:", json.dumps({k: v for k, v in out["tolerance"].items() if k.startswith(("top", "logsum", "observed_top", "observed_logsum"))}))


if __name__ == "__main__":
    main()
