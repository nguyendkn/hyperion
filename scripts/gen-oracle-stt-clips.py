#!/usr/bin/env python3
"""Select the fixed STT oracle clips and extract their audio (ADR 0013, speech track W4).

Rule, declared here before any model runs: from each dataset split (LibriSpeech `clean`/`test`, FLEURS `vi_vn`/`test`) take the clips whose
duration is 4 to 12 s, sort them by id (LibriSpeech: string id; FLEURS: integer id), and take every `step`-th one starting at the first,
`step = eligible // 20`, until 20 clips are chosen. No randomness, so any later session or pod picks the same clips.

Writes `testdata/oracle/<model>.stt-clips.json` (committed: rule, dataset repo and revision, sha256 of the parquet, and per clip id, reference
text, duration, sha256 of the raw audio bytes, file name) and the raw audio bytes (flac or wav, as stored in the dataset) under
`%LOCALAPPDATA%/hyperion/data/speech/clips/<lang>/` (outside the repo; the sha256 in the JSON says whether they are the right files).

    python scripts/gen-oracle-stt-clips.py [--data-dir DIR]

Needs pyarrow and soundfile (the reference venv `%LOCALAPPDATA%/hyperion/venvs/oracle-qwen3asr` has them).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
N = 20
MIN_S, MAX_S = 4.0, 12.0
SPLITS = {
    "en": {"file": "librispeech_test_clean.parquet", "repo": "openslr/librispeech_asr", "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
           "config": "clean", "split": "test", "text": "text", "id": "id", "ext": "flac", "license": "CC-BY-4.0", "force_language": "English"},
    "vi": {"file": "fleurs_vi_test.parquet", "repo": "google/fleurs", "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
           "config": "vi_vn", "split": "test", "text": "transcription", "id": "id", "ext": "wav", "license": "CC-BY-4.0", "force_language": "Vietnamese"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def select(lang: str, spec: dict, data_dir: Path) -> tuple[dict, list[tuple[dict, bytes]]]:
    path = data_dir / spec["file"]
    table = pq.ParquetFile(path).read(columns=[spec["id"], spec["text"], "audio"]).to_pylist()
    rows = []
    for r in table:
        raw = r["audio"]["bytes"]
        info = sf.info(io.BytesIO(raw))
        if info.samplerate != 16000 or info.channels != 1:
            raise SystemExit(f"{spec['repo']} {r[spec['id']]}: expected 16 kHz mono, got {info.samplerate} Hz x {info.channels}")
        if MIN_S <= info.duration <= MAX_S:
            rows.append((r[spec["id"]], r[spec["text"]], info.duration, raw))
    rows.sort(key=lambda x: x[0] if isinstance(x[0], int) else str(x[0]))
    step = len(rows) // N
    chosen = [rows[i * step] for i in range(N)]
    meta = {k: spec[k] for k in ("repo", "revision", "config", "split", "license", "force_language")}
    meta |= {"parquet": spec["file"], "parquet_sha256": sha256_file(path), "eligible": len(rows), "step": step}
    clips = [({"id": str(i), "reference_text": t, "duration_s": round(d, 3), "audio_sha256": hashlib.sha256(raw).hexdigest(), "file": f"{lang}/{i}.{spec['ext']}"}, raw) for i, t, d, raw in chosen]
    return meta, clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(os.path.expandvars("%LOCALAPPDATA%")) / "hyperion" / "data" / "speech")
    ap.add_argument("--model", default="qwen3-asr-1.7b")
    a = ap.parse_args()
    out: dict = {"_comment": "Fixed STT oracle clips. Rule in scripts/gen-oracle-stt-clips.py: duration 4 to 12 s, sorted by id, every (eligible // 20)-th, 20 per language.", "datasets": {}, "clips": {}}
    for lang, spec in SPLITS.items():
        meta, clips = select(lang, spec, a.data_dir)
        out["datasets"][lang] = meta
        out["clips"][lang] = [c for c, _ in clips]
        (a.data_dir / "clips" / lang).mkdir(parents=True, exist_ok=True)
        for c, raw in clips:
            (a.data_dir / "clips" / c["file"]).write_bytes(raw)
        print(f"{lang}: {meta['eligible']} eligible, step {meta['step']}, {len(clips)} clips, {sum(c['duration_s'] for c, _ in clips):.0f} s")
    dest = ROOT / "testdata" / "oracle" / f"{a.model}.stt-clips.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", dest)


if __name__ == "__main__":
    main()
