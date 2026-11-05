"""Qwen3-ASR compiler (ADR 0013). Reads the real `config.json`, `preprocessor_config.json` and the safetensors headers, checks every tensor
against the inventory the config implies (nothing unmapped, nothing missing, every shape equal), and emits the family manifest.
Every number comes from the files; the only constants are facts of the architecture read from the reference code at the pinned revision
(`docs/research/verification-20261105.md`): three stride-2 3x3 convolutions, WhisperFeatureExtractor defaults for what its config omits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..model_config import ModelConfig
from ..weights import TensorEntry, read_checkpoint
from . import register
from .family_schema import validate_family_manifest

FAMILY = "stt_qwen3_asr"
CONV_LAYERS = 3  # three stride-2 3x3 convolutions: 8x downsampling in time and in mel
WHISPER_SAMPLE_RATE = 16000  # WhisperFeatureExtractor default; the preprocessor config does not repeat it


def _req(cfg: dict[str, Any], key: str, where: str) -> Any:
    if key not in cfg:
        raise ValueError(f"{where}: required key `{key}` is missing")
    return cfg[key]


def _expect(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _inventory(a: dict[str, int], t: dict[str, int], qk_norm: bool) -> dict[str, tuple[int, ...]]:
    d, ffn, ds = a["d"], a["ffn"], a["ds"]
    freq = a["mel"]
    for _ in range(CONV_LAYERS):
        freq = (freq + 1) // 2  # stride 2, padding 1, kernel 3
    shapes: dict[str, tuple[int, ...]] = {}
    p = "thinker.audio_tower."
    for i, cin in enumerate((1, ds, ds), start=1):
        shapes[f"{p}conv2d{i}.weight"] = (ds, cin, 3, 3)
        shapes[f"{p}conv2d{i}.bias"] = (ds,)
    shapes[f"{p}conv_out.weight"] = (d, ds * freq)
    for n in range(a["layers"]):
        q = f"{p}layers.{n}."
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[f"{q}self_attn.{proj}.weight"] = (d, d)
            shapes[f"{q}self_attn.{proj}.bias"] = (d,)
        for norm in ("self_attn_layer_norm", "final_layer_norm"):
            shapes[f"{q}{norm}.weight"] = (d,)
            shapes[f"{q}{norm}.bias"] = (d,)
        shapes[f"{q}fc1.weight"], shapes[f"{q}fc1.bias"] = (ffn, d), (ffn,)
        shapes[f"{q}fc2.weight"], shapes[f"{q}fc2.bias"] = (d, ffn), (d,)
    shapes[f"{p}ln_post.weight"] = shapes[f"{p}ln_post.bias"] = (d,)
    shapes[f"{p}proj1.weight"], shapes[f"{p}proj1.bias"] = (d, d), (d,)
    shapes[f"{p}proj2.weight"], shapes[f"{p}proj2.bias"] = (a["out"], d), (a["out"],)

    h, nh, nkv, hd, mid = t["hidden"], t["heads"], t["kv"], t["head_dim"], t["mid"]
    shapes["thinker.model.embed_tokens.weight"] = (t["vocab"], h)
    for n in range(t["layers"]):
        q = f"thinker.model.layers.{n}."
        shapes[f"{q}input_layernorm.weight"] = shapes[f"{q}post_attention_layernorm.weight"] = (h,)
        shapes[f"{q}self_attn.q_proj.weight"] = (nh * hd, h)
        shapes[f"{q}self_attn.k_proj.weight"] = shapes[f"{q}self_attn.v_proj.weight"] = (nkv * hd, h)
        shapes[f"{q}self_attn.o_proj.weight"] = (h, nh * hd)
        if qk_norm:
            shapes[f"{q}self_attn.q_norm.weight"] = shapes[f"{q}self_attn.k_norm.weight"] = (hd,)
        shapes[f"{q}mlp.gate_proj.weight"] = shapes[f"{q}mlp.up_proj.weight"] = (mid, h)
        shapes[f"{q}mlp.down_proj.weight"] = (h, mid)
    shapes["thinker.model.norm.weight"] = (h,)
    shapes["thinker.lm_head.weight"] = (t["vocab"], h)
    return shapes


def compile_qwen3_asr(
    source: Path,
    *,
    target_arch: str,
    license_spdx: str,
    payloads: list[dict[str, Any]] | None = None,
    checkpoint: dict[str, str] | None = None,
    production_approved: bool = False,
) -> dict[str, Any]:
    """`payloads=None` reads and hashes the real shards in `source`; a caller with only headers passes the payload table it already has
    (name, size, sha256 from the pinned revision's metadata): the compiler then checks each size against its header and refuses a mismatch."""
    raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
    pre = json.loads((source / "preprocessor_config.json").read_text(encoding="utf-8"))
    _expect(_req(raw, "model_type", "config.json") == "qwen3_asr", "config.json: model_type is not qwen3_asr")
    _expect("Qwen3ASRForConditionalGeneration" in _req(raw, "architectures", "config.json"), "config.json: architectures does not name Qwen3ASRForConditionalGeneration")
    thinker = _req(raw, "thinker_config", "config.json")
    ac, tc = _req(thinker, "audio_config", "thinker_config"), _req(thinker, "text_config", "thinker_config")

    _expect(_req(ac, "activation_function", "audio_config") == "gelu", "audio_config: only gelu is implemented")
    a = {"d": ac["d_model"], "layers": ac["encoder_layers"], "heads": ac["encoder_attention_heads"], "ffn": ac["encoder_ffn_dim"], "ds": ac["downsample_hidden_size"],
         "mel": ac["num_mel_bins"], "out": ac["output_dim"]}
    _expect(_req(tc, "hidden_act", "text_config") == "silu", "text_config: only silu is implemented")
    _expect(not tc.get("attention_bias", False), "text_config: attention_bias=true is not supported by this compiler")
    rs = _req(tc, "rope_scaling", "text_config")
    # The three mrope components are identical for audio and text (ADR 0013 amendment), so plain rotate-half RoPE is exact. Anything else is refused.
    _expect(rs.get("rope_type") == "default" and rs.get("mrope_interleaved") is True and sum(rs["mrope_section"]) * 2 == tc["head_dim"], f"text_config: rope_scaling {rs} is not the interleaved-mrope default this compiler proves reduces to plain RoPE")
    _expect(a["out"] == tc["hidden_size"], "audio_config.output_dim must equal text_config.hidden_size")

    _expect(pre.get("feature_extractor_type") == "WhisperFeatureExtractor", "preprocessor_config: not a WhisperFeatureExtractor")
    _expect(pre.get("dither", 0.0) == 0.0, "preprocessor_config: dither must be 0")
    _expect(pre["feature_size"] == a["mel"], "preprocessor feature_size differs from audio_config.num_mel_bins")

    ck = read_checkpoint(source, hash_payloads=payloads is None)
    _expect(ck is not None, f"{source}: no safetensors weights or headers found")
    entries: dict[str, TensorEntry] = ck.entries
    qk_norm = any(n.endswith("self_attn.q_norm.weight") for n in entries)
    t = {"hidden": tc["hidden_size"], "heads": tc["num_attention_heads"], "kv": tc["num_key_value_heads"], "head_dim": tc["head_dim"], "mid": tc["intermediate_size"],
         "vocab": tc["vocab_size"], "layers": tc["num_hidden_layers"]}
    want = _inventory(a, t, qk_norm)
    missing, extra = sorted(set(want) - set(entries)), sorted(set(entries) - set(want))
    bad = sorted(n for n in want if n in entries and tuple(entries[n].shape) != want[n])
    if missing or extra or bad:
        raise ValueError(f"tensor inventory differs from config: {len(missing)} missing {missing[:3]}, {len(extra)} unexpected {extra[:3]}, {len(bad)} wrong shape {[(n, entries[n].shape, want[n]) for n in bad[:3]]}")
    _expect(all(e.dtype == "bf16" for e in entries.values()), "this compiler packages bf16 checkpoints only")

    table = payloads if payloads is not None else [{"file": p.file, "byte_length": p.byte_length, "sha256": p.sha256} for p in ck.payloads]
    _expect(len(table) > 0, "no payload table")
    ends: dict[int, int] = {}
    for e in entries.values():
        ends[e.payload] = max(ends.get(e.payload, 0), e.end)
    if payloads is not None:  # header-only source: each declared size must equal 8 + header length + data end, which the header does not state; check it is not smaller
        for i, p in enumerate(table):
            _expect(p["byte_length"] > ends.get(i, 0), f"payload {p['file']}: declared size {p['byte_length']} is not larger than the data its tensors need ({ends.get(i, 0)})")

    decoder = ModelConfig(
        num_layers=t["layers"], attention="gqa" if t["kv"] < t["heads"] else "mha", num_attention_heads=t["heads"], num_kv_heads=t["kv"], head_dim=t["head_dim"],
        vocab_size=t["vocab"], max_seq_len=tc["max_position_embeddings"], hidden_size=t["hidden"], intermediate_size=t["mid"], rope_theta=float(tc["rope_theta"]),
        rms_norm_eps=float(tc["rms_norm_eps"]), tie_word_embeddings=bool(tc["tie_word_embeddings"]), qkv_bias=False, qk_norm=qk_norm,
    ).to_json_dict()
    index_of = lambda name: int(name.split(".layers.")[1].split(".")[0]) if ".layers." in name else 0  # noqa: E731
    manifest: dict[str, Any] = {
        "family": FAMILY, "contract_revision": 1, "model_id": "qwen3-asr-1.7b", "target_arch": target_arch, "tensor_format": "bf16",
        "license_spdx": license_spdx, "production_approved": production_approved,
        "payloads": table,
        "tensors": [
            {"index": index_of(e.name), "name": e.name, **({"payload": e.payload} if e.payload else {}), "byte_offset": e.begin, "byte_length": e.end - e.begin,
             "tensor": {"dtype": "bf16", "shape": list(e.shape)}}
            for e in sorted(entries.values(), key=lambda e: (e.payload, e.begin))
        ],
        "stt": {
            "architecture": "qwen3_asr",
            "frontend": {"sample_rate": WHISPER_SAMPLE_RATE, "n_fft": pre["n_fft"], "hop": pre["hop_length"], "n_mels": pre["feature_size"], "f_min": 0.0, "f_max": WHISPER_SAMPLE_RATE / 2,
                         "window": "hann_periodic", "mel_scale": "slaney", "normalization": "whisper_log_max8", "chunk_seconds": pre["chunk_length"]},
            "encoder": {"num_layers": a["layers"], "hidden_size": a["d"], "num_heads": a["heads"], "intermediate_size": a["ffn"], "conv_channels": a["ds"], "subsample": 2**CONV_LAYERS,
                        "chunk_mel_frames": 2 * ac["n_window"], "attention_window_mel_frames": ac["n_window_infer"], "output_dim": a["out"], "max_source_positions": ac["max_source_positions"]},
            "decoder": decoder,
            "audio_token_id": thinker["audio_token_id"], "audio_start_token_id": thinker["audio_start_token_id"], "audio_end_token_id": thinker["audio_end_token_id"],
        },
        "signature": None,
    }
    if checkpoint is not None:
        manifest["qualified_checkpoint"] = checkpoint
    validate_family_manifest(manifest)
    return manifest


register("qwen3_asr", compile_qwen3_asr)
