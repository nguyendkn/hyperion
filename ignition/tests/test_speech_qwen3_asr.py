"""Qwen3-ASR compiler against the real config and the real safetensors headers of the pinned revision (testdata/models-speech.json).
Skipped with a message when the files are absent (`python scripts/fetch-speech-fixtures.py`); HYPERION_REQUIRE_MODEL_FIXTURES=1 makes absence a failure."""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from hyperion_ignition.compiler.speech import compile_speech

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / "testdata" / "models-speech.json").read_text(encoding="utf-8"))["models"]
MODEL = "qwen3-asr-1.7b"
SOURCE = ROOT / "testdata" / "models" / MODEL


def _have() -> bool:
    return all((SOURCE / name).is_file() for name in REGISTRY[MODEL]["files"])


@pytest.fixture(scope="module")
def source() -> Path:
    if not _have():
        if os.environ.get("HYPERION_REQUIRE_MODEL_FIXTURES") == "1":
            pytest.fail(f"{MODEL} fixtures missing; run python scripts/fetch-speech-fixtures.py")
        pytest.skip(f"{MODEL} fixtures missing; run python scripts/fetch-speech-fixtures.py")
    return SOURCE


def _payloads() -> list[dict]:
    # real size and sha256 from the pinned revision's LFS metadata (the registry), not invented
    return [{"file": n, "byte_length": w["bytes"], "sha256": w["sha256"]} for n, w in REGISTRY[MODEL]["weights"].items()]


def _compile(src: Path, **kw):
    return compile_speech("qwen3_asr", src, target_arch="sm_90", license_spdx="Apache-2.0", payloads=_payloads(), **kw)


def test_compiles_the_real_checkpoint_and_validates_against_the_schema(source):
    m = _compile(source)
    assert m["family"] == "stt_qwen3_asr" and m["production_approved"] is False
    assert len(m["tensors"]) == 708  # every tensor of the index mapped, none left over
    stt = m["stt"]
    assert (stt["encoder"]["num_layers"], stt["encoder"]["hidden_size"], stt["encoder"]["chunk_mel_frames"], stt["encoder"]["attention_window_mel_frames"]) == (24, 1024, 100, 800)
    d = stt["decoder"]
    assert (d["num_layers"], d["hidden_size"], d["num_attention_heads"], d["num_kv_heads"], d["head_dim"], d["qk_norm"], d["tie_word_embeddings"], d["rope_theta"]) == (28, 2048, 16, 8, 128, True, True, 1e6)
    assert (stt["audio_token_id"], stt["frontend"]["n_mels"], stt["frontend"]["n_fft"], stt["frontend"]["hop"]) == (151676, 128, 400, 160)
    assert {t["payload"] if "payload" in t else 0 for t in m["tensors"]} == {0, 1}


def test_tensor_offsets_stay_inside_their_payload(source):
    m = _compile(source)
    for t in m["tensors"]:
        size = m["payloads"][t.get("payload", 0)]["byte_length"]
        assert t["byte_offset"] + t["byte_length"] < size, t["name"]


def _copy(source: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "src"
    shutil.copytree(source, dst)
    return dst


def test_refuses_a_changed_shape_and_a_missing_tensor(source, tmp_path):
    dst = _copy(source, tmp_path)
    cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    cfg["thinker_config"]["audio_config"]["encoder_ffn_dim"] = 2048  # the headers still say 4096
    (dst / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong shape"):
        _compile(dst)
    cfg["thinker_config"]["audio_config"]["encoder_ffn_dim"] = 4096
    cfg["thinker_config"]["audio_config"]["encoder_layers"] = 25  # a layer the checkpoint does not have
    (dst / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        _compile(dst)


def test_refuses_what_it_cannot_prove(source, tmp_path):
    dst = _copy(source, tmp_path)
    base = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    for path, value, message in [
        (("thinker_config", "text_config", "rope_scaling", "rope_type"), "yarn", "rope_scaling"),
        (("thinker_config", "text_config", "hidden_act"), "gelu", "silu"),
        (("thinker_config", "audio_config", "activation_function"), "relu", "gelu"),
        (("model_type",), "other", "model_type"),
    ]:
        cfg = copy.deepcopy(base)
        node = cfg
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        (dst / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            _compile(dst)
    (dst / "config.json").write_text(json.dumps(base), encoding="utf-8")
    pre = json.loads((dst / "preprocessor_config.json").read_text(encoding="utf-8"))
    pre["dither"] = 1e-5
    (dst / "preprocessor_config.json").write_text(json.dumps(pre), encoding="utf-8")
    with pytest.raises(ValueError, match="dither"):
        _compile(dst)


def test_a_too_small_declared_payload_is_refused(source):
    small = _payloads()
    small[0]["byte_length"] = 1000
    with pytest.raises(ValueError, match="not larger"):
        compile_speech("qwen3_asr", source, target_arch="sm_90", license_spdx="Apache-2.0", payloads=small)


GOLDEN = ROOT / "specification" / "fixtures" / "family" / "qwen3-asr-1.7b-sm_90.json"


def test_golden_manifest_is_what_the_compiler_writes(source):
    """The Rust side parses the same file (`specification/tests/family.rs`). `UPDATE_FIXTURES=1` rewrites it."""
    text = json.dumps(_compile(source, checkpoint={"repo": REGISTRY[MODEL]["repo"], "revision": REGISTRY[MODEL]["revision"]}), indent=1, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_FIXTURES") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(text, encoding="utf-8", newline="\n")
    assert GOLDEN.read_text(encoding="utf-8").replace("\r\n", "\n") == text
