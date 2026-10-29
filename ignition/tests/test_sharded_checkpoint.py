"""Checkpoints split over several safetensors shards.

The fixture `testdata/synthetic/qwen2-tiny-sharded` is a tiny random Qwen2 saved by Hugging Face's
own `save_pretrained(max_shard_size=...)`, so it has the real shard layout and a real
`model.safetensors.index.json` (`scripts/gen-sharded-fixture.py` writes it). The manifest must name
each shard with its size and sha256, and every tensor must point at the shard the index says.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import pytest

from hyperion_ignition.compiler.registry import build_compiler
from hyperion_ignition.compiler.weights import is_bare_file_name, map_qwen2_layers, read_checkpoint
from hyperion_ignition.compiler.model_config import model_config_from_hf
from hyperion_ignition.hardware.registry import build_recipe
from hyperion_ignition.manifest_schema import validate_manifest

FIXTURE = Path(__file__).resolve().parents[2] / "testdata" / "synthetic" / "qwen2-tiny-sharded"


def _compile(source: Path) -> dict:
    return build_compiler("qwen2").compile(source=source, recipe=build_recipe("sm_90")).to_json_dict()


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "checkpoint"
    shutil.copytree(FIXTURE, target)
    return target


def _hf_name(layer_name: str) -> str:
    # The manifest drops the `model.` prefix; `lm_head.weight` never had one.
    return layer_name if layer_name.startswith("lm_head") else "model." + layer_name


def _data_start(path: Path) -> int:
    with path.open("rb") as handle:
        return 8 + struct.unpack("<Q", handle.read(8))[0]


def test_a_real_sharded_checkpoint_compiles_to_a_payload_table_with_sizes_and_hashes():
    manifest = _compile(FIXTURE)
    shards = sorted(FIXTURE.glob("model-*-of-*.safetensors"))
    assert len(shards) >= 3
    assert [p["file"] for p in manifest["payloads"]] == [s.name for s in shards]
    for payload, shard in zip(manifest["payloads"], shards):
        assert payload["byte_length"] == shard.stat().st_size
        assert payload["sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()


def test_every_tensor_points_at_the_shard_the_index_names():
    manifest = _compile(FIXTURE)
    weight_map = json.loads((FIXTURE / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    assert len(manifest["layers"]) == len(weight_map)
    for layer in manifest["layers"]:
        shard = manifest["payloads"][layer.get("payload", 0)]["file"]
        assert shard == weight_map[_hf_name(layer["name"])], layer["name"]
    assert len({layer.get("payload", 0) for layer in manifest["layers"]}) == len(manifest["payloads"]), "every shard holds a tensor"


def test_the_bytes_at_the_manifest_offsets_are_the_tensors_hugging_face_wrote():
    manifest = _compile(FIXTURE)
    expected = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    for layer in manifest["layers"]:
        want = expected[_hf_name(layer["name"])]
        path = FIXTURE / manifest["payloads"][layer.get("payload", 0)]["file"]
        assert path.name == want["shard"]
        assert layer["byte_length"] == want["length"]
        with path.open("rb") as handle:
            handle.seek(_data_start(path) + layer["byte_offset"])
            data = handle.read(layer["byte_length"])
        assert data[:16].hex() == want["first16"] and data[-16:].hex() == want["last16"], layer["name"]


def test_the_manifest_satisfies_the_committed_schema_and_equals_the_committed_golden():
    manifest = _compile(FIXTURE)
    validate_manifest(manifest)
    golden = json.loads((FIXTURE / "model.manifest.json").read_text(encoding="utf-8"))
    assert manifest == golden, "run hyperion-compile on the fixture to refresh model.manifest.json"


def test_an_untied_head_is_mapped_from_lm_head():
    manifest = _compile(FIXTURE)
    assert "lm_head.weight" in [layer["name"] for layer in manifest["layers"]]


def test_a_single_file_checkpoint_records_one_payload_and_no_per_layer_index(tmp_path: Path):
    source = _copy(tmp_path)
    shards = sorted(source.glob("model-*-of-*.safetensors"))
    tensors, order = {}, []
    for shard in shards:
        raw = shard.read_bytes()
        header = json.loads(raw[8 : 8 + struct.unpack("<Q", raw[:8])[0]])
        header.pop("__metadata__", None)
        data = raw[8 + struct.unpack("<Q", raw[:8])[0] :]
        for name, spec in header.items():
            begin, end = spec["data_offsets"]
            tensors[name] = (spec["dtype"], spec["shape"], data[begin:end])
            order.append(name)
        shard.unlink()
    (source / "model.safetensors.index.json").unlink()
    header, blob = {}, b""
    for name in order:
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    encoded = json.dumps(header).encode()
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)

    manifest = _compile(source)
    assert [p["file"] for p in manifest["payloads"]] == ["model.safetensors"]
    assert manifest["payloads"][0]["sha256"] == hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()
    assert all("payload" not in layer for layer in manifest["layers"])
    validate_manifest(manifest)


def test_a_shard_index_next_to_a_single_file_is_refused(tmp_path: Path):
    source = _copy(tmp_path)
    (source / "model.safetensors").write_bytes(b"\0" * 16)
    with pytest.raises(ValueError, match="both"):
        _compile(source)


def test_a_missing_shard_is_refused(tmp_path: Path):
    source = _copy(tmp_path)
    sorted(source.glob("model-*-of-*.safetensors"))[1].unlink()
    with pytest.raises(ValueError, match="missing"):
        _compile(source)


def test_an_index_that_disagrees_with_a_shard_header_is_refused(tmp_path: Path):
    source = _copy(tmp_path)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    name, shard = next(iter(index["weight_map"].items()))
    other = next(s for s in sorted(set(index["weight_map"].values())) if s != shard)
    index["weight_map"][name] = other
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        _compile(source)


@pytest.mark.parametrize("bad", ["../evil.safetensors", "sub/dir.safetensors", "C:evil.safetensors", "..", ""])
def test_a_shard_name_that_is_a_path_is_refused(tmp_path: Path, bad: str):
    source = _copy(tmp_path)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    first = next(iter(index["weight_map"]))
    index["weight_map"][first] = bad
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="bare file name|must map"):
        _compile(source)


def test_the_bare_file_name_rule_matches_the_rust_spec():
    for good in ["model.safetensors", "model-00001-of-00004.safetensors", "weights.v2.bin"]:
        assert is_bare_file_name(good), good
    for bad in ["", ".", "..", "../x", "a/b", "a\\b", "/abs", "C:evil", "stream:name", "x\0y", "a" * 256]:
        assert not is_bare_file_name(bad), repr(bad)


def test_header_only_shards_map_tensors_to_shards_without_hashing(tmp_path: Path):
    source = _copy(tmp_path)
    for shard in sorted(source.glob("model-*-of-*.safetensors")):
        raw = shard.read_bytes()
        header_end = 8 + struct.unpack("<Q", raw[:8])[0]
        (source / (shard.name + ".header")).write_bytes(raw[:header_end])
        shard.unlink()
    with pytest.raises(ValueError, match="missing"):
        read_checkpoint(source)  # hashing needs the real files
    checkpoint = read_checkpoint(source, hash_payloads=False)
    assert checkpoint is not None and checkpoint.payloads == []
    weight_map = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    shards = sorted(set(weight_map.values()))
    for name, entry in checkpoint.entries.items():
        assert shards[entry.payload] == weight_map[name]
    config = model_config_from_hf(source, ("Qwen2ForCausalLM",))
    layers = map_qwen2_layers(checkpoint.entries, config, tie_word_embeddings=False)
    assert {layer.payload for layer in layers} == set(range(len(shards)))
