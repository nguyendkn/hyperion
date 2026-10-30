"""The GEMM bandwidth script's arithmetic and step detection, on a profile written in Nsight Systems' SQLite schema
(`StringIds`, `CUPTI_ACTIVITY_KIND_KERNEL`). The kernel durations are chosen numbers, so this checks the script's
logic (which kernels are GEMMs, which call site each is, bytes over time), not any real GPU measurement."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gemm-bandwidth.py"
spec = importlib.util.spec_from_file_location("gemm_bandwidth", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

TINY = dict(layers=2, hidden=64, intermediate=128, heads=4, kv_heads=2, head_dim=16, vocab=256)


def write_profile(path: Path, steps: int, fused: bool | str, gemm_us: dict[str, float]) -> None:
    """A profile of `steps` decode steps: embedding, per layer the own kernels between cuBLAS-named GEMMs, argmax."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (shortName INTEGER, demangledName INTEGER, start INTEGER, end INTEGER)")
    ids: dict[str, int] = {}

    def name_id(name: str) -> int:
        if name not in ids:
            ids[name] = len(ids) + 1
            db.execute("INSERT INTO StringIds VALUES (?, ?)", (ids[name], name))
        return ids[name]

    clock = 0

    def kernel(name: str, nanos: int) -> None:
        nonlocal clock
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)", (name_id(name), name_id(name), clock, clock + nanos))
        clock += nanos + 100

    layer_sites = tool.sites(TINY, fused)
    for _ in range(steps):
        kernel("embedding_gather_bf16", 2000)
        for _layer in range(TINY["layers"]):
            for index, (site, _bytes) in enumerate(layer_sites):
                kernel(f"sm90_xmma_gemv_{site}", int(gemm_us[site] * 1000))
                kernel("rope_bf16" if index == 0 else "attention_split_16_bf16", 3000)
        kernel("sm90_xmma_gemv_head", int(gemm_us["lm_head"] * 1000))
        kernel("argmax_chain_bf16", 5000)
    db.commit()
    db.close()


TIMES = {"qkv": 4.0, "o_proj": 2.0, "gate_up": 8.0, "down": 4.0, "lm_head": 6.0,
         "q": 2.0, "k": 1.0, "v": 1.0, "gate": 4.0, "up": 4.0}


@pytest.mark.parametrize("fused", ["fused", "qkv", "separate"])
def test_bandwidth_per_call_site_is_weight_bytes_over_median_time(tmp_path, fused):
    path = tmp_path / "decode.sqlite"
    write_profile(path, steps=8, fused=fused, gemm_us=TIMES)
    layout, rows = tool.analyse(tool.read_kernels(str(path)), TINY, peak_gbps=1000.0)
    assert layout == fused
    by_name = {name: (nbytes, micros, gbps, percent, count) for name, nbytes, micros, gbps, percent, count in rows}
    for site, nbytes in tool.sites(TINY, fused) + [("lm_head", TINY["vocab"] * TINY["hidden"] * 2)]:
        got_bytes, micros, gbps, percent, count = by_name[site]
        assert got_bytes == nbytes
        assert micros == pytest.approx(TIMES[site])
        assert gbps == pytest.approx(nbytes / (TIMES[site] * 1000.0))
        assert percent == pytest.approx(gbps / 10.0)
        assert count == (1 if site == "lm_head" else TINY["layers"])


def test_a_step_with_another_number_of_gemms_is_skipped_and_none_usable_is_an_error(tmp_path):
    path = tmp_path / "decode.sqlite"
    write_profile(path, steps=5, fused=True, gemm_us=TIMES)
    # 7 * 2 + 1 = 15 or 4 * 2 + 1 = 9 GEMMs are the known layouts; claim a model with 3 layers: neither matches.
    with pytest.raises(SystemExit, match="no decode step"):
        tool.analyse(tool.read_kernels(str(path)), {**TINY, "layers": 3}, peak_gbps=1000.0)


def test_the_first_steps_are_skipped_as_warm_up(tmp_path):
    path = tmp_path / "decode.sqlite"
    write_profile(path, steps=3, fused=True, gemm_us=TIMES)
    with pytest.raises(SystemExit):
        tool.analyse(tool.read_kernels(str(path)), TINY, peak_gbps=1000.0, skip=3)
    assert tool.analyse(tool.read_kernels(str(path)), TINY, peak_gbps=1000.0, skip=0)[0] == "fused"


def test_own_kernels_are_never_counted_as_gemms():
    for name in ["embedding_gather_bf16", "add_bf16", "swiglu_bf16", "rope_bf16", "attention_split_128_bf16", "attention_combine_bf16",
                 "attention_chunk_bf16", "kv_append_bf16", "kv_append_chunk_bf16", "bias_cast_bf16", "argmax_chain_bf16", "rmsnorm_bf16", "add_rmsnorm_bf16"]:
        assert tool.OWN_KERNELS.match(name), name
    for name in ["sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x64x64", "gemv2T_kernel_val", "cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8", "void cublasLt::splitKreduce_kernel"]:
        assert not tool.OWN_KERNELS.match(name), name


def test_a_prefill_with_separate_projections_next_to_many_fused_decode_steps_does_not_change_the_layout(tmp_path):
    path = tmp_path / "decode.sqlite"
    write_profile(path, steps=12, fused=True, gemm_us=TIMES)
    db = sqlite3.connect(path)
    # One extra piece with the separate-projection GEMM count (a prefill chunk plus its output head), appended at the end.
    ids = {name: i for i, name in db.execute("SELECT id, value FROM StringIds")}
    at = db.execute("SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] + 100
    rows = [("embedding_gather_bf16", 2000)] + [("sm90_xmma_gemv_q", 9000)] * (7 * TINY["layers"] + 1)
    for name, nanos in rows:
        if name not in ids:
            ids[name] = max(ids.values()) + 1
            db.execute("INSERT INTO StringIds VALUES (?, ?)", (ids[name], name))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)", (ids[name], ids[name], at, at + nanos))
        at += nanos + 100
    db.commit()
    db.close()
    layout, rows = tool.analyse(tool.read_kernels(str(path)), TINY, peak_gbps=1000.0)
    assert layout == "fused"
    assert {name for name, *_ in rows} == {"qkv", "o_proj", "gate_up", "down", "lm_head"}
