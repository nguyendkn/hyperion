#!/usr/bin/env python3
"""Achieved memory bandwidth of every matrix multiply of a decode step, from an Nsight Systems profile.

A batch-1 decode step reads every weight once, so a matrix-vector product is memory bound and its speed is
`weight bytes / kernel time`. This script prints that per call site (QKV, attention output, gate/up, down, output
head) as GB/s and as a percentage of the device's peak, taking the median over the steps of the profile.

    nsys profile --cuda-graph-trace=node -o decode  <run decoding>      # one profile, many decode steps
    nsys export --type sqlite --force-overwrite true decode.nsys-rep     # -> decode.sqlite
    python scripts/gemm-bandwidth.py decode.sqlite --model 7b            # or --dims layers=28,hidden=3584,...

Which kernels are GEMMs: everything in the step that is not one of Hyperion's own kernels (the names below), so the
cuBLAS and cuBLASLt kernels whatever their generated names. A step starts at the embedding kernel. The number of
GEMMs per step tells the layout: `7 * layers + 1` is the separate Q, K, V, gate, up projections, `4 * layers + 1`
the fused ones (QKV and gate/up concatenated). Steps with another count (a prefill, a partial step) are skipped.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import statistics
import sys

OWN_KERNELS = re.compile(
    r"^(embedding_gather|add|swiglu|rope|attention(_\w+)?|kv_append(_\w+)?|bias_cast|argmax_chain|rmsnorm|add_rmsnorm)_?(\w*bf16)?"
)
H100_PEAK_GBPS = 3350.0  # HBM3 of the SXM part; override with --peak-gbps

MODELS = {
    "7b": dict(layers=28, hidden=3584, intermediate=18944, heads=28, kv_heads=4, head_dim=128, vocab=152064),
    "0.5b": dict(layers=24, hidden=896, intermediate=4864, heads=14, kv_heads=2, head_dim=64, vocab=151936),
}


def sites(d: dict, fused: bool) -> list[tuple[str, int]]:
    """The GEMMs of one layer in launch order as (name, weight bytes), bf16 weights."""
    h, i = d["hidden"], d["intermediate"]
    q, kv = d["heads"] * d["head_dim"], d["kv_heads"] * d["head_dim"]
    bytes_of = lambda out, inn: out * inn * 2  # noqa: E731
    if fused:
        return [("qkv", bytes_of(q + 2 * kv, h)), ("o_proj", bytes_of(h, q)), ("gate_up", bytes_of(2 * i, h)), ("down", bytes_of(h, i))]
    return [("q", bytes_of(q, h)), ("k", bytes_of(kv, h)), ("v", bytes_of(kv, h)), ("o_proj", bytes_of(h, q)),
            ("gate", bytes_of(i, h)), ("up", bytes_of(i, h)), ("down", bytes_of(h, i))]


def read_kernels(path: str) -> list[tuple[str, int]]:
    """(kernel name, duration in ns) in launch order from an `nsys export --type sqlite` file."""
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    rows = db.execute("SELECT shortName, demangledName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    return [(names.get(short) or names.get(demangled) or "?", end - start) for short, demangled, start, end in rows]


def steps(kernels: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
    """Cut the kernel list at each embedding kernel: one decode step (or prefill) per piece."""
    pieces: list[list[tuple[str, int]]] = []
    for name, duration in kernels:
        if name.startswith("embedding_gather"):
            pieces.append([])
        if pieces:
            pieces[-1].append((name, duration))
    return pieces


def analyse(kernels: list[tuple[str, int]], dims: dict, peak_gbps: float, skip: int = 3) -> tuple[str, list[tuple[str, int, float, float, float, int]]]:
    layers = dims["layers"]
    per_layout = {7 * layers + 1: False, 4 * layers + 1: True}
    seen: dict[int, int] = {}
    usable: dict[bool, list[list[int]]] = {False: [], True: []}
    for step in steps(kernels)[skip:]:
        gemms = [duration for name, duration in step if not OWN_KERNELS.match(name)]
        seen[len(gemms)] = seen.get(len(gemms), 0) + 1
        if len(gemms) in per_layout:
            usable[per_layout[len(gemms)]].append(gemms)
    if not any(usable.values()):
        raise SystemExit(
            f"no decode step with {sorted(per_layout)} GEMMs found (after skipping {skip} steps); GEMM kernels per step seen: {seen or 'no steps'}. "
            "A library that splits a product over two kernels (a GEMM and a split-K reduction) changes the count: list the kernel names with "
            "`nsys stats --report cuda_gpu_kern_sum` and extend OWN_KERNELS or the layout table."
        )
    # A profile can hold a prefill (separate projections) next to many decode steps (fused ones): the layout with the most steps is the decode.
    fused = len(usable[True]) >= len(usable[False])
    layout = "fused" if fused else "separate"
    layer_sites = sites(dims, fused)
    by_site: dict[str, list[float]] = {}
    order: list[str] = []
    for gemms in usable[fused]:
        for index, duration in enumerate(gemms[:-1]):
            name, _ = layer_sites[index % len(layer_sites)]
            by_site.setdefault(name, []).append(duration)
            if name not in order:
                order.append(name)
        by_site.setdefault("lm_head", []).append(gemms[-1])
        if "lm_head" not in order:
            order.append("lm_head")
    all_sites = {name: nbytes for name, nbytes in layer_sites}
    all_sites["lm_head"] = dims["vocab"] * dims["hidden"] * 2
    result = []
    for name in order:
        median_ns = statistics.median(by_site[name])
        gbps = all_sites[name] / median_ns  # bytes per ns is GB/s
        count = layers if name != "lm_head" else 1
        result.append((name, all_sites[name], median_ns / 1000.0, gbps, 100.0 * gbps / peak_gbps, count))
    return layout, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sqlite", help="`nsys export --type sqlite` output")
    parser.add_argument("--model", choices=sorted(MODELS), help="dimensions of a known model")
    parser.add_argument("--dims", help="comma-separated key=value: layers, hidden, intermediate, heads, kv_heads, head_dim, vocab")
    parser.add_argument("--peak-gbps", type=float, default=H100_PEAK_GBPS)
    args = parser.parse_args(argv)
    if args.model:
        dims = dict(MODELS[args.model])
    elif args.dims:
        dims = {k: int(v) for k, v in (pair.split("=") for pair in args.dims.split(","))}
    else:
        parser.error("give --model or --dims")
    layout, rows = analyse(read_kernels(args.sqlite), dims, args.peak_gbps)
    print(f"layout: {layout} projections; peak {args.peak_gbps:.0f} GB/s; median over the profiled decode steps")
    print(f"{'site':<10}{'weight MiB':>12}{'time us':>10}{'GB/s':>9}{'% peak':>8}{'per step':>10}")
    total_bytes = total_us = 0.0
    for name, nbytes, micros, gbps, percent, count in rows:
        print(f"{name:<10}{nbytes / 2**20:>12.1f}{micros:>10.1f}{gbps:>9.0f}{percent:>8.1f}{count:>10}")
        total_bytes += nbytes * count
        total_us += micros * count
    print(f"{'all GEMMs':<10}{total_bytes / 2**20:>12.1f}{total_us:>10.1f}{total_bytes / total_us / 1000.0:>9.0f}{100.0 * total_bytes / total_us / 1000.0 / args.peak_gbps:>8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
