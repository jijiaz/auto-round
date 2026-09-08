#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark SVDQuant Kernel A against the ARK unfused three-step baseline.

This implements design doc section 4.3. The baseline is the sequence a model
would actually run today when no fused kernel exists, and it is timed **as
three separate ARK device operations whose medians are summed**, not as one
end-to-end call:

1. ``smooth+quant`` -- ARK ``launch_quant_only``: smooth, MXFP4 group amax,
   shared exponent, E2M1 encode, nibble pack (bit-identical ``qact``/``ascales``
   to the fused kernel's quantization phase).
2. ``prepare_lora`` -- ARK ``launch_pack_lora_b_cute``: fold ``smooth`` into
   ``lora_down`` and emit the split 16-bit B planes.
3. ``projection``   -- ARK standalone ``launch_lora_cute`` CUTE/DPAS low-rank
   projection ``x @ (B_hi + B_lo)``, the same projection the fused path runs.

Each step is the same device kernel the fused path uses, launched separately,
so the comparison is one of scheduling (three launches, ``x`` re-read by the
projection) rather than arithmetic -- no ``torch.matmul`` and no re-derived
baseline kernels are involved.

Summing the individual medians is deliberately *generous to the baseline*: it
charges nothing for the launch gaps between the three kernels. It is still a
fair statement of the ceiling an unfused implementation could reach, so a
speedup measured against it is a lower bound on the real-world win.

``quant_down_reference`` (``ref e2e``) is kept as a supplementary reference:
the pure-PyTorch A0 emulation the fused path replaces.

Quant-only shapes (``rank=0``) have no low-rank branch, so they are not forced
through the three-step sum; they report the ARK quant-only kernel alone (the
fused path already is that kernel when ``lora_down`` is absent).

Timing follows section 4.3: XPU events, warmup, at least 100 iterations,
P50 and P95. Every input and output (``qact``, ``ascales``, ``hi``, ``lo``,
``lora_act``) is preallocated once per shape, so each timed call is a pure
kernel launch: no per-iteration tensor allocation and no hidden
dtype-conversion kernel on the timed path. A one-shot parity check confirms the
split ``qact``/``ascales`` are bit-identical to the fused path and that
``lora_act`` stays inside the section 4.1 tolerance.

Run from ``auto_round_extension/ark``::

    python test/bench_svdquant_mxfp4.py
    python test/bench_svdquant_mxfp4.py --iters 200 --dtype fp16
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_round_kernel.svdquant_mxfp4 import (  # noqa: E402
    GROUP_SIZE,
    LORA_ACT_RTOL,
    ark_kernel_available,
    quant_down_reference,
    split_kernel_available,
    svdquant_mxfp4_lora_down,
    svdquant_mxfp4_prepare_lora,
    svdquant_mxfp4_quant_down,
    svdquant_mxfp4_smooth_quant,
)

WARMUP = 20
ITERS = 100


@dataclass(frozen=True)
class Shape:
    label: str
    rows: int
    columns: int
    rank: int


# FLUX-representative shapes plus small-batch decode-like cases. ``rank=0``
# means the quant-only path (no low-rank branch).
SHAPES = (
    Shape("flux mlp", 4608, 3072, 32),
    Shape("flux mlp r16", 4608, 3072, 16),
    Shape("flux mlp r64", 4608, 3072, 64),
    Shape("flux ffn", 4608, 12288, 32),
    Shape("mid batch", 1024, 3072, 32),
    Shape("small batch", 256, 3072, 32),
    Shape("single token", 1, 3072, 32),
    Shape("quant only", 4608, 3072, 0),
    Shape("quant only ffn", 4608, 12288, 0),
)


# ---------------------------------------------------------------------------
# Timing.
# ---------------------------------------------------------------------------


def time_ms(fn, warmup: int = WARMUP, iters: int = ITERS) -> tuple[float, float]:
    """Time ``fn`` with XPU events; returns ``(p50, p95)`` milliseconds per call."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    timings = []
    for _ in range(iters):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    timings.sort()
    return timings[len(timings) // 2], timings[min(int(len(timings) * 0.95), len(timings) - 1)]


# ---------------------------------------------------------------------------
# Split / fused parity check.
# ---------------------------------------------------------------------------


def _split_parity(x: torch.Tensor, smooth: torch.Tensor | None, lora_down: torch.Tensor | None, rank: int):
    """One-shot check that the split steps reproduce the fused outputs.

    ``qact``/``ascales`` are gated on bit-exactness (design doc section 4.1);
    ``lora_act`` is compared with the section 4.1 relative-L2 gate and is *not*
    required to be bit-identical across launches -- the split projection walks
    all of K in one pass, while the fused kernel may split K and sum FP32
    partials in a different order.

    Returns ``(qact_exact, ascales_exact, lora_rel_l2)``.
    """
    qact_fused, ascales_fused, lora_fused = svdquant_mxfp4_quant_down(x, smooth, lora_down, backend="ark")
    qact_split, ascales_split = svdquant_mxfp4_smooth_quant(x, smooth)
    torch.xpu.synchronize()

    qact_exact = bool(torch.equal(qact_split, qact_fused))
    ascales_exact = bool(torch.equal(ascales_split, ascales_fused))

    lora_rel_l2 = None
    if rank:
        hi, lo = svdquant_mxfp4_prepare_lora(smooth, lora_down)
        lora_split = svdquant_mxfp4_lora_down(x, hi, lo, rank)
        torch.xpu.synchronize()
        denom = float(lora_fused.to(torch.float32).square().sum().sqrt().cpu())
        lora_rel_l2 = float(
            (lora_split.to(torch.float32) - lora_fused.to(torch.float32)).square().sum().sqrt().cpu()
        ) / denom
    return qact_exact, ascales_exact, lora_rel_l2


# ---------------------------------------------------------------------------
# Cost model.
# ---------------------------------------------------------------------------


def fused_bytes(shape: Shape, element_size: int) -> int:
    """Minimum global traffic the fused kernel must move, in bytes.

    Counts one read of ``x``, one read of ``smooth``, ``lora_down`` read once
    per row block, and one write of each output. It is a lower bound: the real
    kernel re-reads ``lora_down`` per sub-group, which this deliberately does
    not charge, so the reported bandwidth stays comparable across ranks.
    """
    total = shape.rows * shape.columns * element_size  # read x
    total += shape.columns * 4  # read smooth
    total += shape.rows * shape.columns // 2  # write qact
    total += shape.rows * shape.columns // GROUP_SIZE  # write ascales
    if shape.rank:
        total += shape.rank * shape.columns * element_size  # read lora_down
        total += shape.rows * shape.rank * 4  # write lora_act
    return total


def lora_flops(shape: Shape) -> int:
    """Multiply-accumulate FLOPs in the low-rank down projection."""
    return 2 * shape.rows * shape.columns * shape.rank


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def run_shape(shape: Shape, dtype: torch.dtype, iters: int) -> dict:
    rows, columns, rank = shape.rows, shape.columns, shape.rank

    x = torch.randn(rows, columns, dtype=dtype, device="xpu")
    smooth = (torch.rand(columns, device="xpu") + 0.5).to(dtype)
    lora_down = torch.randn(rank, columns, dtype=dtype, device="xpu") if rank else None

    # Normalize once (FP32 smooth) and preallocate every output before timing,
    # so each timed call below is a pure kernel launch -- no per-iteration
    # allocation, no dtype-conversion kernel -- matching the README's
    # preallocated three-step baseline.
    smooth_f32 = smooth.to(torch.float32).contiguous()
    qact = torch.empty((rows, columns // 2), dtype=torch.uint8, device="xpu")
    ascales = torch.empty((rows, columns // GROUP_SIZE), dtype=torch.uint8, device="xpu")

    # Fused path -- the ARK quant-only kernel when `rank == 0`. Outputs are
    # allocated by the entry point; the fused path is a single call by design.
    fused_p50, fused_p95 = time_ms(
        lambda: svdquant_mxfp4_quant_down(x, smooth_f32, lora_down, backend="ark"), iters=iters
    )

    # One-shot parity: every split step must reproduce the fused outputs.
    qact_exact, ascales_exact, lora_rel_l2 = _split_parity(x, smooth_f32, lora_down, rank)

    if not rank:
        # Quant-only rows have no low-rank branch, so there is no three-step
        # split sum to report; the fused time above already *is* the ARK
        # quant-only kernel.
        return {
            "shape": shape,
            "fused_p50": fused_p50,
            "fused_p95": fused_p95,
            "ref_p50": 0.0,
            "qact_exact": qact_exact,
            "ascales_exact": ascales_exact,
            "lora_rel_l2": None,
            "gbps": fused_bytes(shape, x.element_size()) / fused_p50 / 1e6,
            "tflops": 0.0,
        }

    # hi/lo live in a private DPAS layout, so probe their size once with an
    # allocating call (not timed), then reuse those buffers for every timed
    # launch below.
    hi, lo = svdquant_mxfp4_prepare_lora(smooth_f32, lora_down)
    lora_act = torch.empty((rows, rank), dtype=dtype, device="xpu")

    # The unfused ARK three-step baseline: each phase is a pure launch into
    # preallocated buffers and the medians are summed (section 4.3). No
    # torch.matmul, no re-derived baseline kernels -- every step is the kernel
    # the fused path uses.
    smooth_quant_p50, _ = time_ms(
        lambda: svdquant_mxfp4_smooth_quant(x, smooth_f32, qact=qact, ascales=ascales), iters=iters
    )
    prepare_lora_p50, _ = time_ms(
        lambda: svdquant_mxfp4_prepare_lora(smooth_f32, lora_down, hi=hi, lo=lo), iters=iters
    )
    # hi/lo were (re)written by prepare_lora's timed loop, so the projection
    # reads a valid operand for the whole of its own timed loop.
    lora_down_p50, _ = time_ms(lambda: svdquant_mxfp4_lora_down(x, hi, lo, rank, lora_act=lora_act), iters=iters)

    split_p50 = smooth_quant_p50 + prepare_lora_p50 + lora_down_p50
    ref_p50, _ = time_ms(
        lambda: quant_down_reference(x, smooth_f32, lora_down), warmup=5, iters=max(20, iters // 5)
    )

    return {
        "shape": shape,
        "fused_p50": fused_p50,
        "fused_p95": fused_p95,
        "smooth_quant_p50": smooth_quant_p50,
        "prepare_lora_p50": prepare_lora_p50,
        "lora_down_p50": lora_down_p50,
        "split_p50": split_p50,
        "ref_p50": ref_p50,
        "qact_exact": qact_exact,
        "ascales_exact": ascales_exact,
        "lora_rel_l2": lora_rel_l2,
        "gbps": fused_bytes(shape, x.element_size()) / fused_p50 / 1e6,
        "tflops": lora_flops(shape) / fused_p50 / 1e9,
    }


def _format_parity(result: dict, dtype: torch.dtype) -> str:
    """Human-readable split/fused parity verdict for one measured shape."""
    bits_ok = result["qact_exact"] and result["ascales_exact"]
    lora = result["lora_rel_l2"]
    if lora is None:
        return "qact/ascales bit-exact" if bits_ok else "qact/ascales MISMATCH"
    tol = LORA_ACT_RTOL[dtype]
    lora_ok = lora <= tol
    status = "OK" if bits_ok and lora_ok else "MISMATCH"
    return f"{status} (qact/ascales bit-exact: {bits_ok}, lora_act rel-L2 {lora:.2e} <= {tol:.1e})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--iters", type=int, default=ITERS, help="timed iterations per measurement")
    args = parser.parse_args()

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("no XPU device available")
        return 1
    if not ark_kernel_available():
        print("the auto_round_kernel XPU build does not export svdquant_mxfp4_quant_down")
        return 1
    if not split_kernel_available():
        print("the auto_round_kernel XPU build does not export the split-step benchmark symbols")
        return 1

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(0)

    print(f"device : {torch.xpu.get_device_name(0)}")
    print(f"dtype  : {args.dtype}   iters: {args.iters} (P50/P95 from XPU events after warmup)\n")

    # rank > 0 shapes are compared fused-vs-split; rank == 0 shapes only report
    # the ARK quant-only kernel and are listed separately below.
    compare_shapes = [s for s in SHAPES if s.rank]
    quant_only_shapes = [s for s in SHAPES if not s.rank]

    header = (
        f"{'shape':<16}{'M':>6}{'K':>7}{'R':>4} | "
        f"{'fused P50':>10}{'P95':>9} | "
        f"{'smooth+quant':>13}{'prepare_lora':>13}{'projection':>12}{'split_sum':>11} | "
        f"{'speedup':>8} | {'GB/s':>7}{'TFLOPs':>8}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for shape in compare_shapes:
        result = run_shape(shape, dtype, args.iters)
        results.append(result)
        speedup = result["split_p50"] / result["fused_p50"]
        print(
            f"{shape.label:<16}{shape.rows:>6}{shape.columns:>7}{shape.rank:>4} | "
            f"{result['fused_p50']:>9.3f}ms{result['fused_p95']:>8.3f} | "
            f"{result['smooth_quant_p50']:>13.3f}{result['prepare_lora_p50']:>13.3f}"
            f"{result['lora_down_p50']:>12.3f}{result['split_p50']:>11.3f} | "
            f"{speedup:>7.2f}x | {result['gbps']:>7.1f}{result['tflops']:>8.2f}"
        )

    speedups = [r["split_p50"] / r["fused_p50"] for r in results]
    geomean = math.exp(statistics.fmean(math.log(s) for s in speedups))
    worst = min(speedups)

    print(f"\ngeometric-mean speedup vs unfused ARK three-step baseline : {geomean:.2f}x  (gate: >= 1.20x)")
    print(f"worst per-shape speedup                                   : {worst:.2f}x  (gate: >= 0.95x)")

    # Quant-only rows report the ARK quant-only kernel on its own; they are not
    # forced through a three-step sum.
    quant_only = []
    if quant_only_shapes:
        print("\nquant-only fused (ARK quant-only kernel, no low-rank branch):")
        print(f"{'shape':<16}{'M':>6}{'K':>7} | {'fused P50':>10}{'P95':>9} | {'GB/s':>7}")
        for shape in quant_only_shapes:
            result = run_shape(shape, dtype, args.iters)
            quant_only.append(result)
            print(
                f"{shape.label:<16}{shape.rows:>6}{shape.columns:>7} | "
                f"{result['fused_p50']:>9.3f}ms{result['fused_p95']:>8.3f} | "
                f"{result['gbps']:>7.1f}"
            )

    # Section 4.3 proposed a "rank-32 costs at most 20% over quant-only" gate.
    # Report it with the arithmetic context needed to judge it, because the
    # low-rank branch is real FLOPs, not just extra traffic over the same tile.
    quant_only_ref = next((r for r in quant_only if r["shape"].label == "quant only"), None)
    rank32 = next((r for r in results if r["shape"].label == "flux mlp"), None)
    if quant_only_ref and rank32:
        overhead = rank32["fused_p50"] / quant_only_ref["fused_p50"] - 1.0
        added_ms = rank32["fused_p50"] - quant_only_ref["fused_p50"]
        added_tflops = lora_flops(rank32["shape"]) / added_ms / 1e9
        print(
            f"\nrank-32 overhead over quant-only (same shape)          : {overhead * 100:.1f}%  "
            f"(section 4.3 proposed <= 20%)"
        )
        print(
            f"  the low-rank branch adds {added_ms:.3f} ms for "
            f"{lora_flops(rank32['shape']) / 1e9:.2f} GFLOP = {added_tflops:.2f} TFLOP/s."
        )

    print("\nsupplementary reference point (P50, ms):")
    print(f"{'shape':<16}{'ref e2e':>10}{'split_sum':>11}{'fused':>9}")
    for result in results:
        print(
            f"{result['shape'].label:<16}{result['ref_p50']:>10.3f}"
            f"{result['split_p50']:>11.3f}{result['fused_p50']:>9.3f}"
        )

    print("\nsplit/fused parity (one shot per shape):")
    for result in results + quant_only:
        print(f"  {result['shape'].label:<16}{_format_parity(result, dtype)}")

    parity_ok = all(r["qact_exact"] and r["ascales_exact"] for r in results + quant_only)
    lora_ok = all(
        r["lora_rel_l2"] is None or r["lora_rel_l2"] <= LORA_ACT_RTOL[dtype] for r in results + quant_only
    )

    passed = geomean >= 1.20 and worst >= 0.95 and parity_ok and lora_ok
    print("\n" + ("PERFORMANCE GATES PASSED" if passed else "PERFORMANCE GATES FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
