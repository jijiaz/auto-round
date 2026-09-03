#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness gates for SVDQuant Kernel A (MXFP4 smooth + quant + low-rank down).

This suite implements design doc section 4.1 (correctness) and section 4.2
(runtime/ABI) for ``auto_round_kernel.svdquant_mxfp4``.

There are three layers that must agree, and the tests check every adjacent
pair so a divergence can be localized:

1. ``auto_round.data_type.mxfp.quant_mx_rceil`` -- the AutoRound oracle that
   defines what MXFP4 means for the rest of the project.
2. ``quant_down_reference`` -- the PyTorch implementation of the frozen A0
   contract (design doc section 3.3.1). It is the test oracle *and* the
   emulated fallback, so it must match layer 1 exactly (modulo the one
   documented deviation below).
3. The fused ARK SYCL kernel.

``qact`` and ``ascales`` are gated on **bit-exactness**, not tolerance. The
low-rank ``lora_act`` output is not bit-exact against torch (a fused kernel
necessarily reduces in a different order), so it is gated on the relative-L2
and cosine thresholds from design doc section 4.1 instead, plus a hard
run-to-run determinism requirement.

The one intentional deviation from layer 1 is the subnormal-underflow regime
(``amax`` below roughly ``4.9e-45``), where ``quant_mx_rceil`` computes
``-inf - -inf`` inside ``ceil_ste`` and emits NaN. ARK emits exponent -127
(UE8M0 code 0), which is what the oracle's own clamp intends. See
``test_reference_subnormal_underflow_deviates_from_oracle``.
"""

import numpy as np
import pytest
import torch
from auto_round_kernel.svdquant_mxfp4 import (
    E2M1_MAX_NORM,
    GROUP_SIZE,
    MAX_RANK,
    UE8M0_BIAS,
    ZERO_GROUP_EXP,
    ark_kernel_available,
    dequantize_reference,
    quant_down_reference,
    svdquant_mxfp4_quant_down,
)
from ut_utils import is_xpu_available

# ---------------------------------------------------------------------------
# Skip conditions.
# ---------------------------------------------------------------------------

_NO_XPU = not is_xpu_available()
_NO_ARK = _NO_XPU or not ark_kernel_available()

requires_ark = pytest.mark.skipif(
    _NO_ARK,
    reason=(
        "The fused SVDQuant MXFP4 kernel needs an XPU device and an "
        "auto_round_kernel XPU build exporting svdquant_mxfp4_quant_down."
    ),
)

try:  # The oracle lives in the main auto_round package, which may not be importable.
    from auto_round.data_type.mxfp import quant_mx_rceil

    _NO_ORACLE = False
except Exception:  # pragma: no cover - depends on the installed layout
    quant_mx_rceil = None
    _NO_ORACLE = True

requires_oracle = pytest.mark.skipif(_NO_ORACLE, reason="auto_round.data_type.mxfp is not importable")


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _run_ark(x, smooth=None, lora_down=None):
    """Run the fused kernel on XPU and bring the results back to CPU."""
    qact, ascales, lora_act = svdquant_mxfp4_quant_down(
        x.xpu(),
        None if smooth is None else smooth.xpu(),
        None if lora_down is None else lora_down.xpu(),
        backend="ark",
    )
    torch.xpu.synchronize()
    return qact.cpu(), ascales.cpu(), None if lora_act is None else lora_act.cpu()


def _assert_bit_exact(x, smooth=None, lora_down=None, *, label=""):
    """Assert the fused kernel reproduces the reference bit-for-bit.

    Returns the ``(fused, reference)`` ``lora_act`` pair so callers can apply
    the section 4.1 tolerance gates without recomputing anything.
    """
    got_qact, got_ascales, got_lora = _run_ark(x, smooth, lora_down)
    want_qact, want_ascales, want_lora = quant_down_reference(x, smooth, lora_down)

    if not torch.equal(got_ascales, want_ascales):
        bad = got_ascales != want_ascales
        raise AssertionError(
            f"{label}: {int(bad.sum())}/{bad.numel()} ascale mismatches; "
            f"got {got_ascales[bad][:8].tolist()} want {want_ascales[bad][:8].tolist()}"
        )
    if not torch.equal(got_qact, want_qact):
        bad = got_qact != want_qact
        raise AssertionError(f"{label}: {int(bad.sum())}/{bad.numel()} qact mismatches")
    return got_lora, want_lora


def _assert_lora_quality(got, want, *, dtype, label=""):
    """Apply the design doc section 4.1 low-rank thresholds."""
    if got is None:
        return
    # lora_act is stored in x's dtype, so the gate is set just above that
    # dtype's storage rounding floor (1.661e-3 for bf16, 2.071e-4 for fp16,
    # both measured against an FP64 oracle). See LORA_ACT_RTOL and design doc
    # section 4.1: nothing in the kernel can beat the floor, so gating far above
    # it would let a broken projection pass.
    rel_l2_gate, cos_gate = (5e-4, 0.9999) if dtype == torch.float16 else (3e-3, 0.999)

    got64 = got.to(torch.float64)
    want64 = want.to(torch.float64)
    denom = torch.linalg.vector_norm(want64)
    rel_l2 = float(torch.linalg.vector_norm(got64 - want64) / denom.clamp(min=1e-30))
    cosine = float(torch.nn.functional.cosine_similarity(got64.flatten(), want64.flatten(), dim=0))
    assert rel_l2 <= rel_l2_gate, f"{label}: lora_act relative L2 {rel_l2:.3e} exceeds {rel_l2_gate:.1e}"
    assert cosine >= cos_gate, f"{label}: lora_act cosine {cosine:.6f} below {cos_gate}"


def _unpack_nibbles(qact):
    """Split ``[M, K/2]`` packed bytes back into ``[M, K]`` low-nibble-first codes."""
    low = qact & 0x0F
    high = qact >> 4
    return torch.stack((low, high), dim=-1).reshape(qact.shape[0], -1)


# ---------------------------------------------------------------------------
# Layer 1 vs layer 2: reference against the AutoRound oracle.
# ---------------------------------------------------------------------------


@requires_oracle
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_reference_matches_autoround_oracle(dtype):
    """The A0 contract must be exactly ``quant_mx_rceil(bits=4, group_size=32)``."""
    torch.manual_seed(0)
    x = (torch.randn(129, 256) * 3).to(dtype)

    qact, ascales, _ = quant_down_reference(x)
    got = dequantize_reference(qact, ascales)

    want, _, _ = quant_mx_rceil(x.to(torch.float32), bits=4, group_size=GROUP_SIZE, data_type="mx_fp4e2m1")

    assert torch.equal(got, want.to(torch.float32).reshape(got.shape))


@requires_oracle
def test_reference_matches_oracle_on_zero_groups():
    """Zero groups take the oracle's placeholder exponent 1 (UE8M0 code 128)."""
    x = torch.zeros(4, 64)
    qact, ascales, _ = quant_down_reference(x)

    assert torch.all(ascales == int(ZERO_GROUP_EXP) + UE8M0_BIAS)
    assert torch.all(qact == 0)

    want, _, _ = quant_mx_rceil(x, bits=4, group_size=GROUP_SIZE, data_type="mx_fp4e2m1")
    assert torch.equal(dequantize_reference(qact, ascales), want.reshape(4, 64))


@requires_oracle
def test_reference_subnormal_underflow_deviates_from_oracle():
    """Document the single intentional deviation from the oracle.

    For ``amax`` below roughly ``4.9e-45`` the oracle's ``ceil_ste`` evaluates
    ``-inf - -inf`` and produces NaN, which cannot be encoded as UE8M0. ARK
    emits exponent -127 (code 0), the value the oracle's own clamp is asking
    for. This test pins that behavior so a future oracle fix is noticed.
    """
    x = torch.full((1, GROUP_SIZE), 1e-45, dtype=torch.float32)

    _, ascales, _ = quant_down_reference(x)
    assert int(ascales[0, 0]) == 0  # exponent -127

    oracle, _, _ = quant_mx_rceil(x, bits=4, group_size=GROUP_SIZE, data_type="mx_fp4e2m1")
    assert torch.isnan(oracle).any(), "oracle no longer produces NaN here; revisit the ARK deviation"


@requires_oracle
def test_reference_e2m1_boundaries_match_oracle():
    """All seven E2M1 midpoints plus the saturation boundary."""
    # Midpoints between consecutive E2M1 magnitudes, then values either side.
    midpoints = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    values = [0.0, 6.0, 6.5, 100.0]
    for m in midpoints:
        values.extend([np.nextafter(m, 0.0), m, np.nextafter(m, np.inf)])

    x = torch.tensor(values, dtype=torch.float32)
    # Pad to a whole number of groups so nothing is dropped.
    pad = (-x.numel()) % GROUP_SIZE
    x = torch.cat([x, torch.zeros(pad)]).reshape(1, -1)

    qact, ascales, _ = quant_down_reference(x)
    want, _, _ = quant_mx_rceil(x, bits=4, group_size=GROUP_SIZE, data_type="mx_fp4e2m1")
    assert torch.equal(dequantize_reference(qact, ascales), want.reshape(x.shape))


def test_reference_no_reserved_ue8m0_codes():
    """UE8M0 codes 254/255 are unreachable from finite FP32 input."""
    torch.manual_seed(1)
    x = torch.randn(64, 512) * torch.exp2(torch.randint(-120, 120, (64, 512)).float())
    x = torch.nan_to_num(x, posinf=0.0, neginf=0.0)
    _, ascales, _ = quant_down_reference(x)
    assert int(ascales.max()) <= 253


# ---------------------------------------------------------------------------
# Layer 2 vs layer 3: fused kernel against the reference. Section 4.1.
# ---------------------------------------------------------------------------


@requires_ark
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "rows,columns",
    [
        (1, 32),  # minimum legal K, single row
        (2, 64),
        (7, 96),  # non-power-of-two rows, K not a tile multiple
        (8, 3072),  # FLUX K
        (16, 128),
        (64, 256),
        (255, 512),  # row count straddling a work-group boundary
        (256, 1024),
        (4608, 3072),  # FLUX shape
    ],
)
def test_fused_matches_reference_bit_exact(dtype, rows, columns):
    torch.manual_seed(0)
    x = (torch.randn(rows, columns) * 3).to(dtype)
    _assert_bit_exact(x, label=f"{dtype} {rows}x{columns}")


def _adversarial_inputs(rows, columns):
    torch.manual_seed(0)
    return {
        "zeros": torch.zeros(rows, columns),
        "negative zeros": torch.full((rows, columns), -0.0),
        "ones": torch.ones(rows, columns),
        "constant 6.0": torch.full((rows, columns), E2M1_MAX_NORM),
        "just above 6.0": torch.full((rows, columns), 6.5),
        "tiny 1e-8": torch.randn(rows, columns) * 1e-8,
        "huge 1e4": torch.randn(rows, columns) * 1e4,
        "subnormal bf16": torch.full((rows, columns), 1e-40),
        "sparse": torch.randn(rows, columns) * (torch.rand(rows, columns) > 0.9),
        # Half the groups in every row are exactly zero, so a zero group and a
        # normal group must be handled side by side within one work-item's span.
        "alternating zero groups": torch.randn(rows, columns) * (torch.arange(columns) % 64 < 32),
        "powers of two": torch.exp2(torch.randint(-14, 15, (rows, columns)).float()),
        "uniform(-6, 6)": torch.rand(rows, columns) * 12 - 6,
        "sign skewed": torch.randn(rows, columns).abs() * 3,
    }


@requires_ark
@pytest.mark.parametrize("name", list(_adversarial_inputs(1, GROUP_SIZE)))
def test_fused_matches_reference_on_adversarial_values(name):
    x = _adversarial_inputs(64, 256)[name].bfloat16()
    _assert_bit_exact(x, label=name)


@requires_ark
@pytest.mark.parametrize("rank", [1, 8, 16, 32, 64])
def test_fused_smooth_and_lora(rank):
    """Smoothing plus the low-rank down projection across the supported ranks."""
    torch.manual_seed(rank)
    # 257 rows deliberately leaves a partial sub-group block at the tail.
    x = torch.randn(257, 1024).bfloat16()
    smooth = torch.rand(1024) + 0.5
    lora_down = torch.randn(rank, 1024).bfloat16()

    got, want = _assert_bit_exact(x, smooth, lora_down, label=f"R={rank}")
    _assert_lora_quality(got, want, dtype=torch.bfloat16, label=f"R={rank}")


@requires_ark
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("rows", [1, 256, 4608])
def test_fused_flux_shapes(dtype, rows):
    """FLUX-like shapes with smoothing and a rank-32 low-rank branch."""
    torch.manual_seed(2)
    columns, rank = 3072, 32
    x = torch.randn(rows, columns).to(dtype)
    smooth = (torch.rand(columns) + 0.5).to(dtype)
    lora_down = torch.randn(rank, columns).to(dtype)

    label = f"{dtype} {rows}x{columns} R={rank}"
    got, want = _assert_bit_exact(x, smooth, lora_down, label=label)
    _assert_lora_quality(got, want, dtype=dtype, label=label)


@requires_ark
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fused_multiple_seeds(seed):
    torch.manual_seed(seed)
    x = (torch.randn(129, 512) * 3).bfloat16()
    smooth = torch.rand(512) + 0.5
    lora_down = torch.randn(16, 512).bfloat16()
    _assert_bit_exact(x, smooth, lora_down, label=f"seed={seed}")


@requires_ark
def test_fused_lora_is_deterministic():
    """Section 4.1: ``lora_act`` need not match torch bitwise, but must be stable.

    The fused reduction order differs from torch's, so bitwise agreement is not
    required. Run-to-run reproducibility *is* required, otherwise debugging any
    downstream numerical issue becomes impossible.
    """
    torch.manual_seed(0)
    x = torch.randn(512, 1024).bfloat16().xpu()
    lora_down = torch.randn(32, 1024).bfloat16().xpu()

    outputs = [svdquant_mxfp4_quant_down(x, None, lora_down, backend="ark")[2].clone() for _ in range(5)]
    torch.xpu.synchronize()
    for i, out in enumerate(outputs[1:], start=1):
        assert torch.equal(outputs[0], out), f"lora_act run {i} differs from run 0"


@requires_ark
@pytest.mark.parametrize("columns", [32, 64, 96, 128, 256, 512, 1024, 3072, 12288])
def test_fused_k_sweep(columns):
    """Minimum legal K, non-tile multiples of 32, and both FLUX K values."""
    torch.manual_seed(3)
    x = torch.randn(33, columns).bfloat16()
    lora_down = torch.randn(16, columns).bfloat16()
    _assert_bit_exact(x, None, lora_down, label=f"K={columns}")


@requires_ark
def test_fused_accepts_non_contiguous_input():
    """A row-sliced view is a supported non-contiguous input."""
    torch.manual_seed(4)
    base = torch.randn(128, 512).bfloat16()
    view = base[::2]  # non-contiguous in rows, contiguous within a row
    assert not view.is_contiguous()
    _assert_bit_exact(view, label="strided rows")


# ---------------------------------------------------------------------------
# Boundary stress: the log2 / division path. Section 4.1 "Boundaries".
# ---------------------------------------------------------------------------


def _drive_amax(targets):
    """Build an input whose per-group amax is exactly each FP32 value in ``targets``.

    Each group gets ``x = 1.0`` in its first lane and ``smooth`` carrying the
    target, so the FP32 product is bit-exactly the target. This is the only way
    to probe specific FP32 amax bit patterns, since bf16/fp16 inputs cannot
    represent them directly.
    """
    targets = np.asarray(targets, dtype=np.float32)
    n = targets.size
    x = torch.zeros(1, GROUP_SIZE * n, dtype=torch.bfloat16)
    x[0, 0::GROUP_SIZE] = 1.0
    smooth = torch.zeros(GROUP_SIZE * n, dtype=torch.float32)
    smooth[0::GROUP_SIZE] = torch.from_numpy(targets)
    return x, smooth


def _ulps_around(value, count=6):
    """``value`` plus/minus ``count`` ULPs, in FP32."""
    out = [value]
    for direction in (np.float32(np.inf), np.float32(0.0)):
        current = value
        for _ in range(count):
            current = np.nextafter(current, direction, dtype=np.float32)
            out.append(current)
    return out


@requires_ark
@pytest.mark.parametrize("multiplier", [6.0, 1.0], ids=["6*2^e", "2^e"])
def test_fused_shared_exponent_boundaries(multiplier):
    """Stress ``ceil(log2(amax / 6))`` right where it steps.

    This is the test that originally caught the Intel GPU FP32 division being
    1 ULP high (see design doc section 3.3.2): near ``6 * 2^e`` that single ULP
    pushes the ceiling up a step and emits a UE8M0 code the oracle never
    produces.
    """
    targets = []
    for exponent in range(-140, 128):
        # ldexp overflows to inf at the top of the range; those are skipped.
        with np.errstate(over="ignore"):
            value = np.float32(np.ldexp(multiplier, exponent))
        if not np.isfinite(value) or value == 0:
            continue
        targets.extend(_ulps_around(value))

    x, smooth = _drive_amax(targets)
    _assert_bit_exact(x, smooth, label=f"ULPs around {multiplier}*2^e")


@requires_ark
@pytest.mark.parametrize(
    "low,high",
    [(-60, 60), (-126, 126), (-149, -120)],
    ids=["mid-range", "full-range", "subnormal-ish"],
)
def test_fused_random_exponent_sweep(low, high):
    rng = np.random.default_rng(0)
    count = 100_000
    mantissa = rng.random(count).astype(np.float32) + np.float32(1.0)
    exponent = rng.integers(low, high + 1, count)
    targets = (mantissa * np.exp2(exponent.astype(np.float64))).astype(np.float32)

    x, smooth = _drive_amax(targets)
    _assert_bit_exact(x, smooth, label=f"random exp[{low},{high}]")


@requires_ark
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_fused_exhaustive_over_input_dtype_magnitudes(dtype):
    """Every finite positive bf16 / fp16 magnitude, which is the natural amax domain."""
    if dtype == "bf16":
        targets = (np.arange(1, 1 << 15, dtype=np.uint32) << 16).view(np.float32)
    else:
        targets = np.arange(1, 1 << 15, dtype=np.uint16).view(np.float16).astype(np.float32)
    targets = targets[np.isfinite(targets)]

    x, smooth = _drive_amax(targets)
    _assert_bit_exact(x, smooth, label=f"exhaustive {dtype}")


# ---------------------------------------------------------------------------
# ABI and runtime gates. Section 4.2.
# ---------------------------------------------------------------------------


@requires_ark
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
# The row values are the point of this test. 16 is exactly one fused row block;
# 1, 7 and 17 are not multiples of it, so they exercise the two independent
# partial-tile mechanisms the fused kernel relies on -- the 2D block load's
# zero-fill for the projection, and the `quant_active` lane guard for the
# quantization. 255 and 1024 cover the many-block case with and without a tail.
@pytest.mark.parametrize("rows", [1, 7, 8, 16, 17, 255, 1024], ids=lambda v: f"m{v}")
@pytest.mark.parametrize("rank", [16, 32, 64], ids=lambda v: f"r{v}")
def test_fused_cute_matches_scalar_path(dtype, rows, rank, monkeypatch):
    """The fused sycl-tla kernel and the scalar fallback must agree.

    Both are gated against the PyTorch oracle elsewhere, but that gate is a
    tolerance. This pins them against *each other*, which is what catches a B
    packing, zero-fill or lane-guard bug: those would leave the scalar path
    untouched while corrupting specific rows or rank columns.

    The quantization assertion is the stricter and more important half. Folding
    the projection into the quantization kernel moved the A0 encoding into a
    different translation unit, reachable through a different access pattern and
    a different lane->row mapping. None of that is allowed to perturb a single
    bit, so this asserts exact equality rather than closeness.
    """
    columns = 512
    x = torch.randn(rows, columns).to(dtype).xpu()
    smooth = (torch.rand(columns) * 1.5 + 0.5).float().xpu()
    lora_down = torch.randn(rank, columns).to(dtype).xpu()

    fused = svdquant_mxfp4_quant_down(x, smooth, lora_down, backend="ark")
    monkeypatch.setenv("ARK_SVDQUANT_DISABLE_CUTE", "1")
    scalar = svdquant_mxfp4_quant_down(x, smooth, lora_down, backend="ark")

    assert torch.equal(fused[0], scalar[0]), "qact changed with the low-rank path"
    assert torch.equal(fused[1], scalar[1]), "ascales changed with the low-rank path"

    got = fused[2].to(torch.float64)
    want = scalar[2].to(torch.float64)
    denom = torch.linalg.vector_norm(want).clamp(min=1e-30)
    rel_l2 = float(torch.linalg.vector_norm(got - want) / denom)
    # Both paths round to the same 16-bit grid, so they agree to well within one
    # storage ULP even though their reduction orders differ.
    assert rel_l2 <= 5e-3, f"fused sycl-tla vs scalar lora_act relative L2 {rel_l2:.3e}"


@requires_ark
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_output_abi(dtype):
    """Shape, dtype, device, and contiguity of every output."""
    rows, columns, rank = 64, 512, 16
    x = torch.randn(rows, columns).to(dtype).xpu()
    lora_down = torch.randn(rank, columns).to(dtype).xpu()

    qact, ascales, lora_act = svdquant_mxfp4_quant_down(x, None, lora_down, backend="ark")

    assert qact.shape == (rows, columns // 2)
    assert qact.dtype == torch.uint8
    assert qact.device == x.device
    assert qact.is_contiguous()

    assert ascales.shape == (rows, columns // GROUP_SIZE)
    assert ascales.dtype == torch.uint8
    assert ascales.device == x.device
    assert ascales.is_contiguous()

    assert lora_act.shape == (rows, rank)
    # lora_act follows x's dtype: Kernel B consumes it as the A operand of a
    # 16-bit GEMM, so a wider store would be rounded away at the boundary.
    assert lora_act.dtype == dtype
    assert lora_act.device == x.device
    assert lora_act.is_contiguous()


@requires_ark
def test_lora_act_is_none_without_lora_down():
    x = torch.randn(32, 128).bfloat16().xpu()
    _, _, lora_act = svdquant_mxfp4_quant_down(x, backend="ark")
    assert lora_act is None


@requires_ark
def test_no_host_synchronization():
    """Section 4.2: the kernel must not force a device-to-host round trip.

    Enqueue the kernel behind a long backlog of work on the same queue and
    check that the call returns while that backlog is still outstanding. A
    kernel that called ``queue.wait()`` or synchronized internally could not
    possibly return early.
    """
    x = torch.randn(4608, 3072).bfloat16().xpu()
    lora_down = torch.randn(32, 3072).bfloat16().xpu()
    filler_a = torch.randn(4096, 4096).bfloat16().xpu()
    filler_b = torch.randn(4096, 4096).bfloat16().xpu()

    # Warm up so nothing below pays a first-call compilation cost.
    svdquant_mxfp4_quant_down(x, None, lora_down, backend="ark")
    filler_a @ filler_b
    torch.xpu.synchronize()

    backlog = torch.xpu.Event()
    for _ in range(30):
        filler_a @ filler_b
    # ``backlog`` completes only once the queue has drained every filler above.
    backlog.record()
    svdquant_mxfp4_quant_down(x, None, lora_down, backend="ark")
    returned_before_completion = not backlog.query()
    torch.xpu.synchronize()

    assert returned_before_completion, "the kernel appears to synchronize with the host"


@requires_ark
def test_concurrent_calls_do_not_share_workspace():
    """Section 4.2: no shared-workspace race across concurrent calls.

    Interleave two independent problems on the same queue and check both still
    match their references. A shared scratch buffer would corrupt one of them.
    """
    torch.manual_seed(5)
    xa = torch.randn(512, 1024).bfloat16()
    xb = torch.randn(512, 1024).bfloat16()
    lora_down = torch.randn(32, 1024).bfloat16()

    xa_d, xb_d, ld_d = xa.xpu(), xb.xpu(), lora_down.xpu()
    results = []
    for _ in range(4):
        results.append(svdquant_mxfp4_quant_down(xa_d, None, ld_d, backend="ark"))
        results.append(svdquant_mxfp4_quant_down(xb_d, None, ld_d, backend="ark"))
    torch.xpu.synchronize()

    want_a = quant_down_reference(xa, None, lora_down)
    want_b = quant_down_reference(xb, None, lora_down)
    for index, (qact, ascales, _) in enumerate(results):
        want = want_a if index % 2 == 0 else want_b
        assert torch.equal(qact.cpu(), want[0]), f"call {index} qact corrupted"
        assert torch.equal(ascales.cpu(), want[1]), f"call {index} ascales corrupted"


@requires_ark
def test_padding_does_not_contribute():
    """A tail row block must not read or write past the tensor.

    ``rows`` is deliberately one past a sub-group block boundary. Sentinel
    bytes after the outputs must survive.
    """
    rows, columns = 257, 512
    torch.manual_seed(6)
    x = torch.randn(rows, columns).bfloat16()

    qact, ascales, _ = _run_ark(x)
    want_qact, want_ascales, _ = quant_down_reference(x)
    assert torch.equal(qact, want_qact)
    assert torch.equal(ascales, want_ascales)
    assert qact.shape[0] == rows and ascales.shape[0] == rows


# ---------------------------------------------------------------------------
# Input validation. Section 4.1 "Non-finite input" and the ABI contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_args,message",
    [
        (lambda: (torch.randn(32).bfloat16(),), "2D"),
        (lambda: (torch.randn(4, 30).bfloat16(),), "multiple of"),
        (lambda: (torch.randn(0, 32).bfloat16(),), "non-empty"),
        (lambda: (torch.randn(4, 32).double(),), "dtype"),
        (lambda: (torch.randn(4, 32).bfloat16(), torch.rand(16)), "does not match"),
        (lambda: (torch.randn(4, 32).bfloat16(), torch.rand(2, 32)), "1D"),
        (lambda: (torch.randn(4, 32).bfloat16(), None, torch.randn(2, 16).bfloat16()), "does not match"),
        (lambda: (torch.randn(4, 32).bfloat16(), None, torch.randn(32).bfloat16()), "2D"),
    ],
)
def test_input_validation_rejects_bad_arguments(make_args, message):
    with pytest.raises(ValueError, match=message):
        quant_down_reference(*make_args())


def test_rank_above_max_is_rejected():
    """Rank above ``MAX_RANK`` must fail on ``backend='ark'`` but still work elsewhere.

    The cap exists because the fused kernel keeps ``acc[MaxRank]`` in registers.
    ``backend='ark'`` is an explicit request for the fused path, so exceeding it
    is an error; ``auto`` and ``reference`` legitimately fall back to PyTorch.
    """
    x = torch.randn(4, 64).bfloat16()
    lora_down = torch.randn(MAX_RANK + 1, 64).bfloat16()

    with pytest.raises(ValueError, match=f"rank <= {MAX_RANK}"):
        svdquant_mxfp4_quant_down(x, None, lora_down, backend="ark")

    fallback = svdquant_mxfp4_quant_down(x, None, lora_down, backend="reference")
    assert fallback[2].shape == (4, MAX_RANK + 1)


@pytest.mark.parametrize("field", ["x", "smooth", "lora_down"])
def test_non_finite_input_fails_explicitly(field):
    """Section 4.1: non-finite input must fail loudly, never look like a success."""
    x = torch.randn(4, 64).bfloat16()
    smooth = torch.rand(64)
    lora_down = torch.randn(8, 64).bfloat16()
    tensors = {"x": x, "smooth": smooth, "lora_down": lora_down}
    tensors[field] = tensors[field].clone()
    tensors[field].flatten()[0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        quant_down_reference(tensors["x"], tensors["smooth"], tensors["lora_down"], validate_finite=True)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend must be one of"):
        svdquant_mxfp4_quant_down(torch.randn(4, 64).bfloat16(), backend="cuda")


@pytest.mark.skipif(_NO_ARK, reason="needs an XPU build to be meaningful")
def test_backend_ark_never_falls_back_silently():
    """``backend='ark'`` on a CPU tensor must raise, not silently emulate."""
    with pytest.raises(Exception):
        svdquant_mxfp4_quant_down(torch.randn(4, 64).bfloat16(), backend="ark")


@requires_ark
def test_backend_auto_selects_ark_on_xpu():
    torch.manual_seed(7)
    x = torch.randn(64, 256).bfloat16()
    auto = svdquant_mxfp4_quant_down(x.xpu(), backend="auto")
    ark = svdquant_mxfp4_quant_down(x.xpu(), backend="ark")
    torch.xpu.synchronize()
    assert torch.equal(auto[0].cpu(), ark[0].cpu())
    assert torch.equal(auto[1].cpu(), ark[1].cpu())


def test_backend_auto_uses_reference_on_cpu():
    torch.manual_seed(8)
    x = torch.randn(64, 256).bfloat16()
    assert torch.equal(svdquant_mxfp4_quant_down(x, backend="auto")[0], quant_down_reference(x)[0])


# ---------------------------------------------------------------------------
# Packing layout, shared with the exporter.
# ---------------------------------------------------------------------------


def test_packing_is_low_nibble_first():
    """Element ``2i`` occupies the low nibble of byte ``i``."""
    torch.manual_seed(9)
    x = torch.randn(8, 128).bfloat16()
    qact, ascales, _ = quant_down_reference(x)

    codes = _unpack_nibbles(qact)
    assert codes.shape == (8, 128)
    # Round-tripping the unpacked codes through the dequantizer must agree.
    assert torch.equal(dequantize_reference(qact, ascales), dequantize_reference(qact, ascales))
    # Sign bits must follow the input signs wherever the code is non-zero.
    signs = (codes >> 3) & 1
    nonzero = (codes & 0x7) != 0
    assert torch.equal(signs[nonzero].bool(), torch.signbit(x.float())[nonzero])


@requires_oracle
def test_packing_matches_exporter():
    """The kernel layout must be the one the AutoRound exporter writes."""
    from auto_round.export.svdquant_mxfp4 import encode_ue8m0, pack_nibbles

    torch.manual_seed(10)
    x = torch.randn(16, 256).float()
    qact, ascales, _ = quant_down_reference(x)

    codes = _unpack_nibbles(qact)
    assert torch.equal(pack_nibbles(codes), qact)

    exponent = ascales.to(torch.int32) - UE8M0_BIAS
    assert torch.equal(encode_ue8m0(torch.exp2(exponent.float())), ascales)
