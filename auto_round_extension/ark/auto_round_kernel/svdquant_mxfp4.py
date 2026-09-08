# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ARK SVDQuant MXFP4 Kernel A: smooth + dynamic MXFP4 activation quant + low-rank down.

This module owns the frozen A0 numerical contract described in section 3.3.1 of
``ark_SVDQuant_mxfp4_design_doc.md``. It provides:

* :func:`quant_down_reference` -- a pure-PyTorch implementation of A0, device
  agnostic, used as the test oracle and as the emulated fallback path.
* :func:`svdquant_mxfp4_quant_down` -- the public entry point, which dispatches
  to the fused ARK SYCL kernel on XPU and otherwise raises or falls back
  explicitly (never silently).

A0 in one line::

    xh = x * smooth
    ascales, qact = MXFP4_RCEIL_QUANT(xh)     # group_size 32, E2M1 + UE8M0
    lora_act = xh @ lora_down.T               # FP32 accumulation, 16-bit store

The zero-group exponent is 1 (UE8M0 code 128), not 0. This intentionally matches
``auto_round.data_type.mxfp.quant_mx_rceil`` and the existing
``auto_round.export.svdquant_mxfp4`` exporter rather than a "cleaner" scale of
1.0. See design doc section 3.3.1 for the rationale.
"""

from __future__ import annotations

import torch

GROUP_SIZE = 32
"""MXFP4 micro-scaling group size. Fixed by the A0 contract."""

LORA_ACT_RTOL = {torch.bfloat16: 3e-3, torch.float16: 5e-4, torch.float32: 1e-5}
"""Relative L2 gate for ``lora_act``, per storage dtype (design doc section 4.1).

These are set just above the *storage rounding floor*, measured against an FP64
oracle at ``4608x3072 R=32``: 1.661e-3 for BF16 and 2.071e-4 for FP16. Nothing
implemented in the kernel can beat those floors, because they are the cost of
writing the result out at all. Gating anywhere near 1e-2 (the value used while
``lora_act`` was FP32) would let a genuinely broken projection pass.

The split-BF16 DPAS path lands exactly on both floors, so it is the accuracy
reference here, not a compromise against it.
"""

E2M1_MAX_NORM = 6.0
"""Largest magnitude representable by E2M1."""

UE8M0_BIAS = 127
"""Exponent bias for the UE8M0 activation scale encoding."""

SCALE_EXP_MIN = -127
SCALE_EXP_MAX = 127
"""Shared-exponent clamp range. Keeps UE8M0 codes inside ``[0, 254]`` so the
reserved code 255 is never emitted."""

ZERO_GROUP_EXP = 1.0
"""Shared exponent used when a group's amax is exactly zero (A0 step 3)."""

MAX_RANK = 64
"""Largest low-rank width the fused kernel accepts. Must match ``kMaxRank`` in
``wrapper/include/sycl_svdquant_mxfp4.hpp``. SVDQuant itself uses 16 or 32."""

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
"""E2M1 magnitude codebook indexed by the 3-bit magnitude code."""

E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
"""Midpoints between adjacent codebook entries. Exact hits are ties and resolve
to the even code index (round-half-to-even on the code index).

Documentation only: :func:`e2m1_magnitude_code` deliberately reproduces the
oracle's FP32 arithmetic instead of using these thresholds, because the two
disagree within one ULP of a midpoint. See that function's docstring."""

_SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_SMOOTH_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_REFERENCE_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _floor_log2_clamped(magnitude: torch.Tensor) -> torch.Tensor:
    """``max(floor(log2(a)), 0)`` computed from FP32 exponent bits.

    For a normal FP32 value, ``floor(log2(a))`` is exactly the unbiased
    exponent, so extracting it from the bit pattern is both cheaper and more
    reliable than calling ``log2``, whose GPU implementation is only accurate to
    a few ULP and can therefore misclassify exact powers of two. Zero and
    subnormal inputs have a zero exponent field, yielding ``-127``, which the
    clamp maps to ``0`` -- the same answer the oracle produces via its
    ``clip(min=min_exp)`` with ``min_exp == 0`` for E2M1.
    """

    exponent_field = (magnitude.view(torch.int32) >> 23) & 0xFF
    return (exponent_field - 127).clamp_(min=0)


def e2m1_magnitude_code(magnitude: torch.Tensor) -> torch.Tensor:
    """Map non-negative FP32 magnitudes in ``[0, 6]`` to 3-bit E2M1 codes.

    Implements A0 step 5 by reproducing the exact FP32 operation sequence of
    ``auto_round.data_type.mxfp.quant_element(ebits=2, mbits=3, max_norm=6.0)``
    with the default ``"even"`` mantissa rounding::

        pe = max(floor(log2(a)), 0)          # in {0, 1, 2} for a in [0, 6]
        s  = a * 2**(1 - pe)
        q  = floor(s + 0.5) - ((s - 0.5) % 2 == 0)
        code = q + 2 * pe

    The final ``code = q + 2 * pe`` identity holds because the oracle's
    magnitude is ``q * 2**(pe - 1)``, and enumerating the reachable
    ``(pe, q)`` pairs maps that one-to-one onto the codebook index.

    Reproducing the arithmetic matters rather than thresholding on
    :data:`E2M1_MIDPOINTS`: ``floor(s + 0.5)`` rounds in FP32, so a value one ULP
    *below* a midpoint can still round up. For example ``a = 0.24999998509``
    (one ULP below the 0.25 tie) makes ``s + 0.5`` an exact FP32 tie that
    round-half-to-even resolves to ``1.0``, so the oracle returns code 1 where a
    plain midpoint comparison would return code 0. These cases are rare but
    reachable with real data, so the kernel reproduces them to keep the
    section 4.1 bit-exactness gate free of carve-outs.
    """

    private_exponent = _floor_log2_clamped(magnitude)
    scaled = torch.ldexp(magnitude, 1 - private_exponent)
    tie = torch.remainder(scaled - 0.5, 2.0) == 0
    quotient = torch.floor(scaled + 0.5) - tie.to(scaled.dtype)
    return (quotient.to(torch.int32) + 2 * private_exponent).to(torch.uint8)


def _shared_exponent(amax: torch.Tensor) -> torch.Tensor:
    """A0 step 3: rceil shared exponent with the zero-group and degenerate cases.

    Three regimes, in priority order:

    1. ``amax == 0`` -> exponent 1 (UE8M0 code 128). Deliberately matches
       ``quant_mx_rceil``; see the module docstring and design doc 3.3.1.
    2. ``amax / 6`` underflows to zero (``amax <= ~4.9e-45``, i.e. the whole
       group is subnormal) -> ``log2`` is ``-inf`` and the clamp yields the
       minimum exponent ``-127``. The oracle instead returns **NaN** here,
       because its ``ceil_ste(x) = (x.ceil() - x).detach() + x`` evaluates
       ``-inf - -inf``. That is a straight-through-estimator artifact leaking
       into the forward value, not a contract, so ARK takes the value the clamp
       was written to produce. Every code in such a group is 0 either way.
    3. Otherwise ``ceil(log2(amax / 6))`` clamped to ``[-127, 127]``.

    The ``ceil(log2(.))`` must be evaluated as *round the real logarithm to FP32,
    then ceil*. Computing ``floor(log2)`` from exponent bits and adding one for
    non-powers-of-two is **not** equivalent: for ``v`` just above ``2**e`` the
    true logarithm ``e + 1.7e-7`` rounds back to exactly ``e`` in FP32 whenever
    ``ulp(e)`` exceeds that offset, which is true for ``|e| >= 4``. This is
    simply ``torch.log2`` on the FP32 ratio, which is what the oracle computes.
    """

    ratio = amax.to(torch.float32) / E2M1_MAX_NORM
    logarithm = torch.log2(ratio)
    exponent = torch.where(amax == 0, torch.full_like(logarithm, ZERO_GROUP_EXP), torch.ceil(logarithm))
    # ``-inf`` from the underflow regime clamps to SCALE_EXP_MIN, which is the
    # intended behavior described above.
    return exponent.clamp_(min=SCALE_EXP_MIN, max=SCALE_EXP_MAX)


def _validate_inputs(
    x: torch.Tensor,
    smooth: torch.Tensor | None,
    lora_down: torch.Tensor | None,
    *,
    allowed_dtypes: tuple[torch.dtype, ...] = _SUPPORTED_INPUT_DTYPES,
) -> None:
    """Enforce the section 4.2 ABI contract before any compute happens."""

    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("x must be a 2D torch.Tensor of shape [M, K]")
    if x.dtype not in allowed_dtypes:
        raise ValueError(f"x dtype must be one of {allowed_dtypes}, got {x.dtype}")
    rows, columns = x.shape
    if rows <= 0 or columns <= 0:
        raise ValueError("x dimensions must be non-empty")
    if columns % GROUP_SIZE:
        raise ValueError(f"x K dimension must be a multiple of {GROUP_SIZE}, got {columns}")

    if smooth is not None:
        if not isinstance(smooth, torch.Tensor) or smooth.ndim != 1:
            raise ValueError("smooth must be a 1D torch.Tensor of shape [K]")
        if smooth.shape[0] != columns:
            raise ValueError(f"smooth length {smooth.shape[0]} does not match x K dimension {columns}")
        if smooth.dtype not in _SUPPORTED_SMOOTH_DTYPES:
            raise ValueError(f"smooth dtype must be one of {_SUPPORTED_SMOOTH_DTYPES}, got {smooth.dtype}")
        if smooth.device != x.device:
            raise ValueError(f"smooth must be on device {x.device}, got {smooth.device}")

    if lora_down is not None:
        if not isinstance(lora_down, torch.Tensor) or lora_down.ndim != 2:
            raise ValueError("lora_down must be a 2D torch.Tensor of shape [R, K]")
        if lora_down.shape[1] != columns:
            raise ValueError(f"lora_down K dimension {lora_down.shape[1]} does not match x K dimension {columns}")
        if lora_down.shape[0] <= 0:
            raise ValueError("lora_down rank must be positive")
        if lora_down.dtype not in allowed_dtypes:
            raise ValueError(f"lora_down dtype must be one of {allowed_dtypes}, got {lora_down.dtype}")
        if lora_down.device != x.device:
            raise ValueError(f"lora_down must be on device {x.device}, got {lora_down.device}")


def quant_down_reference(
    x: torch.Tensor,
    smooth: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    *,
    validate_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Pure-PyTorch implementation of the frozen A0 contract.

    Args:
        x: ``[M, K]`` activations, FP16 or BF16. FP32 is additionally accepted
            here because the reference doubles as the test oracle and golden
            vectors are easiest to express in FP32; the fused kernel ABI stays
            FP16/BF16 only.
        smooth: optional ``[K]`` smoothing vector. ``None`` means multiply by 1.
        lora_down: optional ``[R, K]`` low-rank down factor.
        validate_finite: when True, raise on non-finite inputs. This is the
            explicit debug path required by design doc section 4.1; it is off by
            default because it forces a device-to-host synchronization.

    Returns:
        ``(qact, ascales, lora_act)`` where ``qact`` is ``[M, K/2]`` uint8
        low-nibble-first packed E2M1 codes, ``ascales`` is ``[M, K/32]`` uint8
        UE8M0 codes, and ``lora_act`` is ``[M, R]`` in ``x``'s dtype, or
        ``None`` when ``lora_down`` is ``None``. The projection accumulates in
        FP32 and only the store narrows; see the note on :data:`LORA_ACT_RTOL`.
    """

    _validate_inputs(x, smooth, lora_down, allowed_dtypes=_REFERENCE_INPUT_DTYPES)

    if validate_finite:
        if not bool(torch.isfinite(x).all()):
            raise ValueError("x must contain only finite values")
        if smooth is not None and not bool(torch.isfinite(smooth).all()):
            raise ValueError("smooth must contain only finite values")
        if lora_down is not None and not bool(torch.isfinite(lora_down).all()):
            raise ValueError("lora_down must contain only finite values")

    rows, columns = x.shape
    num_groups = columns // GROUP_SIZE

    # Step 1: smooth in FP32.
    smoothed = x.to(torch.float32)
    if smooth is not None:
        smoothed = smoothed * smooth.to(torch.float32)

    grouped = smoothed.reshape(rows, num_groups, GROUP_SIZE)

    # Steps 2 and 3: per-group amax and rceil shared exponent.
    amax = grouped.abs().amax(dim=-1)
    exponent = _shared_exponent(amax)
    scale = torch.exp2(exponent)

    # Step 4: normalize and clamp into the E2M1 representable range.
    normalized = (grouped / scale.unsqueeze(-1)).clamp_(min=-E2M1_MAX_NORM, max=E2M1_MAX_NORM)

    # Steps 5 and 6: magnitude code plus sign bit.
    codes = e2m1_magnitude_code(normalized.abs())
    codes |= torch.signbit(normalized).to(torch.uint8) << 3

    # Step 7: low-nibble-first packing.
    pairs = codes.reshape(rows, columns // 2, 2)
    qact = (pairs[..., 0] | (pairs[..., 1] << 4)).contiguous()

    ascales = (exponent.to(torch.int32) + UE8M0_BIAS).to(torch.uint8).contiguous()

    # Step 8: low-rank down projection against the smoothed, unquantized input.
    lora_act = None
    if lora_down is not None:
        lora_act = smoothed @ lora_down.to(torch.float32).t()
        # Accumulate in FP32, store in x's dtype. Kernel B consumes this as the
        # A operand of a 16-bit DPAS GEMM, so a wider store would be rounded
        # away at the kernel boundary and only cost bandwidth.
        lora_act = lora_act.to(x.dtype).contiguous()

    return qact, ascales, lora_act


_ARK_SYMBOL = "svdquant_mxfp4_quant_down"

_BACKENDS = ("auto", "ark", "reference")


def ark_kernel_available() -> bool:
    """True when the fused ARK SYCL kernel is importable on this build."""

    try:
        from .xpu_loader import ensure_xpu_lib

        ensure_xpu_lib(required_symbols=(_ARK_SYMBOL,))
    except Exception:
        return False
    return True


def _quant_down_ark(
    x: torch.Tensor,
    smooth: torch.Tensor | None,
    lora_down: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Fused ARK SYCL path. Assumes :func:`_validate_inputs` already ran."""

    from . import cvt_dtype, get_stream
    from .xpu_loader import ensure_xpu_lib

    library = ensure_xpu_lib(required_symbols=(_ARK_SYMBOL,))

    # The kernel ABI is row-major contiguous throughout, FP32 smooth, and a
    # low-rank factor whose dtype matches x. Normalizing here keeps the C++
    # side free of layout and dtype fallbacks.
    x = x.contiguous()
    rows, columns = x.shape
    smooth = None if smooth is None else smooth.to(torch.float32).contiguous()
    lora_down = None if lora_down is None else lora_down.to(x.dtype).contiguous()
    rank = 0 if lora_down is None else lora_down.shape[0]

    qact = torch.empty((rows, columns // 2), dtype=torch.uint8, device=x.device)
    ascales = torch.empty((rows, columns // GROUP_SIZE), dtype=torch.uint8, device=x.device)
    lora_act = None if lora_down is None else torch.empty((rows, rank), dtype=x.dtype, device=x.device)

    # Scratch for the DPAS low-rank path. The kernel owns the packed layout, so
    # the size comes from the kernel; 0 means "this shape has no DPAS path" and
    # the kernel falls back to the scalar fused loop on a null pointer.
    workspace_elements = library.svdquant_mxfp4_workspace_elements(m=rows, k=columns, r=rank)
    workspace = None if workspace_elements == 0 else torch.empty(workspace_elements, dtype=x.dtype, device=x.device)

    library.svdquant_mxfp4_quant_down(
        stream=get_stream(x),
        x=x.data_ptr(),
        smooth=0 if smooth is None else smooth.data_ptr(),
        lora_down=0 if lora_down is None else lora_down.data_ptr(),
        qact=qact.data_ptr(),
        ascales=ascales.data_ptr(),
        lora_act=0 if lora_act is None else lora_act.data_ptr(),
        workspace=0 if workspace is None else workspace.data_ptr(),
        m=rows,
        k=columns,
        r=rank,
        x_dtype=cvt_dtype(x.dtype),
        lora_dtype=cvt_dtype(x.dtype if lora_down is None else lora_down.dtype),
    )
    return qact, ascales, lora_act


# ---------------------------------------------------------------------------
# Benchmark split-step wrappers.
#
# These expose each phase of the fused kernel as its own standalone device
# launch so ``bench_svdquant_mxfp4.py`` can time the unfused three-step baseline
# (design doc section 4.3) against the fused path:
#
#   1. smooth + quant   -> launch_quant_only            (bit-identical qact/ascales)
#   2. prepare_lora     -> launch_pack_lora_b_cute      (hi + lo split B planes)
#   3. lora_down        -> launch_lora_cute             (CUTE/DPAS projection)
#
# They are benchmark interfaces only -- each forwards to the same device kernel
# the fused path uses and adds no new device logic. Outputs and any workspace
# are allocated here; callers pass torch tensors, never raw pointers.
# ---------------------------------------------------------------------------

_SPLIT_SYMBOLS = (
    "svdquant_mxfp4_smooth_quant",
    "svdquant_mxfp4_prepare_lora",
    "svdquant_mxfp4_lora_down",
    "svdquant_mxfp4_lora_plane_elements",
)


def split_kernel_available() -> bool:
    """True when the XPU build exports the split-step benchmark symbols."""

    try:
        from .xpu_loader import ensure_xpu_lib

        ensure_xpu_lib(required_symbols=_SPLIT_SYMBOLS)
    except Exception:
        return False
    return True


def svdquant_mxfp4_smooth_quant(
    x: torch.Tensor,
    smooth: torch.Tensor | None = None,
    *,
    qact: torch.Tensor | None = None,
    ascales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split step 1: ARK smooth + MXFP4 quantize + pack.

    Standalone launch of the fused kernel's quantization phase. ``qact`` and
    ``ascales`` are bit-identical to what :func:`svdquant_mxfp4_quant_down`
    produces for the same ``x`` / ``smooth``.

    ``qact`` / ``ascales`` may be preallocated output buffers: when given, the
    kernel writes in place and nothing is allocated on the launch path. The
    benchmark uses this to time a pure kernel launch (the README's preallocated
    three-step baseline).
    """

    _validate_inputs(x, smooth, None)
    from . import cvt_dtype, get_stream
    from .xpu_loader import ensure_xpu_lib

    library = ensure_xpu_lib(required_symbols=_SPLIT_SYMBOLS)

    x = x.contiguous()
    rows, columns = x.shape
    smooth = None if smooth is None else smooth.to(torch.float32).contiguous()

    if qact is None:
        qact = torch.empty((rows, columns // 2), dtype=torch.uint8, device=x.device)
    elif qact.shape != (rows, columns // 2) or qact.dtype != torch.uint8 or qact.device != x.device:
        raise ValueError(f"qact must be uint8 of shape {(rows, columns // 2)} on device {x.device}")
    if ascales is None:
        ascales = torch.empty((rows, columns // GROUP_SIZE), dtype=torch.uint8, device=x.device)
    elif (
        ascales.shape != (rows, columns // GROUP_SIZE) or ascales.dtype != torch.uint8 or ascales.device != x.device
    ):
        raise ValueError(f"ascales must be uint8 of shape {(rows, columns // GROUP_SIZE)} on device {x.device}")

    library.svdquant_mxfp4_smooth_quant(
        stream=get_stream(x),
        x=x.data_ptr(),
        smooth=0 if smooth is None else smooth.data_ptr(),
        qact=qact.data_ptr(),
        ascales=ascales.data_ptr(),
        m=rows,
        k=columns,
        x_dtype=cvt_dtype(x.dtype),
    )
    return qact, ascales


def svdquant_mxfp4_prepare_lora(
    smooth: torch.Tensor,
    lora_down: torch.Tensor,
    *,
    hi: torch.Tensor | None = None,
    lo: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split step 2: ARK LoRA operand preparation.

    Folds ``smooth`` into ``lora_down`` and emits the two 16-bit split planes
    (``hi`` + ``lo``) in the exact layout the fused CUTE/DPAS path consumes.
    Size each plane with the private-layout query the kernel exposes.

    ``hi`` / ``lo`` may be preallocated output buffers: when given, the kernel
    writes in place and nothing is allocated on the launch path. The benchmark
    uses this to time a pure kernel launch.
    """

    if not isinstance(smooth, torch.Tensor) or smooth.ndim != 1:
        raise ValueError("smooth must be a 1D torch.Tensor of shape [K]")
    if not isinstance(lora_down, torch.Tensor) or lora_down.ndim != 2:
        raise ValueError("lora_down must be a 2D torch.Tensor of shape [R, K]")
    if smooth.shape[0] != lora_down.shape[1]:
        raise ValueError(f"smooth length {smooth.shape[0]} does not match lora_down K dimension {lora_down.shape[1]}")
    if lora_down.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise ValueError(f"lora_down dtype must be one of {_SUPPORTED_INPUT_DTYPES}, got {lora_down.dtype}")
    if smooth.device != lora_down.device:
        raise ValueError(f"smooth must be on device {lora_down.device}, got {smooth.device}")

    from . import cvt_dtype, get_stream
    from .xpu_loader import ensure_xpu_lib

    library = ensure_xpu_lib(required_symbols=_SPLIT_SYMBOLS)

    smooth = smooth.to(torch.float32).contiguous()
    lora_down = lora_down.contiguous()
    columns, rank = lora_down.shape[1], lora_down.shape[0]

    plane = int(library.svdquant_mxfp4_lora_plane_elements(k=columns, r=rank))
    if hi is None:
        hi = torch.empty(plane, dtype=lora_down.dtype, device=lora_down.device)
    elif hi.shape != (plane,) or hi.dtype != lora_down.dtype or hi.device != lora_down.device:
        raise ValueError(f"hi must be {lora_down.dtype} of shape {(plane,)} on device {lora_down.device}")
    if lo is None:
        lo = torch.empty(plane, dtype=lora_down.dtype, device=lora_down.device)
    elif lo.shape != (plane,) or lo.dtype != lora_down.dtype or lo.device != lora_down.device:
        raise ValueError(f"lo must be {lora_down.dtype} of shape {(plane,)} on device {lora_down.device}")

    library.svdquant_mxfp4_prepare_lora(
        stream=get_stream(lora_down),
        smooth=smooth.data_ptr(),
        lora_down=lora_down.data_ptr(),
        hi=hi.data_ptr(),
        lo=lo.data_ptr(),
        k=columns,
        r=rank,
        dtype=cvt_dtype(lora_down.dtype),
    )
    return hi, lo


def svdquant_mxfp4_lora_down(
    x: torch.Tensor,
    hi: torch.Tensor,
    lo: torch.Tensor,
    rank: int,
    *,
    lora_act: torch.Tensor | None = None,
) -> torch.Tensor:
    """Split step 3: ARK standalone low-rank down projection.

    ``lora_act[M, R] = x[M, K] @ (B_hi + B_lo)`` over the planes produced by
    :func:`svdquant_mxfp4_prepare_lora`. Uses the same CUTE/DPAS projection as
    the fused path -- never ``torch.matmul``.

    ``lora_act`` may be a preallocated output buffer: when given, the kernel
    writes in place and nothing is allocated on the launch path. The benchmark
    uses this to time a pure kernel launch.
    """

    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("x must be a 2D torch.Tensor of shape [M, K]")
    if rank <= 0 or rank > MAX_RANK:
        raise ValueError(f"rank must be in [1, {MAX_RANK}], got {rank}")
    if x.device.type != "xpu":
        raise RuntimeError("svdquant_mxfp4_lora_down requires an XPU tensor")
    if x.device != hi.device or x.device != lo.device:
        raise ValueError("x, hi and lo must be on the same device")

    from . import cvt_dtype, get_stream
    from .xpu_loader import ensure_xpu_lib

    library = ensure_xpu_lib(required_symbols=_SPLIT_SYMBOLS)

    x = x.contiguous()
    rows, columns = x.shape

    if lora_act is None:
        lora_act = torch.empty((rows, rank), dtype=x.dtype, device=x.device)
    elif lora_act.shape != (rows, rank) or lora_act.dtype != x.dtype or lora_act.device != x.device:
        raise ValueError(f"lora_act must be {x.dtype} of shape {(rows, rank)} on device {x.device}")
    library.svdquant_mxfp4_lora_down(
        stream=get_stream(x),
        x=x.data_ptr(),
        hi=hi.data_ptr(),
        lo=lo.data_ptr(),
        lora_act=lora_act.data_ptr(),
        m=rows,
        k=columns,
        r=rank,
        x_dtype=cvt_dtype(x.dtype),
    )
    return lora_act


def svdquant_mxfp4_quant_down(
    x: torch.Tensor,
    smooth: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    *,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Public Kernel A entry point.

    Args:
        x: ``[M, K]`` activations, FP16 or BF16, ``K % 32 == 0``.
        smooth: optional ``[K]`` smoothing vector.
        lora_down: optional ``[R, K]`` low-rank down factor, ``1 <= R <= 64``.
        backend: ``"ark"`` forces the fused SYCL kernel and raises if it is
            unavailable; ``"reference"`` forces the PyTorch path; ``"auto"``
            picks the fused kernel on XPU when it is loadable and the reference
            otherwise.

    Returns:
        ``(qact, ascales, lora_act)`` as documented on
        :func:`quant_down_reference`.

    The dispatch is deliberately explicit: ``backend="ark"`` never degrades to
    PyTorch behind the caller's back, because a silent fallback would turn a
    kernel regression into a performance mystery rather than a test failure.
    """

    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")

    _validate_inputs(x, smooth, lora_down)
    if lora_down is not None and lora_down.shape[0] > MAX_RANK:
        if backend == "ark":
            raise ValueError(f"fused kernel supports rank <= {MAX_RANK}, got {lora_down.shape[0]}")
        return quant_down_reference(x, smooth, lora_down)

    if backend == "reference":
        return quant_down_reference(x, smooth, lora_down)

    if backend == "ark":
        if x.device.type != "xpu":
            raise RuntimeError(f"backend='ark' requires an XPU tensor, got device {x.device}")
        return _quant_down_ark(x, smooth, lora_down)

    if x.device.type == "xpu" and ark_kernel_available():
        return _quant_down_ark(x, smooth, lora_down)
    return quant_down_reference(x, smooth, lora_down)


def dequantize_reference(qact: torch.Tensor, ascales: torch.Tensor) -> torch.Tensor:
    """Decode ``qact``/``ascales`` back to FP32. Test and debug helper only."""

    if not isinstance(qact, torch.Tensor) or qact.dtype != torch.uint8 or qact.ndim != 2:
        raise ValueError("qact must be a 2D uint8 torch.Tensor")
    if not isinstance(ascales, torch.Tensor) or ascales.dtype != torch.uint8 or ascales.ndim != 2:
        raise ValueError("ascales must be a 2D uint8 torch.Tensor")
    rows, packed_columns = qact.shape
    columns = packed_columns * 2
    if ascales.shape != (rows, columns // GROUP_SIZE):
        raise ValueError(f"ascales shape must be {(rows, columns // GROUP_SIZE)}, got {tuple(ascales.shape)}")

    codes = torch.stack((qact & 0x0F, qact >> 4), dim=-1).reshape(rows, columns)
    codebook = torch.tensor(
        (*E2M1_MAGNITUDES, *(-value for value in E2M1_MAGNITUDES)),
        dtype=torch.float32,
        device=qact.device,
    )
    values = codebook[codes.to(torch.long)].reshape(rows, columns // GROUP_SIZE, GROUP_SIZE)
    scale = torch.exp2((ascales.to(torch.int32) - UE8M0_BIAS).to(torch.float32))
    return (values * scale.unsqueeze(-1)).reshape(rows, columns)
