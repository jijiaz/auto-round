# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the block (M-by-K) symmetric integer QDQ used by the SVDQuant INT4 study."""

import pytest
import torch

from auto_round.data_type.int import quant_block_tensor_sym, quant_tensor_sym
from auto_round.data_type.utils import get_quant_func


def _reference_block_qdq(tensor, block, bits=4, scale_dtype=torch.float16):
    """Independent reference: full-range symmetric INT QDQ with one scale per (M, K) tile."""
    maxq = 2 ** (bits - 1)
    m, k = block
    out = torch.zeros_like(tensor, dtype=torch.float32)
    scales = []
    for i in range(0, tensor.shape[0], m):
        row = []
        for j in range(0, tensor.shape[1], k):
            tile = tensor[i : i + m, j : j + k].to(torch.float32)
            wmin = min(tile.min().item(), 0.0)
            wmax = max(tile.max().item(), 0.0)
            # Full-range convention inherited from ``int_sym``: the dominant magnitude is
            # mapped onto the -2**(bits-1) code, so the scale carries the opposite sign.
            max_v = -wmin if abs(wmax) < abs(wmin) else -wmax
            scale = torch.tensor(max_v / maxq).to(scale_dtype)
            scale = torch.clamp(scale, max=-1e-5) if scale < 0 else torch.clamp(scale, min=1e-5)
            q = torch.clamp(torch.round(tile / scale.to(torch.float32)), -maxq, maxq - 1)
            out[i : i + m, j : j + k] = q * scale.to(torch.float32)
            row.append(scale)
        scales.append(row)
    return out


class TestBlockIntRegistration:
    def test_registered_for_tuple_group_size(self):
        func, name = get_quant_func("int", bits=4, sym=True, disable_opt_rtn=True, group_size=(128, 128), iters=0)
        assert name == "block_int_sym"
        assert func is quant_block_tensor_sym

    def test_tuple_dispatch_wins_over_rtn_alias(self):
        # ``rtn_int_sym`` must not shadow the block implementation for a 2D group size.
        for iters in (0, 200):
            _, name = get_quant_func("int", bits=4, sym=True, disable_opt_rtn=False, group_size=(64, 64), iters=iters)
            assert name == "block_int_sym"

    def test_scalar_group_size_unchanged(self):
        _, name = get_quant_func("int", bits=4, sym=True, disable_opt_rtn=True, group_size=32, iters=200)
        assert name == "int_sym"


class TestBlockIntSemantics:
    def test_matches_int_sym_when_block_height_is_one(self):
        """A (1, K) block is exactly K-axis grouping, so it must match ``int_sym`` bit for bit."""
        weight = torch.randn(8, 128)
        for group_size in (16, 32, 64, 128):
            block_qdq, _, _ = quant_block_tensor_sym(weight, bits=4, group_size=(1, group_size))
            ref_qdq, _, _ = quant_tensor_sym(weight, bits=4, group_size=group_size)
            assert torch.equal(block_qdq, ref_qdq)

    def test_matches_hand_computed_reference(self):
        weight = torch.randn(32, 96)
        qdq, scale, maxq = quant_block_tensor_sym(weight, bits=4, group_size=(16, 32))
        assert maxq == 8
        assert scale.shape == (2, 3)
        expected = _reference_block_qdq(weight, (16, 32))
        assert torch.allclose(qdq, expected, atol=0, rtol=0)

    def test_exact_small_tensor(self):
        weight = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
        qdq, scale, _ = quant_block_tensor_sym(weight, bits=4, group_size=(2, 2))
        # max magnitude is 2.0 -> |scale| = 0.25, codes are round(w / scale) clamped to [-8, 7]
        assert scale.numel() == 1
        assert pytest.approx(abs(scale.item()), rel=1e-3) == 0.25
        assert torch.allclose(qdq, torch.tensor([[1.0, -2.0], [0.5, 0.25]]), atol=1e-3)

    def test_code_range_is_signed_full_range(self):
        weight = torch.randn(64, 64) * 3.0
        qdq, scale, maxq = quant_block_tensor_sym(weight, bits=4, group_size=(32, 32))
        codes = (qdq / scale.repeat_interleave(32, 0).repeat_interleave(32, 1)).round()
        assert codes.min() >= -maxq
        assert codes.max() <= maxq - 1

    def test_zero_block_is_finite_and_zero(self):
        weight = torch.zeros(32, 32)
        weight[16:, 16:] = torch.randn(16, 16)
        qdq, scale, _ = quant_block_tensor_sym(weight, bits=4, group_size=(16, 16))
        assert torch.isfinite(qdq).all()
        assert torch.count_nonzero(qdq[:16, :16]) == 0
        assert torch.isfinite(scale).all()
        assert (scale.abs() > 0).all()

    @pytest.mark.parametrize("shape", [(20, 30), (33, 65), (7, 7)])
    @pytest.mark.parametrize("block", [(16, 16), (32, 32)])
    def test_non_divisible_shapes(self, shape, block):
        weight = torch.randn(*shape)
        qdq, scale, _ = quant_block_tensor_sym(weight, bits=4, group_size=block)
        assert qdq.shape == weight.shape
        assert torch.isfinite(qdq).all()
        expected_scale_shape = (
            -(-shape[0] // block[0]),
            -(-shape[1] // block[1]),
        )
        assert tuple(scale.shape) == expected_scale_shape

    def test_padding_does_not_change_shared_scale(self):
        """Zero padding of a partial block must not widen the block range."""
        weight = torch.randn(16, 24)
        padded = torch.zeros(16, 32)
        padded[:, :24] = weight
        qdq_partial, scale_partial, _ = quant_block_tensor_sym(weight, bits=4, group_size=(16, 32))
        qdq_padded, scale_padded, _ = quant_block_tensor_sym(padded, bits=4, group_size=(16, 32))
        assert torch.equal(scale_partial, scale_padded)
        assert torch.equal(qdq_partial, qdq_padded[:, :24])

    @pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scale_dtype_materialization(self, scale_dtype):
        weight = torch.randn(32, 32)
        qdq, scale, _ = quant_block_tensor_sym(weight, bits=4, group_size=(32, 32), scale_dtype=scale_dtype)
        assert scale.dtype == scale_dtype
        assert torch.isfinite(qdq).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_dtype_preserved(self, dtype):
        weight = torch.randn(32, 64).to(dtype)
        qdq, _, _ = quant_block_tensor_sym(weight, bits=4, group_size=(16, 16))
        assert qdq.dtype == dtype
        assert qdq.shape == weight.shape

    @pytest.mark.parametrize(
        "block",
        [(1, 16), (1, 32), (1, 64), (1, 128), (16, 16), (16, 32), (32, 32), (32, 64), (64, 64), (128, 128)],
    )
    def test_candidate_block_shapes(self, block):
        weight = torch.randn(256, 256)
        qdq, scale, _ = quant_block_tensor_sym(weight, bits=4, group_size=block)
        assert qdq.shape == weight.shape
        assert tuple(scale.shape) == (256 // block[0], 256 // block[1])
        assert torch.isfinite(qdq).all()
        # Smaller blocks can never be worse than larger ones on average.
        assert (qdq - weight).abs().mean() < weight.abs().mean()

    def test_error_decreases_monotonically_with_block_size(self):
        torch.manual_seed(0)
        weight = torch.randn(256, 256)
        weight[0, 0] = 50.0  # outlier, the case SVDQuant's low-rank branch targets
        errors = []
        for k in (16, 32, 64, 128, 256):
            qdq, _, _ = quant_block_tensor_sym(weight, bits=4, group_size=(1, k))
            errors.append((qdq - weight).square().mean().item())
        assert errors == sorted(errors)

    def test_rejects_invalid_group_size(self):
        with pytest.raises(ValueError):
            quant_block_tensor_sym(torch.randn(16, 16), bits=4, group_size=16)
        with pytest.raises(ValueError):
            quant_block_tensor_sym(torch.randn(16, 16), bits=4, group_size=(16, 16, 16))

    def test_rejects_non_2d_tensor(self):
        with pytest.raises(ValueError):
            quant_block_tensor_sym(torch.randn(2, 16, 16), bits=4, group_size=(16, 16))

    def test_handles_non_finite_free_input(self):
        weight = torch.randn(32, 32)
        weight[0, 0] = 1e30
        qdq, scale, _ = quant_block_tensor_sym(weight, bits=4, group_size=(32, 32), scale_dtype=torch.float32)
        assert torch.isfinite(qdq).all()
        assert torch.isfinite(scale).all()
