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
"""Steps 1 and 2 of the SVDQuant INT4 reference study.

These checks are deliberately hardware-light (they run on CPU with a sliced
FLUX-shaped block) because they gate the expensive model-level sweep:

* Step 1 - the frozen INT4 numerical contract is what the QDQ actually does.
* Step 2 - the INT4 QDQ is applied only to the SVDQuant residual branch while
  the low-rank branch stays BF16, and the transformed layer numerically matches
  an independently computed ``Q(R) + U @ V``.
"""

from types import SimpleNamespace

import pytest
import torch

from auto_round.algorithms.composer import AlgorithmComposer, BlockContext
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.transforms.svdquant import SVDQuantConfig, SVDQuantLinear
from auto_round.algorithms.transforms.svdquant.residual import ResidualQuantScheme, rtn_qdq_residual, truncated_svd
from auto_round.data_type.int import quant_block_tensor_sym, quant_tensor_sym
from auto_round.data_type.utils import get_quant_func
from auto_round.schemes import QuantizationScheme
from auto_round.utils.device_manager import device_manager

from .contract import BLOCK_CANDIDATES, INT4_CONTRACT, K_AXIS_CANDIDATES, Candidate, weight_only_candidates


class TestInt4Contract:
    """Step 1: the PyTorch QDQ represents the proposed kernel exactly."""

    @pytest.mark.parametrize("block", BLOCK_CANDIDATES)
    def test_block_contract(self, block):
        weight = torch.randn(256, 256)
        qdq, scale, maxq = quant_block_tensor_sym(weight, bits=4, group_size=block)
        assert maxq == 8
        assert INT4_CONTRACT["code_range"] == [-8, 7]
        codes = qdq / scale.repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
        codes = codes.round()
        assert codes.min() >= -8 and codes.max() <= 7
        assert torch.isfinite(qdq).all()
        assert qdq.shape == weight.shape

    @pytest.mark.parametrize("group_size", [g for g in K_AXIS_CANDIDATES if g > 0])
    def test_k_axis_contract_matches_block_contract(self, group_size):
        """The K-axis and M-by-K paths must be one contract, not two."""
        weight = torch.randn(64, 256)
        k_axis, _, _ = quant_tensor_sym(weight, bits=4, group_size=group_size)
        block, _, _ = quant_block_tensor_sym(weight, bits=4, group_size=(1, group_size))
        assert torch.equal(k_axis, block)

    def test_scheme_resolves_to_the_contract_function(self):
        for candidate in weight_only_candidates():
            scheme = candidate.to_scheme()
            func, name = get_quant_func(
                scheme.data_type, scheme.bits, scheme.sym, disable_opt_rtn=True, group_size=scheme.group_size, iters=0
            )
            if isinstance(scheme.group_size, tuple):
                assert name == "block_int_sym", candidate.name
                assert func is quant_block_tensor_sym
            else:
                assert name.endswith("int_sym"), candidate.name

    def test_scheme_construction(self):
        scheme = Candidate(name="w64", weight_block=64).to_scheme()
        assert isinstance(scheme, QuantizationScheme)
        assert scheme.bits == 4 and scheme.sym and scheme.data_type == "int"
        assert scheme.act_bits == 16

        w4a4 = Candidate(name="w64_a32", weight_block=64, act_group_size=32, act_bits=4).to_scheme()
        assert w4a4.act_bits == 4
        assert w4a4.act_group_size == 32
        assert w4a4.act_dynamic is True


class SlicedFluxBlock(torch.nn.Module):
    """A thin slice of a FLUX transformer block: one projection per group."""

    def __init__(self, in_features=64, out_features=48):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.to_q = torch.nn.Linear(in_features, out_features)

    def forward(self, hidden_states):
        return self.attn.to_q(hidden_states)


class SlicedFluxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([SlicedFluxBlock()])


def _prepare_model(scheme: QuantizationScheme) -> SlicedFluxModel:
    # The fixed experiment contract runs FLUX.1-dev in BF16.
    model = SlicedFluxModel().to(torch.bfloat16)
    for name, module in model.named_modules():
        module.global_name = name
        if isinstance(module, torch.nn.Linear):
            module.bits = scheme.bits
            module.group_size = scheme.group_size
            module.sym = scheme.sym
            module.data_type = scheme.data_type
            module.act_bits = scheme.act_bits or 16
            module.act_group_size = scheme.act_group_size or 32
            module.act_sym = True
            module.act_data_type = scheme.act_data_type or "int"
            module.act_dynamic = True
            module.scale_dtype = torch.float32
    return model


def _orchestrator(model, block_name, scheme):
    return SimpleNamespace(
        model_context=SimpleNamespace(
            model=model,
            amp=False,
            amp_dtype=torch.bfloat16,
            is_diffusion=True,
            is_moe_model=False,
            output_config=None,
        ),
        compress_context=SimpleNamespace(
            enable_torch_compile=False,
            low_gpu_mem_usage=False,
            cache_device=torch.device("cpu"),
            clear_memory=lambda: None,
        ),
        calibration_context=SimpleNamespace(batch_size=1, batch_dim=0),
        scheme_context=scheme,
        scale_dtype=None,
        nblocks=1,
        quant_block_list=[[block_name]],
        data_type=scheme.data_type,
        batch_dim=0,
        batch_size=1,
        cache_device="cpu",
        amp=False,
        amp_dtype=torch.bfloat16,
        shared_cache_keys=(),
    )


@pytest.mark.parametrize("weight_block", [64, (32, 32)], ids=["k_axis", "block"])
def test_svdquant_workflow_applies_int4_to_the_residual_only(monkeypatch, weight_block):
    """Step 2: residual branch is INT4-QDQ, low-rank branch stays BF16, forward matches Q(R) + U @ V."""
    monkeypatch.setattr(device_manager, "_device_map", "cpu")
    monkeypatch.setattr(device_manager, "_device_list", ["cpu"])
    monkeypatch.setattr(device_manager, "_major_device", "cpu")

    rank = 4
    scheme = Candidate(name="probe", weight_block=weight_block).to_scheme()
    model = _prepare_model(scheme)
    block_name = "blocks.0"
    original_weight = model.blocks[0].attn.to_q.weight.detach().clone().float()

    composer = AlgorithmComposer(
        [
            SVDQuantConfig(
                rank=rank,
                smooth_enabled=False,
                residual_iters=1,
                low_rank_dtype="bf16",
                target_modules=["to_q"],
            ),
            RTNConfig(disable_opt_rtn=True),
        ],
        orchestrator=_orchestrator(model, block_name, scheme),
    )
    composer.prepare_run()
    block = model.blocks[0]
    inputs = torch.randn(1, 2, 64, dtype=torch.bfloat16)
    composer.compress_block(
        block,
        [inputs],
        {},
        BlockContext(model=model, block_names=[block_name], block_name=block_name, block_index=0, block_cnt=1),
    )

    layer = block.attn.to_q
    assert isinstance(layer, SVDQuantLinear)

    # Low-rank branch is untouched BF16 of the requested rank.
    assert layer.lora_down.weight.dtype == torch.bfloat16
    assert layer.lora_up.weight.dtype == torch.bfloat16
    assert layer.lora_down.weight.shape[0] == rank
    assert layer.lora_up.weight.shape[1] == rank

    # Residual branch carries INT4 codes on the frozen contract.
    assert layer.residual_linear.weight.dtype == torch.bfloat16
    residual_weight = layer.residual_linear.weight.detach().float()
    scheme_obj = ResidualQuantScheme(data_type="int", bits=4, group_size=weight_block, sym=True)
    # QDQ is idempotent: re-quantizing an already quantized residual is a no-op.
    requantized = rtn_qdq_residual(layer.residual_linear.weight.detach(), scheme_obj).float()
    assert torch.equal(residual_weight, requantized), "residual weight is not on the INT4 grid"

    # Independent reference: Q(R) + U @ V must reconstruct the effective weight.
    low_rank = layer.lora_up.weight.float() @ layer.lora_down.weight.float()
    _, down, up = truncated_svd(original_weight, rank)
    reference_low_rank = up.to(torch.bfloat16).float() @ down.to(torch.bfloat16).float()
    reference_residual = original_weight - reference_low_rank
    reference_qdq = rtn_qdq_residual(reference_residual.to(torch.bfloat16), scheme_obj).float()
    effective = residual_weight + low_rank
    reference_effective = reference_qdq + reference_low_rank
    assert torch.allclose(effective, reference_effective, atol=5e-2)

    # Forward is finite, shape-preserving, and equals the two-branch definition.
    output = layer(inputs)
    assert output.shape == (1, 2, 48)
    assert torch.isfinite(output).all()
    smoothed = inputs.float() * layer.smooth.float()
    expected = smoothed @ residual_weight.t() + smoothed @ low_rank.t()
    if layer.residual_linear.bias is not None:
        expected = expected + layer.residual_linear.bias.float()
    assert torch.allclose(output.float(), expected, atol=5e-2)


@pytest.mark.parametrize("weight_block", [64, (32, 32)], ids=["k_axis", "block"])
def test_low_rank_branch_absorbs_outliers(weight_block):
    """The INT4 residual branch must beat quantizing the full weight on outlier-carrying layers."""
    torch.manual_seed(0)
    weight = torch.randn(256, 256)
    weight[:, ::37] *= 8.0  # input-channel outliers, the case SVDQuant targets
    weight = weight.to(torch.bfloat16)

    scheme_obj = ResidualQuantScheme(data_type="int", bits=4, group_size=weight_block, sym=True)
    _, down, up = truncated_svd(weight.float(), 32)
    low_rank = up.to(torch.bfloat16).float() @ down.to(torch.bfloat16).float()
    residual_qdq = rtn_qdq_residual((weight.float() - low_rank).to(torch.bfloat16), scheme_obj).float()
    svdquant_error = (residual_qdq + low_rank - weight.float()).square().mean()
    direct_error = (rtn_qdq_residual(weight, scheme_obj).float() - weight.float()).square().mean()
    assert svdquant_error < direct_error
