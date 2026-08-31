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
"""Frozen experiment contract for the SVDQuant INT4 reference accuracy study.

This module is the single source of truth shared by the tests, the coarse proxy
sweep, the full model-level sweep, and the aggregation/gating script. Changing a
value here changes every stage of the study, so treat it as the experiment
manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from auto_round.schemes import QuantizationScheme

# ---------------------------------------------------------------------------
# Step 1: the INT4 numerical contract
# ---------------------------------------------------------------------------
#
# The PyTorch reference must represent the proposed kernel exactly, so the
# contract below is stated explicitly instead of being implied by a data type
# name. It is implemented by ``auto_round.data_type.int.quant_block_tensor_sym``
# (registered as ``block_int_sym``) for M-by-K block scales, and by the existing
# ``int_sym`` for ordinary K-axis groups. ``block_int_sym`` is bit-exact with
# ``int_sym`` when the block height is 1, so both layouts share one contract.
INT4_CONTRACT: dict[str, Any] = {
    "code_range": [-8, 7],
    "signed": True,
    "scale_equation": "scale = -sign(dominant) * max(|w_min|, |w_max|) / 8, clamped to |scale| >= 1e-5",
    "scale_dtype": "float16 (materialized before use; float32 optional for study diagnostics)",
    "rounding_mode": "round-half-to-even, then clamp to the code range",
    "clamping_order": "round -> clamp (clamp never feeds back into the scale)",
    "weight_block_axes": "(out_features, in_features) = (M, K); K is the reduction axis",
    "activation_block_axes": "K only (per-token dynamic groups); activation blocks are 1-D",
    "weight_activation_coupling": "independent; the sweep scans them one at a time",
    "padding": "zero padding of partial blocks; padding never widens a block range and is cropped",
    "zero_block": "scale clamped to +1e-5, all codes 0, output exactly 0",
    "non_finite": "inputs are asserted finite; QDQ must not create NaN/Inf",
    "scale_layout": "2-D (ceil(M_out/M), ceil(K/K_block)) for block_int_sym; 1-D K groups for int_sym",
    "data_type_weight": "block_int_sym (M > 1) or int_sym (M == 1)",
    "data_type_activation": "int_sym, dynamic, per-token K groups",
}

# ---------------------------------------------------------------------------
# Fixed experiment contract (Steps 3-5)
# ---------------------------------------------------------------------------
FIXED_CONTRACT: dict[str, Any] = {
    "model": "black-forest-labs/FLUX.1-dev",
    "model_dtype": "bf16",
    "low_rank_dtype": "bf16",
    "rank": 32,
    "calibration_samples": 128,
    "calibration_dataset": "coco2017 captions (tsv, id + caption columns)",
    "num_inference_steps": 50,
    "signround_iters": 200,
    "residual_iters": 20,
    "metrics": ["clip", "clip-iqa", "imagereward"],
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "batch_size": 1,
    "seeds": [12345],
    "eval_limit": 100,
}

# ---------------------------------------------------------------------------
# Step 6: pre-registered acceptance gates
# ---------------------------------------------------------------------------
#
# Absolute floor = the worst published MXFP4 SVDQuant RTN result for FLUX.1-dev.
# It is a conservative provisional envelope until dedicated INT4 thresholds are
# approved. Relative gates are applied against the *fresh* controls measured in
# the same environment and are preferred whenever controls are available.
ABSOLUTE_GATES: dict[str, float] = {
    "clip": 25.9624,
    "clip-iqa": 0.946939,
    "imagereward": 0.934579,
}

# Maximum tolerated regression against the fresh BF16 control. Derived from the
# published BF16 -> worst-MXFP4-RTN deltas (CLIP -0.0565, CLIP-IQA -0.0074,
# ImageReward -0.0838), rounded outward slightly to absorb sampling noise.
RELATIVE_GATES_VS_BF16: dict[str, float] = {
    "clip": -0.10,
    "clip-iqa": -0.010,
    "imagereward": -0.09,
}

# Historical in-repository reference points. Not automatically reproducible;
# every sweep must record fresh controls in the same environment.
HISTORICAL_REFERENCE: dict[str, dict[str, float]] = {
    "bf16": {"clip": 26.0189, "clip-iqa": 0.954360, "imagereward": 1.018340},
    "mxfp4_smooth_signround": {"clip": 26.1039, "clip-iqa": 0.962655, "imagereward": 1.021020},
    "mxfp4_nosmooth_signround": {"clip": 26.0727, "clip-iqa": 0.959363, "imagereward": 1.002380},
    "mxfp4_smooth_rtn": {"clip": 25.9719, "clip-iqa": 0.947763, "imagereward": 0.939392},
    "mxfp4_nosmooth_rtn": {"clip": 25.9624, "clip-iqa": 0.946939, "imagereward": 0.934579},
}

# ---------------------------------------------------------------------------
# Candidate block sizes
# ---------------------------------------------------------------------------
# K-axis candidates with a per-row scale (M = 1). ``-1`` means per-output-channel.
K_AXIS_CANDIDATES: tuple[int, ...] = (16, 32, 64, 128, 256, -1)

# Two-dimensional (M, K) candidates for a block-scaled kernel contract.
BLOCK_CANDIDATES: tuple[tuple[int, int], ...] = (
    (16, 16),
    (16, 32),
    (32, 32),
    (32, 64),
    (64, 64),
    (128, 128),
)

# Activation group sizes scanned while the weight block is held fixed.
ACT_GROUP_CANDIDATES: tuple[int, ...] = (16, 32, 64, 128, -1)


@dataclass(frozen=True)
class Candidate:
    """One point of the block-size sweep."""

    name: str
    weight_block: int | tuple[int, int]
    act_group_size: int | None = None
    act_bits: int = 16
    notes: str = ""

    def to_scheme(self) -> QuantizationScheme:
        """Build the AutoRound scheme implementing this candidate's contract."""
        scheme = QuantizationScheme(
            bits=4,
            group_size=self.weight_block,
            sym=True,
            data_type="int",
            act_bits=self.act_bits,
        )
        if self.act_bits < 16:
            scheme.act_data_type = "int"
            scheme.act_group_size = self.act_group_size if self.act_group_size is not None else 32
            scheme.act_sym = True
            scheme.act_dynamic = True
        return scheme

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scheme"] = self.to_scheme().to_dict()
        return payload


def weight_only_candidates(blocks=None) -> list[Candidate]:
    """W4A16 candidates: scan the weight block size with activations in BF16."""
    blocks = blocks if blocks is not None else (*K_AXIS_CANDIDATES, *BLOCK_CANDIDATES)
    return [Candidate(name=_block_name(block), weight_block=block) for block in blocks]


def w4a4_candidates(weight_block: int | tuple[int, int], act_groups=None) -> list[Candidate]:
    """W4A4 candidates: hold the weight block fixed and scan the activation group."""
    act_groups = act_groups if act_groups is not None else ACT_GROUP_CANDIDATES
    return [
        Candidate(
            name=f"{_block_name(weight_block)}_a{act if act != -1 else 'perchannel'}",
            weight_block=weight_block,
            act_group_size=act,
            act_bits=4,
        )
        for act in act_groups
    ]


def _block_name(block: int | tuple[int, int]) -> str:
    if isinstance(block, tuple):
        return f"w{block[0]}x{block[1]}"
    return "wperchannel" if block == -1 else f"w{block}"


@dataclass
class ExperimentManifest:
    """Machine-readable manifest stored with every sweep."""

    stage: str
    int4_contract: dict[str, Any] = field(default_factory=lambda: dict(INT4_CONTRACT))
    fixed_contract: dict[str, Any] = field(default_factory=lambda: dict(FIXED_CONTRACT))
    absolute_gates: dict[str, float] = field(default_factory=lambda: dict(ABSOLUTE_GATES))
    relative_gates_vs_bf16: dict[str, float] = field(default_factory=lambda: dict(RELATIVE_GATES_VS_BF16))
    historical_reference: dict[str, dict[str, float]] = field(default_factory=lambda: dict(HISTORICAL_REFERENCE))
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def environment_revisions() -> dict[str, str]:
    """Collect the software revisions that must be stored with every run."""
    revisions: dict[str, str] = {}
    for module_name in ("torch", "transformers", "diffusers", "auto_round", "torchmetrics"):
        try:
            module = __import__(module_name)
        except ImportError:
            revisions[module_name] = "not installed"
            continue
        revisions[module_name] = str(getattr(module, "__version__", "unknown"))
    return revisions
