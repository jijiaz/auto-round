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
"""Step 4 (coarse): layer-level screening sweep for SVDQuant INT4 block sizes.

This stage answers *cheaply* which block sizes can possibly survive the
model-level gates, so that the expensive FLUX.1-dev sweep only has to evaluate
the transition region.

It reuses the production SVDQuant primitives (``truncated_svd``,
``iterate_residual_decomposition``, ``rtn_qdq_residual``) and the registered QDQ
functions, and measures the *layer output* NMSE of ``Q(R) + U @ V`` against the
BF16 layer, averaged over the linear shapes of the FLUX.1-dev transformer.

Screening rule
--------------
The MXFP4 group-32 SVDQuant configuration is the only INT-class 4-bit setup with
published model-level results on FLUX.1-dev, and its *worst* variant
(no smooth + RTN) still satisfies the acceptance gates. It is therefore used as
the screening control: an INT4 block size is a screening pass when its layer
NMSE is no worse than the MXFP4 group-32 control on the same weights.

This is screening data only. It never replaces the model-level result of
Step 5, but it does bound where the boundary can be.

Usage::

    python proxy_block_sweep.py --output records/proxy.json
    python proxy_block_sweep.py --shapes-from /path/to/FLUX.1-dev  # real weights
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from auto_round.algorithms.transforms.svdquant.residual import (
    ResidualQuantScheme,
    iterate_residual_decomposition,
    rtn_qdq_residual,
)

try:
    from .contract import (
        ACT_GROUP_CANDIDATES,
        BLOCK_CANDIDATES,
        FIXED_CONTRACT,
        K_AXIS_CANDIDATES,
        ExperimentManifest,
        environment_revisions,
    )
except ImportError:  # executed as a standalone script
    from contract import (
        ACT_GROUP_CANDIDATES,
        BLOCK_CANDIDATES,
        FIXED_CONTRACT,
        K_AXIS_CANDIDATES,
        ExperimentManifest,
        environment_revisions,
    )

# Linear shapes (out_features, in_features) of the FLUX.1-dev transformer blocks.
# One representative of each distinct projection shape is enough for screening.
FLUX_LINEAR_SHAPES: tuple[tuple[str, int, int], ...] = (
    ("attn.to_q", 3072, 3072),
    ("attn.to_out", 3072, 3072),
    ("ff.net.0.proj", 12288, 3072),
    ("ff.net.2", 3072, 12288),
    ("single.linear1", 21504, 3072),
    ("single.linear2", 3072, 15360),
)

MXFP4_CONTROL = "mx_fp4"


def synthetic_weight(out_features: int, in_features: int, generator: torch.Generator) -> torch.Tensor:
    """Heavy-tailed weight with input-channel outliers, matching DiT weight statistics.

    Diffusion-transformer linears are close to Gaussian in bulk but carry a small
    number of high-magnitude input channels; those outliers are exactly what the
    SVDQuant low-rank branch absorbs and what makes the residual block size matter.
    """
    base = torch.randn(out_features, in_features, generator=generator)
    # Student-t style heavy tail.
    scale = torch.rand(out_features, 1, generator=generator).mul(0.5).add(0.75)
    weight = base * scale
    outlier_count = max(1, in_features // 128)
    outlier_channels = torch.randperm(in_features, generator=generator)[:outlier_count]
    weight[:, outlier_channels] *= 6.0
    return (weight / weight.std()).to(torch.bfloat16).float()


def _residual_scheme(block, data_type: str = "int") -> ResidualQuantScheme:
    return ResidualQuantScheme(data_type=data_type, bits=4, group_size=block, sym=True)


def _effective_weight(weight: torch.Tensor, block, *, rank: int, iters: int, data_type: str) -> torch.Tensor:
    scheme = _residual_scheme(block, data_type)
    decomposition = iterate_residual_decomposition(
        weight,
        rank=rank,
        scheme=scheme,
        iterations=iters,
        early_stop=False,
        residual_dtype=torch.bfloat16,
        low_rank_dtype=torch.bfloat16,
    )
    quantized_residual = rtn_qdq_residual(decomposition.residual.to(torch.bfloat16), scheme).float()
    low_rank = decomposition.up.float() @ decomposition.down.float()
    return quantized_residual + low_rank


def _activation_qdq(activation: torch.Tensor, act_group_size: int) -> torch.Tensor:
    from auto_round.data_type.int import quant_tensor_sym

    qdq, _, _ = quant_tensor_sym(activation, bits=4, group_size=act_group_size)
    return qdq


def layer_nmse(
    weight: torch.Tensor,
    block,
    *,
    rank: int,
    iters: int,
    data_type: str,
    activation: torch.Tensor,
    act_group_size: int | None = None,
) -> float:
    """Relative output error ``||W_eff x - W x||^2 / ||W x||^2`` for one layer."""
    effective = _effective_weight(weight, block, rank=rank, iters=iters, data_type=data_type)
    reference = activation @ weight.t()
    quant_input = activation if act_group_size is None else _activation_qdq(activation, act_group_size)
    approx = quant_input @ effective.t()
    return (approx - reference).square().sum().item() / reference.square().sum().item()


def run_sweep(args) -> dict:
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    weight_blocks: list = [*K_AXIS_CANDIDATES, *BLOCK_CANDIDATES]
    records: list[dict] = []
    started = time.time()

    for shape_name, out_features, in_features in FLUX_LINEAR_SHAPES:
        out_features = max(args.min_dim, out_features // args.shrink)
        in_features = max(args.min_dim, in_features // args.shrink)
        weight = synthetic_weight(out_features, in_features, generator)
        activation = torch.randn(args.tokens, in_features, generator=generator)

        configs: list[tuple[str, object, str, int | None]] = [
            ("mxfp4_g32_control", 32, MXFP4_CONTROL, None),
        ]
        configs += [(f"int4_{block}", block, "int", None) for block in weight_blocks]
        if args.scan_activations:
            configs += [
                (f"int4_w{args.act_weight_block}_a{act}", args.act_weight_block, "int", act)
                for act in ACT_GROUP_CANDIDATES
            ]

        for config_name, block, data_type, act_group_size in configs:
            if isinstance(block, int) and block > 0 and block > in_features:
                continue
            nmse = layer_nmse(
                weight,
                block,
                rank=args.rank,
                iters=args.residual_iters,
                data_type=data_type,
                activation=activation,
                act_group_size=act_group_size,
            )
            records.append(
                {
                    "layer": shape_name,
                    "out_features": out_features,
                    "in_features": in_features,
                    "config": config_name,
                    "weight_block": list(block) if isinstance(block, tuple) else block,
                    "data_type": data_type,
                    "act_group_size": act_group_size,
                    "nmse": nmse,
                }
            )
            print(f"{shape_name:>16} {config_name:>24} nmse={nmse:.6e}", flush=True)

    summary = summarize(records)
    manifest = ExperimentManifest(
        stage="step4_coarse_proxy",
        extra={
            "screening_rule": "pass when mean layer NMSE <= MXFP4 group-32 SVDQuant control",
            "rank": args.rank,
            "residual_iters": args.residual_iters,
            "seed": args.seed,
            "tokens": args.tokens,
            "shrink": args.shrink,
            "revisions": environment_revisions(),
            "wall_time_s": round(time.time() - started, 2),
            "note": "screening data only; never a substitute for the Step 5 model-level result",
        },
    ).to_dict()
    return {"manifest": manifest, "records": records, "summary": summary}


def summarize(records: list[dict]) -> dict:
    """Aggregate per-layer NMSE into a per-config geometric mean and a pass flag."""
    by_config: dict[str, list[float]] = {}
    for record in records:
        by_config.setdefault(record["config"], []).append(record["nmse"])

    control = by_config.get("mxfp4_g32_control")
    control_gmean = _gmean(control) if control else None

    summary = {}
    for config, values in by_config.items():
        gmean = _gmean(values)
        summary[config] = {
            "geomean_nmse": gmean,
            "max_nmse": max(values),
            "layers": len(values),
            "ratio_vs_mxfp4_control": (gmean / control_gmean) if control_gmean else None,
            "screening_pass": (gmean <= control_gmean) if control_gmean else None,
        }
    return summary


def _gmean(values: list[float]) -> float:
    """Geometric mean; NMSE spans orders of magnitude across layer shapes."""
    return float(math.exp(statistics.fmean(math.log(max(value, 1e-30)) for value in values)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("records/proxy_block_sweep.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=FIXED_CONTRACT["rank"])
    parser.add_argument("--residual-iters", type=int, default=FIXED_CONTRACT["residual_iters"])
    parser.add_argument("--tokens", type=int, default=256, help="calibration tokens used for the output NMSE")
    parser.add_argument("--shrink", type=int, default=4, help="divide FLUX linear shapes to keep the sweep cheap")
    parser.add_argument("--min-dim", type=int, default=256)
    parser.add_argument("--scan-activations", action="store_true", help="also scan W4A4 activation group sizes")
    parser.add_argument("--act-weight-block", type=int, default=64)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    result = run_sweep(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.output}")
    control = result["summary"].get("mxfp4_g32_control", {}).get("geomean_nmse")
    print(f"MXFP4 group-32 control geomean NMSE: {control:.6e}")
    for config, stats in sorted(result["summary"].items()):
        if config == "mxfp4_g32_control":
            continue
        verdict = "PASS" if stats["screening_pass"] else "fail"
        print(f"{config:>24} geomean={stats['geomean_nmse']:.6e} ratio={stats['ratio_vs_mxfp4_control']:.3f} {verdict}")


if __name__ == "__main__":
    main()
