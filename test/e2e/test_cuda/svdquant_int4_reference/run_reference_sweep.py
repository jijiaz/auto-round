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
"""Steps 3 and 5: FLUX.1-dev controls and the model-level INT4 block-size sweep.

The script keeps the whole reference path in PyTorch: it quantizes in memory
with ``AutoRound.quantize()`` and evaluates the resulting pipeline with
``auto_round.compressors.diffusion.eval.diffusion_eval``. It never calls the
``svdquant_nunchaku`` exporter (restricted to MXFP4 group size 32) and never
calls an ARK INT4 kernel; ARK parity and performance are separate follow-up
gates.

Requires a FLUX-capable accelerator (Intel XPU is the validated SVDQuant path;
CUDA works as well) plus ``diffusers``, ``torchmetrics`` and ``image-reward``.

Examples::

    # Step 3: fresh controls in this environment
    python run_reference_sweep.py --stage controls \
        --model /models/FLUX.1-dev --prompt-file coco2017_captions.tsv \
        --output-dir records

    # Step 5: full model-level sweep around the screened boundary
    python run_reference_sweep.py --stage sweep --weight-blocks 64 128 256 \
        --model /models/FLUX.1-dev --prompt-file coco2017_captions.tsv \
        --output-dir records
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from .contract import (
        FIXED_CONTRACT,
        Candidate,
        ExperimentManifest,
        environment_revisions,
        w4a4_candidates,
        weight_only_candidates,
    )
except ImportError:  # executed as a standalone script
    from contract import (
        FIXED_CONTRACT,
        Candidate,
        ExperimentManifest,
        environment_revisions,
        w4a4_candidates,
        weight_only_candidates,
    )


@dataclass
class RunSpec:
    """One evaluated configuration."""

    name: str
    method: str  # bf16 | mxfp4 | int4
    scheme: object | None  # QuantizationScheme or preset name; None for BF16
    smooth: bool
    terminal: str  # rtn | signround
    weight_block: object | None = None
    act_group_size: int | None = None


def _parse_block(value: str):
    """Parse ``64`` (K-axis) or ``32x64`` (M-by-K block) from the command line."""
    if "x" in value:
        m, k = value.split("x", 1)
        return (int(m), int(k))
    return int(value)


def build_specs(args) -> list[RunSpec]:
    specs: list[RunSpec] = []
    if args.stage in ("controls", "all"):
        specs.append(RunSpec("bf16", "bf16", None, smooth=False, terminal="none"))
        specs.append(RunSpec("mxfp4_smooth_signround", "mxfp4", "MXFP4", smooth=True, terminal="signround"))
        specs.append(RunSpec("mxfp4_nosmooth_signround", "mxfp4", "MXFP4", smooth=False, terminal="signround"))
        if args.include_rtn_controls:
            specs.append(RunSpec("mxfp4_smooth_rtn", "mxfp4", "MXFP4", smooth=True, terminal="rtn"))
            specs.append(RunSpec("mxfp4_nosmooth_rtn", "mxfp4", "MXFP4", smooth=False, terminal="rtn"))

    if args.stage in ("sweep", "all"):
        blocks = [_parse_block(value) for value in args.weight_blocks] if args.weight_blocks else None
        candidates: list[Candidate] = weight_only_candidates(blocks)
        if args.act_bits == 4:
            candidates = w4a4_candidates(candidates[0].weight_block)
        for candidate in candidates:
            for terminal in args.terminals:
                specs.append(
                    RunSpec(
                        name=f"int4_{candidate.name}_{'smooth' if args.smooth else 'nosmooth'}_{terminal}",
                        method="int4",
                        scheme=candidate.to_scheme(),
                        smooth=args.smooth,
                        terminal=terminal,
                        weight_block=candidate.weight_block,
                        act_group_size=candidate.act_group_size,
                    )
                )
    return specs


def build_autoround(spec: RunSpec, args):
    """Construct AutoRound for one spec using the fixed experiment contract."""
    from auto_round import AutoRound
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    from auto_round.algorithms.transforms.svdquant import SVDQuantConfig

    terminal_config = (
        RTNConfig(disable_opt_rtn=True)
        if spec.terminal == "rtn"
        else SignRoundConfig(iters=FIXED_CONTRACT["signround_iters"])
    )
    alg_configs = [
        SVDQuantConfig(
            rank=FIXED_CONTRACT["rank"],
            smooth_enabled=spec.smooth,
            residual_iters=FIXED_CONTRACT["residual_iters"],
            low_rank_dtype=FIXED_CONTRACT["low_rank_dtype"],
            model_adapter="flux",
        ),
        terminal_config,
    ]
    return AutoRound(
        args.model,
        scheme=spec.scheme,
        model_dtype=FIXED_CONTRACT["model_dtype"],
        alg_configs=alg_configs,
        dataset=args.prompt_file,
        nsamples=FIXED_CONTRACT["calibration_samples"],
        batch_size=FIXED_CONTRACT["batch_size"],
        num_inference_steps=FIXED_CONTRACT["num_inference_steps"],
        iters=FIXED_CONTRACT["signround_iters"] if spec.terminal == "signround" else 0,
        device_map=args.device,
        low_gpu_mem_usage=True,
    )


def load_bf16_pipeline(args):
    from diffusers import AutoPipelineForText2Image

    return AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)


def evaluate(pipe, spec: RunSpec, args, seed: int) -> dict[str, float]:
    """Run ``diffusion_eval`` and return the three model-level metrics."""
    from auto_round.compressors.diffusion.eval import diffusion_eval

    image_dir = Path(args.output_dir) / "images" / f"{spec.name}_seed{seed}"
    image_dir.mkdir(parents=True, exist_ok=True)
    gen_kwargs = {
        "height": FIXED_CONTRACT["height"],
        "width": FIXED_CONTRACT["width"],
        "guidance_scale": FIXED_CONTRACT["guidance_scale"],
        "num_inference_steps": FIXED_CONTRACT["num_inference_steps"],
        "generator": torch.Generator(device="cpu").manual_seed(seed),
    }
    result = diffusion_eval(
        pipe,
        prompt_file=args.prompt_file,
        metrics=FIXED_CONTRACT["metrics"],
        image_save_dir=str(image_dir),
        batch_size=FIXED_CONTRACT["batch_size"],
        gen_kwargs=gen_kwargs,
        limit=args.limit,
    )
    # ``diffusion_eval`` prints a table and (depending on the version) returns a
    # dict; fall back to parsing the returned mapping defensively.
    if isinstance(result, dict):
        return {key: float(value) for key, value in result.items()}
    raise RuntimeError(
        "diffusion_eval did not return a metric mapping; capture the printed table "
        "and record it manually before continuing the sweep."
    )


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "model_level_records.jsonl"

    manifest = ExperimentManifest(
        stage=f"step{'3' if args.stage == 'controls' else '5'}_{args.stage}",
        extra={
            "model": args.model,
            "prompt_file": args.prompt_file,
            "limit": args.limit,
            "device": str(args.device),
            "seeds": args.seeds,
            "revisions": environment_revisions(),
        },
    ).to_dict()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for spec in build_specs(args):
        for seed in args.seeds:
            started = time.time()
            if spec.method == "bf16":
                pipe = load_bf16_pipeline(args)
            else:
                autoround = build_autoround(spec, args)
                autoround.quantize()
                pipe = autoround.pipe if hasattr(autoround, "pipe") else autoround.model
            metrics = evaluate(pipe, spec, args, seed)
            peak_memory = None
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            record = {
                "run": spec.name,
                "method": spec.method,
                "smooth": spec.smooth,
                "terminal": spec.terminal,
                "weight_block": list(spec.weight_block) if isinstance(spec.weight_block, tuple) else spec.weight_block,
                "act_group_size": spec.act_group_size,
                "seed": seed,
                "metrics": metrics,
                "wall_time_s": round(time.time() - started, 2),
                "peak_memory_bytes": peak_memory,
                "manifest": manifest["fixed_contract"],
            }
            with records_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nrecords written to {records_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["controls", "sweep", "all"], default="all")
    parser.add_argument("--model", default=FIXED_CONTRACT["model"])
    parser.add_argument("--prompt-file", required=True, help="COCO2017 caption tsv (id + caption columns)")
    parser.add_argument("--output-dir", default="records")
    parser.add_argument("--device", default="xpu:0" if hasattr(torch, "xpu") else "cuda:0")
    parser.add_argument("--limit", type=int, default=FIXED_CONTRACT["eval_limit"])
    parser.add_argument("--seeds", type=int, nargs="+", default=FIXED_CONTRACT["seeds"])
    parser.add_argument(
        "--weight-blocks",
        nargs="+",
        help="block sizes to evaluate, e.g. `64 128 256` or `32x64 64x64`; defaults to every candidate",
    )
    parser.add_argument("--act-bits", type=int, default=16, choices=[4, 16])
    parser.add_argument("--smooth", action="store_true", default=True)
    parser.add_argument("--no-smooth", dest="smooth", action="store_false")
    parser.add_argument("--terminals", nargs="+", default=["signround"], choices=["rtn", "signround"])
    parser.add_argument("--include-rtn-controls", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
