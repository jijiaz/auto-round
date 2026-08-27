---
description: "Use when implementing or running the FLUX.1-dev SVDQuant INT4 PyTorch reference accuracy study, block-size sweeps, model-level quality evaluation, or go/no-go analysis."
name: "SVDQuant INT4 Reference Accuracy"
applyTo: "test/e2e/test_cuda/svdquant_int4_reference/**"
---

# SVDQuant INT4 Reference Accuracy Workflow

## Purpose

Determine the largest INT4 scale-sharing block size that satisfies model-level
quality requirements for SVDQuant before investing in packing, export, or
runtime kernels.

Keep the reference path in PyTorch. Reuse AutoRound's SVDQuant transform,
quantization registry, diffusion calibration, and diffusion quality metrics.
Do not use kernel throughput or an export round trip as a substitute for the
reference accuracy result.

The primary scope is FLUX.1-dev because it is the only SVDQuant end-to-end path
currently validated by AutoRound.

## Existing Reference Points

Reuse these implementations instead of duplicating them:

- SVD decomposition and residual iteration:
  `auto_round/algorithms/transforms/svdquant/residual.py`
- Structural transform and projection grouping:
  `auto_round/algorithms/transforms/svdquant/apply.py`
- PyTorch residual plus low-rank forward:
  `auto_round/algorithms/transforms/svdquant/wrapper.py`
- Registered INT and MXFP quantize-dequantize functions:
  `auto_round/data_type/`
- Diffusion generation and CLIP, CLIP-IQA, and ImageReward evaluation:
  `auto_round/compressors/diffusion/eval.py`
- Existing SVDQuant configurations and measured quality:
  `docs/step_by_step.md` and `docs/svdquant_details.md`

The published in-repository FLUX.1-dev reference values are:

| Configuration | CLIP | CLIP-IQA | ImageReward |
|---|---:|---:|---:|
| BF16 | 26.0189 | 0.954360 | 1.018340 |
| MXFP4, smooth + SVDQuant + SignRound | 26.1039 | 0.962655 | 1.021020 |
| MXFP4, no smooth + SVDQuant + SignRound | 26.0727 | 0.959363 | 1.002380 |
| MXFP4, smooth + SVDQuant + RTN | 25.9719 | 0.947763 | 0.939392 |
| MXFP4, no smooth + SVDQuant + RTN | 25.9624 | 0.946939 | 0.934579 |

Treat these values as historical reference points, not automatically
reproducible baselines. Record a fresh BF16 and MXFP4 control in the same
environment as every full INT4 sweep.

## Fixed Experiment Contract

Unless the experiment explicitly studies one of these variables, keep the
following settings fixed:

- Model: `black-forest-labs/FLUX.1-dev`
- Model and low-rank dtype: BF16
- Low-rank rank: 32
- Calibration samples: 128 COCO2017 captions
- Inference steps: 50
- SignRound tuning iterations: 200
- Residual iterations: 20
- Metrics: CLIP, CLIP-IQA, and ImageReward
- Prompt file, prompt order, generation seeds, image dimensions, guidance
  scale, scheduler, batch size, and software revisions

Use the same prompts and per-prompt seeds for BF16, MXFP4 controls, and every
INT4 block size. Save the resolved configuration and revisions with each run.
Before claiming direct comparability with the historical values, recover and
match any settings not documented above, especially the evaluation prompt set,
sample limit, image dimensions, guidance scale, and seeds.

## Step 1: Freeze the INT4 Numerical Contract

**Goal:** Make the PyTorch QDQ exactly represent the proposed kernel rather
than a generic interpretation of INT4.

Document and test:

- Signed code range, for example `[-8, 7]` or `[-7, 7]`
- Scale equation and scale dtype
- Rounding mode and clamping order
- Weight and activation block axes
- Whether weight and activation block sizes are coupled
- Padding and partial-block behavior
- Zero-block, NaN, infinity, underflow, and overflow behavior
- Whether scales are one-dimensional K groups or two-dimensional M-by-K blocks

Reuse `int.py` for ordinary K-axis symmetric INT4 only when its behavior
matches the kernel contract. For M-by-K scaling, add and register a dedicated
`block_int` PyTorch QDQ implementation. Do not alter the semantics of the
existing `int` datatype for this experiment.

**Completion criterion:** Unit tests cover exact hand-computed tensors, zero
blocks, non-divisible dimensions, BF16/FP16 scale materialization, and all
candidate block shapes.

## Step 2: Prove SVDQuant Workflow Integration

**Goal:** Confirm that the proposed INT4 QDQ is applied only to the SVDQuant
residual branch while the low-rank branch remains BF16.

Construct AutoRound with:

```python
alg_configs=[
    SVDQuantConfig(
        rank=32,
        smooth_enabled=...,
        residual_iters=20,
        low_rank_dtype="bf16",
        model_adapter="flux",
    ),
    RTNConfig(disable_opt_rtn=True),
]
```

Build a `QuantizationScheme` for each candidate block size and pass it directly
to AutoRound. Use `quantize()` and evaluate the in-memory pipeline. Do not call
the `svdquant_nunchaku` exporter: it is intentionally restricted to MXFP4
group size 32 and is not part of this reference study.

Check representative transformed layers for:

- `SVDQuantLinear` replacement
- INT4-QDQ residual weights
- Dynamic INT4 activation QDQ when W4A4 is being studied
- BF16 `lora_down` and `lora_up`
- Finite outputs and unchanged model output shapes

**Completion criterion:** A sliced FLUX unit/integration case completes an
in-memory forward and numerically matches an independently computed
`Q(R) + U @ V` reference.

## Step 3: Reproduce Controls

**Goal:** Establish that the environment and evaluation harness are capable of
reproducing meaningful model-level results before scanning INT4.

Run, in the same environment:

1. BF16
2. MXFP4, smooth + SVDQuant + SignRound
3. MXFP4, no smooth + SVDQuant + SignRound
4. Optionally, both existing RTN configurations for diagnostic comparison

Use `diffusion_eval()` for all three quality metrics. Also retain basic image
sanity checks from `test_diffusion_quantize_e2e.py`, but never treat image
shape, dtype, or standard deviation as quality acceptance.

Compare fresh controls with the historical values above. Investigate material
differences before running the expensive sweep; do not tune thresholds to hide
a control mismatch.

**Completion criterion:** BF16 and at least one MXFP4 control are stable across
repeated runs and the cause of any difference from the historical values is
documented.

## Step 4: Run a Coarse Block-Size Sweep

**Goal:** Locate the transition region cheaply before running the full
model-level matrix.

Start with kernel-valid K-axis candidates such as:

```text
16, 32, 64, 128, 256, per-channel
```

For a two-dimensional scale contract, scan the hardware-valid M-by-K shapes
instead. Keep M fixed while finding the K limit, then scan M around the
surviving K values.

Use a reduced but fixed prompt subset and fewer generation steps only for this
coarse stage. Record these results as screening data, never as final accuracy.
Run each candidate with identical prompts and seeds. Scan one variable at a
time:

1. Weight block size with activation block size fixed
2. Activation block size with weight block size fixed
3. A small joint matrix around the surviving boundary

**Completion criterion:** Identify the largest likely passing block size and
the first clearly failing size for the full evaluation.

## Step 5: Run the Full Model-Level Sweep

**Goal:** Determine the supported block-size boundary using the established
SVDQuant quality standard.

Run the fresh BF16 control, fresh MXFP4 control, and candidate INT4 sizes using
the fixed experiment contract. Evaluate at least the two sizes around the
coarse boundary, plus one smaller control size. Use repeated paired seeds when
resources permit and report the mean, standard deviation, and worst run.

The primary feasibility configuration is smooth SVDQuant plus SignRound,
matching the strongest existing AutoRound result. Also run no-smooth plus RTN
at the final candidate boundary if the intended product path is RTN; do not
infer RTN feasibility solely from a SignRound result.

Store one structured record per model, method, block size, seed, and metric.
Include raw metric values, deltas from the fresh BF16 run, deltas from the
fresh MXFP4 control, wall time, peak memory, and complete configuration.

**Completion criterion:** Every final candidate has complete CLIP, CLIP-IQA,
and ImageReward results from the same prompt/seed set.

## Step 6: Apply Acceptance Gates

**Goal:** Select the maximum block size without changing criteria after seeing
the sweep.

Use the matching fresh MXFP4 configuration as the primary quality control. A
candidate passes only if all three metrics satisfy their pre-registered
thresholds on the aggregate result and no repeated run is a clear outlier
failure.

Until dedicated INT4 thresholds are approved, use the existing validated
MXFP4 RTN envelope as a conservative provisional floor:

```text
CLIP        >= 25.9624
CLIP-IQA    >= 0.946939
ImageReward >= 0.934579
```

This envelope corresponds to the worst existing MXFP4 SVDQuant RTN result
relative to the in-repository BF16 reference. Prefer fresh-run relative gates
once controls are reproducible. Record the approved absolute and relative
gates in the experiment configuration before the full sweep.

The block-size upper bound is the largest tested size for which:

- All three model-level quality gates pass
- All outputs are finite and image sanity checks pass
- The result is reproducible with the required seeds
- All smaller hardware-valid sizes expected to be monotonic have either passed
  or any non-monotonic result has been rerun and explained

Do not average a failed metric away using a stronger unrelated metric.

## Step 7: Make the Kernel Go/No-Go Decision

**Goal:** Decide whether the surviving accuracy region offers enough system
value to justify production development.

For every passing block size, estimate:

- INT4 weight bytes and scale metadata bytes
- Dynamic activation scale traffic
- Residual INT4 GEMM cost
- BF16 low-rank down/up GEMM and intermediate-tensor cost
- Hardware tile and memory-layout compatibility
- Expected bandwidth, occupancy, latency, and throughput benefit
- Export, loader, backend, and parity-test implementation cost

Recommend kernel development only when at least one passing block size is
hardware-efficient and materially improves the target workload over the
existing MXFP4 path. A small passing block that erases memory or throughput
benefits is an accuracy success but a product no-go.

## Required Outputs

Keep generated checkpoints and images out of git. The working directory should
contain only reusable scripts, tests, and small configuration files. Produce:

- A machine-readable experiment manifest
- Raw per-seed metric records
- An aggregated block-size comparison table
- The selected maximum passing block size
- A documented go/no-go recommendation and its performance assumptions

Any new Python, YAML, or shell file must carry the repository Apache 2.0
header. Any committed Markdown change must include the equivalent `_CN`
translation.
