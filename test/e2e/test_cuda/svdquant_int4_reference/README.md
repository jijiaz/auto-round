# FLUX.1-dev SVDQuant INT4 Reference Accuracy Study

PyTorch reference study that determines the largest INT4 scale-sharing block
size that still satisfies model-level quality for SVDQuant on FLUX.1-dev,
before any packing, export, or kernel work is funded.

Everything here stays in PyTorch and reuses the production AutoRound pieces:
the SVDQuant transform (`auto_round/algorithms/transforms/svdquant/`), the
registered quantize/dequantize functions (`auto_round/data_type/`), and the
diffusion metrics (`auto_round/compressors/diffusion/eval.py`). No ARK INT4
kernel and no `svdquant_nunchaku` export round trip is used as a substitute for
the reference accuracy result.

## Contents

| File | Stage | Purpose |
|---|---|---|
| `contract.py` | 1, 6 | Frozen INT4 numerical contract, fixed experiment settings, candidate block sizes, pre-registered acceptance gates, run manifest |
| `test_int4_reference_contract.py` | 1, 2 | Contract tests and the SVDQuant workflow integration test (CPU, sliced FLUX block) |
| `proxy_block_sweep.py` | 4 | Cheap layer-level screening sweep that locates the transition region |
| `run_reference_sweep.py` | 3, 5 | Fresh BF16/MXFP4 controls and the model-level INT4 block-size sweep on FLUX.1-dev |
| `aggregate_results.py` | 6 | Applies the pre-registered gates and selects the maximum passing block size |

Generated checkpoints, images, and record files are not committed
(`records/` and `images/` are ignored).

## Step 1: the INT4 numerical contract

The reference QDQ has to be the proposed kernel, not a generic INT4
interpretation, so the contract is stated explicitly in
`contract.py::INT4_CONTRACT` and implemented by two registered functions:

| Property | Value |
|---|---|
| Code range | signed `[-8, 7]` (full range) |
| Scale | `scale = -sign(dominant) * max(abs(w_min), abs(w_max)) / 8`, magnitude clamped to `1e-5` |
| Scale dtype | FP16 by default, materialized before use |
| Rounding | round-half-to-even, then clamp (clamp never feeds back into the scale) |
| Weight axes | `(M, K)` over `(out_features, in_features)`; `K` is the reduction axis |
| Activation axes | 1-D `K` groups, dynamic per token |
| Coupling | weight and activation block sizes are independent and scanned separately |
| Padding | zero padding of partial blocks; padding never widens a block range and is cropped |
| Zero block | scale clamped to `+1e-5`, all codes 0, output exactly 0 |
| Scale layout | 2-D `(ceil(M_out/M), ceil(K/K_block))` for `block_int_sym`; 1-D `K` groups for `int_sym` |

`auto_round/data_type/int.py` gained `block_int_sym` (aliases `block_int_sym`,
`block_int4_sym`) for M-by-K scales. It is **bit-exact with the existing
`int_sym`** when the block height is 1, so the K-axis and block layouts are one
contract rather than two, and the semantics of the existing `int` datatype were
not changed. `get_quant_func` now resolves the `block_*` implementations for a
tuple `group_size` *before* the `rtn_*`/`opt_rtn_*` aliases, whose K-axis
semantics do not match a 2-D block contract.

Unit tests: `test/unit/test_cpu/data_type/test_block_int.py` (exact
hand-computed tensors, zero blocks, non-divisible dimensions, FP16/BF16/FP32
scale materialization, and every candidate block shape).

## Step 2: workflow integration

`test_int4_reference_contract.py` builds AutoRound's algorithm composer with

```
SVDQuantConfig(rank=..., smooth_enabled=..., residual_iters=..., low_rank_dtype="bf16", model_adapter="flux")
RTNConfig(disable_opt_rtn=True)
```

and a `QuantizationScheme` produced by `contract.Candidate.to_scheme()`, then
checks on a sliced FLUX-shaped block that the layer becomes an
`SVDQuantLinear`, the residual weight lies exactly on the INT4 grid (QDQ is
idempotent), `lora_down`/`lora_up` stay BF16 at the requested rank, the forward
is finite and shape-preserving, and the effective weight matches an
independently computed `Q(R) + U @ V`.

Run it on CPU:

```bash
PYTHONPATH=. pytest test/e2e/test_cuda/svdquant_int4_reference/ -q
PYTHONPATH=. pytest test/unit/test_cpu/data_type/test_block_int.py -q
```

## Step 4: coarse screening sweep (already executed)

`proxy_block_sweep.py` measures the *layer output* NMSE of `Q(R) + U @ V`
against BF16, averaged over the six distinct FLUX.1-dev linear shapes, using
the production `truncated_svd` / `iterate_residual_decomposition` /
`rtn_qdq_residual` primitives with rank 32 and 20 residual iterations.

Screening rule: MXFP4 group 32 is the only 4-bit SVDQuant configuration with
published FLUX.1-dev model-level numbers, and even its worst variant
(no smooth + RTN) satisfies the acceptance gates. An INT4 configuration is a
screening pass when its layer NMSE is **no worse than the matching MXFP4
group-32 control** (W4A16 candidates against the W4A16 control, W4A4 candidates
against the W4A4 control).

```bash
PYTHONPATH=. python test/e2e/test_cuda/svdquant_int4_reference/proxy_block_sweep.py \
    --shrink 4 --residual-iters 20 --scan-activations --output records/proxy.json
```

Result (geometric mean over the six shapes, ratio versus the matching MXFP4
group-32 control, lower is better):

| Weight scale layout | Elements per scale | NMSE ratio vs MXFP4 g32 | Screening |
|---|---:|---:|---|
| K-axis 16 | 16 | 0.43 | pass |
| K-axis 32 | 32 | 0.55 | pass |
| K-axis 64 | 64 | 0.68 | pass |
| K-axis 128 | 128 | 0.80 | pass |
| K-axis 256 | 256 | 0.93 | pass |
| K-axis 512 | 512 | 1.03 | fail (marginal) |
| K-axis 1024 | 1024 | 1.26 | fail |
| K-axis per-channel | >= 768 (full row) | 1.22 | fail |
| Block 16x16 | 256 | 1.02 | fail (marginal) |
| Block 16x32 | 512 | 1.18 | fail |
| Block 32x32 | 1024 | 1.34 | fail |
| Block 32x64 | 2048 | 1.50 | fail |
| Block 64x64 | 4096 | 1.66 | fail |
| Block 128x128 | 16384 | 2.00 | fail |

Two conclusions drive the rest of the study:

1. **1-D K-axis grouping dominates 2-D block scaling at equal metadata cost.**
   `K-axis 256` and `16x16` both share one scale per 256 weights, yet the
   K-axis layout is ~10% better in NMSE and lands on the passing side of the
   control. Rows of a DiT projection have very different dynamic ranges, so a
   block that spans 16 output channels pays for the widest row in the tile.
   A 2-D block contract is therefore not recommended for INT4 SVDQuant.
2. **The weight boundary sits between K = 256 and K = 512.** K = 256 is still
   slightly better than the MXFP4 control (0.93x); K = 512 is the first clear
   failure (1.03x) and everything above it degrades monotonically.

Two independent seeds agree to within 1% on every configuration, so the
ordering and the location of the boundary are stable screening results.

Activation scan (weight block fixed at K = 64, W4A4, screened against a *W4A4*
MXFP4 group-32 control, since the published MXFP4 SVDQuant configuration also
quantizes activations):

| Activation group | NMSE ratio vs MXFP4 g32 W4A4 | Screening |
|---|---:|---|
| 16 | 0.52 | pass |
| 32 | 0.60 | pass |
| 64 | 0.68 | pass |
| 128 | 0.76 | pass |
| per-token | 1.03 | fail (marginal) |

Activation quantization raises the absolute error by roughly 3x relative to
W4A16 (2.0e-2 versus 7.0e-3 for the controls), but INT4 activations are
*relatively* better than MXFP4 activations at every group size up to 128. The
activation boundary is therefore also between 128 and per-token, and W4A4
remains the higher-risk path only in absolute terms.

## Steps 3 and 5: controls and the model-level sweep

These require a FLUX-capable accelerator (Intel XPU is the validated SVDQuant
path; CUDA works too) plus `diffusers`, `torchmetrics` and `image-reward`, and
they are **not** run in this repository's CI.

```bash
# Step 3: fresh BF16 + MXFP4 controls in the target environment
PYTHONPATH=. python test/e2e/test_cuda/svdquant_int4_reference/run_reference_sweep.py \
    --stage controls --model /models/FLUX.1-dev \
    --prompt-file coco2017_captions.tsv --output-dir records --include-rtn-controls

# Step 5: model-level sweep around the screened boundary
PYTHONPATH=. python test/e2e/test_cuda/svdquant_int4_reference/run_reference_sweep.py \
    --stage sweep --weight-blocks 128 256 512 --terminals signround \
    --model /models/FLUX.1-dev --prompt-file coco2017_captions.tsv --output-dir records
```

Fixed settings (`contract.py::FIXED_CONTRACT`): BF16 model and low-rank dtype,
rank 32, 128 COCO2017 captions, 50 inference steps, 200 SignRound iterations,
20 residual iterations, 1024x1024, guidance 3.5, batch size 1, shared prompts
and per-prompt seeds across BF16, MXFP4 and every INT4 block size. Each run
stores its metrics, deltas, wall time, peak memory, resolved configuration and
software revisions.

The primary feasibility configuration is smooth SVDQuant + SignRound. If the
product path is RTN, run no-smooth + RTN at the final boundary as well; RTN
feasibility must not be inferred from a SignRound result.

## Step 6: acceptance gates

Gates are pre-registered in `contract.py` and must not be edited after seeing
sweep results.

Absolute floor (worst published MXFP4 SVDQuant RTN result, a conservative
provisional envelope until dedicated INT4 thresholds are approved):

```
CLIP        >= 25.9624
CLIP-IQA    >= 0.946939
ImageReward >= 0.934579
```

Relative budget against the *fresh* BF16 control (preferred once controls are
reproducible): CLIP `>= -0.10`, CLIP-IQA `>= -0.010`, ImageReward `>= -0.09`.

A candidate passes only when all three metrics pass on the aggregate **and** the
worst repeated run also clears the absolute floor. A failed metric is never
averaged away with a stronger unrelated metric.

```bash
PYTHONPATH=. python test/e2e/test_cuda/svdquant_int4_reference/aggregate_results.py \
    records/model_level_records.jsonl --output records/summary.json
```

The script prints the block-size comparison table, checks monotonicity of the
smaller sizes, and emits the selected maximum passing block size.

## Current recommendation

Screening evidence (Step 4), pending the model-level confirmation of Step 5:

* **Recommended INT4 weight contract: 1-D K-axis groups, `group_size = 128`.**
  It is 0.80x the MXFP4 group-32 control error, leaving a comfortable margin
  below the screened boundary at K = 256, and 128 is the standard tile-friendly
  K granularity for INT4 GEMM kernels. K = 64 (0.68x) is the conservative
  fallback if the model-level run at 128 is marginal.
* **Screened upper bound: K = 256**; K = 512 is the first failure. Step 5 should
  evaluate 256 and 512 (plus 128 as the smaller control) and, if a 2-D contract
  is still under consideration, 16x16.
* **2-D M-by-K block scaling is not recommended**: at equal metadata cost it is
  strictly worse than K-axis grouping, and even 16x16 already screens as a
  marginal failure.
* **W4A4**: dynamic INT4 activations with a group of 128 or smaller screen
  better than the matching MXFP4 W4A4 control; per-token activation scales do
  not. Activations still contribute ~3x more absolute error than weights, so
  the W4A4 boundary must be confirmed at model level before it is promised.

## Step 7: go/no-go inputs

For each passing block size, the decision needs INT4 weight bytes and scale
metadata bytes, dynamic activation scale traffic, residual INT4 GEMM cost, the
BF16 low-rank down/up GEMM and intermediate-tensor cost, hardware tile and
memory-layout compatibility, expected bandwidth/occupancy/latency benefit, and
the export/loader/backend/parity implementation cost.

With K = 128 the scale metadata is 1/128 of a FP16 value per weight
(~0.125 bit/weight on top of 4 bits), versus 1/32 of an E8M0 exponent
(~0.25 bit/weight) for MXFP4 group 32, so the accuracy-passing INT4 point is
*cheaper* in metadata than the existing MXFP4 path and uses a K granularity
that maps onto standard INT4 GEMM tiles. That is a favourable go signal for a
W4A16 INT4 SVDQuant kernel, but it stays provisional until the FLUX.1-dev
model-level sweep confirms the boundary on real hardware, since a small passing
block that erased the memory or throughput benefit would be an accuracy success
and a product no-go.
