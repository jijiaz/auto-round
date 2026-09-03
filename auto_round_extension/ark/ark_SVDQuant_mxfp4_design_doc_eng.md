# ARK MXFP4 SVDQuant Two-Kernel Design

## 1. Goal and scope

The inference path is split into two fused kernels:

1. **Kernel A: smoothing, dynamic MXFP4 activation quantization, and low-rank down projection.**
2. **Kernel B: MXFP4 W4A4 GEMM and low-rank up projection.**

The current B60 machine cannot execute the target MXFP4 SIMD/matrix instructions. The development plan therefore completes and validates Kernel A first. Kernel B development on B60 completes all instruction-independent components and prepares the native mainloop boundary, tests, and benchmarks, without requiring an end-to-end native Kernel B.

Offline SVD decomposition, smoothing search, calibration, export, and graph rewriting are outside the main scope except for providing stable kernel inputs.

## 2. Computation split

```text
Xh = X * S
QX, SX = MXFP4_RCEIL_QUANT(Xh)
L  = Xh @ Ld.T
Y  = MXFP4_GEMM(QX, SX, Wq, SW) + L @ Lu.T + bias

Kernel A: X, S, Ld -> QX, SX, L
Kernel B: QX, SX, Wq, SW, L, Lu, bias -> Y
```

This follows the SVDQuant/Nunchaku boundary: activation quantization and low-rank down reuse the same smoothed input, while W4A4 compute and low-rank up share the output tile.

## 3. Kernel A: quantize + low-rank down

### 3.1 Fused operations

One launch performs:

1. Load `X[M,K]`.
2. Compute `Xh = X * smooth`.
3. Compute one dynamic MXFP4 scale per 32 contiguous K values.
4. Apply the AutoRound rceil E2M1 contract and pack two codes per byte.
5. Store UE8M0 activation scales.
6. Compute `L = Xh @ lora_down.T` from the same input tile.

Kernel A does not require an MXFP4 matrix instruction. Its primary gain is removing duplicate input reads, smoothing, and intermediates, so it can be completed on B60.

### 3.2 Logical API

```cpp
void svdquant_mxfp4_quant_down(
    Tensor x, Tensor smooth, Tensor lora_down,
    Tensor qact, Tensor ascales, Tensor lora_act,
    Queue current_queue);
```

V1 constraints:

- `x`: `[M,K]`, BF16/FP16; `K > 0` and `K % 32 == 0`.
- `smooth`: `[K]`, BF16/FP16/FP32, optional.
- `lora_down`: `[R,K]`, BF16/FP16.
- `qact`: `[M,K/2]`, low-nibble-first logical E2M1.
- `ascales`: `[M,K/32]`, UE8M0.
- `lora_act`: `[M,R]`, **same dtype as `x`** (BF16 or FP16). Accumulation stays FP32; only the
  store narrows.

  This was revised in 2026-09. The contract used to be FP32, justified as "avoiding extra low
  precision error between the two kernels", but that justification does not hold: §5.1 step 4 has
  Kernel B compute `lora_act @ lora_up.T`, which is a 16-bit DPAS GEMM, so the first thing Kernel B
  does is round the FP32 input back to 16-bit. The extra precision is discarded at the kernel
  boundary and only costs bandwidth. It was also internally inconsistent: `lora_down` was already
  16-bit on the input side (matching the paper's 16-bit low-rank branch) while the output was FP32.
  Following `x`'s dtype additionally keeps this consistent with the existing "`lora_down` dtype must
  equal `x` dtype" rule, and is exactly the operand type Kernel B's GEMM wants.
- Rank 32 is the first optimized path; multiples of 16 remain extensible.
- The current PyTorch XPU queue is used without implicit host synchronization.

### 3.2.1 Aligning with existing ARK engineering conventions

Kernel A's Python/C++ boundary should reuse the conventions already present in
`auto_round_extension/ark` rather than invent new ones, to keep review cost low and stay consistent
with other kernels in this repo:

- **Queue retrieval**: reuse `get_stream(tensor)` from `auto_round_kernel/__init__.py` (derives
  `torch.xpu.current_stream().sycl_queue` from the input tensor's device); do not define a separate
  queue-passing convention for Kernel A.
- **Dtype enum**: reuse `ARK_DT` (already includes `float8_e8m0`, directly usable for `ascales`). Add
  E2M1/MXFP4 code constants to this enum instead of introducing a parallel dtype encoding.
- **Python-side shape/dtype validation**: reuse the `_validate_packed_blob`-style pre-checks found in
  `qlinear.py`/`__init__.py` to re-verify the ABI contract (Section 4.2) in Python before calling the
  native kernel, instead of relying solely on C++ assertions.
- **Extension loading**: reuse `ensure_xpu_lib`/`load_xpu_lib` from `xpu_loader.py`; the new kernel's
  `.so` is exposed to Python through the same `required_symbols` mechanism, no new loader.

### 3.3 Implementation rules

- Threads cooperate on each K-group `amax`.
- Compute the exponent per the frozen contract in 3.3.1 and clamp to `[-127, 127]`, so UE8M0 codes land
  in `[0, 254]` and reserved code 255 is never emitted.
- Match AutoRound `quant_mx_rceil` scale and code rounding bit-for-bit.
- Reuse the smoothed tile from registers/SLM for packing and down projection.
- Accumulate down projection in FP32.
- Mask M/K tails; padding must not affect scale reduction.
- Do not freeze Kernel-B-specific swizzles, target instruction tiles, or speculative subgroup settings on B60.

### 3.3.1 Frozen numerical contract (A0)

This section is the **sole** numerical authority for Kernel A, frozen before implementation began. It is
equivalent to `quant_mx_rceil(..., bits=4, group_size=32, data_type="mx_fp4e2m1")` plus the
`svdquant_mxfp4.py` codecs. Every implementation (SYCL or PyTorch reference) must reproduce these steps
bit-for-bit.

**Step 1 — smooth**: `xh = float32(x) * float32(smooth[k])`; a `None` smooth is equivalent to
multiplying by 1. All subsequent math is done in FP32.

**Step 2 — group amax**: `amax = max(|xh|)` over each 32 contiguous K elements.

**Step 3 — shared exponent** (note the zero-group and degenerate cases):

```text
shared_exp = (amax == 0) ? 1.0 : ceil( fp32( log2(amax / 6.0) ) )
shared_exp = clamp(shared_exp, -127, 127)
scale      = 2^shared_exp
ue8m0_code = shared_exp + 127          // lands in [0, 254]
```

> **The zero-group special case is intentional.** When `amax == 0`, `quant_mx_rceil` emits
> `shared_exp = 1` (scale `2.0`, UE8M0 code `128`), not 0/code 127. This originates from mxfp.py's
> `torch.where(max_val == 0, torch.ones_like(max_val), ...)` using 1 as a placeholder, and the existing
> exporter `svdquant_mxfp4.py::pack_residual` has already inherited and serialized that behavior. ARK
> chooses to **align strictly with the oracle and exporter** (option A) so the §4.1 "bit-exact" gate
> needs no carve-out. All E2M1 codes in a zero group are 0, so the dequantized result is independent of
> the exponent; this choice affects only the `ascales` byte, not end-to-end numerics. Implementation
> cost is identical either way.

> **Degenerate underflow (the single intentional deviation from the oracle).** When an entire group is
> subnormal enough that `amax / 6.0` underflows to zero in FP32 (roughly `amax <= 4.9e-45`), `log2`
> yields `-inf` and the clamp should give `-127` (code 0). The oracle instead returns **NaN**, because
> `ceil_ste(x) = (x.ceil() - x).detach() + x` evaluates `-inf - (-inf)` -- a straight-through-estimator
> artifact leaking into the forward value, which then turns the whole group's qdq into NaN. The existing
> exporter has no usable contract here either (`encode_ue8m0(NaN)` silently degrades to code 127, while
> `encode_e2m1` raises ValueError). ARK takes the value the clamp was written to produce, `-127`: every
> code in such a group is 0 regardless, the dequantized result is 0, and NaN is never propagated
> downstream. This regime is essentially unreachable with real activations (it needs all 32 values in a
> group to be `< 5e-45`).

> **`ceil(log2(.))` must be evaluated as "round the real logarithm to FP32, then ceil".** The integer
> shortcut of taking `floor(log2)` from the exponent bits and adding one for non-powers-of-two is
> **not** equivalent: for `v` just above `2^e`, the true logarithm `e + 1.7e-7` rounds back to exactly
> `e` whenever `ulp(e)` exceeds that offset, which holds for `|e| >= 4` -- almost always. The integer
> shortcut disagreed with the oracle at 246 of 254 `nextafter(2^e)` sample points. The reliable
> engineering form is to **evaluate log2 in double and narrow to FP32**: this matched torch bit-for-bit
> across 16M adversarial samples clustered around powers of two, and torch's own `log2` agrees between
> CPU and XPU on that same set, so the contract is device independent and reproducible.

**Step 4 — normalize and clamp**: `t = clamp(xh / scale, -6.0, 6.0)`.

**Step 5 — E2M1 quantization (round-half-to-even on the code index)**:
The E2M1 magnitude codebook is `{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}` (codes 0..7).
The implementation **must reproduce the oracle's FP32 operation sequence bit-for-bit** rather than
comparing against midpoint thresholds:

```text
pe   = max(floor(log2(a)), 0)          // a = |t|; pe in {0,1,2}, taken from FP32 exponent bits
s    = a * 2^(1 - pe)
q    = floor(s + 0.5) - ((s - 0.5) mod 2 == 0)      // round-half-to-even
code = q + 2 * pe
```

`code = q + 2*pe` holds because the oracle's magnitude is `q * 2^(pe-1)`, and enumerating the reachable
`(pe, q)` pairs maps that one-to-one onto the codebook index. Deriving `pe` from exponent bits rather
than `log2` is both faster and more reliable (GPU `log2` is only accurate to a few ULP and can
misclassify exact powers of two); zero and subnormal inputs have a zero exponent field, yielding `-127`,
which `max(., 0)` maps to 0 -- matching the oracle's `clip(min=min_exp)` with `min_exp == 0` for E2M1.

The mathematical intent is equivalent to the following table (**for understanding only; not an
implementation**):

| `a` range | code | magnitude |
|---|---:|---:|
| `[0, 0.25]` | 0 | 0.0 |
| `(0.25, 0.75)` | 1 | 0.5 |
| `[0.75, 1.25]` | 2 | 1.0 |
| `(1.25, 1.75)` | 3 | 1.5 |
| `[1.75, 2.5]` | 4 | 2.0 |
| `(2.5, 3.5)` | 5 | 3.0 |
| `[3.5, 5.0]` | 6 | 4.0 |
| `(5.0, 6.0]` | 7 | 6.0 |

> **Why the midpoint table cannot be used as the implementation.** `floor(s + 0.5)` itself rounds in
> FP32, so a value one ULP *below* a midpoint can still carry. For example `a = 0.24999998509` (one ULP
> below the 0.25 tie) makes `s + 0.5` land exactly on an FP32 tie that round-half-to-even resolves to
> `1.0`, so the oracle returns code 1 where the table comparison returns code 0. Such points are
> reachable with real data (roughly one per 1e7 elements for random input, so a FLUX layer of 14M
> elements hits several), so the kernel must reproduce the behavior for the §4.1 bit-exactness gate to
> stay carve-out free. The sequence above showed zero disagreement with `quant_element` over 600K
> structured samples (including a +-4 ULP neighborhood of every tie) and 3M random samples.

**Step 6 — sign bit**: `code |= (signbit(t) ? 8 : 0)`. The sign comes from the normalized value, and
`-0.0` has sign bit 1 (consistent with `torch.signbit` and `encode_e2m1`).

**Step 7 — packing**: low-nibble-first, `qact[m, j] = code[m, 2j] | (code[m, 2j+1] << 4)` (matching
`svdquant_mxfp4.py::pack_nibbles`).

**Step 8 — low-rank down**: `lora_act[m, r] = sum_k xh[m, k] * float32(lora_down[r, k])`, FP32
accumulation, stored in `x`'s dtype. Note this multiplies the **smoothed but unquantized** `xh`, not
the dequantized value.

The narrowing happens **only at the store**: the accumulator is FP32 throughout, and the DPAS path's
tail epilogue goes through the same narrowing statement, so tiled rows and tail rows carry identical
rounding.

**Non-finite input**: behavior is undefined if `x`/`smooth`/`lora_down` contain NaN/Inf; debug builds
must raise explicitly and the release path performs no detection (see §4.1).

### 3.3.2 Implementation notes (Kernel A, implemented and validated)

This section records the engineering constraints found while implementing A0 that cannot be
derived from the contract itself. All three are of the "silently wrong if you don't know"
variety, so anyone rewriting Kernel A must read this first.

**File layout**

| File | Role |
|---|---|
| `auto_round_kernel/svdquant_mxfp4.py` | PyTorch reference for A0 plus the public entry point `svdquant_mxfp4_quant_down(backend=auto/ark/reference)`; doubles as the test oracle and the emulated fallback |
| `auto_round_kernel/wrapper/include/sycl_svdquant_mxfp4.hpp` | Device-side numerical primitives (`e2m1_magnitude_code`, `shared_exponent`, `divide_by_e2m1_max_norm`, `encode_group`) and the host entry declaration; Kernel B and the standalone SYCL tests reuse the same encode path |
| `auto_round_kernel/svdquant_mxfp4_kernel.cpp` | SYCL parallel decomposition and launcher |
| `ark.cpp` | `svdquant_mxfp4_quant_down` pybind binding (thin, pointer casts only) |
| `test/test_svdquant_mxfp4.py` | The section 4.1 / 4.2 correctness and ABI suite (see section 4.1) |
| `test/bench_svdquant_mxfp4.py` | The section 4.3 benchmark against the unfused three-step baseline (see section 4.3) |

**Finding 1: Intel GPU FP32 division is not IEEE correctly-rounded by default.**
The offload backend expands `fdiv` into a reciprocal approximation plus refinement, which can
land one ULP above the correctly-rounded quotient. When `amax / 6` falls near `6 * 2^e`, that one
ULP pushes `ceil(log2(.))` up a step and emits a UE8M0 code the oracle never produces (measured on
B60: `amax = 0x3cc00002` gives code 120 where the oracle gives 119).

- `-fp-model=precise` does **not** fix it: the division is expanded in the offload backend at link
  time, not in the front end.
- `-foffload-fp32-prec-div` does fix it, but only takes effect at final link and therefore applies
  to **every** kernel in the module.
- Adopted instead: correct the division inside the kernel with Markstein's method, which needs no
  global flag -- `q0 = a * y; r = fma(-b, q0, a); q = fma(r, y, q0)` with `y = 1/6` folded on the
  host and therefore correctly rounded. Validated bit-for-bit against IEEE FP32 division over 4.2M
  normal values on B60.
- For the same reason, step 4's `xh / scale` became a multiply by `ldexp(1.0f, -exp)`: the two are
  mathematically identical for a power-of-two divisor, and the multiply avoids the approximate
  divide entirely.

**Finding 2: this translation unit must be compiled with `-fp-model=precise`.** icpx defaults to
`-fp-model=fast`, whose reassociation collapses the Markstein correction above straight back into a
single ordinary (approximate) division, undoing it. Set per-file via
`set_source_files_properties` in `CMakeLists.txt` so no other kernel's performance is affected.

**Finding 3: the device `log2` itself is trustworthy.** On B60, `sycl::log2(float)` agrees
bit-for-bit with torch/numpy FP32 `log2` on every input tested (+-6 ULP around `2^e` and `6*2^e`,
exhaustive over all positive bf16 and fp16 magnitudes, and 1.2M random values spanning the full
exponent range). The FP32 path is therefore the default; `ARK_SVDQ_LOG2_FP64=1` keeps an FP64
escape hatch for parts whose FP32 `log2` is less accurate. The earlier conclusion that a
"double then narrow" formulation was required attributed the division error to `log2`; that has
been corrected.

**Parallel decomposition**: two specialised paths, selected by whether a low-rank down projection
was requested.

*Quant-only path* (`launch_quant_only`, no `lora_down`): a flat 1-D range with one work-item per
32-element group, `kQuantWorkGroupSize = 256`. There is no cross-row reuse to exploit when the
low-rank projection is absent, so any row-blocked shape only throws away parallelism. This path is
memory bound (see numbers below).

*Fused path* (`launch_fused`): **zero shared local memory**, built entirely on sub-group
collectives. One sub-group (`kWorkGroupSize = 16` lanes) owns `kRowsPerSubGroup` rows and walks the
whole K dimension; `kFusedWorkGroupSize = 256` puts 16 sub-groups in a work-group. Walking full K
inside one sub-group is what makes fusing the low-rank down projection worthwhile -- `lora_down` is
re-read per sub-group rather than per row. Within a sub-group the lanes work *cooperatively* on the
same 32-element group, lane `l` owning the element pair `(2l, 2l+1)`; this makes both the `x` and
the `lora_down` loads fully coalesced, and lets `amax` come from a single
`reduce_over_group(sycl::maximum)`. Rank accumulators `acc[MaxRank]` live in registers and are
reduced once at the very end with `reduce_over_group(sycl::plus)`, with the per-rank stores spread
across lanes rather than funnelled through lane 0. The loop is unrolled `kGroupsPerStep` groups at a
time; this is essential, because the per-group `reduce_over_group` for `amax` otherwise forms a
serial latency chain that dominates the kernel.

Because the reduction order is fixed, `lora_act` is run-to-run deterministic, but it is **not**
required to match PyTorch bit-for-bit (different reduction order; §4.1 gates it on tolerance).

**Tuning constants** (`svdquant_mxfp4_kernel.cpp`, all measured on B60 -- re-sweep on new hardware):
`kGroupsPerStep = 16`, `kRowsPerSubGroup = 1`. The sweep is strongly non-monotonic and the two
interact, so change them together and re-measure. Observed at `4608x3072 R=32`: `(4,1)` 0.83 ms,
`(4,2)` 0.91 ms, `(8,2)` 0.61 ms, `(16,1)` **0.53 ms**, `(16,2)` 1.09 ms, `(32,1)` 0.77 ms.
`kRowsPerSubGroup = 2` is the only setting that meaningfully helps very large K (it halves
`lora_down` re-reads: `4608x12288` goes 2.98 -> 2.62 ms with `(8,2)`) but it costs 15-100% everywhere
else because `acc[kRowsPerSubGroup][MaxRank]` runs into register pressure at `R = 32`. A K-dependent
dispatch is a legitimate future option; it was not worth the extra template instantiation yet.

**Validated**: `qact` / `ascales` are bit-identical to the oracle across every §4.1 case
(bf16/fp16; M in {1,2,7,8,16,64,255,256,4608}; K in {32,64,96,128,256,512,1024,3072,12288};
R in {1,8,16,32,64}; zero groups, subnormals, all-6.0, negative zero, powers of two, sparse).
`lora_act` relative L2 at `4608x3072 R=32` (against an FP64 oracle): 1.658e-3 for bf16 input (DPAS and
scalar alike, both sitting exactly on the 1.661e-3 bf16 storage floor); for fp16 input, 2.078e-4 for
scalar (again on its 2.071e-4 floor) and 2.927e-4 for DPAS.

The 1.4x gap for DPAS under fp16 is **explainable**: the split residual `lo = value - float(hi)` has
to be stored back in the operand type T, and fp16's exponent range is narrow -- `lo` is on the order
of `value * 2^-11`, so once `value`'s exponent drops below -3 the residual falls into fp16
subnormals and starts losing bits. BF16 does not suffer this because it keeps fp32's exponent range.
The gap is well inside the §4.1 gate of 5e-4 and is therefore accepted; if it ever needs closing, the
fix is to store both split planes as bf16 rather than T, at the cost of converting A (DPAS requires
both operands to share an element type).

**Measured performance (B60, bf16, P50, vs. the PyTorch reference path)**:

| shape | R | ARK | reference | speedup |
|---|---|---|---|---|
| `4608x3072` | 32 | 0.232 ms | 7.366 ms | 32.3x |
| `4608x3072` | 16 | 0.226 ms | 7.395 ms | 32.8x |
| `4608x3072` | 64 | 0.254 ms | 7.584 ms | 29.7x |
| `1024x3072` | 32 | 0.088 ms | 1.565 ms | 17.4x |
| `4608x12288` | 32 | 1.130 ms | 29.333 ms | 25.9x |
| `256x3072` | 32 | 0.068 ms | 0.421 ms | 6.0x |
| `1x3072` | 32 | 0.061 ms | 0.415 ms | 6.8x |
| `4608x3072` quant-only | - | 0.134 ms | - | 267 GB/s |
| `4608x12288` quant-only | - | 0.430 ms | - | 334 GB/s |

(Geomean 23.3x, worst 6.0x; both clear the §4.3 gates. Progression at `4608x3072 R=32`:
scalar 0.530 ms / geomean 12.5x, joint_matrix DPAS 0.248 ms / geomean 18.3x, sycl-tla 0.232 ms /
geomean 23.3x. The largest single jumps were not on that shape but at R=64 and M=1 -- see §3.3.4.)

The quant-only path is essentially at the memory roof: a bare device-side read of `x` at
`4608x3072` costs 0.092 ms (314 GB/s).

**Remaining headroom**: the honest ceiling for the low-rank branch is the **memory roof, not the
FLOP roof**. At `4608x3072 R=32` the arithmetic intensity is `2*M*K*R / (M*K*2)` ~ 32 FLOP/byte
against a B60 machine balance of 88e12/386e9 ~ 228 FLOP/byte, so it is firmly memory bound: 28.5 MB
of operands at the measured 386 GB/s is ~0.074 ms. Do not set TFLOP/s targets for this branch.

Untried levers, roughly in order of expected value: **fusing the low-rank projection into the
quantization kernel so `x` is read once rather than twice** -- now the single largest remaining gap,
since at `4608x3072` the fused kernel costs 0.232 ms against 0.134 ms for quant-only and the
difference is very nearly one extra pass over `x`; K-splitting for small M, much less pressing now
that §3.3.4's row blocking recovered most of it; and inspecting actual register spills via compiler
diagnostics, which has still never been done directly. The sycl-tla 2D block load previously listed
here is **done** -- see §3.3.4, where it also turned out to help for a different reason than
predicted. Performance optimization is a separate phase from correctness and does not change the
§3.3.1 contract.

**Note on fusion being the right call**: an unfused formulation is *not* competitive under this
contract, because §3.3.1 requires the smoothing to happen in FP32. Materialising that FP32
intermediate for torch to consume -- `(x.float() * smooth).bfloat16() @ lora_down.T` -- already
costs 0.986 ms at `4608x3072`, i.e. more than the entire fused kernel.

### 3.3.3 DPAS low-rank projection (implemented and validated)

In its scalar form the low-rank branch reached only ~2.5 TFLOP/s, and the cause was **operand fetch**
rather than arithmetic: the inner loop read two bytes of `lora_down` per FMA (bytes/FMA = 2.00, zero
register reuse), so `lora_down` was re-read once per row -- 906 MB at R=32 against only 28 MB for `x`.
DPAS fixes this structurally: the systolic array supplies operand reuse for free, and moving the
accumulators into dedicated accumulator registers also relieves the register pressure that had
blocked row blocking.

**The key algebraic rewrite (fold `smooth` into B).** The obvious mapping (A = the smoothed
activations) is the wrong one: it would force the FP32 product `x * smooth` to be materialised,
rounded to 16-bit and staged through SLM before `joint_matrix_load` could reach it, and SLM occupancy
is exactly what made an earlier version slow. Since `smooth` is per-column (per-K), it can be folded
into the *other* operand instead:

```
lora_act = (x * smooth) @ lora_down^T  ==  x @ (smooth * lora_down)^T
```

Three benefits: A becomes the raw `x` exactly as it sits in memory -- no staging, no SLM, and **zero
rounding on A**; the folded operand `smooth * lora_down` is `[K, R<=64]`, tiny and cache-resident;
and all precision loss is concentrated in that one small matrix, where it is cheap to correct.

**Precision: split (two passes).** DPAS has no FP32 x FP32 mode, so the folded operand must land in
16-bit. Rounding it once costs relative L2 1.65e-3. Splitting it into a high plane and a residual
plane and running two DPAS passes into the same FP32 accumulator recovers 2.44e-6. Because this
kernel is memory bound and the matrix is already cache-resident, the second pass is close to free.
As recorded in §4.1: under the 16-bit output contract the split lands exactly on the storage floor
while a single pass is 1.4x-8x worse -- **the 16-bit output did not make the split redundant**.

**VNNI layout.** The prologue emits B already tiled and VNNI-interleaved; packer and consumer share
`packed_b_offset()`, so the two cannot disagree. The two B planes live in a caller-supplied
workspace whose size comes from `svdquant_workspace_elements()`, so Python never duplicates the
layout knowledge.

**The M gate is structural, not tuned.** `dpas_lora_supported` requires `m >= kTileM (= 8)`. Below
one tile height there are no tiled rows at all and the entire projection falls to
`launch_lora_dpas_tail`, a scalar epilogue with only `m * r` work items (32 of them at M=1, R=32, on
a 160-EU device). Measured at K=3072: M=4 runs at 0.24x the scalar fused path while M=8 is already
2.05x it, with the cliff in exactly the same place at K=12288.

**`ARK_SVDQUANT_DISABLE_DPAS=1`** forces the scalar path; the tests use it to compare the two paths
against each other. This env var is **deliberately not cached in a function-local static**: the tests
flip it between two launches in the same process, and caching would silently make that comparison
vacuous (which it did, once).

**`ARK_SVDQUANT_DISABLE_CUTE=1`** falls back from the sycl-tla path of §3.3.4 to this one, under the
same no-caching rule and for the same reason.

**Status.** This path is superseded as the default by §3.3.4, which is faster on every measured
shape. It is retained because it is the fallback when `ARK_SYCL_TLA` is off, and because the two
paths pin each other in `test_cute_matches_joint_matrix_path`.

### 3.3.4 sycl-tla (CuTe) low-rank projection (implemented and validated; current default)

The §3.3.3 path left the A load at 161 GB/s against a 386 GB/s roof, because `joint_matrix_load`
fetches a fixed 8 x 16 tile -- 8 rows of 32 bytes, i.e. half a cache line each. The hypothesis was
that the Xe 2D block-load atoms, which expose `XE_LOAD_2D<Bits, Height, Width, BlockWidth>` with
`Height <= 32` and `Bits * Width <= 512`, would close that gap by fetching 32 rows x 64 bytes.

This was implemented at the **CuTe arch layer** (raw `XE_LOAD_2D` / `XE_DPAS_TT` plus the
`__builtin_IB_subgroup_createBlock2DAddressPayload` intrinsics) rather than the device layer.
`GemmUniversalAdapter` was rejected for the reason already given in §3.3.3: its `<_256,_256,_32>`
tile leaves MMA utilisation at 12.5% when `N = R <= 64`, and it is a separate launch that would
re-read `x`.

**Three hardware contracts, each established by probe rather than by reading.** Each produces
plausible-looking but silently wrong numbers if assumed incorrectly, so each was measured on a B60
against a host double reference *before* the kernel was written:

1. **Register order for a `Count=2` load.** `XE_LOAD_2D<16, 32, 32, 16>` fetches 32 rows x 32
   columns as two 16-column blocks, and the destination registers are laid out **block-major**: A
   tile `(kt, mt)` sits at `a[kt * MTiles + mt]`. Row-major would have been an equally plausible
   guess and would have silently transposed the K contributions.
2. **`XE_LOAD_2D_VNNI` performs the VNNI interleave in hardware**, reading a plain row-major `[K, N]`
   surface. The host-side packer of §3.3.3 is therefore *deleted* on this path, not reimplemented:
   B is emitted as an ordinary row-major `[K, R]` matrix.
3. **Out-of-bounds rows read back as zero.** The load bounds-checks against the surface height
   recorded in the payload. This is why this path needs **no scalar tail kernel and has no minimum
   M**: partial row tiles are handled by the same code with only the stores guarded.

**The hypothesis was wrong, and the measurement says so.** Sweeping the row block over 8 / 16 / 32:

| rows per sub-group | 8 | 16 | 32 |
| --- | --- | --- | --- |
| geomean speedup | **23.33x** | 22.07x | 20.35x |
| flux mlp 4608x3072 R32 | **0.231 ms** | 0.238 | 0.261 |
| flux ffn 4608x12288 R32 | **1.130 ms** | 1.236 | 1.266 |
| mid batch 1024x3072 R32 | **0.087 ms** | 0.100 | 0.126 |

Widening the row block made things monotonically *worse*. Two effects swamp the fetch width:

* **Occupancy.** Rows per sub-group divide into the sub-group count, so wide row blocking starves
  the machine -- at M = 1024 with 32 rows the entire grid is 2 work-groups on a 160-EU device.
* **Register pressure.** The accumulator is `Rows/8 * NTiles` vectors of 8 floats per lane, which
  spilled outright at `Rows = 32, R = 64`: 0.763 ms, against 0.266 ms once the rows were halved.

So the row blocking is simply 8, and **the sycl-tla path wins for reasons other than the one it was
built on**. At `Rows = 8` its row block is exactly `joint_matrix`'s height; what actually remains is
the *K* width of the load (two K-tiles per instruction, so 64-byte rows rather than 32), the
hardware VNNI transform, and the absence of a separate tail kernel.

`ARK_SVDQUANT_CUTE_ROWS` (8/16/32) keeps the wider instantiations reachable so this sweep can be
repeated on hardware with a different register file or EU count without a rebuild. Values that would
reintroduce the spill are clamped rather than honoured, and
`test_cute_row_blocking_override_is_equivalent` pins all three to the same result.

**Measured against §3.3.3** (P50, bf16, lower is better):

| shape | sycl-tla | joint_matrix | gain |
| --- | --- | --- | --- |
| flux mlp 4608x3072 R32 | 0.232 ms | 0.249 | 1.07x |
| flux mlp R16 | 0.226 ms | 0.230 | 1.02x |
| flux mlp R64 | 0.254 ms | 0.390 | **1.54x** |
| flux ffn 4608x12288 R32 | 1.130 ms | 1.240 | 1.10x |
| mid batch 1024x3072 | 0.088 ms | 0.111 | 1.26x |
| small batch 256x3072 | 0.068 ms | 0.093 | 1.37x |
| single token M=1 | 0.061 ms | 0.156 | **2.56x** |
| geomean vs unfused baseline | **23.31x** | 18.34x | |
| worst per-shape | **5.99x** | 2.55x | |

The two largest gains are exactly where the two structural differences bite: R=64 is where the
joint_matrix accumulator ran out of registers, and M=1 is where its tail kernel took over.

**Build note.** The kernel body sits inside `#ifdef __SYCL_DEVICE_ONLY__` because the 2D block-load
builtins have no host declaration. That makes an implicit `[=]` capture *nothing* during host
compilation and eight things during device compilation, which trips SYCL's lambda-size static
assert. The capture list must therefore be written out explicitly -- the same applies to any future
kernel using these intrinsics.

### 3.3.5 Fusing the projection into the quantization kernel (implemented; **currently a regression**)

Until now the matrix path issued two launches over `x`: `launch_fused_cute` did the projection and
`launch_quant_only` did the encoding. Section 4.4's criterion 2 ("one launch over `x`") was
therefore unmet, and the gap between the two-launch total (0.232 ms at flux mlp) and quant-only
alone (0.134 ms) is very nearly one extra DRAM pass over `x`. Closing it was the obvious next step.

**Why the fusion is structurally clean.** `kGroupSize == kCuteKStep == 32`. One DPAS K-step is
exactly one micro-scaling group per row, so each loop iteration can finish the quantization of the
rows it just loaded -- no leftover state, no second traversal. `kFusedCuteRows` is pinned to
`kDpasSubGroup == 16` so the quantization phase maps lane -> row 1:1, which lets `load_smoothed`,
`encode_group` and `store_group` be reused *verbatim*. That verbatim reuse is deliberate: those
three functions are what the bit-exact PyTorch-oracle tests pin, and the A0 contract must not move.
It worked -- `test_fused_cute_matches_scalar_path` asserts `torch.equal` on `qact` and `ascales`
across M in {1, 7, 8, 16, 17, 255, 1024} and both dtypes, and all 138 tests pass.

The two phases want `x` in different per-lane layouts, so the quantization re-reads it rather than
decoding the DPAS A registers. That re-read is an L1 hit (the `XE_LOAD_2D` just pulled the same
16x64B tile); the saving being chased is the DRAM round trip. Decoding from the A registers was
rejected because it would need cross-lane shuffles to re-pair elements for the nibble packing,
risking the frozen A0 contract for no bandwidth gain.

**It is nonetheless slower.** Geomean fell from 23.31x to 11.09x; flux mlp went 0.232 -> 0.402 ms
and M=1 went 0.061 -> 0.309 ms. Wall-clock is almost flat in M -- 0.29 ms at M=16 and 0.30 ms at
M=1024, a 64x increase in work for 4% more time -- which is the signature of a kernel that is not
throughput-bound at all.

**Measured cause: parallelism starvation, not register spill.** Spill was the first hypothesis
(it was the correct one for the 32-row blocking in 3.3.4). It is wrong here. Holding the total
element count fixed at 4608x3072 and trading rows against columns -- which moves work between the
*parallel* M dimension and the *serial* in-kernel K loop while changing neither the FLOPs nor the
bytes -- gives:

| M | K | subgroups (M/16) | K iterations | fused P50 |
| --- | --- | --- | --- | --- |
| 4608 | 3072 | 288 | 96 | 0.611 ms |
| 9216 | 1536 | 576 | 48 | 0.212 ms |
| 18432 | 768 | 1152 | 24 | **0.149 ms** |
| 36864 | 384 | 2304 | 12 | 0.162 ms |
| 73728 | 192 | 4608 | 6 | 0.166 ms |

Identical work, 4.1x apart, monotonic in the subgroup count until it saturates around ~1000
subgroups. The fused kernel exposes only `M / 16` subgroups because the K dimension is consumed by
a serial loop, whereas `launch_quant_only` exposed `M * K / 32` independent work-items. At
4608x3072 that is 288 subgroups (4608 work-items) against a device that holds roughly 20k
work-items in flight -- about 22% occupancy, with no other warps to hide the dependent
load -> DPAS -> `load_smoothed` -> `encode_group` chain. The projection alone tolerated this because
it is a small fraction of the runtime; the encoding, which dominates, does not.

So the fusion removed one DRAM pass and paid for it with a ~4x loss of latency hiding. On this
machine the second pass was the cheaper of the two.

**The fix: split K (implemented).** Each row block is given `S` sub-groups, each owning a
contiguous slice of the K range. Quantization needs no cooperation at all -- micro-scaling groups
are independent by construction and each slice writes disjoint bytes of `qact` and `ascales` -- so
slicing it costs exactly nothing. Only the projection accumulates along K, so only it needs a
reduction: each slice writes an FP32 `[M, R]` partial plane into the workspace and a small
`launch_reduce_partials_cute` sums them in slice order. A deterministic partial buffer is used
rather than FP32 atomics so that `lora_act` stays reproducible run to run.

`S` targets a fixed sub-group count (`cute_k_slices`, `kCuteTargetSubGroups = 1024`) rather than
being a constant, because the right value depends entirely on how much parallelism M already
supplies. The sweep behind that number, and the reason the curve turns back upwards above it, is
documented on the constant itself in `sycl_svdquant_mxfp4_cute.hpp`; the short version is that the
partial buffer's own traffic eventually costs more than the occupancy it buys.
`ARK_SVDQUANT_CUTE_TARGET_SUBGROUPS` re-opens the sweep on new hardware without a rebuild.

**Result.** The regression is not merely recovered, it is beaten -- and criterion 2 of section 4.4
is now met, since the projection no longer needs its own pass over `x`.

| shape | two-launch | fused, unsliced | fused + K-split |
| --- | --- | --- | --- |
| flux mlp 4608x3072 R32 | 0.232 ms | 0.402 | **0.195** |
| flux mlp R16 | 0.226 ms | 0.369 | **0.179** |
| flux mlp R64 | **0.254 ms** | 0.507 | 0.281 |
| flux ffn 4608x12288 R32 | 1.130 ms | 1.818 | **0.877** |
| mid batch 1024x3072 | 0.088 ms | 0.329 | **0.070** |
| small batch 256x3072 | 0.068 ms | 0.325 | **0.062** |
| single token M=1 | **0.061 ms** | 0.309 | 0.062 |
| geomean vs unfused baseline | 23.31x | 11.09x | **25.79x** |
| R=32 overhead over quant-only | -- | 197.7% | **45.1%** |

R=64 is the one shape still behind the two-launch configuration. It is the NTiles=4 instantiation,
where the accumulator alone is 64 floats per lane on top of the A vectors and the 32-float
`smoothed` scratch, so it is the first to run short of registers -- the same pressure that decided
the row blocking in 3.3.4, arriving here through the rank instead. Splitting the *N* dimension
across sub-groups would halve the accumulator, at the cost of either duplicating the quantization or
gating it to one N-half; it has not been tried.

## 4. Kernel A acceptance

### 4.1 Correctness

The §3.3.1 A0 contract is authoritative; it is equivalent to AutoRound
`quant_mx_rceil(..., bits=4, group_size=32, data_type="mx_fp4e2m1")` **including its zero-group
exponent=1 behavior**.

| Item | Gate |
|---|---|
| Activation scales | Bit-exact for all finite inputs (including zero-group code 128) |
| E2M1 codes | Unpacked codes and packed bytes are bit-exact |
| Zero groups | Exponent 1, all codes 0; no NaN/Inf or code 255 |
| Boundaries | Cover all 7 midpoints in the table above, `amax=6`, values just above 6, and exponent limits |
| Non-finite input | Explicit debug failure; never a success-shaped result |
| FP16 `lora_act` | Relative L2 `<= 5e-4`, cosine `>= 0.9999` |
| BF16 `lora_act` | Relative L2 `<= 3e-3`, cosine `>= 0.999` |

The low-rank thresholds were tightened in 2026-09 alongside the 16-bit contract (they used to be
5e-3 / 1e-2). The new values are not arbitrary: they sit just above the **storage rounding floor**.
Measured against an FP64 oracle at `4608x3072 R=32`, merely writing a mathematically exact result
out as BF16 already costs relative L2 1.661e-3, and as FP16 2.071e-4. No kernel implementation can
beat those numbers, because they are the cost of storing the result at all. With a floor that high,
keeping the old 1e-2 gate would let a genuinely broken projection pass.

Measured comparison (relative L2 vs. FP64):

| Variant | FP32 out | stored BF16 | stored FP16 |
|---|---|---|---|
| scalar FP32 | 5.0e-7 | 1.661e-3 | 2.07e-4 |
| single-pass bf16 B | 1.65e-3 | 2.34e-3 | 1.66e-3 |
| split-bf16, 2 passes | 2.44e-6 | **1.661e-3** | **2.07e-4** |
| pure storage floor | — | 1.661e-3 | 2.07e-4 |

Two non-obvious conclusions: (1) the split-bf16 two-pass scheme lands exactly on the storage floor,
contributing nothing of its own, while single-pass bf16 is 1.4x (bf16 out) to 8x (fp16 out) worse —
**moving to a 16-bit output did not make the split redundant, it made it the only scheme that
extracts what the 16-bit format can hold**; (2) FP16 storage is 8x more accurate than BF16, but
fp16's overflow headroom is only 100-200x (`|lora_act|max` already reaches 618 at K=12288), which is
why the output follows `x`'s dtype rather than always using fp16.

These remain ARK's first-release gates, not values claimed by the paper; if reference distributions
from real FLUX layers show they are too loose, tighten them rather than relaxing them to pass.

The low-rank thresholds are proposed ARK gates, not paper claims. They may be tightened using real FLUX distributions but must not be relaxed merely to pass.

**Harness**: `test/test_svdquant_mxfp4.py`. Run it from `auto_round_extension/ark/test`:

```bash
PYTHONPATH=<repo>/auto_round_extension/ark:<repo> python -m pytest test_svdquant_mxfp4.py -q
```

The suite checks **both** adjacent layer pairs so any divergence can be localized:
`quant_mx_rceil` oracle <-> `quant_down_reference` <-> fused SYCL kernel. It also pins the one
intentional deviation from the oracle (the subnormal-underflow NaN, see §3.3.1) as an explicit
test, so a future oracle fix is noticed rather than silently absorbed. Runtime is ~5 s.

Beyond the coverage listed above it includes: the FP32-division/`log2` boundary stress that
originally caught the §3.3.2 bug (every ULP within +-6 of `6*2^e` and `2^e` across the whole
exponent range, plus exhaustive sweeps over *every* finite positive bf16 and fp16 magnitude);
the §4.2 runtime gates (output ABI, no host sync, no shared-workspace race across concurrent
calls, tail row blocks); input-validation and backend-dispatch behavior (`backend="ark"` must
never fall back silently); and packing-layout parity with `auto_round.export.svdquant_mxfp4`.

Coverage includes:

- Random, zero, constant, very small, very large, and sign-skewed inputs.
- M: `1, 2, 7, 16, 64, 256, 4608`.
- Minimum legal K, non-tile K multiples of 32, and FLUX K=3072/12288.
- R=16/32, with full performance validation for R=32.
- BF16/FP16, contiguous and supported non-contiguous inputs, multiple seeds, and deterministic reruns.

### 4.2 Runtime

- ABI-compliant shape, dtype, and device for every output.
- No out-of-bounds access or padding contribution.
- Same-queue producer/consumer operation.
- No `queue.wait()`, `torch.xpu.synchronize()`, or CPU round-trip.
- No shared-workspace race across streams or concurrent calls.
- Checked 64-bit size arithmetic.

### 4.3 Performance

The baseline is three separate operations on the same XPU:

```text
smooth kernel
MXFP4 reference quant/pack kernel
BF16/FP16 lora_down GEMM
```

Use XPU events after warmup for at least 100 iterations; report P50/P95 with preallocated input, output, and workspace.

| Item | Proposed gate |
|---|---|
| Launches | One Kernel A launch |
| Input traffic | Profiler shows one global read of the main X tile |
| Synchronization | No host sync |
| Representative shapes | At least 1.20x geometric-mean speedup over the three-step baseline |
| Per-shape regression | At least 0.95x baseline, otherwise dispatch/fallback |
| Rank-32 overhead | No more than 20% over quant-only Kernel A |

These are project targets rather than paper guarantees. Target hardware must rerun and freeze final thresholds.

**Harness**: `test/bench_svdquant_mxfp4.py`. It times the three baseline steps *separately* and
sums their medians, which is deliberately generous to the baseline (it charges nothing for the
launch gaps between the three kernels), so the reported speedup is a lower bound on the real win.
Run it from `auto_round_extension/ark`:

```bash
PYTHONPATH=. python test/bench_svdquant_mxfp4.py            # bf16, 100 iterations
PYTHONPATH=. python test/bench_svdquant_mxfp4.py --dtype fp16 --iters 200
```

**Measured (B60, bf16, P50, 100 iterations after 20 warmup)**:

| shape | M | K | R | fused | smooth | quant+pack | GEMM | sum | speedup | GB/s | TFLOP/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flux mlp | 4608 | 3072 | 32 | 0.535 ms | 0.595 | 6.575 | 0.305 | 7.474 | 13.96x | 68.4 | 1.69 |
| flux mlp r16 | 4608 | 3072 | 16 | 0.370 ms | 0.595 | 6.579 | 0.212 | 7.385 | 19.94x | 97.9 | 1.22 |
| flux mlp r64 | 4608 | 3072 | 64 | 0.968 ms | 0.594 | 6.578 | 0.360 | 7.533 | 7.78x | 38.6 | 1.87 |
| flux ffn | 4608 | 12288 | 32 | 3.066 ms | 2.325 | 26.064 | 0.912 | 29.302 | 9.56x | 47.2 | 1.18 |
| mid batch | 1024 | 3072 | 32 | 0.146 ms | 0.119 | 1.371 | 0.049 | 1.540 | 10.54x | 56.8 | 1.38 |
| small batch | 256 | 3072 | 32 | 0.138 ms | 0.043 | 0.320 | 0.047 | 0.410 | 2.97x | 16.2 | 0.36 |
| single token | 1 | 3072 | 32 | 0.126 ms | 0.043 | 0.315 | 0.045 | 0.403 | 3.20x | 1.7 | - |
| quant only | 4608 | 3072 | - | 0.134 ms | 0.595 | 6.593 | - | 7.188 | 53.52x | 266.9 | - |
| quant only ffn | 4608 | 12288 | - | 0.430 ms | 2.333 | 26.117 | - | 28.450 | 66.19x | 333.6 | - |

Geometric-mean speedup **12.48x** (gate 1.20x), worst per-shape **2.97x** (gate 0.95x). Both gates
pass with a wide margin.

#### Measured device rooflines (B60)

Every performance claim below is relative to these, which were measured rather than assumed:

| Roof | Measured |
|---|---|
| bf16 matrix engine (XMX), 4096^3 `torch.matmul` | **88 TFLOP/s** |
| fp16 matrix engine | 93 TFLOP/s |
| FP32 vector, 4096^3 `torch.matmul` | 11.6 TFLOP/s (theoretical ~12.3) |
| Global memory, read / copy | **386 / 391 GB/s** |

**Do not gate Kernel B on `has_subgroup_matrix_multiply_accumulate`.** On this driver
(`1.15.38308+4`) the SYCL device property reports **`False`**, yet the device clearly has a working
matrix engine: bf16 GEMM reaches 88 TFLOP/s against 11.6 TFLOP/s for FP32, a 7.6x ratio that vector
units cannot produce. The property is a runtime reporting artifact, not an absence of hardware.
A capability gate written against it would wrongly conclude this machine has no matrix engine.

#### Reading the TFLOP/s column

The column counts **only** the low-rank projection (`2*M*K*R`), because that is the only part of
Kernel A that does meaningful arithmetic; smoothing, amax reduction, `log2`, E2M1 encoding and
nibble packing produce almost no FLOPs but dominate the runtime. At `4608x3072 R=32` that is
0.906 GFLOP, which is **1.04%** of the 87 GFLOP in the corresponding main GEMM (`4608x3072x3072`)
that Kernel B will do. So this number must not be compared against published W4A4 GEMM throughput
(e.g. Nunchaku's 600-1000 TFLOP/s): that figure describes the *main GEMM* on NVIDIA hardware with
FP4 tensor cores, whereas this one describes a thin rank-32 side branch inside a bandwidth-bound
quantization kernel. Even a perfect Kernel B on B60 is capped at the 88 TFLOP/s above.

The meaningful gate for Kernel A is **bandwidth**, and by that measure the quant-only path is
essentially done: 334 GB/s of the 386 GB/s roof (87%) at `4608x12288`.

#### The rank-32 overhead gate does not hold, and has been reclassified as a diagnostic

Measured overhead over quant-only at the same shape is ~296%, not 20%. The cause is now measured,
and the earlier explanation in this document ("compute bound") was **wrong**:

| R | total | lora increment | ms/rank | bytes/FMA |
|---|---|---|---|---|
| 8 | 0.286 ms | 0.113 ms | 0.0141 | 2.00 |
| 16 | 0.370 ms | 0.197 ms | 0.0123 | 2.00 |
| 32 | 0.528 ms | 0.355 ms | 0.0111 | 2.00 |
| 64 | 0.934 ms | 0.761 ms | 0.0119 | 2.00 |

The branch is bound by **operand fetch, not by the FMA units**. Two signatures show this: `ms/rank`
is flat, so time scales strictly linearly in R; and `bytes/FMA` is exactly 2.00, meaning every
single FMA loads two bytes of `lora_down` with **zero register reuse**. Consequently `lora_down` is
re-read once per row -- **906 MB at R=32 against only 28 MB for `x`**, a 32x multiplier. It remains
tolerable only because `lora_down` is 0.2 MB and therefore cache-resident. The resulting
~2.5 TFLOP/s is about 20% of the 12.3 TFLOP/s FP32 vector roof.

The fix is operand reuse, and there are two routes:

1. **Register blocking over rows** (`kRowsPerSubGroup > 1`) halves `bytes/FMA` to 1.00. It measurably
   helps at large K (`4608x12288`: 2.98 -> 2.62 ms) but is currently blocked at `R = 32` by register
   pressure from `acc[kRowsPerSubGroup][MaxRank]`; worth retrying under large-GRF mode.
2. **A `joint_matrix` / DPAS formulation** of the low-rank projection, which gets operand reuse for
   free from the systolic array. ARK already has substantial `joint_matrix` infrastructure to model
   this on (`sycl_tla_dense_gemm.hpp`, `sycl_tla_moe_prefill_s4_dpas.hpp`, and ~10 other headers).

**The low-rank accumulation is therefore the single largest remaining optimization target for
Kernel A**, ahead of memory-side work, which is already near the roof.

### 4.4 Milestone

Kernel A is complete when:

1. Golden, random, tail, and FLUX-shape correctness gates pass.
2. It uses one launch and the current queue without host sync.
3. Profiling proves smoothed-input reuse.
4. Quant-only, down-only, unfused, and fused benchmarks are reproducible.
5. The API does not depend on an unvalidated Kernel B physical layout.

**Status against these criteria.** 1, 4 and 5 are met (138 pytest cases; the §4.3 harness reports
quant-only, unfused and fused side by side; the ABI is frozen independently of Kernel B). 3 is met in
the sense that matters -- the smoothed input is never materialised at all, because §3.3.4 folds
`smooth` into the low-rank operand instead.

**Criterion 2 is now met.** Section 3.3.5 folded the projection into the quantization kernel, so
there is exactly one pass over `x`. Two auxiliary launches remain and neither reads `x`: the B
prologue touches a `[K, R<=64]` matrix, and the partial reduction touches `[S, M, R]` when K is
split. All of them go on the caller's queue with no host synchronization.

## 5. Kernel B: W4A4 GEMM + low-rank up

### 5.1 Final fused operations

The target implementation will:

1. Consume Kernel A qact/ascales.
2. Consume native-packed qweight/wscales.
3. Execute MXFP4 W4A4 GEMM with FP32 residual accumulation.
4. Accumulate `lora_act @ lora_up.T` into the same output tile.
5. Add bias.
6. Cast to the input dtype and store `[M,N]`.

It must not materialize complete BF16/FP16 residual activations or weights.

### 5.2 Work completed on B60

- Stable C++/pybind/SYCL shape, dtype, device, queue, and error contracts.
- Workspace query and overflow/alignment/bounds validation.
- Canonical-to-native opaque blob header/version contract. **Directly reuse** the existing
  three-part API shape in `auto_round_kernel/ark.cpp` (`packed_weight_size` / `repack_quantized_weight`
  / `unpack_weight`: query packed size → pack from canonical weight → restore from blob); do not invent
  a separate naming or call sequence for Kernel B.
- Clearly labeled reference/emulated B: decode, FP32 GEMM, low-rank up, bias, and cast.
- Independent or pluggable SYCL low-rank-up/bias/output epilogue.
- Residual-only, low-rank-only, combined, bias/no-bias, and M/N-tail tests.
- Mock mainloop proving an accumulator tile can feed the epilogue without an intermediate tensor.
- FLUX shape benchmark harness and target bring-up commands.
- A capability-gate skeleton: mirroring `moe_prefill_dpas_s4_enabled()` (env-flag toggle) and
  `moe_prefill_dpas_s4_pergroup_shape_ok()` (shape precondition) in
  `wrapper/include/sycl_tla_moe_prefill_s4_dpas.hpp`, reserve isomorphic
  `svdquant_mxfp4_dpas_enabled()` / `svdquant_mxfp4_shape_ok()` placeholders. On B60 these may
  unconditionally return `false` (the instruction is unavailable), but the signatures, the env-flag
  naming convention (`ARK_SVDQUANT_MXFP4_DPAS`), and call sites must be frozen at this stage; Phase 3
  only replaces the function bodies.

### 5.3 Target-only work

- MXFP4 matrix instruction selection and validation.
- Native qweight/qact/scale swizzles and tile layouts.
- W4A4 mainloop and subgroup/prefetch/SLM/register tuning.
- Final native blob payload layout.
- Kernel B performance gates and productization decision.
- The real implementation of `svdquant_mxfp4_dpas_enabled()` / `svdquant_mxfp4_shape_ok()` (B60 keeps
  these as constant `false`/reference-shape-only).

No placeholder may return fabricated output. Until the native mainloop exists, explicit fused mode reports unsupported; emulated mode is clearly named and logged.

### 5.4 B60 milestone

B60 requires only:

1. Kernel B ABI matches Kernel A outputs.
2. Emulated B matches the PyTorch reference.
3. A mock accumulator drives low-rank up, bias, and cast.
4. Native insertion points do not leak into portable checkpoints.
5. Capability gates never select fused B on B60.

A working native launch, BF16-GEMM speedup, and frozen target layout are not required.

### 5.5 Relationship to existing INT4/INT8 DPAS kernels (important clarification)

`auto_round_extension/ark/auto_round_kernel/wrapper/include/sycl_tla_moe_prefill_s4_dpas.hpp` and
`sycl_tla_moe_prefill_int_dpas.hpp` already implement, and have run on real Xe DPAS hardware, a
"packed-nibble weight → per-K-group dequant into a workspace → DPAS INT8/BF16 mainloop" path, complete
with a persistent scheduler, policy template classes, and an env-flag capability gate. This is not the
MXFP4 matrix instruction, but it is the **only** low-bit GEMM mainloop in this repo that is already
validated on real Xe DPAS hardware, so:

- When designing Kernel B's dispatcher, launcher, and workspace lifecycle in Phase 2/3, treat these two
  files as the structural template (scheduler shape, policy specialization style, env-flag naming)
  rather than designing from scratch.
- If bring-up time for the native MXFP4 matrix instruction on the target machine is uncertain, the
  already-validated "S4 → workspace dequant → INT8/BF16 DPAS" path can serve as a **transitional
  fallback** for Kernel B (decode MXFP4 into the same kind of workspace, then feed the existing
  INT8/BF16 DPAS mainloop), to get a measurable on-hardware performance baseline faster before deciding
  whether the native MXFP4 instruction route is worth the investment. Whether to adopt this fallback
  should be recorded as an explicit decision point at Phase 3 kickoff, not assumed as the default path.
- The `comp_n`/`comp_k`/`mem_k`/`num_n_lanes` constants in `NunchakuMXFP4Packer` (see
  `auto_round/export/svdquant_mxfp4.py`) encode a CUDA warp-level MMA physical layout that only serves
  Nunchaku CUDA runtime checkpoint interop. They are **not** the ARK/Xe native layout and must not be
  used as a reference for Kernel B's target tile/swizzle; ARK's native opaque blob must be redesigned
  for Xe DPAS in Phase 3 and produced from the canonical weight via a `repack_quantized_weight`-style API.

## 6. Phases

### Phase 1: Kernel A

Freeze the logical ABI and oracle, implement reduction/E2M1 packing, fuse smoothing and down projection, complete correctness/runtime tests, then pass XPU profiling and benchmark gates.

### Phase 2: Kernel B preparation

Freeze the logical ABI and opaque packing boundary, complete emulated B and the epilogue, add a mock mainloop and benchmark harness, and produce the target-machine bring-up checklist.

### Phase 3: Target bring-up

Select the MXFP4 instruction route, freeze native layout and tiles, implement the W4A4 mainloop, and validate end-to-end parity, performance, and FLUX integration.

## 7. References and reuse

### 7.1 Paper and cross-repo numerical/fusion semantics

- SVDQuant paper, arXiv:2411.05007.
- Nunchaku:
  - `src/Linear.cpp`
  - `src/kernels/zgemm/gemm_w4a4_launch_impl.cuh`
  - `src/kernels/zgemm/gemm_w4a4.cuh`
- AutoRound:
  - `auto_round/data_type/mxfp.py`: `quant_mx_rceil` is Kernel A's sole numerical ground truth.
  - `auto_round/export/svdquant_mxfp4.py`: `encode_e2m1`/`decode_e2m1`/`encode_ue8m0`/`decode_ue8m0`/
    `pack_nibbles`/`unpack_nibbles` are pure, hardware-independent codecs that Kernel A's `qact`/`ascales`
    outputs should match bit-for-bit; `NunchakuMXFP4Packer` is a CUDA-only physical layout (see Section
    5.5; do not treat it as the ARK native layout).
  - `test/unit/common/export/test_svdquant_mxfp4.py`: reuse its random/boundary vectors as Kernel A
    golden-test seeds.
  - `auto_round/algorithms/transforms/svdquant/` (`residual.py`/`apply.py`/`wrapper.py`): the SVD
    decomposition, grouping, and `Q(R) + U@V` PyTorch forward reference — the sole authority for
    end-to-end Kernel A+B semantics.

### 7.2 Engineering patterns reusable from within this repo

Not SVDQuant-specific, but existing conventions in this repo that Kernel A/B should reuse rather than
reinvent:

| Concern | Reuse target | Note |
|---|---|---|
| Queue retrieval | `auto_round_kernel/__init__.py::get_stream` | Derives `sycl_queue` from the input tensor |
| Dtype enum | `auto_round_kernel/__init__.py::ARK_DT` (includes `float8_e8m0`) | Add E2M1/MXFP4 code constants here |
| Canonical-to-native blob boundary | `ark.cpp::packed_weight_size` / `repack_quantized_weight` / `unpack_weight` | Template for Kernel B's native blob query/pack/restore API |
| Python-side ABI pre-validation | `_validate_packed_blob`-style checks in `qlinear.py`/`__init__.py` | Re-check the Section 4.2 ABI contract before calling the native kernel |
| Extension loading | `xpu_loader.py::ensure_xpu_lib`/`load_xpu_lib` | New kernel `.so` exposed via the same `required_symbols` mechanism |
| Low-bit DPAS mainloop structural template | `wrapper/include/sycl_tla_moe_prefill_s4_dpas.hpp`, `sycl_tla_moe_prefill_int_dpas.hpp` | Ready-made persistent scheduler, policy templates, env-flag capability gate (details in Section 5.5) |
| pybind symbol registration | The `m.def(...)` list inside `ark.cpp`'s `PYBIND11_MODULE(PY_NAME, m)` | Register new kernel entry points here; no new pybind module |
| Build registration | `auto_round_kernel/CMakeLists.txt` (`file(GLOB SRCS *.cpp)` auto-collects, but pybind symbols still need manual registration) | New `.cpp` files under this directory compile automatically, but must be added to `ark.cpp`'s `m.def` list |

### 7.3 Reuse boundary

Reuse numerical semantics and fusion boundaries, not CUDA-specific warp, layout, or launch parameters.
The in-repo engineering patterns (Section 7.2) may be reused for implementation details and calling
conventions, but Kernel B's native tile/swizzle must still be redesigned on the target machine (see
Section 5.5).

## 8. Gaps beyond kernel development

Completing Kernel A/B alone does not make this feature production-ready. The following gaps are
independent of the B60/target-machine hardware constraint and need owners and exit criteria in each
phase; they should not be discovered only after the kernels are done.

### 8.1 Python integration and dispatch

No ARK SVDQuant inference-time entry point currently exists: the runtime forward in
`auto_round/algorithms/transforms/svdquant/wrapper.py` only has a PyTorch reference and a
Nunchaku/CUDA-facing export path (`auto_round/export/svdquant_nunchaku.py`); the vLLM-side
`auto_round_extension/vllm_ext/linear_impl_mxfp4.py` only covers the vLLM backend. A new module
(suggested location: `auto_round_extension/ark/auto_round_kernel/svdquant_mxfp4_linear.py`, alongside
`qlinear.py`) is needed to:

- Provide a Python wrapper function/`nn.Module` callable by `SVDQuantLinear` or standalone tests,
  shaped after `qlinear.py::QuantLinear`.
- Select one of three paths based on the Section 5.4 capability gate: "fused Kernel A + native Kernel
  B", "fused Kernel A + emulated Kernel B", or an explicit error — never a silent fallback to plain
  PyTorch.
- Freezing this integration point is part of the Phase 1 exit criteria (Kernel A's ABI must be consumed
  once through this layer to prove it works under a real call pattern), not deferred to Phase 3.

### 8.2 Build, symbol registration, and packaging

- New `.cpp`/`.hpp` files go under `auto_round_kernel/`; `CMakeLists.txt`'s `file(GLOB SRCS *.cpp)`
  picks them up automatically, but **pybind symbols must be manually added to `ark.cpp`'s
  `PYBIND11_MODULE` list**, or Python cannot see the entry point.
- `ARK_DT` in `auto_round_kernel/__init__.py` needs new MXFP4/E2M1 constants; the `required_symbols`
  list passed to `ensure_xpu_lib` by callers must include the new symbol names, or the loader will
  silently find a stale `.so` and only fail with `AttributeError` at call time.
- `auto_round_kernel/version.py` should be bumped per repo convention when new capability lands.
- These are pure engineering checklist items — track them explicitly in the Phase 1 PR description
  rather than skipping them; otherwise "kernel compiles but Python can't call it" rework is likely.

### 8.3 Checkpoint / export format boundary

- The current `svdquant_nunchaku` export format (`auto_round/export/svdquant_nunchaku.py`,
  `auto_round/export/formats/backends/svdquant_nunchaku.py`) produces a physical layout loadable only by
  the Nunchaku CUDA runtime — distinct from ARK's native blob.
- An explicit decision is needed: should ARK consume (a) the **logical/canonical** qact/qweight/scale
  from `svdquant_mxfp4.py` (simple, portable, but requires an online `repack_quantized_weight` at load
  time), or (b) a new `svdquant_ark` export format that writes the native blob directly (faster load,
  but tightly couples the export format to hardware; future target-layout changes require re-export)?
- This decision does not block Kernel A/B development itself, but it blocks "can we actually run a real
  FLUX checkpoint" — decide it at the end of Phase 2 / start of Phase 3, before the native layout is
  frozen.

### 8.4 Test and benchmark harness reuse

- `auto_round_extension/ark/test/` already has shared fixtures (`conftest.py`, `ut_utils.py`) and a
  directory pattern where the same kernel has both an accuracy test and a perf test (e.g.
  `test_moe_prefill_accuracy.py`/`test_moe_prefill_perf.py`). Kernel A/B tests should live in the same
  directory and reuse the same fixtures/timing utilities (the Section 4.3 XPU-event timing method should
  be extracted from the existing perf tests, not rewritten).
- `benchmarks/` already has an independent benchmark script style (e.g. `bench_sparse_topk.py`); Kernel
  A's FLUX-shape benchmark should follow the same CLI/output format for easy cross-kernel comparison.

### 8.5 Bilingual documentation parity

- The Chinese and English versions of this design doc must stay structurally and substantively in sync;
  per repo convention, any `.md` change requires a matching `_CN`/English update. If user-facing docs for
  this feature later change (e.g. `ark/README.md`/`README_CN.md`, or top-level
  `docs/svdquant_details.md`/`_CN.md`), keep them in sync and check this explicitly in review.

### 8.6 Placeholder convention for hardware capability probing

- Until the target machine is available, `svdquant_mxfp4_dpas_enabled()` (Section 5.2) may
  unconditionally return `false`. This is the **only** judgment point allowed to be hard-coded; every
  other shape/dtype/ABI check must be a real check — "we don't have the target machine yet" is not an
  excuse to simplify them.
- Once the target machine is available, the first Phase 3 step is replacing this function with a real
  probe (e.g. querying a SYCL device aspect or a known device-id allowlist), not redesigning the call
  sites.

### 8.7 Source of model-level acceptance baselines (optional, for future reference)

- If this feature ever needs model-level (not just kernel-numerical) acceptance, this repo already has a
  methodology doc for an SVDQuant INT4 reference-accuracy study
  (`.github/instructions/svdquant-int4-reference.instructions.md`, on the unmerged
  `copilot/svdquant-int4-reference-accuracy` branch) recording BF16 and MXFP4 SVDQuant CLIP/CLIP-IQA/
  ImageReward baselines on FLUX.1-dev, plus a "reproduce controls first, then sweep, then gate"
  methodology. If ARK ever reaches model-level acceptance, port/reference that methodology and baseline
  numbers instead of reinventing the thresholds. This is optional follow-up work beyond Phase 3, not a
  Phase 1/2 exit criterion for this design doc.
