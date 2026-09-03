// ARK SVDQuant MXFP4 Kernel A -- smooth + dynamic MXFP4 activation quantization
// + low-rank down projection.
//
// This header exposes the host-side entry point and the numerical device
// helpers of "Kernel A". The helpers live in the header (rather than the .cpp)
// so that the Kernel B preparation work and the standalone SYCL tests can reuse
// exactly the same encode path, which is what keeps the bit-exactness gate in
// section 4.1 of `ark_SVDQuant_mxfp4_design_doc.md` free of carve-outs.
//
// The frozen A0 contract is specified in design doc section 3.3.1 and mirrored
// by the PyTorch reference in `auto_round_kernel/svdquant_mxfp4.py`. Any change
// here must be made in all three places at once.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

namespace ark {
namespace svdquant {

// A0 contract constants. Keep in sync with svdquant_mxfp4.py.
inline constexpr int kGroupSize = 32;
inline constexpr float kE2M1MaxNorm = 6.0f;
inline constexpr int kUE8M0Bias = 127;
inline constexpr float kScaleExpMin = -127.0f;
inline constexpr float kScaleExpMax = 127.0f;
inline constexpr float kZeroGroupExp = 1.0f;

// Maximum low-rank width the fused path accepts. Beyond this the Python
// wrapper must fall back to the unfused path; SVDQuant itself uses 16 or 32.
inline constexpr int kMaxRank = 64;

#ifdef ARK_XPU

// ---------------------------------------------------------------------------
// A0 step 5: E2M1 magnitude code.
//
// Reproduces the exact FP32 operation sequence of
// `auto_round.data_type.mxfp.quant_element(ebits=2, mbits=3, max_norm=6.0)`
// with the default "even" mantissa rounding:
//
//     pe   = max(floor(log2(a)), 0)          # in {0, 1, 2} for a in [0, 6]
//     s    = a * 2**(1 - pe)
//     q    = floor(s + 0.5) - ((s - 0.5) % 2 == 0)
//     code = q + 2 * pe
//
// `code = q + 2 * pe` holds because the oracle's magnitude is `q * 2**(pe - 1)`
// and the reachable (pe, q) pairs map one-to-one onto the codebook index.
//
// Thresholding against the codebook midpoints would NOT be equivalent:
// `floor(s + 0.5)` rounds in FP32, so a value one ULP *below* a midpoint can
// still round up (e.g. a = 0.24999998509 returns code 1, not 0).
//
// `pe` is taken from the exponent field rather than from `log2` because the
// device `log2` is only accurate to a few ULP and would misclassify exact
// powers of two. A zero or subnormal input has exponent field 0, giving -127,
// which the clamp maps to 0 -- the same answer the oracle's `clip(min=0)`
// produces.
// ---------------------------------------------------------------------------
inline uint8_t e2m1_magnitude_code(float a) {
  const uint32_t bits = sycl::bit_cast<uint32_t>(a);
  int pe = static_cast<int>((bits >> 23) & 0xFFu) - 127;
  if (pe < 0) pe = 0;

  // s = a * 2**(1 - pe) with 1 - pe in {1, 0, -1}. Building the power of two
  // from its exponent field and multiplying is exact here (a is in [0, 6], so
  // nothing can over- or underflow) and avoids a call to ldexp in the innermost
  // loop.
  const float s = a * sycl::bit_cast<float>(static_cast<uint32_t>(128 - pe) << 23);

  // A tie is "s - 0.5 is an even integer". s lies in [0, 4) for every reachable
  // a, so the only even integers in range are 0 and 2 and the general fmod
  // reduces to two equality tests.
  const bool tie = (s == 0.5f) | (s == 2.5f);
  const float q = sycl::floor(s + 0.5f) - (tie ? 1.0f : 0.0f);

  return static_cast<uint8_t>(static_cast<int>(q) + 2 * pe);
}

// ---------------------------------------------------------------------------
// IEEE correctly-rounded FP32 division by a compile-time constant.
//
// Intel GPUs expand FP32 `fdiv` into a reciprocal approximation plus refinement
// rather than the correctly-rounded quotient, and neither `-fp-model=precise`
// nor any per-translation-unit flag changes that (the expansion happens in the
// offload backend at link time). The error is one ULP, which is invisible
// almost everywhere but not here: `amax / 6` landing one ULP high near 6*2^e
// pushes `ceil(log2(.))` up by one and emits a UE8M0 code the oracle never
// produces.
//
// Markstein's correction fixes it locally, with no global flag and no effect on
// any other kernel in the module: given a correctly-rounded reciprocal `y`,
//
//     q0 = a * y;  r = fma(-b, q0, a);  q = fma(r, y, q0)
//
// is the correctly-rounded quotient. `y` is folded on the host, where it is
// correctly rounded by construction.
//
// Validated bit-for-bit against IEEE FP32 division over 4.2M values spanning
// the normal range on Arc B60.
// ---------------------------------------------------------------------------
inline float divide_by_e2m1_max_norm(float a) {
  constexpr float kInvE2M1MaxNorm = 1.0f / kE2M1MaxNorm;
  const float approx = a * kInvE2M1MaxNorm;
  const float residual = sycl::fma(-kE2M1MaxNorm, approx, a);
  return sycl::fma(residual, kInvE2M1MaxNorm, approx);
}

// ---------------------------------------------------------------------------
// A0 step 3: rceil shared exponent, in three regimes.
//
//   1. amax == 0                 -> exponent 1 (UE8M0 code 128). Deliberately
//                                   matches quant_mx_rceil, which uses 1 as a
//                                   placeholder for empty groups.
//   2. amax / 6 underflows to 0  -> log2 is -inf and the clamp yields -127.
//                                   The oracle returns NaN here because its
//                                   straight-through ceil evaluates -inf - -inf;
//                                   that is an autograd artifact, not a
//                                   contract, so ARK takes the clamp's value.
//                                   This is the single intentional deviation.
//   3. otherwise                 -> ceil(log2(amax / 6)) clamped to [-127, 127].
//
// The logarithm must be evaluated as "round the real log to FP32, then ceil".
// Computing floor(log2) from the exponent field and adding one for
// non-powers-of-two is NOT equivalent: for v just above 2**e the true log
// e + 1.7e-7 rounds back to exactly e whenever ulp(e) exceeds that offset,
// which holds for |e| >= 4.
//
// `UseDoubleLog` selects how that rounding is realised. The double path
// computes in a wider format and narrows; the float path calls the device
// builtin directly. On Arc B60 the two were measured to agree bit-for-bit with
// each other and with torch on every FP32 input tested, so the float path is
// the default and the double path is kept as an escape hatch for parts whose
// FP32 `log2` turns out to be less accurate.
// ---------------------------------------------------------------------------
template <bool UseDoubleLog>
inline float shared_exponent(float amax) {
  const float ratio = divide_by_e2m1_max_norm(amax);
  float logarithm;
  if constexpr (UseDoubleLog) {
    logarithm = static_cast<float>(sycl::log2(static_cast<double>(ratio)));
  } else {
    logarithm = sycl::log2(ratio);
  }
  const float exponent = (amax == 0.0f) ? kZeroGroupExp : sycl::ceil(logarithm);
  // fmax turns the -inf of regime 2 into kScaleExpMin, as intended.
  return sycl::fmin(sycl::fmax(exponent, kScaleExpMin), kScaleExpMax);
}

// Reciprocal of the exact power-of-two scale, built from its exponent field.
// Multiplying by this is mathematically identical to dividing by the scale and
// avoids the offload backend's approximate FP32 divide. The exponent field
// 127 - e is in [1, 254] for every reachable e: e is at most 126 for finite
// input (amax <= FLT_MAX < 6 * 2**127) and at least -127 by the clamp.
inline float inverse_scale(float exponent) {
  return sycl::bit_cast<float>(static_cast<uint32_t>(127 - static_cast<int>(exponent)) << 23);
}

// A0 steps 4-6 for one already-smoothed value: normalize, clamp, encode, sign.
inline uint8_t e2m1_code(float xh, float inv_scale) {
  const float v = sycl::fmin(sycl::fmax(xh * inv_scale, -kE2M1MaxNorm), kE2M1MaxNorm);
  const uint8_t sign = static_cast<uint8_t>(sycl::bit_cast<uint32_t>(v) >> 31);
  return static_cast<uint8_t>(e2m1_magnitude_code(sycl::fabs(v)) | (sign << 3));
}

// A0 steps 2-7 for one group of 32 already-smoothed FP32 values.
// Fills four 32-bit words of low-nibble-first packed codes (so the caller can
// issue a single 16-byte store) and returns the UE8M0 scale byte.
template <bool UseDoubleLog>
inline uint8_t encode_group(const float* xh, uint32_t* packed_out) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kGroupSize; ++i) {
    amax = sycl::fmax(amax, sycl::fabs(xh[i]));
  }

  const float exponent = shared_exponent<UseDoubleLog>(amax);
  const float inv_scale = inverse_scale(exponent);

  uint8_t codes[kGroupSize];
#pragma unroll
  for (int i = 0; i < kGroupSize; ++i) {
    codes[i] = e2m1_code(xh[i], inv_scale);
  }

#pragma unroll
  for (int w = 0; w < kGroupSize / 8; ++w) {
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = 8 * w + 2 * i;
      word |= static_cast<uint32_t>(codes[c] | (codes[c + 1] << 4)) << (8 * i);
    }
    packed_out[w] = word;
  }

  return static_cast<uint8_t>(static_cast<int>(exponent) + kUE8M0Bias);
}

// Load one 32-element group and apply the A0 step-1 smoothing.
// The reads go through 32-bit words so the compiler emits wide loads: `x` is
// 16-byte aligned at every group boundary (K is a multiple of 32 and torch
// allocations are over-aligned), and both FP16 and BF16 pack two elements per
// word.
template <typename T>
inline void load_smoothed(const T* x_row, const float* smooth_col, float* out) {
  if (smooth_col != nullptr) {
#pragma unroll
    for (int i = 0; i < kGroupSize; ++i) {
      out[i] = static_cast<float>(x_row[i]) * smooth_col[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < kGroupSize; ++i) {
      out[i] = static_cast<float>(x_row[i]);
    }
  }
}

inline void store_group(uint8_t* dst, const uint32_t* words) {
  // 16-byte aligned by construction (see load_smoothed).
  auto* out = reinterpret_cast<sycl::vec<uint32_t, 4>*>(dst);
  *out = sycl::vec<uint32_t, 4>(words[0], words[1], words[2], words[3]);
}

#endif  // ARK_XPU

// ---------------------------------------------------------------------------
// Host entry point.
//
//   x         [M, K]      FP16 or BF16, row-major contiguous
//   smooth    [K]         FP32, or null for "multiply by 1"
//   lora_down [R, K]      FP16 or BF16 row-major contiguous, or null
//   qact      [M, K/2]    uint8, low-nibble-first packed E2M1 (output)
//   ascales   [M, K/32]   uint8, UE8M0 (output)
//   lora_act  [M, R]      same dtype as x (output), ignored when lora_down is
//                         null. The projection accumulates in FP32 and only the
//                         store narrows: Kernel B consumes this as the A operand
//                         of a 16-bit GEMM, so a wider store would be rounded
//                         away at the kernel boundary and only cost bandwidth.
//   workspace             scratch of `svdquant_workspace_elements(m, k, r)`
//                         elements of x's dtype, or null
//
// `x_dtype` and `lora_dtype` are BTLA_DTYPE codes (ARK_DT in Python).
// K must be a multiple of 32; R must be in [1, kMaxRank].
//
// The workspace, when supplied, unlocks the fused sycl-tla path (see
// wrapper/include/sycl_svdquant_mxfp4_cute.hpp). It is optional in the strict
// sense: with a null workspace the scalar fused path produces the same qact and
// ascales bit-for-bit and a lora_act that differs only within the section 4.1
// tolerance. Callers that cannot allocate scratch simply lose the speedup.
// ---------------------------------------------------------------------------
void quant_down(void* stream, const void* x, const void* smooth, const void* lora_down, void* qact, void* ascales,
                void* lora_act, void* workspace, int m, int k, int r, int x_dtype, int lora_dtype);

// Number of elements (of x's dtype) the DPAS path needs as scratch, or 0 when
// this shape has no DPAS path and the caller should pass a null workspace.
// The caller must not reimplement this: the packed layout is private to the
// DPAS header.
std::size_t svdquant_workspace_elements(int m, int k, int r);

}  // namespace svdquant
}  // namespace ark
