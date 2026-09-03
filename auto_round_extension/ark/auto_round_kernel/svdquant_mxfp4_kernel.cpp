// ARK SVDQuant MXFP4 Kernel A -- SYCL implementation.
//
// See `wrapper/include/sycl_svdquant_mxfp4.hpp` for the numerical helpers and
// design doc section 3.3.1 for the frozen A0 contract.
//
// Parallel decomposition
// ----------------------
// One work-group owns `kRowsPerGroup` consecutive rows of `x` and walks the
// whole K dimension. Work-item `lid` owns micro-scaling groups
// `lid, lid + WGS, lid + 2*WGS, ...`, so a work-group covers a contiguous
// `WGS * 32` element span of every row it owns before moving on.
//
// Walking full K inside one work-group is what makes fusing the low-rank down
// projection worthwhile: `lora_down` is streamed once per work-group instead of
// once per row, which cuts its read traffic by `kRowsPerGroup`. The partial
// dot products live in SLM (indexed by work-item, then row, then rank) and are
// reduced in a fixed order at the end, so the result is run-to-run
// deterministic.
//
// The low-rank output is FP32-accumulated but is *not* required to be
// bit-identical to the PyTorch reference: the reduction order differs from
// torch's blocked GEMM. Section 4.1 gates it on tolerance, and gates only
// `qact` / `ascales` on bit-exactness.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "bestla/bestla/bestla.h"
#include "sycl_svdquant_mxfp4.hpp"
#include "sycl_svdquant_mxfp4_cute.hpp"

namespace ark {
namespace svdquant {

#ifdef ARK_XPU

namespace {

constexpr int kWorkGroupSize = 16;   // one sub-group on Xe
constexpr int kQuantWorkGroupSize = 256;  // quant-only path: plain 1-D mapping
constexpr int kFusedWorkGroupSize = 256;  // fused path: 16 sub-groups, one row each
constexpr int kGroupsPerStep = 16;        // scalar fused path: groups per loop iteration (ILP)
constexpr int kRowsPerSubGroup = 1;       // scalar fused path: rows sharing each lora_down load

// Read once. The FP32 and FP64 log2 paths were measured to agree bit-for-bit
// with each other and with torch on Arc B60, so FP32 is the default; the FP64
// path stays reachable as an escape hatch for parts with a weaker FP32 log2.
bool use_double_log() {
  static const bool value = []() {
    const char* raw = std::getenv("ARK_SVDQ_LOG2_FP64");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return value;
}

// Quant-only path: one work-item per micro-scaling group, flat 1-D range.
//
// Without the low-rank projection there is nothing to reuse across rows, so the
// row-blocked decomposition below only costs parallelism. A flat mapping gives
// M * K / 32 work-items -- two orders of magnitude more than the fused path at
// the same problem size -- and makes consecutive work-items read consecutive
// 64-byte spans.
template <typename T, bool UseDoubleLog>
void launch_quant_only(sycl::queue* queue, const T* x, const float* smooth, uint8_t* qact, uint8_t* ascales, int m,
                       int k) {
  const int num_groups = k / kGroupSize;
  const int packed_row_stride = k / 2;
  const size_t total = static_cast<size_t>(m) * num_groups;
  const size_t padded = ((total + kQuantWorkGroupSize - 1) / kQuantWorkGroupSize) * kQuantWorkGroupSize;

  queue->submit([&](sycl::handler& handler) {
    handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(kQuantWorkGroupSize)),
                         [=](sycl::nd_item<1> item) {
                           const size_t tid = item.get_global_linear_id();
                           if (tid >= total) return;
                           const int row = static_cast<int>(tid / num_groups);
                           const int group = static_cast<int>(tid % num_groups);
                           const int col_base = group * kGroupSize;

                           float smoothed[kGroupSize];
                           load_smoothed(x + static_cast<size_t>(row) * k + col_base,
                                         smooth == nullptr ? nullptr : smooth + col_base, smoothed);

                           uint32_t words[kGroupSize / 8];
                           ascales[static_cast<size_t>(row) * num_groups + group] =
                               encode_group<UseDoubleLog>(smoothed, words);
                           store_group(qact + static_cast<size_t>(row) * packed_row_stride + col_base / 2, words);
                         });
  });
}

// Fused path: quantization plus the low-rank down projection.
//
// One sub-group owns one row of `x`. Within a row the sub-group walks the
// micro-scaling groups *together*: for group `g`, lane `l` owns the element
// pair `(2l, 2l + 1)`, so 16 lanes cover all 32 elements.
//
// Having the whole sub-group sit on the same group is what makes the low-rank
// inner loop fast. `lora_down[rank][col_base + 2l .. +1]` is then contiguous
// across lanes, so each rank costs one coalesced 64-byte load. The earlier
// lane-per-group shape made the same access a 16-way gather (lanes 64 bytes
// apart) and ran ~10x slower.
//
// Each lane keeps a private `acc[rank]` covering only its own element pair,
// accumulated across every group of the row, so the sub-group reduction happens
// once per rank at the very end rather than once per group. `reduce_over_group`
// uses a fixed shuffle tree, so `lora_act` is run-to-run deterministic; it is
// not bit-identical to a torch GEMM (different reduction order) and §4.1 gates
// it on tolerance.
//
// No SLM is used at all. An earlier version staged the smoothed tile and the
// partial dot products in SLM, costing 32 KB per work-group; that capped
// residency at a handful of work-groups per Xe-core and ran ~25x slower than
// the memory-bandwidth bound despite doing identical arithmetic.
template <typename T, bool UseDoubleLog, int MaxRank>
void launch_fused(sycl::queue* queue, const T* x, const float* smooth, const T* lora_down, uint8_t* qact,
                  uint8_t* ascales, T* lora_act, int m, int k, int r) {
  const int num_groups = k / kGroupSize;
  const int packed_row_stride = k / 2;
  constexpr int kSubGroupsPerGroup = kFusedWorkGroupSize / kWorkGroupSize;
  const int rows_per_block = kSubGroupsPerGroup * kRowsPerSubGroup;
  const int num_row_blocks = (m + rows_per_block - 1) / rows_per_block;

  queue->submit([&](sycl::handler& handler) {
    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_row_blocks) * kFusedWorkGroupSize),
                          sycl::range<1>(kFusedWorkGroupSize)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kWorkGroupSize)]] {
          const sycl::sub_group sg = item.get_sub_group();
          const int lane = static_cast<int>(sg.get_local_linear_id());
          const int row_base = (static_cast<int>(item.get_group(0)) * kSubGroupsPerGroup +
                                static_cast<int>(sg.get_group_linear_id())) *
                               kRowsPerSubGroup;
          // The whole sub-group leaves together, so no collective is skipped.
          if (row_base >= m) return;

          // Rows past the end alias row 0 so the unrolled bodies stay branch
          // free; their stores are masked off instead.
          int rows[kRowsPerSubGroup];
          bool live[kRowsPerSubGroup];
#pragma unroll
          for (int t = 0; t < kRowsPerSubGroup; ++t) {
            live[t] = (row_base + t) < m;
            rows[t] = live[t] ? row_base + t : row_base;
          }

          float acc[kRowsPerSubGroup][MaxRank];
#pragma unroll
          for (int t = 0; t < kRowsPerSubGroup; ++t) {
#pragma unroll
            for (int rank = 0; rank < MaxRank; ++rank) acc[t][rank] = 0.0f;
          }

          const int lane_offset = 2 * lane;
          constexpr int kStep = kGroupsPerStep;
          const int steps = num_groups / kStep;

          for (int step = 0; step < steps; ++step) {
            float a[kRowsPerSubGroup][kStep], b[kRowsPerSubGroup][kStep];
            int col[kStep];
#pragma unroll
            for (int u = 0; u < kStep; ++u) {
              col[u] = (step * kStep + u) * kGroupSize + lane_offset;
            }
#pragma unroll
            for (int t = 0; t < kRowsPerSubGroup; ++t) {
              const T* x_row = x + static_cast<size_t>(rows[t]) * k;
#pragma unroll
              for (int u = 0; u < kStep; ++u) {
                a[t][u] = static_cast<float>(x_row[col[u]]);
                b[t][u] = static_cast<float>(x_row[col[u] + 1]);
                if (smooth != nullptr) {
                  a[t][u] *= smooth[col[u]];
                  b[t][u] *= smooth[col[u] + 1];
                }
              }
            }

#pragma unroll
            for (int t = 0; t < kRowsPerSubGroup; ++t) {
              if (!live[t]) continue;
              uint8_t* qact_row = qact + static_cast<size_t>(rows[t]) * packed_row_stride;
              uint8_t* ascale_row = ascales + static_cast<size_t>(rows[t]) * num_groups;
#pragma unroll
              for (int u = 0; u < kStep; ++u) {
                const float amax = sycl::reduce_over_group(
                    sg, sycl::fmax(sycl::fabs(a[t][u]), sycl::fabs(b[t][u])), sycl::maximum<float>());
                const float exponent = shared_exponent<UseDoubleLog>(amax);
                const float inv_scale = inverse_scale(exponent);
                qact_row[(step * kStep + u) * (kGroupSize / 2) + lane] = static_cast<uint8_t>(
                    e2m1_code(a[t][u], inv_scale) | (e2m1_code(b[t][u], inv_scale) << 4));
                if (lane == 0) {
                  ascale_row[step * kStep + u] = static_cast<uint8_t>(static_cast<int>(exponent) + kUE8M0Bias);
                }
              }
            }

            // Each lora_down element loaded here feeds kRowsPerSubGroup rows,
            // which is what keeps its re-read traffic bounded at large K.
            for (int rank = 0; rank < r; ++rank) {
              const T* w = lora_down + static_cast<size_t>(rank) * k;
              float wa[kStep], wb[kStep];
#pragma unroll
              for (int u = 0; u < kStep; ++u) {
                wa[u] = static_cast<float>(w[col[u]]);
                wb[u] = static_cast<float>(w[col[u] + 1]);
              }
#pragma unroll
              for (int t = 0; t < kRowsPerSubGroup; ++t) {
#pragma unroll
                for (int u = 0; u < kStep; ++u) {
                  acc[t][rank] = sycl::fma(a[t][u], wa[u], sycl::fma(b[t][u], wb[u], acc[t][rank]));
                }
              }
            }
          }

          // Tail: groups not covered by a full step.
          for (int group = steps * kStep; group < num_groups; ++group) {
            const int col = group * kGroupSize + lane_offset;
            float a[kRowsPerSubGroup], b[kRowsPerSubGroup];
#pragma unroll
            for (int t = 0; t < kRowsPerSubGroup; ++t) {
              const T* x_row = x + static_cast<size_t>(rows[t]) * k;
              a[t] = static_cast<float>(x_row[col]);
              b[t] = static_cast<float>(x_row[col + 1]);
              if (smooth != nullptr) {
                a[t] *= smooth[col];
                b[t] *= smooth[col + 1];
              }
              const float amax =
                  sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(a[t]), sycl::fabs(b[t])), sycl::maximum<float>());
              const float exponent = shared_exponent<UseDoubleLog>(amax);
              const float inv_scale = inverse_scale(exponent);
              if (live[t]) {
                qact[static_cast<size_t>(rows[t]) * packed_row_stride + group * (kGroupSize / 2) + lane] =
                    static_cast<uint8_t>(e2m1_code(a[t], inv_scale) | (e2m1_code(b[t], inv_scale) << 4));
                if (lane == 0) {
                  ascales[static_cast<size_t>(rows[t]) * num_groups + group] =
                      static_cast<uint8_t>(static_cast<int>(exponent) + kUE8M0Bias);
                }
              }
            }
            for (int rank = 0; rank < r; ++rank) {
              const T* w = lora_down + static_cast<size_t>(rank) * k + col;
              const float wa = static_cast<float>(w[0]);
              const float wb = static_cast<float>(w[1]);
#pragma unroll
              for (int t = 0; t < kRowsPerSubGroup; ++t) {
                acc[t][rank] = sycl::fma(a[t], wa, sycl::fma(b[t], wb, acc[t][rank]));
              }
            }
          }

          for (int rank = 0; rank < r; ++rank) {
#pragma unroll
            for (int t = 0; t < kRowsPerSubGroup; ++t) {
              const float total = sycl::reduce_over_group(sg, acc[t][rank], sycl::plus<float>());
              // Spread the stores across lanes instead of funnelling them through lane 0.
              if (live[t] && lane == (rank & (kWorkGroupSize - 1))) {
                lora_act[static_cast<size_t>(rows[t]) * r + rank] = static_cast<T>(total);
              }
            }
          }
        });
  });
}

// `ARK_SVDQUANT_DISABLE_CUTE=1` forces the scalar fused path, which is what the
// tests use to compare the two against each other.
//
// Deliberately NOT cached in a function-local static: the tests flip this
// between two launches in the same process, and a cached value would silently
// make that comparison vacuous (it did, once, with the equivalent flag on the
// since-deleted joint_matrix path). getenv runs once per launch on the host,
// which is nothing next to the kernel itself.
bool cute_disabled() {
#ifdef ARK_SYCL_TLA
  const char* value = std::getenv("ARK_SVDQUANT_DISABLE_CUTE");
  return value != nullptr && value[0] == '1';
#else
  return true;
#endif
}

// Split the fused sycl-tla kernel over the column-tile count. R is a multiple
// of kTileN and at most kMaxNTiles * kTileN here, both enforced by
// cute_lora_supported, so exactly one arm is reachable.
//
// Row blocking is fixed at kFusedCuteRows and therefore not a dispatch
// dimension: the fused kernel pins one lane to one row for the quantization
// phase, so it cannot vary independently.
template <typename T>
void launch_fused_cute_dispatch(sycl::queue* queue, const T* x, const float* smooth, const T* hi, const T* lo,
                                uint8_t* qact, uint8_t* ascales, T* lora_act, float* partial, int m, int k, int r,
                                int slices) {
#ifdef ARK_SYCL_TLA
  const bool double_log = use_double_log();
  switch (lora_n_tiles(r)) {
    case 1:
      if (double_log) launch_fused_cute<T, true, 1>(queue, x, smooth, hi, lo, qact, ascales, lora_act, partial, m, k, r, slices);
      else launch_fused_cute<T, false, 1>(queue, x, smooth, hi, lo, qact, ascales, lora_act, partial, m, k, r, slices);
      break;
    case 2:
      if (double_log) launch_fused_cute<T, true, 2>(queue, x, smooth, hi, lo, qact, ascales, lora_act, partial, m, k, r, slices);
      else launch_fused_cute<T, false, 2>(queue, x, smooth, hi, lo, qact, ascales, lora_act, partial, m, k, r, slices);
      break;
    case 3:
      if (double_log) launch_fused_cute<T, true, 3>(queue, x, smooth, hi, lo, qact, ascales, lora_act, partial, m, k, r, slices);
      else launch_fused_cute<T, false, 3>(queue, x, smooth, hi, lo, qact, ascales, lora_act, partial, m, k, r, slices);
      break;
    default:
      if (double_log) launch_fused_cute<T, true, 4>(queue, x, smooth, hi, lo, qact, ascales, lora_act, partial, m, k, r, slices);
      else launch_fused_cute<T, false, 4>(queue, x, smooth, hi, lo, qact, ascales, lora_act, partial, m, k, r, slices);
      break;
  }
#endif
}

template <typename T>
void dispatch_flags(sycl::queue* queue, const void* x, const void* smooth, const void* lora_down, void* qact,
                    void* ascales, void* lora_act, void* workspace, int m, int k, int r) {
  const T* x_ptr = static_cast<const T*>(x);
  const float* smooth_ptr = static_cast<const float*>(smooth);
  const T* lora_ptr = static_cast<const T*>(lora_down);
  uint8_t* qact_ptr = static_cast<uint8_t*>(qact);
  uint8_t* ascales_ptr = static_cast<uint8_t*>(ascales);
  T* lora_act_ptr = static_cast<T*>(lora_act);

  const bool use_cute = lora_ptr != nullptr && workspace != nullptr && !cute_disabled() && cute_lora_supported(m, k, r);

  if (use_cute) {
    // Single launch over `x` for both branches. The prologue that builds the
    // split B planes is a separate launch, but it only touches a [K, R<=64]
    // matrix and never reads `x`.
    T* hi = static_cast<T*>(workspace);
    T* lo = hi + cute_b_plane_elements(k, r);
    // The FP32 partial planes share the same allocation, past the two B planes.
    // The caller sized it with svdquant_workspace_elements, which uses the very
    // same cute_k_slices, so the two cannot disagree.
    const int slices = cute_k_slices(m, k, r);
    float* partial = slices > 1 ? reinterpret_cast<float*>(lo + cute_b_plane_elements(k, r)) : nullptr;
    launch_pack_lora_b_cute<T>(queue, smooth_ptr, lora_ptr, hi, lo, k, r);
    launch_fused_cute_dispatch<T>(queue, x_ptr, smooth_ptr, hi, lo, qact_ptr, ascales_ptr, lora_act_ptr, partial, m, k,
                                  r, slices);
    if (slices > 1) launch_reduce_partials_cute<T>(queue, partial, lora_act_ptr, m, r, slices);
    return;
  }

  if (lora_ptr == nullptr) {
    if (use_double_log()) {
      launch_quant_only<T, true>(queue, x_ptr, smooth_ptr, qact_ptr, ascales_ptr, m, k);
    } else {
      launch_quant_only<T, false>(queue, x_ptr, smooth_ptr, qact_ptr, ascales_ptr, m, k);
    }
    return;
  }
  // MaxRank bounds the per-lane register accumulator; picking the smallest
  // bucket that fits keeps register pressure (and therefore occupancy) down.
  if (use_double_log()) {
    if (r <= 16) launch_fused<T, true, 16>(queue, x_ptr, smooth_ptr, lora_ptr, qact_ptr, ascales_ptr, lora_act_ptr, m, k, r);
    else if (r <= 32) launch_fused<T, true, 32>(queue, x_ptr, smooth_ptr, lora_ptr, qact_ptr, ascales_ptr, lora_act_ptr, m, k, r);
    else launch_fused<T, true, kMaxRank>(queue, x_ptr, smooth_ptr, lora_ptr, qact_ptr, ascales_ptr, lora_act_ptr, m, k, r);
  } else {
    if (r <= 16) launch_fused<T, false, 16>(queue, x_ptr, smooth_ptr, lora_ptr, qact_ptr, ascales_ptr, lora_act_ptr, m, k, r);
    else if (r <= 32) launch_fused<T, false, 32>(queue, x_ptr, smooth_ptr, lora_ptr, qact_ptr, ascales_ptr, lora_act_ptr, m, k, r);
    else launch_fused<T, false, kMaxRank>(queue, x_ptr, smooth_ptr, lora_ptr, qact_ptr, ascales_ptr, lora_act_ptr, m, k, r);
  }
}

}  // namespace

#endif  // ARK_XPU

std::size_t svdquant_workspace_elements(int m, int k, int r) {
  // Two B planes (hi and lo) followed by the FP32 projection partials. Counted
  // in *output-dtype* elements, which are 16-bit, so each FP32 partial costs
  // two. The partial region is absent when a single K slice covers the problem.
  if (!cute_lora_supported(m, k, r)) return 0;
  const std::size_t b = 2 * cute_b_plane_elements(k, r);
  return b + 2 * cute_partial_elements(m, r, cute_k_slices(m, k, r));
}

void quant_down(void* stream, const void* x, const void* smooth, const void* lora_down, void* qact, void* ascales,
                void* lora_act, void* workspace, int m, int k, int r, int x_dtype, int lora_dtype) {
#ifdef ARK_XPU
  if (stream == nullptr) throw std::invalid_argument("ark::svdquant::quant_down: stream is null");
  if (x == nullptr || qact == nullptr || ascales == nullptr) {
    throw std::invalid_argument("ark::svdquant::quant_down: x, qact and ascales are required");
  }
  if (m <= 0 || k <= 0) throw std::invalid_argument("ark::svdquant::quant_down: m and k must be positive");
  if (k % kGroupSize != 0) {
    throw std::invalid_argument("ark::svdquant::quant_down: k must be a multiple of 32");
  }
  if (lora_down != nullptr) {
    if (r <= 0 || r > kMaxRank) {
      throw std::invalid_argument("ark::svdquant::quant_down: rank must be in [1, 64] when lora_down is given");
    }
    if (lora_act == nullptr) {
      throw std::invalid_argument("ark::svdquant::quant_down: lora_act is required when lora_down is given");
    }
    if (lora_dtype != x_dtype) {
      throw std::invalid_argument("ark::svdquant::quant_down: lora_down dtype must match x dtype");
    }
  }

  sycl::queue* queue = static_cast<sycl::queue*>(stream);
  switch (static_cast<BTLA_DTYPE>(x_dtype)) {
    case BTLA_DTYPE::F16:
      dispatch_flags<sycl::half>(queue, x, smooth, lora_down, qact, ascales, lora_act, workspace, m, k, r);
      break;
    case BTLA_DTYPE::BF16:
      dispatch_flags<sycl::ext::oneapi::bfloat16>(queue, x, smooth, lora_down, qact, ascales, lora_act, workspace, m,
                                                  k, r);
      break;
    default:
      throw std::invalid_argument("ark::svdquant::quant_down: x dtype must be float16 or bfloat16");
  }
#else
  (void)stream; (void)x; (void)smooth; (void)lora_down; (void)qact; (void)ascales; (void)lora_act; (void)workspace;
  (void)m; (void)k; (void)r; (void)x_dtype; (void)lora_dtype;
  throw std::runtime_error("ark::svdquant::quant_down is only supported on XPU");
#endif
}

}  // namespace svdquant
}  // namespace ark
