// Copyright (c) 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// sycl-tla (CuTe) low-rank down projection for SVDQuant Kernel A.
//
// The motivation, the algebraic rewrite, the split-16-bit precision scheme and
// the three probed hardware contracts are all documented at the top of
// sycl_svdquant_mxfp4_cute.hpp. None of it is repeated here.

#include "wrapper/include/sycl_svdquant_mxfp4_cute.hpp"

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

#include <sycl/sycl.hpp>

#include <cstdlib>

#include <cute/tensor.hpp>

#include <cute/arch/copy_xe_2d.hpp>
#include <cute/arch/mma_xe.hpp>

namespace ark {
namespace svdquant {

namespace {

constexpr int kCuteSubGroupsPerWG = 16;
constexpr int kCuteWorkGroupSize = kCuteSubGroupsPerWG * kDpasSubGroup;
constexpr int kCutePackWorkGroupSize = 256;

// The DPAS atoms are declared over CuTe's own 16-bit types. These are
// layout-compatible with the SYCL types the rest of the kernel is templated on
// (both are trivially copyable 16-bit wrappers), and only ever reach the atoms
// as raw register storage, so the mapping is a reinterpret and nothing more.
template <typename T>
struct CuteElem;
template <>
struct CuteElem<sycl::ext::oneapi::bfloat16> {
  using type = cute::bfloat16_t;
};
template <>
struct CuteElem<sycl::half> {
  using type = cute::half_t;
};

// Row blocking as a static member rather than a constexpr local: a lambda with
// an explicit capture list cannot implicitly capture a function-local variable
// even when it is constexpr, but static members need no capture at all.
template <int Rows>
struct CuteTile {
  static constexpr int MTiles = Rows / kTileM;
};

}  // namespace

int cute_rows_for(int m, int n_tiles) {
  const int chosen = cute_rows_for_shape(m, n_tiles);
  const char* override_value = std::getenv("ARK_SVDQUANT_CUTE_ROWS");
  if (override_value == nullptr) return chosen;
  const int requested = std::atoi(override_value);
  if (requested != 8 && requested != 16 && requested != 32) return chosen;
  // Never let the override reintroduce the accumulator spill.
  const int register_cap = n_tiles <= 2 ? 32 : 16;
  return requested < register_cap ? requested : register_cap;
}

// ---------------------------------------------------------------------------
// Prologue: B[k, r] = smooth[k] * lora_down[r, k] in FP32, split so that
// B ~= hi + lo with both parts in the DPAS element type T.
//
// Emitted as a plain row-major [K, cute_b_cols(r)] matrix: the VNNI interleave
// the joint_matrix path had to do by hand is performed by the load
// instruction. Padding columns are zeroed so the padded surface is inert.
// ---------------------------------------------------------------------------
int cute_target_subgroups() {
  const char* value = std::getenv("ARK_SVDQUANT_CUTE_TARGET_SUBGROUPS");
  if (value == nullptr) return kCuteTargetSubGroups;
  const int parsed = std::atoi(value);
  return parsed > 0 ? parsed : kCuteTargetSubGroups;
}

template <typename T>
void launch_pack_lora_b_cute(sycl::queue* queue, const float* smooth, const T* lora_down, T* hi, T* lo, int k, int r) {
  const int cols = cute_b_cols(r);
  const size_t total = static_cast<size_t>(k) * cols;
  const size_t rounded = ((total + kCutePackWorkGroupSize - 1) / kCutePackWorkGroupSize) * kCutePackWorkGroupSize;

  queue->submit([&](sycl::handler& handler) {
    handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(rounded), sycl::range<1>(kCutePackWorkGroupSize)),
                         [=](sycl::nd_item<1> item) {
                           const size_t index = item.get_global_linear_id();
                           if (index >= total) return;

                           const int k_index = static_cast<int>(index / cols);
                           const int r_index = static_cast<int>(index % cols);

                           float value = 0.0f;
                           if (r_index < r) {
                             // lora_down is [R, K], so the stride between ranks is k.
                             value = static_cast<float>(lora_down[static_cast<size_t>(r_index) * k + k_index]);
                             if (smooth != nullptr) value *= smooth[k_index];
                           }

                           const T value_hi = static_cast<T>(value);
                           const T value_lo = static_cast<T>(value - static_cast<float>(value_hi));

                           hi[index] = value_hi;
                           lo[index] = value_lo;
                         });
  });
}

// ---------------------------------------------------------------------------
// lora_act[M, R] = x[M, K] @ (B_hi + B_lo).
//
// One sub-group owns kCuteRows = 32 rows and all NTiles column tiles, so the
// A registers fetched for a K step are reused across every column tile and
// both split planes -- A is read from global memory exactly once.
//
// Rows past M are handled by the load's own bounds checking (contract 3), so
// there is no tail kernel and no minimum M; only the stores are guarded.
// ---------------------------------------------------------------------------
template <typename T, int NTiles, int Rows>
void launch_lora_cute(sycl::queue* queue, const T* x, const T* hi, const T* lo, T* lora_act, int m, int k, int r) {
  using Elem = typename CuteElem<T>::type;
  using Tile = CuteTile<Rows>;
  using LoadA = cute::XE_LOAD_2D<16, Rows, kCuteKStep, kTileK>;
  using LoadB = cute::XE_LOAD_2D_VNNI<16, kTileK, kTileN>;
  using Dpas = cute::XE_DPAS_TT<kTileM, float, Elem, Elem, float>;

  const int row_tiles = (m + Rows - 1) / Rows;
  const int work_groups = (row_tiles + kCuteSubGroupsPerWG - 1) / kCuteSubGroupsPerWG;
  const int b_cols = cute_b_cols(r);

  queue->submit([&](sycl::handler& handler) {
    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(work_groups) * kCuteWorkGroupSize),
                          sycl::range<1>(kCuteWorkGroupSize)),
        // The kernel body sits inside #ifdef __SYCL_DEVICE_ONLY__ because the
        // 2D block-load builtins have no host declaration. That makes an
        // implicit [=] capture *nothing* during host compilation and eight
        // things during device compilation, which trips the SYCL lambda-size
        // static assert. The capture list is therefore explicit.
        [x, hi, lo, lora_act, m, k, r, b_cols](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kDpasSubGroup)]] {
#ifdef __SYCL_DEVICE_ONLY__
          const auto sg = item.get_sub_group();
          const int sg_index = static_cast<int>(item.get_group(0)) * kCuteSubGroupsPerWG +
                               static_cast<int>(sg.get_group_linear_id());
          const int row0 = sg_index * Rows;
          // Uniform across the sub-group: the 2D loads below are sub-group
          // collectives and must not be entered by only part of a sub-group.
          if (row0 >= m) return;

          // Payload fields are "minus one" encoded; width and pitch are in
          // bytes, height in rows. Recording the true height (m, k) is what
          // makes the hardware zero-fill rather than read past the end.
          int* pa = __builtin_IB_subgroup_createBlock2DAddressPayload(
              reinterpret_cast<long>(x), k * 2 - 1, m - 1, k * 2 - 1, 0, row0, kTileK, Rows, kCuteKTiles);
          int* pb_hi = __builtin_IB_subgroup_createBlock2DAddressPayload(
              reinterpret_cast<long>(hi), b_cols * 2 - 1, k - 1, b_cols * 2 - 1, 0, 0, kTileN, kTileK, 1);
          int* pb_lo = __builtin_IB_subgroup_createBlock2DAddressPayload(
              reinterpret_cast<long>(lo), b_cols * 2 - 1, k - 1, b_cols * 2 - 1, 0, 0, kTileN, kTileK, 1);

          typename Dpas::DVector acc[NTiles][Tile::MTiles] = {};

          for (int k0 = 0; k0 < k; k0 += kCuteKStep) {
            // A tile (kt, mt) lands at a[kt * Tile::MTiles + mt] -- contract 1.
            typename Dpas::AVector a[kCuteKTiles * Tile::MTiles];
            __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(pa, k0);
            LoadA::copy(pa, reinterpret_cast<Elem*>(&a[0]));

#pragma unroll
            for (int n = 0; n < NTiles; ++n) {
              __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(pb_hi, n * kTileN);
              __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(pb_lo, n * kTileN);
#pragma unroll
              for (int kt = 0; kt < kCuteKTiles; ++kt) {
                typename Dpas::BVector b_hi;
                typename Dpas::BVector b_lo;
                __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(pb_hi, k0 + kt * kTileK);
                __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(pb_lo, k0 + kt * kTileK);
                LoadB::copy(pb_hi, reinterpret_cast<Elem*>(&b_hi));
                LoadB::copy(pb_lo, reinterpret_cast<Elem*>(&b_lo));

#pragma unroll
                for (int mt = 0; mt < Tile::MTiles; ++mt) {
                  const auto& a_vec = a[kt * Tile::MTiles + mt];
                  // Both split planes fold into the same FP32 accumulator, so
                  // the residual is summed exactly once and never re-rounded.
                  typename Dpas::DVector tmp;
                  Dpas::fma(tmp, a_vec, b_hi, acc[n][mt]);
                  Dpas::fma(acc[n][mt], a_vec, b_lo, tmp);
                }
              }
            }
          }

          // Accumulator layout: lane `l` of tile (n, mt) holds column
          // n * kTileN + l of rows row0 + mt * kTileM + i.
          const int lane = static_cast<int>(sg.get_local_linear_id());
#pragma unroll
          for (int n = 0; n < NTiles; ++n) {
#pragma unroll
            for (int mt = 0; mt < Tile::MTiles; ++mt) {
#pragma unroll
              for (int i = 0; i < kTileM; ++i) {
                const int row = row0 + mt * kTileM + i;
                if (row < m) lora_act[static_cast<size_t>(row) * r + n * kTileN + lane] = static_cast<T>(acc[n][mt][i]);
              }
            }
          }
#endif  // __SYCL_DEVICE_ONLY__
        });
  });
}

// ---------------------------------------------------------------------------
// FUSED: quantization + low-rank projection in a single launch.
//
// This is the kernel the design doc's §4.4 criterion 2 asks for. The separate
// launch_lora_cute above leaves `x` being read twice -- once by the quantization
// kernel and once by the projection -- and at 4608x3072 that second DRAM pass
// was very nearly the entire difference between quant-only (0.134 ms) and the
// two-launch fused result (0.232 ms).
//
// THE DECOMPOSITION HINGES ON kGroupSize == kCuteKStep
//
// Both are 32. That is what makes the fusion clean rather than a compromise:
// one K step of the DPAS loop covers exactly 32 columns, which is exactly one
// micro-scaling group per row. So each iteration can finish the quantization of
// the rows it just loaded, with no leftover state and no second traversal.
//
// A sub-group owns kFusedCuteRows = 16 rows, chosen so that the two phases fit
// the same 16 lanes:
//
//   * Quantization: lane `l` owns row `row0 + l` entirely, and runs the *same*
//     load_smoothed / encode_group / store_group sequence as the standalone
//     quant kernel. Reusing those three verbatim is deliberate -- the A0
//     numerical contract of §3.3.1 is bit-exact validated against a PyTorch
//     oracle, and re-deriving the encoding from the DPAS register layout would
//     have put that at risk for no benefit.
//   * Projection: 16 rows is 2 DPAS M-tiles, and lane `l` holds accumulator
//     column `l`.
//
// The two phases therefore want `x` in different per-lane layouts, which is why
// the quantization re-reads it rather than working out of the A registers. That
// re-read is an L1 hit: XE_LOAD_2D has just pulled the identical 16 x 64 B tile
// into cache in the same iteration. The saving being chased here is the DRAM
// round trip, not the L1 access.
// ---------------------------------------------------------------------------
template <typename T, bool UseDoubleLog, int NTiles>
void launch_fused_cute(sycl::queue* queue, const T* x, const float* smooth, const T* hi, const T* lo, uint8_t* qact,
                       uint8_t* ascales, T* lora_act, float* partial, int m, int k, int r, int slices) {
  using Elem = typename CuteElem<T>::type;
  constexpr int Rows = kFusedCuteRows;
  constexpr int MTiles = Rows / kTileM;
  using LoadA = cute::XE_LOAD_2D<16, Rows, kCuteKStep, kTileK>;
  using LoadB = cute::XE_LOAD_2D_VNNI<16, kTileK, kTileN>;
  using Dpas = cute::XE_DPAS_TT<kTileM, float, Elem, Elem, float>;

  const int row_tiles = (m + Rows - 1) / Rows;
  const int num_groups = k / kGroupSize;
  const int groups_per_slice = (num_groups + slices - 1) / slices;
  const int sub_groups = row_tiles * slices;
  const int work_groups = (sub_groups + kCuteFusedSubGroupsPerWG - 1) / kCuteFusedSubGroupsPerWG;
  const int b_cols = cute_b_cols(r);
  const int packed_row_stride = k / 2;
  const bool split = slices > 1;

  queue->submit([&](sycl::handler& handler) {
    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(work_groups) * kCuteFusedWorkGroupSize),
                          sycl::range<1>(kCuteFusedWorkGroupSize)),
        // Explicit capture list: see the note on launch_lora_cute above.
        [x, smooth, hi, lo, qact, ascales, lora_act, partial, m, k, r, b_cols, num_groups, packed_row_stride, slices,
         groups_per_slice, sub_groups, split](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kDpasSubGroup)]] {
#ifdef __SYCL_DEVICE_ONLY__
          const auto sg = item.get_sub_group();
          const int sg_index = static_cast<int>(item.get_group(0)) * kCuteFusedSubGroupsPerWG +
                               static_cast<int>(sg.get_group_linear_id());
          // Uniform across the sub-group: the 2D loads are sub-group
          // collectives and must not be entered by only part of a sub-group.
          if (sg_index >= sub_groups) return;

          // Slice varies fastest so that the four sub-groups sharing a
          // work-group also share a row block, and therefore the same rows of
          // `x` -- they read disjoint K ranges of them, but from pages the
          // memory system has already been asked for.
          const int slice = sg_index % slices;
          const int row0 = (sg_index / slices) * Rows;

          const int g0 = slice * groups_per_slice;
          int g1 = g0 + groups_per_slice;
          if (g1 > num_groups) g1 = num_groups;

          const int lane = static_cast<int>(sg.get_local_linear_id());
          const int quant_row = row0 + lane;
          const bool quant_active = quant_row < m;

          int* pa = __builtin_IB_subgroup_createBlock2DAddressPayload(
              reinterpret_cast<long>(x), k * 2 - 1, m - 1, k * 2 - 1, 0, row0, kTileK, Rows, kCuteKTiles);
          int* pb_hi = __builtin_IB_subgroup_createBlock2DAddressPayload(
              reinterpret_cast<long>(hi), b_cols * 2 - 1, k - 1, b_cols * 2 - 1, 0, 0, kTileN, kTileK, 1);
          int* pb_lo = __builtin_IB_subgroup_createBlock2DAddressPayload(
              reinterpret_cast<long>(lo), b_cols * 2 - 1, k - 1, b_cols * 2 - 1, 0, 0, kTileN, kTileK, 1);

          typename Dpas::DVector acc[NTiles][MTiles] = {};

          for (int g = g0; g < g1; ++g) {
            const int k0 = g * kGroupSize;

            // --- projection: A tile (kt, mt) lands at a[kt * MTiles + mt] ---
            typename Dpas::AVector a[kCuteKTiles * MTiles];
            __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(pa, k0);
            LoadA::copy(pa, reinterpret_cast<Elem*>(&a[0]));

#pragma unroll
            for (int n = 0; n < NTiles; ++n) {
              __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(pb_hi, n * kTileN);
              __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(pb_lo, n * kTileN);
#pragma unroll
              for (int kt = 0; kt < kCuteKTiles; ++kt) {
                typename Dpas::BVector b_hi;
                typename Dpas::BVector b_lo;
                __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(pb_hi, k0 + kt * kTileK);
                __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(pb_lo, k0 + kt * kTileK);
                LoadB::copy(pb_hi, reinterpret_cast<Elem*>(&b_hi));
                LoadB::copy(pb_lo, reinterpret_cast<Elem*>(&b_lo));

#pragma unroll
                for (int mt = 0; mt < MTiles; ++mt) {
                  const auto& a_vec = a[kt * MTiles + mt];
                  typename Dpas::DVector tmp;
                  Dpas::fma(tmp, a_vec, b_hi, acc[n][mt]);
                  Dpas::fma(acc[n][mt], a_vec, b_lo, tmp);
                }
              }
            }

            // --- quantization of the same 16 x 32 tile, one row per lane ---
            //
            // No cross-slice interaction: this slice owns groups [g0, g1) for
            // these rows outright, and every store below is to bytes no other
            // sub-group touches.
            if (quant_active) {
              float smoothed[kGroupSize];
              load_smoothed(x + static_cast<size_t>(quant_row) * k + k0, smooth == nullptr ? nullptr : smooth + k0,
                            smoothed);
              uint32_t words[kGroupSize / 8];
              ascales[static_cast<size_t>(quant_row) * num_groups + g] = encode_group<UseDoubleLog>(smoothed, words);
              store_group(qact + static_cast<size_t>(quant_row) * packed_row_stride + k0 / 2, words);
            }
          }

          // Accumulator: lane `l` of tile (n, mt) holds column n * kTileN + l.
          // With one slice this is the final answer and lands in `lora_act`
          // directly; otherwise it is a partial sum for the epilogue.
          float* out_f32 = split ? partial + static_cast<size_t>(slice) * m * r : nullptr;
#pragma unroll
          for (int n = 0; n < NTiles; ++n) {
#pragma unroll
            for (int mt = 0; mt < MTiles; ++mt) {
#pragma unroll
              for (int i = 0; i < kTileM; ++i) {
                const int row = row0 + mt * kTileM + i;
                if (row >= m) continue;
                const size_t offset = static_cast<size_t>(row) * r + n * kTileN + lane;
                if (split) out_f32[offset] = acc[n][mt][i];
                else lora_act[offset] = static_cast<T>(acc[n][mt][i]);
              }
            }
          }
#endif  // __SYCL_DEVICE_ONLY__
        });
  });
}

// Sum the per-slice FP32 planes and narrow to the output dtype.
//
// Summed in slice order in FP32, so the result is deterministic -- which is why
// this exists instead of an FP32 atomic add in the kernel above. It reads
// slices * M * R floats, a few MB, and is negligible beside the `x` pass.
template <typename T>
void launch_reduce_partials_cute(sycl::queue* queue, const float* partial, T* lora_act, int m, int r, int slices) {
  const size_t total = static_cast<size_t>(m) * static_cast<size_t>(r);
  const size_t plane = total;
  const size_t rounded = ((total + kCutePackWorkGroupSize - 1) / kCutePackWorkGroupSize) * kCutePackWorkGroupSize;

  queue->submit([&](sycl::handler& handler) {
    handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(rounded), sycl::range<1>(kCutePackWorkGroupSize)),
                         [=](sycl::nd_item<1> item) {
                           const size_t index = item.get_global_linear_id();
                           if (index >= total) return;
                           float sum = 0.0f;
                           for (int s = 0; s < slices; ++s) sum += partial[static_cast<size_t>(s) * plane + index];
                           lora_act[index] = static_cast<T>(sum);
                         });
  });
}

// Explicit instantiations: the dispatcher picks NTiles from R at runtime.
#define ARK_SVDQ_CUTE_INSTANTIATE(T)                                                                   \
  template void launch_pack_lora_b_cute<T>(sycl::queue*, const float*, const T*, T*, T*, int, int);   \
  template void launch_reduce_partials_cute<T>(sycl::queue*, const float*, T*, int, int, int); \
  template void launch_lora_cute<T, 1, 8>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_lora_cute<T, 1, 16>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_lora_cute<T, 1, 32>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_lora_cute<T, 2, 8>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_lora_cute<T, 2, 16>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_lora_cute<T, 2, 32>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_lora_cute<T, 3, 8>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_lora_cute<T, 3, 16>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_lora_cute<T, 3, 32>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_lora_cute<T, 4, 8>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_lora_cute<T, 4, 16>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_lora_cute<T, 4, 32>(sycl::queue*, const T*, const T*, const T*, T*, int, int, int); \
  template void launch_fused_cute<T, true, 1>(sycl::queue*, const T*, const float*, const T*, const T*, uint8_t*, uint8_t*, T*, float*, int, int, int, int); \
  template void launch_fused_cute<T, true, 2>(sycl::queue*, const T*, const float*, const T*, const T*, uint8_t*, uint8_t*, T*, float*, int, int, int, int); \
  template void launch_fused_cute<T, true, 3>(sycl::queue*, const T*, const float*, const T*, const T*, uint8_t*, uint8_t*, T*, float*, int, int, int, int); \
  template void launch_fused_cute<T, true, 4>(sycl::queue*, const T*, const float*, const T*, const T*, uint8_t*, uint8_t*, T*, float*, int, int, int, int); \
  template void launch_fused_cute<T, false, 1>(sycl::queue*, const T*, const float*, const T*, const T*, uint8_t*, uint8_t*, T*, float*, int, int, int, int); \
  template void launch_fused_cute<T, false, 2>(sycl::queue*, const T*, const float*, const T*, const T*, uint8_t*, uint8_t*, T*, float*, int, int, int, int); \
  template void launch_fused_cute<T, false, 3>(sycl::queue*, const T*, const float*, const T*, const T*, uint8_t*, uint8_t*, T*, float*, int, int, int, int); \
  template void launch_fused_cute<T, false, 4>(sycl::queue*, const T*, const float*, const T*, const T*, uint8_t*, uint8_t*, T*, float*, int, int, int, int);

ARK_SVDQ_CUTE_INSTANTIATE(sycl::ext::oneapi::bfloat16)
ARK_SVDQ_CUTE_INSTANTIATE(sycl::half)

#undef ARK_SVDQ_CUTE_INSTANTIATE

}  // namespace svdquant
}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
