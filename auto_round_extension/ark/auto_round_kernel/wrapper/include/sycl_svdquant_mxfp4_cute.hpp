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
// ---------------------------------------------------------------------------
// SVDQuant Kernel A: sycl-tla (CuTe) low-rank down projection.
//
// This is the only matrix implementation of the projection. An earlier
// joint_matrix version existed and was deleted once this one beat it on every
// measured shape; the parts of its rationale that still apply are reproduced
// below rather than lost with it.
//
// WHY A MATRIX UNIT AT ALL
//
// The scalar formulation accumulates the projection with FP32 vector FMA, which
// measured ~2.5 TFLOP/s -- about 20% of this device's FP32 vector roof. The
// limit was operand fetch, not arithmetic: its inner loop read two bytes of
// `lora_down` per FMA (bytes/FMA = 2.00, zero register reuse), so `lora_down`
// was re-read once per row -- 906 MB at R=32 against only 28 MB for `x`. A
// systolic array supplies that operand reuse for free, and moves the
// accumulators into dedicated registers, relieving the pressure that had
// blocked row blocking.
//
// THE KEY ALGEBRAIC REWRITE (fold `smooth` into B)
//
// The obvious mapping (A = the smoothed activations) is the wrong one. It would
// force the FP32 product `x * smooth` to be materialised, rounded to 16-bit and
// staged through SLM before the matrix unit could reach it, and SLM occupancy
// is exactly what made an earlier version slow. Since `smooth` is per-column
// (per-K), fold it into the *other* operand instead:
//
//     lora_act = (x * smooth) @ lora_down^T  ==  x @ (smooth * lora_down)^T
//
// Three consequences: A becomes the raw `x` exactly as it sits in memory, so it
// is loaded straight from global memory with no staging, no SLM and **no
// rounding at all** on A; the folded operand `smooth * lora_down` is [K, R<=64],
// tiny and cache-resident; and all the precision loss is concentrated in that
// one small matrix, where it is cheap to correct.
//
// PRECISION: SPLIT 16-BIT, TWO PASSES
//
// DPAS has no FP32 x FP32 mode, so the folded operand must land in 16 bits.
// Rounding it once costs relative L2 1.65e-3. Splitting it into a high plane and
// a residual plane and running two passes into the same FP32 accumulator
// recovers 2.44e-6. Because this kernel is memory bound and the operand is
// already cache-resident, the second pass is close to free. Under the 16-bit
// output contract the split lands exactly on the storage floor while a single
// pass is 1.4x-8x worse -- the 16-bit output did **not** make the split
// redundant.
//
// WHY THE CuTe ARCH LAYER RATHER THAN THE DEVICE LAYER
//
// This is built on raw XE_LOAD_2D / XE_DPAS_TT plus the
// __builtin_IB_subgroup_createBlock2DAddressPayload intrinsics, not on
// GemmUniversalAdapter. The device-level BMG GEMM uses a <256, 256, 32> tile,
// which at N = R <= 64 leaves MMA utilisation at 12.5%, and it is a separate
// kernel launch that would have to re-read `x` from global memory -- precisely
// what design doc section 3.1 sets out to avoid.
//
// THREE HARDWARE CONTRACTS, ALL ESTABLISHED BY PROBE RATHER THAN BY READING
//
// Each of these produces plausible-looking but silently wrong numbers if
// assumed incorrectly, so each was measured on a B60 against a host double
// reference before this kernel was written:
//
//  1. Register order for a Count=2 load. XE_LOAD_2D<16, Rows, 32, 16> fetches
//     Rows x 32 elements as two 16-column blocks, and the destination registers
//     are laid out *block-major*: A tile (kt, mt) is at a[kt * MTiles + mt].
//     Row-major would have been an equally plausible guess and would have
//     silently transposed the K contributions.
//  2. XE_LOAD_2D_VNNI performs the VNNI interleave in hardware, reading a plain
//     row-major [K, N] surface. B is therefore emitted as an ordinary row-major
//     [K, R] matrix and there is no host-side VNNI packer at all.
//  3. Out-of-bounds rows read back as zero. The 2D block load bounds-checks
//     against the surface height recorded in the payload, so a tile that hangs
//     off the end of `x` contributes nothing rather than reading garbage. This
//     is why there is no scalar tail kernel and no minimum M: small-M problems
//     run the same code, with only the stores guarded.
// ---------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#include "sycl_svdquant_mxfp4.hpp"

namespace ark {
namespace svdquant {

// DPAS tile shape. These are the Xe 16-bit DPAS dimensions, not free
// parameters: XE_DPAS_TT is declared over exactly this shape.
constexpr int kTileM = 8;
constexpr int kTileN = 16;
constexpr int kTileK = 16;

// Sub-group size required by DPAS on Xe.
constexpr int kDpasSubGroup = 16;

// Accumulator tiles along N one sub-group owns. R <= 64 and R is a multiple of
// kTileN on this path, so at most 4.
constexpr int kMaxNTiles = 4;

inline int lora_n_tiles(int r) { return r / kTileN; }

// Rows one sub-group owns.
//
// A sweep of 8 / 16 / 32 over every benchmark shape found 8 uniformly best
// (geomean speedup 23.33x / 22.07x / 20.35x; flux mlp 0.231 / 0.238 / 0.261 ms;
// flux ffn 1.130 / 1.236 / 1.266 ms), so this is simply 8.
//
// That result refutes the hypothesis this path was built on. The expectation
// was that fetching 32 rows per instruction would lift the A-load bandwidth
// joint_matrix was leaving on the table; instead every widening of the row
// block made things monotonically worse. Two effects swamp the fetch width:
//
//   * Occupancy. Rows per sub-group divide into the sub-group count, so wide
//     row blocking starves the machine -- at M = 1024 and 32 rows the entire
//     grid is 2 work-groups on a 160-EU device.
//   * Register pressure. The accumulator is Rows/kTileM * NTiles vectors of
//     kTileM floats per lane, which spilled outright at Rows = 32 with R = 64
//     (0.763 ms, against 0.266 ms once the rows were halved).
//
// The sycl-tla path still beats joint_matrix on every measured shape, but not
// for the advertised reason: at Rows = 8 the row block is exactly joint_matrix's
// height, and what actually remains is the *K* width of the load (two K-tiles,
// so 64-byte rows rather than 32), the hardware VNNI transform, and the absence
// of a separate tail kernel.
constexpr int kCuteMaxRows = 32;

// Rows per sub-group in the *fused* kernel. Not a free parameter and not the
// result of the sweep above: the fused kernel maps one lane to one row for the
// quantization phase, so this must equal the DPAS sub-group size. It is also
// exactly 2 M-tiles, which is what the projection phase consumes.
constexpr int kFusedCuteRows = kDpasSubGroup;

// `ARK_SVDQUANT_CUTE_ROWS` overrides the row blocking to 8, 16 or 32. The wider
// instantiations are kept solely so this sweep can be repeated on hardware with
// a different register file or EU count without a rebuild; values that would
// reintroduce the accumulator spill are clamped rather than honoured.
int cute_rows_for(int m, int n_tiles);

inline constexpr int cute_rows_for_shape(int, int) { return 8; }

// K advanced per A load. Two K-tiles wide so each fetched row spans a full
// 64-byte cache line; Bits * Width = 16 * 32 = 512 is exactly the atom's limit.
constexpr int kCuteKTiles = 2;
constexpr int kCuteKStep = kTileK * kCuteKTiles;

// Columns allocated for the B operand.
//
// B is [K, R] row-major, so its surface pitch is r_padded * 2 bytes. Intel's 2D
// block load documents a 64-byte minimum surface width, which R = 16 alone
// would violate at 32 bytes. A B60 probe accepted the narrow surface anyway,
// but that is the sort of leniency that differs between steppings, so the
// buffer is padded to two N tiles. The kernel only ever loops over the real
// tiles, so the padding is never read and costs a few KB.
inline constexpr int cute_b_cols(int r) {
  const int n_tiles = (r + kTileN - 1) / kTileN;
  return (n_tiles < 2 ? 2 : n_tiles) * kTileN;
}

// Elements in one B plane (hi or lo). The workspace holds two back to back.
inline constexpr std::size_t cute_b_plane_elements(int k, int r) {
  return static_cast<std::size_t>(k) * cute_b_cols(r);
}

// --- K splitting -----------------------------------------------------------
//
// The fused kernel consumes K in a serial loop, so on its own it exposes only
// ceil(M / kFusedCuteRows) sub-groups of parallelism. That is far too few: see
// design doc 3.3.5 for the measurement, where holding the total element count
// fixed and trading rows against columns moved the runtime by 4.1x purely on
// the sub-group count. Splitting K restores it.
//
// The split is nearly free because the two halves of the kernel have completely
// different dependence structure across K:
//
//   * Quantization needs no cooperation whatsoever. Micro-scaling groups are
//     independent by construction and each slice writes disjoint bytes of
//     `qact` and `ascales`, so slicing costs exactly nothing.
//   * The projection accumulates along K, so it needs a reduction. Each slice
//     writes an FP32 [M, R] partial plane and a second, tiny kernel sums them.
//
// A partial-buffer reduction is used rather than FP32 atomics so that
// `lora_act` is reproducible run to run.
//
// The target below was swept on B60 at M = 4608, K = 3072 by pinning the slice
// count directly (target = row_tiles * s), which isolates it from every other
// variable:
//
//   slices    1     2     3     4     5     6     7     8    10
//   R=32   0.460 0.213 0.173 0.178 0.212 0.217 0.196 0.204 0.207 ms
//   R=64   0.503 0.323 0.256 0.253 0.315 0.338 0.323 0.313 0.359 ms
//
// The curve is a clean U with its floor at 3-4 slices, i.e. ~900-1150
// sub-groups, and 1024 sits in the middle of that floor for every shape
// measured. Two forces set the two arms:
//
//   * Below it, latency hiding. One slice is 2.7x off the floor -- that is the
//     parallelism starvation of 3.3.5, unmitigated.
//   * Above it, the partial buffer. Its traffic is slices * M * R * 4 bytes
//     written and read again; at 8 slices, M = 4608, R = 32 that is 9.4 MB of
//     round trip against the 28 MB the `x` pass moves, so it stops being
//     rounding error and starts costing more than the occupancy it buys.
//
// Divisibility was checked and is *not* a factor: 6 and 8 divide the 96 groups
// exactly and are both slower than 5 and 7, which do not.
constexpr int kCuteTargetSubGroups = 1024;

// Sub-groups per work-group and work-group size for the fused kernel.
constexpr int kCuteFusedSubGroupsPerWG = 4;
constexpr int kCuteFusedWorkGroupSize = kCuteFusedSubGroupsPerWG * kDpasSubGroup;

// Number of K slices. Chosen to hit a fixed sub-group target rather than being
// a constant, because the right value depends entirely on how much parallelism
// M already supplies: at M = 4608 the row blocks alone give 288 sub-groups and
// only ~7 slices are wanted, while at M = 1 every slice is pure gain.
//
// Capped at the group count so that no slice is empty, and clamped so the
// partial buffer cannot outgrow the tensors it serves.
// `ARK_SVDQUANT_CUTE_TARGET_SUBGROUPS` overrides the target above. It exists so
// the sweep can be repeated on hardware with a different EU count or register
// file without a rebuild; it is read on the host, once per launch.
int cute_target_subgroups();

inline int cute_k_slices(int m, int k, int r) {
  const int row_tiles = (m + kFusedCuteRows - 1) / kFusedCuteRows;
  const int num_groups = k / kGroupSize;
  int slices = cute_target_subgroups() / (row_tiles > 0 ? row_tiles : 1);
  if (slices < 1) slices = 1;
  if (slices > num_groups) slices = num_groups;
  // Re-derive from the per-slice group count so that the last slice is never
  // empty: ceil(num_groups / ceil(num_groups / slices)) <= slices.
  const int groups_per_slice = (num_groups + slices - 1) / slices;
  return (num_groups + groups_per_slice - 1) / groups_per_slice;
}

// FP32 elements in the projection partial buffer. Zero when a single slice
// covers K, in which case the kernel writes `lora_act` directly and the
// reduction pass is skipped entirely.
inline std::size_t cute_partial_elements(int m, int r, int slices) {
  if (slices <= 1) return 0;
  return static_cast<std::size_t>(slices) * static_cast<std::size_t>(m) * static_cast<std::size_t>(r);
}

// True when the CuTe path can serve this problem.
//
// Note the absence of an M bound: contract 3 above means partial row tiles are
// handled by the hardware, so this is valid down to
// M = 1. K must be a multiple of the A load width because the K loop is not
// bounds-checked -- only the row dimension is.
inline bool cute_lora_supported(int m, int k, int r) {
  return m > 0 && r > 0 && r % kTileN == 0 && r <= kMaxNTiles * kTileN && k % kCuteKStep == 0;
}

#ifdef ARK_XPU

// Prologue: fold `smooth` into `lora_down` and emit the split B operand as two
// plain row-major [K, cute_b_cols(r)] planes.
//
// Unlike the joint_matrix packer this does no VNNI interleaving -- contract 2
// above puts that in the load instruction. `smooth` may be null, meaning
// "multiply by 1".
template <typename T>
void launch_pack_lora_b_cute(sycl::queue* queue, const float* smooth, const T* lora_down, T* hi, T* lo, int k, int r);

// lora_act[M, R] = x[M, K] @ (B_hi + B_lo), all rows including partial tiles.
// Standalone form: assumes the quantization already ran in a separate launch.
template <typename T, int NTiles, int Rows>
void launch_lora_cute(sycl::queue* queue, const T* x, const T* hi, const T* lo, T* lora_act, int m, int k, int r);

// Fused form: quantization and projection in one launch, one pass over `x`.
// `hi` / `lo` are the split planes produced by launch_pack_lora_b_cute.
//
// `slices` comes from cute_k_slices. When it is 1 the kernel writes `lora_act`
// directly and `partial` is ignored; otherwise each slice writes its own FP32
// [M, R] plane into `partial` and launch_reduce_partials_cute must follow.
template <typename T, bool UseDoubleLog, int NTiles>
void launch_fused_cute(sycl::queue* queue, const T* x, const float* smooth, const T* hi, const T* lo, uint8_t* qact,
                       uint8_t* ascales, T* lora_act, float* partial, int m, int k, int r, int slices);

// Epilogue: lora_act[M, R] = sum over the `slices` FP32 partial planes.
template <typename T>
void launch_reduce_partials_cute(sycl::queue* queue, const float* partial, T* lora_act, int m, int r, int slices);

#endif  // ARK_XPU

}  // namespace svdquant
}  // namespace ark
