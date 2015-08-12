// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// pack_neon.h: optimized NEON specializations of the templates in pack.h.

#ifndef GEMMLOWP_INTERNAL_PACK_NEON_H_
#define GEMMLOWP_INTERNAL_PACK_NEON_H_

#include "pack.h"

#include <arm_neon.h>

namespace gemmlowp {

typedef SideMap<const std::uint8_t, SideMapOrder::WidthMajor> WidthMajorUint8SideMap;

template <int Cells>
using DepthMajorSideFormatNCells4x2 = KernelSideFormat<CellFormat<4, 2>, Cells>;

template <int Cells>
class PackingRegisterBlock<WidthMajorUint8SideMap, DepthMajorSideFormatNCells4x2<Cells> >
  : public PackingRegisterBlockBase<WidthMajorUint8SideMap, DepthMajorSideFormatNCells4x2<Cells> >
{
 public:
  typedef DepthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Store(PackedSideBlock<KernelSideFormat>* dst, int start_width) {
    ScopedProfilingLabel label("PackingRegisterBlock::Store (NEON intrinsics)");

    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* const src_ptr = this->loaded_src.data();
    const int stride = this->loaded_src.stride();
    uint8x16_t src_lines[4 * kCells];
    for (int i = 0; i < 4 * kCells; i++) {
      src_lines[i] = vld1q_u8(src_ptr + i * stride);
    }
    uint8x16x2_t src_lines_intertwined_2x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_2x[2 * i] = vzipq_u8(src_lines[4 * i], src_lines[4 * i + 2]);
      src_lines_intertwined_2x[2 * i + 1] = vzipq_u8(src_lines[4 * i + 1], src_lines[4 * i + 3]);
    }
    uint8x16x2_t src_lines_intertwined_4x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_4x[2 * i] = vzipq_u8(src_lines_intertwined_2x[2 * i].val[0], src_lines_intertwined_2x[2 * i + 1].val[0]);
      src_lines_intertwined_4x[2 * i + 1] = vzipq_u8(src_lines_intertwined_2x[2 * i].val[1], src_lines_intertwined_2x[2 * i + 1].val[1]);
    }
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        for (int cell = 0; cell < kCells; cell++) {
          vst1_u8(dst_ptr, vget_low_u8(src_lines_intertwined_4x[2 * cell + outer].val[inner]));
          dst_ptr += 8;
        }
        for (int cell = 0; cell < kCells; cell++) {
          vst1_u8(dst_ptr, vget_high_u8(src_lines_intertwined_4x[2 * cell + outer].val[inner]));
          dst_ptr += 8;
        }
      }
    }
    uint16x8_t sums_of_2_cells[kCells][4];
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        int i = 2 * outer + inner;
        for (int cell = 0; cell < kCells; cell++) {
          sums_of_2_cells[cell][i] = vaddl_u8(vget_low_u8(src_lines_intertwined_4x[2 * cell + outer].val[inner]),
                                              vget_high_u8(src_lines_intertwined_4x[2 * cell + outer].val[inner]));
        }
      }
    }
    int32x4_t sums_of_4_cells[kCells][4];
    for (int i = 0; i < 4; i++) {
      for (int cell = 0; cell < kCells; cell++) {
        sums_of_4_cells[cell][i] = vreinterpretq_s32_u32(vaddl_u16(vget_low_u16(sums_of_2_cells[cell][i]), vget_high_u16(sums_of_2_cells[cell][i])));
      }
    }

    for (int cell = 0; cell < kCells; cell++) {
      int32x4_t s01 = vaddq_s32(sums_of_4_cells[cell][0], sums_of_4_cells[cell][1]);
      int32x4_t s23 = vaddq_s32(sums_of_4_cells[cell][2], sums_of_4_cells[cell][3]);
      int32x4_t s = vaddq_s32(s01, s23);
      int32x4_t u = vmulq_n_s32(s, dst->rank_one_update_multiplier());
      std::int32_t* rank_one_update_ptr = dst->rank_one_update() + start_width + 4 * cell;
      vst1q_s32(rank_one_update_ptr, vaddq_s32(u, vld1q_s32(rank_one_update_ptr)));
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

#if 0 // The asm paths below are checked in only so that they're in the
      // history for future reference. The effective speedup on the overall
      // GEMM is too low to justify having all this asm code for now.

// The paths here are specifically arm 64bit assembly, not arm 32bit.
#ifdef GEMMLOWP_NEON64

typedef KernelSideFormat<CellFormat<4, 2>, 3> DepthMajorSideFormat3Cells4x2;

template <>
class PackingRegisterBlock<WidthMajorUint8SideMap, DepthMajorSideFormat3Cells4x2>
  : public PackingRegisterBlockBase<WidthMajorUint8SideMap, DepthMajorSideFormat3Cells4x2>
{
 public:
  typedef DepthMajorSideFormat3Cells4x2 KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Store(PackedSideBlock<KernelSideFormat>* dst, int start_width) {
    ScopedProfilingLabel label("PackingRegisterBlock::Store (NEON)");

    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* src_ptr = this->loaded_src.data();
    const int stride = this->loaded_src.stride();
    std::int32_t* rank_one_update_ptr = dst->rank_one_update() + start_width;
    const std::int32_t rank_one_update_multiplier = dst->rank_one_update_multiplier();

    asm volatile(
      "ld1 {v0.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v1.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v2.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v3.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v4.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v5.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v6.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v7.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v8.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v9.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v10.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v11.16b}, [%[src_ptr]], %[stride]\n"
      
      "zip1 v12.16b, v0.16b, v2.16b\n"
      "zip2 v13.16b, v0.16b, v2.16b\n"
      "zip1 v14.16b, v1.16b, v3.16b\n"
      "zip2 v15.16b, v1.16b, v3.16b\n"
      "zip1 v16.16b, v4.16b, v6.16b\n"
      "zip2 v17.16b, v4.16b, v6.16b\n"
      "zip1 v18.16b, v5.16b, v7.16b\n"
      "zip2 v19.16b, v5.16b, v7.16b\n"
      "zip1 v20.16b, v8.16b, v10.16b\n"
      "zip2 v21.16b, v8.16b, v10.16b\n"
      "zip1 v22.16b, v9.16b, v11.16b\n"
      "zip2 v23.16b, v9.16b, v11.16b\n"
      
      "zip1 v0.16b, v12.16b, v14.16b\n"
      "zip2 v3.16b, v12.16b, v14.16b\n"
      "zip1 v6.16b, v13.16b, v15.16b\n"
      "zip2 v9.16b, v13.16b, v15.16b\n"
      "zip1 v1.16b, v16.16b, v18.16b\n"
      "zip2 v4.16b, v16.16b, v18.16b\n"
      "zip1 v7.16b, v17.16b, v19.16b\n"
      "zip2 v10.16b, v17.16b, v19.16b\n"
      "zip1 v2.16b, v20.16b, v22.16b\n"
      "zip2 v5.16b, v20.16b, v22.16b\n"
      "zip1 v8.16b, v21.16b, v23.16b\n"
      "zip2 v11.16b, v21.16b, v23.16b\n"
      
      "st3 {v0.2d, v1.2d, v2.2d}, [%[dst_ptr]], #48\n"
      "st3 {v3.2d, v4.2d, v5.2d}, [%[dst_ptr]], #48\n"
      "st3 {v6.2d, v7.2d, v8.2d}, [%[dst_ptr]], #48\n"
      "st3 {v9.2d, v10.2d, v11.2d}, [%[dst_ptr]], #48\n"

      "uaddl v12.8h, v0.8b, v3.8b\n"
      "uaddl v13.8h, v1.8b, v4.8b\n"
      "uaddl v14.8h, v2.8b, v5.8b\n"
      "uaddl2 v15.8h, v0.16b, v3.16b\n"
      "uaddl2 v16.8h, v1.16b, v4.16b\n"
      "uaddl2 v17.8h, v2.16b, v5.16b\n"
      "uaddl v18.8h, v6.8b, v9.8b\n"
      "uaddl v19.8h, v7.8b, v10.8b\n"
      "uaddl v20.8h, v8.8b, v11.8b\n"
      "uaddl2 v21.8h, v6.16b, v9.16b\n"
      "uaddl2 v22.8h, v7.16b, v10.16b\n"
      "uaddl2 v23.8h, v8.16b, v11.16b\n"
      
      "uaddl v0.4s, v12.4h, v15.4h\n"
      "uaddl v1.4s, v13.4h, v16.4h\n"
      "uaddl v2.4s, v14.4h, v17.4h\n"
      "uaddl2 v3.4s, v12.8h, v15.8h\n"
      "uaddl2 v4.4s, v13.8h, v16.8h\n"
      "uaddl2 v5.4s, v14.8h, v17.8h\n"
      "uaddl v6.4s, v18.4h, v21.4h\n"
      "uaddl v7.4s, v19.4h, v22.4h\n"
      "uaddl v8.4s, v20.4h, v23.4h\n"
      "uaddl2 v9.4s, v18.8h, v21.8h\n"
      "uaddl2 v10.4s, v19.8h, v22.8h\n"
      "uaddl2 v11.4s, v20.8h, v23.8h\n"
      
      "add v0.4s, v0.4s, v6.4s\n"
      "add v1.4s, v1.4s, v7.4s\n"
      "add v2.4s, v2.4s, v8.4s\n"
      "add v3.4s, v3.4s, v9.4s\n"
      "add v4.4s, v4.4s, v10.4s\n"
      "add v5.4s, v5.4s, v11.4s\n"

      "mov x0, %[rank_one_update_ptr]\n"
      "ld1 {v12.4s}, [x0], #16\n"
      "ld1 {v13.4s}, [x0], #16\n"
      "ld1 {v14.4s}, [x0], #16\n"
      "dup v15.4s, %w[rank_one_update_multiplier]\n"
      "add v16.4s, v0.4s, v3.4s\n"
      "add v17.4s, v1.4s, v4.4s\n"
      "add v18.4s, v2.4s, v5.4s\n"
      "mul v16.4s, v16.4s, v15.4s\n"
      "mul v17.4s, v17.4s, v15.4s\n"
      "mul v18.4s, v18.4s, v15.4s\n"
      "add v12.4s, v12.4s, v16.4s\n"
      "add v13.4s, v13.4s, v17.4s\n"
      "add v14.4s, v14.4s, v18.4s\n"
      "mov x0, %[rank_one_update_ptr]\n"
      "st1 {v12.4s}, [x0], #16\n"
      "st1 {v13.4s}, [x0], #16\n"
      "st1 {v14.4s}, [x0], #16\n"
      
      :  // outputs
      [dst_ptr] "+r"(dst_ptr), [src_ptr] "+r"(src_ptr)
      :  //inputs
      [stride] "r"(stride),
      [rank_one_update_ptr] "r"(rank_one_update_ptr),
      [rank_one_update_multiplier] "r"(rank_one_update_multiplier)
      :  //clobbers
      "cc", "memory", "x0",
      "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
      "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
      "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
      "v31");
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

typedef KernelSideFormat<CellFormat<4, 2>, 2> DepthMajorSideFormat2Cells4x2;

template <>
class PackingRegisterBlock<WidthMajorUint8SideMap, DepthMajorSideFormat2Cells4x2>
  : public PackingRegisterBlockBase<WidthMajorUint8SideMap, DepthMajorSideFormat2Cells4x2>
{
 public:
  typedef DepthMajorSideFormat2Cells4x2 KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Store(PackedSideBlock<KernelSideFormat>* dst, int start_width) {
    ScopedProfilingLabel label("PackingRegisterBlock::Store (NEON)");

    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* src_ptr = this->loaded_src.data();
    const int stride = this->loaded_src.stride();
    std::int32_t* rank_one_update_ptr = dst->rank_one_update() + start_width;
    const std::int32_t rank_one_update_multiplier = dst->rank_one_update_multiplier();

    asm volatile(
      "ld1 {v0.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v1.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v2.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v3.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v4.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v5.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v6.16b}, [%[src_ptr]], %[stride]\n"
      "ld1 {v7.16b}, [%[src_ptr]], %[stride]\n"
      
      "zip1 v12.16b, v0.16b, v2.16b\n"
      "zip2 v13.16b, v0.16b, v2.16b\n"
      "zip1 v14.16b, v1.16b, v3.16b\n"
      "zip2 v15.16b, v1.16b, v3.16b\n"
      "zip1 v16.16b, v4.16b, v6.16b\n"
      "zip2 v17.16b, v4.16b, v6.16b\n"
      "zip1 v18.16b, v5.16b, v7.16b\n"
      "zip2 v19.16b, v5.16b, v7.16b\n"
      
      "zip1 v0.16b, v12.16b, v14.16b\n"
      "zip2 v2.16b, v12.16b, v14.16b\n"
      "zip1 v4.16b, v13.16b, v15.16b\n"
      "zip2 v6.16b, v13.16b, v15.16b\n"
      "zip1 v1.16b, v16.16b, v18.16b\n"
      "zip2 v3.16b, v16.16b, v18.16b\n"
      "zip1 v5.16b, v17.16b, v19.16b\n"
      "zip2 v7.16b, v17.16b, v19.16b\n"
      
      "st2 {v0.2d, v1.2d}, [%[dst_ptr]], #32\n"
      "st2 {v2.2d, v3.2d}, [%[dst_ptr]], #32\n"
      "st2 {v4.2d, v5.2d}, [%[dst_ptr]], #32\n"
      "st2 {v6.2d, v7.2d}, [%[dst_ptr]], #32\n"

      "uaddl v12.8h, v0.8b, v2.8b\n"
      "uaddl v13.8h, v1.8b, v3.8b\n"
      "uaddl2 v14.8h, v0.16b, v2.16b\n"
      "uaddl2 v15.8h, v1.16b, v3.16b\n"
      "uaddl v16.8h, v4.8b, v6.8b\n"
      "uaddl v17.8h, v5.8b, v7.8b\n"
      "uaddl2 v18.8h, v4.16b, v6.16b\n"
      "uaddl2 v19.8h, v5.16b, v7.16b\n"
      
      "uaddl v0.4s, v12.4h, v14.4h\n"
      "uaddl v1.4s, v13.4h, v15.4h\n"
      "uaddl2 v2.4s, v12.8h, v14.8h\n"
      "uaddl2 v3.4s, v13.8h, v15.8h\n"
      "uaddl v4.4s, v16.4h, v18.4h\n"
      "uaddl v5.4s, v17.4h, v19.4h\n"
      "uaddl2 v6.4s, v16.8h, v18.8h\n"
      "uaddl2 v7.4s, v17.8h, v19.8h\n"
      
      "add v0.4s, v0.4s, v4.4s\n"
      "add v1.4s, v1.4s, v5.4s\n"
      "add v2.4s, v2.4s, v6.4s\n"
      "add v3.4s, v3.4s, v7.4s\n"

      "mov x0, %[rank_one_update_ptr]\n"
      "ld1 {v12.4s}, [x0], #16\n"
      "ld1 {v13.4s}, [x0], #16\n"
      "dup v15.4s, %w[rank_one_update_multiplier]\n"
      "add v16.4s, v0.4s, v2.4s\n"
      "add v17.4s, v1.4s, v3.4s\n"
      "mul v16.4s, v16.4s, v15.4s\n"
      "mul v17.4s, v17.4s, v15.4s\n"
      "add v12.4s, v12.4s, v16.4s\n"
      "add v13.4s, v13.4s, v17.4s\n"
      "mov x0, %[rank_one_update_ptr]\n"
      "st1 {v12.4s}, [x0], #16\n"
      "st1 {v13.4s}, [x0], #16\n"
      
      :  // outputs
      [dst_ptr] "+r"(dst_ptr), [src_ptr] "+r"(src_ptr)
      :  //inputs
      [stride] "r"(stride),
      [rank_one_update_ptr] "r"(rank_one_update_ptr),
      [rank_one_update_multiplier] "r"(rank_one_update_multiplier)
      :  //clobbers
      "cc", "memory", "x0",
      "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
      "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
      "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
      "v31");
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};
#endif  // GEMMLOWP_NEON64

#endif

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_PACK_NEON_H_
