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

template <int Cells, typename BitDepth>
class PackingRegisterBlock<
        WidthMajorUint8SideMap,
        PackedSideBlock<DepthMajorSideFormatNCells4x2<Cells>, BitDepth> >
  : public PackingRegisterBlockBase<WidthMajorUint8SideMap, PackedSideBlock<DepthMajorSideFormatNCells4x2<Cells>, BitDepth> >
{
 public:
  typedef DepthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Pack(PackedSideBlock<KernelSideFormat, BitDepth>* dst, int start_width) {
    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* const src_ptr = this->complete_src_.data();
    const int stride = this->complete_src_.stride();
    // Load raw source WidthMajor data
    uint8x16_t src_lines[4 * kCells];
    for (int i = 0; i < 4 * kCells; i++) {
      src_lines[i] = vld1q_u8(src_ptr + i * stride);
    }
    if (BitDepth::kBits < 8) {
      for (int i = 0; i < 4 * kCells; i++) {
        src_lines[i] = vshrq_n_u8(src_lines[i], 8 - BitDepth::kBits);
      }
    }
    // Reorder the data within registers to make DepthMajor 4x2 cells
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
    // Store the resulting DepthMajor 4x2 cells in the destination packed block
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
    // Compute sums across the depth dimension
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
    // Update the rank_one_update vector
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

// The paths here are specifically arm 32bit assembly, not arm 64bit.
#ifdef GEMMLOWP_NEON32

// We provide a PackAlignedRunImpl specialization for the very important case
// of the RHS format of our default GEMM kernel. Indeed, profiles show a
// substantial amount of time spent packing the RHS, even with the above
// PackingRegisterBlock specialization.
typedef KernelSideFormat<CellFormat<4, 2>, 1> DepthMajorSideFormat1Cell4x2;

template <>
class PackAlignedRunImpl<
        WidthMajorUint8SideMap,
        PackedSideBlock<DepthMajorSideFormat1Cell4x2, BitDepth<8> > >
{
 public:
  typedef WidthMajorUint8SideMap SrcMapType;
  typedef DepthMajorSideFormat1Cell4x2 KernelSideFormat;

  static void PackAlignedRun(
    PackedSideBlock<KernelSideFormat, BitDepth<8> >* packed_side_block,
    const SrcMapType& src_map,
    int start_width, int /*width*/, int start_depth, int depth)
  {
    const std::uint8_t* src_line0_ptr = src_map.data(start_width, start_depth);
    const std::uint8_t* src_line1_ptr = src_line0_ptr + src_map.stride();
    const std::uint8_t* src_line2_ptr = src_line1_ptr + src_map.stride();
    const std::uint8_t* src_line3_ptr = src_line2_ptr + src_map.stride();
    std::uint8_t* dst_ptr = packed_side_block->current_data();
    std::uint8_t* dst_end_ptr = dst_ptr + 4 * depth;
    std::int32_t* rank_one_update_ptr = packed_side_block->rank_one_update() + start_width;
    std::int32_t rank_one_update_multiplier = packed_side_block->rank_one_update_multiplier();
    asm volatile(
      // Clear accumulator register for the rank-one-update sum.
      "vmov.s32 q4, #0\n"
      // Main loop across the depth dimension.
      "loop_PackAlignedRun_WidthMajor_to_DepthMajorSideFormat1Cell4x2_%=:\n"
      // Load source data (width-major).
      "vld1.u8 {d16,d17}, [%[src_line0_ptr]]!\n"
      "vld1.u8 {d18,d19}, [%[src_line1_ptr]]!\n"
      "vld1.u8 {d20,d21}, [%[src_line2_ptr]]!\n"
      "vld1.u8 {d22,d23}, [%[src_line3_ptr]]!\n"
      // Sum across the depth dimension
      "vpaddl.u8 q0, q8\n"
      "vpaddl.u8 q1, q9\n"
      "vpaddl.u8 q2, q10\n"
      "vpaddl.u8 q3, q11\n"
      "vpaddl.u16 q0, q0\n"
      "vpaddl.u16 q1, q1\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vtrn.32 d0, d2\n"
      "vtrn.32 d4, d6\n"
      "vmov d1, d4\n"
      "vmov d3, d6\n"
      "vadd.u32 q0, q0, q1\n"
      "vadd.u32 q4, q4, q0\n"
      // Store depth-major cells; the storage order flip is handled by vst4.
      "vst4.u8 {d16, d18, d20, d22}, [%[dst_ptr]]!\n"
      "vst4.u8 {d17, d19, d21, d23}, [%[dst_ptr]]!\n"
      // Loop.
      "cmp %[dst_ptr], %[dst_end_ptr]\n"
      "bne loop_PackAlignedRun_WidthMajor_to_DepthMajorSideFormat1Cell4x2_%=\n"
      // Update the rank-one-update data.
      "vld1.32 {d10, d11}, [%[rank_one_update_ptr]]\n"
      "vdup.32 d12, %[rank_one_update_multiplier]\n"
      "vmla.s32 q5, q4, d12[0]\n"
      "vst1.32 {d10, d11}, [%[rank_one_update_ptr]]\n"
        :  // outputs
        [src_line0_ptr] "+r"(src_line0_ptr),
        [src_line1_ptr] "+r"(src_line1_ptr),
        [src_line2_ptr] "+r"(src_line2_ptr),
        [src_line3_ptr] "+r"(src_line3_ptr),
        [dst_ptr] "+r"(dst_ptr)
        :  // inputs
        [dst_end_ptr] "r"(dst_end_ptr),
        [rank_one_update_multiplier] "r"(rank_one_update_multiplier),
        [rank_one_update_ptr] "r"(rank_one_update_ptr)
        :  // clobbers
        "cc", "memory",
        // note: someone on internet says that quad registers are
        // unsupported in the clobber list!
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31");
    packed_side_block->seek_forward_n_cells(
      KernelSideFormat::kCells * depth / KernelSideFormat::kDepth);
  }
};
#endif  // GEMMLOWP_NEON32

template <int Cells>
using WidthMajorSideFormatNCells4x2 = KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, Cells>;

template <int Cells, typename BitDepth>
class PackingRegisterBlock<
        WidthMajorUint8SideMap,
        PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells>, BitDepth> >
  : public PackingRegisterBlockBase<WidthMajorUint8SideMap, PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells>, BitDepth> >
{
 public:
  typedef WidthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Pack(PackedSideBlock<KernelSideFormat, BitDepth>* dst, int start_width) {
    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* src_ptr = this->complete_src_.data();
    const int stride = this->complete_src_.stride();
    // Load raw source WidthMajor data
    uint16x8_t src_lines[kCells * 4];
    for (int i = 0; i < 4 * kCells; i++) {
      src_lines[i] = vreinterpretq_u16_u8(vld1q_u8(src_ptr));
      src_ptr += stride;
    }
    if (BitDepth::kBits < 8) {
      for (int i = 0; i < 4 * kCells; i++) {
        src_lines[i] = vreinterpretq_u16_u8(vshrq_n_u8(vreinterpretq_u8_u16(src_lines[i]), 8 - BitDepth::kBits));
      }
    }
    // Reorder the data within registers to make WidthMajor 4x2 cells
    uint16x8x2_t src_lines_intertwined_2x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_2x[2 * i] = vzipq_u16(src_lines[4 * i], src_lines[4 * i + 2]);
      src_lines_intertwined_2x[2 * i + 1] = vzipq_u16(src_lines[4 * i + 1], src_lines[4 * i + 3]);
    }
    uint16x8x2_t src_lines_intertwined_4x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_4x[2 * i] = vzipq_u16(src_lines_intertwined_2x[2 * i].val[0], src_lines_intertwined_2x[2 * i + 1].val[0]);
      src_lines_intertwined_4x[2 * i + 1] = vzipq_u16(src_lines_intertwined_2x[2 * i].val[1], src_lines_intertwined_2x[2 * i + 1].val[1]);
    }
    // Store the resulting WidthMajor 4x2 cells in the destination packed block
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        for (int cell = 0; cell < kCells; cell++) {
          vst1_u8(dst_ptr, vreinterpret_u8_u16(vget_low_u16(src_lines_intertwined_4x[2 * cell + outer].val[inner])));
          dst_ptr += 8;
        }
        for (int cell = 0; cell < kCells; cell++) {
          vst1_u8(dst_ptr, vreinterpret_u8_u16(vget_high_u16(src_lines_intertwined_4x[2 * cell + outer].val[inner])));
          dst_ptr += 8;
        }
      }
    }
    // Compute sums across the depth dimension
    uint16x8_t sums_of_2[kCells][4];
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        int i = 2 * outer + inner;
        for (int cell = 0; cell < kCells; cell++) {
          sums_of_2[cell][i] = vpaddlq_u8(vreinterpretq_u8_u16(src_lines_intertwined_4x[2 * cell + outer].val[inner]));
        }
      }
    }
    uint16x8_t sums_of_4[kCells][2];
    for (int i = 0; i < 2; i++) {
      for (int cell = 0; cell < kCells; cell++) {
        sums_of_4[cell][i] = vaddq_u16(sums_of_2[cell][2 * i], sums_of_2[cell][2 * i + 1]);
      }
    }
    uint16x8_t sums_of_8[kCells];
    for (int cell = 0; cell < kCells; cell++) {
      sums_of_8[cell] = vaddq_u16(sums_of_4[cell][0], sums_of_4[cell][1]);
    }

    uint16x4_t sums_of_16[kCells];
    for (int cell = 0; cell < kCells; cell++) {
      sums_of_16[cell] = vadd_u16(vget_low_u16(sums_of_8[cell]), vget_high_u16(sums_of_8[cell]));
    }
    // Update the rank_one_update vector
    for (int cell = 0; cell < kCells; cell++) {
      int32x4_t s = vreinterpretq_s32_u32(vmovl_u16(sums_of_16[cell]));
      int32x4_t u = vmulq_n_s32(s, dst->rank_one_update_multiplier());
      std::int32_t* rank_one_update_ptr = dst->rank_one_update() + start_width + 4 * cell;
      vst1q_s32(rank_one_update_ptr, vaddq_s32(u, vld1q_s32(rank_one_update_ptr)));
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_PACK_NEON_H_
