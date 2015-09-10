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

// Variant of PseudoRandomNonzeroBytesGenerator that produces
// random NEON 128-bit vectors.
class NEONPseudoRandomNonzeroBytesGenerator {
 public:
  NEONPseudoRandomNonzeroBytesGenerator() {
    uint8_t s = 1;
    std::uint8_t a[16];
    for (int i = 0; i < 16; i++) {
      a[i] = s;
      // Xorshift8(7,7,1). Very important to choose a different
      // xorshift than we do in get(), otherwise lanes would contain
      // the same values!
      s ^= s << 7;
      s ^= s >> 7;
      s ^= s << 1;
    }
    x_ = vld1q_u8(a);
  }

  uint8x16_t get() {
    // Xorshift8(7,5,3)
    x_ = veorq_u8(x_, vshlq_n_u8(x_, 7));
    x_ = veorq_u8(x_, vshrq_n_u8(x_, 5));
    x_ = veorq_u8(x_, vshlq_n_u8(x_, 3));
    return x_;
  }

 private:
  // State
  uint8x16_t x_;
};

// Requantizes source uint8 values in [0..255] range
// to the range specified by BitDepth, [0..((2^bits)-1)].
// Bias must be avoided. Currently this is achieved
// by probabilistic rounding.
template <typename BitDepth, RoundingMode Rounding>
uint8x16_t Requantize(uint8x16_t raw_src_data,
                      NEONPseudoRandomNonzeroBytesGenerator* prng) {
  static const int kBits = BitDepth::kBits;
  static const std::uint8_t kMaxVal = (1 << kBits) - 1;

  if (kBits == 8) {
    return raw_src_data;
  }

  // We will need to temporarily work in 16 bit precision.
  uint16x8_t x[2];

  // Multiply source values by 2^kBits.
  x[0] = vshll_n_u8(vget_low_u8(raw_src_data), kBits);
  x[1] = vshll_n_u8(vget_high_u8(raw_src_data), kBits);

  // Subtract source values, so we effectively have them
  // multiplied by (2^kBits) - 1.
  x[0] = vsubw_u8(x[0], vget_low_u8(raw_src_data));
  x[1] = vsubw_u8(x[1], vget_high_u8(raw_src_data));

  // Compute the rounding offset.
  uint8x16_t rounding_offset;
  switch (Rounding) {
    case RoundingMode::Nearest:
      // 128 is the midpoint in [1..255], and [1..255] is the interval
      // that we use for offsets here, see the comment below on
      // the Probabilistic case.
      rounding_offset = vdupq_n_u8(128);
      break;
    case RoundingMode::Probabilistic:
      // Take nonzero bytes in [1..255].
      // In principle we want a value in [0..254], but:
      //   1) Below we will be multiplying by 257/256 instead of 256/255,
      //      which is slightly too low, and this helps compensate for that.
      //      (One checks this on paper and this also gives better results
      //      on TestWithRealData).
      //   1 bis) Note also that this 257/256 != 256/255 helps ensure
      //      that no overflow can happen, even with offset=255.
      //   2) Our PRNG, xorshift, inherently wants to generate values
      //      in [1..255] so this saves a couple of instructions.
      rounding_offset = prng->get();
      break;
    default:
      assert(false);
      rounding_offset = vdupq_n_u8(0);
  }

  // Add the rounding offset.
  x[0] = vaddw_u8(x[0], vget_low_u8(rounding_offset));
  x[1] = vaddw_u8(x[1], vget_high_u8(rounding_offset));

  // Multiply by 257/256, which is close enough to 256/255.
  x[0] = vaddq_u16(x[0], vshrq_n_u16(x[0], 8));
  x[1] = vaddq_u16(x[1], vshrq_n_u16(x[1], 8));

  uint8x8_t y[2];
  // Divide again by 256.
  y[0] = vshrn_n_u16(x[0], 8);
  y[1] = vshrn_n_u16(x[1], 8);

  return vcombine_u8(y[0], y[1]);
}

typedef SideMap<const std::uint8_t, SideMapOrder::WidthMajor>
    WidthMajorUint8SideMap;

template <int Cells>
using DepthMajorSideFormatNCells4x2 = KernelSideFormat<CellFormat<4, 2>, Cells>;

template <int Cells>
class PackingRegisterBlock<
    WidthMajorUint8SideMap,
    PackedSideBlock<DepthMajorSideFormatNCells4x2<Cells> > >
    : public PackingRegisterBlockBase<
          WidthMajorUint8SideMap,
          PackedSideBlock<DepthMajorSideFormatNCells4x2<Cells> > > {
 public:
  typedef DepthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  typedef NEONPseudoRandomNonzeroBytesGenerator
      PseudoRandomNonzeroBytesGenerator;

  template <typename BitDepth, RoundingMode Rounding>
  void Pack(PackedSideBlock<KernelSideFormat>* dst, int start_width,
            PseudoRandomNonzeroBytesGenerator* prng) {
    static const int kBits = BitDepth::kBits;
    static const std::uint16_t kMaxVal = (1 << kBits) - 1;
    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* const src_ptr = this->complete_src_.data();
    const int stride = this->complete_src_.stride();
    // Load and requantize source WidthMajor data
    uint8x16_t src_lines[4 * kCells];
    for (int i = 0; i < 4 * kCells; i++) {
      src_lines[i] = Requantize<BitDepth, Rounding>(
        vld1q_u8(src_ptr + i * stride), prng);
    }
    // Reorder the data within registers to make DepthMajor 4x2 cells
    uint8x16x2_t src_lines_intertwined_2x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_2x[2 * i] =
          vzipq_u8(src_lines[4 * i], src_lines[4 * i + 2]);
      src_lines_intertwined_2x[2 * i + 1] =
          vzipq_u8(src_lines[4 * i + 1], src_lines[4 * i + 3]);
    }
    uint8x16x2_t src_lines_intertwined_4x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_4x[2 * i] =
          vzipq_u8(src_lines_intertwined_2x[2 * i].val[0],
                   src_lines_intertwined_2x[2 * i + 1].val[0]);
      src_lines_intertwined_4x[2 * i + 1] =
          vzipq_u8(src_lines_intertwined_2x[2 * i].val[1],
                   src_lines_intertwined_2x[2 * i + 1].val[1]);
    }
    // Store the resulting DepthMajor 4x2 cells in the destination packed block
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        for (int cell = 0; cell < kCells; cell++) {
          uint8x8_t value = vget_low_u8(
              src_lines_intertwined_4x[2 * cell + outer].val[inner]);
          vst1_u8(dst_ptr, value);
          dst_ptr += 8;
        }
        for (int cell = 0; cell < kCells; cell++) {
          uint8x8_t value = vget_high_u8(
              src_lines_intertwined_4x[2 * cell + outer].val[inner]);
          vst1_u8(dst_ptr, value);
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
          sums_of_2_cells[cell][i] = vaddl_u8(
              vget_low_u8(
                  src_lines_intertwined_4x[2 * cell + outer].val[inner]),
              vget_high_u8(
                  src_lines_intertwined_4x[2 * cell + outer].val[inner]));
        }
      }
    }
    int32x4_t sums_of_4_cells[kCells][4];
    for (int i = 0; i < 4; i++) {
      for (int cell = 0; cell < kCells; cell++) {
        sums_of_4_cells[cell][i] = vreinterpretq_s32_u32(
            vaddl_u16(vget_low_u16(sums_of_2_cells[cell][i]),
                      vget_high_u16(sums_of_2_cells[cell][i])));
      }
    }
    // Update the rank_one_update vector
    for (int cell = 0; cell < kCells; cell++) {
      int32x4_t s01 =
          vaddq_s32(sums_of_4_cells[cell][0], sums_of_4_cells[cell][1]);
      int32x4_t s23 =
          vaddq_s32(sums_of_4_cells[cell][2], sums_of_4_cells[cell][3]);
      int32x4_t s = vaddq_s32(s01, s23);
      int32x4_t u = vmulq_n_s32(s, dst->rank_one_update_multiplier());
      std::int32_t* rank_one_update_ptr =
          dst->rank_one_update() + start_width + 4 * cell;
      vst1q_s32(rank_one_update_ptr,
                vaddq_s32(u, vld1q_s32(rank_one_update_ptr)));
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

template <int Cells>
using WidthMajorSideFormatNCells4x2 =
    KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, Cells>;

template <int Cells>
class PackingRegisterBlock<
    WidthMajorUint8SideMap,
    PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells> > >
    : public PackingRegisterBlockBase<
          WidthMajorUint8SideMap,
          PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells> > > {
 public:
  typedef WidthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  typedef NEONPseudoRandomNonzeroBytesGenerator
      PseudoRandomNonzeroBytesGenerator;

  template <typename BitDepth, RoundingMode Rounding>
  void Pack(PackedSideBlock<KernelSideFormat>* dst, int start_width,
            PseudoRandomNonzeroBytesGenerator* prng) {
    static const int kBits = BitDepth::kBits;
    static const std::uint16_t kMaxVal = (1 << kBits) - 1;
    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* src_ptr = this->complete_src_.data();
    const int stride = this->complete_src_.stride();
    // Load and requantize source WidthMajor data
    uint16x8_t src_lines[kCells * 4];
    for (int i = 0; i < kCells; i++) {
// This packing path is used with our current
// less-than-8-bit kernel, and the partial unrolling of this loop
// results in substantially faster code (thanks to better
// register allocation) on Nexus 5.

#define GEMMLOWP_UNROLLED_LOOP_ITER(k)                     \
  src_lines[4 * i + k] =                                   \
      vreinterpretq_u16_u8(Requantize<BitDepth, Rounding>( \
        vld1q_u8(src_ptr), prng));                         \
  src_ptr += stride;

      GEMMLOWP_UNROLLED_LOOP_ITER(0)
      GEMMLOWP_UNROLLED_LOOP_ITER(1)
      GEMMLOWP_UNROLLED_LOOP_ITER(2)
      GEMMLOWP_UNROLLED_LOOP_ITER(3)

#undef GEMMLOWP_UNROLLED_LOOP_ITER
    }
    // Reorder the data within registers to make WidthMajor 4x2 cells
    uint16x8x2_t src_lines_intertwined_2x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_2x[2 * i] =
          vzipq_u16(src_lines[4 * i], src_lines[4 * i + 2]);
      src_lines_intertwined_2x[2 * i + 1] =
          vzipq_u16(src_lines[4 * i + 1], src_lines[4 * i + 3]);
    }
    uint16x8x2_t src_lines_intertwined_4x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_4x[2 * i] =
          vzipq_u16(src_lines_intertwined_2x[2 * i].val[0],
                    src_lines_intertwined_2x[2 * i + 1].val[0]);
      src_lines_intertwined_4x[2 * i + 1] =
          vzipq_u16(src_lines_intertwined_2x[2 * i].val[1],
                    src_lines_intertwined_2x[2 * i + 1].val[1]);
    }
    // Store the resulting WidthMajor 4x2 cells in the destination packed block
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        for (int cell = 0; cell < kCells; cell++) {
          uint8x8_t value = vreinterpret_u8_u16(vget_low_u16(
              src_lines_intertwined_4x[2 * cell + outer].val[inner]));
          vst1_u8(dst_ptr, value);
          dst_ptr += 8;
        }
        for (int cell = 0; cell < kCells; cell++) {
          uint8x8_t value = vreinterpret_u8_u16(vget_high_u16(
              src_lines_intertwined_4x[2 * cell + outer].val[inner]));
          vst1_u8(dst_ptr, value);
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
          sums_of_2[cell][i] = vpaddlq_u8(vreinterpretq_u8_u16(
              src_lines_intertwined_4x[2 * cell + outer].val[inner]));
        }
      }
    }
    uint16x8_t sums_of_4[kCells][2];
    for (int i = 0; i < 2; i++) {
      for (int cell = 0; cell < kCells; cell++) {
        sums_of_4[cell][i] =
            vaddq_u16(sums_of_2[cell][2 * i], sums_of_2[cell][2 * i + 1]);
      }
    }
    uint16x8_t sums_of_8[kCells];
    for (int cell = 0; cell < kCells; cell++) {
      sums_of_8[cell] = vaddq_u16(sums_of_4[cell][0], sums_of_4[cell][1]);
    }

    uint16x4_t sums_of_16[kCells];
    for (int cell = 0; cell < kCells; cell++) {
      sums_of_16[cell] = vadd_u16(vget_low_u16(sums_of_8[cell]),
                                  vget_high_u16(sums_of_8[cell]));
    }
    // Update the rank_one_update vector
    for (int cell = 0; cell < kCells; cell++) {
      int32x4_t s = vreinterpretq_s32_u32(vmovl_u16(sums_of_16[cell]));
      int32x4_t u = vmulq_n_s32(s, dst->rank_one_update_multiplier());
      std::int32_t* rank_one_update_ptr =
          dst->rank_one_update() + start_width + 4 * cell;
      vst1q_s32(rank_one_update_ptr,
                vaddq_s32(u, vld1q_s32(rank_one_update_ptr)));
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_PACK_NEON_H_
