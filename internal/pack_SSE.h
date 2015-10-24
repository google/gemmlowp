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

// pack_SSE.h: optimized SSE specializations of the templates in pack.h.

#ifndef GEMMLOWP_INTERNAL_PACK_SSE_H_
#define GEMMLOWP_INTERNAL_PACK_SSE_H_

#include "pack.h"
#include <xmmintrin.h>

namespace gemmlowp {

// Requantizes source values pointed by raw_src_ptr in [0..255] range
// to the range specified by BitDepth, [0..((2^bits)-1)].
// This is in-place requantization, where the input is 
// not modified if 8bit integers are used. SSE does not
// have less than 8bit kernels currently. Altought SSE registers
// hold 16 uint8_t elements, only first 8 uint8_t elements are 
// requantized. The packing only use first 8 uint8_t elements 
// of the SSE registers. Therefore, requantizing all 16 uint8_t 
// elements will be wasteful computation.
template <typename QuantizationParams>
void SSERequantize(
    __m128i* raw_src_ptr,
    ScalarRoundingOffsetGenerator<QuantizationParams::kRoundingMode>*
        rounding_offset_generator) {
  static const int kBits = QuantizationParams::BitDepth::kBits;
  static const std::uint8_t kMaxVal = (1 << kBits) - 1;
  if (kBits == 8) {
    return;
  }

  std::uint8_t* raw_src_ui8_ptr = (std::uint8_t*) &raw_src_ptr[0];

  // modify only first 8 elements in the register (see note above)
  for (int i = 0; i < 8; ++i) {
    std::uint16_t scaled = static_cast<std::uint16_t>(raw_src_ui8_ptr[i]) * kMaxVal;
    std::uint8_t rounding_offset = rounding_offset_generator->get();
    raw_src_ui8_ptr[i] = (scaled + rounding_offset) / 255;
  }
}

// TODO: Add DepthMajorUint8SideMap

typedef SideMap<const std::uint8_t, SideMapOrder::WidthMajor>
    WidthMajorUint8SideMap;

template <int Cells>
using WidthMajorSideFormatNCells4x2 =
    KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, Cells>;

template <typename QuantizationParams, int Cells>
class PackingRegisterBlock<
    QuantizationParams, WidthMajorUint8SideMap,
    PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells> > >
    : public PackingRegisterBlockBase<
          QuantizationParams, WidthMajorUint8SideMap,
          PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells> > > {
 public:
  typedef WidthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;
  static const int kKernelWidth2 = (kKernelWidth / kCellWidth / 2) * kCellWidth;

  typedef ScalarRoundingOffsetGenerator<QuantizationParams::kRoundingMode>
      RoundingOffsetGenerator;

  void Pack(PackedSideBlock<KernelSideFormat>* dst, int start_width,
      RoundingOffsetGenerator* rounding_offset_generator) {
    int rank_one_mult = dst->rank_one_update_multiplier();
    int is_mult_lt_zero = rank_one_mult < 0;
    int rank_one_mult_abs = is_mult_lt_zero ? -rank_one_mult : rank_one_mult;
    // extract high and low parts of the multiplier
    int16_t rank_one_mult_lo = (rank_one_mult_abs << 16) >> 16;
    int16_t rank_one_mult_hi = rank_one_mult_abs >> 16;
    int is_bit15_set = (rank_one_mult_lo & 0x8000);

    std::uint8_t* dst_ptr = dst->current_data();
    const int width_stride = this->complete_src_.width_stride();
    int depth_step = 8;

    // If the multiplier cannot be represented by int16_t, then we need to do exra 
    // work for integer multiplication. Hopefully, this is not the case most
    // of the time since this involves more computations.
    //
    // Splits (ab)x(c) integer multiplication into two parts:
    //
    // a|b
    //  |c
    // ----
    // (a*c)<<16 + c*b
    //
    // where, a,b are 16bit parts of 32bit integer and c is a 16bit integer.
    //
    // Here, special care is required if sign bit location of b is set. 
    // The code protected with is_bit15_set handles this case.
    if (rank_one_mult_hi || is_bit15_set) {
      // erase the sign bit if it is set
      rank_one_mult_lo &= (~0x8000);
      rank_one_mult_lo = is_mult_lt_zero ? -rank_one_mult_lo : rank_one_mult_lo;
      rank_one_mult_hi = is_mult_lt_zero ? -rank_one_mult_hi : rank_one_mult_hi;
      __m128i mult_lo_xmm = _mm_set1_epi16(rank_one_mult_lo);
      __m128i mult_hi_xmm = _mm_set1_epi16(rank_one_mult_hi);

      __m128i mult_shift_xmm = _mm_set_epi32(0, 0, 0, 16);
      __m128i bit15_mult = _mm_set1_epi16(0);
      __m128i bit15_sign = _mm_set1_epi32(0);

      if (is_bit15_set) {
        bit15_mult = _mm_set1_epi16(0x4000);
        if (is_mult_lt_zero) {
          bit15_sign = _mm_set1_epi32(0x80000000);
        }
      }

      for (int cell_start_depth = 0; cell_start_depth < kRegisterSize;
          cell_start_depth += depth_step) {

        for (int cell_start_width = 0; cell_start_width < kKernelWidth;
            cell_start_width += kCellWidth) {
          std::int32_t* cell_rank_one_update_ptr =
            dst->rank_one_update() + start_width + cell_start_width;
          const std::uint8_t* src_data = this->complete_src_.data(cell_start_width, cell_start_depth);

          __m128i xmm1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src_data[0]));
          __m128i xmm2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src_data[1*width_stride]));
          __m128i xmm3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src_data[2*width_stride]));
          __m128i xmm4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src_data[3*width_stride]));

          __m128i xmm5 = _mm_unpacklo_epi16(xmm1, xmm2);
          __m128i xmm8 = _mm_shuffle_epi32 (xmm5, 0x31);

          __m128i xmm6 = _mm_unpacklo_epi16(xmm3, xmm4);
          __m128i xmm7 = _mm_shuffle_epi32 (xmm6, 0x80);

          __m128i xmm9  = _mm_blend_epi16(xmm5, xmm7, 0xcc);
          SSERequantize<QuantizationParams>(&xmm9, rounding_offset_generator);

          __m128i xmm10 = _mm_blend_epi16(xmm8, xmm6, 0xcc);
          SSERequantize<QuantizationParams>(&xmm10, rounding_offset_generator);

          _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst_ptr[0]), xmm9);
          _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst_ptr[kCellSize*kCells]), xmm10);

          __m128i xmm11 = _mm_shuffle_epi32(xmm9 , 0xee);
          SSERequantize<QuantizationParams>(&xmm11, rounding_offset_generator);

          __m128i xmm12 = _mm_shuffle_epi32(xmm10, 0xee);
          SSERequantize<QuantizationParams>(&xmm12, rounding_offset_generator);

          _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst_ptr[2*kCellSize*kCells]), xmm11);
          _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst_ptr[3*kCellSize*kCells]), xmm12);

          xmm1 = _mm_cvtepu8_epi16(xmm9);
          // low-16bits multiplication
          xmm2 = _mm_madd_epi16(xmm1, mult_lo_xmm);
          // high-16bits multiplication
          xmm3 = _mm_madd_epi16(xmm1, mult_hi_xmm);
          xmm3 = _mm_sll_epi32(xmm3, mult_shift_xmm);
          xmm2 = _mm_add_epi32(xmm3, xmm2);
          // branch-free code for handlig 15th-bit (treated as sign bit)
          xmm4 = _mm_madd_epi16(xmm1, bit15_mult);
          xmm4 = _mm_or_si128(bit15_sign, xmm4);
          xmm2 = _mm_add_epi32(xmm2, xmm4);
          xmm2 = _mm_add_epi32(xmm2, xmm4);
          __m128i rank_one_update_xmm = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&cell_rank_one_update_ptr[0]));
          rank_one_update_xmm = _mm_add_epi32(rank_one_update_xmm, xmm2);

          xmm1 = _mm_cvtepu8_epi16(xmm10);
          // low-16bits multiplication
          xmm2 = _mm_madd_epi16(xmm1, mult_lo_xmm);
          // high-16bits multiplication
          xmm3 = _mm_madd_epi16(xmm1, mult_hi_xmm);
          xmm3 = _mm_sll_epi32(xmm3, mult_shift_xmm);
          xmm2 = _mm_add_epi32(xmm3, xmm2);
          // branch-free code for handlig 15th-bit (treated as sign bit)
          xmm4 = _mm_madd_epi16(xmm1, bit15_mult);
          xmm4 = _mm_or_si128(bit15_sign, xmm4);
          xmm2 = _mm_add_epi32(xmm2, xmm4);
          xmm2 = _mm_add_epi32(xmm2, xmm4);
          rank_one_update_xmm = _mm_add_epi32(rank_one_update_xmm, xmm2);

          xmm1 = _mm_cvtepu8_epi16(xmm11);
          // low-16bits multiplication
          xmm2 = _mm_madd_epi16(xmm1, mult_lo_xmm);
          // high-16bits multiplication
          xmm3 = _mm_madd_epi16(xmm1, mult_hi_xmm);
          xmm3 = _mm_sll_epi32(xmm3, mult_shift_xmm);
          xmm2 = _mm_add_epi32(xmm3, xmm2);
          // branch-free code for handlig 15th-bit (treated as sign bit)
          xmm4 = _mm_madd_epi16(xmm1, bit15_mult);
          xmm4 = _mm_or_si128(bit15_sign, xmm4);
          xmm2 = _mm_add_epi32(xmm2, xmm4);
          xmm2 = _mm_add_epi32(xmm2, xmm4);
          rank_one_update_xmm = _mm_add_epi32(rank_one_update_xmm, xmm2);

          xmm1 = _mm_cvtepu8_epi16(xmm12);
          // low-16bits multiplication
          xmm2 = _mm_madd_epi16(xmm1, mult_lo_xmm);
          // high-16bits multiplication
          xmm3 = _mm_madd_epi16(xmm1, mult_hi_xmm);
          xmm3 = _mm_sll_epi32(xmm3, mult_shift_xmm);
          xmm2 = _mm_add_epi32(xmm3, xmm2);
          // branch-free code for handlig 15th-bit (treated as sign bit)
          xmm4 = _mm_madd_epi16(xmm1, bit15_mult);
          xmm4 = _mm_or_si128(bit15_sign, xmm4);
          xmm2 = _mm_add_epi32(xmm2, xmm4);
          xmm2 = _mm_add_epi32(xmm2, xmm4);
          rank_one_update_xmm = _mm_add_epi32(rank_one_update_xmm, xmm2);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(&cell_rank_one_update_ptr[0]), rank_one_update_xmm);

          dst_ptr += kCellSize;
        }
        dst_ptr += 3*kCellSize*kCells;
      }
    } else {
      // safe to use only low 16 bit part of the multiplier
      rank_one_mult_lo = is_mult_lt_zero ?  -rank_one_mult_lo : rank_one_mult_lo;
      __m128i mult_lo_xmm = _mm_set1_epi16(rank_one_mult_lo);
      for (int cell_start_depth = 0; cell_start_depth < kRegisterSize;
          cell_start_depth += depth_step) {

        for (int cell_start_width = 0; cell_start_width < kKernelWidth;
            cell_start_width += kCellWidth) {
          std::int32_t* cell_rank_one_update_ptr =
            dst->rank_one_update() + start_width + cell_start_width;
          const std::uint8_t* src_data = this->complete_src_.data(cell_start_width, cell_start_depth);

          __m128i xmm1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src_data[0]));
          __m128i xmm2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src_data[1*width_stride]));
          __m128i xmm3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src_data[2*width_stride]));
          __m128i xmm4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src_data[3*width_stride]));

          __m128i xmm5 = _mm_unpacklo_epi16(xmm1, xmm2);
          __m128i xmm8 = _mm_shuffle_epi32 (xmm5, 0x31);

          __m128i xmm6 = _mm_unpacklo_epi16(xmm3, xmm4);
          __m128i xmm7 = _mm_shuffle_epi32 (xmm6, 0x80);

          __m128i xmm9  = _mm_blend_epi16(xmm5, xmm7, 0xcc);
          SSERequantize<QuantizationParams>(&xmm9, rounding_offset_generator);

          __m128i xmm10 = _mm_blend_epi16(xmm8, xmm6, 0xcc);
          SSERequantize<QuantizationParams>(&xmm10, rounding_offset_generator);

          _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst_ptr[0]), xmm9);
          _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst_ptr[kCellSize*kCells]), xmm10);

          __m128i xmm11 = _mm_shuffle_epi32(xmm9 , 0xee);
          SSERequantize<QuantizationParams>(&xmm11, rounding_offset_generator);

          __m128i xmm12 = _mm_shuffle_epi32(xmm10, 0xee);
          SSERequantize<QuantizationParams>(&xmm12, rounding_offset_generator);

          _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst_ptr[2*kCellSize*kCells]), xmm11);
          _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst_ptr[3*kCellSize*kCells]), xmm12);

          xmm1 = _mm_cvtepu8_epi16(xmm9);
          xmm2 = _mm_madd_epi16(xmm1, mult_lo_xmm);
          __m128i rank_one_update_xmm = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&cell_rank_one_update_ptr[0]));
          rank_one_update_xmm = _mm_add_epi32(rank_one_update_xmm, xmm2);

          xmm1 = _mm_cvtepu8_epi16(xmm10);
          xmm2 = _mm_madd_epi16(xmm1, mult_lo_xmm);
          rank_one_update_xmm = _mm_add_epi32(rank_one_update_xmm, xmm2);

          xmm1 = _mm_cvtepu8_epi16(xmm11);
          xmm2 = _mm_madd_epi16(xmm1, mult_lo_xmm);
          rank_one_update_xmm = _mm_add_epi32(rank_one_update_xmm, xmm2);

          xmm1 = _mm_cvtepu8_epi16(xmm12);
          xmm2 = _mm_madd_epi16(xmm1, mult_lo_xmm);
          rank_one_update_xmm = _mm_add_epi32(rank_one_update_xmm, xmm2);

          _mm_storeu_si128(reinterpret_cast<__m128i*>(&cell_rank_one_update_ptr[0]), rank_one_update_xmm);
          dst_ptr += kCellSize;
        }
        dst_ptr += 3*kCellSize*kCells;
      }
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_PACK_SSE_H_
