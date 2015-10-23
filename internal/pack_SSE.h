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
// TODO: Delete below
#include <iostream>

namespace gemmlowp {

template <RoundingMode tRoundingMode>
class SSERoundingOffsetGenerator {
 public:
  std::uint8_t get() {
      assert(false);
      return 0;
#if 0
    assert(false);  // This generic path should never be called.
    return vdupq_n_u8(0);
#endif
  }
};

// A RoundingOffsetGenerator for rounding-to-nearest, always returning
// the midpoint value 127.
template <>
class SSERoundingOffsetGenerator<RoundingMode::Nearest> {
 public:
  std::uint8_t get() { 
      return 127;
  }
};

// Variant of SSERoundingOffsetGenerator that produces
// random SSE 128-bit vectors using a 8-bit Xorshift.
template <>
class SSERoundingOffsetGenerator<RoundingMode::ProbabilisticXorshift> {
 public:
  SSERoundingOffsetGenerator() {
    x_ = 128;
  }

  std::uint8_t get() {

    std::uint8_t result = x_ - 1;
    // Xorshift8(7,5,3)
    x_ ^= x_ << 7;
    x_ ^= x_ >> 5;
    x_ ^= x_ << 3;
    return result;
  }

 private:
  std::uint8_t x_;
};

// Variant of SSERoundingOffsetGenerator that produces
// rounding vectors using an 8-bit add/mod low-discrepancy sequence.
template <>
class SSERoundingOffsetGenerator<RoundingMode::ProbabilisticAddmod> {
  static const std::uint8_t AddConst = 97;
 public:
  SSERoundingOffsetGenerator() {
    x_ = 1;
  }

  std::uint8_t get() {
    x_ += (AddConst + (x_ >= (255-AddConst)));
    return x_;
  }

 private:
  // State
  std::uint8_t x_;
};


// Requantizes source uint8 values in [0..255] range
// to the range specified by BitDepth, [0..((2^bits)-1)].
// Bias must be avoided. Currently this is achieved
// by probabilistic rounding.
template <typename QuantizationParams>
std::uint8_t Requantize(
    std::uint8_t raw_src_val,
    SSERoundingOffsetGenerator<QuantizationParams::kRoundingMode>*
        rounding_offset_generator) {
  static const int kBits = QuantizationParams::BitDepth::kBits;
  static const std::uint8_t kMaxVal = (1 << kBits) - 1;
  if (kBits == 8) {
    return raw_src_val;
  }

  std::uint16_t scaled = static_cast<std::uint16_t>(raw_src_val) * kMaxVal;
  std::uint8_t rounding_offset = rounding_offset_generator->get();
  return (scaled + rounding_offset) / 255;
}

template <typename QuantizationParams>
void SSERequantize(
    __m128i* raw_src_ptr,
    SSERoundingOffsetGenerator<QuantizationParams::kRoundingMode>*
        rounding_offset_generator) {
  static const int kBits = QuantizationParams::BitDepth::kBits;
  static const std::uint8_t kMaxVal = (1 << kBits) - 1;
  if (kBits == 8) {
    return;
  }

  std::uint8_t* raw_src_ui8_ptr = (std::uint8_t*) &raw_src_ptr[0];

  for (int i = 0; i < 8; ++i) {
    std::uint16_t scaled = static_cast<std::uint16_t>(raw_src_ui8_ptr[i]) * kMaxVal;
    std::uint8_t rounding_offset = rounding_offset_generator->get();
    raw_src_ui8_ptr[i] = (scaled + rounding_offset) / 255;
  }
}



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

  typedef SSERoundingOffsetGenerator<QuantizationParams::kRoundingMode>
      RoundingOffsetGenerator;

  void Pack(PackedSideBlock<KernelSideFormat>* dst, int start_width,
            RoundingOffsetGenerator* rounding_offset_generator) {
    // std::cout<< "Efe is here\n" << std::endl;
    union scratch_type { 
        std::uint8_t data[16];
        std::aligned_storage<sizeof(std::uint8_t)*16, 64> alignment_dummy;
    };
    int rank_one_mult = dst->rank_one_update_multiplier();
    int16_t rank_one_mult_lo = (rank_one_mult << 16) >> 16;
    int16_t rank_one_mult_hi = rank_one_mult >> 16;
    int16_t rank_one_mult_lo_array[8] = {rank_one_mult_lo, rank_one_mult_lo, 
      rank_one_mult_lo, rank_one_mult_lo, 
      rank_one_mult_lo, rank_one_mult_lo,
      rank_one_mult_lo, rank_one_mult_lo};
    int16_t rank_one_mult_hi_array[8] = {rank_one_mult_hi, rank_one_mult_hi, 
      rank_one_mult_hi, rank_one_mult_hi, 
      rank_one_mult_hi, rank_one_mult_hi, 
      rank_one_mult_hi, rank_one_mult_hi};
    __m128i mult_lo_xmm = _mm_loadu_si128((__m128i*) &rank_one_mult_lo_array[0]);
    __m128i mult_hi_xmm = _mm_loadu_si128((__m128i*) &rank_one_mult_hi_array[0]);


    std::uint8_t* dst_ptr = dst->current_data();
    int depth_step = 8;
    const int width_stride = this->complete_src_.width_stride();
    for (int cell_start_depth = 0; cell_start_depth < kRegisterSize;
         cell_start_depth += depth_step) {

      for (int cell_start_width = 0; cell_start_width < kKernelWidth;
           cell_start_width += kCellWidth) {
        std::int32_t* cell_rank_one_update_ptr =
          dst->rank_one_update() + start_width + cell_start_width;
        const std::uint8_t* src_data = this->complete_src_.data(cell_start_width, cell_start_depth);

#if 1
        __m128i xmm1 = _mm_loadl_epi64((__m128i*) &src_data[0]);
        __m128i xmm2 = _mm_loadl_epi64((__m128i*) &src_data[1*width_stride]);
        __m128i xmm3 = _mm_loadl_epi64((__m128i*) &src_data[2*width_stride]);
        __m128i xmm4 = _mm_loadl_epi64((__m128i*) &src_data[3*width_stride]);

        __m128i xmm5 = _mm_unpacklo_epi16(xmm1, xmm2);
        __m128i xmm8 = _mm_shuffle_epi32 (xmm5, 0x31);

        __m128i xmm6 = _mm_unpacklo_epi16(xmm3, xmm4);
        __m128i xmm7 = _mm_shuffle_epi32 (xmm6, 0x80);

        __m128i xmm9  = _mm_blend_epi16(xmm5, xmm7, 0xcc);
        SSERequantize<QuantizationParams>(&xmm9, rounding_offset_generator);

        __m128i xmm10 = _mm_blend_epi16(xmm8, xmm6, 0xcc);
        SSERequantize<QuantizationParams>(&xmm10, rounding_offset_generator);

        _mm_storel_epi64((__m128i*) &dst_ptr[0], xmm9);
        _mm_storel_epi64((__m128i*) &dst_ptr[kCellSize*kCells], xmm10);

        __m128i xmm11 = _mm_shuffle_epi32(xmm9 , 0xee);
        SSERequantize<QuantizationParams>(&xmm11, rounding_offset_generator);

        __m128i xmm12 = _mm_shuffle_epi32(xmm10, 0xee);
        SSERequantize<QuantizationParams>(&xmm12, rounding_offset_generator);

        _mm_storel_epi64((__m128i*) &dst_ptr[2*kCellSize*kCells] , xmm11);
        _mm_storel_epi64((__m128i*) &dst_ptr[3*kCellSize*kCells], xmm12);

        xmm1 = _mm_cvtepu8_epi16(xmm9);
        xmm2 = _mm_madd_epi16(xmm1, mult_lo_xmm);
        __m128i rank_one_update_xmm = _mm_loadu_si128((__m128i*) &cell_rank_one_update_ptr[0]);
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

        _mm_storeu_si128((__m128i*) &cell_rank_one_update_ptr[0], rank_one_update_xmm);

#else
        scratch_type scratch_pad;
        for (int i = 0; i < 8; ++i) {
          scratch_pad.data[2*i]   = Requantize<QuantizationParams>(
              src_data[i*width_stride], rounding_offset_generator);
          scratch_pad.data[2*i+1] = Requantize<QuantizationParams>(
              src_data[1+i*width_stride], rounding_offset_generator);
        }

        __m128i src_8bit_xmm = _mm_loadl_epi64((__m128i*) &scratch_pad.data[0]);
        _mm_storel_epi64((__m128i*) dst_ptr, src_8bit_xmm);

        __m128i src_16bit_xmm0 = _mm_cvtepu8_epi16(src_8bit_xmm);
        __m128i rank_one_update_xmm0 = _mm_madd_epi16(src_16bit_xmm0, mult_lo_xmm);
        __m128i rank_one_update_xmm2 = _mm_loadu_si128((__m128i*) &cell_rank_one_update_ptr[0]);
        rank_one_update_xmm0 = _mm_add_epi32(rank_one_update_xmm0, rank_one_update_xmm2);
        _mm_storeu_si128((__m128i*) &cell_rank_one_update_ptr[0], rank_one_update_xmm0);
#endif
        dst_ptr += kCellSize;
      }
      dst_ptr += 3*kCellSize*kCells;
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_PACK_SSE_H_
