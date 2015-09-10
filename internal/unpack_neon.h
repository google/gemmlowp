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

// unpack_neon.h: optimized NEON specializations of the templates in unpack.h.

#ifndef GEMMLOWP_INTERNAL_UNPACK_NEON_H_
#define GEMMLOWP_INTERNAL_UNPACK_NEON_H_

#include "unpack.h"

#include <arm_neon.h>

namespace gemmlowp {

template <std::uint32_t numerator, std::uint32_t denominator>
int32x4_t MultiplyByConstantFraction(int32x4_t x) {
  static_assert(numerator > 0 && denominator > 0,
                "only supporting positive num/denom");

  if (numerator == denominator) {
    return x;
  }

  static const std::int32_t int_quotient =
      (numerator + denominator / 2) / denominator;
  static const std::int32_t remaining_numerator =
      numerator - int_quotient * denominator;
  static const std::int32_t scaled_remaining_numerator =
      static_cast<std::int32_t>(
          (static_cast<std::int64_t>(remaining_numerator) << 31) / denominator);
  const int32x4_t remaining_product =
      vqrdmulhq_n_s32(x, scaled_remaining_numerator);

  return vmlaq_n_s32(remaining_product, x, int_quotient);
}

template <BitDepthSetting BitDepth, typename PackedResultType>
struct UnpackResultImpl<BitDepth, MatrixMap<std::uint8_t, MapOrder::ColMajor>,
                        PackedResultType> {
  typedef MatrixMap<std::uint8_t, MapOrder::ColMajor> ResultBlockType;
  static void Unpack(ResultBlockType* dst, const PackedResultType& src,
                     int depth, const std::int32_t* lhs_rank_one_update,
                     const std::int32_t* rhs_rank_one_update,
                     std::int32_t lhs_offset, std::int32_t rhs_offset,
                     std::int32_t result_offset, std::int32_t result_mult_int,
                     std::int32_t result_shift) {
    ScopedProfilingLabel label("optimized path (NEON)");
    const int kLhsBits = LhsBitDepth<BitDepth>::kBits;
    const int kRhsBits = RhsBitDepth<BitDepth>::kBits;
    const std::int32_t kLhsMax = (1 << kLhsBits) - 1;
    const std::int32_t kRhsMax = (1 << kRhsBits) - 1;
    auto src_map = src.Map();
    const std::int32_t term_11 =
        lhs_offset * rhs_offset * depth + result_offset;
    const int32x4_t shift_reg = vdupq_n_s32(-result_shift);
    const std::int32_t preshift_offset = 1 << (result_shift - 1);
    const int32x4_t preshift_offset_reg = vdupq_n_s32(preshift_offset);
    for (int c = 0; c < dst->cols(); c++) {
      std::uint8_t* dst_ptr = dst->data(0, c);
      const std::int32_t* src_ptr = src_map.data(0, c);
      const std::int32_t* rank_one_update_ptr = lhs_rank_one_update;
      const std::int32_t raw_1x = rhs_rank_one_update[c];
      const std::int32_t term_1x =
          MultiplyByConstantFraction<255, kRhsMax>(raw_1x);
      const std::int32_t term_1x_plus_term_11 = term_1x + term_11;

      int dst_rows_aligned16 = RoundDown<16>(dst->rows());
      for (int r = 0; r < dst_rows_aligned16; r += 16) {
        int32x4_t raw_xx[4];
        for (int i = 0; i < 4; i++) {
          raw_xx[i] = vld1q_s32(src_ptr);
          src_ptr += 4;
        }
        int32x4_t raw_x1[4];
        for (int i = 0; i < 4; i++) {
          raw_x1[i] = vld1q_s32(rank_one_update_ptr);
          rank_one_update_ptr += 4;
        }
        int32x4_t term_xx[4];
        for (int i = 0; i < 4; i++) {
          term_xx[i] =
              MultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(
                  raw_xx[i]);
        }
        int32x4_t term_x1[4];
        for (int i = 0; i < 4; i++) {
          term_x1[i] = MultiplyByConstantFraction<255, kLhsMax>(raw_x1[i]);
        }
        int32x4_t q[4];
        for (int i = 0; i < 4; i++) {
          q[i] = vaddq_s32(vaddq_s32(term_xx[i], term_x1[i]),
                           vdupq_n_s32(term_1x_plus_term_11));
        }
        for (int i = 0; i < 4; i++) {
          q[i] = vmulq_n_s32(q[i], result_mult_int);
        }
        for (int i = 0; i < 4; i++) {
          q[i] = vshlq_s32(vaddq_s32(q[i], preshift_offset_reg), shift_reg);
        }
        int16x4_t q16[4];
        for (int i = 0; i < 4; i++) {
          q16[i] = vqmovn_s32(q[i]);
        }
        uint8x8_t q8[4];
        for (int i = 0; i < 4; i++) {
          q8[i] = vqmovun_s16(vcombine_s16(q16[i], q16[i]));
        }
        for (int i = 0; i < 4; i++) {
          vst1_lane_u32(reinterpret_cast<std::uint32_t*>(dst_ptr),
                        vreinterpret_u32_u8(q8[i]), 0);
          dst_ptr += 4;
        }
      }
      // We have finished handling groups of 16 entries at once; now
      // try to handle 4 entries at once.
      int dst_rows_aligned4 = RoundDown<4>(dst->rows());
      for (int r = dst_rows_aligned16; r < dst_rows_aligned4; r += 4) {
        const int32x4_t raw_xx = vld1q_s32(src_ptr);
        const int32x4_t term_xx =
            MultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(raw_xx);
        const int32x4_t raw_x1 = vld1q_s32(rank_one_update_ptr);
        const int32x4_t term_x1 =
            MultiplyByConstantFraction<255, kLhsMax>(raw_x1);
        int32x4_t q = vaddq_s32(vaddq_s32(term_xx, term_x1),
                                vdupq_n_s32(term_1x_plus_term_11));
        q = vmulq_n_s32(q, result_mult_int);
        q = vshlq_s32(vaddq_s32(q, preshift_offset_reg), shift_reg);
        int16x4_t q16 = vqmovn_s32(q);
        uint8x8_t q8 = vqmovun_s16(vcombine_s16(q16, q16));
        vst1_lane_u32(reinterpret_cast<std::uint32_t*>(dst_ptr),
                      vreinterpret_u32_u8(q8), 0);
        dst_ptr += 4;
        src_ptr += 4;
        rank_one_update_ptr += 4;
      }
      // We have finished handling 4 entries at once; now handle
      // remaining entries one by one.
      for (int r = dst_rows_aligned4; r < dst->rows(); r++) {
        std::int32_t raw_xx = src_map(r, c);
        std::int32_t raw_x1 = lhs_rank_one_update[r];
        std::int32_t term_xx =
            MultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(raw_xx);
        std::int32_t term_x1 =
            MultiplyByConstantFraction<255, kLhsMax>(raw_x1);
        std::int32_t sum = term_xx + term_x1 + term_1x_plus_term_11;
        std::int32_t result =
            (sum * result_mult_int + (1 << (result_shift - 1))) >> result_shift;
        (*dst)(r, c) = result > 255 ? 255 : result < 0 ? 0 : result;
      }
    }
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_UNPACK_NEON_H_
