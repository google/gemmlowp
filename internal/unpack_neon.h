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

template <>
struct UnpackResultImpl<MatrixMap<std::uint8_t, MapOrder::ColMajor>> {
  typedef MatrixMap<std::uint8_t, MapOrder::ColMajor> ResultBlockType;
  static void Unpack(ResultBlockType* dst, const PackedResultInt32& src,
                     int depth, const std::int32_t* lhs_rank_one_update,
                     const std::int32_t* rhs_rank_one_update,
                     std::int32_t lhs_offset, std::int32_t rhs_offset,
                     std::int32_t result_offset, std::int32_t result_mult_int,
                     std::int32_t result_shift) {
    ScopedProfilingLabel label("optimized path (NEON)");

    auto src_map = src.Map();
    std::int32_t rank0update = lhs_offset * rhs_offset * depth;
    std::int32_t preshift_offset = 1 << (result_shift - 1);
    int32x4_t shift_reg = vdupq_n_s32(-result_shift);
    for (int c = 0; c < dst->cols(); c++) {
      std::uint8_t* dst_ptr = dst->data(0, c);
      const std::int32_t* src_ptr = src_map.data(0, c);
      const std::int32_t* rank_one_update_ptr = lhs_rank_one_update;
      std::int32_t rank1update = rhs_rank_one_update[c];
      std::int32_t constant_offset = rank1update + rank0update + result_offset;

      int dst_rows_aligned16 = RoundDown<16>(dst->rows());
      for (int r = 0; r < dst_rows_aligned16; r += 16) {
        int32x4_t q0 = vld1q_s32(src_ptr);
        int32x4_t q1 = vld1q_s32(src_ptr + 4);
        int32x4_t q2 = vld1q_s32(src_ptr + 8);
        int32x4_t q3 = vld1q_s32(src_ptr + 12);
        q0 = vaddq_s32(q0, vld1q_s32(rank_one_update_ptr));
        q1 = vaddq_s32(q1, vld1q_s32(rank_one_update_ptr + 4));
        q2 = vaddq_s32(q2, vld1q_s32(rank_one_update_ptr + 8));
        q3 = vaddq_s32(q3, vld1q_s32(rank_one_update_ptr + 12));
        int32x4_t o = vdupq_n_s32(constant_offset);
        q0 = vaddq_s32(q0, o);
        q1 = vaddq_s32(q1, o);
        q2 = vaddq_s32(q2, o);
        q3 = vaddq_s32(q3, o);
        q0 = vmulq_n_s32(q0, result_mult_int);
        q1 = vmulq_n_s32(q1, result_mult_int);
        q2 = vmulq_n_s32(q2, result_mult_int);
        q3 = vmulq_n_s32(q3, result_mult_int);
        o = vdupq_n_s32(preshift_offset);
        q0 = vaddq_s32(q0, o);
        q1 = vaddq_s32(q1, o);
        q2 = vaddq_s32(q2, o);
        q3 = vaddq_s32(q3, o);
        q0 = vshlq_s32(q0, shift_reg);
        q1 = vshlq_s32(q1, shift_reg);
        q2 = vshlq_s32(q2, shift_reg);
        q3 = vshlq_s32(q3, shift_reg);
        int16x4_t q0_16 = vqmovn_s32(q0);
        int16x4_t q1_16 = vqmovn_s32(q1);
        int16x4_t q2_16 = vqmovn_s32(q2);
        int16x4_t q3_16 = vqmovn_s32(q3);
        uint8x8_t q0_8 = vqmovun_s16(vcombine_s16(q0_16, q0_16));
        uint8x8_t q1_8 = vqmovun_s16(vcombine_s16(q1_16, q1_16));
        uint8x8_t q2_8 = vqmovun_s16(vcombine_s16(q2_16, q2_16));
        uint8x8_t q3_8 = vqmovun_s16(vcombine_s16(q3_16, q3_16));
        vst1_lane_u32(reinterpret_cast<std::uint32_t*>(dst_ptr),
                      vreinterpret_u32_u8(q0_8), 0);
        vst1_lane_u32(reinterpret_cast<std::uint32_t*>(dst_ptr + 4),
                      vreinterpret_u32_u8(q1_8), 0);
        vst1_lane_u32(reinterpret_cast<std::uint32_t*>(dst_ptr + 8),
                      vreinterpret_u32_u8(q2_8), 0);
        vst1_lane_u32(reinterpret_cast<std::uint32_t*>(dst_ptr + 12),
                      vreinterpret_u32_u8(q3_8), 0);
        dst_ptr += 16;
        src_ptr += 16;
        rank_one_update_ptr += 16;
      }

      // We have finished handling groups of 16 entries at once; now
      // try to handle 4 entries at once.
      int dst_rows_aligned4 = RoundDown<4>(dst->rows());
      for (int r = dst_rows_aligned16; r < dst_rows_aligned4; r += 4) {
        int32x4_t q = vld1q_s32(src_ptr);
        q = vaddq_s32(q, vld1q_s32(rank_one_update_ptr));
        q = vaddq_s32(q, vdupq_n_s32(constant_offset));
        q = vmulq_n_s32(q, result_mult_int);
        q = vaddq_s32(q, vdupq_n_s32(preshift_offset));
        q = vshlq_s32(q, shift_reg);
        int16x4_t q_16 = vqmovn_s32(q);
        uint8x8_t q_8 = vqmovun_s16(vcombine_s16(q_16, q_16));
        vst1_lane_u32(reinterpret_cast<std::uint32_t*>(dst_ptr),
                      vreinterpret_u32_u8(q_8), 0);
        dst_ptr += 4;
        src_ptr += 4;
        rank_one_update_ptr += 4;
      }
      // We have finished handling 4 entries at once; now handle
      // remaining entries one by one.
      for (int r = dst_rows_aligned4; r < dst->rows(); r++) {
        std::int32_t q = src_map(r, c);
        q += lhs_rank_one_update[r] + rank1update + rank0update;
        q = ((q + result_offset) * result_mult_int +
             (1 << (result_shift - 1))) >>
            result_shift;
        (*dst)(r, c) = q > 255 ? 255 : q < 0 ? 0 : q;
      }
    }
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_UNPACK_NEON_H_
