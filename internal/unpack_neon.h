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

// The paths here are specifically arm 32bit assembly, not arm 64bit.
#ifdef GEMMLOWP_NEON32

template <>
struct UnpackResultImpl<MatrixMap<std::uint8_t, MapOrder::ColMajor>> {
  typedef MatrixMap<std::uint8_t, MapOrder::ColMajor> ResultBlockType;
  static void Unpack(ResultBlockType* dst, const PackedResultInt32& src,
                     int depth, const std::int32_t* lhs_rank_one_update,
                     const std::int32_t* rhs_rank_one_update,
                     std::int32_t lhs_offset, std::int32_t rhs_offset,
                     std::int32_t result_offset, std::int32_t result_mult_int,
                     std::int32_t result_shift) {
    ScopedProfilingLabel label("optimized kernel");

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

      int dst_rows_aligned4 = RoundDown<4>(dst->rows());
      int dst_rows_aligned16 = RoundDown<16>(dst->rows());

      if (dst_rows_aligned16) {
        std::uint8_t* dst_end_ptr = dst_ptr + dst_rows_aligned16;

        asm volatile(
            "vdup.32 q12, %[constant_offset]\n"
            "vdup.32 q13, %[preshift_offset]\n"
            "neg r3, %[result_shift]\n"
            "vdup.32 q14, r3\n"
            "vdup.32 q15, %[result_mult_int]\n"

            "loop_UnpackResultImplNEON_%=:\n"

            // Load a 16x1 block of the packed result matrix
            // (so 16 contiguous entries in one column).
            "vld1.32 {d0, d1, d2, d3}, [%[src_ptr]]!\n"
            "vld1.32 {d4, d5, d6, d7}, [%[src_ptr]]!\n"

            // Load entries the LHS rank one update vector.
            "vld1.32 {d8, d9, d10, d11}, "
            "[%[rank_one_update_ptr]:256]!\n"
            "vld1.32 {d12, d13, d14, d15}, "
            "[%[rank_one_update_ptr]:256]!\n"

            // Apply the LHS rank one update.
            "vadd.s32 q0, q0, q4\n"
            "vadd.s32 q1, q1, q5\n"
            "vadd.s32 q2, q2, q6\n"
            "vadd.s32 q3, q3, q7\n"

            // Add the constant offset
            // (which includes the RHS rank one update, see above).
            "vadd.s32 q0, q0, q12\n"
            "vadd.s32 q1, q1, q12\n"
            "vadd.s32 q2, q2, q12\n"
            "vadd.s32 q3, q3, q12\n"

            // Multiply by the result multiplier
            "vmul.s32 q0, q0, q15\n"
            "vmul.s32 q1, q1, q15\n"
            "vmul.s32 q2, q2, q15\n"
            "vmul.s32 q3, q3, q15\n"

            // Add the pre-shift offset (so that the shift is rounding)
            "vadd.s32 q0, q0, q13\n"
            "vadd.s32 q1, q1, q13\n"
            "vadd.s32 q2, q2, q13\n"
            "vadd.s32 q3, q3, q13\n"

            // Shift right (shift left by negative offset).
            "vshl.s32 q0, q0, q14\n"
            "vshl.s32 q1, q1, q14\n"
            "vshl.s32 q2, q2, q14\n"
            "vshl.s32 q3, q3, q14\n"

            // So far we had signed 32bit values; now we cast them down
            // to unsigned 8bit, saturating.
            "vqmovn.s32 d8, q0\n"
            "vqmovn.s32 d9, q1\n"
            "vqmovn.s32 d10, q2\n"
            "vqmovn.s32 d11, q3\n"
            "vqmovun.s16 d0, q4\n"
            "vqmovun.s16 d1, q5\n"

            // Store result into the destination matrix.
            "vst1.8 {d0, d1}, [%[dst_ptr]]!\n"

            // End of the loop.
            "cmp %[dst_ptr], %[dst_end_ptr]\n"
            "bne loop_UnpackResultImplNEON_%=\n"

            :  // outputs
            [dst_ptr] "+r"(dst_ptr), [src_ptr] "+r"(src_ptr),
            [rank_one_update_ptr] "+r"(rank_one_update_ptr)
            :  // inputs
            [dst_end_ptr] "r"(dst_end_ptr),
            [constant_offset] "r"(constant_offset),
            [result_mult_int] "r"(result_mult_int),
            [preshift_offset] "r"(preshift_offset),
            [result_shift] "r"(result_shift)
            :  // clobbers
            "cc", "memory", "r3",
            // note: someone on internet says that quad registers are
            // unsupported in the clobber list!
            "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
            "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
            "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
            "d29", "d30", "d31");
      }

      // We have finished handling groups of 16 entries at once; now
      // try to handle 4 entries at once.
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

#endif  // GEMMLOWP_NEON32

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_UNPACK_NEON_H_
