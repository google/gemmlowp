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

// fixedpoint_SSE.h: optimized SSE specializations of the templates
// in fixedpoint.h.

#ifndef GEMMLOWP_INTERNAL_FIXEDPOINT_SSE_H_
#define GEMMLOWP_INTERNAL_FIXEDPOINT_SSE_H_

#include <smmintrin.h>
#include "fixedpoint.h"

namespace gemmlowp {

template <>
struct FixedPointRawTypeTraits<__m128i> {
  typedef std::int32_t ScalarRawType;
  static const int kLanes = 4;
};

template <>
inline __m128i BitAnd(__m128i a, __m128i b) {
  return _mm_and_si128(a, b);
}

template <>
inline __m128i BitOr(__m128i a, __m128i b) {
  return _mm_or_si128(a, b);
}

template <>
inline __m128i BitXor(__m128i a, __m128i b) {
  return _mm_xor_si128(a, b);
}

template <>
inline __m128i BitNot(__m128i a) {
  return _mm_andnot_si128(a, _mm_set1_epi32(-1));
}

template <>
inline __m128i Add(__m128i a, __m128i b) {
  return _mm_add_epi32(a, b);
}

template <>
inline __m128i Mul(__m128i a, __m128i b) {
  return _mm_mullo_epi32(a, b);
}

template <>
inline __m128i Sub(__m128i a, __m128i b) {
  return _mm_sub_epi32(a, b);
}

template <>
inline __m128i Neg(__m128i a) {
  return _mm_sign_epi32(a, _mm_set1_epi32(-1));
}

template <>
inline __m128i ShiftLeft(__m128i a, int offset) {
  return _mm_slli_epi32(a, offset);
}

template <>
inline __m128i ShiftRight(__m128i a, int offset) {
  return _mm_srai_epi32(a, offset);
}

template <>
inline __m128i SelectUsingMask(__m128i if_mask, __m128i then_val,
                               __m128i else_val) {
  return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(else_val),
                                        _mm_castsi128_ps(then_val),
                                        _mm_castsi128_ps(if_mask)));
}

template <>
inline __m128i MaskIfEqual(__m128i a, __m128i b) {
  return _mm_cmpeq_epi32(a, b);
}

template <>
inline __m128i MaskIfNotEqual(__m128i a, __m128i b) {
  return BitNot(MaskIfEqual(a, b));
}

template <>
inline __m128i MaskIfZero(__m128i a) {
  return MaskIfEqual(a, _mm_set1_epi32(0));
}

template <>
inline __m128i MaskIfNonZero(__m128i a) {
  return MaskIfNotEqual(a, _mm_set1_epi32(0));
}

template <>
inline __m128i MaskIfGreaterThan(__m128i a, __m128i b) {
  return _mm_cmpgt_epi32(a, b);
}

template <>
inline __m128i MaskIfLessThan(__m128i a, __m128i b) {
  return _mm_cmplt_epi32(a, b);
}

template <>
inline __m128i MaskIfGreaterThanOrEqual(__m128i a, __m128i b) {
  return BitNot(MaskIfLessThan(a, b));
}

template <>
inline __m128i MaskIfLessThanOrEqual(__m128i a, __m128i b) {
  return BitNot(MaskIfGreaterThan(a, b));
}

/* Assumptions:
   - All and Any are used on masks.
   - masks are all_ones for true lanes, all_zeroes otherwise.
Hence, All means all 128bits set, and Any means any bit set.
*/

template <>
inline bool All(__m128i a) {
  return _mm_testc_si128(a, a);
}

template <>
inline bool Any(__m128i a) {
  return BitNot(_mm_testz_si128(a, a));
}

template <>
inline __m128i RoundingHalfSum(__m128i a, __m128i b) {
  /* __m128i round_bit_mask, a_over_2, b_over_2, round_bit, sum; */
  /* We divide the inputs before the add to avoid the overflow and costly test
   */
  /* of checking if an overflow occured on signed add */
  /* round_bit_mask = _mm_set1_epi32(1); */
  /* a_over_2 = _mm_srai_epi32(a, 1); */
  /* b_over_2 = _mm_srai_epi32(b, 1); */
  /* sum = Add(a_over_2, b_over_2); */
  /* round_bit = _mm_sign_epi32(BitAnd(BitOr(a,b), round_bit_mask), sum); */
  /* return Add(sum, round_bit); */

  /* Other possibility detecting overflow and xor the sign if an overflow
   * happened*/
  __m128i one, sign_bit_mask, sum, rounded_half_sum, overflow, result;
  one = _mm_set1_epi32(1);
  sign_bit_mask = _mm_set1_epi32(0x80000000);
  sum = Add(a, b);
  rounded_half_sum = _mm_srai_epi32(Add(sum, one), 1);
  overflow =
      BitAnd(BitAnd(BitXor(a, rounded_half_sum), BitXor(b, rounded_half_sum)),
             sign_bit_mask);
  result = BitXor(rounded_half_sum, overflow);
  return result;
}

template <>
inline __m128i SaturatingRoundingDoublingHighMul(__m128i a, __m128i b) {
  __m128i min, saturation_mask, a0_a2, a1_a3, b0_b2, b1_b3;
  __m128i a0b0_a2b2, a1b1_a3b3, a0b0_a2b2_rounded, a1b1_a3b3_rounded;
  __m128i a0b0_a2b2_rounded_2x, a1b1_a3b3_rounded_2x, result;
  __m128i nudge;

  // saturation only happen if a == b == INT_MIN
  min = _mm_set1_epi32(std::numeric_limits<std::int32_t>::min());
  saturation_mask = BitAnd(MaskIfEqual(a, b), MaskIfEqual(a, min));

  // a = a0 | a1 | a2 | a3
  // b = b0 | b1 | b2 | b3
  a0_a2 = a;
  a1_a3 = _mm_srli_si128(a, 4);
  b0_b2 = b;
  b1_b3 = _mm_srli_si128(b, 4);

  a0b0_a2b2 = _mm_mul_epi32(a0_a2, b0_b2);
  a1b1_a3b3 = _mm_mul_epi32(a1_a3, b1_b3);

  // do the rounding and take into account that it will be doubled
  nudge = _mm_set1_epi64x(1 << 30);
  a0b0_a2b2_rounded = _mm_add_epi64(a0b0_a2b2, nudge);
  a1b1_a3b3_rounded = _mm_add_epi64(a1b1_a3b3, nudge);

  // do the doubling
  a0b0_a2b2_rounded_2x = _mm_slli_epi64(a0b0_a2b2_rounded, 1);
  a1b1_a3b3_rounded_2x = _mm_slli_epi64(a1b1_a3b3_rounded, 1);

  // get the high part of the products
  result = _mm_blend_epi16(_mm_srli_si128(a0b0_a2b2_rounded_2x, 4),
                           a1b1_a3b3_rounded_2x, 0xcc);

  // saturate those which overflowed
  return SelectUsingMask(saturation_mask, min, result);
}

template <>
inline __m128i Dup<__m128i>(std::int32_t x) {
  return _mm_set1_epi32(x);
}

}  // end namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_FIXEDPOINT_SSE_H_
