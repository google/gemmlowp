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

// output_neon.h: optimized NEON specializations of the templates in output.h.

#ifndef GEMMLOWP_INTERNAL_OUTPUT_NEON_H_
#define GEMMLOWP_INTERNAL_OUTPUT_NEON_H_

#include "output.h"

#include <arm_neon.h>

namespace gemmlowp {

// The code in unpack_neon.h will whenever possible process
// 4 NEON SIMD vectors (4x 128 bits) at once,
// to offer the compiler better optimization opportunities, reducing
// register dependencies. From the perspective of interfacing with the output
// pipeline, this takes the form of passing int32x4x4_t data. In most cases,
// such data is handled simply by handling separately its 4 int32x4_t
// components.
// This partial specialization handles that for arbitrary output stages
// implementing
// a int32x4_t path. Only some output stages below will override this to
// use custom code to handle int32x4x4_t data all at once (see
// OutputStageSaturatingCastToUint8 below).
template <typename OutputStageType>
struct EvalOutputStageImpl<OutputStageType, int32x4x4_t> {
  typedef int32x4x4_t InputType;
  typedef int32x4x4_t OutputType;

  static OutputType Eval(const OutputStageType& output_stage, InputType input) {
    OutputType retval;
    typedef EvalOutputStageImpl<OutputStageType, int32x4_t> ImplInt32x4;
    for (int i = 0; i < 4; i++) {
      retval.val[i] = ImplInt32x4::Eval(output_stage, input.val[i]);
    }
    return retval;
  }
};

// Implementation of OutputStageQuantizeDownInt32ToUint8Scale for NEON int32x4_t
template <>
struct EvalOutputStageImpl<OutputStageQuantizeDownInt32ToUint8Scale,
                           int32x4_t> {
  typedef int32x4_t InputType;
  typedef int32x4_t OutputType;

  static OutputType Eval(
      const OutputStageQuantizeDownInt32ToUint8Scale& output_stage,
      InputType input) {
    const std::int32_t result_shift = output_stage.result_shift;
    const std::int32_t result_mult_int = output_stage.result_mult_int;
    const std::int32_t result_offset = output_stage.result_offset;
    const std::int32_t preshift_offset =
        (result_shift < 1) ? 0 : (1 << (result_shift - 1));
    const int32x4_t a = vaddq_s32(input, vdupq_n_s32(result_offset));
    const int32x4_t b =
        vmlaq_n_s32(vdupq_n_s32(preshift_offset), a, result_mult_int);
    return vshlq_s32(b, vdupq_n_s32(-result_shift));
  }
};

// Dummy data type to represent a NEON SIMD vector of 4 uint8 values.
// Useful as the result of 32bit -> 8bit casts on a single int32x4_t input.
struct pseudo_uint8x4_t {
  uint8x8_t val;
};

// Implementation of OutputStageSaturatingCastToUint8 for NEON int32x4_t
template <>
struct EvalOutputStageImpl<OutputStageSaturatingCastToUint8, int32x4_t> {
  typedef int32x4_t InputType;
  typedef pseudo_uint8x4_t OutputType;

  static OutputType Eval(const OutputStageSaturatingCastToUint8& output_stage,
                         InputType input) {
    int16x8_t q16 = vcombine_s16(vqmovn_s32(input), vdup_n_s16(0));
    pseudo_uint8x4_t retval;
    retval.val = vqmovun_s16(q16);
    return retval;
  }
};

// In the case of OutputStageSaturatingCastToUint8, the handling of int32x4x4_t
// data can be made much more efficient by handling it all at once, instead of
// as 4 separate int32x4 values as in the above generic partial specialization.
// This also avoids the poor (50%) register utilization of pseudo_uint8x4_t:
// by handling 16 scalar values at once, we are able to fill a uint8x16_t.
template <>
struct EvalOutputStageImpl<OutputStageSaturatingCastToUint8, int32x4x4_t> {
  typedef int32x4x4_t InputType;
  typedef uint8x16_t OutputType;

  static OutputType Eval(const OutputStageSaturatingCastToUint8& output_stage,
                         InputType input) {
    int16x8_t q16[2];
    for (int i = 0; i < 2; i++) {
      q16[i] = vcombine_s16(vqmovn_s32(input.val[2 * i]),
                            vqmovn_s32(input.val[2 * i + 1]));
    }
    return vcombine_u8(vqmovun_s16(q16[0]), vqmovun_s16(q16[1]));
  }
};

// Specialization of StoreUnpackOutput for pseudo_uint8x4_t.
// This is quite inefficient, but we have no choice: instructions storing 32bit
// at once
// also assume 32bit alignment. In practice, this slowness is not a problem
// because
// we use the int32x4x4_t -> uint8x16_t path for most values.
template <>
inline std::size_t StoreUnpackOutput(pseudo_uint8x4_t value,
                                     std::uint8_t* dst) {
  vst1_lane_u8(dst++, value.val, 0);
  vst1_lane_u8(dst++, value.val, 1);
  vst1_lane_u8(dst++, value.val, 2);
  vst1_lane_u8(dst++, value.val, 3);
  return 4;
}

// Specialization of StoreUnpackOutput for uint8x16_t.
template <>
inline std::size_t StoreUnpackOutput(uint8x16_t value, std::uint8_t* dst) {
  vst1q_u8(dst, value);
  return 16;
}

// Specialization of StoreUnpackOutput for int32x4_t, storing into a
// int32 destination.
template <>
inline std::size_t StoreUnpackOutput(int32x4_t value, std::int32_t* dst) {
  vst1q_s32(dst, value);
  return 4;
}

// Specialization of StoreUnpackOutput for int32x4x4_t, storing into a
// int32 destination.
template <>
inline std::size_t StoreUnpackOutput(int32x4x4_t value, std::int32_t* dst) {
  vst1q_s32(dst + 0, value.val[0]);
  vst1q_s32(dst + 4, value.val[1]);
  vst1q_s32(dst + 8, value.val[2]);
  vst1q_s32(dst + 12, value.val[3]);
  return 16;
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_OUTPUT_NEON_H_
