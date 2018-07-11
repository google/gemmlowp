// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
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

#ifndef GEMMLOWP_META_TRANSFORM_KERNELS_ARM_32_H_
#define GEMMLOWP_META_TRANSFORM_KERNELS_ARM_32_H_

#ifdef GEMMLOWP_NEON_32

#include <cassert>
#include <cstdint>

namespace gemmlowp {
namespace meta {

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 0>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 0>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #0\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"



      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"



      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"



    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 1>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 1>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #1\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0[0]}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vcvt.s32.f32 q8, q8\n"

      "vqmovn.s32 d16, q8\n"

      "vqmovun.s16 d16, q8\n"

      "vst1.8 {d16[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0[0]}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vcvt.s32.f32 q4, q4\n"

      "vqmovn.s32 d8, q4\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.8 {d8[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0[0]}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vfma.f32 q4, q0, q12\n"

      "vcvt.s32.f32 q4, q4\n"

      "vqmovn.s32 d8, q4\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.8 {d8[0]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 2>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 2>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #2\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vcvt.s32.f32 q8, q8\n"

      "vqmovn.s32 d16, q8\n"

      "vqmovun.s16 d16, q8\n"

      "vst1.16 {d16[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vcvt.s32.f32 q4, q4\n"

      "vqmovn.s32 d8, q4\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.16 {d8[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vfma.f32 q4, q0, q12\n"

      "vcvt.s32.f32 q4, q4\n"

      "vqmovn.s32 d8, q4\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.16 {d8[0]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 3>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 3>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #3\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0}, [%[input]]!\n"
      "vld1.32 {d1[0]}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vcvt.s32.f32 q8, q8\n"

      "vqmovn.s32 d16, q8\n"

      "vqmovun.s16 d16, q8\n"

      "vst1.16 {d16[0]}, [%[output]]!\n"
      "vst1.8 {d16[2]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0}, [%[input]]!\n"
      "vld1.32 {d1[0]}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vcvt.s32.f32 q4, q4\n"

      "vqmovn.s32 d8, q4\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.16 {d8[0]}, [%[output]]!\n"
      "vst1.8 {d8[2]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0}, [%[input]]!\n"
      "vld1.32 {d1[0]}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vfma.f32 q4, q0, q12\n"

      "vcvt.s32.f32 q4, q4\n"

      "vqmovn.s32 d8, q4\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.16 {d8[0]}, [%[output]]!\n"
      "vst1.8 {d8[2]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 4>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 4>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #4\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vcvt.s32.f32 q8, q8\n"

      "vqmovn.s32 d16, q8\n"

      "vqmovun.s16 d16, q8\n"

      "vst1.32 {d16[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vcvt.s32.f32 q4, q4\n"

      "vqmovn.s32 d8, q4\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.32 {d8[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vfma.f32 q4, q0, q12\n"

      "vcvt.s32.f32 q4, q4\n"

      "vqmovn.s32 d8, q4\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.32 {d8[0]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 5>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 5>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #5\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2[0]}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovun.s16 d16, q8\n"

      "vst1.32 {d16[0]}, [%[output]]!\n"
      "vst1.8 {d16[4]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2[0]}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.32 {d8[0]}, [%[output]]!\n"
      "vst1.8 {d8[4]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vmov.f32 q5, q13\n"
      "vld1.32 {d2[0]}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vcvt.f32.s32 q1, q1\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.32 {d8[0]}, [%[output]]!\n"
      "vst1.8 {d8[4]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 6>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 6>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #6\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovun.s16 d16, q8\n"

      "vst1.32 {d16[0]}, [%[output]]!\n"
      "vst1.16 {d16[2]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.32 {d8[0]}, [%[output]]!\n"
      "vst1.16 {d8[2]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vmov.f32 q5, q13\n"
      "vld1.32 {d2}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vcvt.f32.s32 q1, q1\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.32 {d8[0]}, [%[output]]!\n"
      "vst1.16 {d8[2]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 7>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 7>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #7\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2}, [%[input]]!\n"
      "vld1.32 {d3[0]}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovun.s16 d16, q8\n"

      "vst1.32 {d16[0]}, [%[output]]!\n"
      "vst1.16 {d16[2]}, [%[output]]!\n"
      "vst1.8 {d16[6]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2}, [%[input]]!\n"
      "vld1.32 {d3[0]}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.32 {d8[0]}, [%[output]]!\n"
      "vst1.16 {d8[2]}, [%[output]]!\n"
      "vst1.8 {d8[6]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vmov.f32 q5, q13\n"
      "vld1.32 {d2}, [%[input]]!\n"
      "vld1.32 {d3[0]}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vcvt.f32.s32 q1, q1\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.32 {d8[0]}, [%[output]]!\n"
      "vst1.16 {d8[2]}, [%[output]]!\n"
      "vst1.8 {d8[6]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 8>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 8>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #8\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovun.s16 d16, q8\n"

      "vst1.32 {d16}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.32 {d8}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vcvt.f32.s32 q1, q1\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovun.s16 d8, q4\n"

      "vst1.32 {d8}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 9>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 9>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #9\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4[0]}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovn.s32 d20, q10\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16}, [%[output]]!\n"
      "vst1.8 {d17[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4[0]}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.8 {d9[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vmov.f32 q6, q13\n"
      "vld1.32 {d4[0]}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vcvt.f32.s32 q1, q1\n"

      "vcvt.f32.s32 q2, q2\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.8 {d9[0]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 10>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 10>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #10\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovn.s32 d20, q10\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16}, [%[output]]!\n"
      "vst1.16 {d17[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.16 {d9[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vmov.f32 q6, q13\n"
      "vld1.32 {d4}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vcvt.f32.s32 q1, q1\n"

      "vcvt.f32.s32 q2, q2\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.16 {d9[0]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 11>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 11>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #11\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4}, [%[input]]!\n"
      "vld1.32 {d5[0]}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovn.s32 d20, q10\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16}, [%[output]]!\n"
      "vst1.16 {d17[0]}, [%[output]]!\n"
      "vst1.8 {d17[2]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4}, [%[input]]!\n"
      "vld1.32 {d5[0]}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.16 {d9[0]}, [%[output]]!\n"
      "vst1.8 {d9[2]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vmov.f32 q6, q13\n"
      "vld1.32 {d4}, [%[input]]!\n"
      "vld1.32 {d5[0]}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vcvt.f32.s32 q1, q1\n"

      "vcvt.f32.s32 q2, q2\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.16 {d9[0]}, [%[output]]!\n"
      "vst1.8 {d9[2]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 12>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 12>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #12\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovn.s32 d20, q10\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16}, [%[output]]!\n"
      "vst1.32 {d17[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.32 {d9[0]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vcvt.f32.s32 q1, q1\n"

      "vcvt.f32.s32 q2, q2\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.32 {d9[0]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 13>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 13>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #13\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6[0]}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16}, [%[output]]!\n"
      "vst1.32 {d17[0]}, [%[output]]!\n"
      "vst1.8 {d17[4]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6[0]}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.32 {d9[0]}, [%[output]]!\n"
      "vst1.8 {d9[4]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vmov.f32 q7, q13\n"
      "vld1.32 {d6[0]}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vcvt.f32.s32 q1, q1\n"

      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.32 {d9[0]}, [%[output]]!\n"
      "vst1.8 {d9[4]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 14>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 14>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #14\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16}, [%[output]]!\n"
      "vst1.32 {d17[0]}, [%[output]]!\n"
      "vst1.16 {d17[2]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.32 {d9[0]}, [%[output]]!\n"
      "vst1.16 {d9[2]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vmov.f32 q7, q13\n"
      "vld1.32 {d6}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vcvt.f32.s32 q1, q1\n"

      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.32 {d9[0]}, [%[output]]!\n"
      "vst1.16 {d9[2]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<int32_t, uint8_t, Requantize, 16, 15>::Transform(const int32_t* input, const Requantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Requantize<int32_t, uint8_t, Requantize, 16, 15>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  const float coefficient = params.one_over_output_range_scale * params.input_range_scale;
  const float offset = params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     );
  asm volatile(
    // Requantize::Prepare
    "vdup.32 q12, %[coefficient]\n"
    "vdup.32 q13, %[offset]\n"

    // Reduce count by leftovers
    "subs %[count], %[count], #15\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %[count], %[count], #16\n"
    "vmov.f32 q4, q13\n"
    "vld1.32 {d0, d1}, [%[input]]!\n"

    "vmov.f32 q5, q13\n"
    "vld1.32 {d2, d3}, [%[input]]!\n"

    "vmov.f32 q6, q13\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"

    "vmov.f32 q7, q13\n"
    "vld1.32 {d6, d7}, [%[input]]!\n"


    "vcvt.f32.s32 q0, q0\n"

    "vcvt.f32.s32 q1, q1\n"

    "vcvt.f32.s32 q2, q2\n"

    "vcvt.f32.s32 q3, q3\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %[count], %[count], #16\n"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %[count], %[count], #16\n"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6, d7}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "vfma.f32 q4, q0, q12\n"
      "vmov.f32 q8, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q5, q1, q12\n"
      "vmov.f32 q9, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q6, q2, q12\n"
      "vmov.f32 q10, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q7, q3, q12\n"
      "vmov.f32 q11, q13\n"
      "vld1.32 {d6}, [%[input]]!\n"
      "vld1.32 {d7[0]}, [%[input]]!\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d8, d9}, [%[output]]!\n"

      "vfma.f32 q8, q0, q12\n"

      "vfma.f32 q9, q1, q12\n"

      "vfma.f32 q10, q2, q12\n"

      "vfma.f32 q11, q3, q12\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"

      "vst1.32 {d16}, [%[output]]!\n"
      "vst1.32 {d17[0]}, [%[output]]!\n"
      "vst1.16 {d17[2]}, [%[output]]!\n"
      "vst1.8 {d17[6]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "vfma.f32 q8, q0, q12\n"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vfma.f32 q9, q1, q12\n"
      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vfma.f32 q10, q2, q12\n"
      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vfma.f32 q11, q3, q12\n"
      "vmov.f32 q7, q13\n"
      "vld1.32 {d6}, [%[input]]!\n"
      "vld1.32 {d7[0]}, [%[input]]!\n"

      "vcvt.s32.f32 q8, q8\n"
      "vcvt.s32.f32 q9, q9\n"
      "vcvt.s32.f32 q10, q10\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q11, q11\n"

      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vcvt.f32.s32 q0, q0\n"

      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vcvt.f32.s32 q1, q1\n"

      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d17, q10\n"
      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vst1.32 {d16, d17}, [%[output]]!\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.32 {d9[0]}, [%[output]]!\n"
      "vst1.16 {d9[2]}, [%[output]]!\n"
      "vst1.8 {d9[6]}, [%[output]]!\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "vmov.f32 q4, q13\n"
      "vld1.32 {d0, d1}, [%[input]]!\n"

      "vmov.f32 q5, q13\n"
      "vld1.32 {d2, d3}, [%[input]]!\n"

      "vmov.f32 q6, q13\n"
      "vld1.32 {d4, d5}, [%[input]]!\n"

      "vmov.f32 q7, q13\n"
      "vld1.32 {d6}, [%[input]]!\n"
      "vld1.32 {d7[0]}, [%[input]]!\n"


      "vcvt.f32.s32 q0, q0\n"

      "vcvt.f32.s32 q1, q1\n"

      "vcvt.f32.s32 q2, q2\n"

      "vcvt.f32.s32 q3, q3\n"

      "vfma.f32 q4, q0, q12\n"

      "vfma.f32 q5, q1, q12\n"

      "vfma.f32 q6, q2, q12\n"

      "vfma.f32 q7, q3, q12\n"

      "vcvt.s32.f32 q4, q4\n"
      "vcvt.s32.f32 q5, q5\n"
      "vcvt.s32.f32 q6, q6\n"
      "pld [%[input], #64]\n"
      "vcvt.s32.f32 q7, q7\n"

      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"

      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"

      "vqmovun.s16 d8, q4\n"
      "vqmovun.s16 d9, q6\n"

      "vst1.32 {d8}, [%[output]]!\n"
      "vst1.32 {d9[0]}, [%[output]]!\n"
      "vst1.16 {d9[2]}, [%[output]]!\n"
      "vst1.8 {d9[6]}, [%[output]]!\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 0>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 0>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 1>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 1>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #1\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovun.s16 d0, q0\n"

    "vst1.8 {d0[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 2>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 2>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #2\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovun.s16 d0, q0\n"

    "vst1.16 {d0[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 3>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 3>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #3\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovun.s16 d0, q0\n"

    "vst1.16 {d0[0]}, [%[output]]!\n"
    "vst1.8 {d0[2]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 4>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 4>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #4\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovun.s16 d0, q0\n"

    "vst1.32 {d0[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 5>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 5>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #5\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d2[0]}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovun.s16 d0, q0\n"

    "vst1.32 {d0[0]}, [%[output]]!\n"
    "vst1.8 {d0[4]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 6>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 6>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #6\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1, d2}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovun.s16 d0, q0\n"

    "vst1.32 {d0[0]}, [%[output]]!\n"
    "vst1.16 {d0[2]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 7>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 7>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #7\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1, d2}, [%[input]]!\n"
    "vld1.32 {d3[0]}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovun.s16 d0, q0\n"

    "vst1.32 {d0[0]}, [%[output]]!\n"
    "vst1.16 {d0[2]}, [%[output]]!\n"
    "vst1.8 {d0[6]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 8>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 8>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #8\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovun.s16 d0, q0\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 9>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 9>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #9\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4[0]}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.8 {d1[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 10>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 10>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #10\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.16 {d1[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 11>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 11>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #11\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4}, [%[input]]!\n"
    "vld1.32 {d5[0]}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.16 {d1[0]}, [%[output]]!\n"
    "vst1.8 {d1[2]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 12>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 12>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #12\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.32 {d1[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 13>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 13>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #13\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5}, [%[input]]!\n"
    "vld1.32 {d6[0]}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.32 {d1[0]}, [%[output]]!\n"
    "vst1.8 {d1[4]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 14>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 14>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #14\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.32 {d1[0]}, [%[output]]!\n"
    "vst1.16 {d1[2]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<float, uint8_t, Quantize, 16, 15>::Transform(const float* input, const Quantize& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Quantize<float, uint8_t, Quantize, 16, 15>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Quantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #15\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6, d7}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "vld1.32 {d0, d1, d2, d3}, [%[input]]!\n"
    "vld1.32 {d4, d5, d6}, [%[input]]!\n"
    "vld1.32 {d7[0]}, [%[input]]!\n"
    "pld [%[input], #64]\n"
    "vsub.f32 q0, q0, q4\n"
    "vsub.f32 q1, q1, q4\n"
    "vsub.f32 q2, q2, q4\n"
    "vsub.f32 q3, q3, q4\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q5\n"
    "vadd.f32 q1, q1, q5\n"
    "vadd.f32 q2, q2, q5\n"
    "vadd.f32 q3, q3, q5\n"
    "vcvt.s32.f32 q0, q0\n"
    "vcvt.s32.f32 q1, q1\n"
    "vcvt.s32.f32 q2, q2\n"
    "vcvt.s32.f32 q3, q3\n"
    "vqmovn.s32 d0, q0\n"
    "vqmovn.s32 d1, q1\n"
    "vqmovn.s32 d4, q2\n"
    "vqmovn.s32 d5, q3\n"
    "vqmovun.s16 d0, q0\n"
    "vqmovun.s16 d1, q2\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.32 {d1[0]}, [%[output]]!\n"
    "vst1.16 {d1[2]}, [%[output]]!\n"
    "vst1.8 {d1[6]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 0>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 0>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 1>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 1>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #1\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.8 {d0[0]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vsub.f32 q0, q0, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vadd.f32 q0, q0, q4\n"

    "vst1.32 {d0[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 2>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 2>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #2\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.16 {d0[0]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vsub.f32 q0, q0, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vadd.f32 q0, q0, q4\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 3>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 3>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #3\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.16 {d0[0]}, [%[input]]!\n"
    "vld1.8 {d0[2]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vsub.f32 q0, q0, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vadd.f32 q0, q0, q4\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.32 {d1[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 4>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 4>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #4\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vsub.f32 q0, q0, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vadd.f32 q0, q0, q4\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 5>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 5>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #5\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "vld1.8 {d0[4]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "vst1.32 {d2[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 6>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 6>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #6\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "vld1.16 {d0[2]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"

    "vst1.32 {d0, d1, d2}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 7>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 7>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #7\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "vld1.16 {d0[2]}, [%[input]]!\n"
    "vld1.8 {d0[6]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"

    "vst1.32 {d0, d1, d2}, [%[output]]!\n"
    "vst1.32 {d3[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 8>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 8>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #8\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 9>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 9>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #9\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.8 {d1[0]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 10>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 10>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #10\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.16 {d1[0]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 11>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 11>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #11\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.16 {d1[0]}, [%[input]]!\n"
    "vld1.8 {d1[2]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4}, [%[output]]!\n"
    "vst1.32 {d5[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 12>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 12>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #12\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 13>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 13>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #13\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "vld1.8 {d1[4]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5}, [%[output]]!\n"
    "vst1.32 {d6[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 14>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 14>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #14\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "vld1.16 {d1[2]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, float, Dequantize, 16, 15>::Transform(const uint8_t* input, const Dequantize& params, float* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") Dequantize<uint8_t, float, Dequantize, 16, 15>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // Dequantize::Prepare
    "vdup.32 q4, %[range_min]\n"
    "vdup.32 q5, %[range_offset]\n"
    "vdup.32 q6, %[range_scale]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #15\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // Dequantize::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6, d7}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "vld1.16 {d1[2]}, [%[input]]!\n"
    "vld1.8 {d1[6]}, [%[input]]!\n"
    "pld [%[input], #32]\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vsub.f32 q0, q0, q5\n"
    "vsub.f32 q1, q1, q5\n"
    "vsub.f32 q2, q2, q5\n"
    "vsub.f32 q3, q3, q5\n"
    "vmul.f32 q0, q0, q6\n"
    "vmul.f32 q1, q1, q6\n"
    "vmul.f32 q2, q2, q6\n"
    "vmul.f32 q3, q3, q6\n"
    "vadd.f32 q0, q0, q4\n"
    "vadd.f32 q1, q1, q4\n"
    "vadd.f32 q2, q2, q4\n"
    "vadd.f32 q3, q3, q4\n"

    "vst1.32 {d0, d1, d2, d3}, [%[output]]!\n"
    "vst1.32 {d4, d5, d6}, [%[output]]!\n"
    "vst1.32 {d7[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 0>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 0>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 1>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 1>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #1\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.8 {d0[0]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.8 {d0[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 2>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 2>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #2\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.16 {d0[0]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.16 {d0[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 3>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 3>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #3\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.16 {d0[0]}, [%[input]]!\n"
    "vld1.8 {d0[2]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.16 {d0[0]}, [%[output]]!\n"
    "vst1.8 {d0[2]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 4>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 4>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #4\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 5>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 5>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #5\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "vld1.8 {d0[4]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0[0]}, [%[output]]!\n"
    "vst1.8 {d0[4]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 6>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 6>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #6\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "vld1.16 {d0[2]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0[0]}, [%[output]]!\n"
    "vst1.16 {d0[2]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 7>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 7>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #7\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "vld1.16 {d0[2]}, [%[input]]!\n"
    "vld1.8 {d0[6]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0[0]}, [%[output]]!\n"
    "vst1.16 {d0[2]}, [%[output]]!\n"
    "vst1.8 {d0[6]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 8>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 8>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #8\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 9>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 9>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #9\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.8 {d1[0]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.8 {d1[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 10>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 10>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #10\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.16 {d1[0]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.16 {d1[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 11>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 11>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #11\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.16 {d1[0]}, [%[input]]!\n"
    "vld1.8 {d1[2]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.16 {d1[0]}, [%[output]]!\n"
    "vst1.8 {d1[2]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 12>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 12>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #12\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.32 {d1[0]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 13>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 13>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #13\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "vld1.8 {d1[4]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.32 {d1[0]}, [%[output]]!\n"
    "vst1.8 {d1[4]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 14>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 14>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #14\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "vld1.16 {d1[2]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.32 {d1[0]}, [%[output]]!\n"
    "vst1.16 {d1[2]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, uint8_t, MinMax<uint8_t>, 16, 15>::Transform(const uint8_t* input, const MinMax<uint8_t>& params, uint8_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") MinMax<uint8_t><uint8_t, uint8_t, MinMax<uint8_t>, 16, 15>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(

    // MinMax::Prepare
    "vdup.8 q4, %[min]\n"
    "vdup.8 q5, %[max]\n"

    // Reduce count by leftovers.
    "subs %[count], %[count], #15\n"
    "beq 2f\n"

    "1:"
    "subs %[count], %[count], #16\n"

    // MinMax::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0, d1}, [%[output]]!\n"
    "pld [%[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "vld1.16 {d1[2]}, [%[input]]!\n"
    "vld1.8 {d1[6]}, [%[input]]!\n"
    "pld [%[input], #16]\n"
    "vmax.u8 q0, q0, q4\n"
    "vmin.u8 q0, q0, q5\n"

    "vst1.32 {d0}, [%[output]]!\n"
    "vst1.32 {d1[0]}, [%[output]]!\n"
    "vst1.16 {d1[2]}, [%[output]]!\n"
    "vst1.8 {d1[6]}, [%[output]]!\n"
    "pld [%[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "d0", "d1", "d8", "d9", "d10", "d11", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 0>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 0>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 1>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 1>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #1\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.8 {d0[0]}, [%[input]]!\n"
    "vld1.8 {d2[0]}, [r1]!\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q2, q12\n"
    "vmovl.u8 q1, d2\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q1, d2\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vfma.f32 q2, q0, q13\n"
    "vfma.f32 q2, q1, q14\n"
    "vcvt.s32.f32 q2, q2\n"

    "vst1.32 {d4[0]}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 2>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 2>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #2\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.16 {d0[0]}, [%[input]]!\n"
    "vld1.16 {d2[0]}, [r1]!\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q2, q12\n"
    "vmovl.u8 q1, d2\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q1, d2\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vfma.f32 q2, q0, q13\n"
    "vfma.f32 q2, q1, q14\n"
    "vcvt.s32.f32 q2, q2\n"

    "vst1.32 {d4}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 3>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 3>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #3\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.16 {d0[0]}, [%[input]]!\n"
    "vld1.8 {d0[2]}, [%[input]]!\n"
    "vld1.16 {d2[0]}, [r1]!\n"
    "vld1.8 {d2[2]}, [r1]!\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q2, q12\n"
    "vmovl.u8 q1, d2\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q1, d2\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vfma.f32 q2, q0, q13\n"
    "vfma.f32 q2, q1, q14\n"
    "vcvt.s32.f32 q2, q2\n"

    "vst1.32 {d4}, [%[output]]!\n"
    "vst1.32 {d5[0]}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 4>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 4>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #4\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "vld1.32 {d2[0]}, [r1]!\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q2, q12\n"
    "vmovl.u8 q1, d2\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q1, d2\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vfma.f32 q2, q0, q13\n"
    "vfma.f32 q2, q1, q14\n"
    "vcvt.s32.f32 q2, q2\n"

    "vst1.32 {d4, d5}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 5>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 5>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #5\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "vld1.8 {d0[4]}, [%[input]]!\n"
    "vld1.32 {d4[0]}, [r1]!\n"
    "vld1.8 {d4[4]}, [r1]!\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q4, q12\n"
    "vmovl.u8 q2, d4\n"
    "vmov.f32 q5, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q3, d5\n"
    "vmovl.s16 q2, d4\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vfma.f32 q4, q0, q13\n"
    "vfma.f32 q5, q1, q13\n"
    "vfma.f32 q4, q2, q14\n"
    "vfma.f32 q5, q3, q14\n"
    "vcvt.s32.f32 q4, q4\n"
    "vcvt.s32.f32 q5, q5\n"

    "vst1.32 {d8, d9}, [%[output]]!\n"
    "vst1.32 {d10[0]}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 6>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 6>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #6\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "vld1.16 {d0[2]}, [%[input]]!\n"
    "vld1.32 {d4[0]}, [r1]!\n"
    "vld1.16 {d4[2]}, [r1]!\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q4, q12\n"
    "vmovl.u8 q2, d4\n"
    "vmov.f32 q5, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q3, d5\n"
    "vmovl.s16 q2, d4\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vfma.f32 q4, q0, q13\n"
    "vfma.f32 q5, q1, q13\n"
    "vfma.f32 q4, q2, q14\n"
    "vfma.f32 q5, q3, q14\n"
    "vcvt.s32.f32 q4, q4\n"
    "vcvt.s32.f32 q5, q5\n"

    "vst1.32 {d8, d9, d10}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 7>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 7>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #7\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0[0]}, [%[input]]!\n"
    "vld1.16 {d0[2]}, [%[input]]!\n"
    "vld1.8 {d0[6]}, [%[input]]!\n"
    "vld1.32 {d4[0]}, [r1]!\n"
    "vld1.16 {d4[2]}, [r1]!\n"
    "vld1.8 {d4[6]}, [r1]!\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q4, q12\n"
    "vmovl.u8 q2, d4\n"
    "vmov.f32 q5, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q3, d5\n"
    "vmovl.s16 q2, d4\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vfma.f32 q4, q0, q13\n"
    "vfma.f32 q5, q1, q13\n"
    "vfma.f32 q4, q2, q14\n"
    "vfma.f32 q5, q3, q14\n"
    "vcvt.s32.f32 q4, q4\n"
    "vcvt.s32.f32 q5, q5\n"

    "vst1.32 {d8, d9, d10}, [%[output]]!\n"
    "vst1.32 {d11[0]}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 8>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 8>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #8\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d4}, [r1]!\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q4, q12\n"
    "vmovl.u8 q2, d4\n"
    "vmov.f32 q5, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q3, d5\n"
    "vmovl.s16 q2, d4\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vfma.f32 q4, q0, q13\n"
    "vfma.f32 q5, q1, q13\n"
    "vfma.f32 q4, q2, q14\n"
    "vfma.f32 q5, q3, q14\n"
    "vcvt.s32.f32 q4, q4\n"
    "vcvt.s32.f32 q5, q5\n"

    "vst1.32 {d8, d9, d10, d11}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 9>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 9>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #9\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.8 {d1[0]}, [%[input]]!\n"
    "vld1.32 {d6}, [r1]!\n"
    "vld1.8 {d7[0]}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q6, q12\n"
    "vmovl.u8 q4, d7\n"
    "vmovl.u8 q3, d6\n"
    "vmov.f32 q7, q12\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q8, q12\n"
    "vmovl.s16 q5, d8\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q4, d7\n"
    "vmovl.s16 q3, d6\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vfma.f32 q6, q0, q13\n"
    "vfma.f32 q7, q1, q13\n"
    "vfma.f32 q8, q2, q13\n"
    "vfma.f32 q6, q3, q14\n"
    "vfma.f32 q7, q4, q14\n"
    "vfma.f32 q8, q5, q14\n"
    "vcvt.s32.f32 q6, q6\n"
    "vcvt.s32.f32 q7, q7\n"
    "vcvt.s32.f32 q8, q8\n"

    "vst1.32 {d12, d13, d14, d15}, [%[output]]!\n"
    "vst1.32 {d16[0]}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 10>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 10>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #10\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.16 {d1[0]}, [%[input]]!\n"
    "vld1.32 {d6}, [r1]!\n"
    "vld1.16 {d7[0]}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q6, q12\n"
    "vmovl.u8 q4, d7\n"
    "vmovl.u8 q3, d6\n"
    "vmov.f32 q7, q12\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q8, q12\n"
    "vmovl.s16 q5, d8\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q4, d7\n"
    "vmovl.s16 q3, d6\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vfma.f32 q6, q0, q13\n"
    "vfma.f32 q7, q1, q13\n"
    "vfma.f32 q8, q2, q13\n"
    "vfma.f32 q6, q3, q14\n"
    "vfma.f32 q7, q4, q14\n"
    "vfma.f32 q8, q5, q14\n"
    "vcvt.s32.f32 q6, q6\n"
    "vcvt.s32.f32 q7, q7\n"
    "vcvt.s32.f32 q8, q8\n"

    "vst1.32 {d12, d13, d14, d15}, [%[output]]!\n"
    "vst1.32 {d16}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 11>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 11>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #11\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.16 {d1[0]}, [%[input]]!\n"
    "vld1.8 {d1[2]}, [%[input]]!\n"
    "vld1.32 {d6}, [r1]!\n"
    "vld1.16 {d7[0]}, [r1]!\n"
    "vld1.8 {d7[2]}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q6, q12\n"
    "vmovl.u8 q4, d7\n"
    "vmovl.u8 q3, d6\n"
    "vmov.f32 q7, q12\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q8, q12\n"
    "vmovl.s16 q5, d8\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q4, d7\n"
    "vmovl.s16 q3, d6\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vfma.f32 q6, q0, q13\n"
    "vfma.f32 q7, q1, q13\n"
    "vfma.f32 q8, q2, q13\n"
    "vfma.f32 q6, q3, q14\n"
    "vfma.f32 q7, q4, q14\n"
    "vfma.f32 q8, q5, q14\n"
    "vcvt.s32.f32 q6, q6\n"
    "vcvt.s32.f32 q7, q7\n"
    "vcvt.s32.f32 q8, q8\n"

    "vst1.32 {d12, d13, d14, d15}, [%[output]]!\n"
    "vst1.32 {d16}, [%[output]]!\n"
    "vst1.32 {d17[0]}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 12>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 12>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #12\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "vld1.32 {d6}, [r1]!\n"
    "vld1.32 {d7[0]}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q6, q12\n"
    "vmovl.u8 q4, d7\n"
    "vmovl.u8 q3, d6\n"
    "vmov.f32 q7, q12\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q8, q12\n"
    "vmovl.s16 q5, d8\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q4, d7\n"
    "vmovl.s16 q3, d6\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vfma.f32 q6, q0, q13\n"
    "vfma.f32 q7, q1, q13\n"
    "vfma.f32 q8, q2, q13\n"
    "vfma.f32 q6, q3, q14\n"
    "vfma.f32 q7, q4, q14\n"
    "vfma.f32 q8, q5, q14\n"
    "vcvt.s32.f32 q6, q6\n"
    "vcvt.s32.f32 q7, q7\n"
    "vcvt.s32.f32 q8, q8\n"

    "vst1.32 {d12, d13, d14, d15}, [%[output]]!\n"
    "vst1.32 {d16, d17}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 13>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 13>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #13\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "vld1.8 {d1[4]}, [%[input]]!\n"
    "vld1.32 {d8}, [r1]!\n"
    "vld1.32 {d9[0]}, [r1]!\n"
    "vld1.8 {d9[4]}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21}, [%[output]]!\n"
    "vst1.32 {d22[0]}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 14>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 14>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #14\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "vld1.16 {d1[2]}, [%[input]]!\n"
    "vld1.32 {d8}, [r1]!\n"
    "vld1.32 {d9[0]}, [r1]!\n"
    "vld1.16 {d9[2]}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

template<>
inline void Transform1DKernel<uint8_t, int32_t, BiasAdd<uint8_t>, 16, 15>::Transform(const uint8_t* input, const BiasAdd<uint8_t>& params, int32_t* output) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__ << ") BiasAdd<uint8_t><uint8_t, int32_t, BiasAdd<uint8_t>, 16, 15>::Transform()" << std::endl << std::flush;
#endif
#endif
  int params_rows_copy = params.rows;
  const float coeff_input = params.input_range_scale * params.one_over_output_range_scale;
  const float coeff_bias = params.bias_range_scale * params.one_over_output_range_scale;
  const float offset = params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      );
  asm volatile(
    "vdup.32 q12, %[offset]\n"
    "vdup.32 q13, %[coeff_input]\n"
    "vdup.32 q14, %[coeff_bias]\n"
    "1:"
    "mov r0, %[count]\n"
    "mov r1, %[bias]\n"
    "subs r0, r0, #15\n"
    "beq 3f\n"
    "2:"
    "subs r0, r0, #16\n"

    // BiasAdd::Transform
    "vld1.32 {d0, d1}, [%[input]]!\n"
    "vld1.32 {d8, d9}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "vld1.32 {d0}, [%[input]]!\n"
    "vld1.32 {d1[0]}, [%[input]]!\n"
    "vld1.16 {d1[2]}, [%[input]]!\n"
    "vld1.8 {d1[6]}, [%[input]]!\n"
    "vld1.32 {d8}, [r1]!\n"
    "vld1.32 {d9[0]}, [r1]!\n"
    "vld1.16 {d9[2]}, [r1]!\n"
    "vld1.8 {d9[6]}, [r1]!\n"
    "vmovl.u8 q1, d1\n"
    "vmovl.u8 q0, d0\n"
    "vmov.f32 q8, q12\n"
    "vmovl.u8 q5, d9\n"
    "vmovl.u8 q4, d8\n"
    "vmov.f32 q9, q12\n"
    "vmovl.s16 q3, d3\n"
    "vmovl.s16 q2, d2\n"
    "vmov.f32 q10, q12\n"
    "vmovl.s16 q7, d11\n"
    "vmovl.s16 q6, d10\n"
    "vmov.f32 q11, q12\n"
    "vmovl.s16 q1, d1\n"
    "vmovl.s16 q0, d0\n"
    "vmovl.s16 q5, d9\n"
    "vmovl.s16 q4, d8\n"
    "vcvt.f32.s32 q0, q0\n"
    "vcvt.f32.s32 q1, q1\n"
    "vcvt.f32.s32 q2, q2\n"
    "vcvt.f32.s32 q3, q3\n"
    "vcvt.f32.s32 q4, q4\n"
    "vcvt.f32.s32 q5, q5\n"
    "vcvt.f32.s32 q6, q6\n"
    "vcvt.f32.s32 q7, q7\n"
    "vfma.f32 q8, q0, q13\n"
    "vfma.f32 q9, q1, q13\n"
    "vfma.f32 q10, q2, q13\n"
    "vfma.f32 q11, q3, q13\n"
    "vfma.f32 q8, q4, q14\n"
    "vfma.f32 q9, q5, q14\n"
    "vfma.f32 q10, q6, q14\n"
    "vfma.f32 q11, q7, q14\n"
    "vcvt.s32.f32 q8, q8\n"
    "vcvt.s32.f32 q9, q9\n"
    "vcvt.s32.f32 q10, q10\n"
    "vcvt.s32.f32 q11, q11\n"

    "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
    "vst1.32 {d20, d21, d22}, [%[output]]!\n"
    "vst1.32 {d23[0]}, [%[output]]!\n"
    "subs %[rows], %[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory"
  );
}

}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm for arm32 requires: GEMMLOWP_NEON_32!"
#endif

#endif  // GEMMLOWP_META_TRANSFORM_KERNELS_ARM_32_H_
