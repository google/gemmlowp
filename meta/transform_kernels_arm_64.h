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

#ifndef GEMMLOWP_META_TRANSFORM_KERNELS_ARM_64_H_
#define GEMMLOWP_META_TRANSFORM_KERNELS_ARM_64_H_

#ifdef GEMMLOWP_NEON_64

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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #0\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.4s}, [%x[output]], #16\n"



      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.4s}, [%x[output]], #16\n"



      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"



    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #1\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.s}[0], [%x[input]], #4\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v8.4h, v8.4s\n"

      "sqxtun v8.8b, v8.8h\n"

      "st1 {v8.b}[0], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.s}[0], [%x[input]], #4\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v4.4h, v4.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.b}[0], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.s}[0], [%x[input]], #4\n"


      "scvtf v0.4s, v0.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v4.4h, v4.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.b}[0], [%x[output]], #1\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #2\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.2s}, [%x[input]], #8\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v8.4h, v8.4s\n"

      "sqxtun v8.8b, v8.8h\n"

      "st1 {v8.h}[0], [%x[output]], #2\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.2s}, [%x[input]], #8\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v4.4h, v4.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.h}[0], [%x[output]], #2\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.2s}, [%x[input]], #8\n"


      "scvtf v0.4s, v0.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v4.4h, v4.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.h}[0], [%x[output]], #2\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #3\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.2s}, [%x[input]], #8\n"
      "ld1 {v0.s}[2], [%x[input]], #4\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v8.4h, v8.4s\n"

      "sqxtun v8.8b, v8.8h\n"

      "st1 {v8.h}[0], [%x[output]], #2\n"
      "st1 {v8.b}[2], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.2s}, [%x[input]], #8\n"
      "ld1 {v0.s}[2], [%x[input]], #4\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v4.4h, v4.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.h}[0], [%x[output]], #2\n"
      "st1 {v4.b}[2], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.2s}, [%x[input]], #8\n"
      "ld1 {v0.s}[2], [%x[input]], #4\n"


      "scvtf v0.4s, v0.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v4.4h, v4.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.h}[0], [%x[output]], #2\n"
      "st1 {v4.b}[2], [%x[output]], #1\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #4\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v8.4h, v8.4s\n"

      "sqxtun v8.8b, v8.8h\n"

      "st1 {v8.s}[0], [%x[output]], #4\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v4.4h, v4.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.s}[0], [%x[output]], #4\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"


      "scvtf v0.4s, v0.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"

      "sqxtn v4.4h, v4.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.s}[0], [%x[output]], #4\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #5\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.s}[0], [%x[input]], #4\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtun v8.8b, v8.8h\n"

      "st1 {v8.s}[0], [%x[output]], #4\n"
      "st1 {v8.b}[4], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.s}[0], [%x[input]], #4\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.s}[0], [%x[output]], #4\n"
      "st1 {v4.b}[4], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "mov v5.16b, v13.16b\n"
      "ld1 {v1.s}[0], [%x[input]], #4\n"


      "scvtf v0.4s, v0.4s\n"

      "scvtf v1.4s, v1.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.s}[0], [%x[output]], #4\n"
      "st1 {v4.b}[4], [%x[output]], #1\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #6\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.2s}, [%x[input]], #8\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtun v8.8b, v8.8h\n"

      "st1 {v8.s}[0], [%x[output]], #4\n"
      "st1 {v8.h}[2], [%x[output]], #2\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.2s}, [%x[input]], #8\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.s}[0], [%x[output]], #4\n"
      "st1 {v4.h}[2], [%x[output]], #2\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "mov v5.16b, v13.16b\n"
      "ld1 {v1.2s}, [%x[input]], #8\n"


      "scvtf v0.4s, v0.4s\n"

      "scvtf v1.4s, v1.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.s}[0], [%x[output]], #4\n"
      "st1 {v4.h}[2], [%x[output]], #2\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #7\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.2s}, [%x[input]], #8\n"
      "ld1 {v1.s}[2], [%x[input]], #4\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtun v8.8b, v8.8h\n"

      "st1 {v8.s}[0], [%x[output]], #4\n"
      "st1 {v8.h}[2], [%x[output]], #2\n"
      "st1 {v8.b}[6], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.2s}, [%x[input]], #8\n"
      "ld1 {v1.s}[2], [%x[input]], #4\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.s}[0], [%x[output]], #4\n"
      "st1 {v4.h}[2], [%x[output]], #2\n"
      "st1 {v4.b}[6], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "mov v5.16b, v13.16b\n"
      "ld1 {v1.2s}, [%x[input]], #8\n"
      "ld1 {v1.s}[2], [%x[input]], #4\n"


      "scvtf v0.4s, v0.4s\n"

      "scvtf v1.4s, v1.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.s}[0], [%x[output]], #4\n"
      "st1 {v4.h}[2], [%x[output]], #2\n"
      "st1 {v4.b}[6], [%x[output]], #1\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #8\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtun v8.8b, v8.8h\n"

      "st1 {v8.2s}, [%x[output]], #8\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"


      "scvtf v0.4s, v0.4s\n"

      "scvtf v1.4s, v1.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtun v4.8b, v4.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #9\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.s}[0], [%x[input]], #4\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtn v10.4h, v10.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.2s}, [%x[output]], #8\n"
      "st1 {v8.b}[8], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.s}[0], [%x[input]], #4\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.b}[8], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "mov v6.16b, v13.16b\n"
      "ld1 {v2.s}[0], [%x[input]], #4\n"


      "scvtf v0.4s, v0.4s\n"

      "scvtf v1.4s, v1.4s\n"

      "scvtf v2.4s, v2.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.b}[8], [%x[output]], #1\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #10\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.2s}, [%x[input]], #8\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtn v10.4h, v10.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.2s}, [%x[output]], #8\n"
      "st1 {v8.h}[4], [%x[output]], #2\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.2s}, [%x[input]], #8\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.h}[4], [%x[output]], #2\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "mov v6.16b, v13.16b\n"
      "ld1 {v2.2s}, [%x[input]], #8\n"


      "scvtf v0.4s, v0.4s\n"

      "scvtf v1.4s, v1.4s\n"

      "scvtf v2.4s, v2.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.h}[4], [%x[output]], #2\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #11\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.2s}, [%x[input]], #8\n"
      "ld1 {v2.s}[2], [%x[input]], #4\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtn v10.4h, v10.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.2s}, [%x[output]], #8\n"
      "st1 {v8.h}[4], [%x[output]], #2\n"
      "st1 {v8.b}[10], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.2s}, [%x[input]], #8\n"
      "ld1 {v2.s}[2], [%x[input]], #4\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.h}[4], [%x[output]], #2\n"
      "st1 {v4.b}[10], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "mov v6.16b, v13.16b\n"
      "ld1 {v2.2s}, [%x[input]], #8\n"
      "ld1 {v2.s}[2], [%x[input]], #4\n"


      "scvtf v0.4s, v0.4s\n"

      "scvtf v1.4s, v1.4s\n"

      "scvtf v2.4s, v2.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.h}[4], [%x[output]], #2\n"
      "st1 {v4.b}[10], [%x[output]], #1\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #12\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtn v10.4h, v10.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.2s}, [%x[output]], #8\n"
      "st1 {v8.s}[2], [%x[output]], #4\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.s}[2], [%x[output]], #4\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"


      "scvtf v0.4s, v0.4s\n"

      "scvtf v1.4s, v1.4s\n"

      "scvtf v2.4s, v2.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.s}[2], [%x[output]], #4\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #13\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.s}[0], [%x[input]], #4\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.2s}, [%x[output]], #8\n"
      "st1 {v8.s}[2], [%x[output]], #4\n"
      "st1 {v8.b}[12], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.s}[0], [%x[input]], #4\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.s}[2], [%x[output]], #4\n"
      "st1 {v4.b}[12], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "mov v7.16b, v13.16b\n"
      "ld1 {v3.s}[0], [%x[input]], #4\n"


      "scvtf v0.4s, v0.4s\n"

      "scvtf v1.4s, v1.4s\n"

      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.s}[2], [%x[output]], #4\n"
      "st1 {v4.b}[12], [%x[output]], #1\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #14\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.2s}, [%x[input]], #8\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.2s}, [%x[output]], #8\n"
      "st1 {v8.s}[2], [%x[output]], #4\n"
      "st1 {v8.h}[6], [%x[output]], #2\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.2s}, [%x[input]], #8\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.s}[2], [%x[output]], #4\n"
      "st1 {v4.h}[6], [%x[output]], #2\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "mov v7.16b, v13.16b\n"
      "ld1 {v3.2s}, [%x[input]], #8\n"


      "scvtf v0.4s, v0.4s\n"

      "scvtf v1.4s, v1.4s\n"

      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.s}[2], [%x[output]], #4\n"
      "st1 {v4.h}[6], [%x[output]], #2\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v12.4s, %w[coefficient]\n"
    "dup v13.4s, %w[offset]\n"

    // Reduce count by leftovers
    "subs %x[count], %x[count], #15\n"
    "beq 4f\n"

    // Prepare initial values
    "subs %x[count], %x[count], #16\n"
    "mov v4.16b, v13.16b\n"
    "ld1 {v0.4s}, [%x[input]], #16\n"

    "mov v5.16b, v13.16b\n"
    "ld1 {v1.4s}, [%x[input]], #16\n"

    "mov v6.16b, v13.16b\n"
    "ld1 {v2.4s}, [%x[input]], #16\n"

    "mov v7.16b, v13.16b\n"
    "ld1 {v3.4s}, [%x[input]], #16\n"


    "scvtf v0.4s, v0.4s\n"

    "scvtf v1.4s, v1.4s\n"

    "scvtf v2.4s, v2.4s\n"

    "scvtf v3.4s, v3.4s\n"

    "beq 2f\n"


    // Requantize::Transform
    "1:"
      // Requantize::Transform::Loop part A
      "subs %x[count], %x[count], #16\n"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "beq 3f\n"

      // Requantize::Transform::Loop part B
      "subs %x[count], %x[count], #16\n"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.4s}, [%x[input]], #16\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "bne 1b\n"

    // Requantize::Transform::Tail A
    "2:"
      "fmla v4.4s, v0.4s, v12.4s\n"
      "mov v8.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v5.4s, v1.4s, v12.4s\n"
      "mov v9.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v6.4s, v2.4s, v12.4s\n"
      "mov v10.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v7.4s, v3.4s, v12.4s\n"
      "mov v11.16b, v13.16b\n"
      "ld1 {v3.2s}, [%x[input]], #8\n"
      "ld1 {v3.s}[2], [%x[input]], #4\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v4.4s}, [%x[output]], #16\n"

      "fmla v8.4s, v0.4s, v12.4s\n"

      "fmla v9.4s, v1.4s, v12.4s\n"

      "fmla v10.4s, v2.4s, v12.4s\n"

      "fmla v11.4s, v3.4s, v12.4s\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"

      "st1 {v8.2s}, [%x[output]], #8\n"
      "st1 {v8.s}[2], [%x[output]], #4\n"
      "st1 {v8.h}[6], [%x[output]], #2\n"
      "st1 {v8.b}[14], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail B
    "3:"
      "fmla v8.4s, v0.4s, v12.4s\n"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "fmla v9.4s, v1.4s, v12.4s\n"
      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "fmla v10.4s, v2.4s, v12.4s\n"
      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "fmla v11.4s, v3.4s, v12.4s\n"
      "mov v7.16b, v13.16b\n"
      "ld1 {v3.2s}, [%x[input]], #8\n"
      "ld1 {v3.s}[2], [%x[input]], #4\n"

      "fcvtzs v8.4s, v8.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v9.4s, v9.4s\n"
      "fcvtzs v10.4s, v10.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v11.4s, v11.4s\n"

      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"

      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "scvtf v1.4s, v1.4s\n"

      "sqxtun v8.8b, v8.8h\n"
      "sqxtun2 v8.16b, v10.8h\n"
      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "st1 {v8.4s}, [%x[output]], #16\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.s}[2], [%x[output]], #4\n"
      "st1 {v4.h}[6], [%x[output]], #2\n"
      "st1 {v4.b}[14], [%x[output]], #1\n"

      "b 5f\n"

    // Requantize::Transform::Tail C
    "4:"
      "mov v4.16b, v13.16b\n"
      "ld1 {v0.4s}, [%x[input]], #16\n"

      "mov v5.16b, v13.16b\n"
      "ld1 {v1.4s}, [%x[input]], #16\n"

      "mov v6.16b, v13.16b\n"
      "ld1 {v2.4s}, [%x[input]], #16\n"

      "mov v7.16b, v13.16b\n"
      "ld1 {v3.2s}, [%x[input]], #8\n"
      "ld1 {v3.s}[2], [%x[input]], #4\n"


      "scvtf v0.4s, v0.4s\n"

      "scvtf v1.4s, v1.4s\n"

      "scvtf v2.4s, v2.4s\n"

      "scvtf v3.4s, v3.4s\n"

      "fmla v4.4s, v0.4s, v12.4s\n"

      "fmla v5.4s, v1.4s, v12.4s\n"

      "fmla v6.4s, v2.4s, v12.4s\n"

      "fmla v7.4s, v3.4s, v12.4s\n"

      "fcvtzs v4.4s, v4.4s\n"
      "prfm pldl2keep, [%x[input], #2048]\n"
      "fcvtzs v5.4s, v5.4s\n"
      "fcvtzs v6.4s, v6.4s\n"
      "prfm pldl1keep, [%x[input], #64]\n"
      "fcvtzs v7.4s, v7.4s\n"

      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"

      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"

      "sqxtun v4.8b, v4.8h\n"
      "sqxtun2 v4.16b, v6.8h\n"

      "st1 {v4.2s}, [%x[output]], #8\n"
      "st1 {v4.s}[2], [%x[output]], #4\n"
      "st1 {v4.h}[6], [%x[output]], #2\n"
      "st1 {v4.b}[14], [%x[output]], #1\n"

    // Requantize::Return
    "5:"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [coefficient] "r"(coefficient), [offset] "r"(offset)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #1\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtun v0.8b, v0.8h\n"

    "st1 {v0.b}[0], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #2\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtun v0.8b, v0.8h\n"

    "st1 {v0.h}[0], [%x[output]], #2\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #3\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtun v0.8b, v0.8h\n"

    "st1 {v0.h}[0], [%x[output]], #2\n"
    "st1 {v0.b}[2], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #4\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtun v0.8b, v0.8h\n"

    "st1 {v0.s}[0], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #5\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v1.s}[0], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtun v0.8b, v0.8h\n"

    "st1 {v0.s}[0], [%x[output]], #4\n"
    "st1 {v0.b}[4], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #6\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v1.2s}, [%x[input]], #8\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtun v0.8b, v0.8h\n"

    "st1 {v0.s}[0], [%x[output]], #4\n"
    "st1 {v0.h}[2], [%x[output]], #2\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #7\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v1.2s}, [%x[input]], #8\n"
    "ld1 {v1.s}[2], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtun v0.8b, v0.8h\n"

    "st1 {v0.s}[0], [%x[output]], #4\n"
    "st1 {v0.h}[2], [%x[output]], #2\n"
    "st1 {v0.b}[6], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #8\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s}, [%x[input]], #32\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtun v0.8b, v0.8h\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #9\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s}, [%x[input]], #32\n"
    "ld1 {v2.s}[0], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.b}[8], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #10\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s}, [%x[input]], #32\n"
    "ld1 {v2.2s}, [%x[input]], #8\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.h}[4], [%x[output]], #2\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #11\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s}, [%x[input]], #32\n"
    "ld1 {v2.2s}, [%x[input]], #8\n"
    "ld1 {v2.s}[2], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.h}[4], [%x[output]], #2\n"
    "st1 {v0.b}[10], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #12\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s}, [%x[input]], #48\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.s}[2], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #13\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s}, [%x[input]], #48\n"
    "ld1 {v3.s}[0], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.s}[2], [%x[output]], #4\n"
    "st1 {v0.b}[12], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #14\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s}, [%x[input]], #48\n"
    "ld1 {v3.2s}, [%x[input]], #8\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.s}[2], [%x[output]], #4\n"
    "st1 {v0.h}[6], [%x[output]], #2\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #15\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[input]], #64\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Quantize::Transform
    "ld1 {v0.4s, v1.4s, v2.4s}, [%x[input]], #48\n"
    "ld1 {v3.2s}, [%x[input]], #8\n"
    "ld1 {v3.s}[2], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #64]\n"
    "fsub v0.4s, v0.4s, v4.4s\n"
    "fsub v1.4s, v1.4s, v4.4s\n"
    "fsub v2.4s, v2.4s, v4.4s\n"
    "fsub v3.4s, v3.4s, v4.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v5.4s\n"
    "fadd v1.4s, v1.4s, v5.4s\n"
    "fadd v2.4s, v2.4s, v5.4s\n"
    "fadd v3.4s, v3.4s, v5.4s\n"
    "fcvtzs v0.4s, v0.4s\n"
    "fcvtzs v1.4s, v1.4s\n"
    "fcvtzs v2.4s, v2.4s\n"
    "fcvtzs v3.4s, v3.4s\n"
    "sqxtn v0.4h, v0.4s\n"
    "sqxtn2 v0.8h, v1.4s\n"
    "sqxtn v2.4h, v2.4s\n"
    "sqxtn2 v2.8h, v3.4s\n"
    "sqxtun v0.8b, v0.8h\n"
    "sqxtun2 v0.16b, v2.8h\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.s}[2], [%x[output]], #4\n"
    "st1 {v0.h}[6], [%x[output]], #2\n"
    "st1 {v0.b}[14], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #1\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.b}[0], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"

    "st1 {v0.s}[0], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #2\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.h}[0], [%x[input]], #2\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #3\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.h}[0], [%x[input]], #2\n"
    "ld1 {v0.b}[2], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.s}[2], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #4\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #5\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "ld1 {v0.b}[4], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "st1 {v1.s}[0], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #6\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "ld1 {v0.h}[2], [%x[input]], #2\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "st1 {v1.2s}, [%x[output]], #8\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #7\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "ld1 {v0.h}[2], [%x[input]], #2\n"
    "ld1 {v0.b}[6], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "st1 {v1.2s}, [%x[output]], #8\n"
    "st1 {v1.s}[2], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #8\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s}, [%x[output]], #32\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #9\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.b}[8], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s}, [%x[output]], #32\n"
    "st1 {v2.s}[0], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #10\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.h}[4], [%x[input]], #2\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s}, [%x[output]], #32\n"
    "st1 {v2.2s}, [%x[output]], #8\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #11\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.h}[4], [%x[input]], #2\n"
    "ld1 {v0.b}[10], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s}, [%x[output]], #32\n"
    "st1 {v2.2s}, [%x[output]], #8\n"
    "st1 {v2.s}[2], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #12\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s}, [%x[output]], #48\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #13\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "ld1 {v0.b}[12], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s}, [%x[output]], #48\n"
    "st1 {v3.s}[0], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #14\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "ld1 {v0.h}[6], [%x[input]], #2\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s}, [%x[output]], #48\n"
    "st1 {v3.2s}, [%x[output]], #8\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.4s, %w[range_min]\n"
    "dup v5.4s, %w[range_offset]\n"
    "dup v6.4s, %w[range_scale]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #15\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // Dequantize::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%x[output]], #64\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // Dequantize::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "ld1 {v0.h}[6], [%x[input]], #2\n"
    "ld1 {v0.b}[14], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #32]\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fsub v0.4s, v0.4s, v5.4s\n"
    "fsub v1.4s, v1.4s, v5.4s\n"
    "fsub v2.4s, v2.4s, v5.4s\n"
    "fsub v3.4s, v3.4s, v5.4s\n"
    "fmul v0.4s, v0.4s, v6.4s\n"
    "fmul v1.4s, v1.4s, v6.4s\n"
    "fmul v2.4s, v2.4s, v6.4s\n"
    "fmul v3.4s, v3.4s, v6.4s\n"
    "fadd v0.4s, v0.4s, v4.4s\n"
    "fadd v1.4s, v1.4s, v4.4s\n"
    "fadd v2.4s, v2.4s, v4.4s\n"
    "fadd v3.4s, v3.4s, v4.4s\n"

    "st1 {v0.4s, v1.4s, v2.4s}, [%x[output]], #48\n"
    "st1 {v3.2s}, [%x[output]], #8\n"
    "st1 {v3.s}[2], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [range_offset] "r"(params.range_offset), [range_scale] "r"(params.range_scale), [range_min] "r"(params.range_min)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #1\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.b}[0], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.b}[0], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #2\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.h}[0], [%x[input]], #2\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.h}[0], [%x[output]], #2\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #3\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.h}[0], [%x[input]], #2\n"
    "ld1 {v0.b}[2], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.h}[0], [%x[output]], #2\n"
    "st1 {v0.b}[2], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #4\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.s}[0], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #5\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "ld1 {v0.b}[4], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.s}[0], [%x[output]], #4\n"
    "st1 {v0.b}[4], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #6\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "ld1 {v0.h}[2], [%x[input]], #2\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.s}[0], [%x[output]], #4\n"
    "st1 {v0.h}[2], [%x[output]], #2\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #7\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "ld1 {v0.h}[2], [%x[input]], #2\n"
    "ld1 {v0.b}[6], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.s}[0], [%x[output]], #4\n"
    "st1 {v0.h}[2], [%x[output]], #2\n"
    "st1 {v0.b}[6], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #8\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #9\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.b}[8], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.b}[8], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #10\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.h}[4], [%x[input]], #2\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.h}[4], [%x[output]], #2\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #11\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.h}[4], [%x[input]], #2\n"
    "ld1 {v0.b}[10], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.h}[4], [%x[output]], #2\n"
    "st1 {v0.b}[10], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #12\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.s}[2], [%x[output]], #4\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #13\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "ld1 {v0.b}[12], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.s}[2], [%x[output]], #4\n"
    "st1 {v0.b}[12], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #14\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "ld1 {v0.h}[6], [%x[input]], #2\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.s}[2], [%x[output]], #4\n"
    "st1 {v0.h}[6], [%x[output]], #2\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v4.16b, %w[min]\n"
    "dup v5.16b, %w[max]\n"

    // Reduce count by leftovers.
    "subs %x[count], %x[count], #15\n"
    "beq 2f\n"

    "1:"
    "subs %x[count], %x[count], #16\n"

    // MinMax::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.4s}, [%x[output]], #16\n"
    "prfm pldl1keep, [%x[output]]\n"

    "bne 1b\n"
    "2:"

    // Handle leftovers.

    // MinMax::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "ld1 {v0.h}[6], [%x[input]], #2\n"
    "ld1 {v0.b}[14], [%x[input]], #1\n"
    "prfm pldl1keep, [%x[input], #16]\n"
    "umax v0.16b, v0.16b, v4.16b\n"
    "umin v0.16b, v0.16b, v5.16b\n"

    "st1 {v0.2s}, [%x[output]], #8\n"
    "st1 {v0.s}[2], [%x[output]], #4\n"
    "st1 {v0.h}[6], [%x[output]], #2\n"
    "st1 {v0.b}[14], [%x[output]], #1\n"
    "prfm pldl1keep, [%x[output]]\n"
    : [count] "+r"(params_count_copy), [input] "+r"(input), [output] "+r"(output)
    : [max] "r"(params.max), [min] "r"(params.min)
    : "v0", "v4", "v5", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #1\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.b}[0], [%x[input]], #1\n"
    "ld1 {v1.b}[0], [x1], #1\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v2.16b, v12.16b\n"
    "uxtl v1.8h, v1.8b\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl v1.4s, v1.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "fmla v2.4s, v0.4s, v13.4s\n"
    "fmla v2.4s, v1.4s, v14.4s\n"
    "fcvtzs v2.4s, v2.4s\n"

    "st1 {v2.s}[0], [%x[output]], #4\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #2\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.h}[0], [%x[input]], #2\n"
    "ld1 {v1.h}[0], [x1], #2\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v2.16b, v12.16b\n"
    "uxtl v1.8h, v1.8b\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl v1.4s, v1.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "fmla v2.4s, v0.4s, v13.4s\n"
    "fmla v2.4s, v1.4s, v14.4s\n"
    "fcvtzs v2.4s, v2.4s\n"

    "st1 {v2.2s}, [%x[output]], #8\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #3\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.h}[0], [%x[input]], #2\n"
    "ld1 {v0.b}[2], [%x[input]], #1\n"
    "ld1 {v1.h}[0], [x1], #2\n"
    "ld1 {v1.b}[2], [x1], #1\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v2.16b, v12.16b\n"
    "uxtl v1.8h, v1.8b\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl v1.4s, v1.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "fmla v2.4s, v0.4s, v13.4s\n"
    "fmla v2.4s, v1.4s, v14.4s\n"
    "fcvtzs v2.4s, v2.4s\n"

    "st1 {v2.2s}, [%x[output]], #8\n"
    "st1 {v2.s}[2], [%x[output]], #4\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #4\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "ld1 {v1.s}[0], [x1], #4\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v2.16b, v12.16b\n"
    "uxtl v1.8h, v1.8b\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl v1.4s, v1.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "fmla v2.4s, v0.4s, v13.4s\n"
    "fmla v2.4s, v1.4s, v14.4s\n"
    "fcvtzs v2.4s, v2.4s\n"

    "st1 {v2.4s}, [%x[output]], #16\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #5\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "ld1 {v0.b}[4], [%x[input]], #1\n"
    "ld1 {v2.s}[0], [x1], #4\n"
    "ld1 {v2.b}[4], [x1], #1\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v4.16b, v12.16b\n"
    "uxtl v2.8h, v2.8b\n"
    "mov v5.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v3.4s, v2.8h\n"
    "sxtl v2.4s, v2.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fmla v4.4s, v0.4s, v13.4s\n"
    "fmla v5.4s, v1.4s, v13.4s\n"
    "fmla v4.4s, v2.4s, v14.4s\n"
    "fmla v5.4s, v3.4s, v14.4s\n"
    "fcvtzs v4.4s, v4.4s\n"
    "fcvtzs v5.4s, v5.4s\n"

    "st1 {v4.4s}, [%x[output]], #16\n"
    "st1 {v5.s}[0], [%x[output]], #4\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #6\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "ld1 {v0.h}[2], [%x[input]], #2\n"
    "ld1 {v2.s}[0], [x1], #4\n"
    "ld1 {v2.h}[2], [x1], #2\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v4.16b, v12.16b\n"
    "uxtl v2.8h, v2.8b\n"
    "mov v5.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v3.4s, v2.8h\n"
    "sxtl v2.4s, v2.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fmla v4.4s, v0.4s, v13.4s\n"
    "fmla v5.4s, v1.4s, v13.4s\n"
    "fmla v4.4s, v2.4s, v14.4s\n"
    "fmla v5.4s, v3.4s, v14.4s\n"
    "fcvtzs v4.4s, v4.4s\n"
    "fcvtzs v5.4s, v5.4s\n"

    "st1 {v4.4s}, [%x[output]], #16\n"
    "st1 {v5.2s}, [%x[output]], #8\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #7\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.s}[0], [%x[input]], #4\n"
    "ld1 {v0.h}[2], [%x[input]], #2\n"
    "ld1 {v0.b}[6], [%x[input]], #1\n"
    "ld1 {v2.s}[0], [x1], #4\n"
    "ld1 {v2.h}[2], [x1], #2\n"
    "ld1 {v2.b}[6], [x1], #1\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v4.16b, v12.16b\n"
    "uxtl v2.8h, v2.8b\n"
    "mov v5.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v3.4s, v2.8h\n"
    "sxtl v2.4s, v2.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fmla v4.4s, v0.4s, v13.4s\n"
    "fmla v5.4s, v1.4s, v13.4s\n"
    "fmla v4.4s, v2.4s, v14.4s\n"
    "fmla v5.4s, v3.4s, v14.4s\n"
    "fcvtzs v4.4s, v4.4s\n"
    "fcvtzs v5.4s, v5.4s\n"

    "st1 {v4.4s}, [%x[output]], #16\n"
    "st1 {v5.2s}, [%x[output]], #8\n"
    "st1 {v5.s}[2], [%x[output]], #4\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #8\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v2.2s}, [x1], #8\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v4.16b, v12.16b\n"
    "uxtl v2.8h, v2.8b\n"
    "mov v5.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v3.4s, v2.8h\n"
    "sxtl v2.4s, v2.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "fmla v4.4s, v0.4s, v13.4s\n"
    "fmla v5.4s, v1.4s, v13.4s\n"
    "fmla v4.4s, v2.4s, v14.4s\n"
    "fmla v5.4s, v3.4s, v14.4s\n"
    "fcvtzs v4.4s, v4.4s\n"
    "fcvtzs v5.4s, v5.4s\n"

    "st1 {v4.4s, v5.4s}, [%x[output]], #32\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #9\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.b}[8], [%x[input]], #1\n"
    "ld1 {v3.2s}, [x1], #8\n"
    "ld1 {v3.b}[8], [x1], #1\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v6.16b, v12.16b\n"
    "uxtl2 v4.8h, v3.16b\n"
    "uxtl v3.8h, v3.8b\n"
    "mov v7.16b, v12.16b\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v8.16b, v12.16b\n"
    "sxtl v5.4s, v4.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v4.4s, v3.8h\n"
    "sxtl v3.4s, v3.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "fmla v6.4s, v0.4s, v13.4s\n"
    "fmla v7.4s, v1.4s, v13.4s\n"
    "fmla v8.4s, v2.4s, v13.4s\n"
    "fmla v6.4s, v3.4s, v14.4s\n"
    "fmla v7.4s, v4.4s, v14.4s\n"
    "fmla v8.4s, v5.4s, v14.4s\n"
    "fcvtzs v6.4s, v6.4s\n"
    "fcvtzs v7.4s, v7.4s\n"
    "fcvtzs v8.4s, v8.4s\n"

    "st1 {v6.4s, v7.4s}, [%x[output]], #32\n"
    "st1 {v8.s}[0], [%x[output]], #4\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #10\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.h}[4], [%x[input]], #2\n"
    "ld1 {v3.2s}, [x1], #8\n"
    "ld1 {v3.h}[4], [x1], #2\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v6.16b, v12.16b\n"
    "uxtl2 v4.8h, v3.16b\n"
    "uxtl v3.8h, v3.8b\n"
    "mov v7.16b, v12.16b\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v8.16b, v12.16b\n"
    "sxtl v5.4s, v4.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v4.4s, v3.8h\n"
    "sxtl v3.4s, v3.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "fmla v6.4s, v0.4s, v13.4s\n"
    "fmla v7.4s, v1.4s, v13.4s\n"
    "fmla v8.4s, v2.4s, v13.4s\n"
    "fmla v6.4s, v3.4s, v14.4s\n"
    "fmla v7.4s, v4.4s, v14.4s\n"
    "fmla v8.4s, v5.4s, v14.4s\n"
    "fcvtzs v6.4s, v6.4s\n"
    "fcvtzs v7.4s, v7.4s\n"
    "fcvtzs v8.4s, v8.4s\n"

    "st1 {v6.4s, v7.4s}, [%x[output]], #32\n"
    "st1 {v8.2s}, [%x[output]], #8\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #11\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.h}[4], [%x[input]], #2\n"
    "ld1 {v0.b}[10], [%x[input]], #1\n"
    "ld1 {v3.2s}, [x1], #8\n"
    "ld1 {v3.h}[4], [x1], #2\n"
    "ld1 {v3.b}[10], [x1], #1\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v6.16b, v12.16b\n"
    "uxtl2 v4.8h, v3.16b\n"
    "uxtl v3.8h, v3.8b\n"
    "mov v7.16b, v12.16b\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v8.16b, v12.16b\n"
    "sxtl v5.4s, v4.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v4.4s, v3.8h\n"
    "sxtl v3.4s, v3.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "fmla v6.4s, v0.4s, v13.4s\n"
    "fmla v7.4s, v1.4s, v13.4s\n"
    "fmla v8.4s, v2.4s, v13.4s\n"
    "fmla v6.4s, v3.4s, v14.4s\n"
    "fmla v7.4s, v4.4s, v14.4s\n"
    "fmla v8.4s, v5.4s, v14.4s\n"
    "fcvtzs v6.4s, v6.4s\n"
    "fcvtzs v7.4s, v7.4s\n"
    "fcvtzs v8.4s, v8.4s\n"

    "st1 {v6.4s, v7.4s}, [%x[output]], #32\n"
    "st1 {v8.2s}, [%x[output]], #8\n"
    "st1 {v8.s}[2], [%x[output]], #4\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #12\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "ld1 {v3.2s}, [x1], #8\n"
    "ld1 {v3.s}[2], [x1], #4\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v6.16b, v12.16b\n"
    "uxtl2 v4.8h, v3.16b\n"
    "uxtl v3.8h, v3.8b\n"
    "mov v7.16b, v12.16b\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v8.16b, v12.16b\n"
    "sxtl v5.4s, v4.4h\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v4.4s, v3.8h\n"
    "sxtl v3.4s, v3.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "fmla v6.4s, v0.4s, v13.4s\n"
    "fmla v7.4s, v1.4s, v13.4s\n"
    "fmla v8.4s, v2.4s, v13.4s\n"
    "fmla v6.4s, v3.4s, v14.4s\n"
    "fmla v7.4s, v4.4s, v14.4s\n"
    "fmla v8.4s, v5.4s, v14.4s\n"
    "fcvtzs v6.4s, v6.4s\n"
    "fcvtzs v7.4s, v7.4s\n"
    "fcvtzs v8.4s, v8.4s\n"

    "st1 {v6.4s, v7.4s, v8.4s}, [%x[output]], #48\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #13\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "ld1 {v0.b}[12], [%x[input]], #1\n"
    "ld1 {v4.2s}, [x1], #8\n"
    "ld1 {v4.s}[2], [x1], #4\n"
    "ld1 {v4.b}[12], [x1], #1\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s}, [%x[output]], #48\n"
    "st1 {v11.s}[0], [%x[output]], #4\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #14\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "ld1 {v0.h}[6], [%x[input]], #2\n"
    "ld1 {v4.2s}, [x1], #8\n"
    "ld1 {v4.s}[2], [x1], #4\n"
    "ld1 {v4.h}[6], [x1], #2\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s}, [%x[output]], #48\n"
    "st1 {v11.2s}, [%x[output]], #8\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
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
    "dup v12.4s, %w[offset]\n"
    "dup v13.4s, %w[coeff_input]\n"
    "dup v14.4s, %w[coeff_bias]\n"
    "1:"
    "mov x0, %x[count]\n"
    "mov x1, %x[bias]\n"
    "subs x0, x0, #15\n"
    "beq 3f\n"
    "2:"
    "subs x0, x0, #16\n"

    // BiasAdd::Transform
    "ld1 {v0.4s}, [%x[input]], #16\n"
    "ld1 {v4.4s}, [x1], #16\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%x[output]], #64\n"
    "bne 2b\n"
    "3:"

    // BiasAdd::Transform
    "ld1 {v0.2s}, [%x[input]], #8\n"
    "ld1 {v0.s}[2], [%x[input]], #4\n"
    "ld1 {v0.h}[6], [%x[input]], #2\n"
    "ld1 {v0.b}[14], [%x[input]], #1\n"
    "ld1 {v4.2s}, [x1], #8\n"
    "ld1 {v4.s}[2], [x1], #4\n"
    "ld1 {v4.h}[6], [x1], #2\n"
    "ld1 {v4.b}[14], [x1], #1\n"
    "uxtl2 v1.8h, v0.16b\n"
    "uxtl v0.8h, v0.8b\n"
    "mov v8.16b, v12.16b\n"
    "uxtl2 v5.8h, v4.16b\n"
    "uxtl v4.8h, v4.8b\n"
    "mov v9.16b, v12.16b\n"
    "sxtl2 v3.4s, v1.8h\n"
    "sxtl v2.4s, v1.4h\n"
    "mov v10.16b, v12.16b\n"
    "sxtl2 v7.4s, v5.8h\n"
    "sxtl v6.4s, v5.4h\n"
    "mov v11.16b, v12.16b\n"
    "sxtl2 v1.4s, v0.8h\n"
    "sxtl v0.4s, v0.4h\n"
    "sxtl2 v5.4s, v4.8h\n"
    "sxtl v4.4s, v4.4h\n"
    "scvtf v0.4s, v0.4s\n"
    "scvtf v1.4s, v1.4s\n"
    "scvtf v2.4s, v2.4s\n"
    "scvtf v3.4s, v3.4s\n"
    "scvtf v4.4s, v4.4s\n"
    "scvtf v5.4s, v5.4s\n"
    "scvtf v6.4s, v6.4s\n"
    "scvtf v7.4s, v7.4s\n"
    "fmla v8.4s, v0.4s, v13.4s\n"
    "fmla v9.4s, v1.4s, v13.4s\n"
    "fmla v10.4s, v2.4s, v13.4s\n"
    "fmla v11.4s, v3.4s, v13.4s\n"
    "fmla v8.4s, v4.4s, v14.4s\n"
    "fmla v9.4s, v5.4s, v14.4s\n"
    "fmla v10.4s, v6.4s, v14.4s\n"
    "fmla v11.4s, v7.4s, v14.4s\n"
    "fcvtzs v8.4s, v8.4s\n"
    "fcvtzs v9.4s, v9.4s\n"
    "fcvtzs v10.4s, v10.4s\n"
    "fcvtzs v11.4s, v11.4s\n"

    "st1 {v8.4s, v9.4s, v10.4s}, [%x[output]], #48\n"
    "st1 {v11.2s}, [%x[output]], #8\n"
    "st1 {v11.s}[2], [%x[output]], #4\n"
    "subs %x[rows], %x[rows], #1\n"
    "bne 1b\n"
    : [input] "+r"(input), [output] "+r"(output)
    : [count] "r"(params.count), [rows] "r"(params_rows_copy), [bias] "r"(params.bias), [coeff_bias] "r"(coeff_bias), [coeff_input] "r"(coeff_input), [offset] "r"(offset)
    : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory"
  );
}

}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm for arm64 requires: GEMMLOWP_NEON_64!"
#endif

#endif  // GEMMLOWP_META_TRANSFORM_KERNELS_ARM_64_H_
