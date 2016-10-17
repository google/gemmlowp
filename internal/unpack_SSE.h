// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
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

// unpack_SSE.h: SSE-specialized unpacking the result blocks computed by 
// compute.h, storing them into the destination matrix.

#ifndef GEMMLOWP_INTERNAL_UNPACK_SSE_H_
#define GEMMLOWP_INTERNAL_UNPACK_SSE_H_

#define ACTIVATE_SSE_OUTPUT

#include "allocator.h"
#include "block_params.h"
#include "output_sse.h"
#ifdef ACTIVATE_SSE_OUTPUT
#include "pack_SSE.h"
#else
#include "pack.h"
#endif

#include <iostream>

#include <cmath>

namespace gemmlowp {

// Non-optimized rounding routine since SSE does not have
// optimized kernels for less than 8bit input values
template <std::uint32_t numerator, std::uint32_t denominator>
void SSERoundingMultiplyByConstantFraction(__m128i* x) {
  if (numerator == denominator) {
    return; 
  }
  std::int32_t *x_array = reinterpret_cast<int32_t*>(x);
  for (int i=0; i<4; ++i) {
    x_array[i] = RoundingMultiplyByConstantFraction<numerator, denominator>(x_array[i]);
  }
}

template <typename tScalar, VectorShape tShape>
__m128i get_m128i_and_inc(
    ConstIterator<VectorMap<tScalar, tShape>>* iterator) {
  const __m128i result = _mm_loadu_si128((const __m128i*) iterator->get());
  *iterator += 4;
  return result;
}

template <typename tScalar, VectorShape tShape>
__m128i get_m128i_and_inc(
    ConstIterator<VectorDup<tScalar, tShape>>* iterator) {
  const __m128i result = _mm_set1_epi32(**iterator);
  // Increment really does nothing for VectorDup.
  *iterator += 4;
  return result;
}

template <typename vectype>
__m128i mul32i_xdup_y(vectype x, vectype y) {
  __m128i y1 = _mm_shuffle_epi32(y, 0x50);
  vectype res1 = _mm_mul_epi32(x, y1);
  __m128i y2 = _mm_shuffle_epi32(y, 0xfa);
  vectype res2 = _mm_mul_epi32(x, y2);
  vectype res  = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(res1), _mm_castsi128_ps(res2), 0x88));
  return res;
}


template <typename BitDepthParams, typename PackedResultType, 
          typename OutputScalar, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType>
struct UnpackResultImpl<BitDepthParams,
                        MatrixMap<OutputScalar, MapOrder::ColMajor>,
                        PackedResultType, LhsOffset, RhsOffset,
                        OutputPipelineType> {
  typedef MatrixMap<OutputScalar, MapOrder::ColMajor> ResultBlockType;
  static void Unpack(ResultBlockType* dst, const MatrixBlockBounds& dst_block,
                     const PackedResultType& src, int depth,
                     const std::int32_t* lhs_sums_of_each_slice,
                     const std::int32_t* rhs_sums_of_each_slice,
                     const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                     const OutputPipelineType& output_pipeline) {
    ScopedProfilingLabel label("optimized path (SSE)");
    std::cout << "SSE unpack code is activated" << std::endl;
    assert(dst_block.start_row >= 0);
    assert(dst_block.start_row + dst_block.rows <= dst->rows());
    assert(dst_block.start_col >= 0);
    assert(dst_block.start_col + dst_block.cols <= dst->cols());
    auto src_map = src.Map();
    // No top-level blocking in the depth dimension at the moment.
    // Too much loss of precision.
    const int kLhsBits = BitDepthParams::LhsBitDepth::kBits;
    const int kRhsBits = BitDepthParams::RhsBitDepth::kBits;
    const std::int32_t kLhsMax = (1 << kLhsBits) - 1;
    const std::int32_t kRhsMax = (1 << kRhsBits) - 1;
    __m128i depth_xmm = _mm_set1_epi32((int32_t)depth);
#ifdef ACTIVATE_SSE_OUTPUT
    OutputPipelineExecutor<OutputPipelineType, SSEFragmentInt32x4x1>
            int32x4x1_output_pipeline_executor(output_pipeline);
#endif
    OutputPipelineExecutor<OutputPipelineType, FragmentInt32x1x1>
        output_pipeline_executor(output_pipeline);
#if 1
    for (int c = 0; c < dst_block.cols; c++) {
      int c_dst = c + dst_block.start_col;
      const std::int32_t* src_ptr = src_map.data(0, c);
      const std::int32_t* lhs_sums_of_each_slice_ptr = lhs_sums_of_each_slice;
      auto lhs_offset_iter = const_iterator(lhs_offset, dst_block.start_row);
      const std::int32_t rhs_offset_c = rhs_offset(c_dst);
      const std::int32_t rhs_sums_of_each_slice_c = rhs_sums_of_each_slice[c];

      __m128i rhs_offset_xmm = _mm_set1_epi32(rhs_offset_c);
      __m128i rhs_sums_xmm   = _mm_set1_epi32(rhs_sums_of_each_slice_c);

      // Handle 4 values at once for higher performance
      int dst_rows_aligned4 = RoundDown<4>(dst_block.rows);
      for (int r = 0; r < dst_rows_aligned4; 
           r += 4, src_ptr += 4, lhs_sums_of_each_slice_ptr += 4) {
        int r_dst = r + dst_block.start_row;

        // xx term: src_map(r:r+4,c)
        __m128i term_xx_xmm = _mm_loadu_si128((const __m128i*) src_ptr);
        SSERoundingMultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax> (&term_xx_xmm);
        // x1 term: lhs_sums_of_each_slice[r:r+4] * rhs_offset(c_dst)
        __m128i term_x1_xmm = _mm_loadu_si128((const __m128i*) lhs_sums_of_each_slice_ptr);
        // term_x1_xmm = _mm_mul_epi32(term_x1_xmm, rhs_offset_xmm);
        term_x1_xmm = mul32i_xdup_y(rhs_offset_xmm, term_x1_xmm);
        SSERoundingMultiplyByConstantFraction<255, kLhsMax> (&term_x1_xmm);
        // 1x term: rhs_sums_of_each_slice[c] * lhs_offset(r_dst:r_dst+4)
        __m128i lhs_offset_xmm = get_m128i_and_inc(&lhs_offset_iter);
        //__m128i term_1x_xmm = _mm_mul_epi32(rhs_sums_xmm, lhs_offset_xmm);
        __m128i term_1x_xmm = mul32i_xdup_y(rhs_sums_xmm, lhs_offset_xmm);
        SSERoundingMultiplyByConstantFraction<255, kLhsMax> (&term_1x_xmm);
        // 11 term: lhs_offset(r_dst:r_dst+4) * rhs_offset(c_dst) * depth
        //__m128i term_11_xmm = _mm_mul_epi32(depth_xmm,
        //_mm_mul_epi32(lhs_offset_xmm, rhs_offset_xmm));
        __m128i term_11_xmm_ = mul32i_xdup_y(rhs_offset_xmm, lhs_offset_xmm);
        __m128i term_11_xmm  = mul32i_xdup_y(depth_xmm, term_11_xmm_);

        SSERoundingMultiplyByConstantFraction<255, kLhsMax> (&term_11_xmm);
        // sum xx, x1, 1x, 11
        __m128i sum_xmm = _mm_add_epi32(
            _mm_add_epi32(term_xx_xmm, term_x1_xmm), 
            _mm_add_epi32(term_1x_xmm, term_11_xmm));
#ifdef ACTIVATE_SSE_OUTPUT
        SSEFragmentInt32x4x1 f(sum_xmm);
        int32x4x1_output_pipeline_executor.Execute(f, dst, r_dst, c_dst);
#else
        std::int32_t* sum_array = reinterpret_cast<int32_t*>(&sum_xmm);
        for(int isse = 0; isse < 4; ++isse) {
          FragmentInt32x1x1 sum = sum_array[isse];
          output_pipeline_executor.Execute(sum, dst, r_dst + isse, c_dst);
        }
#endif
      }

      for (int r = dst_rows_aligned4; r < dst_block.rows; r++) {
        int r_dst = r + dst_block.start_row;
        // To understand this code, read
        //   doc/low-precision.txt
        //   doc/less-than-8-bit.txt
        // We have 4 terms to sum: xx, x1, 1x, 11.
        // In case of requantization, we first need to scale them back
        // to the original scale, using RoundingMultiplyByConstantFraction.
        std::int32_t raw_xx = src_map(r, c);
        std::int32_t raw_x1 = lhs_sums_of_each_slice[r] * rhs_offset(c_dst);
        std::int32_t raw_1x = rhs_sums_of_each_slice[c] * lhs_offset(r_dst);
        std::int32_t term_xx =
            RoundingMultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(
                raw_xx);
        std::int32_t term_x1 =
            RoundingMultiplyByConstantFraction<255, kLhsMax>(raw_x1);
        std::int32_t term_1x =
            RoundingMultiplyByConstantFraction<255, kRhsMax>(raw_1x);
        std::int32_t term_11 = lhs_offset(r_dst) * rhs_offset(c_dst) * depth;
        // Sum the 4 terms.
        FragmentInt32x1x1 sum = term_xx + term_x1 + term_1x + term_11;
        output_pipeline_executor.Execute(sum, dst, r_dst, c_dst);
      }
    }
#else
    for (int c = 0; c < dst_block.cols; c++) {
      int c_dst = c + dst_block.start_col;
      for (int r = 0; r < dst_block.rows; r++) {
        int r_dst = r + dst_block.start_row;
        // To understand this code, read
        //   doc/low-precision.txt
        //   doc/less-than-8-bit.txt
        // We have 4 terms to sum: xx, x1, 1x, 11.
        // In case of requantization, we first need to scale them back
        // to the original scale, using RoundingMultiplyByConstantFraction.
        std::int32_t raw_xx = src_map(r, c);
        std::int32_t raw_x1 = lhs_sums_of_each_slice[r] * rhs_offset(c_dst);
        std::int32_t raw_1x = rhs_sums_of_each_slice[c] * lhs_offset(r_dst);
        std::int32_t term_xx =
            RoundingMultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(
                raw_xx);
        std::int32_t term_x1 =
            RoundingMultiplyByConstantFraction<255, kLhsMax>(raw_x1);
        std::int32_t term_1x =
            RoundingMultiplyByConstantFraction<255, kRhsMax>(raw_1x);
        std::int32_t term_11 = lhs_offset(r_dst) * rhs_offset(c_dst) * depth;
        // Sum the 4 terms.
        FragmentInt32x1x1 sum = term_xx + term_x1 + term_1x + term_11;

        output_pipeline_executor.Execute(sum, dst, r_dst, c_dst);
      }
    }
#endif
  }
};
}  // namespace gemmlowp


#endif  // GEMMLOWP_INTERNAL_UNPACK_H_
