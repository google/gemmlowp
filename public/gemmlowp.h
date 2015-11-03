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

// gemmlowp.h: the main public interface header of gemmlowp.

#ifndef GEMMLOWP_PUBLIC_GEMMLOWP_H_
#define GEMMLOWP_PUBLIC_GEMMLOWP_H_

#include "bit_depth.h"
#include "map.h"
#include "../internal/multi_thread_gemm.h"
#include "../internal/kernel_default.h"

namespace gemmlowp {

inline bool IsRequantizationWorthIt(int rows, int cols) {
  // We pack depth*(rows+cols) and compute depth*rows*cols.
  // Thus the ratio of compute/packing cost is rows*cols/(rows+cols)
  // In the square case rows==cols==N, it becomes N/2.
  return 2 * rows * cols >= (rows + cols) * kMinimumWidthForRequantization;
}

class GemmContext : public MultiThreadGemmContext {};

// Computes a general matrix product ("GEMM").
// The meaning of the offsets, result_mult_int and result_shift
// parameters is the same as in the standard EightBitIntGemm interface
// (which is also implemented in the eight_bit_int_gemm directory).
template <typename Scalar, typename BitDepthParams, MapOrder LhsOrder,
          MapOrder RhsOrder, MapOrder ResultOrder>
void Gemm(GemmContext* context, const MatrixMap<const Scalar, LhsOrder>& lhs,
          const MatrixMap<const Scalar, RhsOrder>& rhs,
          MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
          int rhs_offset, int result_offset, int result_mult_int,
          int result_shift) {
  assert(lhs.cols() == rhs.rows());

  int rows = result->rows();
  int cols = result->cols();
  int depth = lhs.cols();

  if (rows == 0 || cols == 0 || depth == 0) {
    // Vacuous GEMM, return early to avoid having to deal with
    // zero sizes below.
    return;
  }

  if (cols == 1) {
    if (IsRequantizationWorthIt(rows, cols)) {
      typedef DefaultKernel<KernelFamily::Gemv, BitDepthParams> Kernel;
      MultiThreadGemm<typename Kernel::Format, std::uint8_t, BitDepthParams>(
          context, Kernel(), lhs, rhs, result, lhs_offset, rhs_offset,
          result_offset, result_mult_int, result_shift);
    } else {
      typedef DefaultKernel<KernelFamily::Gemv, DefaultL8R8BitDepthParams>
          Kernel;
      MultiThreadGemm<typename Kernel::Format, std::uint8_t,
                      DefaultL8R8BitDepthParams>(
          context, Kernel(), lhs, rhs, result, lhs_offset, rhs_offset,
          result_offset, result_mult_int, result_shift);
    }
  } else {
    if (IsRequantizationWorthIt(rows, cols)) {
      typedef DefaultKernel<KernelFamily::Gemm, BitDepthParams> Kernel;
      MultiThreadGemm<typename Kernel::Format, std::uint8_t, BitDepthParams>(
          context, Kernel(), lhs, rhs, result, lhs_offset, rhs_offset,
          result_offset, result_mult_int, result_shift);
    } else {
      typedef DefaultKernel<KernelFamily::Gemm, DefaultL8R8BitDepthParams>
          Kernel;
      MultiThreadGemm<typename Kernel::Format, std::uint8_t,
                      DefaultL8R8BitDepthParams>(
          context, Kernel(), lhs, rhs, result, lhs_offset, rhs_offset,
          result_offset, result_mult_int, result_shift);
    }
  }
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_GEMMLOWP_H_
