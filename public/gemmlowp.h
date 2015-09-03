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

class GemmContext : public MultiThreadGemmContext {};

// Computes a general matrix product ("GEMM").
// The meaning of the offsets, result_mult_int and result_shift
// parameters is the same as in the standard EightBitIntGemm interface
// (which is also implemented in the eight_bit_int_gemm directory).
template <typename Scalar, BitDepthSetting BitDepth, MapOrder LhsOrder,
          MapOrder RhsOrder, MapOrder ResultOrder>
void Gemm(GemmContext* context, const MatrixMap<const Scalar, LhsOrder>& lhs,
          const MatrixMap<const Scalar, RhsOrder>& rhs,
          MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
          int rhs_offset, int result_offset, int result_mult_int,
          int result_shift) {
  if (rhs.cols() == 1) {
    MultiThreadGemm<typename DefaultKernelForGemv<BitDepth>::Format,
                    std::uint8_t, BitDepth>(
        context, DefaultKernelForGemv<BitDepth>(), lhs, rhs, result, lhs_offset,
        rhs_offset, result_offset, result_mult_int, result_shift);
  } else {
    MultiThreadGemm<typename DefaultKernelForGemm<BitDepth>::Format,
                    std::uint8_t, BitDepth>(
        context, DefaultKernelForGemm<BitDepth>(), lhs, rhs, result, lhs_offset,
        rhs_offset, result_offset, result_mult_int, result_shift);
  }
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_GEMMLOWP_H_
