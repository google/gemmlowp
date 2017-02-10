// Copyright 2017 The Gemmlowp Authors. All Rights Reserved.
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

// dispatch_gemm_shape.h: dispatch GEMM calls according to their shape

#ifndef GEMMLOWP_INTERNAL_DISPATCH_GEMM_SHAPE_H_
#define GEMMLOWP_INTERNAL_DISPATCH_GEMM_SHAPE_H_

#include "multi_thread_gemm.h"

namespace gemmlowp {

template <typename InputScalar, typename OutputScalar, typename BitDepthParams,
          MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder,
          typename LhsOffset, typename RhsOffset, typename OutputPipelineType,
          typename GemmContextType>
void DispatchGemmShape(GemmContextType* context,
                       const MatrixMap<const InputScalar, LhsOrder>& lhs,
                       const MatrixMap<const InputScalar, RhsOrder>& rhs,
                       MatrixMap<OutputScalar, ResultOrder>* result,
                       const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                       const OutputPipelineType& output_pipeline) {
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
    typedef DefaultKernel<KernelFamily::Gemv, BitDepthParams> Kernel;
    MultiThreadGemm<typename Kernel::Format, InputScalar, OutputScalar,
                    BitDepthParams>(context, Kernel(), lhs, rhs, result,
                                    lhs_offset, rhs_offset, output_pipeline);
  } else {
    typedef DefaultKernel<KernelFamily::Gemm, BitDepthParams> Kernel;
    MultiThreadGemm<typename Kernel::Format, InputScalar, OutputScalar,
                    BitDepthParams>(context, Kernel(), lhs, rhs, result,
                                    lhs_offset, rhs_offset, output_pipeline);
  }
}

}  // end namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_DISPATCH_GEMM_SHAPE_H_
