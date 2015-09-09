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

// single_thread_gemm.h: Single-threaded GEMM implementation.
// This is a good place to start reading code, as it shows the overall
// structure of a GEMM and is much simpler than multi_thread_gemm.h.

#ifndef GEMMLOWP_INTERNAL_SINGLE_THREAD_GEMM_H_
#define GEMMLOWP_INTERNAL_SINGLE_THREAD_GEMM_H_

#include <cassert>

#include "../public/map.h"
#include "allocator.h"
#include "pack.h"
#include "unpack.h"
#include "compute.h"
#include "kernel.h"

namespace gemmlowp {

class SingleThreadGemmContext {
 public:
  Allocator* allocator() { return &allocator_; }

 protected:
  Allocator allocator_;
};

template <typename KernelFormat, typename Scalar, BitDepthSetting BitDepth,
          MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
void SingleThreadGemm(SingleThreadGemmContext* context,
                      const KernelBase& kernel,
                      const MatrixMap<const Scalar, LhsOrder>& lhs,
                      const MatrixMap<const Scalar, RhsOrder>& rhs,
                      MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
                      int rhs_offset, int result_offset, int result_mult_int,
                      int result_shift) {
  ScopedProfilingLabel label("gemmlowp::SingleThreadGemm");

  assert(lhs.cols() == rhs.rows());

  int rows = result->rows();
  int cols = result->cols();
  int depth = lhs.cols();

  Allocator* allocator = context->allocator();

  BlockParams block_params;
  block_params.Init<KernelFormat>(rows, cols, depth, 1);

  PackedSideBlock<typename KernelFormat::Lhs>
      packed_lhs(Side::Lhs, allocator, block_params, rhs_offset);
  PackedSideBlock<typename KernelFormat::Rhs>
      packed_rhs(Side::Rhs, allocator, block_params, lhs_offset);

  PackedResult packed_result(allocator, block_params);

  allocator->Commit();

  const bool pack_rhs_once = block_params.l2_cols == cols;

  if (pack_rhs_once) {
    PackRhs<BitDepth>(&packed_rhs, rhs);
  }

  for (int r = 0; r < rows; r += block_params.l2_rows) {
    int rs = std::min(block_params.l2_rows, rows - r);

    PackLhs<BitDepth>(&packed_lhs, lhs.block(r, 0, rs, depth));

    for (int c = 0; c < cols; c += block_params.l2_cols) {
      int cs = std::min(block_params.l2_cols, cols - c);

      if (!pack_rhs_once) {
        PackRhs<BitDepth>(&packed_rhs, rhs.block(0, c, depth, cs));
      }

      Compute(kernel, block_params, &packed_result, packed_lhs, packed_rhs);

      auto result_block = result->block(r, c, rs, cs);
      UnpackResult<BitDepth>(
        &result_block, packed_result, depth,
        packed_lhs.rank_one_update(), packed_rhs.rank_one_update(),
        lhs_offset, rhs_offset, result_offset, result_mult_int,
        result_shift);
    }
  }

  allocator->Decommit();
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_SINGLE_THREAD_GEMM_H_
