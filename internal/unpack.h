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

// unpack.h: unpacking the result blocks computed by compute.h,
// storing them into the destination matrix.

#ifndef GEMMLOWP_INTERNAL_UNPACK_H_
#define GEMMLOWP_INTERNAL_UNPACK_H_

#include "allocator.h"
#include "block_params.h"
#include "pack.h"

namespace gemmlowp {

class PackedResultInt32 {
  Allocator* allocator_;
  Allocator::Handle matrix_handle_;
  const BlockParams& block_params_;

 public:
  PackedResultInt32(Allocator* _allocator, const BlockParams& _block_params)
      : allocator_(_allocator), block_params_(_block_params) {
    matrix_handle_ = allocator_->Reserve<std::int32_t>(block_params_.l2_rows *
                                                       block_params_.l2_cols);
  }

  ~PackedResultInt32() {}

  MatrixMap<std::int32_t, MapOrder::ColMajor> Map() {
    return MatrixMap<std::int32_t, MapOrder::ColMajor>(
        allocator_->GetPointer<std::int32_t>(matrix_handle_),
        block_params_.l2_rows, block_params_.l2_cols, block_params_.l2_rows);
  }

  MatrixMap<const std::int32_t, MapOrder::ColMajor> Map() const {
    return MatrixMap<const std::int32_t, MapOrder::ColMajor>(
        allocator_->GetPointer<const std::int32_t>(matrix_handle_),
        block_params_.l2_rows, block_params_.l2_cols, block_params_.l2_rows);
  }
};

template <typename ResultBlockType>
struct UnpackResultImplGeneric {
  static void Unpack(ResultBlockType* dst, const PackedResultInt32& src,
                     int depth, const std::int32_t* lhs_rank_one_update,
                     const std::int32_t* rhs_rank_one_update,
                     std::int32_t lhs_offset, std::int32_t rhs_offset,
                     std::int32_t result_offset, std::int32_t result_mult_int,
                     std::int32_t result_shift) {
    std::int32_t rank0update = lhs_offset * rhs_offset * depth;
    auto src_map = src.Map();
    // No top-level blocking in the depth dimension at the moment.
    // Too much loss of precision.
    for (int c = 0; c < dst->cols(); c++) {
      for (int r = 0; r < dst->rows(); r++) {
        std::int32_t q = *src_map.data(r, c);
        q += lhs_rank_one_update[r] + rhs_rank_one_update[c] + rank0update;
        q = ((q + result_offset) * result_mult_int +
             (1 << (result_shift - 1))) >>
            result_shift;
        (*dst)(r, c) = q > 255 ? 255 : q < 0 ? 0 : q;
      }
    }
  }
};

template <typename ResultBlockType>
struct UnpackResultImpl : UnpackResultImplGeneric<ResultBlockType> {};

template <typename ResultBlockType>
void UnpackResult(ResultBlockType* dst, const PackedResultInt32& src, int depth,
                  const std::int32_t* lhs_rank_one_update,
                  const std::int32_t* rhs_rank_one_update,
                  std::int32_t lhs_offset, std::int32_t rhs_offset,
                  std::int32_t result_offset, std::int32_t result_mult_int,
                  std::int32_t result_shift) {
  ScopedProfilingLabel label("unpack");
  UnpackResultImpl<ResultBlockType>::Unpack(
      dst, src, depth, lhs_rank_one_update, rhs_rank_one_update, lhs_offset,
      rhs_offset, result_offset, result_mult_int, result_shift);
}

}  // namespace gemmlowp

#ifdef GEMMLOWP_NEON
#include "unpack_neon.h"
#endif

#endif  // GEMMLOWP_INTERNAL_UNPACK_H_
