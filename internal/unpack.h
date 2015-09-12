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

#include <cmath>

namespace gemmlowp {

class PackedResult {
 public:
  PackedResult(Allocator* _allocator, const BlockParams& _block_params)
      : allocator_(_allocator), block_params_(_block_params) {
    matrix_handle_ = allocator_->Reserve<std::int32_t>(block_params_.l2_rows *
                                                       block_params_.l2_cols);
  }

  ~PackedResult() {}

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

 private:
  Allocator* allocator_;
  Allocator::Handle matrix_handle_;
  const BlockParams& block_params_;
};

template <std::uint32_t numerator, std::uint32_t denominator>
std::int32_t MultiplyByConstantFraction(std::int32_t x) {
  if (numerator == denominator) {
    return x;
  }

  // We'll use only signed arithmetic here. This is
  // simpler (since this function operates on signed int32's) and
  // more friendly to ARM NEON, where this allows us to use the
  // VQRDMULH instruction.
  static const std::int32_t int_quotient =
      (numerator + denominator / 2) / denominator;
  static const std::int32_t remaining_numerator =
      numerator - int_quotient * denominator;
  static const std::int32_t scaled_remaining_numerator =
      static_cast<std::int32_t>(
          (static_cast<std::int64_t>(remaining_numerator) << 31) / denominator);

  const std::int64_t scaled_remaining_product =
      static_cast<std::int64_t>(x) *
      static_cast<std::int64_t>(scaled_remaining_numerator);

  const std::int32_t scaled_remaining_product_nudge =
      (scaled_remaining_product > 0 ? 1 : -1) * (1 << 30);

  const std::int32_t remaining_product =
      (scaled_remaining_product + scaled_remaining_product_nudge) / (1u << 31);

  return x * int_quotient + remaining_product;
}

template <BitDepthSetting BitDepth, 
  typename ResultBlockType, typename PackedResultType>
struct UnpackResultImplGeneric {
  static void Unpack(ResultBlockType* dst, const PackedResultType& src,
                     int depth, const std::int32_t* lhs_rank_one_update,
                     const std::int32_t* rhs_rank_one_update,
                     std::int32_t lhs_offset, std::int32_t rhs_offset,
                     std::int32_t result_offset, std::int32_t result_mult_int,
                     std::int32_t result_shift) {
    std::int32_t term_11 = lhs_offset * rhs_offset * depth + result_offset;
    auto src_map = src.Map();
    // No top-level blocking in the depth dimension at the moment.
    // Too much loss of precision.
    const int kLhsBits = LhsBitDepth<BitDepth>::kBits;
    const int kRhsBits = RhsBitDepth<BitDepth>::kBits;
    const std::int32_t kLhsMax = (1 << kLhsBits) - 1;
    const std::int32_t kRhsMax = (1 << kRhsBits) - 1;
    for (int c = 0; c < dst->cols(); c++) {
      for (int r = 0; r < dst->rows(); r++) {
        std::int32_t raw_xx = src_map(r, c);
        std::int32_t raw_x1 = lhs_rank_one_update[r];
        std::int32_t raw_1x = rhs_rank_one_update[c];
        std::int32_t term_xx =
            MultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(raw_xx);
        std::int32_t term_x1 =
            MultiplyByConstantFraction<255, kLhsMax>(raw_x1);
        std::int32_t term_1x =
            MultiplyByConstantFraction<255, kRhsMax>(raw_1x);
        std::int32_t sum = term_xx + term_x1 + term_1x + term_11;
        std::int32_t result =
            (sum * result_mult_int + (1 << (result_shift - 1))) >> result_shift;
        (*dst)(r, c) = result > 255 ? 255 : result < 0 ? 0 : result;
      }
    }
  }
};

template <BitDepthSetting BitDepth, 
  typename ResultBlockType, typename PackedResultType>
struct UnpackResultImpl
    : UnpackResultImplGeneric<BitDepth, ResultBlockType, PackedResultType> {};

template <BitDepthSetting BitDepth, 
  typename ResultBlockType, typename PackedResultType>
void UnpackResult(ResultBlockType* dst, const PackedResultType& src, int depth,
                  const std::int32_t* lhs_rank_one_update,
                  const std::int32_t* rhs_rank_one_update,
                  std::int32_t lhs_offset, std::int32_t rhs_offset,
                  std::int32_t result_offset, std::int32_t result_mult_int,
                  std::int32_t result_shift) {
  ScopedProfilingLabel label("unpack");
  UnpackResultImpl<BitDepth, ResultBlockType, PackedResultType>::Unpack(
      dst, src, depth, lhs_rank_one_update, rhs_rank_one_update, lhs_offset,
      rhs_offset, result_offset, result_mult_int, result_shift);
}

}  // namespace gemmlowp

#ifdef GEMMLOWP_NEON
#include "unpack_neon.h"
#endif

#endif  // GEMMLOWP_INTERNAL_UNPACK_H_
