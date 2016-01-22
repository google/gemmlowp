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

// multi_thread_gemm.h: Entry point to the multithreaded version of the
// generated (meta) gemm library.

#ifndef GEMMLOWP_META_MULTI_THREAD_GEMM_H_
#define GEMMLOWP_META_MULTI_THREAD_GEMM_H_

#ifdef GEMMLOWP_NEON_32

#include "multi_thread_common.h"
#include "single_thread_gemm.h"

namespace gemmlowp {
namespace meta {
namespace internal {

const std::int32_t kMaxCacheFriendlySize = 24 * 1024;

template <typename IN_TYPE, typename OUT_TYPE, typename F>
void CacheFriendlyMatrixMatrix(std::uint8_t* scratch, const IN_TYPE* lhs,
                               const IN_TYPE* rhs, std::int32_t m,
                               std::int32_t n, std::int32_t k, OUT_TYPE* result,
                               std::int32_t result_stride, const F& operation) {
  const std::int32_t rhs_size = n * k * sizeof(IN_TYPE);
  if (rhs_size > kMaxCacheFriendlySize) {
    const std::int32_t optimal_n =
        std::max(1, 3 * (kMaxCacheFriendlySize / (k * 3)));
    const std::int32_t chunks_count_less_one = n / optimal_n - 1;
    const std::int32_t chunk_size = optimal_n * k;
    for (int i = 0; i < chunks_count_less_one; ++i) {
      operation.ExecuteCacheFriendlyMatrixMatrix(
          scratch, lhs, rhs + i * chunk_size, m, optimal_n, k,
          result + i * optimal_n, result_stride);
    }
    const std::int32_t n_left = n - chunks_count_less_one * optimal_n;
    operation.ExecuteCacheFriendlyMatrixMatrix(
        scratch, lhs, rhs + chunks_count_less_one * chunk_size, m, n_left, k,
        result + chunks_count_less_one * optimal_n, result_stride);
  } else {
    operation.ExecuteCacheFriendlyMatrixMatrix(scratch, lhs, rhs, m, n, k,
                                               result, result_stride);
  }
}

class GemmQuantized8BitOperation {
 public:
  GemmQuantized8BitOperation(std::int32_t lhs_offset, std::int32_t rhs_offset,
                             std::int32_t sum_offset, std::int32_t multiplier,
                             std::int32_t shift)
      : lhs_offset(lhs_offset),
        rhs_offset(rhs_offset),
        sum_offset(sum_offset),
        multiplier(multiplier),
        shift(shift) {}

  void ExecuteMatrixMatrix(std::uint8_t* scratch, const std::uint8_t* lhs,
                           const std::uint8_t* rhs, std::int32_t m,
                           std::int32_t n, std::int32_t k, std::uint8_t* result,
                           std::int32_t result_stride) const {
    CacheFriendlyMatrixMatrix(scratch, lhs, rhs, m, n, k, result, result_stride,
                              *this);
  }

  void ExecuteCacheFriendlyMatrixMatrix(std::uint8_t* scratch,
                                        const std::uint8_t* lhs,
                                        const std::uint8_t* rhs, std::int32_t m,
                                        std::int32_t n, std::int32_t k,
                                        std::uint8_t* result,
                                        std::int32_t result_stride) const {
    gemm_q8_strided(scratch, lhs, rhs, m, n, k, lhs_offset, rhs_offset,
                    sum_offset, multiplier, shift, result, result_stride);
  }

  static std::int32_t ScratchPerThread(std::int32_t m, std::int32_t n,
                                       std::int32_t k) {
    return 128 * 1024;
  }

 private:
  std::int32_t lhs_offset;
  std::int32_t rhs_offset;
  std::int32_t sum_offset;
  std::int32_t multiplier;
  std::int32_t shift;
};

class GemmFloatOperation {
 public:
  GemmFloatOperation(std::int32_t lhs_offset, std::int32_t rhs_offset,
                     float result_offset)
      : lhs_offset(lhs_offset),
        rhs_offset(rhs_offset),
        result_offset(result_offset) {}

  void ExecuteMatrixMatrix(std::uint8_t* scratch, const std::uint8_t* lhs,
                           const std::uint8_t* rhs, std::int32_t m,
                           std::int32_t n, std::int32_t k, float* result,
                           std::int32_t result_stride) const {
    CacheFriendlyMatrixMatrix(scratch, lhs, rhs, m, n, k, result, result_stride,
                              *this);
  }

  void ExecuteCacheFriendlyMatrixMatrix(std::uint8_t* scratch,
                                        const std::uint8_t* lhs,
                                        const std::uint8_t* rhs, std::int32_t m,
                                        std::int32_t n, std::int32_t k,
                                        float* result,
                                        std::int32_t result_stride) const {
    gemm_f_strided(scratch, lhs, rhs, m, n, k, lhs_offset, rhs_offset,
                   result_offset, result, result_stride);
  }

  static std::int32_t ScratchPerThread(std::int32_t m, std::int32_t n,
                                       std::int32_t k) {
    return 128 * 1024;
  }

 private:
  std::int32_t lhs_offset;
  std::int32_t rhs_offset;
  float result_offset;
};

class GemmInt32Operation {
 public:
  GemmInt32Operation(std::int32_t lhs_offset, std::int32_t rhs_offset)
      : lhs_offset(lhs_offset), rhs_offset(rhs_offset) {}

  void ExecuteMatrixMatrix(std::uint8_t* scratch, const std::uint8_t* lhs,
                           const std::uint8_t* rhs, std::int32_t m,
                           std::int32_t n, std::int32_t k, std::int32_t* result,
                           std::int32_t result_stride) const {
    CacheFriendlyMatrixMatrix(scratch, lhs, rhs, m, n, k, result, result_stride,
                              *this);
  }

  void ExecuteCacheFriendlyMatrixMatrix(std::uint8_t* scratch,
                                        const std::uint8_t* lhs,
                                        const std::uint8_t* rhs, std::int32_t m,
                                        std::int32_t n, std::int32_t k,
                                        std::int32_t* result,
                                        std::int32_t result_stride) const {
    gemm_i32_strided(scratch, lhs, rhs, m, n, k, lhs_offset, rhs_offset, result,
                     result_stride);
  }

  static std::int32_t ScratchPerThread(std::int32_t m, std::int32_t n,
                                       std::int32_t k) {
    return 128 * 1024;
  }

 private:
  std::int32_t lhs_offset;
  std::int32_t rhs_offset;
};

}  // namespace internal

std::int32_t gemm_q8_scratch(std::int32_t m, std::int32_t n, std::int32_t k,
                             std::int32_t max_threads) {
  return internal::ResolveMaxThreads(max_threads) *
         internal::GemmQuantized8BitOperation::ScratchPerThread(m, n, k);
}

void multi_thread_gemm_q8(gemmlowp::WorkersPool* pool, std::int32_t max_threads,
                          std::uint8_t* scratch, const std::uint8_t* lhs,
                          const std::uint8_t* rhs, std::int32_t m,
                          std::int32_t n, std::int32_t k,
                          std::int32_t lhs_offset, std::int32_t rhs_offset,
                          std::int32_t sum_offset, std::int32_t multiplier,
                          std::int32_t shift, std::uint8_t* result) {
  internal::GemmQuantized8BitOperation operation(lhs_offset, rhs_offset,
                                                 sum_offset, multiplier, shift);
  internal::MultiThreadedMatrixMatrix(pool, max_threads, scratch, lhs, rhs, m,
                                      n, k, result, n, operation);
}

std::int32_t gemm_f_scratch(std::int32_t m, std::int32_t n, std::int32_t k,
                            std::int32_t max_threads) {
  return internal::ResolveMaxThreads(max_threads) *
         internal::GemmFloatOperation::ScratchPerThread(m, n, k);
}

void multi_thread_gemm_f(gemmlowp::WorkersPool* pool, std::int32_t max_threads,
                         std::uint8_t* scratch, const std::uint8_t* lhs,
                         const std::uint8_t* rhs, std::int32_t m,
                         std::int32_t n, std::int32_t k,
                         std::int32_t lhs_offset, std::int32_t rhs_offset,
                         float result_offset, float* result) {
  internal::GemmFloatOperation operation(lhs_offset, rhs_offset, result_offset);
  internal::MultiThreadedMatrixMatrix(pool, max_threads, scratch, lhs, rhs, m,
                                      n, k, result, n, operation);
}

std::int32_t gemm_i32_scratch(std::int32_t m, std::int32_t n, std::int32_t k,
                              std::int32_t max_threads) {
  return internal::ResolveMaxThreads(max_threads) *
         internal::GemmInt32Operation::ScratchPerThread(m, n, k);
}

void multi_thread_gemm_i32(gemmlowp::WorkersPool* pool,
                           std::int32_t max_threads, std::uint8_t* scratch,
                           const std::uint8_t* lhs, const std::uint8_t* rhs,
                           std::int32_t m, std::int32_t n, std::int32_t k,
                           std::int32_t lhs_offset, std::int32_t rhs_offset,
                           std::int32_t* result) {
  internal::GemmInt32Operation operation(lhs_offset, rhs_offset);
  internal::MultiThreadedMatrixMatrix(pool, max_threads, scratch, lhs, rhs, m,
                                      n, k, result, n, operation);
}

}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm fast-path requires GEMMLOWP_NEON_32!"
#endif

#endif  // GEMMLOWP_META_MULTI_THREAD_GEMM_H_
