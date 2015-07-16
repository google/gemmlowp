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

#include "test.h"

#include <unistd.h>
#include <iostream>
#include <ctime>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <string>

#include "../public/gemmlowp.h"
#include "../internal/kernel_reference.h"
#include "../eight_bit_int_gemm/eight_bit_int_gemm.h"

namespace gemmlowp {

struct ReferenceEightBitIntGemmContext {
  ReferenceEightBitIntGemmContext()
      : saturated_0_values(0), saturated_255_values(0) {}

  int saturated_0_values, saturated_255_values;
};

void ReferenceEightBitIntGemm(ReferenceEightBitIntGemmContext* context,
                              bool transpose_a, bool transpose_b,
                              bool transpose_c, int m, int n, int k,
                              const uint8_t* a, int32_t a_offset, int lda,
                              const uint8_t* b, int32_t b_offset, int ldb,
                              uint8_t* c, int32_t c_offset, int32_t c_mult_int,
                              int32_t c_shift, int ldc) {
  context->saturated_0_values = 0;
  context->saturated_255_values = 0;
  assert((c_shift > 0) && (c_shift < 32));
  assert(a != nullptr);
  assert(b != nullptr);
  assert(c != nullptr);
  int a_i_stride;
  int a_l_stride;
  if (transpose_a == transpose_c) {
    a_i_stride = 1;
    a_l_stride = lda;
  } else {
    a_i_stride = lda;
    a_l_stride = 1;
  }
  int b_j_stride;
  int b_l_stride;
  if (transpose_b == transpose_c) {
    b_j_stride = ldb;
    b_l_stride = 1;
  } else {
    b_j_stride = 1;
    b_l_stride = ldb;
  }
  int c_i_stride;
  int c_j_stride;
  if (transpose_c) {
    c_i_stride = ldc;
    c_j_stride = 1;
  } else {
    c_i_stride = 1;
    c_j_stride = ldc;
  }
  int i, j, l;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      int32_t total = 0;
      for (l = 0; l < k; l++) {
        const int a_index = i * a_i_stride + l * a_l_stride;
        const uint8_t a_as_byte = a[a_index];
        const int32_t a_as_int = static_cast<int32_t>(a_as_byte) + a_offset;
        const int b_index = j * b_j_stride + l * b_l_stride;
        const uint8_t b_as_byte = b[b_index];
        const int32_t b_as_int = static_cast<int32_t>(b_as_byte) + b_offset;
        const int32_t mult_as_int = a_as_int * b_as_int;
        total += mult_as_int;
      }
      int32_t output =
          (((total + c_offset) * c_mult_int) + (1 << (c_shift - 1))) >> c_shift;
      if (output > 255) {
        output = 255;
        context->saturated_255_values++;
      }
      if (output < 0) {
        output = 0;
        context->saturated_0_values++;
      }
      const int c_index = i * c_i_stride + j * c_j_stride;
      c[c_index] = static_cast<uint8_t>(output);
    }
  }
}

// *GemmWrapper's allow to wrap various Gemm functions in a uniform
// interface, so we can use the same testing code to test all of them

template <typename Kernel, typename Scalar>
struct SingleThreadGemmWrapper {
  static const int kLhsBitDepth = Kernel::kLhsBitDepth;
  static const int kRhsBitDepth = Kernel::kRhsBitDepth;

  static const char* Name() {
    static char buf[256];
    snprintf(buf, sizeof(buf), "SingleThreadGemm, Kernel: %s", Kernel().Name());
    return buf;
  }

  typedef SingleThreadGemmContext Context;

  template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
  static void Gemm(Context* context,
                   const MatrixMap<const Scalar, LhsOrder>& lhs,
                   const MatrixMap<const Scalar, RhsOrder>& rhs,
                   MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
                   int rhs_offset, int result_offset, int result_mult_int,
                   int result_shift) {
    SingleThreadGemm<typename Kernel::Format, Scalar, LhsOrder, RhsOrder,
                     ResultOrder>(context, Kernel(), lhs, rhs, result,
                                  lhs_offset, rhs_offset, result_offset,
                                  result_mult_int, result_shift);
  }
};

template <typename Kernel, typename Scalar>
struct MultiThreadGemmWrapper {
  static const int kLhsBitDepth = Kernel::kLhsBitDepth;
  static const int kRhsBitDepth = Kernel::kRhsBitDepth;

  static const char* Name() {
    static char buf[256];
    snprintf(buf, sizeof(buf), "MultiThreadGemm, Kernel: %s", Kernel().Name());
    return buf;
  }

  typedef MultiThreadGemmContext Context;

  template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
  static void Gemm(Context* context,
                   const MatrixMap<const Scalar, LhsOrder>& lhs,
                   const MatrixMap<const Scalar, RhsOrder>& rhs,
                   MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
                   int rhs_offset, int result_offset, int result_mult_int,
                   int result_shift) {
    MultiThreadGemm<typename Kernel::Format, Scalar, LhsOrder, RhsOrder,
                    ResultOrder>(context, Kernel(), lhs, rhs, result,
                                 lhs_offset, rhs_offset, result_offset,
                                 result_mult_int, result_shift);
  }
};

template <typename Scalar>
struct PublicGemmWrapper {
  static const int kLhsBitDepth = 8;
  static const int kRhsBitDepth = 8;

  static const char* Name() { return "public Gemm"; }

  typedef GemmContext Context;

  template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
  static void Gemm(Context* context,
                   const MatrixMap<const Scalar, LhsOrder>& lhs,
                   const MatrixMap<const Scalar, RhsOrder>& rhs,
                   MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
                   int rhs_offset, int result_offset, int result_mult_int,
                   int result_shift) {
    gemmlowp::Gemm<uint8_t, LhsOrder, RhsOrder, ResultOrder>(
        context, lhs, rhs, result, lhs_offset, rhs_offset, result_offset,
        result_mult_int, result_shift);
  }
};

template <typename Scalar>
struct EightBitIntGemmWrapper {
  static const int kLhsBitDepth = 8;
  static const int kRhsBitDepth = 8;

  static const char* Name() { return "EightBitIntGemm"; }

  typedef void Context;

  template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
  static void Gemm(Context*, const MatrixMap<const Scalar, LhsOrder>& lhs,
                   const MatrixMap<const Scalar, RhsOrder>& rhs,
                   MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
                   int rhs_offset, int result_offset, int result_mult_int,
                   int result_shift) {
    const bool transpose_c = ResultOrder == MapOrder::ColMajor;
    const bool transpose_a =
        RhsOrder == MapOrder::RowMajor ? transpose_c : !transpose_c;
    const bool transpose_b =
        LhsOrder == MapOrder::RowMajor ? transpose_c : !transpose_c;

    eight_bit_int_gemm::EightBitIntGemm(
        transpose_a, transpose_b, transpose_c, rhs.cols(), lhs.rows(),
        lhs.cols(), rhs.data(), rhs_offset, rhs.stride(), lhs.data(),
        lhs_offset, lhs.stride(), result->data(), result_offset,
        result_mult_int, result_shift, result->stride());
  }
};

template <typename Scalar>
struct ReferenceEightBitIntGemmWrapper {
  static const int kLhsBitDepth = 8;
  static const int kRhsBitDepth = 8;

  static const char* Name() { return "ReferenceEightBitIntGemm"; }

  typedef ReferenceEightBitIntGemmContext Context;

  template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
  static void Gemm(Context* context, bool transpose_a, bool transpose_b,
                   bool transpose_c,
                   const MatrixMap<const Scalar, LhsOrder>& lhs,
                   const MatrixMap<const Scalar, RhsOrder>& rhs,
                   MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
                   int rhs_offset, int result_offset, int result_mult_int,
                   int result_shift) {
    ReferenceEightBitIntGemm(context, transpose_a, transpose_b, transpose_c,
                             rhs.cols(), lhs.rows(), lhs.cols(), rhs.data(),
                             rhs_offset, rhs.stride(), lhs.data(), lhs_offset,
                             lhs.stride(), result->data(), result_offset,
                             result_mult_int, result_shift, result->stride());
  }
};

const char* OrderName(MapOrder order) {
  return order == MapOrder::ColMajor ? "ColMajor" : "RowMajor";
}

// Our approach to choosing result_shift values for testing, is bisection.
// This function takes an interval, [result_shift_min .. result_shift_max].
// If too much saturation occurred in either direction, it bisects accordingly,
// recursing until the interval contains only one value.
// The primary reason why we prefer this over computing optimal shift values,
// is that we actually want to exercise some saturation, as there is nontrivial
// code handling that in gemmlowp.
// Secondarily, this is faster than computing optimal shifts, since in 90% of
// cases the first-tried shift value 16 turns out to be good enough.
template <typename GemmWrapper, typename LhsType, typename RhsType,
          typename ResultType>
void test_gemm_impl(typename GemmWrapper::Context* context, const LhsType& lhs,
                    const RhsType& rhs, ResultType* result, int lhs_offset,
                    int rhs_offset, int result_offset, int result_mult_int,
                    int result_shift_min, int result_shift_max) {
  const int rows = lhs.rows();
  const int cols = rhs.cols();
  Check(lhs.cols() == rhs.rows());
  const int depth = lhs.cols();

  const int result_shift = (result_shift_min + result_shift_max) / 2;

  GemmWrapper::Gemm(context, lhs.const_map(), rhs.const_map(), &result->map(),
                    lhs_offset, rhs_offset, result_offset, result_mult_int,
                    result_shift);

  typedef typename ResultType::Scalar Scalar;
  static const MapOrder kLhsOrder = LhsType::kOrder;
  static const MapOrder kRhsOrder = RhsType::kOrder;
  static const MapOrder kResultOrder = ResultType::kOrder;
  ResultType ref_result(rows, cols);
  ReferenceEightBitIntGemmContext reference_context;
  const bool transpose_c = kResultOrder == MapOrder::ColMajor;
  const bool transpose_a =
      kRhsOrder == MapOrder::RowMajor ? transpose_c : !transpose_c;
  const bool transpose_b =
      kLhsOrder == MapOrder::RowMajor ? transpose_c : !transpose_c;
  printf("ta=%d tb=%d tc=%d\n", int(transpose_a), int(transpose_b),
         int(transpose_c));
  ReferenceEightBitIntGemmWrapper<Scalar>::Gemm(
      &reference_context, transpose_a, transpose_b, transpose_c,
      lhs.const_map(), rhs.const_map(), &ref_result.map(), lhs_offset,
      rhs_offset, result_offset, result_mult_int, result_shift);

  const bool good = *result == ref_result;
  printf(
      "%s: %dx%dx%d %s x %s -> %s, %s, offsets %d/%d/%d, mult %d, shift %d\n",
      good ? "PASS" : "FAIL", rows, depth, cols, OrderName(kLhsOrder),
      OrderName(kRhsOrder), OrderName(kResultOrder), GemmWrapper::Name(),
      lhs_offset, rhs_offset, result_offset, result_mult_int, result_shift);

  if (!good) {
    int maxdiff = 0;
    int countdiff = 0;
    for (int c = 0; c < result->cols(); c++) {
      for (int r = 0; r < result->rows(); r++) {
        int a = (*result)(r, c);
        int b = ref_result(r, c);
        if (a != b) {
          countdiff++;
          maxdiff = std::max(maxdiff, std::abs(a - b));
        }
      }
    }
    printf("max difference: %d\n", maxdiff);
    printf("number of different places: %d\n", countdiff);
    int bad_coeffs_printed = 0;
    for (int c = 0; c < result->cols() && bad_coeffs_printed < 20; c++) {
      for (int r = 0; r < result->rows() && bad_coeffs_printed < 20; r++) {
        if (ref_result(r, c) != (*result)(r, c)) {
          printf("bad coeff: at (%d, %d), expected %d, got %d\n", r, c,
                 ref_result(r, c), (*result)(r, c));
          bad_coeffs_printed++;
        }
      }
    }
  }

  Check(good);

  if (result_shift_min != result_shift_max) {
    const int max_allowed_saturated_values = result->size() / 16;

    int new_result_shift_min = result_shift_min;
    int new_result_shift_max = result_shift_max;
    bool retry = false;

    if (reference_context.saturated_0_values > max_allowed_saturated_values) {
      new_result_shift_max = (result_shift_min + result_shift_max) / 2;
      retry = true;
    }

    if (reference_context.saturated_255_values > max_allowed_saturated_values) {
      new_result_shift_min = (result_shift_min + result_shift_max) / 2;
      retry = true;
    }

    if (retry) {
      test_gemm_impl<GemmWrapper>(context, lhs, rhs, result, lhs_offset,
                                  rhs_offset, result_offset, result_mult_int,
                                  new_result_shift_min, new_result_shift_max);
    }
  }
}

template <typename GemmWrapper, typename LhsType, typename RhsType,
          typename ResultType>
void test_gemm(typename GemmWrapper::Context* context, const LhsType& lhs,
               const RhsType& rhs, ResultType* result, int lhs_offset,
               int rhs_offset, int result_offset, int result_mult_int) {
  test_gemm_impl<GemmWrapper>(context, lhs, rhs, result, lhs_offset, rhs_offset,
                              result_offset, result_mult_int, 0, 32);
}

enum class WhatParamsToTest {
  All,
  OnlyGenericCase,
};

template <typename GemmWrapper, MapOrder LhsOrder, MapOrder RhsOrder,
          MapOrder ResultOrder>
void test_gemm(typename GemmWrapper::Context* context, int rows, int depth,
               int cols, WhatParamsToTest params_to_test) {
  typedef std::uint8_t Scalar;
  typedef Matrix<Scalar, LhsOrder> LhsType;
  LhsType lhs(rows, depth);
  MakeRandom(&lhs, GemmWrapper::kLhsBitDepth);
  typedef Matrix<Scalar, RhsOrder> RhsType;
  RhsType rhs(depth, cols);
  MakeRandom(&rhs, GemmWrapper::kRhsBitDepth);
  typedef Matrix<Scalar, ResultOrder> ResultType;
  ResultType result(rows, cols);
  MakeZero(&result);

  if (params_to_test == WhatParamsToTest::All) {
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 0, 0, 0, 1);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 10, 0, 0, 1);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 0, 10, 0, 1);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 0, 0, 10, 1);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 0, 0, 0, 10);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 10, 10, 10, 10);
    test_gemm<GemmWrapper>(context, lhs, rhs, &result, 256, 1, 17, 4);
  }
  test_gemm<GemmWrapper>(context, lhs, rhs, &result, -75, -91, 74980, 123);
}

enum class WhatOrdersToTest { All, OnlyRCC };

template <typename GemmWrapper>
void test_gemm(typename GemmWrapper::Context* context, int rows, int depth,
               int cols, WhatParamsToTest params_to_test,
               WhatOrdersToTest orders_to_test) {
#define GEMMLOWP_ONE_TEST(LhsOrder, RhsOrder, ResultOrder)         \
  do {                                                             \
    test_gemm<GemmWrapper, MapOrder::LhsOrder, MapOrder::RhsOrder, \
              MapOrder::ResultOrder>(context, rows, depth, cols,   \
                                     params_to_test);              \
  } while (false)

  if (orders_to_test == WhatOrdersToTest::All) {
    GEMMLOWP_ONE_TEST(ColMajor, ColMajor, ColMajor);
    GEMMLOWP_ONE_TEST(RowMajor, ColMajor, ColMajor);
    GEMMLOWP_ONE_TEST(ColMajor, RowMajor, ColMajor);
    GEMMLOWP_ONE_TEST(RowMajor, RowMajor, ColMajor);

    GEMMLOWP_ONE_TEST(ColMajor, ColMajor, RowMajor);
    GEMMLOWP_ONE_TEST(RowMajor, ColMajor, RowMajor);
    GEMMLOWP_ONE_TEST(ColMajor, RowMajor, RowMajor);
    GEMMLOWP_ONE_TEST(RowMajor, RowMajor, RowMajor);
  } else {
    GEMMLOWP_ONE_TEST(RowMajor, ColMajor, ColMajor);
  }

#undef GEMMLOWP_ONE_TEST
}

template <typename Kernel>
void test_gemm_kernel(MultiThreadGemmContext* context) {
  typedef MultiThreadGemmWrapper<Kernel, uint8_t> GemmWrapper;
  test_gemm<GemmWrapper>(context, 1, 1, 1, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 2, 2, 2, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 3, 3, 3, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 4, 4, 4, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 5, 5, 5, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 9, 11, 13, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 50, 50, 50, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 500, 500, 500,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::All);
  test_gemm<GemmWrapper>(context, 100, 5000, 100,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
}

template <typename GemmWrapper>
void test_gemm(typename GemmWrapper::Context* context) {
  test_gemm<GemmWrapper>(context, 1, 1, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 2, 1, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1, 2, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1, 1, 2, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 2, 2, 2, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 3, 3, 3, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 4, 4, 4, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 5, 5, 5, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 6, 6, 6, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 3, 5, 7, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 7, 3, 5, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 5, 7, 3, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 8, 8, 8, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);

  test_gemm<GemmWrapper>(context, 16, 16, 16, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 32, 32, 32, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 64, 64, 64, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 128, 128, 128, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);

  test_gemm<GemmWrapper>(context, 17, 24, 31, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 37, 55, 73, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 57, 87, 117, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 93, 83, 73, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 109, 89, 99, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 78, 101, 82, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);

  test_gemm<GemmWrapper>(context, 512, 512, 512,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1024, 1024, 1024,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 567, 2345, 123,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 100, 5000, 100,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1, 1, 1000, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1000, 1, 1, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1, 1000, 1, WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1, 1000, 1000,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1000, 1, 1000,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 1000, 1000, 1,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 777, 3456, 1,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 4567, 555, 1,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::OnlyRCC);

  // Test all storage orders
  test_gemm<GemmWrapper>(context, 70, 90, 110, WhatParamsToTest::All,
                         WhatOrdersToTest::All);
  test_gemm<GemmWrapper>(context, 300, 400, 500,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::All);
}

template <typename GemmWrapper>
void test_gemv(typename GemmWrapper::Context* context) {
  test_gemm<GemmWrapper>(context, 2, 2, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 3, 3, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 4, 4, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 5, 5, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 6, 6, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 3, 5, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 7, 3, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 5, 7, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 8, 8, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 32, 32, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 128, 128, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);
  test_gemm<GemmWrapper>(context, 321, 123, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::OnlyRCC);

  // Test all storage orders
  test_gemm<GemmWrapper>(context, 70, 90, 1, WhatParamsToTest::All,
                         WhatOrdersToTest::All);
  test_gemm<GemmWrapper>(context, 300, 400, 1,
                         WhatParamsToTest::OnlyGenericCase,
                         WhatOrdersToTest::All);
}

void test() {
#ifdef GEMMLOWP_TEST_PROFILE
  RegisterCurrentThreadForProfiling();
  StartProfiling();
#endif

  GemmContext context;

  // Test the internal GEMM interfaces
  test_gemm<SingleThreadGemmWrapper<DefaultKernelForGEMM, uint8_t>>(&context);

  test_gemm<MultiThreadGemmWrapper<DefaultKernelForGEMM, uint8_t>>(&context);

  // Test the public GEMM interfaces
  test_gemm<PublicGemmWrapper<uint8_t>>(&context);

  test_gemm<EightBitIntGemmWrapper<uint8_t>>(&context);

  // Test GEMV cases (internal interfaces)
  test_gemv<SingleThreadGemmWrapper<DefaultKernelForGEMV, uint8_t>>(&context);

  test_gemv<MultiThreadGemmWrapper<DefaultKernelForGEMV, uint8_t>>(&context);

  // Test GEMV cases (public interfaces)
  test_gemv<PublicGemmWrapper<uint8_t>>(&context);

  test_gemv<EightBitIntGemmWrapper<uint8_t>>(&context);

  // Test specific kernels with various different formats,
  // to exercises corner cases especially in the packing code.
  test_gemm_kernel<
      ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<1, 1>, 1>,
                                   KernelSideFormat<CellFormat<1, 1>, 1>>>>(
      &context);

  test_gemm_kernel<
      ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<3, 4>, 2>,
                                   KernelSideFormat<CellFormat<5, 4>, 3>>>>(
      &context);

  test_gemm_kernel<
      ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<5, 3>, 3>,
                                   KernelSideFormat<CellFormat<4, 3>, 2>>>>(
      &context);

  test_gemm_kernel<
      ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<4, 3>, 3>,
                                   KernelSideFormat<CellFormat<4, 3>, 1>>>>(
      &context);

  test_gemm_kernel<
      ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<4, 3>, 3>,
                                   KernelSideFormat<CellFormat<2, 3>, 2>>>>(
      &context);

// Test all our optimized kernels, even if they are not used
// at the moment, as they might be handy later and so it's
// useful to keep them functional for now.
#ifdef GEMMLOWP_NEON32
  test_gemm_kernel<gemmlowp::NEON32Kernel12x4Depth2>(&context);
  test_gemm_kernel<gemmlowp::NEON32Kernel20x1Depth4>(&context);
  test_gemm_kernel<gemmlowp::NEON32Kernel8x1Depth4>(&context);
#endif

#ifdef GEMMLOWP_NEON64
  test_gemm_kernel<gemmlowp::NEON64Kernel12x4Depth2>(&context);
#endif


#ifdef GEMMLOWP_TEST_PROFILE
  FinishProfiling();
#endif

  std::cerr << "All tests passed." << std::endl;

  // We have been testing the eight_bit_int_gemm, so we should free its
  // persistent
  // resources now to avoid having leak-checking tools report leaks.
  eight_bit_int_gemm::FreePersistentResources();
}

}  // end namespace gemmlowp

int main() { gemmlowp::test(); }
