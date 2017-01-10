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

// test_fixedpoint.cc: unit tests covering the fixedpoint/ directory.

#define GEMMLOWP_ENABLE_FIXEDPOINT_CONSTANTS_CHECKS

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include "test.h"

#include "../fixedpoint/fixedpoint.h"

namespace gemmlowp {

namespace {

// Explanation of SimdVector type and associated functions
// (LoadSimdVector, StoreSimdVector):
// The fixedpoint stuff being tested here is generic in an underlying
// integer type which may be either scalar (int32_t) or SIMD (e.g.
// NEON int32x4_t). We want to write uniform tests that can test
// both the scalar and SIMD paths. We achieve this by having this
// generic SimdVector abstraction, local to this test.

#ifdef GEMMLOWP_NEON
using SimdVector = int32x4_t;
constexpr std::size_t SimdVectorSize = 4;
SimdVector LoadSimdVector(const std::int32_t* src) { return vld1q_s32(src); }
void StoreSimdVector(std::int32_t* dst, SimdVector v) { vst1q_s32(dst, v); }
#elif defined(GEMMLOWP_SSE4)
using SimdVector = __m128i;
constexpr std::size_t SimdVectorSize = 4;
SimdVector LoadSimdVector(const std::int32_t* src) {
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
}
void StoreSimdVector(std::int32_t* dst, SimdVector v) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), v);
}
#else
using SimdVector = std::int32_t;
constexpr std::size_t SimdVectorSize = 1;
SimdVector LoadSimdVector(const std::int32_t* src) { return *src; }
void StoreSimdVector(std::int32_t* dst, SimdVector v) { *dst = v; }
#endif

// Explanation of UnaryOpBase, its *Op subclasses below, and TestUnaryOp:
// Most (though not all) of the fixedpoint functionality being tested
// consists of functions taking one fixedpoint value and returning one
// fixedpoint value, e.g. "exp" or "tanh". We call them "unary operators".
// We factor a lot of testing boilerplate into a common TestUnaryOp function
// taking a "unary op" object that fully describes the function to be tested.
// These objects inherit UnaryOpBase mostly as a means to share some default
// values for some properties.
//
// An important design element here is that the fixed-point values are passed
// around as raw integers (e.g. int32_t or SIMD types such as int32x4_t), not
// as higher-level FixedPoint objects. The motivation for this design is 1) to
// avoid having to templatize everything in the tIntegerBits parameter of
// class FixedPoint, and 2) to allow directly testing low-level functions
// operating on raw types (e.g. RoundingDivideByPOT) without needlessly
// requiring
// wrapping raw values in FixedPoint objects.
class UnaryOpBase {
 public:
  // Min bound of the input range of this op. For example, an op only handling
  // nonnegative values would return 0.
  std::int32_t MinInput() const {
    return std::numeric_limits<std::int32_t>::min();
  }
  // Max bound of the input range of this op. For example, an op only handling
  // nonpositive values would return 0.
  std::int32_t MaxInput() const {
    return std::numeric_limits<std::int32_t>::max();
  }
  // Tolerated difference between actual and reference int32 values.
  // Note that the corresponding real-numbers tolerance depends on the number
  // of integer bits of the fixed-point representation of the results of this
  // op.
  // For example, for an op returning fixed-point values with 0 integer bits,
  // the correspondence between real-number values and raw values is
  // real_number = (2^31) * raw_value.
  std::int32_t Tolerance() const { return 0; }
};

// Op wrapping RoundingDivideByPOT
class RoundingDivideByPOTOp final : public UnaryOpBase {
 public:
  RoundingDivideByPOTOp(int exponent) : exponent_(exponent) {}
  std::int32_t ReferenceOp(std::int32_t x) const {
    const double d = static_cast<double>(x) / (1ll << exponent_);
    return static_cast<std::int32_t>(std::round(d));
  }
  template <typename tRawType>
  tRawType Op(tRawType x) const {
    return RoundingDivideByPOT(x, exponent_);
  }

 private:
  const int exponent_;
};

// Op wrapping SaturatingRoundingMultiplyByPOT
template <int tExponent>
class SaturatingRoundingMultiplyByPOTOp final : public UnaryOpBase {
 public:
  std::int32_t ReferenceOp(std::int32_t x) const {
    const double d = static_cast<double>(x) * std::pow(2., tExponent);
    const double clamp_min = std::numeric_limits<std::int32_t>::min();
    const double clamp_max = std::numeric_limits<std::int32_t>::max();
    const double clamped = std::min(clamp_max, std::max(clamp_min, d));
    return static_cast<std::int32_t>(std::round(clamped));
  }
  template <typename tRawType>
  tRawType Op(tRawType x) const {
    return SaturatingRoundingMultiplyByPOT<tExponent>(x);
  }
};

// Op wrapping exp_on_interval_between_negative_one_quarter_and_0_excl
class ExpOnIntervalBetweenNegativeOneQuarterAnd0ExclOp final
    : public UnaryOpBase {
 public:
  std::int32_t MinInput() const { return -(1 << 29); }
  std::int32_t MaxInput() const { return 0; }
  std::int32_t Tolerance() const { return 500; }
  std::int32_t ReferenceOp(std::int32_t x) const {
    using F = FixedPoint<std::int32_t, 0>;
    const double d = ToDouble(F::FromRaw(x));
    const double e = std::exp(d);
    return F::FromDouble(e).raw();
  }
  template <typename tRawType>
  tRawType Op(tRawType x) const {
    using F = FixedPoint<tRawType, 0>;
    const F f = F::FromRaw(x);
    const F e = exp_on_interval_between_negative_one_quarter_and_0_excl(f);
    return e.raw();
  }
};

// Op wrapping exp_on_negative_values
template <int tIntegerBits>
class ExpOnNegativeValuesOp final : public UnaryOpBase {
 public:
  std::int32_t MaxInput() const { return 0; }
  std::int32_t Tolerance() const { return 500; }
  std::int32_t ReferenceOp(std::int32_t x) const {
    using F = FixedPoint<std::int32_t, tIntegerBits>;
    using F0 = FixedPoint<std::int32_t, 0>;
    const double d = ToDouble(F::FromRaw(x));
    const double e = std::exp(d);
    return F0::FromDouble(e).raw();
  }
  template <typename tRawType>
  tRawType Op(tRawType x) const {
    using F = FixedPoint<tRawType, tIntegerBits>;
    const F f = F::FromRaw(x);
    return exp_on_negative_values(f).raw();
  }
};

// Op wrapping one_minus_x_over_one_plus_x_for_x_in_0_1
class OneMinusXOverOnePlusXForXIn01Op final : public UnaryOpBase {
 public:
  std::int32_t MinInput() const { return 0; }
  std::int32_t Tolerance() const { return 12; }
  std::int32_t ReferenceOp(std::int32_t x) const {
    using F = FixedPoint<std::int32_t, 0>;
    const double d = ToDouble(F::FromRaw(x));
    const double e = (1 - d) / (1 + d);
    return F::FromDouble(e).raw();
  }
  template <typename tRawType>
  tRawType Op(tRawType x) const {
    using F = FixedPoint<tRawType, 0>;
    const F f = F::FromRaw(x);
    return one_minus_x_over_one_plus_x_for_x_in_0_1(f).raw();
  }
};

// Op wrapping tanh
template <int tIntegerBits>
class TanhOp final : public UnaryOpBase {
 public:
  std::int32_t Tolerance() const { return 310; }
  std::int32_t ReferenceOp(std::int32_t x) const {
    using F = FixedPoint<std::int32_t, tIntegerBits>;
    using F0 = FixedPoint<std::int32_t, 0>;
    const double d = ToDouble(F::FromRaw(x));
    const double e = std::tanh(d);
    return F0::FromDouble(e).raw();
  }
  template <typename tRawType>
  tRawType Op(tRawType x) const {
    using F = FixedPoint<tRawType, tIntegerBits>;
    const F f = F::FromRaw(x);
    return tanh(f).raw();
  }
};

// Op wrapping one_over_one_plus_x_for_x_in_0_1
class OneOverOnePlusXForXIn01Op final : public UnaryOpBase {
 public:
  std::int32_t MinInput() const { return 0; }
  std::int32_t Tolerance() const { return 6; }
  std::int32_t ReferenceOp(std::int32_t x) const {
    using F = FixedPoint<std::int32_t, 0>;
    const double d = ToDouble(F::FromRaw(x));
    const double e = 1 / (1 + d);
    return F::FromDouble(e).raw();
  }
  template <typename tRawType>
  tRawType Op(tRawType x) const {
    using F = FixedPoint<tRawType, 0>;
    const F f = F::FromRaw(x);
    return one_over_one_plus_x_for_x_in_0_1(f).raw();
  }
};

// Op wrapping logistic
template <int tIntegerBits>
class LogisticOp final : public UnaryOpBase {
 public:
  std::int32_t Tolerance() const { return 155; }
  std::int32_t ReferenceOp(std::int32_t x) const {
    using F = FixedPoint<std::int32_t, tIntegerBits>;
    using F0 = FixedPoint<std::int32_t, 0>;
    const double d = ToDouble(F::FromRaw(x));
    const double e = 1 / (1 + std::exp(-d));
    return F0::FromDouble(e).raw();
  }
  template <typename tRawType>
  tRawType Op(tRawType x) const {
    using F = FixedPoint<tRawType, tIntegerBits>;
    const F f = F::FromRaw(x);
    return logistic(f).raw();
  }
};

// Tests a given op, on a given list of int32 input values.
template <typename tUnaryOpType>
void TestUnaryOp(const tUnaryOpType& unary_op,
                 const std::vector<std::int32_t>& testvals_int32) {
  Check(0 == (testvals_int32.size() % SimdVectorSize));
  for (std::size_t i = 0; i < testvals_int32.size(); i += SimdVectorSize) {
    // First, clamp input int32 values accoding to the MinInput() and MaxInput()
    // bounds returned by the op.
    std::int32_t input[SimdVectorSize] = {0};
    for (std::size_t j = 0; j < SimdVectorSize; j++) {
      const std::int32_t raw_input = testvals_int32[i + j];
      input[j] = std::min(unary_op.MaxInput(),
                          std::max(unary_op.MinInput(), raw_input));
    }
    // Compute reference results and check that the actual results on
    // scalar inputs agree with them, to the Tolerance() returned by the op.
    std::int32_t reference[SimdVectorSize] = {0};
    std::int32_t actual_scalar[SimdVectorSize] = {0};
    for (std::size_t j = 0; j < SimdVectorSize; j++) {
      reference[j] = unary_op.ReferenceOp(input[j]);
      actual_scalar[j] = unary_op.Op(input[j]);
      const std::int64_t diff = static_cast<std::int64_t>(actual_scalar[j]) -
                                static_cast<std::int64_t>(reference[j]);
      Check(std::abs(diff) <= unary_op.Tolerance());
    }
    // Check that the actual results on SIMD inputs agree *exactly* with the
    // actual results on scalar inputs. I.e. SIMD must make absolutely no
    // difference
    // to the results, regardless of the fact that both scalar and SIMD results
    // may differ from the reference results.
    std::int32_t actual_simd[SimdVectorSize] = {0};
    StoreSimdVector(actual_simd, unary_op.Op(LoadSimdVector(input)));
    for (std::size_t j = 0; j < SimdVectorSize; j++) {
      Check(actual_simd[j] == actual_scalar[j]);
    }
  }
}

template <int tIntegerBits>
void test_convert(FixedPoint<std::int32_t, tIntegerBits> x) {
  typedef FixedPoint<std::int32_t, tIntegerBits> F;
  F y = F::FromDouble(ToDouble(x));
  Check(y == x);
}

template <int tIntegerBits_a, int tIntegerBits_b>
void test_Rescale(FixedPoint<std::int32_t, tIntegerBits_a> a) {
  FixedPoint<std::int32_t, tIntegerBits_b> actual = Rescale<tIntegerBits_b>(a);
  FixedPoint<std::int32_t, tIntegerBits_b> expected =
      FixedPoint<std::int32_t, tIntegerBits_b>::FromDouble(ToDouble(a));
  Check(actual == expected);
}

template <int tIntegerBits_a, int tIntegerBits_b>
void test_Rescale(const std::vector<std::int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    FixedPoint<std::int32_t, tIntegerBits_a> aq;
    aq.raw() = a;
    test_Rescale<tIntegerBits_a, tIntegerBits_b>(aq);
  }
}

template <int tIntegerBits_a, int tIntegerBits_b>
void test_mul(FixedPoint<std::int32_t, tIntegerBits_a> a,
              FixedPoint<std::int32_t, tIntegerBits_b> b) {
  static const int ProductIntegerBits = tIntegerBits_a + tIntegerBits_b;
  using ProductFixedPoint = FixedPoint<std::int32_t, ProductIntegerBits>;
  ProductFixedPoint ab;
  ab = a * b;
  double a_double = ToDouble(a);
  double b_double = ToDouble(b);
  double ab_double = a_double * b_double;
  ProductFixedPoint expected = ProductFixedPoint::FromDouble(ab_double);
  std::int64_t diff = std::int64_t(ab.raw()) - std::int64_t(expected.raw());
  Check(std::abs(diff) <= 1);
}

template <int tIntegerBits_a, int tIntegerBits_b>
void test_mul(const std::vector<std::int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    for (auto b : testvals_int32) {
      FixedPoint<std::int32_t, tIntegerBits_a> aq;
      FixedPoint<std::int32_t, tIntegerBits_b> bq;
      aq.raw() = a;
      bq.raw() = b;
      test_mul(aq, bq);
    }
  }
}

template <int tExponent, int tIntegerBits_a>
void test_ExactMulByPot(FixedPoint<std::int32_t, tIntegerBits_a> a) {
  double x = ToDouble(a) * std::pow(2.0, tExponent);
  double y = ToDouble(ExactMulByPot<tExponent>(a));
  Check(x == y);
}

template <int tExponent, int tIntegerBits_a>
void test_ExactMulByPot(const std::vector<std::int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    FixedPoint<std::int32_t, tIntegerBits_a> aq;
    aq.raw() = a;
    test_ExactMulByPot<tExponent, tIntegerBits_a>(aq);
  }
}

// Make the list of test values to test each op against.
std::vector<std::int32_t> MakeTestValsInt32() {
  std::vector<std::int32_t> testvals_int32;

  for (int i = 0; i < 31; i++) {
    testvals_int32.push_back((1 << i) - 2);
    testvals_int32.push_back((1 << i) - 1);
    testvals_int32.push_back((1 << i));
    testvals_int32.push_back((1 << i) + 1);
    testvals_int32.push_back((1 << i) + 2);
    testvals_int32.push_back(-(1 << i) - 2);
    testvals_int32.push_back(-(1 << i) - 1);
    testvals_int32.push_back(-(1 << i));
    testvals_int32.push_back(-(1 << i) + 1);
    testvals_int32.push_back(-(1 << i) + 2);
  }
  testvals_int32.push_back(std::numeric_limits<std::int32_t>::min());
  testvals_int32.push_back(std::numeric_limits<std::int32_t>::min() + 1);
  testvals_int32.push_back(std::numeric_limits<std::int32_t>::min() + 2);
  testvals_int32.push_back(std::numeric_limits<std::int32_t>::max() - 2);
  testvals_int32.push_back(std::numeric_limits<std::int32_t>::max() - 1);
  testvals_int32.push_back(std::numeric_limits<std::int32_t>::max());

  std::mt19937 random_engine;
  std::uniform_int_distribution<std::int32_t> uniform_distribution(
      std::numeric_limits<std::int32_t>::min(),
      std::numeric_limits<std::int32_t>::max());
  for (int i = 0; i < 1000; i++) {
    testvals_int32.push_back(uniform_distribution(random_engine));
  }

  // SIMD tests will require the length of testvals_int32 to be a multiple
  // of SIMD vector size.
  while (testvals_int32.size() % SimdVectorSize) {
    testvals_int32.push_back(0);
  }

  std::sort(testvals_int32.begin(), testvals_int32.end());
  return testvals_int32;
}

}  // end anonymous namespace

}  // end namespace gemmlowp

int main() {
  using namespace gemmlowp;

  const std::vector<std::int32_t> testvals_int32 = MakeTestValsInt32();

  for (int s = 0; s < 32; s++) {
    TestUnaryOp(RoundingDivideByPOTOp(s), testvals_int32);
  }

  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-31>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-30>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-29>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-17>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-16>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-15>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-4>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-3>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-2>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<-1>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<0>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<1>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<2>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<3>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<4>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<15>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<16>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<17>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<29>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<30>(), testvals_int32);
  TestUnaryOp(SaturatingRoundingMultiplyByPOTOp<31>(), testvals_int32);

  TestUnaryOp(ExpOnIntervalBetweenNegativeOneQuarterAnd0ExclOp(),
              testvals_int32);
  TestUnaryOp(ExpOnNegativeValuesOp<0>(), testvals_int32);
  TestUnaryOp(ExpOnNegativeValuesOp<1>(), testvals_int32);
  TestUnaryOp(ExpOnNegativeValuesOp<2>(), testvals_int32);
  TestUnaryOp(ExpOnNegativeValuesOp<3>(), testvals_int32);
  TestUnaryOp(ExpOnNegativeValuesOp<4>(), testvals_int32);
  TestUnaryOp(ExpOnNegativeValuesOp<5>(), testvals_int32);
  TestUnaryOp(ExpOnNegativeValuesOp<6>(), testvals_int32);

  TestUnaryOp(OneMinusXOverOnePlusXForXIn01Op(), testvals_int32);
  TestUnaryOp(TanhOp<0>(), testvals_int32);
  TestUnaryOp(TanhOp<1>(), testvals_int32);
  TestUnaryOp(TanhOp<2>(), testvals_int32);
  TestUnaryOp(TanhOp<3>(), testvals_int32);
  TestUnaryOp(TanhOp<4>(), testvals_int32);
  TestUnaryOp(TanhOp<5>(), testvals_int32);
  TestUnaryOp(TanhOp<6>(), testvals_int32);

  TestUnaryOp(OneOverOnePlusXForXIn01Op(), testvals_int32);
  TestUnaryOp(LogisticOp<0>(), testvals_int32);
  TestUnaryOp(LogisticOp<1>(), testvals_int32);
  TestUnaryOp(LogisticOp<2>(), testvals_int32);
  TestUnaryOp(LogisticOp<3>(), testvals_int32);
  TestUnaryOp(LogisticOp<4>(), testvals_int32);
  TestUnaryOp(LogisticOp<5>(), testvals_int32);
  TestUnaryOp(LogisticOp<6>(), testvals_int32);

  for (auto a : testvals_int32) {
    FixedPoint<std::int32_t, 4> x;
    x.raw() = a;
    test_convert(x);
  }

  test_mul<0, 0>(testvals_int32);
  test_mul<0, 1>(testvals_int32);
  test_mul<2, 0>(testvals_int32);
  test_mul<1, 1>(testvals_int32);
  test_mul<4, 4>(testvals_int32);
  test_mul<3, 5>(testvals_int32);
  test_mul<7, 2>(testvals_int32);
  test_mul<14, 15>(testvals_int32);

  test_Rescale<0, 0>(testvals_int32);
  test_Rescale<0, 1>(testvals_int32);
  test_Rescale<2, 0>(testvals_int32);
  test_Rescale<4, 4>(testvals_int32);
  test_Rescale<4, 5>(testvals_int32);
  test_Rescale<6, 3>(testvals_int32);
  test_Rescale<13, 9>(testvals_int32);

  test_ExactMulByPot<0, 0>(testvals_int32);
  test_ExactMulByPot<0, 4>(testvals_int32);
  test_ExactMulByPot<1, 4>(testvals_int32);
  test_ExactMulByPot<3, 2>(testvals_int32);
  test_ExactMulByPot<-4, 5>(testvals_int32);
  test_ExactMulByPot<-2, 6>(testvals_int32);

  std::cerr << "All tests passed." << std::endl;
}
