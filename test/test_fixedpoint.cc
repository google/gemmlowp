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

#define GEMMLOWP_ENABLE_FIXEDPOINT_CONSTANTS_CHECKS

#include "test.h"

#include "../fixedpoint/fixedpoint.h"

using namespace gemmlowp;

void test_RoundingDivideByPOT(std::int32_t x) {
  double d = x;
  for (int s = 0; s < 32; s++) {
    const std::int32_t actual = RoundingDivideByPOT(x, s);
    const std::int32_t expected = std::round(d);
    Check(actual == expected);
    d /= 2;
  }
}

void test_RoundingDivideByPOT(const std::vector<std::int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    test_RoundingDivideByPOT(a);
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

void test_exp_on_interval_between_negative_one_quarter_and_0_excl(
    FixedPoint<std::int32_t, 0> a) {
  double a_double = ToDouble(a);
  double expected = std::exp(a_double);
  double actual =
      ToDouble(exp_on_interval_between_negative_one_quarter_and_0_excl(a));
  double error = expected - actual;
  Check(std::abs(error) < 3e-7);
}

void test_exp_on_interval_between_negative_one_quarter_and_0_excl(
    const std::vector<std::int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    typedef FixedPoint<std::int32_t, 0> F;
    F aq = SaturatingRoundingMultiplyByPOT<-3>(F::FromRaw(a)) -
           F::ConstantPOT<-3>();
    test_exp_on_interval_between_negative_one_quarter_and_0_excl(aq);
  }
}

template <int tIntegerBits>
void test_exp_on_negative_values(FixedPoint<std::int32_t, tIntegerBits> a) {
  double a_double = ToDouble(a);
  double expected = std::exp(a_double);
  double actual = ToDouble(exp_on_negative_values(a));
  double error = expected - actual;
  Check(std::abs(error) < 3e-7);
}

template <int tIntegerBits>
void test_exp_on_negative_values(const std::vector<std::int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    if (a < 0) {
      FixedPoint<std::int32_t, tIntegerBits> aq;
      aq.raw() = a;
      test_exp_on_negative_values(aq);
    }
  }
}

void test_one_minus_x_over_one_plus_x_for_x_in_0_1(FixedPoint<std::int32_t, 0> a) {
  double a_double = ToDouble(a);
  double expected = (1 - a_double) / (1 + a_double);
  FixedPoint<std::int32_t, 0> retval = one_minus_x_over_one_plus_x_for_x_in_0_1(a);
  double actual = ToDouble(retval);
  double error = expected - actual;
  Check(std::abs(error) < 6e-9);
}

void test_one_minus_x_over_one_plus_x_for_x_in_0_1(
    const std::vector<std::int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    if (a > 0) {
      FixedPoint<std::int32_t, 0> aq;
      aq.raw() = a;
      test_one_minus_x_over_one_plus_x_for_x_in_0_1(aq);
    }
  }
}

template <int tIntegerBits>
void test_tanh(FixedPoint<std::int32_t, tIntegerBits> a) {
  double a_double = ToDouble(a);
  double expected = std::tanh(a_double);
  double actual = ToDouble(tanh(a));
  double error = expected - actual;
  Check(std::abs(error) < 1.5e-7);
}

template <int tIntegerBits>
void test_tanh(const std::vector<std::int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    FixedPoint<std::int32_t, tIntegerBits> aq;
    aq.raw() = a;
    test_tanh(aq);
  }
}

void test_one_over_one_plus_x_for_x_in_0_1(FixedPoint<std::int32_t, 0> a) {
  double a_double = ToDouble(a);
  double expected = 1. / (1 + a_double);
  FixedPoint<std::int32_t, 0> retval = one_over_one_plus_x_for_x_in_0_1(a);
  double actual = ToDouble(retval);
  double error = expected - actual;
  Check(std::abs(error) < 3e-9);
}

void test_one_over_one_plus_x_for_x_in_0_1(
    const std::vector<std::int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    if (a > 0) {
      FixedPoint<std::int32_t, 0> aq;
      aq.raw() = a;
      test_one_over_one_plus_x_for_x_in_0_1(aq);
    }
  }
}

template <int tIntegerBits>
void test_logistic(FixedPoint<std::int32_t, tIntegerBits> a) {
  double a_double = ToDouble(a);
  double expected = 1. / (1 + std::exp(-a_double));
  double actual = ToDouble(logistic(a));
  double error = expected - actual;
  Check(std::abs(error) < 8e-8);
}

template <int tIntegerBits>
void test_logistic(const std::vector<std::int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    FixedPoint<std::int32_t, tIntegerBits> aq;
    aq.raw() = a;
    test_logistic(aq);
  }
}

#ifdef GEMMLOWP_NEON
void test_int32x4(const std::vector<std::int32_t>& testvals_int32) {
  size_t n = testvals_int32.size();
  size_t n4 = n - (n % 4);
  std::vector<std::int32_t> results_int32(n4);
  std::vector<std::int32_t> results_int32x4(n4);

  for (size_t i = 0; i < n4; i++) {
    results_int32[i] =
        tanh(FixedPoint<std::int32_t, 4>::FromRaw(testvals_int32[i])).raw();
  }
  for (size_t i = 0; i < n4; i++) {
    vst1q_s32(
        &results_int32x4[i],
        tanh(FixedPoint<int32x4_t, 4>::FromRaw(vld1q_s32(&testvals_int32[i])))
            .raw());
  }

  for (size_t i = 0; i < n4; i++) {
    Check(results_int32[i] == results_int32x4[i]);
  }
}
#endif  // GEMMLOWP_NEON

#ifdef GEMMLOWP_SSE4
#define LOAD_SI128(ptr)                                       \
  (((unsigned long)(ptr)&15) == 0)                            \
      ? _mm_load_si128(reinterpret_cast<const __m128i*>(ptr)) \
      : _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))
#define STORE_SI128(ptr, val)                                 \
  (((unsigned long)(ptr)&15) == 0)                            \
      ? _mm_store_si128(reinterpret_cast<__m128i*>(ptr), val) \
      : _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), val)

template <int tIntegerBits>
void test_tanh_m128i(const std::vector<std::int32_t>& testvals_int32) {
  size_t n = testvals_int32.size();
  size_t n4 = n / 4;
  std::uint32_t results_m128i[4];

  for (size_t i = 0; i < n4; i += 4) {
    typedef FixedPoint<std::int32_t, tIntegerBits> F_input;
    typedef FixedPoint<__m128i, tIntegerBits> F4_input;
    typedef FixedPoint<std::int32_t, 0> F_output;
    typedef FixedPoint<__m128i, 0> F4_output;

    __m128i arguments = LOAD_SI128(&testvals_int32[i]);
    F4_output results = tanh(F4_input::FromRaw(arguments));

    STORE_SI128(results_m128i, results.raw());
    for (size_t j = 0; j < 4; j++) {
      double expected =
          std::tanh(ToDouble(F_input::FromRaw(testvals_int32[i + j])));
      double computed = ToDouble(F_output::FromRaw(results_m128i[j]));
      double error = std::abs(expected - computed);
      Check(error < 1.5e-7);
    }
  }
}
#endif  // GEMMLOWP_SSE4

int main() {
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

  std::uint32_t random = 1;
  for (int i = 0; i < 1000; i++) {
    random = random * 1664525 + 1013904223;
    testvals_int32.push_back(static_cast<std::int32_t>(random));
  }

  std::sort(testvals_int32.begin(), testvals_int32.end());

  test_RoundingDivideByPOT(testvals_int32);

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

  test_exp_on_interval_between_negative_one_quarter_and_0_excl(testvals_int32);
  test_exp_on_negative_values<1>(testvals_int32);
  test_exp_on_negative_values<2>(testvals_int32);
  test_exp_on_negative_values<3>(testvals_int32);
  test_exp_on_negative_values<4>(testvals_int32);
  test_exp_on_negative_values<5>(testvals_int32);
  test_exp_on_negative_values<6>(testvals_int32);

  test_one_minus_x_over_one_plus_x_for_x_in_0_1(testvals_int32);
  test_tanh<1>(testvals_int32);
  test_tanh<2>(testvals_int32);
  test_tanh<3>(testvals_int32);
  test_tanh<4>(testvals_int32);
  test_tanh<5>(testvals_int32);
  test_tanh<6>(testvals_int32);

  test_one_over_one_plus_x_for_x_in_0_1(testvals_int32);
  test_logistic<1>(testvals_int32);
  test_logistic<2>(testvals_int32);
  test_logistic<3>(testvals_int32);
  test_logistic<4>(testvals_int32);
  test_logistic<5>(testvals_int32);
  test_logistic<6>(testvals_int32);

#ifdef GEMMLOWP_NEON
  test_int32x4(testvals_int32);
#endif  // GEMMLOWP_NEON

#ifdef GEMMLOWP_SSE4
  test_tanh_m128i<1>(testvals_int32);
  test_tanh_m128i<2>(testvals_int32);
  test_tanh_m128i<3>(testvals_int32);
  test_tanh_m128i<4>(testvals_int32);
  test_tanh_m128i<5>(testvals_int32);
  test_tanh_m128i<6>(testvals_int32);
#endif  // GEMMLOWP_SSE4

  std::cerr << "All tests passed." << std::endl;
}
