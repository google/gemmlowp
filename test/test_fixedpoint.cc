#define GEMMLOWP_ENABLE_FIXEDPOINT_CONSTANTS_CHECKS

#include "test.h"

#include "../internal/fixedpoint.h"

using namespace gemmlowp;

template <int tIntegerBits>
void test_convert(FixedPoint<int32_t, tIntegerBits> x) {
  typedef FixedPoint<int32_t, tIntegerBits> F;
  F y = ToFixedPoint<int32_t, tIntegerBits>(ToDouble(x));
  Check(y == x);
}

template <int tIntegerBits_a, int tIntegerBits_b>
void test_Rescale(FixedPoint<int32_t, tIntegerBits_a> a) {
  FixedPoint<int32_t, tIntegerBits_b> actual = Rescale<tIntegerBits_b>(a);
  FixedPoint<int32_t, tIntegerBits_b> expected =
      ToFixedPoint<int32_t, tIntegerBits_b>(ToDouble(a));
  Check(actual == expected);
}

template <int tIntegerBits_a, int tIntegerBits_b>
void test_Rescale(const std::vector<int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    FixedPoint<int32_t, tIntegerBits_a> aq;
    aq.raw() = a;
    test_Rescale<tIntegerBits_a, tIntegerBits_b>(aq);
  }
}

template <int tIntegerBits_a, int tIntegerBits_b>
void test_mul(FixedPoint<int32_t, tIntegerBits_a> a,
              FixedPoint<int32_t, tIntegerBits_b> b) {
  static const int IntegerBits_ab = tIntegerBits_a + tIntegerBits_b;
  FixedPoint<int32_t, IntegerBits_ab> ab;
  ab = a * b;
  double a_double = ToDouble(a);
  double b_double = ToDouble(b);
  double ab_double = a_double * b_double;
  FixedPoint<int32_t, IntegerBits_ab> expected =
      ToFixedPoint<int32_t, IntegerBits_ab>(ab_double);
  int64_t diff = int64_t(ab.raw()) - int64_t(expected.raw());
  Check(std::abs(diff) <= 1);
}

template <int tIntegerBits_a, int tIntegerBits_b>
void test_mul(const std::vector<int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    for (auto b : testvals_int32) {
      FixedPoint<int32_t, tIntegerBits_a> aq;
      FixedPoint<int32_t, tIntegerBits_b> bq;
      aq.raw() = a;
      bq.raw() = b;
      test_mul(aq, bq);
    }
  }
}

template <int tExponent, int tIntegerBits_a>
void test_ExactMulByPot(FixedPoint<int32_t, tIntegerBits_a> a) {
  double x = ToDouble(a) * std::pow(2.0, tExponent);
  double y = ToDouble(ExactMulByPot<tExponent>(a));
  Check(x == y);
}

template <int tExponent, int tIntegerBits_a>
void test_ExactMulByPot(const std::vector<int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    FixedPoint<int32_t, tIntegerBits_a> aq;
    aq.raw() = a;
    test_ExactMulByPot<tExponent, tIntegerBits_a>(aq);
  }
}

void test_exp_on_interval_between_negative_one_quarter_and_0_excl(
    FixedPoint<int32_t, 0> a) {
  double a_double = ToDouble(a);
  double expected = std::exp(a_double);
  double actual =
      ToDouble(exp_on_interval_between_negative_one_quarter_and_0_excl(a));
  double error = expected - actual;
  Check(std::abs(error) < 3e-7);
}

void test_exp_on_interval_between_negative_one_quarter_and_0_excl(
    const std::vector<int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    typedef FixedPoint<int32_t, 0> F;
    F aq = SaturatingRoundingMultiplyByPOT<-3>(F::FromRaw(a)) -
           F::ConstantPOT<-3>();
    test_exp_on_interval_between_negative_one_quarter_and_0_excl(aq);
  }
}

template <int tIntegerBits>
void test_exp_on_negative_values(FixedPoint<int32_t, tIntegerBits> a) {
  double a_double = ToDouble(a);
  double expected = std::exp(a_double);
  double actual = ToDouble(exp_on_negative_values(a));
  double error = expected - actual;
  Check(std::abs(error) < 3e-7);
}

template <int tIntegerBits>
void test_exp_on_negative_values(const std::vector<int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    if (a < 0) {
      FixedPoint<int32_t, tIntegerBits> aq;
      aq.raw() = a;
      test_exp_on_negative_values(aq);
    }
  }
}

void test_one_minus_x_over_one_plus_x_for_x_in_0_1(FixedPoint<int32_t, 0> a) {
  double a_double = ToDouble(a);
  double expected = (1 - a_double) / (1 + a_double);
  FixedPoint<int32_t, 0> retval = one_minus_x_over_one_plus_x_for_x_in_0_1(a);
  double actual = ToDouble(retval);
  double error = expected - actual;
  Check(std::abs(error) < 6e-9);
}

void test_one_minus_x_over_one_plus_x_for_x_in_0_1(
    const std::vector<int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    if (a > 0) {
      FixedPoint<int32_t, 0> aq;
      aq.raw() = a;
      test_one_minus_x_over_one_plus_x_for_x_in_0_1(aq);
    }
  }
}

template <int tIntegerBits>
void test_tanh(FixedPoint<int32_t, tIntegerBits> a) {
  double a_double = ToDouble(a);
  double expected = std::tanh(a_double);
  double actual = ToDouble(tanh(a));
  double error = expected - actual;
  Check(std::abs(error) < 1.5e-7);
}

template <int tIntegerBits>
void test_tanh(const std::vector<int32_t>& testvals_int32) {
  for (auto a : testvals_int32) {
    FixedPoint<int32_t, tIntegerBits> aq;
    aq.raw() = a;
    test_tanh(aq);
  }
}

#ifdef GEMMLOWP_NEON
void test_int32x4(const std::vector<int32_t>& testvals_int32) {
  size_t n = testvals_int32.size();
  size_t n4 = n - (n % 4);
  std::vector<int32_t> results_int32(n4);
  std::vector<int32_t> results_int32x4(n4);

  for (size_t i = 0; i < n4; i++) {
    results_int32[i] =
        tanh(FixedPoint<int32_t, 4>::FromRaw(testvals_int32[i])).raw();
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

int main() {
  std::vector<int32_t> testvals_int32;

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
  testvals_int32.push_back(std::numeric_limits<int32_t>::min());
  testvals_int32.push_back(std::numeric_limits<int32_t>::min() + 1);
  testvals_int32.push_back(std::numeric_limits<int32_t>::min() + 2);
  testvals_int32.push_back(std::numeric_limits<int32_t>::max() - 2);
  testvals_int32.push_back(std::numeric_limits<int32_t>::max() - 1);
  testvals_int32.push_back(std::numeric_limits<int32_t>::max());

  uint32_t random = 1;
  for (int i = 0; i < 1000; i++) {
    random = random * 1664525 + 1013904223;
    testvals_int32.push_back(static_cast<int32_t>(random));
  }

  std::sort(testvals_int32.begin(), testvals_int32.end());

  for (auto a : testvals_int32) {
    FixedPoint<int32_t, 4> x;
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

#ifdef GEMMLOWP_NEON
  test_int32x4(testvals_int32);
#endif  // GEMMLOWP_NEON

  std::cerr << "All tests passed." << std::endl;
}
