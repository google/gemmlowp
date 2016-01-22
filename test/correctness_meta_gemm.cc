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

#include <unistd.h>
#ifdef __APPLE__
#include <sys/time.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <vector>

#include "../public/gemmlowp.h"
#include "../meta/multi_thread_gemm.h"
#include "test.h"

#if defined(__arm__) && !defined(GEMMLOWP_NEON)
#warning "Building without NEON support on ARM, check your compiler setup!"
#endif

double time() {
#ifdef __APPLE__
  timeval t;
  gettimeofday(&t, nullptr);
  return t.tv_sec + 1e-6 * t.tv_usec;
#else
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
#endif
}

void prepare_test_data(std::uint8_t* data, std::int32_t rows, std::int32_t cols,
                       std::int32_t seed, std::int32_t seed_2) {
  int32_t value = seed;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      data[i * cols + j] = static_cast<std::uint8_t>(value);
      value = ((value * seed_2) + seed) % 256;
    }
  }
}

bool verbose = false;
bool quiet = true;

void check_result(std::uint8_t* left, std::uint8_t* right, std::uint8_t* result,
                  std::int32_t rows, std::int32_t cols, std::int32_t depth,
                  std::int32_t lhs_offset, std::int32_t rhs_offset,
                  std::int32_t sum_offset, std::int32_t mul_offset,
                  std::int32_t shift) {
  std::int32_t rounding = (1 << (shift - 1));
  std::int32_t wrong = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::int32_t expected = 0;
      for (int k = 0; k < depth; ++k) {
        expected +=
            (static_cast<std::int32_t>(left[depth * i + k]) + lhs_offset) *
            (static_cast<std::int32_t>(right[depth * j + k]) + rhs_offset);
      }
      expected += sum_offset;
      expected *= mul_offset;
      expected += rounding;
      expected = (expected >> shift);
      if (expected < 0) {
        expected = 0;
      } else if (expected > 255) {
        expected = 255;
      }
      expected = static_cast<std::int32_t>(static_cast<std::uint8_t>(expected));
      std::int32_t actual = static_cast<std::int32_t>(result[i * cols + j]);
      if (actual == expected) {
        if (!quiet) {
          if (verbose) {
            std::cout << expected << "==" << actual << " ";
          } else {
            std::cout << ".";
          }
        }
      } else {
        if (!quiet) {
          if (verbose) {
            std::cout << expected << "!=" << actual << " ";
          } else {
            std::cout << "x";
          }
        }
        wrong++;
      }
    }
    if (!quiet) {
      std::cout << std::endl;
    }
  }
  if (wrong > 0) {
    std::cout << "Wrong: " << wrong << std::endl;
  } else {
    std::cout << "." << std::flush;
  }
}

void check_result_f(std::uint8_t* left, std::uint8_t* right, float* result,
                    std::int32_t rows, std::int32_t cols, std::int32_t depth,
                    std::int32_t lhs_offset, std::int32_t rhs_offset,
                    float result_offset) {
  std::int32_t wrong = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::int32_t expected = 0;
      for (int k = 0; k < depth; ++k) {
        expected +=
            (static_cast<std::int32_t>(left[depth * i + k]) + lhs_offset) *
            (static_cast<std::int32_t>(right[depth * j + k]) + rhs_offset);
      }
      float expected_float = static_cast<float>(expected) * result_offset;
      float actual_float = result[i * cols + j];
      if (actual_float == expected_float) {
        if (!quiet) {
          if (verbose) {
            std::cout << expected_float << "==" << actual_float << " ";
          } else {
            std::cout << ".";
          }
        }
      } else {
        if (!quiet) {
          if (verbose) {
            std::cout << expected_float << "!=" << actual_float << " ";
          } else {
            std::cout << "x";
          }
        }
        wrong++;
      }
    }
    if (!quiet) {
      std::cout << std::endl;
    }
  }
  if (wrong > 0) {
    std::cout << "Wrong: " << wrong << std::endl;
  } else {
    std::cout << "." << std::flush;
  }
}

template <typename T>
void clear(T* result, std::int32_t rows, std::int32_t cols) {
  for (int i = 0; i < rows * cols; ++i) {
    result[i] = static_cast<T>(0);
  }
}

void test(std::uint8_t* scratch, std::uint8_t* lhs, std::uint8_t* rhs,
          std::int32_t m, std::int32_t n, std::int32_t k, std::uint8_t* result,
          gemmlowp::WorkersPool* pool, std::int32_t pool_size) {
  prepare_test_data(lhs, m, k, 11, 13);
  prepare_test_data(rhs, n, k, 177, 19);

  clear(result, m, n);
  gemmlowp::meta::multi_thread_gemm_q8(pool, pool_size, scratch, lhs, rhs, m, n,
                                       k, -127, -127, 127 * k, 1, 7, result);
  check_result(lhs, rhs, result, m, n, k, -127, -127, 127 * k, 1, 7);
}

void test_f(std::uint8_t* scratch, std::uint8_t* lhs, std::uint8_t* rhs,
            std::int32_t m, std::int32_t n, std::int32_t k, float* result,
            gemmlowp::WorkersPool* pool, std::int32_t pool_size) {
  prepare_test_data(lhs, m, k, 11, 13);
  prepare_test_data(rhs, n, k, 177, 19);

  clear(result, m, n);
  float scale = 1.0f / 1234567.8f;
  gemmlowp::meta::multi_thread_gemm_f(pool, pool_size, scratch, lhs, rhs, m, n,
                                      k, -127, -127, scale, result);
  check_result_f(lhs, rhs, result, m, n, k, -127, -127, scale);
}

int main() {
  const std::int32_t min_n = 256;
  const std::int32_t min_m = 256;
  const std::int32_t min_k = 256;

  const std::int32_t max_n = 1024;
  const std::int32_t max_m = 1024;
  const std::int32_t max_k = 512;

  std::uint8_t* left = new std::uint8_t[max_m * max_k];
  std::uint8_t* right = new std::uint8_t[max_n * max_k];
  std::uint8_t* result = new std::uint8_t[max_m * max_n];
  float* result_float = new float[max_m * max_n];
  std::uint8_t* scratch = new std::uint8_t[1024 * 1024 * 64];

  gemmlowp::WorkersPool pool;
  pool.CreateWorkers(3);

  std::cout << "Quantized 8 bit." << std::endl << std::flush;

  for (int m = min_m; m < max_m; m += 128) {
    for (int n = min_n; n < max_n; n += 128) {
      for (int k = min_k; k < max_k; k += 13) {
        test(scratch, left, right, m, n, k, result, &pool, 4);
      }
    }
  }

  std::cout << std::endl << "Floats." << std::endl << std::flush;

  for (int m = min_m; m < max_m; m += 128) {
    for (int n = min_n; n < max_n; n += 128) {
      for (int k = min_k; k < max_k; k += 13) {
        test_f(scratch, left, right, m, n, k, result_float, &pool, 4);
      }
    }
  }

  std::cout << std::endl << "Done." << std::endl << std::flush;
}
