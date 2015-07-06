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

#include <iostream>
#include <ctime>
#include <cstdint>
#include <vector>
#include <map>
#include <cstdlib>

#include "test/test.h"
#include "public/gemmlowp.h"

namespace gemmlowp {

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

const double min_accurate_duration = 1e-1;
const std::size_t min_working_set_size = 16 * 1024 * 1024;

template <typename Kernel, typename LhsType, typename RhsType,
          typename ResultType>
double gflops_for_gemm_size(GemmContext* context, int rows, int depth,
                            int cols) {
  typedef std::uint8_t Scalar;

  // set up the matrix pool

  const std::size_t combined_three_matrices_sizes =
      sizeof(Scalar) * (rows * depth + depth * cols + rows * cols);

  const std::size_t matrix_pool_size =
      1 + min_working_set_size / combined_three_matrices_sizes;

  std::vector<LhsType> lhs(matrix_pool_size);
  std::vector<RhsType> rhs(matrix_pool_size);
  std::vector<ResultType> result(matrix_pool_size);

  lhs[0].Resize(rows, depth);
  MakeConstant(&lhs[0], 128);
  rhs[0].Resize(depth, cols);
  MakeConstant(&rhs[0], 128);
  result[0].Resize(rows, cols);
  MakeZero(&result[0]);

  for (std::size_t i = 1; i < matrix_pool_size; i++) {
    lhs[i] = lhs[0];
    rhs[i] = rhs[0];
    result[i] = result[0];
  }

  const int depth_shift = static_cast<int>(
      std::ceil(0.5 * std::log(static_cast<float>(depth)) / std::log(2.0f)));

  // main benchmark loop

  int iters_at_a_time = 1;
  float time_per_iter = 0.0f;
  std::size_t matrix_index = 0;

  while (true) {
    double starttime = time();
    for (int i = 0; i < iters_at_a_time; i++) {
      Gemm(context, lhs[matrix_index].const_map(),
           rhs[matrix_index].const_map(), &result[matrix_index].map(), -75, -91,
           74980, 123, 18 + depth_shift);

      matrix_index++;
      if (matrix_index == matrix_pool_size) {
        matrix_index = 0;
      }
    }
    double endtime = time();

    const float timing = static_cast<float>(endtime - starttime);

    if (timing >= min_accurate_duration) {
      time_per_iter = timing / iters_at_a_time;
      break;
    }

    iters_at_a_time *= 2;
  }

  return 2e-9 * rows * depth * cols / time_per_iter;
}

void benchmark(GemmContext* context) {
#ifdef GEMMLOWP_TEST_KERNEL
  typedef gemmlowp::GEMMLOWP_TEST_KERNEL KernelForGEMM;
  typedef gemmlowp::GEMMLOWP_TEST_KERNEL KernelForGEMV;
#else
  typedef gemmlowp::DefaultKernelForGEMM KernelForGEMM;
  typedef gemmlowp::DefaultKernelForGEMV KernelForGEMV;
#endif

  std::map<std::tuple<int, int, int>, std::vector<double>> benchmark_results;

  std::vector<std::tuple<int, int, int>> benchmark_sizes;
  benchmark_sizes.emplace_back(10, 10, 10);
  benchmark_sizes.emplace_back(20, 20, 20);
  benchmark_sizes.emplace_back(30, 30, 30);
  benchmark_sizes.emplace_back(40, 40, 40);
  benchmark_sizes.emplace_back(50, 50, 50);
  benchmark_sizes.emplace_back(60, 60, 60);
  benchmark_sizes.emplace_back(64, 256, 147);
  benchmark_sizes.emplace_back(100, 100, 1);
  benchmark_sizes.emplace_back(100, 100, 100);
  benchmark_sizes.emplace_back(100, 1000, 100);
  benchmark_sizes.emplace_back(1000, 1000, 1);
  benchmark_sizes.emplace_back(1000, 1000, 10);
  benchmark_sizes.emplace_back(1000, 1000, 100);
  benchmark_sizes.emplace_back(1000, 1000, 1000);

  const int repeat = 2;

  typedef Matrix<std::uint8_t, MapOrder::RowMajor> LhsType;
  typedef Matrix<std::uint8_t, MapOrder::ColMajor> RhsType;
  typedef Matrix<std::uint8_t, MapOrder::ColMajor> ResultType;

#ifdef GEMMLOWP_TEST_PROFILE
  gemmlowp::RegisterCurrentThreadForProfiling();
  gemmlowp::StartProfiling();
#endif

  // We don't record the first repetition, it's just warm-up.
  for (int r = 0; r < repeat + 1; r++) {
    std::cout << "repetition " << r + 1 << "/" << repeat + 1 << "...\r"
              << std::flush;
    for (auto s : benchmark_sizes) {
      double gflops = 0;
      int rows = std::get<0>(s);
      int depth = std::get<1>(s);
      int cols = std::get<2>(s);
      if (cols > KernelForGEMM::Format::kCols / 2) {
        gflops =
            gflops_for_gemm_size<KernelForGEMM, LhsType, RhsType, ResultType>(
                context, rows, depth, cols);
      } else {
        gflops =
            gflops_for_gemm_size<KernelForGEMV, LhsType, RhsType, ResultType>(
                context, rows, depth, cols);
      }
      if (r > 0) {
        benchmark_results[s].emplace_back(gflops);
      }
    }
  }

  std::cout << "                                                \r"
            << std::flush;

#ifdef GEMMLOWP_TEST_PROFILE
  gemmlowp::FinishProfiling();
#endif

  std::cout.precision(4);

  for (auto b : benchmark_results) {
    sort(b.second.begin(), b.second.end());
    std::cout << std::get<0>(b.first) << "x" << std::get<1>(b.first) << "x"
              << std::get<2>(b.first) << " : " << b.second.back() << " GFlops/s"
              << std::endl;
  }
  std::cout << std::endl;
}

}  // end namespace gemmlowp

int main() {
  {
    gemmlowp::GemmContext context;
    std::cout << "Benchmarking default mode (typically multi-threaded)..."
              << std::endl;
    gemmlowp::benchmark(&context);
  }

  {
    gemmlowp::GemmContext context;
    context.set_max_num_threads(1);
    std::cout << "Benchmarking single-threaded mode..." << std::endl;
    gemmlowp::benchmark(&context);
  }
}
