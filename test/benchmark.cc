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

#include "test.h"
#include "../public/gemmlowp.h"

#if defined(__arm__) && !defined(GEMMLOWP_NEON)
#warning "Building without NEON support on ARM, check your compiler setup!"
#endif

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
double time_for_gemm_size(GemmContext* context, int rows, int depth,
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

  return time_per_iter;
}

template <typename Kernel, typename LhsType, typename RhsType,
          typename ResultType>
double gflops_for_gemm_size(GemmContext* context, int rows, int depth,
                            int cols) {
  const double time_per_iter =
      time_for_gemm_size<Kernel, LhsType, RhsType, ResultType>(
          context, rows, depth, cols);
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

void benchmark_googlenet(GemmContext* context) {

#ifdef GEMMLOWP_TEST_KERNEL
  typedef gemmlowp::GEMMLOWP_TEST_KERNEL KernelForGEMM;
  typedef gemmlowp::GEMMLOWP_TEST_KERNEL KernelForGEMV;
#else
  typedef gemmlowp::DefaultKernelForGEMM KernelForGEMM;
  typedef gemmlowp::DefaultKernelForGEMV KernelForGEMV;
#endif

  // These are the m, n, k sizes for a typical GoogLeNet.
  const int googlenet_gemm_sizes[] = {
    12544, 64, 147,
    3136, 64, 64,
    3136, 192, 576,
    784, 64, 192,
    784, 96, 192,
    784, 128, 864,
    784, 16, 192,
    784, 32, 400,
    784, 32, 192,
    784, 128, 256,
    784, 128, 256,
    784, 192, 1152,
    784, 32, 256,
    784, 96, 800,
    784, 64, 256,
    196, 192, 480,
    196, 96, 480,
    196, 204, 864,
    196, 16, 480,
    196, 48, 400,
    196, 64, 480,
    196, 160, 508,
    196, 112, 508,
    196, 224, 1008,
    196, 24, 508,
    196, 64, 600,
    196, 64, 508,
    196, 128, 512,
    196, 128, 512,
    196, 256, 1152,
    196, 24, 512,
    196, 64, 600,
    196, 64, 512,
    196, 112, 512,
    196, 144, 512,
    196, 288, 1296,
    196, 32, 512,
    196, 64, 800,
    196, 64, 512,
    196, 256, 528,
    196, 160, 528,
    196, 320, 1440,
    196, 32, 528,
    196, 128, 800,
    196, 128, 528,
    49, 256, 832,
    49, 160, 832,
    49, 320, 1440,
    49, 48, 832,
    49, 128, 1200,
    49, 128, 832,
    49, 384, 832,
    49, 192, 832,
    49, 384, 1728,
    49, 48, 832,
    49, 128, 1200,
    49, 128, 832,
    16, 128, 508,
    1, 1024, 2048,
    1, 1008, 1024,
    16, 128, 528,
    1, 1024, 2048,
    1, 1008, 1024,
    1, 1008, 1024,
  };
  const int param_count =
      sizeof(googlenet_gemm_sizes) / sizeof(googlenet_gemm_sizes[0]);
  const int gemm_count = param_count / 3;

  const int repeat = 2;

  typedef Matrix<std::uint8_t, MapOrder::RowMajor> LhsType;
  typedef Matrix<std::uint8_t, MapOrder::ColMajor> RhsType;
  typedef Matrix<std::uint8_t, MapOrder::ColMajor> ResultType;

#ifdef GEMMLOWP_TEST_PROFILE
  gemmlowp::RegisterCurrentThreadForProfiling();
  gemmlowp::StartProfiling();
#endif

  float total_time = 0;

  // We don't record the first repetition, it's just warm-up.
  for (int r = 0; r < repeat + 1; r++) {
    std::cout << "repetition " << r + 1 << "/" << repeat + 1 << "...\r"
              << std::flush;
    for (int gemm_index = 0; gemm_index < gemm_count; ++gemm_index) {
      float gemm_time = 0;
      const int rows = googlenet_gemm_sizes[(gemm_index * 3) + 1];
      const int cols = googlenet_gemm_sizes[(gemm_index * 3) + 0];
      const int depth = googlenet_gemm_sizes[(gemm_index * 3) + 2];
      if (cols > KernelForGEMM::Format::kCols / 2) {
        gemm_time =
            time_for_gemm_size<KernelForGEMM, LhsType, RhsType, ResultType>(
                context, rows, depth, cols);
      } else {
        gemm_time =
            time_for_gemm_size<KernelForGEMV, LhsType, RhsType, ResultType>(
                context, rows, depth, cols);
      }
      if (r > 0) {
        total_time += gemm_time;
      }
    }
  }

#ifdef GEMMLOWP_TEST_PROFILE
  gemmlowp::FinishProfiling();
#endif

  const float ms_per_network = (total_time / repeat) * 1000.0f;
  std::cout.precision(4);
  std::cout << "GoogLeNet GEMMs took " << ms_per_network << "ms" << std::endl;
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

  {
    gemmlowp::GemmContext context;
    std::cout << "Benchmarking typical GoogLeNet GEMMs..." << std::endl;
    gemmlowp::benchmark_googlenet(&context);
  }
}
