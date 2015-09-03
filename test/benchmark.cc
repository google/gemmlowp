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

#ifndef GEMMLOWP_TEST_BIT_DEPTH
#define GEMMLOWP_TEST_BIT_DEPTH L8R8
#endif

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

struct gemm_t {
  int rows, depth, cols;
  gemm_t() : rows(0), depth(0), cols(0) {}
  gemm_t(int r, int d, int c) : rows(r), depth(d), cols(c) {}
};

bool operator<(const gemm_t& a, const gemm_t& b) {
  return a.rows < b.rows ||
         (a.rows <= b.rows &&
          (a.depth < b.depth || (a.depth <= b.depth && (a.cols < b.cols))));
}

template <typename LhsType, typename RhsType, typename ResultType>
double time_for_gemms(GemmContext* context, const std::vector<gemm_t>& gemms) {
  typedef std::uint8_t Scalar;

  // set up the matrix pool

  std::size_t combined_gemm_sizes = 0;
  for (auto gemm : gemms) {
    int rows = gemm.rows;
    int depth = gemm.depth;
    int cols = gemm.cols;
    combined_gemm_sizes +=
        sizeof(Scalar) * (rows * depth + depth * cols + rows * cols);
  }

  const std::size_t pool_size = 1 + min_working_set_size / combined_gemm_sizes;

  std::vector<LhsType> lhs(pool_size * gemms.size());
  std::vector<RhsType> rhs(pool_size * gemms.size());
  std::vector<ResultType> result(pool_size * gemms.size());

  for (std::size_t i = 0; i < pool_size; i++) {
    for (std::size_t j = 0; j < gemms.size(); j++) {
      int k = i * gemms.size() + j;
      lhs[k].Resize(gemms[j].rows, gemms[j].depth);
      MakeConstant(&lhs[k], 0);
      rhs[k].Resize(gemms[j].depth, gemms[j].cols);
      MakeConstant(&rhs[k], 0);
      result[k].Resize(gemms[j].rows, gemms[j].cols);
      MakeConstant(&result[k], 0);
    }
  }

  // main benchmark loop

  int iters_at_a_time = 1;
  float time_per_iter = 0.0f;
  std::size_t pool_index = 0;

  while (true) {
    double starttime = time();
    for (int i = 0; i < iters_at_a_time; i++) {
      for (int j = 0; j < gemms.size(); j++) {
        int k = pool_index * gemms.size() + j;
        Gemm<std::uint8_t, BitDepthSetting::GEMMLOWP_TEST_BIT_DEPTH>(
            context, lhs[k].const_map(), rhs[k].const_map(), &result[k].map(),
            -75, -91, 74980, 123, 20);
      }
      pool_index++;
      if (pool_index == pool_size) {
        pool_index = 0;
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

template <typename LhsType, typename RhsType, typename ResultType>
double gflops_for_gemms(GemmContext* context,
                        const std::vector<gemm_t>& gemms) {
  const double time_per_iter =
      time_for_gemms<LhsType, RhsType, ResultType>(context, gemms);
  double ops = 0;
  for (auto gemm : gemms) {
    ops += 2.0 * gemm.rows * gemm.depth * gemm.cols;
  }
  return 1e-9 * ops / time_per_iter;
}

void benchmark(GemmContext* context) {
  std::map<gemm_t, std::vector<double>> benchmark_results;

  std::vector<gemm_t> benchmark_gemms;
  benchmark_gemms.emplace_back(10, 10, 10);
  benchmark_gemms.emplace_back(20, 20, 20);
  benchmark_gemms.emplace_back(30, 30, 30);
  benchmark_gemms.emplace_back(40, 40, 40);
  benchmark_gemms.emplace_back(50, 50, 50);
  benchmark_gemms.emplace_back(60, 60, 60);
  benchmark_gemms.emplace_back(64, 256, 147);
  benchmark_gemms.emplace_back(100, 100, 1);
  benchmark_gemms.emplace_back(100, 100, 100);
  benchmark_gemms.emplace_back(100, 1000, 100);
  benchmark_gemms.emplace_back(1000, 1000, 1);
  benchmark_gemms.emplace_back(1000, 1000, 10);
  benchmark_gemms.emplace_back(1000, 1000, 100);
  benchmark_gemms.emplace_back(1000, 1000, 1000);

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
    for (auto gemm : benchmark_gemms) {
      double gflops = 0;
      std::vector<gemm_t> unique_gemm;
      unique_gemm.push_back(gemm);
      gflops =
          gflops_for_gemms<LhsType, RhsType, ResultType>(context, unique_gemm);
      if (r > 0) {
        benchmark_results[gemm].emplace_back(gflops);
      }
    }
  }

#ifdef GEMMLOWP_TEST_PROFILE
  gemmlowp::FinishProfiling();
#endif

  std::cout << "                                                \r"
            << std::flush;

  std::cout.precision(4);

  for (auto b : benchmark_results) {
    sort(b.second.begin(), b.second.end());
    std::cout << b.first.rows << "x" << b.first.depth << "x" << b.first.cols
              << " : " << b.second.back() << " GFlops/s" << std::endl;
  }
  std::cout << std::endl;
}

void benchmark_googlenet(GemmContext* context) {
  // These are the m, n, k sizes for a typical GoogLeNet.
  const int googlenet_gemm_sizes[] = {
      12544, 64,  147, 3136, 64,   64,   3136, 192,  576,  784, 64,   192,
      784,   96,  192, 784,  128,  864,  784,  16,   192,  784, 32,   400,
      784,   32,  192, 784,  128,  256,  784,  128,  256,  784, 192,  1152,
      784,   32,  256, 784,  96,   800,  784,  64,   256,  196, 192,  480,
      196,   96,  480, 196,  204,  864,  196,  16,   480,  196, 48,   400,
      196,   64,  480, 196,  160,  508,  196,  112,  508,  196, 224,  1008,
      196,   24,  508, 196,  64,   600,  196,  64,   508,  196, 128,  512,
      196,   128, 512, 196,  256,  1152, 196,  24,   512,  196, 64,   600,
      196,   64,  512, 196,  112,  512,  196,  144,  512,  196, 288,  1296,
      196,   32,  512, 196,  64,   800,  196,  64,   512,  196, 256,  528,
      196,   160, 528, 196,  320,  1440, 196,  32,   528,  196, 128,  800,
      196,   128, 528, 49,   256,  832,  49,   160,  832,  49,  320,  1440,
      49,    48,  832, 49,   128,  1200, 49,   128,  832,  49,  384,  832,
      49,    192, 832, 49,   384,  1728, 49,   48,   832,  49,  128,  1200,
      49,    128, 832, 16,   128,  508,  1,    1024, 2048, 1,   1008, 1024,
      16,    128, 528, 1,    1024, 2048, 1,    1008, 1024, 1,   1008, 1024,
  };
  assert(sizeof(googlenet_gemm_sizes) % (3 * sizeof(googlenet_gemm_sizes[0])) ==
         0);
  const std::size_t num_googlenet_gemms =
      sizeof(googlenet_gemm_sizes) / (3 * sizeof(googlenet_gemm_sizes[0]));

  std::vector<gemm_t> googlenet_gemms(num_googlenet_gemms);
  for (std::size_t i = 0; i < num_googlenet_gemms; i++) {
    googlenet_gemms[i].rows = googlenet_gemm_sizes[3 * i + 1];
    googlenet_gemms[i].depth = googlenet_gemm_sizes[3 * i + 2];
    googlenet_gemms[i].cols = googlenet_gemm_sizes[3 * i + 0];
  }

  typedef Matrix<std::uint8_t, MapOrder::RowMajor> LhsType;
  typedef Matrix<std::uint8_t, MapOrder::ColMajor> RhsType;
  typedef Matrix<std::uint8_t, MapOrder::ColMajor> ResultType;

  std::vector<float> gemm_times;
  const double mintime = 20.0;
  std::cout << "running for " << mintime << " seconds..." << std::endl;

#ifdef GEMMLOWP_TEST_PROFILE
  gemmlowp::RegisterCurrentThreadForProfiling();
  gemmlowp::StartProfiling();
#endif

  double starttime = time();
  while (time() < starttime + mintime) {
    gemm_times.push_back(
        time_for_gemms<LhsType, RhsType, ResultType>(context, googlenet_gemms));
  }

#ifdef GEMMLOWP_TEST_PROFILE
  gemmlowp::FinishProfiling();
#endif

  std::sort(gemm_times.begin(), gemm_times.end());
  const std::size_t omit = gemm_times.size() / 4;
  float sum = 0;
  float count = 0;
  for (std::size_t i = omit; i < gemm_times.size() - omit; i++) {
    sum += gemm_times[i];
    count++;
  }
  const float avg = sum / count;
  const float ms_per_network = avg * 1000.0f;
  std::cout.precision(4);
  std::cout << "GoogLeNet GEMMs took " << ms_per_network << "ms" << std::endl;
}

}  // end namespace gemmlowp

int main() {
  {
    gemmlowp::GemmContext context;
    std::cout << "Benchmarking typical GoogLeNet GEMMs..." << std::endl;
    gemmlowp::benchmark_googlenet(&context);
  }

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
