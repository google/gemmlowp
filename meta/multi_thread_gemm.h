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

#include "single_thread_gemm.h"
#include "../internal/multi_thread_gemm.h"

namespace gemmlowp {
namespace meta {
namespace internal {

struct MetaTask : gemmlowp::Task {
  std::uint8_t* scratch;
  const std::uint8_t* lhs;
  const std::uint8_t* rhs;
  std::int32_t n;
  std::int32_t m;
  std::int32_t k;
  std::int32_t lhs_offset;
  std::int32_t rhs_offset;
  std::int32_t sum_offset;
  std::int32_t multiplier;
  std::int32_t shift;
  std::uint8_t* result;

  MetaTask(std::uint8_t* scratch, const std::uint8_t* lhs,
           const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
           std::int32_t k, std::int32_t lhs_offset, std::int32_t rhs_offset,
           std::int32_t sum_offset, std::int32_t multiplier, std::int32_t shift,
           std::uint8_t* result)
      : scratch(scratch),
        lhs(lhs),
        rhs(rhs),
        n(n),
        m(m),
        k(k),
        lhs_offset(lhs_offset),
        rhs_offset(rhs_offset),
        sum_offset(sum_offset),
        multiplier(multiplier),
        shift(shift),
        result(result) {}

  void Run() const override {
    gemm(scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset, sum_offset,
         multiplier, shift, result);
  }
};

std::int32_t ToMegs(std::int32_t bytes) {
  const std::int32_t MB = 1024 * 1024;
  return ((bytes + MB - 1) / (MB)) * MB;
}

std::int32_t ScratchPerThread(std::int32_t n, std::int32_t m, std::int32_t k) {
  return ToMegs((m + 32) * (k + 32) + 32 * (m + 32));
}

std::int32_t ResolveMaxThreads(std::int32_t max_threads) {
  if (max_threads == 0) {
    static const int hardware_threads_count =
        static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
    return hardware_threads_count;
  }
  return max_threads;
}

}  // namespace internal

std::int32_t RequiredScratch(std::int32_t n, std::int32_t m, std::int32_t k,
                             std::int32_t max_threads) {
  return internal::ScratchPerThread(n, m, k) *
         internal::ResolveMaxThreads(max_threads);
}

void multi_thread_gemm(gemmlowp::WorkersPool* pool, std::int32_t max_threads,
                       std::uint8_t* scratch, const std::uint8_t* lhs,
                       const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                       std::int32_t k, std::int32_t lhs_offset,
                       std::int32_t rhs_offset, std::int32_t sum_offset,
                       std::int32_t multiplier, std::int32_t shift,
                       std::uint8_t* result) {
  max_threads = internal::ResolveMaxThreads(max_threads);
  pool->CreateWorkers(max_threads - 1);

  std::int32_t max_tasks_size = (n * m * k) / 100000;
  std::int32_t max_tasks_n = n / 6;

  std::int32_t real_tasks =
      std::max(1, std::min(max_threads, std::min(max_tasks_n, max_tasks_size)));

  if (real_tasks == 1) {
    gemm(scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset, sum_offset,
         multiplier, shift, result);
    return;
  }

  std::int32_t row_chunk_size = n / real_tasks;

  pool->counter_to_decrement_when_ready().Reset(real_tasks - 1);

  std::uint8_t* task_scratch = scratch;

  for (int i = 0; i < real_tasks - 1; ++i) {
    auto task = new internal::MetaTask(
        task_scratch, lhs + i * k * row_chunk_size, rhs, row_chunk_size, m, k,
        lhs_offset, rhs_offset, sum_offset, multiplier, shift,
        result + i * m * row_chunk_size);
    pool->StartWorker(i, task);
    task_scratch += internal::ScratchPerThread(n, m, k);
  }

  auto task = new internal::MetaTask(
      task_scratch, lhs + (real_tasks - 1) * k * row_chunk_size, rhs,
      n - (real_tasks - 1) * row_chunk_size, m, k, lhs_offset, rhs_offset,
      sum_offset, multiplier, shift,
      result + (real_tasks - 1) * m * row_chunk_size);
  task->Run();
  delete task;

  pool->counter_to_decrement_when_ready().Wait();
}

}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm fast-path requires GEMMLOWP_NEON_32!"
#endif

#endif  // GEMMLOWP_META_MULTI_THREAD_GEMM_H_
