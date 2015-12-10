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

const std::int32_t kMinTaskSize = 10000;
const std::int32_t kMinTaskDimension = 6;
const std::int32_t kMaxCacheFriendlySize = 24 * 1024;
const std::int32_t kCacheOptimalChunkSize = 24 * 1024;

void CacheFriendlyGemm(std::uint8_t* scratch, const std::uint8_t* lhs,
                       const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                       std::int32_t k, std::int32_t lhs_offset,
                       std::int32_t rhs_offset, std::int32_t sum_offset,
                       std::int32_t multiplier, std::int32_t shift,
                       std::uint8_t* result, std::int32_t result_stride) {
  const std::int32_t rhs_size = m * k;
  if (rhs_size > kMaxCacheFriendlySize) {
    const std::int32_t optimal_m =
        std::max(1, 3 * (kCacheOptimalChunkSize / (k * 3)));
    const std::int32_t chunks_count_less_one = m / optimal_m - 1;
    const std::int32_t chunk_size = optimal_m * k;
    for (int i = 0; i < chunks_count_less_one; ++i) {
      gemm_strided(scratch, lhs, rhs + i * chunk_size, n, optimal_m, k,
                   lhs_offset, rhs_offset, sum_offset, multiplier, shift,
                   result + i * optimal_m, result_stride);
    }
    const std::int32_t m_left = m - chunks_count_less_one * optimal_m;
    gemm_strided(scratch, lhs, rhs + chunks_count_less_one * chunk_size, n,
                 m_left, k, lhs_offset, rhs_offset, sum_offset, multiplier,
                 shift, result + chunks_count_less_one * optimal_m,
                 result_stride);
  } else {
    gemm_strided(scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset, sum_offset,
                 multiplier, shift, result, result_stride);
  }
}

struct TaskRect {
  std::int32_t n_offset;
  std::int32_t n;
  std::int32_t m_offset;
  std::int32_t m;

  TaskRect(std::int32_t n_offset, std::int32_t n, std::int32_t m_offset,
           std::int32_t m)
      : n_offset(n_offset), n(n), m_offset(m_offset), m(m) {}
};

struct MetaTask : gemmlowp::Task {
  TaskRect task_rect;
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

  MetaTask(TaskRect& task_rect, std::uint8_t* scratch, const std::uint8_t* lhs,
           const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
           std::int32_t k, std::int32_t lhs_offset, std::int32_t rhs_offset,
           std::int32_t sum_offset, std::int32_t multiplier, std::int32_t shift,
           std::uint8_t* result)
      : task_rect(task_rect),
        scratch(scratch),
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
    const std::uint8_t* task_lhs = lhs + task_rect.n_offset * k;
    const std::uint8_t* task_rhs = rhs + task_rect.m_offset * k;
    std::uint8_t* task_result =
        result + task_rect.n_offset * m + task_rect.m_offset;
    CacheFriendlyGemm(scratch, task_lhs, task_rhs, task_rect.n, task_rect.m, k,
                      lhs_offset, rhs_offset, sum_offset, multiplier, shift,
                      task_result, m);
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

void PrepareTasks(std::int32_t max_tasks, std::int32_t n, std::int32_t m,
                  std::int32_t k, std::vector<internal::TaskRect>* tasks) {
  const std::int32_t max_tasks_by_size = (n * m * k) / kMinTaskSize;
  const std::int32_t max_tasks_n = n / kMinTaskDimension;
  const std::int32_t max_tasks_m = m / kMinTaskDimension;
  const std::int32_t max_tasks_dimension = std::max(max_tasks_n, max_tasks_m);

  std::int32_t real_tasks = std::max(
      1, std::min(max_tasks, std::min(max_tasks_by_size, max_tasks_dimension)));

  if (real_tasks == 1) {
    tasks->push_back(TaskRect(0, n, 0, m));
    return;
  }

  if (max_tasks_n > max_tasks_m) {
    const std::int32_t n_chunk = n / real_tasks;
    for (int i = 0; i < real_tasks - 1; ++i) {
      tasks->push_back(TaskRect(i * n_chunk, n_chunk, 0, m));
    }
    const std::int32_t last_n_offset = (real_tasks - 1) * n_chunk;
    tasks->push_back(TaskRect(last_n_offset, n - last_n_offset, 0, m));
  } else {
    const std::int32_t m_chunk = m / real_tasks;
    for (int i = 0; i < real_tasks - 1; ++i) {
      tasks->push_back(TaskRect(0, n, i * m_chunk, m_chunk));
    }
    const std::int32_t last_m_offset = (real_tasks - 1) * m_chunk;
    tasks->push_back(TaskRect(0, n, last_m_offset, m - last_m_offset));
  }
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
  if (max_threads > 1) {
    pool->CreateWorkers(max_threads - 1);
  }
  std::vector<internal::TaskRect> task_rects;
  PrepareTasks(max_threads, n, m, k, &task_rects);

  if (task_rects.size() == 1) {
    internal::CacheFriendlyGemm(scratch, lhs, rhs, n, m, k, lhs_offset,
                                rhs_offset, sum_offset, multiplier, shift,
                                result, m);
    return;
  }

  pool->counter_to_decrement_when_ready().Reset(task_rects.size() - 1);

  std::uint8_t* task_scratch = scratch;

  for (int i = 0; i < static_cast<int>(task_rects.size()) - 1; ++i) {
    auto task = new internal::MetaTask(task_rects[i], task_scratch, lhs, rhs, n,
                                       m, k, lhs_offset, rhs_offset, sum_offset,
                                       multiplier, shift, result);
    pool->StartWorker(i, task);
    task_scratch += internal::ScratchPerThread(n, m, k);
  }

  auto task = new internal::MetaTask(task_rects.back(), task_scratch, lhs, rhs,
                                     n, m, k, lhs_offset, rhs_offset,
                                     sum_offset, multiplier, shift, result);
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
