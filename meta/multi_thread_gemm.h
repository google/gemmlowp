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

#ifndef GEMMLOWP_META_MULTI_THREAD_GEMM_H_
#define GEMMLOWP_META_MULTI_THREAD_GEMM_H_

#include "../internal/multi_thread_gemm.h"
#include "single_thread_gemm.h"

namespace gemmlowp {
namespace meta {
namespace internal {

const std::int32_t kMinTaskSize = 16000;
const std::int32_t kMinTaskDimension = 4;

int ResolveMaxThreads(int max_threads) {
  if (max_threads == 0) {
    static const int hardware_threads_count =
        static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
    return hardware_threads_count;
  }
  return max_threads;
}

template <typename Executor, typename Params>
uint8_t* PrepareTask(const Params& params, int kernel_m, int kernel_n,
                     int kernel_k, uint8_t* scratch, int m_start, int m,
                     int n_start, int n, std::vector<Params>* tasks) {
  tasks->push_back(params);
  Params& task = tasks->back();
  task.scratch = scratch;

  task.m = m;
  task.lhs =
      StreamUtil<typename Params::InType, typename Params::LeftStream>::Offset(
          params.left_stream, params.lhs, m_start, 0);

  task.n = n;
  task.rhs =
      StreamUtil<typename Params::InType, typename Params::RightStream>::Offset(
          params.right_stream, params.rhs, n_start, 0);

  task.result =
      StreamUtil<typename Params::OutType, typename Params::OutputStream>::
          Offset(params.fused_kernel.output_stream, params.result, m_start,
                 n_start);

  return scratch + Executor::template EstimateScratchSize<Params>(
                       task, kernel_m, kernel_n, kernel_k);
}

template <typename MultiThreadingContext, typename Executor, typename Params>
bool PrepareTasks(MultiThreadingContext* context, const Params& params,
                  int kernel_m, int kernel_n, int kernel_k,
                  std::vector<Params>* tasks) {
  const int max_threads = ResolveMaxThreads(context->max_num_threads());
  const int max_tasks_by_size = (params.m * params.n * params.k) / kMinTaskSize;
  const int max_tasks_m = params.m / kMinTaskDimension;
  const int max_tasks_n = params.n / kMinTaskDimension;
  const int max_tasks_dimension = std::max(max_tasks_m, max_tasks_n);

  const int real_tasks = std::max(
      1,
      std::min(max_threads, std::min(max_tasks_by_size, max_tasks_dimension)));

  if (real_tasks == 1) {
    return false;
  }

  uint8_t* scratch = params.scratch;

  if (max_tasks_m > max_tasks_n) {
    const int m_chunk = params.m / real_tasks;
    for (int i = 0; i < real_tasks - 1; ++i) {
      scratch = PrepareTask<Executor, Params>(params, kernel_m, kernel_n,
                                              kernel_k, scratch, i * m_chunk,
                                              m_chunk, 0, params.n, tasks);
    }
    const int sum_m = (real_tasks - 1) * m_chunk;
    PrepareTask<Executor, Params>(params, kernel_m, kernel_n, kernel_k, scratch,
                                  sum_m, params.m - sum_m, 0, params.n, tasks);
  } else {
    const int n_chunk = params.n / real_tasks;
    for (int i = 0; i < real_tasks - 1; ++i) {
      scratch = PrepareTask<Executor, Params>(params, kernel_m, kernel_n,
                                              kernel_k, scratch, 0, params.m,
                                              i * n_chunk, n_chunk, tasks);
    }
    int sum_n = (real_tasks - 1) * n_chunk;
    PrepareTask<Executor, Params>(params, kernel_m, kernel_n, kernel_k, scratch,
                                  0, params.m, sum_n, params.n - sum_n, tasks);
  }

  return true;
}

template <typename Executor, typename Params, int kernel_m, int kernel_n,
          int kernel_k>
struct TaskRunner : gemmlowp::Task {
  TaskRunner(const Params& params) : params(params) {}

  void Run() const override {
    Gemm<Executor, Params, kernel_m, kernel_n, kernel_k>(params);
  }

  Params params;
};

}  // namespace internal

template <typename MultiThreadingContext, typename Executor, typename Params,
          int kernel_m, int kernel_n, int kernel_k>
void MultiThreadGemm(MultiThreadingContext* context, const Params& params) {
  typedef internal::TaskRunner<Executor, Params, kernel_m, kernel_n, kernel_k>
      TaskRunnerType;

  std::vector<Params> tasks;
  if (!internal::PrepareTasks<MultiThreadingContext, Executor, Params>(
          context, params, kernel_m, kernel_n, kernel_k, &tasks)) {
    Gemm<Executor, Params, kernel_m, kernel_n, kernel_k>(params);
    return;
  }

  int worker_tasks_count = tasks.size() - 1;
  auto workers_pool = context->workers_pool();

  workers_pool->CreateWorkers(worker_tasks_count);
  workers_pool->Prepare(worker_tasks_count);

  for (int i = 0; i < worker_tasks_count; ++i) {
    workers_pool->StartWorker(i, new TaskRunnerType(tasks[i]));
  }
  Gemm<Executor, Params, kernel_m, kernel_n, kernel_k>(tasks.back());
  workers_pool->Wait();
}

template <typename WorkersPool>
class SimpleContext {
 public:
  SimpleContext(int max_num_threads, WorkersPool* pool)
      : max_num_threads_(max_num_threads), pool_(pool) {}

  WorkersPool* workers_pool() { return pool_; }

  int max_num_threads() { return max_num_threads_; }

 private:
  int max_num_threads_;
  WorkersPool* pool_;
};

}  // namespace meta
}  // namespace gemmlowp

#endif  // GEMMLOWP_META_MULTI_THREAD_GEMM_H_
