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

#ifndef GEMMLOWP_META_MULTI_THREAD_TRANSFORM_H_
#define GEMMLOWP_META_MULTI_THREAD_TRANSFORM_H_

#include "multi_thread_common.h"
#include "single_thread_transform.h"

namespace gemmlowp {
namespace meta {
namespace internal {

const int kTransformTaskOverhead = 128000;
const int kMinTransformTaskSize = 32000;

template <typename MultiThreadingContext, typename Params>
inline bool PrepareTransform1DTasks(MultiThreadingContext* context,
                                    const Params& params, int kernel_size,
                                    std::vector<Params>* tasks) {
  typedef Transform1DUtil<typename Params::InType, typename Params::OutType,
                          typename Params::Kernel>
      Util;

  const int max_threads = ResolveMaxThreads(context->max_num_threads());
  const int task_size = Util::EstimateComputeCost(params.kernel);
  const int max_tasks_by_size =
      (task_size - kTransformTaskOverhead) / kMinTransformTaskSize;

  const int real_tasks = std::max(1, std::min(max_threads, max_tasks_by_size));

  if (real_tasks == 1) {
    return false;
  }

  const int chunk = params.kernel.count / real_tasks;
  for (int i = 0; i < real_tasks - 1; ++i) {
    tasks->push_back(params);
    Params& task = tasks->back();
    task.kernel.count = chunk;
    task.input = Util::OffsetInput(params.kernel, params.input, i * chunk);
    task.output = Util::OffsetOutput(params.kernel, params.output, i * chunk);
  }
  tasks->push_back(params);
  Params& task = tasks->back();
  const int sum_chunk = (real_tasks - 1) * chunk;
  task.kernel.count = params.kernel.count - sum_chunk;
  task.input = Util::OffsetInput(params.kernel, params.input, sum_chunk);
  task.output = Util::OffsetOutput(params.kernel, params.output, sum_chunk);
  return true;
}

template <typename Params, int kernel_size>
struct Transform1DTaskRunner : gemmlowp::Task {
  Transform1DTaskRunner(const Params& params) : params(params) {}

  void Run() const override { Transform1D<Params, kernel_size>(params); }

  Params params;
};

}  // namespace internal

template <typename MultiThreadingContext, typename Params, int kernel_size>
inline void MultiThreadTransform1D(MultiThreadingContext* context,
                                   const Params& params) {
  typedef internal::Transform1DTaskRunner<Params, kernel_size> TaskRunnerType;

  std::vector<Params> tasks;
  if (!internal::PrepareTransform1DTasks<MultiThreadingContext, Params>(
          context, params, kernel_size, &tasks)) {
    Transform1D<Params, kernel_size>(params);
    return;
  }

  const int worker_tasks_count = tasks.size() - 1;
  auto workers_pool = context->workers_pool();

  workers_pool->CreateWorkers(worker_tasks_count);
  workers_pool->Prepare(worker_tasks_count);

  for (int i = 0; i < worker_tasks_count; ++i) {
    workers_pool->StartWorker(i, new TaskRunnerType(tasks[i]));
  }
  Transform1D<Params, kernel_size>(tasks.back());
  workers_pool->Wait();
}

}  // namespace meta
}  // namespace gemmlowp

#endif  // GEMMLOWP_META_MULTI_THREAD_TRANSFORM_H_
