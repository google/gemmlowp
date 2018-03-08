// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
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

// multi_thread_gemm.h: Multi-threaded GEMM entry point.
// Readers note: To understand this file, it is useful to first
// read and understand the much simpler single_thread_gemm.h.

#ifndef GEMMLOWP_INTERNAL_MULTI_THREAD_GEMM_H_
#define GEMMLOWP_INTERNAL_MULTI_THREAD_GEMM_H_

#include <vector>

#include "single_thread_gemm.h"

namespace gemmlowp {

// On X86 and ARM platforms we enable a busy-wait spinlock before waiting on a
// pthread conditional variable. In order to implement that correctly we need
// to put some explicit memory load/store barriers.

#if defined(GEMMLOWP_ALLOW_INLINE_ASM) && !defined(GEMMLOWP_NO_BUSYWAIT) && \
    (defined(GEMMLOWP_ARM) || defined(GEMMLOWP_X86))

#define GEMMLOWP_USE_BUSYWAIT

const int kMaxBusyWaitNOPs = 32 * 1000 * 1000;

#define GEMMLOWP_NOP "nop\n"

#define GEMMLOWP_STRING_CONCAT_4(X) X X X X
#define GEMMLOWP_NOP4 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP)
#define GEMMLOWP_NOP16 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP4)
#define GEMMLOWP_NOP64 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP16)

inline int Do256NOPs() {
  asm volatile(GEMMLOWP_NOP64);
  return 64;
}

#undef GEMMLOWP_STRING_CONCAT_4
#undef GEMMLOWP_NOP256
#undef GEMMLOWP_NOP64
#undef GEMMLOWP_NOP16
#undef GEMMLOWP_NOP4
#undef GEMMLOWP_NOP

inline void WriteBarrier() {
#if defined(_MSC_VER)
  MemoryBarrier();
#elif defined(GEMMLOWP_ARM_32)
  asm volatile("" ::: "memory");
#elif defined(GEMMLOWP_ARM_64)
  asm volatile("dmb ishst" ::: "memory");
#elif defined(GEMMLOWP_X86)
  asm volatile("sfence" ::: "memory");
#else
#error "Unsupported architecture for WriteBarrier."
#endif
}

inline void ReadBarrier() {
#if defined(_MSC_VER)
  MemoryBarrier();
#elif defined(GEMMLOWP_ARM_32)
  asm volatile("" ::: "memory");
#elif defined(GEMMLOWP_ARM_64)
  asm volatile("dmb ishld" ::: "memory");
#elif defined(GEMMLOWP_X86)
  asm volatile("lfence" ::: "memory");
#else
#error "Unsupported architecture for ReadBarrier."
#endif
}

#endif

// Waits until *var != initial_value.
//
// Returns the new value of *var. The guarantee here is that
// the return value is different from initial_value, and that that
// new value has been taken by *var at some point during the
// execution of this function. There is no guarantee that this is
// still the value of *var when this function returns, since *var is
// not assumed to be guarded by any lock.
//
// First does some busy-waiting for a fixed number of no-op cycles,
// then falls back to passive waiting for the given condvar, guarded
// by the given mutex.
//
// The idea of doing some initial busy-waiting is to help get
// better and more consistent multithreading benefits for small GEMM sizes.
// Busy-waiting help ensuring that if we need to wake up soon after having
// started waiting, then we can wake up quickly (as opposed to, say,
// having to wait to be scheduled again by the OS). On the other hand,
// we must still eventually revert to passive waiting for longer waits
// (e.g. worker threads having finished a GEMM and waiting until the next GEMM)
// so as to avoid permanently spinning.
//
template <typename T>
T WaitForVariableChange(volatile T* var, T initial_value, pthread_cond_t* cond,
                        pthread_mutex_t* mutex) {
#ifdef GEMMLOWP_USE_BUSYWAIT
  // If we are on a platform that supports it, spin for some time.
  {
    int nops = 0;
    // First, trivial case where the variable already changed value.
    T new_value = *var;
    if (new_value != initial_value) {
      ReadBarrier();
      return new_value;
    }
    // Then try busy-waiting.
    while (nops < kMaxBusyWaitNOPs) {
      nops += Do256NOPs();
      new_value = *var;
      if (new_value != initial_value) {
        ReadBarrier();
        return new_value;
      }
    }
  }
#endif

  // Finally, do real passive waiting.
  pthread_mutex_lock(mutex);
  T new_value;
  while ((new_value = *var) == initial_value) {
    pthread_cond_wait(cond, mutex);
  }
  assert(new_value != initial_value);
  pthread_mutex_unlock(mutex);
  return new_value;
}

// A BlockingCounter lets one thread to wait for N events to occur.
// This is how the master thread waits for all the worker threads
// to have finished working.
class BlockingCounter {
 public:
  BlockingCounter() : count_(0), initial_count_(0) {
    pthread_cond_init(&cond_, nullptr);
    pthread_mutex_init(&mutex_, nullptr);
  }

  ~BlockingCounter() {
    pthread_cond_destroy(&cond_);
    pthread_mutex_destroy(&mutex_);
  }

  // Sets/resets the counter; initial_count is the number of
  // decrementing events that the Wait() call will be waiting for.
  void Reset(std::size_t initial_count) {
    pthread_mutex_lock(&mutex_);
    assert(count_ == 0);
    initial_count_ = initial_count;
    count_ = initial_count_;
    pthread_mutex_unlock(&mutex_);
  }

  // Decrements the counter; if the counter hits zero, signals
  // the thread that was waiting for that, and returns true.
  // Otherwise (if the decremented count is still nonzero),
  // returns false.
  bool DecrementCount() {
    pthread_mutex_lock(&mutex_);
    assert(count_ > 0);
    count_--;
#ifdef GEMMLOWP_USE_BUSYWAIT
    WriteBarrier();
#endif
    if (count_ == 0) {
      pthread_cond_signal(&cond_);
    }
    bool retval = count_ == 0;
    pthread_mutex_unlock(&mutex_);
    return retval;
  }

  // Waits for the N other threads (N having been set by Reset())
  // to hit the BlockingCounter.
  void Wait() {
    ScopedProfilingLabel label("BlockingCounter::Wait");
    while (count_) {
#ifdef GEMMLOWP_USE_BUSYWAIT
      ReadBarrier();
#else
      // This is likely unnecessary, but is kept to ensure regressions are not
      // introduced.
#ifndef _WIN32
      asm volatile("" ::: "memory");
#endif
#endif
      const std::size_t count_value = count_;
      if (count_value) {
        WaitForVariableChange(&count_, count_value, &cond_, &mutex_);
      }
    }
  }

 private:
  pthread_cond_t cond_;
  pthread_mutex_t mutex_;
  std::size_t count_;
  std::size_t initial_count_;
};

// A workload for a worker.
struct Task {
  Task() : local_allocator(nullptr) {}
  virtual ~Task() {}
  virtual void Run() = 0;
  Allocator* local_allocator;
};

// A worker thread.
class Worker {
 public:
  enum class State {
    ThreadStartup,  // The initial state before the thread main loop runs.
    Ready,          // Is not working, has not yet received new work to do.
    HasWork,        // Has work to do.
    ExitAsSoonAsPossible  // Should exit at earliest convenience.
  };

  explicit Worker(BlockingCounter* counter_to_decrement_when_ready)
      : task_(nullptr),
        state_(State::ThreadStartup),
        counter_to_decrement_when_ready_(counter_to_decrement_when_ready) {
    pthread_cond_init(&state_cond_, nullptr);
    pthread_mutex_init(&state_mutex_, nullptr);
    pthread_create(&thread_, nullptr, ThreadFunc, this);
  }

  ~Worker() {
    ChangeState(State::ExitAsSoonAsPossible);
    pthread_join(thread_, nullptr);
    pthread_cond_destroy(&state_cond_);
    pthread_mutex_destroy(&state_mutex_);
  }

  // Changes State; may be called from either the worker thread
  // or the master thread; however, not all state transitions are legal,
  // which is guarded by assertions.
  void ChangeState(State new_state) {
    ScopedProfilingLabel label("Worker::ChangeState");
    pthread_mutex_lock(&state_mutex_);
    assert(new_state != state_);
    switch (state_) {
      case State::ThreadStartup:
        assert(new_state == State::Ready);
        break;
      case State::Ready:
        assert(new_state == State::HasWork ||
               new_state == State::ExitAsSoonAsPossible);
        break;
      case State::HasWork:
        assert(new_state == State::Ready ||
               new_state == State::ExitAsSoonAsPossible);
        break;
      default:
        abort();
    }
    state_ = new_state;
    pthread_cond_signal(&state_cond_);
    if (state_ == State::Ready) {
      counter_to_decrement_when_ready_->DecrementCount();
    }
    pthread_mutex_unlock(&state_mutex_);
  }

  // Thread entry point.
  void ThreadFunc() {
    ScopedProfilingLabel label("Worker::ThreadFunc");
    RegisterCurrentThreadForProfiling();

    ChangeState(State::Ready);

    // Thread main loop
    while (true) {
      // Get a state to act on
      // In the 'Ready' state, we have nothing to do but to wait until
      // we switch to another state.
      State state_to_act_upon = WaitForVariableChange(
          &state_, State::Ready, &state_cond_, &state_mutex_);

      // We now have a state to act on, so act.
      switch (state_to_act_upon) {
        case State::HasWork:
          // Got work to do! So do it, and then revert to 'Ready' state.
          assert(task_);
          task_->Run();
          task_ = nullptr;
          ChangeState(State::Ready);
          break;
        case State::ExitAsSoonAsPossible:
          return;
        default:
          abort();
      }
    }
  }

  static void* ThreadFunc(void* arg) {
    static_cast<Worker*>(arg)->ThreadFunc();
    return nullptr;
  }

  // Called by the master thead to give this worker work to do.
  // It is only legal to call this if the worker
  void StartWork(Task* task) {
    assert(!task_);
    task->local_allocator = &local_allocator_;
    task_ = task;
#ifdef GEMMLOWP_USE_BUSYWAIT
    WriteBarrier();
#endif
    assert(state_ == State::Ready);
    ChangeState(State::HasWork);
  }

 private:
  // The underlying thread.
  pthread_t thread_;

  // The task to be worked on.
  Task* task_;

  // The condition variable and mutex guarding state changes.
  pthread_cond_t state_cond_;
  pthread_mutex_t state_mutex_;

  // The state enum tells if we're currently working, waiting for work, etc.
  State state_;

  // Each thread had a local allocator so they can allocate temporary
  // buffers without blocking each other.
  Allocator local_allocator_;

  // pointer to the master's thread BlockingCounter object, to notify the
  // master thread of when this worker switches to the 'Ready' state.
  BlockingCounter* const counter_to_decrement_when_ready_;
};

// A very simple pool of workers, that only allows the very
// specific parallelization pattern that we use here:
// a fixed number of workers can be given work, and one then
// waits for all of them to finish.
//
// See MultiThreadGemmContextBase for how other WorkersPool implementations can
// be used. Note that in those implementations, StartWorker can be free to
// ignore the <index> value; that is, the caller of WorkersPool does not rely on
// <index> to order tasks with equal <index>.
class WorkersPool {
 public:
  WorkersPool() {}

  ~WorkersPool() {
    for (auto w : workers_) {
      delete w;
    }
  }

  void Execute(const std::vector<Task*>& tasks) {
    assert(tasks.size() >= 1);
    // One of the tasks will be run on the current thread.
    std::size_t workers_count = tasks.size() - 1;
    CreateWorkers(workers_count);
    assert(workers_count <= workers_.size());
    counter_to_decrement_when_ready_.Reset(workers_count);
    int n = 0;
    std::for_each(tasks.begin(), --tasks.end(),
                  [this, &n](Task* task) { workers_[n++]->StartWork(task); });
    // Execute the remaining workload immediately on the current thread.
    Task* task = tasks.back();
    task->local_allocator = &main_thread_task_allocator_;
    task->Run();
    // Wait for the workers submitted above to finish.
    counter_to_decrement_when_ready_.Wait();
    // Cleanup tasks (best to do this from the same thread that allocated
    // the memory).
    std::for_each(tasks.begin(), tasks.end(), [](Task* task) { delete task; });
  }

 private:
  // Ensures that the pool has at least the given count of workers.
  // If any new worker has to be created, this function waits for it to
  // be ready.
  void CreateWorkers(std::size_t workers_count) {
    if (workers_.size() >= workers_count) {
      return;
    }
    counter_to_decrement_when_ready_.Reset(workers_count - workers_.size());
    while (workers_.size() < workers_count) {
      workers_.push_back(new Worker(&counter_to_decrement_when_ready_));
    }
    counter_to_decrement_when_ready_.Wait();
  }

  // copy construction disallowed
  WorkersPool(const WorkersPool&) = delete;

  // The workers in this pool. They are owned by the pool:
  // the pool creates workers and destroys them in its destructor.
  std::vector<Worker*> workers_;

  // The BlockingCounter used to wait for the workers.
  BlockingCounter counter_to_decrement_when_ready_;

  // For N-threaded operations, we will use only N-1 worker threads
  // while the last task will be run directly on the main thread.
  // It will then use this main_thread_task_allocator_; having a
  // dedicated allocator for that (separate from the base allocator_)
  // allows to use the same code for all tasks regardless of which
  // thread they run on.
  Allocator main_thread_task_allocator_;
};

// The task we use to implement a multi-threaded Gemm: a block of the
// RHS has been packed by the master thread; each worker thread
// then has to pack a block of the LHS and accumulate the Gemm of these
// packed LHS and RHS blocks.
template <typename KernelFormat, typename InputScalar, typename OutputScalar,
          typename BitDepthParams, MapOrder LhsOrder, MapOrder RhsOrder,
          MapOrder ResultOrder, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType, typename GemmContextType>
struct GemmWithPackedRhsTask : Task {
  typedef PackedSideBlock<typename KernelFormat::Lhs> PackedLhs;
  typedef PackedSideBlock<typename KernelFormat::Rhs> PackedRhs;
  GemmWithPackedRhsTask(GemmContextType* _context, const KernelBase& _kernel,
                        const MatrixMap<const InputScalar, LhsOrder>& _lhs,
                        const PackedRhs& _packed_rhs,
                        MatrixMap<OutputScalar, ResultOrder>* _result,
                        const MatrixBlockBounds& _result_block,
                        const LhsOffset& _lhs_offset,
                        const RhsOffset& _rhs_offset,
                        const BlockParams& _block_params,
                        const OutputPipelineType& _output_pipeline)
      : context(_context),
        kernel(_kernel),
        lhs(_lhs),
        packed_rhs(_packed_rhs),
        result(*_result),
        result_block(_result_block),
        lhs_offset(_lhs_offset),
        rhs_offset(_rhs_offset),
        block_params(_block_params),
        output_pipeline(_output_pipeline) {}

  void Run() override {
    ScopedProfilingLabel label("GemmWithPackedRhsTask");

    const int rows = result_block.rows;
    const int cols = result_block.cols;
    const int depth = lhs.cols();

    PackedLhs packed_lhs(Side::Lhs, local_allocator, block_params);

    PackedResult packed_result(local_allocator, block_params);

    local_allocator->Commit();

    for (int c = 0; c < cols; c += block_params.l2_cols) {
      int cs = std::min(block_params.l2_cols, cols - c);

      for (int r = 0; r < rows; r += block_params.l2_rows) {
        int rs = std::min(block_params.l2_rows, rows - r);

        PackLhs(&packed_lhs, lhs.block(r, 0, rs, depth));

        Compute(kernel, block_params, &packed_result, packed_lhs, packed_rhs,
                depth);

        auto curr_result_block = MatrixBlockBounds(
            result_block.start_row + r, result_block.start_col + c, rs, cs);
        UnpackResult<KernelFormat>(
            &result, curr_result_block, packed_result, depth,
            packed_lhs.sums_of_each_slice(), packed_rhs.sums_of_each_slice(),
            lhs_offset.block(curr_result_block.start_row, rs),
            rhs_offset.block(curr_result_block.start_col, cs), output_pipeline);
      }
    }

    local_allocator->Decommit();
  }

  const GemmContextType* context;
  const KernelBase& kernel;
  const MatrixMap<const InputScalar, LhsOrder> lhs;
  const PackedRhs packed_rhs;
  MatrixMap<OutputScalar, ResultOrder> result;
  const MatrixBlockBounds result_block;
  const LhsOffset& lhs_offset;
  const RhsOffset& rhs_offset;
  const BlockParams& block_params;
  const OutputPipelineType& output_pipeline;
};

// This base class for multi-threading allows subclasses to implement their own
// workers_pool() method.  See MultiThreadGemmContext below for an example;
// any other implementation of workers_pool() must return an object with the
// same public methods as WorkersPool.
class MultiThreadGemmContextBase : public SingleThreadGemmContext {
 public:
  void set_max_num_threads(int n) { max_num_threads_ = n; }

  int max_num_threads() const { return max_num_threads_; }

 protected:
  // The maximum number of worker threads to use (including
  // the master thread).
  // The default value 1 means single-threading. That is the default
  // because gemmlowp's primary target is mobile hardware, where thermal
  // constraints usually mean that it may not be realistic to use more
  // than 1 CPU core even if multiple cores are present.
  // The special value 0 means try to detect the number of hardware threads.
  // Note: this assumes that all CPU cores are equivalent. That assumption
  // is defeated on big.LITTLE ARM devices, where we have no API to query
  // the number of big cores (which is typically what we would want to use,
  // leaving aside above-mentioned thermal issues). That is the other reason
  // why the best compromise here is to let max_num_threads_ default to 1,
  // so users who want multi-threading have to make the decision of how many
  // threads to use by themselves.
  int max_num_threads_ = 1;
};

class MultiThreadGemmContext : public MultiThreadGemmContextBase {
 public:
  WorkersPool* workers_pool() { return &workers_pool_; }

 private:
  // The workers pool used by MultiThreadGemm. Making
  // this part of the context allows it to be persistent,
  // avoiding recreating threads on every Gemm.
  WorkersPool workers_pool_;
};

// Determines how many threads should be used for a given Gemm
// operation.
template <int KernelRows>
inline int HowManyThreads(int max_num_threads, int rows, int cols, int depth) {
  // Early-exit in the default case where multi-threading is disabled.
  if (max_num_threads == 1) {
    return 1;
  }

  // Determine the maximum number of threads.
  int max_count = GetHardwareConcurrency(max_num_threads);

  // Basic calculation: take into account max pool size, and
  // how many rows we have to feed our kernel.
  // The motivation for an absolute minimum number of rows per thread,
  // potentially higher than KernelRows, is that very thin thread workload
  // currently defeat assumptions of the AddMod generator, resulting
  // in substantial bias in TestWithRealData on 24 threads.
  // Ideally, the AddMod generator should be aware of global (r,c) coordinates
  // so as to be independent of the number of threads.
  static const int AbsoluteMinRowsPerThread = 16;
  static const int MinRowsPerThread = KernelRows > AbsoluteMinRowsPerThread
                                          ? KernelRows
                                          : AbsoluteMinRowsPerThread;
  int thread_count = std::min(max_count, CeilQuotient(rows, MinRowsPerThread));

  // At this point for small products we already have thread_count==1 so
  // we can avoid doing more work; otherwise, we still want to check
  // that the cubic size (rows*cols*depth) is big enough to keep
  // workers_ busy.
  if (thread_count > 1) {
    // Empirically determined value.
    static const std::uint64_t min_cubic_size_per_thread = 64 * 1024;

    // We can only multiply two out of three sizes without risking overflow
    const std::uint64_t cubic_size =
        std::uint64_t(rows) * std::uint64_t(cols) * std::uint64_t(depth);

    thread_count =
        std::min(thread_count, int(cubic_size / min_cubic_size_per_thread));

    if (thread_count < 1) {
      thread_count = 1;
    }
  }

  assert(thread_count > 0 && thread_count <= max_count);
  return thread_count;
}

// The main multi-threaded Gemm function.
// To understand it, first read the code of SingleThreadGemm().
// The parallelization scheme used here is to have this master function
// pack a block of RHS and then start worker threads to pack a block of LHS
// each, and accumulate the corresponding products.
template <typename KernelFormat, typename InputScalar, typename OutputScalar,
          typename BitDepthParams, MapOrder LhsOrder, MapOrder RhsOrder,
          MapOrder ResultOrder, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType, typename GemmContextType>
void MultiThreadGemm(GemmContextType* context, const KernelBase& kernel,
                     const MatrixMap<const InputScalar, LhsOrder>& lhs,
                     const MatrixMap<const InputScalar, RhsOrder>& rhs,
                     MatrixMap<OutputScalar, ResultOrder>* result,
                     const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                     const OutputPipelineType& output_pipeline) {
  ScopedProfilingLabel label("gemmlowp::MultiThreadGemm");

  assert(lhs.cols() == rhs.rows());

  int rows = result->rows();
  int cols = result->cols();
  int depth = lhs.cols();

  // zero sizes should have been caught earlier and early-returned.
  assert(rows > 0);
  assert(cols > 0);
  assert(depth > 0);

  // The case of rows<cols should have been caught earlier and transposed.
  assert(rows >= cols);

  const int thread_count = HowManyThreads<KernelFormat::kRows>(
      context->max_num_threads(), rows, cols, depth);
  if (thread_count == 1) {
    return SingleThreadGemm<KernelFormat, InputScalar, OutputScalar,
                            BitDepthParams>(context, kernel, lhs, rhs, result,
                                            lhs_offset, rhs_offset,
                                            output_pipeline);
  }
  assert(thread_count > 1);

  // Simple 1:1 mapping of tasks to physical cores, which is very important
  // to getting good multithreaded performance, specially for not-very-large
  // GEMMs, and especially on Android.
  const int task_count = thread_count;

  Allocator* allocator = context->allocator();
  auto* workers_pool = context->workers_pool();

  BlockParams block_params;
  block_params.Init<KernelFormat>(
      rows, cols, depth, task_count, context->l1_bytes_to_use(),
      context->l2_bytes_to_use(), context->l2_rhs_factor());

  PackedSideBlock<typename KernelFormat::Rhs> packed_rhs(Side::Rhs, allocator,
                                                         block_params);
  allocator->Commit();

  // We loop over large blocks of the RHS.
  for (int c = 0; c < cols; c += block_params.l2_cols) {
    int cs = std::min(block_params.l2_cols, cols - c);

    // Pack a large block of the RHS.
    PackRhs(&packed_rhs, rhs.block(0, c, depth, cs));

    // Give work to each worker.
    std::vector<Task*> tasks;
    int next_start_row = 0;
    for (int n = 0; n < task_count; ++n) {
      int start_row = next_start_row;
      next_start_row = std::min(
          rows, RoundUp<KernelFormat::kRows>(rows * (n + 1) / task_count));

      int block_rows = next_start_row - start_row;
      auto lhs_block = lhs.block(start_row, 0, block_rows, depth);
      typedef GemmWithPackedRhsTask<KernelFormat, InputScalar, OutputScalar,
                                    BitDepthParams, LhsOrder, RhsOrder,
                                    ResultOrder, LhsOffset, RhsOffset,
                                    OutputPipelineType, GemmContextType>
          TaskType;
      tasks.push_back(
          new TaskType(context, kernel, lhs_block, packed_rhs, result,
                       MatrixBlockBounds(start_row, c, block_rows, cs),
                       lhs_offset, rhs_offset, block_params, output_pipeline));
    }
    // Execute the work on the workers (and partially on this thread).
    workers_pool->Execute(tasks);
  }

  allocator->Decommit();
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_MULTI_THREAD_GEMM_H_
