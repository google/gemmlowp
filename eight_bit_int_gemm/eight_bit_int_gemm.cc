// Copyright 2014 Google Inc. All Rights Reserved.
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

#include "eight_bit_int_gemm/eight_bit_int_gemm.h"

// gemmlowp symbols should have hidden visibility.
// currently this is ensured in the build system by
// passing -finlines-visibility-hidden. TODO: it would be
// safer to hardcode it here with some #pragma's.
#include "public/gemmlowp.h"

namespace gemmlowp {

namespace eight_bit_int_gemm {

namespace {

// To be used as template parameter for GlobalLock.
// GlobalLock<EightBitIntGemmLockId> is the global lock
// on EightBitIntGemm entry points, protecting
// EightBitIntGemm's global state.
struct EightBitIntGemmLockId;

// Global state: consists of one global GemmContext instance.
GemmContext* global_context;

GemmContext* GetOrCreateGlobalContext() {
  if (!global_context) {
    global_context = new GemmContext;
  }
  return global_context;
}

void DestroyGlobalContext() {
  delete global_context;
  global_context = nullptr;
}

}  // end anonymous namespace

// Public interface entry points

void EightBitIntGemm(int m, int n, int k, const std::uint8_t* a,
                     std::int32_t a_offset, int lda, const std::uint8_t* b,
                     std::int32_t b_offset, int ldb, std::uint8_t* c,
                     std::int32_t c_offset, std::int32_t c_mult_int,
                     std::int32_t c_shift, int ldc) {
  AutoGlobalLock<EightBitIntGemmLockId> lock;
  GemmContext* context = GetOrCreateGlobalContext();

  MatrixMap<const std::uint8_t, MapOrder::RowMajor> lhs(b, n, k, ldb);
  MatrixMap<const std::uint8_t, MapOrder::ColMajor> rhs(a, k, m, lda);
  MatrixMap<std::uint8_t, MapOrder::ColMajor> result(c, n, m, ldc);

  const int lhs_offset = b_offset;
  const int rhs_offset = a_offset;
  const int result_offset = c_offset;
  const int result_mult_int = c_mult_int;
  const int result_shift = c_shift;

  Gemm(context, lhs, rhs, &result, lhs_offset, rhs_offset, result_offset,
       result_mult_int, result_shift);
}

void SetMaxNumThreads(int n) {
  AutoGlobalLock<EightBitIntGemmLockId> lock;
  GemmContext* context = GetOrCreateGlobalContext();
  context->set_max_num_threads(n);
}

void FreePersistentResources() {
  AutoGlobalLock<EightBitIntGemmLockId> lock;
  DestroyGlobalContext();
}

}  // namespace eight_bit_int_gemm

}  // namespace gemmlowp
