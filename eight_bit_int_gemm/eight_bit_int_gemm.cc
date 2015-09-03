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

#include "eight_bit_int_gemm.h"

// gemmlowp symbols should have hidden visibility.
// currently this is ensured in the build system by
// passing -finlines-visibility-hidden. TODO: it would be
// safer to hardcode it here with some #pragma's.
#include "../public/gemmlowp.h"

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

template <bool transpose_a, bool transpose_b, bool transpose_c>
void EightBitIntGemmImpl(GemmContext* context, int m, int n, int k,
                         const std::uint8_t* a, std::int32_t a_offset, int lda,
                         const std::uint8_t* b, std::int32_t b_offset, int ldb,
                         std::uint8_t* c, std::int32_t c_offset,
                         std::int32_t c_mult_int, std::int32_t c_shift, int ldc,
                         BitDepthSetting bit_depth) {
  const int lhs_offset = b_offset;
  const int rhs_offset = a_offset;
  const int result_offset = c_offset;
  const int result_mult_int = c_mult_int;
  const int result_shift = c_shift;

  static const MapOrder ResultOrder =
      transpose_c ? MapOrder::ColMajor : MapOrder::RowMajor;
  static const MapOrder LhsOrder =
      transpose_b == transpose_c ? MapOrder::RowMajor : MapOrder::ColMajor;
  static const MapOrder RhsOrder =
      transpose_a == transpose_c ? MapOrder::RowMajor : MapOrder::ColMajor;

  MatrixMap<const std::uint8_t, LhsOrder> lhs(b, n, k, ldb);
  MatrixMap<const std::uint8_t, RhsOrder> rhs(a, k, m, lda);
  MatrixMap<std::uint8_t, ResultOrder> result(c, n, m, ldc);

  switch (bit_depth) {
#define GEMMLOWP_HANDLE_BIT_DEPTH(AnBn, LnRn)                              \
  case BitDepthSetting::AnBn:                                              \
    Gemm<std::uint8_t, gemmlowp::BitDepthSetting::LnRn>(                   \
        context, lhs, rhs, &result, lhs_offset, rhs_offset, result_offset, \
        result_mult_int, result_shift);                                    \
    return;
    GEMMLOWP_HANDLE_BIT_DEPTH(A8B8, L8R8)
    GEMMLOWP_HANDLE_BIT_DEPTH(A5B7, L7R5)
    default:
      abort();
#undef GEMMLOWP_HANDLE_BIT_DEPTH
  }
}

}  // end anonymous namespace

// Public interface entry points

void EightBitIntGemm(bool transpose_a, bool transpose_b, bool transpose_c,
                     int m, int n, int k, const std::uint8_t* a,
                     std::int32_t a_offset, int lda, const std::uint8_t* b,
                     std::int32_t b_offset, int ldb, std::uint8_t* c,
                     std::int32_t c_offset, std::int32_t c_mult_int,
                     std::int32_t c_shift, int ldc, BitDepthSetting bit_depth) {
  AutoGlobalLock<EightBitIntGemmLockId> lock;
  GemmContext* context = GetOrCreateGlobalContext();

#define GEMMLOWP_HANDLE_CASE(ta, tb, tc)                                    \
  if (transpose_a == ta && transpose_b == tb && transpose_c == tc) {        \
    EightBitIntGemmImpl<ta, tb, tc>(context, m, n, k, a, a_offset, lda, b,  \
                                    b_offset, ldb, c, c_offset, c_mult_int, \
                                    c_shift, ldc, bit_depth);               \
  }

  GEMMLOWP_HANDLE_CASE(false, false, false)
  GEMMLOWP_HANDLE_CASE(false, false, true)
  GEMMLOWP_HANDLE_CASE(false, true, false)
  GEMMLOWP_HANDLE_CASE(false, true, true)
  GEMMLOWP_HANDLE_CASE(true, false, false)
  GEMMLOWP_HANDLE_CASE(true, false, true)
  GEMMLOWP_HANDLE_CASE(true, true, false)
  GEMMLOWP_HANDLE_CASE(true, true, true)

#undef GEMMLOWP_HANDLE_CASE
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
