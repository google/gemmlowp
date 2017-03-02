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

// kernel_default.h: Chooses default GEMM and GEMV kernels for the
// host platform.

#ifndef GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_
#define GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_

#include "../public/bit_depth.h"
#include "common.h"
#include "kernel_reference.h"

namespace gemmlowp {

enum class KernelFamily { Gemm, Gemv };

template <KernelFamily Family, bool MaxProductIsLessThan4096,
          bool LhsAlwaysNonzero>
struct DefaultKernelImpl {};

// Partial specialization implementing the logic that if we want to use
// a kernel for LhsAlwaysNonzero but do not have such a kernel, then we fall
// back to a generic kernel not taking advantage of LhsAlwaysNonzero.
template <KernelFamily Family, bool LhsAlwaysNonzero>
struct DefaultKernelImpl<Family, true, LhsAlwaysNonzero>
    : DefaultKernelImpl<Family, false, LhsAlwaysNonzero> {};

// Partial specialization implementing the logic that if we want to use
// a kernel for MaxProductIsLessThan4096 but do not have such a kernel, then we
// fall back to a generic kernel not taking advantage of
// MaxProductIsLessThan4096.
template <KernelFamily Family, bool MaxProductIsLessThan4096>
struct DefaultKernelImpl<Family, MaxProductIsLessThan4096, true>
    : DefaultKernelImpl<Family, MaxProductIsLessThan4096, false> {};

template <KernelFamily Family, typename BitDepthParams>
struct DefaultKernel
    : DefaultKernelImpl<Family,
                        (BitDepthParams::LhsRange::kMaxValue *
                             BitDepthParams::RhsRange::kMaxValue <
                         4096),
                        (BitDepthParams::LhsRange::kMinValue > 0)> {};

}  // end namespace gemmlowp

#define GEMMLOWP_SET_DEFAULT_KERNEL(Op, MaxProductIsLessThan4096,      \
                                    LhsAlwaysNonzero, Kernel)          \
  namespace gemmlowp {                                                 \
  template <>                                                          \
  struct DefaultKernelImpl<KernelFamily::Op, MaxProductIsLessThan4096, \
                           LhsAlwaysNonzero> : Kernel {};              \
  }

#if defined GEMMLOWP_NEON_32
#include "kernel_neon.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, false, false, NEON_32_Kernel12x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, true, false,
                            NEON_32_Kernel12x4Depth2Assuming12BitProducts)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, false, false, NEONKernel4Nx1Depth2<3>)
#elif defined GEMMLOWP_NEON_64
#include "kernel_neon.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, false, false, NEON_64_Kernel12x8Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, false, true,
                            NEON_64bit_GEMM_Int8Operands_LhsNonzero)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, false, false, NEONKernel4Nx1Depth2<3>)
#elif defined GEMMLOWP_SSE4_32
#include "kernel_sse.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, false, false, SSE4_32_Kernel4x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, false, false, SSE4_32_Kernel4x4Depth2)
#elif defined GEMMLOWP_SSE4_64
#include "kernel_sse.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, false, false, SSE4_64_Kernel12x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, false, false, SSE4_64_Kernel12x4Depth2)
#else
#ifndef GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
#error \
    "SIMD not enabled, you'd be getting a slow software fallback. Consider \
enabling SIMD extensions (for example using -msse4 if you're on modern x86). \
If that's not an option, and you would like to continue with the \
slow fallback, define GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK."
#endif
#include "kernel_reference.h"
namespace gemmlowp {
typedef ReferenceKernel<KernelFormat<
    KernelSideFormat<CellFormat<4, 16, CellOrder::WidthMajor>, 1>,
    KernelSideFormat<CellFormat<4, 16, CellOrder::WidthMajor>, 1> > >
    DefaultReferenceKernel;
}
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, false, false, DefaultReferenceKernel)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, false, false, DefaultReferenceKernel)
#endif

#endif  // GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_
