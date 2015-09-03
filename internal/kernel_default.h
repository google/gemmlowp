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

// kernel_default.h: Chooses default GEMM and GEMV kernels for the
// host platform.

#ifndef GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_
#define GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_

#include "common.h"
#include "../public/bit_depth.h"

namespace gemmlowp {
template <BitDepthSetting BitDepth>
struct DefaultKernelForGemm {};
template <BitDepthSetting BitDepth>
struct DefaultKernelForGemv {};
}

#define GEMMLOWP_SET_DEFAULT_KERNEL(op, bit_depth, kernel)             \
  namespace gemmlowp {                                                 \
  template <>                                                          \
  struct DefaultKernelFor##op<BitDepthSetting::bit_depth> : kernel {}; \
  }

#if defined GEMMLOWP_NEON32
#include "kernel_neon.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, L8R8, NEON32Kernel12x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, L7R5,
                            NEON32Kernel12x4Depth2Assuming12BitProducts)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, L8R8, NEONKernel4Nx1Depth2<3>)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, L7R5, NEONKernel4Nx1Depth2<3>)
#elif defined GEMMLOWP_NEON64
#include "kernel_neon.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, L8R8, NEON64Kernel12x8Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, L7R5, NEON64Kernel12x8Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, L8R8, NEONKernel4Nx1Depth2<3>)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, L7R5, NEONKernel4Nx1Depth2<3>)
#elif defined GEMMLOWP_SSE32
#include "kernel_SSE.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, L8R8, SSE32Kernel4x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, L7R5, SSE32Kernel4x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, L8R8, SSE32Kernel4x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, L7R5, SSE32Kernel4x4Depth2)
#elif defined GEMMLOWP_SSE64
#include "kernel_SSE.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, L8R8, SSE64Kernel12x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, L7R5, SSE64Kernel12x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, L8R8, SSE64Kernel12x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, L7R5, SSE64Kernel12x4Depth2)
#else
#include "kernel_reference.h"
namespace gemmlowp {
typedef ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<4, 4>, 2>,
                                     KernelSideFormat<CellFormat<4, 4>, 2> > >
    DefaultReferenceKernel;
}
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, L8R8, DefaultReferenceKernel)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, L7R5, DefaultReferenceKernel)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, L8R8, DefaultReferenceKernel)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, L7R5, DefaultReferenceKernel)
#endif

#endif  // GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_
