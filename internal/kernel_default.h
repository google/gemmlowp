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

#if defined GEMMLOWP_NEON32
#include "kernel_neon.h"
namespace gemmlowp {
typedef NEON32Kernel12x4Depth2 DefaultKernelForGEMM;
typedef NEON32Kernel8x1Depth4 DefaultKernelForGEMV;
}
#elif defined GEMMLOWP_NEON64
#include "kernel_neon.h"
namespace gemmlowp {
typedef NEON64Kernel12x4Depth2 DefaultKernelForGEMM;
// TODO (benoitjacob): For now we only have a GEMM kernel, we don't have
// a GEMV kernel on Aarch64, so we just use our GEMM kernel there,
// which is inefficient but not as inefficient as using a reference kernel.
typedef NEON64Kernel12x4Depth2 DefaultKernelForGEMV;
}
#else
#include "kernel_reference.h"
namespace gemmlowp {
typedef ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<4, 4>, 2>,
                                     KernelSideFormat<CellFormat<4, 4>, 2> > >
    DefaultKernelForGEMM;
typedef ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<4, 4>, 2>,
                                     KernelSideFormat<CellFormat<1, 4>, 1> > >
    DefaultKernelForGEMV;
}
#endif

#endif  // GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_
