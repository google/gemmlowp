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

// kernel_neon.h: a collection of NEON optimized kernels.
// Check in kernel_default.h which one(s) are actually used by default.
// Others are mere experiments; they are still covered by tests
// in case they might be useful some day.

#ifndef GEMMLOWP_INTERNAL_KERNEL_NEON_H_
#define GEMMLOWP_INTERNAL_KERNEL_NEON_H_

#include "kernel.h"

#include <cassert>

namespace gemmlowp {

// The kernels here are specifically arm 32bit assembly, not arm 64bit.
#ifdef GEMMLOWP_NEON32

// Our main GEMM kernel.
struct NEON32Kernel12x4Depth2 : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 2>, 3>,
                       KernelSideFormat<CellFormat<4, 2>, 1> > Format;

  const char* Name() const override { return "NEON, 12x4, depth 2"; }

  // TODO(benoitjacob): reorder function arguments so dst comes last
  void Run(std::int32_t* dst_ptr, int dst_row_stride, int dst_col_stride,
           const std::uint8_t* lhs_ptr, const std::uint8_t* rhs_ptr,
           int start_depth, int run_depth) const override {
    ScopedProfilingLabel label("optimized kernel");

    assert(dst_row_stride == 1);

    asm volatile(
        // Clear accumulator registers (see layout below)
        "vmov.s32 q4, #0\n"
        "vmov.s32 q8, q4\n"
        "vmov.s32 q12, q4\n"
        "vmov.s32 q5, q4\n"
        "vmov.s32 q9, q4\n"
        "vmov.s32 q13, q4\n"
        "vmov.s32 q6, q4\n"
        "vmov.s32 q10, q4\n"
        "vmov.s32 q14, q4\n"
        "vmov.s32 q7, q4\n"
        "vmov.s32 q11, q4\n"
        "vmov.s32 q15, q4\n"

        /* Main loop */

        "loop_NEONKernel12x4Depth2_%=:\n"

        // Overview of register layout:
        //
        // A 2x4 cell of Rhs is stored in 16bit in d0--d1 (q0).
        // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in d2--d7
        // (q1--q3).
        // A 12x4 block of accumulators is stored in 32bit in q4--q15.
        //
        //                   +-----+-----+-----+-----+
        //                   |d0[0]|d0[1]|d0[2]|d0[3]|
        //              Rhs  +-----+-----+-----+-----+
        //                   |d1[0]|d1[1]|d1[2]|d1[3]|
        //                   +-----+-----+-----+-----+
        //
        //                   |     |     |     |     |
        //
        //    Lhs            |     |     |     |     |
        //
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //
        //                            Accumulator

        // Load 1 Rhs cell of size 2x4
        "vld1.8 {d0}, [%[rhs_ptr]:64]!\n"

        // Load 3 Lhs cells of size 4x2 each
        "vld1.8 {d2}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d4}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d6}, [%[lhs_ptr]:64]!\n"

        // Expand Lhs/Rhs cells to 16 bit.
        "vmovl.u8 q0, d0\n"
        "vmovl.u8 q1, d2\n"
        "vmovl.u8 q2, d4\n"
        "vmovl.u8 q3, d6\n"

        // Multiply-accumulate, level of depth 0
        "vmlal.u16 q4, d2, d0[0]\n"
        "vmlal.u16 q5, d2, d0[1]\n"
        "vmlal.u16 q6, d2, d0[2]\n"
        "vmlal.u16 q7, d2, d0[3]\n"
        "vmlal.u16 q8, d4, d0[0]\n"
        "vmlal.u16 q9, d4, d0[1]\n"
        "vmlal.u16 q10, d4, d0[2]\n"
        "vmlal.u16 q11, d4, d0[3]\n"
        "vmlal.u16 q12, d6, d0[0]\n"
        "vmlal.u16 q13, d6, d0[1]\n"
        "vmlal.u16 q14, d6, d0[2]\n"
        "vmlal.u16 q15, d6, d0[3]\n"

        // Multiply-accumulate, level of depth 1
        "vmlal.u16 q4, d3, d1[0]\n"
        "vmlal.u16 q5, d3, d1[1]\n"
        "vmlal.u16 q6, d3, d1[2]\n"
        "vmlal.u16 q7, d3, d1[3]\n"
        "vmlal.u16 q8, d5, d1[0]\n"
        "vmlal.u16 q9, d5, d1[1]\n"
        "vmlal.u16 q10, d5, d1[2]\n"
        "vmlal.u16 q11, d5, d1[3]\n"
        "vmlal.u16 q12, d7, d1[0]\n"
        "vmlal.u16 q13, d7, d1[1]\n"
        "vmlal.u16 q14, d7, d1[2]\n"
        "vmlal.u16 q15, d7, d1[3]\n"

        // Loop. Decrement loop index (depth) by 2, since we just handled 2
        // levels of depth (Kernel::kDepth=2).
        "subs %[run_depth], #2\n"
        "bne loop_NEONKernel12x4Depth2_%=\n"

        /* end of main loop */

        /* Accumulate our local accumulator registers into the destination block
           */

        // Compute stride between consecutive columns, in bytes
        "mov r0, #4\n"  // multiply by 4 = sizeof(int32)
        "mul %[dst_col_stride], r0\n"

        // If start_depth == 0, then there is no preexisting accumulator
        // to accumulate, so we can simply store our result.
        "cmp %[start_depth], #0\n"
        "beq store_result_NEONKernel12x4Depth2_%=\n"

        "mov r0, %[dst_ptr]\n"

        // Load a column
        "mov r1, r0\n"
        "vld1.32 {d0, d1}, [r1]!\n"
        "vld1.32 {d2, d3}, [r1]!\n"
        "vld1.32 {d4, d5}, [r1]!\n"
        // Accumulate a column
        "vadd.s32 q4, q4, q0\n"
        "vadd.s32 q8, q8, q1\n"
        "vadd.s32 q12, q12, q2\n"

        "add r0, %[dst_col_stride]\n"
        // Load a column
        "mov r1, r0\n"
        "vld1.32 {d0, d1}, [r1]!\n"
        "vld1.32 {d2, d3}, [r1]!\n"
        "vld1.32 {d4, d5}, [r1]!\n"
        // Accumulate a column
        "vadd.s32 q5, q5, q0\n"
        "vadd.s32 q9, q9, q1\n"
        "vadd.s32 q13, q13, q2\n"

        "add r0, %[dst_col_stride]\n"
        // Load a column
        "mov r1, r0\n"
        "vld1.32 {d0, d1}, [r1]!\n"
        "vld1.32 {d2, d3}, [r1]!\n"
        "vld1.32 {d4, d5}, [r1]!\n"
        // Accumulate a column
        "vadd.s32 q6, q6, q0\n"
        "vadd.s32 q10, q10, q1\n"
        "vadd.s32 q14, q14, q2\n"

        "add r0, %[dst_col_stride]\n"
        // Load a column
        "mov r1, r0\n"
        "vld1.32 {d0, d1}, [r1]!\n"
        "vld1.32 {d2, d3}, [r1]!\n"
        "vld1.32 {d4, d5}, [r1]!\n"
        // Accumulate a column
        "vadd.s32 q7, q7, q0\n"
        "vadd.s32 q11, q11, q1\n"
        "vadd.s32 q15, q15, q2\n"

        "store_result_NEONKernel12x4Depth2_%=:\n"

        "mov r0, %[dst_ptr]\n"
        // Store a column
        "mov r1, r0\n"
        "vst1.32 {d8, d9}, [r1]!\n"
        "vst1.32 {d16, d17}, [r1]!\n"
        "vst1.32 {d24, d25}, [r1]!\n"
        // Store a column
        "add r0, %[dst_col_stride]\n"
        "mov r1, r0\n"
        "vst1.32 {d10, d11}, [r1]!\n"
        "vst1.32 {d18, d19}, [r1]!\n"
        "vst1.32 {d26, d27}, [r1]!\n"
        // Store a column
        "add r0, %[dst_col_stride]\n"
        "mov r1, r0\n"
        "vst1.32 {d12, d13}, [r1]!\n"
        "vst1.32 {d20, d21}, [r1]!\n"
        "vst1.32 {d28, d29}, [r1]!\n"
        // Store a column
        "add r0, %[dst_col_stride]\n"
        "mov r1, r0\n"
        "vst1.32 {d14, d15}, [r1]!\n"
        "vst1.32 {d22, d23}, [r1]!\n"
        "vst1.32 {d30, d31}, [r1]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        :  // inputs
        [start_depth] "r"(start_depth),
        [dst_col_stride] "r"(dst_col_stride)
        :  // clobbers
        "cc", "memory", "r0", "r1",
        // note: someone on internet says that quad registers are
        // unsupported in the clobber list!
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31");
  }
};

// Our main GEMV kernel.
struct NEON32Kernel8x1Depth4 : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<8, 4>, 1>,
                       KernelSideFormat<CellFormat<1, 4>, 1> > Format;

  const char* Name() const override { return "NEON, 8x1, depth 4"; }

  void Run(std::int32_t* dst_ptr, int dst_row_stride, int dst_col_stride,
           const std::uint8_t* lhs_ptr, const std::uint8_t* rhs_ptr,
           int start_depth, int run_depth) const override {
    ScopedProfilingLabel label("optimized kernel");

    assert(dst_row_stride == 1);

    asm volatile(
        // Clear accumulator registers (see layout below)
        "vmov.s32 q11, #0\n"
        "vmov.s32 q12, q11\n"

        /* Main loop */

        "loop_NEONKernel8x1Depth4_%=:\n"

        // Overview of register layout:
        //
        // A 4x1 cell of Rhs is stored in 16bit in d0.
        // A 8x4 cell Lhs is stored in 16bit in d2--d9 (q1--q4).
        // A block of accumulators of size 8x1 is stored in 32bit in q11--q12
        //
        //                         +-----+
        //                         |d0[0]|
        //                         +-----+
        //                         |d0[1]|
        //                         +-----+   Rhs
        //                         |d0[2]|
        //                         +-----+
        //                         |d0[3]|
        //                         +-----+
        //                         |     |
        //
        //       Lhs               |     |
        //
        //  +--+--+--+--+ - - - -  +-----+
        //  |d2|d4|d6|d8|          | q11 |
        //  |d2|d4|d6|d8|          | q11 |
        //  |d2|d4|d6|d8|          | q11 |
        //  |d2|d4|d6|d8|          | q11 |
        //  +--+--+--+--+ - - - -  +-----+   Accumulator
        //  |d3|d5|d7|d9|          | q12 |
        //  |d3|d5|d7|d9|          | q12 |
        //  |d3|d5|d7|d9|          | q12 |
        //  |d3|d5|d7|d9|          | q12 |
        //  +--+--+--+--+ - - - -  +-----+

        // Load 1 Rhs cell of size 4x1.
        "vldr.32 d0, [%[rhs_ptr]]\n"
        "add %[rhs_ptr], #4\n"

        // Load 1 Lhs cell of size 8x4. Each vld1 instruction loads 1 col.
        "vld1.8 {d2}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d4}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d6}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d8}, [%[lhs_ptr]:64]!\n"

        // Expand Lhs/Rhs cells to 16 bit.
        "vmovl.u8 q0, d0\n"  // d1 is unused.
        "vmovl.u8 q1, d2\n"
        "vmovl.u8 q2, d4\n"
        "vmovl.u8 q3, d6\n"
        "vmovl.u8 q4, d8\n"

        // Multiply-accumulate, level of depth 0
        "vmlal.u16 q11, d2, d0[0]\n"
        "vmlal.u16 q12, d3, d0[0]\n"

        // Multiply-accumulate, level of depth 1
        "vmlal.u16 q11, d4, d0[1]\n"
        "vmlal.u16 q12, d5, d0[1]\n"

        // Multiply-accumulate, level of depth 2
        "vmlal.u16 q11, d6, d0[2]\n"
        "vmlal.u16 q12, d7, d0[2]\n"

        // Multiply-accumulate, level of depth 3
        "vmlal.u16 q11, d8, d0[3]\n"
        "vmlal.u16 q12, d9, d0[3]\n"

        // Loop. Decrement loop index (depth) by 4, since we just handled 4
        // levels
        // of depth (Kernel::kDepth=4).
        "subs %[run_depth], #4\n"
        "bne loop_NEONKernel8x1Depth4_%=\n"

        /* end of main loop */

        /* Accumulate our local accumulator registers into the destination block
           */

        "cmp %[start_depth], #0\n"
        "beq store_result_NEONKernel8x1Depth4_%=\n"

        "mov r0, %[dst_ptr]\n"
        // Load a column
        "vld1.32 {d0, d1}, [r0]!\n"
        "vld1.32 {d2, d3}, [r0]!\n"

        // Accumulate a column
        "vadd.s32 q11, q11, q0\n"
        "vadd.s32 q12, q12, q1\n"

        "store_result_NEONKernel8x1Depth4_%=:\n"

        "mov r0, %[dst_ptr]\n"
        "vst1.32 {d22, d23}, [r0]!\n"
        "vst1.32 {d24, d25}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [run_depth] "+r"(run_depth)
        :  // inputs
        [dst_col_stride] "r"(dst_col_stride), [start_depth] "r"(start_depth),
        [dst_ptr] "r"(dst_ptr)
        :  // clobbers
        "cc", "memory", "r0",
        // note: someone on internet says that quad registers are
        // unsupported in the clobber list!
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31");
  }
};

// Another GEMV kernel, that performs best by itself;
// unfortunately GEMV is dominated by LHS packing and we haven't succeeded
// so far in making efficient packing code for this very wide (width 20)
// format.
struct NEON32Kernel20x1Depth4 : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 4>, 5>,
                       KernelSideFormat<CellFormat<1, 4>, 1> > Format;

  const char* Name() const override { return "NEON, 20x1, depth 4"; }

  void Run(std::int32_t* dst_ptr, int dst_row_stride, int dst_col_stride,
           const std::uint8_t* lhs_ptr, const std::uint8_t* rhs_ptr,
           int start_depth, int run_depth) const override {
    ScopedProfilingLabel label("optimized kernel");

    assert(dst_row_stride == 1);

    asm volatile(
        // Clear accumulator registers (see layout below)
        "vmov.s32 q11, #0\n"
        "vmov.s32 q12, q11\n"
        "vmov.s32 q13, q11\n"
        "vmov.s32 q14, q11\n"
        "vmov.s32 q15, q11\n"

        /* Main loop */

        "loop_NEONKernel20x1Depth4_%=:\n"

        // Overview of register layout:
        //
        // A 4x1 cell of Rhs is stored in 16bit in d0.
        // 5 cells, of size 4x4 each, are stored in 16bit in d2--d21.
        // A block of accumulators of size 20x1 is stored in 32bit
        // in q11--q15.
        //
        //                             +-----+
        //                             |d0[0]|
        //                             +-----+
        //                             |d0[1]|
        //                             +-----+   Rhs
        //                             |d0[2]|
        //                             +-----+
        //                             |d0[3]|
        //                             +-----+
        //                             |     |
        //
        //       Lhs                   |     |
        //
        //  +---+---+---+---+ - - - -  +-----+
        //  |d2 |d3 |d4 |d5 |          | q11 |
        //  |d2 |d3 |d4 |d5 |          | q11 |
        //  |d2 |d3 |d4 |d5 |          | q11 |
        //  |d2 |d3 |d4 |d5 |          | q11 |
        //  +---+---+---+---+ - - - -  +-----+
        //  |d6 |d7 |d8 |d9 |          | q12 |
        //  |d6 |d7 |d8 |d9 |          | q12 |
        //  |d6 |d7 |d8 |d9 |          | q12 |
        //  |d6 |d7 |d8 |d9 |          | q12 |
        //  +---+---+---+---+ - - - -  +-----+
        //  |d10|d11|d12|d13|          | q13 |
        //  |d10|d11|d12|d13|          | q13 |   Accumulator
        //  |d10|d11|d12|d13|          | q13 |
        //  |d10|d11|d12|d13|          | q13 |
        //  +---+---+---+---+ - - - -  +-----+
        //  |d14|d15|d16|d17|          | q14 |
        //  |d14|d15|d16|d17|          | q14 |
        //  |d14|d15|d16|d17|          | q14 |
        //  |d14|d15|d16|d17|          | q14 |
        //  +---+---+---+---+ - - - -  +-----+
        //  |d18|d19|d20|d21|          | q15 |
        //  |d18|d19|d20|d21|          | q15 |
        //  |d18|d19|d20|d21|          | q15 |
        //  |d18|d19|d20|d21|          | q15 |
        //  +---+---+---+---+ - - - -  +-----+

        // Load 1 Rhs cell
        "vldr.32 d0, [%[rhs_ptr]]\n"
        "add %[rhs_ptr], #4\n"

        // Load 10 Lhs cells
        "vld1.8 {d2}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d4}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d6}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d8}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d10}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d12}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d14}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d16}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d18}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d20}, [%[lhs_ptr]:64]!\n"

        // Expand Lhs/Rhs cells to 16 bit.
        "vmovl.u8 q0, d0\n"
        "vmovl.u8 q1, d2\n"
        "vmovl.u8 q2, d4\n"
        "vmovl.u8 q3, d6\n"
        "vmovl.u8 q4, d8\n"
        "vmovl.u8 q5, d10\n"
        "vmovl.u8 q6, d12\n"
        "vmovl.u8 q7, d14\n"
        "vmovl.u8 q8, d16\n"
        "vmovl.u8 q9, d18\n"
        "vmovl.u8 q10, d20\n"

        // Multiply-accumulate, level of depth 0
        "vmlal.u16 q11, d2, d0[0]\n"
        "vmlal.u16 q12, d6, d0[0]\n"
        "vmlal.u16 q13, d10, d0[0]\n"
        "vmlal.u16 q14, d14, d0[0]\n"
        "vmlal.u16 q15, d18, d0[0]\n"

        // Multiply-accumulate, level of depth 1
        "vmlal.u16 q11, d3, d0[1]\n"
        "vmlal.u16 q12, d7, d0[1]\n"
        "vmlal.u16 q13, d11, d0[1]\n"
        "vmlal.u16 q14, d15, d0[1]\n"
        "vmlal.u16 q15, d19, d0[1]\n"

        // Multiply-accumulate, level of depth 2
        "vmlal.u16 q11, d4, d0[2]\n"
        "vmlal.u16 q12, d8, d0[2]\n"
        "vmlal.u16 q13, d12, d0[2]\n"
        "vmlal.u16 q14, d16, d0[2]\n"
        "vmlal.u16 q15, d20, d0[2]\n"

        // Multiply-accumulate, level of depth 3
        "vmlal.u16 q11, d5, d0[3]\n"
        "vmlal.u16 q12, d9, d0[3]\n"
        "vmlal.u16 q13, d13, d0[3]\n"
        "vmlal.u16 q14, d17, d0[3]\n"
        "vmlal.u16 q15, d21, d0[3]\n"

        // Loop. Decrement loop index (depth) by 4, since we just handled 4
        // levels
        // of depth (Kernel::kDepth=4).
        "subs %[run_depth], #4\n"
        "bne loop_NEONKernel20x1Depth4_%=\n"

        /* end of main loop */

        /* Accumulate our local accumulator registers into the destination block
           */

        "cmp %[start_depth], #0\n"
        "beq store_result_NEONKernel20x1Depth4_%=\n"

        "mov r0, %[dst_ptr]\n"
        // Load a column
        "vld1.32 {d0, d1}, [r0]!\n"
        "vld1.32 {d2, d3}, [r0]!\n"
        "vld1.32 {d4, d5}, [r0]!\n"
        "vld1.32 {d6, d7}, [r0]!\n"
        "vld1.32 {d8, d9}, [r0]!\n"

        // Accumulate a column
        "vadd.s32 q11, q11, q0\n"
        "vadd.s32 q12, q12, q1\n"
        "vadd.s32 q13, q13, q2\n"
        "vadd.s32 q14, q14, q3\n"
        "vadd.s32 q15, q15, q4\n"

        "store_result_NEONKernel20x1Depth4_%=:\n"

        "mov r0, %[dst_ptr]\n"
        "vst1.32 {d22, d23}, [r0]!\n"
        "vst1.32 {d24, d25}, [r0]!\n"
        "vst1.32 {d26, d27}, [r0]!\n"
        "vst1.32 {d28, d29}, [r0]!\n"
        "vst1.32 {d30, d31}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [run_depth] "+r"(run_depth)
        :  // inputs
        [dst_ptr] "r"(dst_ptr), [dst_col_stride] "r"(dst_col_stride),
        [start_depth] "r"(start_depth)
        :  // clobbers
        "cc", "memory", "r0",
        // note: someone on internet says that quad registers are
        // unsupported in the clobber list!
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31");
  }
};

#endif  // GEMMLOWP_NEON32

// The kernels here are specifically arm 64bit assembly, not arm 32bit.
#ifdef GEMMLOWP_NEON64

// Our main GEMM kernel.
struct NEON64Kernel12x4Depth2 : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 2>, 3>,
                       KernelSideFormat<CellFormat<4, 2>, 1> > Format;

  const char* Name() const override { return "NEON, 12x4, depth 2"; }

  // TODO(benoitjacob): reorder function arguments so dst comes last
  void Run(std::int32_t* dst_ptr, int dst_row_stride, int dst_col_stride,
           const std::uint8_t* lhs_ptr, const std::uint8_t* rhs_ptr,
           int start_depth, int run_depth) const override {
    ScopedProfilingLabel label("optimized kernel");

    assert(dst_row_stride == 1);

    asm volatile(
        // Clear accumulator registers (see layout below)
        "dup v4.4s, wzr\n"
        "dup v5.4s, wzr\n"
        "dup v6.4s, wzr\n"
        "dup v7.4s, wzr\n"
        "dup v8.4s, wzr\n"
        "dup v9.4s, wzr\n"
        "dup v10.4s, wzr\n"
        "dup v11.4s, wzr\n"
        "dup v12.4s, wzr\n"
        "dup v13.4s, wzr\n"
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"

        /* Main loop */

        "loop_NEONKernel12x4Depth2_%=:\n"

        // Overview of register layout:
        //
        // A 2x4 cell of Rhs is stored in 16bit in d0--d1 (q0).
        // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in d2--d7
        // (q1--q3).
        // A 12x4 block of accumulators is stored in 32bit in q4--q15.
        //
        //                   +-----+-----+-----+-----+
        //                   |d0[0]|d0[1]|d0[2]|d0[3]|
        //              Rhs  +-----+-----+-----+-----+
        //                   |d1[0]|d1[1]|d1[2]|d1[3]|
        //                   +-----+-----+-----+-----+
        //
        //                   |     |     |     |     |
        //
        //    Lhs            |     |     |     |     |
        //
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //
        //                            Accumulator

        // Load 1 Rhs cell of size 2x4
        "ld1 {v0.8b}, [%[rhs_ptr]], #8\n"

        // Load 3 Lhs cells of size 4x2 each
        "ld1 {v1.8b}, [%[lhs_ptr]], #8\n"
        "ld1 {v2.8b}, [%[lhs_ptr]], #8\n"
        "ld1 {v3.8b}, [%[lhs_ptr]], #8\n"

        // Expand Lhs/Rhs cells to 16 bit.
        "uxtl v0.8h, v0.8b\n"
        "uxtl v1.8h, v1.8b\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"

        // Multiply-accumulate, level of depth 0
        "umlal v4.4s, v1.4h, v0.h[0]\n"
        "umlal v5.4s, v1.4h, v0.h[1]\n"
        "umlal v6.4s, v1.4h, v0.h[2]\n"
        "umlal v7.4s, v1.4h, v0.h[3]\n"
        "umlal v8.4s, v2.4h, v0.h[0]\n"
        "umlal v9.4s, v2.4h, v0.h[1]\n"
        "umlal v10.4s, v2.4h, v0.h[2]\n"
        "umlal v11.4s, v2.4h, v0.h[3]\n"
        "umlal v12.4s, v3.4h, v0.h[0]\n"
        "umlal v13.4s, v3.4h, v0.h[1]\n"
        "umlal v14.4s, v3.4h, v0.h[2]\n"
        "umlal v15.4s, v3.4h, v0.h[3]\n"

        // Multiply-accumulate, level of depth 1
        "umlal2 v4.4s, v1.8h, v0.h[4]\n"
        "umlal2 v5.4s, v1.8h, v0.h[5]\n"
        "umlal2 v6.4s, v1.8h, v0.h[6]\n"
        "umlal2 v7.4s, v1.8h, v0.h[7]\n"
        "umlal2 v8.4s, v2.8h, v0.h[4]\n"
        "umlal2 v9.4s, v2.8h, v0.h[5]\n"
        "umlal2 v10.4s, v2.8h, v0.h[6]\n"
        "umlal2 v11.4s, v2.8h, v0.h[7]\n"
        "umlal2 v12.4s, v3.8h, v0.h[4]\n"
        "umlal2 v13.4s, v3.8h, v0.h[5]\n"
        "umlal2 v14.4s, v3.8h, v0.h[6]\n"
        "umlal2 v15.4s, v3.8h, v0.h[7]\n"

        // Loop. Decrement loop index (depth) by 2, since we just handled 2
        // levels of depth (Kernel::kDepth=2).
        "subs %[run_depth], %[run_depth], #2\n"
        "bne loop_NEONKernel12x4Depth2_%=\n"

        /* end of main loop */

        /* Accumulate our local accumulator registers into the destination block
           */

        // Compute stride between consecutive columns, in bytes
        "mov x0, #4\n"  // multiply by 4 = sizeof(int32)
        "mul %[dst_col_stride], %[dst_col_stride], x0\n"

        // If start_depth == 0, then there is no preexisting accumulator
        // to accumulate, so we can simply store our result.
        "cmp %[start_depth], #0\n"
        "beq store_result_NEONKernel12x4Depth2_%=\n"

        "mov x0, %[dst_ptr]\n"

        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v4.4s, v4.4s, v0.4s\n"
        "add v8.4s, v8.4s, v1.4s\n"
        "add v12.4s, v12.4s, v2.4s\n"

        "add x0, x0, %[dst_col_stride]\n"
        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v5.4s, v5.4s, v0.4s\n"
        "add v9.4s, v9.4s, v1.4s\n"
        "add v13.4s, v13.4s, v2.4s\n"

        "add x0, x0, %[dst_col_stride]\n"
        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v6.4s, v6.4s, v0.4s\n"
        "add v10.4s, v10.4s, v1.4s\n"
        "add v14.4s, v14.4s, v2.4s\n"

        "add x0, x0, %[dst_col_stride]\n"
        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v7.4s, v7.4s, v0.4s\n"
        "add v11.4s, v11.4s, v1.4s\n"
        "add v15.4s, v15.4s, v2.4s\n"

        "store_result_NEONKernel12x4Depth2_%=:\n"

        "mov x0, %[dst_ptr]\n"
        // Store a column
        "mov x1, x0\n"
        "st1 {v4.4s}, [x1], #16\n"
        "st1 {v8.4s}, [x1], #16\n"
        "st1 {v12.4s}, [x1], #16\n"
        // Store a column
        "add x0, x0, %[dst_col_stride]\n"
        "mov x1, x0\n"
        "st1 {v5.4s}, [x1], #16\n"
        "st1 {v9.4s}, [x1], #16\n"
        "st1 {v13.4s}, [x1], #16\n"
        // Store a column
        "add x0, x0, %[dst_col_stride]\n"
        "mov x1, x0\n"
        "st1 {v6.4s}, [x1], #16\n"
        "st1 {v10.4s}, [x1], #16\n"
        "st1 {v14.4s}, [x1], #16\n"
        // Store a column
        "add x0, x0, %[dst_col_stride]\n"
        "mov x1, x0\n"
        "st1 {v7.4s}, [x1], #16\n"
        "st1 {v11.4s}, [x1], #16\n"
        "st1 {v15.4s}, [x1], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        :  // inputs
        [start_depth] "r"(start_depth),
        [dst_col_stride] "r"(dst_col_stride)
        :  // clobbers
        "cc", "memory", "x0", "x1",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
        "v31");
  }
};

#endif  // GEMMLOWP_NEON64

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_KERNEL_NEON_H_
