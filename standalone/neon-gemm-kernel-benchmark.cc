// This is a standalone testbed and benchmark for gemmlowp-style GEMM kernels,
// either doing integer or float arithmetic.
//
// It verifies that a kernel produces correct results, then benchmarks it;
// if multiple CPU cores are detected, it tries to benchmark on each core.
// This is aimed at big.LITTLE systems.
//
// This program is entirely self-contained, and can be compiled manually
// such as suggested in the command lines below.
// It currently supports only Android/ARM but would trivially generalize to
// other OSes (it's mostly standard POSIX) or architectures (each kernel
// targets a specific architecture, one may simply add more).

/*
Build and run this benchmark on Android/ARM/32bit:
export CXX=~/android/toolchains/arm-linux-androideabi-4.8/bin/arm-linux-androideabi-g++
$CXX -fPIE -pie -O3 --std=c++11 neon-gemm-kernel-benchmark.cc -o benchmark -mfloat-abi=softfp -mfpu=neon && adb push benchmark /data/local/tmp && adb shell /data/local/tmp/benchmark

Build and run this benchmark on Android/ARM/64bit:
export CXX=~/android/toolchains/aarch64-linux-android-4.9/bin/aarch64-linux-android-g++
$CXX -fPIE -pie -O3 --std=c++11 neon-gemm-kernel-benchmark.cc -o benchmark && adb push benchmark /data/local/tmp && adb shell /data/local/tmp/benchmark
*/

#include <sched.h>
#include <unistd.h>

#include <type_traits>
#include <random>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <cassert>

#ifdef PRINT_CPUFREQ
#include <fstream>
#include <sstream>
#include <string>
#endif

#if !defined __arm__ && !defined __aarch64__
#error This benchmark assumes ARM (for inline assembly sections).
#endif

// Typically one wants to fit in L1 cache, and GEMM implementations
// are carefully optimized to tune their access patterns to that effect.
// Most devices have at least 16k of L1 cache. The Kraits have exactly 16k.
const int kDefaultCacheSizeK = 16;

const int kCacheLineSize = 64;

// BEGIN code copied from gemmlowp/internal/kernel.h

// Explanation of general gemmlowp terminology
// ===========================================
//
// We use the following abbreviations:
// LHS = "left-hand side"
// RHS = "right-hand side"
// Sometimes when referring to either LHS or RHS, we just say a "Side".
//
// In a matrix product of a MxK matrix times a KxN matrix,
// we call K the 'depth'. Note that M is the number of rows
// of the result (and of the LHS), and N is the number of columns
// of the result (and of the RHS).
//
// In each of the LHS and RHS matrices, we call 'width' the
// other dimension, besides the depth. So in the LHS, 'width'
// is the number of rows, while in the RHS, 'width' is the number
// of columns.
//
//  So in the LHS MxK matrix, the depth is K and the width in M.
// And in the RHS KxN matrix, the depth is K and the width in N.
//
// This is illustrated in this picture:
//
//                             RHS width
//                        <----------------->
//                        +-----------------+ ^
//                        |       RHS       | | Depth
//                        +-----------------+ v
//                 ^ +--+ +-----------------+
//                 | |L | |                 |
//       LHS width | |H | |      Result     |
//                 | |S | |                 |
//                 v +--+ +-----------------+
//                   <-->
//                   Depth

// Explanation of gemmlowp kernel formats and "cells"
// ==================================================
//
// Kernels operate on small LHS and RHS blocks that fit in registers.
// These blocks are stored contiguously in memory, but not always
// in a traditional column-major or row-major order; instead,
// they consist of a number of sub-blocks, which we call "cells",
// that are stored in column-major or row-major order. However,
// what really matters to us is not so much rows vs columns, but
// rather width vs depth. So we refer to "width-major" and "depth-major"
// storage orders. In the LHS, width-major means row-major,
// while in the RHS, width-major means column-major.
// There is also a third possibility, "diagonal order",
// which is unused at the moment.
//
// We aim to treat both sides, LHS and RHS, on an equal footing,
// so we call them both 'sides'. A KernelFormat thus is just a pair
// of KernelSideFormat's, one for LHS and one for RHS; each KernelSideFormat
// contains a CellFormat and a number of cells; cells are only ever
// stacked in the width dimension, which means stacked vertically in the
// LHS and stacked horizondally in the RHS.
//
// Example
// =======
//
// Let's work out the data layout expected by a kernel having the
// following format (the struct names here are defined below in this file):
//
// KernelFormat<
//   KernelSideFormat<CellFormat<3, 4>, 3>,
//   KernelSideFormat<CellFormat<5, 4>, 2>
// >
//
// The LHS format, KernelSideFormat<CellFormat<3, 4>, 3>, means:
// 3 cells, each cell having dimensions (width=3, depth=4), laid out in
// DepthMajor order (the default value, see CellFormat). In the LHS,
// DepthMajor means column-major, so the LHS cells are of size 3x4 in
// column-major order, so the LHS layout is:
//
// 0  3  6  9
// 1  4  7  10
// 2  5  8  11
// 12 15 18 21
// 13 16 19 22
// 14 17 20 23
// 24 27 30 33
// 25 28 31 34
// 26 29 32 35
//
// The RHS format, KernelSideFormat<CellFormat<5, 4>, 2>, means:
// 2 cells each having dimensions (width=5, depth=4), laid out in
// DepthMajor order (the default value, see CellFormat). In the RHS,
// DepthMajor means row-major, so the RHS cells are of size 4x5 in
// row-major order, so the RHS layout is:
//
// 0  1  2  3  4  20 21 22 23 24
// 5  6  7  8  9  25 26 27 28 29
// 10 11 12 13 14 30 31 32 33 34
// 15 16 17 18 19 35 36 37 38 39

// CellOrder enumerates the possible storage orders (=layouts) for
// a cell (see explanation above).
enum class CellOrder { DepthMajor, WidthMajor, Diagonal };

// CellFormat describes how data is laid
// out in a cell. That is, a CellOrder together with actual dimensions.
template <int tWidth, int tDepth, CellOrder tOrder>
struct CellFormat {
  static const int kWidth = tWidth;
  static const int kDepth = tDepth;
  static const CellOrder kOrder = tOrder;

  static const int kSize = kWidth * kDepth;
};

// KernelSideFormat describes how data is laid out in a kernel side
// (i.e. LHS or RHS). That is, a CellFormat together with a number of
// cells. These cells are always stacked in the Width dimension.
// For example, in the LHS case, the Width dimension is the rows dimension,
// se we're saying that in the LHS, cells are stacked vertically.
// We never stack cells in the Depth dimension.
template <typename tCellFormat, int tCells>
struct KernelSideFormat {
  typedef tCellFormat Cell;
  static const int kCells = tCells;
  static const int kWidth = kCells * Cell::kWidth;
  static const int kDepth = Cell::kDepth;
};

// KernelFormat describes fully the input data layout that a kernel expects.
// It consists of two KernelSideFormat's, one for LHS and one for RHS.
template <typename tLhs, typename tRhs>
struct KernelFormat {
  typedef tLhs Lhs;
  typedef tRhs Rhs;

  static_assert(Lhs::Cell::kDepth == Rhs::Cell::kDepth, "");
  static const int kDepth = Lhs::Cell::kDepth;
  static const int kRows = Lhs::Cell::kWidth * Lhs::kCells;
  static const int kCols = Rhs::Cell::kWidth * Rhs::kCells;
};

inline const char* CellOrderName(CellOrder o) {
  switch (o) {
    case CellOrder::DepthMajor:
      return "DepthMajor";
    case CellOrder::WidthMajor:
      return "WidthMajor";
    case CellOrder::Diagonal:
      return "Diagonal";
    default:
      assert(false);
      return nullptr;
  }
}

// Returns the offset into a cell, at which a given coefficient is stored.
template <typename CellFormat>
inline int OffsetIntoCell(int w, int d) {
  switch (CellFormat::kOrder) {
    case CellOrder::DepthMajor:
      return w + d * CellFormat::kWidth;
    case CellOrder::WidthMajor:
      return d + w * CellFormat::kDepth;
    case CellOrder::Diagonal:
      assert(CellFormat::kWidth == CellFormat::kDepth);
      static const int size = CellFormat::kWidth;
      return ((size + w - d) * size + d) % (size * size);
    default:
      assert(false);
      return 0;
  }
}

// END code copied from gemmlowp/internal/kernel.h

#ifdef __arm__

// This is the current standard kernel in gemmlowp, see:
// https://github.com/google/gemmlowp/blob/b1e2a29ff866680028f3080efc244e10e8dd7f46/internal/kernel_neon.h#L33
struct NEON_32bit_GEMM_Uint8Operands_Uint32Accumulators {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Load accumulators
      "mov r0, %[accum_ptr]\n"
      "vld1.32 {d8, d9},   [r0]!\n"
      "vld1.32 {d16, d17}, [r0]!\n"
      "vld1.32 {d24, d25}, [r0]!\n"
      "vld1.32 {d10, d11}, [r0]!\n"
      "vld1.32 {d18, d19}, [r0]!\n"
      "vld1.32 {d26, d27}, [r0]!\n"
      "vld1.32 {d12, d13}, [r0]!\n"
      "vld1.32 {d20, d21}, [r0]!\n"
      "vld1.32 {d28, d29}, [r0]!\n"
      "vld1.32 {d14, d15}, [r0]!\n"
      "vld1.32 {d22, d23}, [r0]!\n"
      "vld1.32 {d30, d31}, [r0]!\n"

      "loop_%=:\n"
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
      "vld1.8 {d0}, [%[rhs_ptr]]!\n"

      // Load 3 Lhs cells of size 4x2 each
      "vld1.8 {d2}, [%[lhs_ptr]]!\n"
      "vld1.8 {d4}, [%[lhs_ptr]]!\n"
      "vld1.8 {d6}, [%[lhs_ptr]]!\n"

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
      // levels of depth.
      "subs %[depth], #2\n"
      "bne loop_%=\n"

      // Store accumulators
      "mov r0, %[accum_ptr]\n"
      "vst1.32 {d8, d9},   [r0]!\n"
      "vst1.32 {d16, d17}, [r0]!\n"
      "vst1.32 {d24, d25}, [r0]!\n"
      "vst1.32 {d10, d11}, [r0]!\n"
      "vst1.32 {d18, d19}, [r0]!\n"
      "vst1.32 {d26, d27}, [r0]!\n"
      "vst1.32 {d12, d13}, [r0]!\n"
      "vst1.32 {d20, d21}, [r0]!\n"
      "vst1.32 {d28, d29}, [r0]!\n"
      "vst1.32 {d14, d15}, [r0]!\n"
      "vst1.32 {d22, d23}, [r0]!\n"
      "vst1.32 {d30, d31}, [r0]!\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "r0",
      "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
      "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
      "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
      "d31");
  }
};

// This is Maciek Chociej's fast kernel not expanding operands,
// from gemmlowp/meta/. Search for
//      mul_3x8_3x8_int32_lhsadd_rhsadd
// in this file:
// https://raw.githubusercontent.com/google/gemmlowp/e4b9d858b6637d5d0058bfa3d869d2b95864251b/meta/single_thread_gemm.h
struct NEON_32bit_GEMM_Uint8Operands_Uint32Accumulators_noexpand {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<3, 8, CellOrder::WidthMajor>, 1>,
                       KernelSideFormat<CellFormat<3, 8, CellOrder::WidthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"
      "vmov.i32 q8, q5\n"

      // Loop head
      "loop_%=:\n"

      // Subtract counter.
      "subs %[depth], %[depth], #8\n"

      "vld1.8 {d18, d19, d20}, [%[rhs_ptr]]!\n"
      "vld1.8 {d21, d22, d23}, [%[lhs_ptr]]!\n"
      "vmull.u8 q12, d18, d21\n"
      "vmull.u8 q13, d18, d22\n"
      "vmull.u8 q14, d18, d23\n"
      "vmull.u8 q15, d19, d21\n"
      "vpadal.u16 q0, q12\n"
      "vpadal.u16 q1, q13\n"
      "vpadal.u16 q2, q14\n"
      "vpadal.u16 q3, q15\n"
      "vmull.u8 q12, d19, d22\n"
      "vmull.u8 q13, d19, d23\n"
      "vmull.u8 q14, d20, d21\n"
      "vmull.u8 q15, d20, d22\n"
      "vmull.u8 q9, d20, d23\n"
      "vpadal.u16 q4, q12\n"
      "vpadal.u16 q5, q13\n"
      "vpadal.u16 q6, q14\n"
      "vpadal.u16 q7, q15\n"
      "vpadal.u16 q8, q9\n"

      // Loop branch
      "bne loop_%=\n"

      // Horizontal reduce aggregators, step 1
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d16, d17\n"

      // Horizontal reduce aggregators, step 2
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"
      "vpadd.u32 d12, d12, d14\n"
      "vpadd.u32 d13, d16, d16\n"

      // Load accumulators
      "mov r0, %[accum_ptr]\n"
      "vld1.32 {d2}, [r0]!\n"
      "vld1.32 {d3[0]}, [r0]!\n"

      "vld1.32 {d8}, [r0]!\n"
      "vld1.32 {d9[0]}, [r0]!\n"

      "vld1.32 {d14}, [r0]!\n"
      "vld1.32 {d15[0]}, [r0]!\n"

      // Accumulate
      "vadd.s32 q0, q0, q1\n"
      "vadd.s32 q3, q3, q4\n"
      "vadd.s32 q6, q6, q7\n"

      // Store accumulators
      "mov r0, %[accum_ptr]\n"
      "vst1.32 {d0}, [r0]!\n"
      "vst1.32 {d1[0]}, [r0]!\n"

      "vst1.32 {d6}, [r0]!\n"
      "vst1.32 {d7[0]}, [r0]!\n"

      "vst1.32 {d12}, [r0]!\n"
      "vst1.32 {d13[0]}, [r0]!\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "r0",
      "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
      "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
      "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
      "d31");
  }
};


// We don't actually use int32*int32 in production. This is just an
// experiment to help dissociate the effect of integer-vs-float, from the
// effect of operands width.
struct NEON_32bit_GEMM_Int32_WithScalar {
  typedef std::int32_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Load accumulators
      "mov r0, %[accum_ptr]\n"
      "vld1.32 {d8, d9},   [r0]!\n"
      "vld1.32 {d16, d17}, [r0]!\n"
      "vld1.32 {d24, d25}, [r0]!\n"
      "vld1.32 {d10, d11}, [r0]!\n"
      "vld1.32 {d18, d19}, [r0]!\n"
      "vld1.32 {d26, d27}, [r0]!\n"
      "vld1.32 {d12, d13}, [r0]!\n"
      "vld1.32 {d20, d21}, [r0]!\n"
      "vld1.32 {d28, d29}, [r0]!\n"
      "vld1.32 {d14, d15}, [r0]!\n"
      "vld1.32 {d22, d23}, [r0]!\n"
      "vld1.32 {d30, d31}, [r0]!\n"

      "loop_%=:\n"

      // Load 1 Rhs cell of size 1x4
      "vld1.32 {d0, d1}, [%[rhs_ptr]]!\n"

      // Load 3 Lhs cells of size 4x1 each
      "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
      "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
      "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

      // Multiply-accumulate
      "vmla.s32 q4, q1, d0[0]\n"
      "vmla.s32 q5, q1, d0[1]\n"
      "vmla.s32 q6, q1, d1[0]\n"
      "vmla.s32 q7, q1, d1[1]\n"
      "vmla.s32 q8, q2, d0[0]\n"
      "vmla.s32 q9, q2, d0[1]\n"
      "vmla.s32 q10, q2, d1[0]\n"
      "vmla.s32 q11, q2, d1[1]\n"
      "vmla.s32 q12, q3, d0[0]\n"
      "vmla.s32 q13, q3, d0[1]\n"
      "vmla.s32 q14, q3, d1[0]\n"
      "vmla.s32 q15, q3, d1[1]\n"

      // Loop. Decrement loop index (depth) by 1, since we just handled 1
      // level of depth.
      "subs %[depth], #1\n"
      "bne loop_%=\n"

      // Store accumulators
      "mov r0, %[accum_ptr]\n"
      "vst1.32 {d8, d9},   [r0]!\n"
      "vst1.32 {d16, d17}, [r0]!\n"
      "vst1.32 {d24, d25}, [r0]!\n"
      "vst1.32 {d10, d11}, [r0]!\n"
      "vst1.32 {d18, d19}, [r0]!\n"
      "vst1.32 {d26, d27}, [r0]!\n"
      "vst1.32 {d12, d13}, [r0]!\n"
      "vst1.32 {d20, d21}, [r0]!\n"
      "vst1.32 {d28, d29}, [r0]!\n"
      "vst1.32 {d14, d15}, [r0]!\n"
      "vst1.32 {d22, d23}, [r0]!\n"
      "vst1.32 {d30, d31}, [r0]!\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "r0",
      "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
      "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
      "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
      "d31");
  }
};

// Not very efficient kernel, just an experiment to see what we can do
// without using NEON multiply-with-scalar instructions.
struct NEON_32bit_GEMM_Float32_MLA_WithVectorDuplicatingScalar {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Load accumulators
      "mov r0, %[accum_ptr]\n"
      "vld1.32 {d8, d9},   [r0]!\n"
      "vld1.32 {d16, d17}, [r0]!\n"
      "vld1.32 {d24, d25}, [r0]!\n"
      "vld1.32 {d10, d11}, [r0]!\n"
      "vld1.32 {d18, d19}, [r0]!\n"
      "vld1.32 {d26, d27}, [r0]!\n"
      "vld1.32 {d12, d13}, [r0]!\n"
      "vld1.32 {d20, d21}, [r0]!\n"
      "vld1.32 {d28, d29}, [r0]!\n"
      "vld1.32 {d14, d15}, [r0]!\n"
      "vld1.32 {d22, d23}, [r0]!\n"
      "vld1.32 {d30, d31}, [r0]!\n"

      "loop_%=:\n"

      // Load 3 Lhs cells of size 4x1 each
      "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
      "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
      "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

      // Multiply-accumulate
      "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
      "vmla.f32 q4, q1, q0\n"
      "vmla.f32 q8, q2, q0\n"
      "vmla.f32 q12, q3, q0\n"
      "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
      "vmla.f32 q5, q1, q0\n"
      "vmla.f32 q9, q2, q0\n"
      "vmla.f32 q13, q3, q0\n"
      "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
      "vmla.f32 q6, q1, q0\n"
      "vmla.f32 q10, q2, q0\n"
      "vmla.f32 q14, q3, q0\n"
      "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
      "vmla.f32 q7, q1, q0\n"
      "vmla.f32 q11, q2, q0\n"
      "vmla.f32 q15, q3, q0\n"

      // Loop. Decrement loop index (depth) by 1, since we just handled 1
      // level of depth.
      "subs %[depth], #1\n"
      "bne loop_%=\n"

      // Store accumulators
      "mov r0, %[accum_ptr]\n"
      "vst1.32 {d8, d9},   [r0]!\n"
      "vst1.32 {d16, d17}, [r0]!\n"
      "vst1.32 {d24, d25}, [r0]!\n"
      "vst1.32 {d10, d11}, [r0]!\n"
      "vst1.32 {d18, d19}, [r0]!\n"
      "vst1.32 {d26, d27}, [r0]!\n"
      "vst1.32 {d12, d13}, [r0]!\n"
      "vst1.32 {d20, d21}, [r0]!\n"
      "vst1.32 {d28, d29}, [r0]!\n"
      "vst1.32 {d14, d15}, [r0]!\n"
      "vst1.32 {d22, d23}, [r0]!\n"
      "vst1.32 {d30, d31}, [r0]!\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "r0",
      "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
      "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
      "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
      "d31");
  }
};

// Not very efficient kernel, just an experiment to see what we can do
// without using NEON multiply-with-scalar instructions.
// This variant is relevant as on ARMv7 FMA does not have a with-scalar variant.
struct NEON_32bit_GEMM_Float32_FMA_WithVectorDuplicatingScalar {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Load accumulators
      "mov r0, %[accum_ptr]\n"
      "vld1.32 {d8, d9},   [r0]!\n"
      "vld1.32 {d16, d17}, [r0]!\n"
      "vld1.32 {d24, d25}, [r0]!\n"
      "vld1.32 {d10, d11}, [r0]!\n"
      "vld1.32 {d18, d19}, [r0]!\n"
      "vld1.32 {d26, d27}, [r0]!\n"
      "vld1.32 {d12, d13}, [r0]!\n"
      "vld1.32 {d20, d21}, [r0]!\n"
      "vld1.32 {d28, d29}, [r0]!\n"
      "vld1.32 {d14, d15}, [r0]!\n"
      "vld1.32 {d22, d23}, [r0]!\n"
      "vld1.32 {d30, d31}, [r0]!\n"

      "loop_%=:\n"

      // Load 3 Lhs cells of size 4x1 each
      "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
      "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
      "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

      // Multiply-accumulate
      "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
      "vfma.f32 q4, q1, q0\n"
      "vfma.f32 q8, q2, q0\n"
      "vfma.f32 q12, q3, q0\n"
      "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
      "vfma.f32 q5, q1, q0\n"
      "vfma.f32 q9, q2, q0\n"
      "vfma.f32 q13, q3, q0\n"
      "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
      "vfma.f32 q6, q1, q0\n"
      "vfma.f32 q10, q2, q0\n"
      "vfma.f32 q14, q3, q0\n"
      "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
      "vfma.f32 q7, q1, q0\n"
      "vfma.f32 q11, q2, q0\n"
      "vfma.f32 q15, q3, q0\n"

      // Loop. Decrement loop index (depth) by 1, since we just handled 1
      // level of depth.
      "subs %[depth], #1\n"
      "bne loop_%=\n"

      // Store accumulators
      "mov r0, %[accum_ptr]\n"
      "vst1.32 {d8, d9},   [r0]!\n"
      "vst1.32 {d16, d17}, [r0]!\n"
      "vst1.32 {d24, d25}, [r0]!\n"
      "vst1.32 {d10, d11}, [r0]!\n"
      "vst1.32 {d18, d19}, [r0]!\n"
      "vst1.32 {d26, d27}, [r0]!\n"
      "vst1.32 {d12, d13}, [r0]!\n"
      "vst1.32 {d20, d21}, [r0]!\n"
      "vst1.32 {d28, d29}, [r0]!\n"
      "vst1.32 {d14, d15}, [r0]!\n"
      "vst1.32 {d22, d23}, [r0]!\n"
      "vst1.32 {d30, d31}, [r0]!\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "r0",
      "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
      "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
      "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
      "d31");
  }
};

// This is the "most natural" kernel, using NEON multiply-with-scalar instructions.
struct NEON_32bit_GEMM_Float32_MLA_WithScalar {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Load accumulators
      "mov r0, %[accum_ptr]\n"
      "vld1.32 {d8, d9},   [r0]!\n"
      "vld1.32 {d16, d17}, [r0]!\n"
      "vld1.32 {d24, d25}, [r0]!\n"
      "vld1.32 {d10, d11}, [r0]!\n"
      "vld1.32 {d18, d19}, [r0]!\n"
      "vld1.32 {d26, d27}, [r0]!\n"
      "vld1.32 {d12, d13}, [r0]!\n"
      "vld1.32 {d20, d21}, [r0]!\n"
      "vld1.32 {d28, d29}, [r0]!\n"
      "vld1.32 {d14, d15}, [r0]!\n"
      "vld1.32 {d22, d23}, [r0]!\n"
      "vld1.32 {d30, d31}, [r0]!\n"

      "loop_%=:\n"

      // Load 1 Rhs cell of size 1x4
      "vld1.32 {d0, d1}, [%[rhs_ptr]]!\n"

      // Load 3 Lhs cells of size 4x1 each
      "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
      "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
      "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

      // Multiply-accumulate
      "vmla.f32 q4, q1, d0[0]\n"
      "vmla.f32 q5, q1, d0[1]\n"
      "vmla.f32 q6, q1, d1[0]\n"
      "vmla.f32 q7, q1, d1[1]\n"
      "vmla.f32 q8, q2, d0[0]\n"
      "vmla.f32 q9, q2, d0[1]\n"
      "vmla.f32 q10, q2, d1[0]\n"
      "vmla.f32 q11, q2, d1[1]\n"
      "vmla.f32 q12, q3, d0[0]\n"
      "vmla.f32 q13, q3, d0[1]\n"
      "vmla.f32 q14, q3, d1[0]\n"
      "vmla.f32 q15, q3, d1[1]\n"

      // Loop. Decrement loop index (depth) by 1, since we just handled 1
      // level of depth.
      "subs %[depth], #1\n"
      "bne loop_%=\n"

      // Store accumulators
      "mov r0, %[accum_ptr]\n"
      "vst1.32 {d8, d9},   [r0]!\n"
      "vst1.32 {d16, d17}, [r0]!\n"
      "vst1.32 {d24, d25}, [r0]!\n"
      "vst1.32 {d10, d11}, [r0]!\n"
      "vst1.32 {d18, d19}, [r0]!\n"
      "vst1.32 {d26, d27}, [r0]!\n"
      "vst1.32 {d12, d13}, [r0]!\n"
      "vst1.32 {d20, d21}, [r0]!\n"
      "vst1.32 {d28, d29}, [r0]!\n"
      "vst1.32 {d14, d15}, [r0]!\n"
      "vst1.32 {d22, d23}, [r0]!\n"
      "vst1.32 {d30, d31}, [r0]!\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "r0",
      "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
      "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
      "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
      "d31");
  }
};

// This rotating variant performs well when permutations (vext) can be dual-issued
// with arithmetic instructions.
struct NEON_32bit_GEMM_Float32_MLA_Rotating {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Load accumulators
      "mov r0, %[accum_ptr]\n"
      "vld1.32 {d8, d9},   [r0]!\n"
      "vld1.32 {d16, d17}, [r0]!\n"
      "vld1.32 {d24, d25}, [r0]!\n"
      "vld1.32 {d10, d11}, [r0]!\n"
      "vld1.32 {d18, d19}, [r0]!\n"
      "vld1.32 {d26, d27}, [r0]!\n"
      "vld1.32 {d12, d13}, [r0]!\n"
      "vld1.32 {d20, d21}, [r0]!\n"
      "vld1.32 {d28, d29}, [r0]!\n"
      "vld1.32 {d14, d15}, [r0]!\n"
      "vld1.32 {d22, d23}, [r0]!\n"
      "vld1.32 {d30, d31}, [r0]!\n"

#define NEON_32BIT_ROTATING_FLOAT_KERNEL_TRANSPOSE_ACCUMULATOR_CELLS \
      "vtrn.32 q4, q5\n" \
      "vtrn.32 q6, q7\n" \
      "vswp d9, d12\n" \
      "vswp d11, d14\n" \
      "vtrn.32 q8, q9\n" \
      "vtrn.32 q10, q11\n" \
      "vswp d17, d20\n" \
      "vswp d19, d22\n" \
      "vtrn.32 q12, q13\n" \
      "vtrn.32 q14, q15\n" \
      "vswp d25, d28\n" \
      "vswp d27, d30\n"

#define NEON_32BIT_ROTATING_FLOAT_KERNEL_ROTATE_ACCUMULATOR_CELLS(a, b, c) \
      NEON_32BIT_ROTATING_FLOAT_KERNEL_TRANSPOSE_ACCUMULATOR_CELLS \
      "vext.32 q5, q5, q5, #" #a "\n" \
      "vext.32 q6, q6, q6, #" #b "\n" \
      "vext.32 q7, q7, q7, #" #c "\n" \
      "vext.32 q9, q9, q9, #" #a "\n" \
      "vext.32 q10, q10, q10, #" #b "\n" \
      "vext.32 q11, q11, q11, #" #c "\n" \
      "vext.32 q13, q13, q13, #" #a "\n" \
      "vext.32 q14, q14, q14, #" #b "\n" \
      "vext.32 q15, q15, q15, #" #c "\n" \
      NEON_32BIT_ROTATING_FLOAT_KERNEL_TRANSPOSE_ACCUMULATOR_CELLS

      NEON_32BIT_ROTATING_FLOAT_KERNEL_ROTATE_ACCUMULATOR_CELLS(1, 2, 3)

      "loop_%=:\n"

      // Load 1 Rhs cell of size 1x4
      "vld1.32 {d0, d1}, [%[rhs_ptr]]!\n"

      // Load 3 Lhs cells of size 4x1 each
      "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
      "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
      "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

      // Multiply-accumulate
      "vmla.f32 q4, q1, q0\n"
      "vmla.f32 q8, q2, q0\n"
      "vmla.f32 q12, q3, q0\n"
      "vext.f32 q0, q0, q0, #1\n"
      "vmla.f32 q5, q1, q0\n"
      "vmla.f32 q9, q2, q0\n"
      "vmla.f32 q13, q3, q0\n"
      "vext.f32 q0, q0, q0, #1\n"
      "vmla.f32 q6, q1, q0\n"
      "vmla.f32 q10, q2, q0\n"
      "vmla.f32 q14, q3, q0\n"
      "vext.f32 q0, q0, q0, #1\n"
      "vmla.f32 q7, q1, q0\n"
      "vmla.f32 q11, q2, q0\n"
      "vmla.f32 q15, q3, q0\n"

      // Loop. Decrement loop index (depth) by 1, since we just handled 1
      // level of depth.
      "subs %[depth], #1\n"
      "bne loop_%=\n"

      // Store accumulators
      "mov r0, %[accum_ptr]\n"

      NEON_32BIT_ROTATING_FLOAT_KERNEL_ROTATE_ACCUMULATOR_CELLS(3, 2, 1)

      "vst1.32 {d8, d9},   [r0]!\n"
      "vst1.32 {d16, d17}, [r0]!\n"
      "vst1.32 {d24, d25}, [r0]!\n"
      "vst1.32 {d10, d11}, [r0]!\n"
      "vst1.32 {d18, d19}, [r0]!\n"
      "vst1.32 {d26, d27}, [r0]!\n"
      "vst1.32 {d12, d13}, [r0]!\n"
      "vst1.32 {d20, d21}, [r0]!\n"
      "vst1.32 {d28, d29}, [r0]!\n"
      "vst1.32 {d14, d15}, [r0]!\n"
      "vst1.32 {d22, d23}, [r0]!\n"
      "vst1.32 {d30, d31}, [r0]!\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "r0",
      "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
      "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
      "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
      "d31");
  }
};

// This rotating variant performs well when permutations (vext) can be dual-issued
// with arithmetic instructions.
// It is relevant as the rotating approach removes the need for multiply-with-scalar
// instructions, and ARMv7 FMA does not have a with-scalar variant.
struct NEON_32bit_GEMM_Float32_FMA_Rotating {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Load accumulators
      "mov r0, %[accum_ptr]\n"
      "vld1.32 {d8, d9},   [r0]!\n"
      "vld1.32 {d16, d17}, [r0]!\n"
      "vld1.32 {d24, d25}, [r0]!\n"
      "vld1.32 {d10, d11}, [r0]!\n"
      "vld1.32 {d18, d19}, [r0]!\n"
      "vld1.32 {d26, d27}, [r0]!\n"
      "vld1.32 {d12, d13}, [r0]!\n"
      "vld1.32 {d20, d21}, [r0]!\n"
      "vld1.32 {d28, d29}, [r0]!\n"
      "vld1.32 {d14, d15}, [r0]!\n"
      "vld1.32 {d22, d23}, [r0]!\n"
      "vld1.32 {d30, d31}, [r0]!\n"

      NEON_32BIT_ROTATING_FLOAT_KERNEL_ROTATE_ACCUMULATOR_CELLS(1, 2, 3)

      "loop_%=:\n"

      // Load 1 Rhs cell of size 1x4
      "vld1.32 {d0, d1}, [%[rhs_ptr]]!\n"

      // Load 3 Lhs cells of size 4x1 each
      "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
      "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
      "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

      // Multiply-accumulate
      "vfma.f32 q4, q1, q0\n"
      "vfma.f32 q8, q2, q0\n"
      "vfma.f32 q12, q3, q0\n"
      "vext.f32 q0, q0, q0, #1\n"
      "vfma.f32 q5, q1, q0\n"
      "vfma.f32 q9, q2, q0\n"
      "vfma.f32 q13, q3, q0\n"
      "vext.f32 q0, q0, q0, #1\n"
      "vfma.f32 q6, q1, q0\n"
      "vfma.f32 q10, q2, q0\n"
      "vfma.f32 q14, q3, q0\n"
      "vext.f32 q0, q0, q0, #1\n"
      "vfma.f32 q7, q1, q0\n"
      "vfma.f32 q11, q2, q0\n"
      "vfma.f32 q15, q3, q0\n"

      // Loop. Decrement loop index (depth) by 1, since we just handled 1
      // level of depth.
      "subs %[depth], #1\n"
      "bne loop_%=\n"

      NEON_32BIT_ROTATING_FLOAT_KERNEL_ROTATE_ACCUMULATOR_CELLS(3, 2, 1)

      // Store accumulators
      "mov r0, %[accum_ptr]\n"
      "vst1.32 {d8, d9},   [r0]!\n"
      "vst1.32 {d16, d17}, [r0]!\n"
      "vst1.32 {d24, d25}, [r0]!\n"
      "vst1.32 {d10, d11}, [r0]!\n"
      "vst1.32 {d18, d19}, [r0]!\n"
      "vst1.32 {d26, d27}, [r0]!\n"
      "vst1.32 {d12, d13}, [r0]!\n"
      "vst1.32 {d20, d21}, [r0]!\n"
      "vst1.32 {d28, d29}, [r0]!\n"
      "vst1.32 {d14, d15}, [r0]!\n"
      "vst1.32 {d22, d23}, [r0]!\n"
      "vst1.32 {d30, d31}, [r0]!\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "r0",
      "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
      "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
      "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
      "d31");
  }
};

#endif  // __arm__

#ifdef __aarch64__

// This is the current standard kernel in gemmlowp, see:
// https://github.com/google/gemmlowp/blob/b1e2a29ff866680028f3080efc244e10e8dd7f46/internal/kernel_neon.h#L646
struct NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Load accumulators
      "mov x0, %[accum_ptr]\n"
      "ld1 {v8.16b}, [x0], #16\n"
      "ld1 {v16.16b}, [x0], #16\n"
      "ld1 {v24.16b}, [x0], #16\n"
      "ld1 {v9.16b}, [x0], #16\n"
      "ld1 {v17.16b}, [x0], #16\n"
      "ld1 {v25.16b}, [x0], #16\n"
      "ld1 {v10.16b}, [x0], #16\n"
      "ld1 {v18.16b}, [x0], #16\n"
      "ld1 {v26.16b}, [x0], #16\n"
      "ld1 {v11.16b}, [x0], #16\n"
      "ld1 {v19.16b}, [x0], #16\n"
      "ld1 {v27.16b}, [x0], #16\n"
      "ld1 {v12.16b}, [x0], #16\n"
      "ld1 {v20.16b}, [x0], #16\n"
      "ld1 {v28.16b}, [x0], #16\n"
      "ld1 {v13.16b}, [x0], #16\n"
      "ld1 {v21.16b}, [x0], #16\n"
      "ld1 {v29.16b}, [x0], #16\n"
      "ld1 {v14.16b}, [x0], #16\n"
      "ld1 {v22.16b}, [x0], #16\n"
      "ld1 {v30.16b}, [x0], #16\n"
      "ld1 {v15.16b}, [x0], #16\n"
      "ld1 {v23.16b}, [x0], #16\n"
      "ld1 {v31.16b}, [x0], #16\n"

      "loop_%=:\n"

      // Overview of register layout:
      //
      // A 2x8 block of 2 2x4 cells of Rhs is stored in 16bit in v0--v1.
      // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in v2--v4.
      // A 12x8 block of accumulators is stored in 32bit in v8--v31.
      //
      //                         +--------+--------+-----+--------+--------+
      //                         |v0.h[0] |v0.h[1] | ... |v1.h[2] |v1.h[3] |
      //                    Rhs  +--------+--------+-----+--------+--------+
      //                         |v0.h[4] |v0.h[5] | ... |v1.h[6] |v1.h[7] |
      //                         +--------+--------+-----+--------+--------+
      //
      //                         |        |        |     |        |        |
      //
      //    Lhs                  |        |        |     |        |        |
      //
      //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
      //  |v2.h[0]|v2.h[4]|      |v8.s[0] |v9.s[0] | ... |v14.s[0]|v15.s[0]|
      //  |v2.h[1]|v2.h[5]|      |v8.s[1] |v9.s[1] | ... |v14.s[1]|v15.s[1]|
      //  |v2.h[2]|v2.h[6]|      |v8.s[2] |v9.s[2] | ... |v14.s[2]|v15.s[2]|
      //  |v2.h[3]|v2.h[7]|      |v8.s[3] |v9.s[3] | ... |v14.s[3]|v15.s[3]|
      //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
      //  |v3.h[0]|v3.h[4]|      |v16.s[0]|v17.s[0]| ... |v22.s[0]|v23.s[0]|
      //  |v3.h[1]|v3.h[5]|      |v16.s[1]|v17.s[1]| ... |v22.s[1]|v23.s[1]|
      //  |v3.h[2]|v3.h[6]|      |v16.s[2]|v17.s[2]| ... |v22.s[2]|v23.s[2]|
      //  |v3.h[3]|v3.h[7]|      |v16.s[3]|v17.s[3]| ... |v22.s[3]|v23.s[3]|
      //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
      //  |v4.h[0]|v4.h[4]|      |v24.s[0]|v25.s[0]| ... |v30.s[0]|v31.s[0]|
      //  |v4.h[1]|v4.h[5]|      |v24.s[1]|v25.s[1]| ... |v30.s[1]|v31.s[1]|
      //  |v4.h[2]|v4.h[6]|      |v24.s[2]|v25.s[2]| ... |v30.s[2]|v31.s[2]|
      //  |v4.h[3]|v4.h[7]|      |v24.s[3]|v25.s[3]| ... |v30.s[3]|v31.s[3]|
      //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
      //
      //                            Accumulator

      // Load 1 Rhs cell of size 2x8
      "ld1 {v0.8b}, [%[rhs_ptr]], #8\n"
      "ld1 {v1.8b}, [%[rhs_ptr]], #8\n"

      // Load 3 Lhs cells of size 4x2 each
      "ld1 {v2.8b}, [%[lhs_ptr]], #8\n"
      "ld1 {v3.8b}, [%[lhs_ptr]], #8\n"
      "ld1 {v4.8b}, [%[lhs_ptr]], #8\n"

      // Expand Lhs/Rhs cells to 16 bit.
      "uxtl v0.8h, v0.8b\n"
      "uxtl v1.8h, v1.8b\n"
      "uxtl v2.8h, v2.8b\n"
      "uxtl v3.8h, v3.8b\n"
      "uxtl v4.8h, v4.8b\n"

      // Multiply-accumulate, level of depth 0
      "umlal v8.4s, v2.4h, v0.h[0]\n"
      "umlal v9.4s, v2.4h, v0.h[1]\n"
      "umlal v10.4s, v2.4h, v0.h[2]\n"
      "umlal v11.4s, v2.4h, v0.h[3]\n"
      "umlal v12.4s, v2.4h, v1.h[0]\n"
      "umlal v13.4s, v2.4h, v1.h[1]\n"
      "umlal v14.4s, v2.4h, v1.h[2]\n"
      "umlal v15.4s, v2.4h, v1.h[3]\n"
      "umlal v16.4s, v3.4h, v0.h[0]\n"
      "umlal v17.4s, v3.4h, v0.h[1]\n"
      "umlal v18.4s, v3.4h, v0.h[2]\n"
      "umlal v19.4s, v3.4h, v0.h[3]\n"
      "umlal v20.4s, v3.4h, v1.h[0]\n"
      "umlal v21.4s, v3.4h, v1.h[1]\n"
      "umlal v22.4s, v3.4h, v1.h[2]\n"
      "umlal v23.4s, v3.4h, v1.h[3]\n"
      "umlal v24.4s, v4.4h, v0.h[0]\n"
      "umlal v25.4s, v4.4h, v0.h[1]\n"
      "umlal v26.4s, v4.4h, v0.h[2]\n"
      "umlal v27.4s, v4.4h, v0.h[3]\n"
      "umlal v28.4s, v4.4h, v1.h[0]\n"
      "umlal v29.4s, v4.4h, v1.h[1]\n"
      "umlal v30.4s, v4.4h, v1.h[2]\n"
      "umlal v31.4s, v4.4h, v1.h[3]\n"

      // Multiply-accumulate, level of depth 1
      "umlal2 v8.4s, v2.8h, v0.h[4]\n"
      "umlal2 v9.4s, v2.8h, v0.h[5]\n"
      "umlal2 v10.4s, v2.8h, v0.h[6]\n"
      "umlal2 v11.4s, v2.8h, v0.h[7]\n"
      "umlal2 v12.4s, v2.8h, v1.h[4]\n"
      "umlal2 v13.4s, v2.8h, v1.h[5]\n"
      "umlal2 v14.4s, v2.8h, v1.h[6]\n"
      "umlal2 v15.4s, v2.8h, v1.h[7]\n"
      "umlal2 v16.4s, v3.8h, v0.h[4]\n"
      "umlal2 v17.4s, v3.8h, v0.h[5]\n"
      "umlal2 v18.4s, v3.8h, v0.h[6]\n"
      "umlal2 v19.4s, v3.8h, v0.h[7]\n"
      "umlal2 v20.4s, v3.8h, v1.h[4]\n"
      "umlal2 v21.4s, v3.8h, v1.h[5]\n"
      "umlal2 v22.4s, v3.8h, v1.h[6]\n"
      "umlal2 v23.4s, v3.8h, v1.h[7]\n"
      "umlal2 v24.4s, v4.8h, v0.h[4]\n"
      "umlal2 v25.4s, v4.8h, v0.h[5]\n"
      "umlal2 v26.4s, v4.8h, v0.h[6]\n"
      "umlal2 v27.4s, v4.8h, v0.h[7]\n"
      "umlal2 v28.4s, v4.8h, v1.h[4]\n"
      "umlal2 v29.4s, v4.8h, v1.h[5]\n"
      "umlal2 v30.4s, v4.8h, v1.h[6]\n"
      "umlal2 v31.4s, v4.8h, v1.h[7]\n"

      // Loop. Decrement loop index (depth) by 2, since we just handled 2
      // levels of depth.
      "subs %[depth], %[depth], #2\n"
      "bne loop_%=\n"

      // Store accumulators
      "mov x0, %[accum_ptr]\n"
      "st1 {v8.16b}, [x0], #16\n"
      "st1 {v16.16b}, [x0], #16\n"
      "st1 {v24.16b}, [x0], #16\n"
      "st1 {v9.16b}, [x0], #16\n"
      "st1 {v17.16b}, [x0], #16\n"
      "st1 {v25.16b}, [x0], #16\n"
      "st1 {v10.16b}, [x0], #16\n"
      "st1 {v18.16b}, [x0], #16\n"
      "st1 {v26.16b}, [x0], #16\n"
      "st1 {v11.16b}, [x0], #16\n"
      "st1 {v19.16b}, [x0], #16\n"
      "st1 {v27.16b}, [x0], #16\n"
      "st1 {v12.16b}, [x0], #16\n"
      "st1 {v20.16b}, [x0], #16\n"
      "st1 {v28.16b}, [x0], #16\n"
      "st1 {v13.16b}, [x0], #16\n"
      "st1 {v21.16b}, [x0], #16\n"
      "st1 {v29.16b}, [x0], #16\n"
      "st1 {v14.16b}, [x0], #16\n"
      "st1 {v22.16b}, [x0], #16\n"
      "st1 {v30.16b}, [x0], #16\n"
      "st1 {v15.16b}, [x0], #16\n"
      "st1 {v23.16b}, [x0], #16\n"
      "st1 {v31.16b}, [x0], #16\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
      "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
      "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
      "v27", "v28", "v29", "v30", "v31");
  }
};

// We don't actually use int32*int32 in production. This is just an
// experiment to help dissociate the effect of integer-vs-float, from the
// effect of operands width.
struct NEON_64bit_GEMM_Int32_WithScalar {
  typedef std::int32_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Load accumulators
      "mov x0, %[accum_ptr]\n"
      "ld1 {v8.16b}, [x0], #16\n"
      "ld1 {v16.16b}, [x0], #16\n"
      "ld1 {v24.16b}, [x0], #16\n"
      "ld1 {v9.16b}, [x0], #16\n"
      "ld1 {v17.16b}, [x0], #16\n"
      "ld1 {v25.16b}, [x0], #16\n"
      "ld1 {v10.16b}, [x0], #16\n"
      "ld1 {v18.16b}, [x0], #16\n"
      "ld1 {v26.16b}, [x0], #16\n"
      "ld1 {v11.16b}, [x0], #16\n"
      "ld1 {v19.16b}, [x0], #16\n"
      "ld1 {v27.16b}, [x0], #16\n"
      "ld1 {v12.16b}, [x0], #16\n"
      "ld1 {v20.16b}, [x0], #16\n"
      "ld1 {v28.16b}, [x0], #16\n"
      "ld1 {v13.16b}, [x0], #16\n"
      "ld1 {v21.16b}, [x0], #16\n"
      "ld1 {v29.16b}, [x0], #16\n"
      "ld1 {v14.16b}, [x0], #16\n"
      "ld1 {v22.16b}, [x0], #16\n"
      "ld1 {v30.16b}, [x0], #16\n"
      "ld1 {v15.16b}, [x0], #16\n"
      "ld1 {v23.16b}, [x0], #16\n"
      "ld1 {v31.16b}, [x0], #16\n"

      "loop_%=:\n"

      // Load 2 Rhs cell of size 1x4 each
      "ld1 {v0.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v1.4s}, [%[rhs_ptr]], #16\n"

      // Load 3 Lhs cells of size 4x1 each
      "ld1 {v2.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v3.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v4.4s}, [%[lhs_ptr]], #16\n"

      // Multiply-accumulate
      "mla v8.4s, v2.4s, v0.s[0]\n"
      "mla v9.4s, v2.4s, v0.s[1]\n"
      "mla v10.4s, v2.4s, v0.s[2]\n"
      "mla v11.4s, v2.4s, v0.s[3]\n"
      "mla v12.4s, v2.4s, v1.s[0]\n"
      "mla v13.4s, v2.4s, v1.s[1]\n"
      "mla v14.4s, v2.4s, v1.s[2]\n"
      "mla v15.4s, v2.4s, v1.s[3]\n"
      "mla v16.4s, v3.4s, v0.s[0]\n"
      "mla v17.4s, v3.4s, v0.s[1]\n"
      "mla v18.4s, v3.4s, v0.s[2]\n"
      "mla v19.4s, v3.4s, v0.s[3]\n"
      "mla v20.4s, v3.4s, v1.s[0]\n"
      "mla v21.4s, v3.4s, v1.s[1]\n"
      "mla v22.4s, v3.4s, v1.s[2]\n"
      "mla v23.4s, v3.4s, v1.s[3]\n"
      "mla v24.4s, v4.4s, v0.s[0]\n"
      "mla v25.4s, v4.4s, v0.s[1]\n"
      "mla v26.4s, v4.4s, v0.s[2]\n"
      "mla v27.4s, v4.4s, v0.s[3]\n"
      "mla v28.4s, v4.4s, v1.s[0]\n"
      "mla v29.4s, v4.4s, v1.s[1]\n"
      "mla v30.4s, v4.4s, v1.s[2]\n"
      "mla v31.4s, v4.4s, v1.s[3]\n"

      // Loop. Decrement loop index (depth) by 1, since we just handled 1
      // level of depth.
      "subs %[depth], %[depth], #1\n"
      "bne loop_%=\n"

      // Store accumulators
      "mov x0, %[accum_ptr]\n"
      "st1 {v8.16b}, [x0], #16\n"
      "st1 {v16.16b}, [x0], #16\n"
      "st1 {v24.16b}, [x0], #16\n"
      "st1 {v9.16b}, [x0], #16\n"
      "st1 {v17.16b}, [x0], #16\n"
      "st1 {v25.16b}, [x0], #16\n"
      "st1 {v10.16b}, [x0], #16\n"
      "st1 {v18.16b}, [x0], #16\n"
      "st1 {v26.16b}, [x0], #16\n"
      "st1 {v11.16b}, [x0], #16\n"
      "st1 {v19.16b}, [x0], #16\n"
      "st1 {v27.16b}, [x0], #16\n"
      "st1 {v12.16b}, [x0], #16\n"
      "st1 {v20.16b}, [x0], #16\n"
      "st1 {v28.16b}, [x0], #16\n"
      "st1 {v13.16b}, [x0], #16\n"
      "st1 {v21.16b}, [x0], #16\n"
      "st1 {v29.16b}, [x0], #16\n"
      "st1 {v14.16b}, [x0], #16\n"
      "st1 {v22.16b}, [x0], #16\n"
      "st1 {v30.16b}, [x0], #16\n"
      "st1 {v15.16b}, [x0], #16\n"
      "st1 {v23.16b}, [x0], #16\n"
      "st1 {v31.16b}, [x0], #16\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
      "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
      "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
      "v27", "v28", "v29", "v30", "v31");
  }
};

// Not very efficient kernel, just an experiment to see what we can do
// without using NEON multiply-with-scalar instructions.
struct NEON_64bit_GEMM_Float32_WithVectorDuplicatingScalar {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Load accumulators
      "mov x0, %[accum_ptr]\n"
      "ld1 {v8.16b}, [x0], #16\n"
      "ld1 {v16.16b}, [x0], #16\n"
      "ld1 {v24.16b}, [x0], #16\n"
      "ld1 {v9.16b}, [x0], #16\n"
      "ld1 {v17.16b}, [x0], #16\n"
      "ld1 {v25.16b}, [x0], #16\n"
      "ld1 {v10.16b}, [x0], #16\n"
      "ld1 {v18.16b}, [x0], #16\n"
      "ld1 {v26.16b}, [x0], #16\n"
      "ld1 {v11.16b}, [x0], #16\n"
      "ld1 {v19.16b}, [x0], #16\n"
      "ld1 {v27.16b}, [x0], #16\n"
      "ld1 {v12.16b}, [x0], #16\n"
      "ld1 {v20.16b}, [x0], #16\n"
      "ld1 {v28.16b}, [x0], #16\n"
      "ld1 {v13.16b}, [x0], #16\n"
      "ld1 {v21.16b}, [x0], #16\n"
      "ld1 {v29.16b}, [x0], #16\n"
      "ld1 {v14.16b}, [x0], #16\n"
      "ld1 {v22.16b}, [x0], #16\n"
      "ld1 {v30.16b}, [x0], #16\n"
      "ld1 {v15.16b}, [x0], #16\n"
      "ld1 {v23.16b}, [x0], #16\n"
      "ld1 {v31.16b}, [x0], #16\n"

      "loop_%=:\n"

      // Load 2 Rhs cell of size 1x4 each
      "ld1 {v5.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v6.4s}, [%[rhs_ptr]], #16\n"

      // Load 3 Lhs cells of size 4x1 each
      "ld1 {v2.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v3.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v4.4s}, [%[lhs_ptr]], #16\n"

      // Multiply-accumulate
      "dup v0.4s, v5.s[0]\n"
      "dup v1.4s, v5.s[1]\n"
      "fmla v8.4s, v2.4s, v0.4s\n"
      "fmla v16.4s, v3.4s, v0.4s\n"
      "fmla v24.4s, v4.4s, v0.4s\n"
      "fmla v9.4s, v2.4s, v1.4s\n"
      "fmla v17.4s, v3.4s, v1.4s\n"
      "fmla v25.4s, v4.4s, v1.4s\n"
      "dup v0.4s, v5.s[2]\n"
      "dup v1.4s, v5.s[3]\n"
      "fmla v10.4s, v2.4s, v0.4s\n"
      "fmla v18.4s, v3.4s, v0.4s\n"
      "fmla v26.4s, v4.4s, v0.4s\n"
      "fmla v11.4s, v2.4s, v1.4s\n"
      "fmla v19.4s, v3.4s, v1.4s\n"
      "fmla v27.4s, v4.4s, v1.4s\n"
      "dup v0.4s, v6.s[0]\n"
      "dup v1.4s, v6.s[1]\n"
      "fmla v12.4s, v2.4s, v0.4s\n"
      "fmla v20.4s, v3.4s, v0.4s\n"
      "fmla v28.4s, v4.4s, v0.4s\n"
      "fmla v13.4s, v2.4s, v1.4s\n"
      "fmla v21.4s, v3.4s, v1.4s\n"
      "fmla v29.4s, v4.4s, v1.4s\n"
      "dup v0.4s, v6.s[2]\n"
      "dup v1.4s, v6.s[3]\n"
      "fmla v14.4s, v2.4s, v0.4s\n"
      "fmla v22.4s, v3.4s, v0.4s\n"
      "fmla v30.4s, v4.4s, v0.4s\n"
      "fmla v15.4s, v2.4s, v1.4s\n"
      "fmla v23.4s, v3.4s, v1.4s\n"
      "fmla v31.4s, v4.4s, v1.4s\n"

      // Loop. Decrement loop index (depth) by 1, since we just handled 1
      // level of depth.
      "subs %[depth], %[depth], #1\n"
      "bne loop_%=\n"

      // Store accumulators
      "mov x0, %[accum_ptr]\n"
      "st1 {v8.16b}, [x0], #16\n"
      "st1 {v16.16b}, [x0], #16\n"
      "st1 {v24.16b}, [x0], #16\n"
      "st1 {v9.16b}, [x0], #16\n"
      "st1 {v17.16b}, [x0], #16\n"
      "st1 {v25.16b}, [x0], #16\n"
      "st1 {v10.16b}, [x0], #16\n"
      "st1 {v18.16b}, [x0], #16\n"
      "st1 {v26.16b}, [x0], #16\n"
      "st1 {v11.16b}, [x0], #16\n"
      "st1 {v19.16b}, [x0], #16\n"
      "st1 {v27.16b}, [x0], #16\n"
      "st1 {v12.16b}, [x0], #16\n"
      "st1 {v20.16b}, [x0], #16\n"
      "st1 {v28.16b}, [x0], #16\n"
      "st1 {v13.16b}, [x0], #16\n"
      "st1 {v21.16b}, [x0], #16\n"
      "st1 {v29.16b}, [x0], #16\n"
      "st1 {v14.16b}, [x0], #16\n"
      "st1 {v22.16b}, [x0], #16\n"
      "st1 {v30.16b}, [x0], #16\n"
      "st1 {v15.16b}, [x0], #16\n"
      "st1 {v23.16b}, [x0], #16\n"
      "st1 {v31.16b}, [x0], #16\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
      "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
      "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
      "v27", "v28", "v29", "v30", "v31");
  }
};

// This is the "most natural" kernel, using NEON multiply-with-scalar instructions.
struct NEON_64bit_GEMM_Float32_WithScalar {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    asm volatile(
      // Load accumulators
      "mov x0, %[accum_ptr]\n"
      "ld1 {v8.16b}, [x0], #16\n"
      "ld1 {v16.16b}, [x0], #16\n"
      "ld1 {v24.16b}, [x0], #16\n"
      "ld1 {v9.16b}, [x0], #16\n"
      "ld1 {v17.16b}, [x0], #16\n"
      "ld1 {v25.16b}, [x0], #16\n"
      "ld1 {v10.16b}, [x0], #16\n"
      "ld1 {v18.16b}, [x0], #16\n"
      "ld1 {v26.16b}, [x0], #16\n"
      "ld1 {v11.16b}, [x0], #16\n"
      "ld1 {v19.16b}, [x0], #16\n"
      "ld1 {v27.16b}, [x0], #16\n"
      "ld1 {v12.16b}, [x0], #16\n"
      "ld1 {v20.16b}, [x0], #16\n"
      "ld1 {v28.16b}, [x0], #16\n"
      "ld1 {v13.16b}, [x0], #16\n"
      "ld1 {v21.16b}, [x0], #16\n"
      "ld1 {v29.16b}, [x0], #16\n"
      "ld1 {v14.16b}, [x0], #16\n"
      "ld1 {v22.16b}, [x0], #16\n"
      "ld1 {v30.16b}, [x0], #16\n"
      "ld1 {v15.16b}, [x0], #16\n"
      "ld1 {v23.16b}, [x0], #16\n"
      "ld1 {v31.16b}, [x0], #16\n"

      "loop_%=:\n"

      // Load 2 Rhs cell of size 1x4 each
      "ld1 {v0.4s}, [%[rhs_ptr]], #16\n"
      "ld1 {v1.4s}, [%[rhs_ptr]], #16\n"

      // Load 3 Lhs cells of size 4x1 each
      "ld1 {v2.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v3.4s}, [%[lhs_ptr]], #16\n"
      "ld1 {v4.4s}, [%[lhs_ptr]], #16\n"

      // Multiply-accumulate
      "fmla v8.4s, v2.4s, v0.s[0]\n"
      "fmla v9.4s, v2.4s, v0.s[1]\n"
      "fmla v10.4s, v2.4s, v0.s[2]\n"
      "fmla v11.4s, v2.4s, v0.s[3]\n"
      "fmla v12.4s, v2.4s, v1.s[0]\n"
      "fmla v13.4s, v2.4s, v1.s[1]\n"
      "fmla v14.4s, v2.4s, v1.s[2]\n"
      "fmla v15.4s, v2.4s, v1.s[3]\n"
      "fmla v16.4s, v3.4s, v0.s[0]\n"
      "fmla v17.4s, v3.4s, v0.s[1]\n"
      "fmla v18.4s, v3.4s, v0.s[2]\n"
      "fmla v19.4s, v3.4s, v0.s[3]\n"
      "fmla v20.4s, v3.4s, v1.s[0]\n"
      "fmla v21.4s, v3.4s, v1.s[1]\n"
      "fmla v22.4s, v3.4s, v1.s[2]\n"
      "fmla v23.4s, v3.4s, v1.s[3]\n"
      "fmla v24.4s, v4.4s, v0.s[0]\n"
      "fmla v25.4s, v4.4s, v0.s[1]\n"
      "fmla v26.4s, v4.4s, v0.s[2]\n"
      "fmla v27.4s, v4.4s, v0.s[3]\n"
      "fmla v28.4s, v4.4s, v1.s[0]\n"
      "fmla v29.4s, v4.4s, v1.s[1]\n"
      "fmla v30.4s, v4.4s, v1.s[2]\n"
      "fmla v31.4s, v4.4s, v1.s[3]\n"

      // Loop. Decrement loop index (depth) by 1, since we just handled 1
      // level of depth.
      "subs %[depth], %[depth], #1\n"
      "bne loop_%=\n"

      // Store accumulators
      "mov x0, %[accum_ptr]\n"
      "st1 {v8.16b}, [x0], #16\n"
      "st1 {v16.16b}, [x0], #16\n"
      "st1 {v24.16b}, [x0], #16\n"
      "st1 {v9.16b}, [x0], #16\n"
      "st1 {v17.16b}, [x0], #16\n"
      "st1 {v25.16b}, [x0], #16\n"
      "st1 {v10.16b}, [x0], #16\n"
      "st1 {v18.16b}, [x0], #16\n"
      "st1 {v26.16b}, [x0], #16\n"
      "st1 {v11.16b}, [x0], #16\n"
      "st1 {v19.16b}, [x0], #16\n"
      "st1 {v27.16b}, [x0], #16\n"
      "st1 {v12.16b}, [x0], #16\n"
      "st1 {v20.16b}, [x0], #16\n"
      "st1 {v28.16b}, [x0], #16\n"
      "st1 {v13.16b}, [x0], #16\n"
      "st1 {v21.16b}, [x0], #16\n"
      "st1 {v29.16b}, [x0], #16\n"
      "st1 {v14.16b}, [x0], #16\n"
      "st1 {v22.16b}, [x0], #16\n"
      "st1 {v30.16b}, [x0], #16\n"
      "st1 {v15.16b}, [x0], #16\n"
      "st1 {v23.16b}, [x0], #16\n"
      "st1 {v31.16b}, [x0], #16\n"
      :  // outputs
      [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
      [depth] "+r"(depth)
      :  // inputs
      [accum_ptr] "r"(accum_ptr)
      :  // clobbers
      "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
      "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
      "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
      "v27", "v28", "v29", "v30", "v31");
  }
};

#endif  // __aarch64__

// BEGIN code copied from gemmlowp/internal/kernel_reference.h

// This kernel is templatized in an arbitrary Format template parameter,
// allowing it to have any arbitrary format.
template <typename tOperandType, typename tAccumulatorType, typename tFormat>
struct ReferenceKernel {
  typedef tOperandType OperandType;
  typedef tAccumulatorType AccumulatorType;
  typedef tFormat Format;

  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr, AccumulatorType* accum_ptr, int depth) {
    const int depth_cells = static_cast<int>(depth / Format::kDepth);

    // The outer loop is over the depth dimension.
    for (int dc = 0; dc < depth_cells; dc++) {
      // The next two loops are over cells of the Lhs (stacked vertically),
      // and over cells of the Rhs (stacked horizontally).
      for (int rc = 0; rc < Format::Lhs::kCells; rc++) {
        const OperandType* lhs_cell_ptr = lhs_ptr +
                                           (dc * Format::Lhs::kCells + rc) *
                                               Format::Lhs::Cell::kWidth *
                                               Format::kDepth;
        for (int cc = 0; cc < Format::Rhs::kCells; cc++) {
          const OperandType* rhs_cell_ptr = rhs_ptr +
                                             (dc * Format::Rhs::kCells + cc) *
                                                 Format::Rhs::Cell::kWidth *
                                                 Format::kDepth;

          // Now we are inside one cell of the Lhs and inside one cell
          // of the Rhs, so the remaining inner loops are just
          // traditional three loops of matrix multiplication.
          for (int di = 0; di < Format::kDepth; di++) {
            for (int ri = 0; ri < Format::Lhs::Cell::kWidth; ri++) {
              for (int ci = 0; ci < Format::Rhs::Cell::kWidth; ci++) {
                const OperandType* lhs_coeff_ptr =
                    lhs_cell_ptr +
                    OffsetIntoCell<typename Format::Lhs::Cell>(ri, di);
                const OperandType* rhs_coeff_ptr =
                    rhs_cell_ptr +
                    OffsetIntoCell<typename Format::Rhs::Cell>(ci, di);
                AccumulatorType* accumulator_coeff_ptr =
                    accum_ptr + (ri + rc * Format::Lhs::Cell::kWidth) +
                    (ci + cc * Format::Rhs::Cell::kWidth) * Format::kRows;
                *accumulator_coeff_ptr +=
                    AccumulatorType(*lhs_coeff_ptr) * AccumulatorType(*rhs_coeff_ptr);
              }
            }
          }
        }
      }
    }
  }
};

// END code copied from gemmlowp/internal/kernel_reference.h

template <typename DataType>
class CacheLineAlignedBuffer
{
public:
  CacheLineAlignedBuffer(std::size_t size)
    : size_(size) {
    data_ = nullptr;
    posix_memalign(reinterpret_cast<void**>(&data_), kCacheLineSize, size_ * sizeof(DataType));
  }

  ~CacheLineAlignedBuffer() {
    free(data_);
  }

  const DataType *data() const { return data_; }
  DataType *data() { return data_; }

  const std::size_t size() const { return size_; }

private:
  const std::size_t size_;
  DataType *data_;
};

template <typename DataType>
void FillRandom(CacheLineAlignedBuffer<DataType>* buffer) {
  static std::mt19937 generator(0);
  // 100 is smaller than any nonzero bound of the range of any data type.
  const DataType kMaxVal = DataType(100);
  const DataType kMinVal = std::is_signed<DataType>::value ? -kMaxVal : DataType(0);
  std::uniform_real_distribution<float> dist(kMinVal, kMaxVal);
  for (std::size_t i = 0; i < buffer->size(); i++) {
    buffer->data()[i] = DataType(dist(generator));
  }
}


template <typename DataType>
void FillZero(CacheLineAlignedBuffer<DataType>* buffer) {
  for (std::size_t i = 0; i < buffer->size(); i++) {
    buffer->data()[i] = DataType(0);
  }
}

template <typename DataType>
void Copy(CacheLineAlignedBuffer<DataType>* dst, const CacheLineAlignedBuffer<DataType>& src) {
  assert(dst->size() == src.size());
  memcpy(dst->data(), src.data(), src.size() * sizeof(DataType));
}

template <typename DataType>
void PrintMatrix(int rows, int cols, int rowstride, int colstride, const DataType* data) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::cerr << double(data[r * rowstride + c * colstride]) << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
}

template <typename DataType>
bool approx_equals(DataType a, DataType b) {
  return a == b;
}

template <>
bool approx_equals(float a, float b) {
  if (!a && !b) {
    return true;
  }
  return std::abs(a - b) < 1e-3f * std::min(std::abs(a), std::abs(b));
}

template <typename Kernel>
void test_kernel(int depth, const char* kernel_name)
{
  typedef typename Kernel::OperandType OperandType;
  typedef typename Kernel::AccumulatorType AccumulatorType;
  typedef typename Kernel::Format Format;
  static const int kLhsWidth = Format::Lhs::kWidth;
  static const int kRhsWidth = Format::Rhs::kWidth;

  typedef ReferenceKernel<OperandType, AccumulatorType, Format>
    ReferenceKernel;

  CacheLineAlignedBuffer<OperandType> lhs(kLhsWidth * depth);
  CacheLineAlignedBuffer<OperandType> rhs(kRhsWidth * depth);
  CacheLineAlignedBuffer<AccumulatorType> accum_initial(kLhsWidth * kRhsWidth);
  CacheLineAlignedBuffer<AccumulatorType> accum(kLhsWidth * kRhsWidth);
  CacheLineAlignedBuffer<AccumulatorType> accum_reference(kLhsWidth * kRhsWidth);

  FillRandom(&lhs);
  FillRandom(&rhs);
  FillRandom(&accum_initial);
  Copy(&accum, accum_initial);
  Copy(&accum_reference, accum_initial);
  

  ReferenceKernel::Run(lhs.data(), rhs.data(), accum_reference.data(), depth);
  Kernel::Run(lhs.data(), rhs.data(), accum.data(), depth);

  for (int l = 0; l < kLhsWidth; l++) {
    for (int r = 0; r < kRhsWidth; r++) {
      const int index = l + kLhsWidth * r;
      if (!approx_equals(accum.data()[index], accum_reference.data()[index])) {
        std::cerr << "Arithmetic error in kernel:" << std::endl << "    " <<
            kernel_name << std::endl <<
            "Wrong accumulator for depth=" << depth << ", " <<
            "at l = " << l << ", r = " << r << std::endl;
        std::cerr << "reference value: " << accum_reference.data()[index] << std::endl;
        std::cerr << "actual value:    " << accum.data()[index] << std::endl;
        if (depth <= 8) {
          std::cerr << "LHS matrix:" << std::endl;
          PrintMatrix(kLhsWidth, depth, 1, kLhsWidth, lhs.data());
          std::cerr << "RHS matrix:" << std::endl;
          PrintMatrix(depth, kRhsWidth, kRhsWidth, 1, rhs.data());
          std::cerr << "Initial Accumulator matrix:" << std::endl;
          PrintMatrix(kLhsWidth, kRhsWidth, 1, kLhsWidth, accum_initial.data());
          std::cerr << "Reference Accumulator matrix:" << std::endl;
          PrintMatrix(kLhsWidth, kRhsWidth, 1, kLhsWidth, accum_reference.data());
          std::cerr << "Actual Accumulator matrix:" << std::endl;
          PrintMatrix(kLhsWidth, kRhsWidth, 1, kLhsWidth, accum.data());
        }
        abort();
      }
    }
  }
}

template <typename Kernel>
int ops(int depth) {
  // 2x the number of multiply-accumulate scalar ops.
  return 2 *
      Kernel::Format::Lhs::kWidth *
      Kernel::Format::Rhs::kWidth *
      depth;
}

template <unsigned Modulus, typename Integer>
Integer RoundDown(Integer i) {
  return i - (i % Modulus);
}

int CacheSizeInKB() {
  static const char* cache_size_k_env = getenv("CACHE_SIZE_KB");
  static const int cache_size_k =
      cache_size_k_env ? atoi(cache_size_k_env) : kDefaultCacheSizeK;
  return cache_size_k;
}

template <typename Kernel>
int BenchmarkDepthToFitInCache() {
  const int cache_size_bytes = 1024 * CacheSizeInKB();

  // Subtract the typical size of a few cache lines, so
  // we don't need to worry too hard about e.g. some stack data.
  const int conservative_cache_size_bytes =
      cache_size_bytes - 2 * kCacheLineSize;

  // We will subtract the memory occupied by accumulators.
  typedef typename Kernel::AccumulatorType AccumulatorType;
  const int kAccumulatorBytes =
      sizeof(AccumulatorType) * Kernel::Format::Lhs::kWidth * Kernel::Format::Rhs::kWidth;

  // Compute the depth.
  typedef typename Kernel::OperandType OperandType;
  const int kBytesPerUnitOfDepth =
      sizeof(OperandType) * (Kernel::Format::Lhs::kWidth + Kernel::Format::Rhs::kWidth);
  const int unrounded_depth =
      (conservative_cache_size_bytes - kAccumulatorBytes) / kBytesPerUnitOfDepth;

  // Cap depth, to avoid unfairly favoring narrower kernels
  const int kMaxDepth = 1024;
  const int clamped_unrounded_depth = std::min(kMaxDepth, unrounded_depth);

  // Round depth down to a multiple of cache line size, which helps because
  // our kernels may crash if depth is not a multiple of the number of
  // depth level that they want to
  // handle at each loop iteration, and we don't want to require kernels
  // to be more complex. Currently all kernels process 1, 2 or 8 levels of
  // depth at a time. The main reason why that might increase in the future
  // is if registers get wider, but I don't suppose that register could
  // ever get wider than cache lines.
  return RoundDown<kCacheLineSize>(clamped_unrounded_depth);
}

double current_time_in_seconds() {
  timespec t;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

template <typename Kernel>
double benchmark() {
  // Minimum duration for this benchmark to run. If the workload finishes
  // sooner, we retry with double the number of iterations.
  static const double min_benchmark_time_in_seconds = 0.5;

  const int depth = BenchmarkDepthToFitInCache<Kernel>();

  typedef typename Kernel::OperandType OperandType;
  typedef typename Kernel::AccumulatorType AccumulatorType;

  CacheLineAlignedBuffer<OperandType> lhs(Kernel::Format::Lhs::kWidth * depth);
  CacheLineAlignedBuffer<OperandType> rhs(Kernel::Format::Rhs::kWidth * depth);
  CacheLineAlignedBuffer<AccumulatorType> accum(Kernel::Format::Lhs::kWidth * Kernel::Format::Rhs::kWidth);

  uint64_t iters_at_a_time = 1;

  for (uint64_t iters_at_a_time = 1; ; iters_at_a_time *= 2) {
    const double t_start = current_time_in_seconds();
    for (uint64_t i = 0; i < iters_at_a_time; i++) {
      Kernel::Run(lhs.data(), rhs.data(), accum.data(), depth);
    }
    const double t_end = current_time_in_seconds();
    const double elapsed = t_end - t_start;
    if (elapsed > min_benchmark_time_in_seconds) {
      return iters_at_a_time * ops<Kernel>(depth) / elapsed;
    }
  }
}

int get_num_cpus() {
  static const int n = sysconf(_SC_NPROCESSORS_CONF);
  return n;
}

#ifdef PRINT_CPUFREQ
void maybe_print_one_word_file(const std::string& filename) {
  std::ifstream file(filename);
  if (file.fail()) {
    // fail silently, the Android /sys filesystem might
    // not be universal...
    return;
  }
  std::string word;
  file >> word;
  std::cout << filename << ": " << word << std::endl;
}

void print_current_cpufreq(int cpu) {
  std::stringstream dir_stream;
  dir_stream << "/sys/devices/system/cpu/cpu" << cpu << "/cpufreq/";
  std::string dir;
  dir_stream >> dir;
  maybe_print_one_word_file(dir + "cpuinfo_cur_freq");
  maybe_print_one_word_file(dir + "scaling_cur_freq");
}
#endif

template <typename Kernel>
void benchmark_and_print_results(const char* kernel_name) {
  test_kernel<Kernel>(Kernel::Format::kDepth, kernel_name);
  test_kernel<Kernel>(2 * Kernel::Format::kDepth, kernel_name);
  test_kernel<Kernel>(1024, kernel_name);
  const int num_cpus = get_num_cpus();
  for (int cpu = 0; cpu < num_cpus; cpu++) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(cpu_set_t), &s);

    std::cout << kernel_name <<
        "(depth=" << BenchmarkDepthToFitInCache<Kernel>() <<
        ") on CPU #" << cpu << ": " <<
        benchmark<Kernel>() * 1e-9f << " Gop/s" << std::endl;

#ifdef PRINT_CPUFREQ
    print_current_cpufreq(cpu);
#endif
  }
}

#define BENCHMARK(Kernel) \
    do { \
      benchmark_and_print_results<Kernel>(#Kernel); \
    } while (false)

int main() {
  std::cout << "There are " << get_num_cpus() << " CPU cores." << std::endl;
  std::cout << "Targeting a cache size of " << CacheSizeInKB() << " K" << std::endl;

#ifdef __arm__
  std::cout << "CPU architecture: ARM 32bit" << std::endl;
  BENCHMARK(NEON_32bit_GEMM_Uint8Operands_Uint32Accumulators);
  BENCHMARK(NEON_32bit_GEMM_Uint8Operands_Uint32Accumulators_noexpand);
  BENCHMARK(NEON_32bit_GEMM_Int32_WithScalar);
  BENCHMARK(NEON_32bit_GEMM_Float32_MLA_WithVectorDuplicatingScalar);
#ifdef __ARM_FEATURE_FMA
  BENCHMARK(NEON_32bit_GEMM_Float32_FMA_WithVectorDuplicatingScalar);
#endif
  BENCHMARK(NEON_32bit_GEMM_Float32_MLA_WithScalar);
  BENCHMARK(NEON_32bit_GEMM_Float32_MLA_Rotating);
#ifdef __ARM_FEATURE_FMA
  BENCHMARK(NEON_32bit_GEMM_Float32_FMA_Rotating);
#endif
#endif

#ifdef __aarch64__
  std::cout << "CPU architecture: ARM 64bit" << std::endl;
  BENCHMARK(NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators);
  BENCHMARK(NEON_64bit_GEMM_Int32_WithScalar);
  BENCHMARK(NEON_64bit_GEMM_Float32_WithVectorDuplicatingScalar);
  BENCHMARK(NEON_64bit_GEMM_Float32_WithScalar);
#endif
}
