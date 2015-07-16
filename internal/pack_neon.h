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

// pack_neon.h: optimized NEON specializations of the templates in pack.h.

#ifndef GEMMLOWP_INTERNAL_PACK_NEON_H_
#define GEMMLOWP_INTERNAL_PACK_NEON_H_

#include "pack.h"

#include <arm_neon.h>

namespace gemmlowp {

typedef SideMap<const std::uint8_t, SideMapOrder::WidthMajor> WidthMajorUint8SideMap;

// The paths here are specifically armv7 assembly, not armv8
#ifdef GEMMLOWP_NEON32

// Specialization for 3 Cells of width 4, depth 2.
// This is the LHS format used by NEONKernel12x4Depth2.
typedef KernelSideFormat<CellFormat<4, 2>, 3> SideFormat3Cells4x2;
template <>
class PackSideBlockImpl<WidthMajorUint8SideMap, SideFormat3Cells4x2>
    : public PackSideBlockImplGeneric<WidthMajorUint8SideMap,
                                      SideFormat3Cells4x2> {
 public:
  typedef SideFormat3Cells4x2 SideFormat;
  typedef WidthMajorUint8SideMap SrcMapType;
  typedef PackSideBlockImplGeneric<SrcMapType, SideFormat> Base;

  PackSideBlockImpl(PackedSideBlock<SideFormat>* packed_side_block,
                    const SrcMapType& src_map)
      : Base(packed_side_block, src_map) {}

 protected:
  static const int KernelRows = SideFormat::kWidth;

  virtual void PackRun(int start_width, int width, int start_depth, int depth) {
    // Fall back to generic path for packing too narrow runs.
    if (width < SideFormat::kWidth) {
      Base::PackRun(start_width, width, start_depth, depth);
      return;
    }

    const std::uint8_t* src_ptr =
        Base::src_map().data(start_width, start_depth);
    const int stride = Base::src_map().stride();
    assert(src_ptr + stride ==
           Base::src_map().data(start_width + 1, start_depth));
    assert(src_ptr + 1 == Base::src_map().data(start_width, start_depth + 1));

    // Prefetch data.
    for (int d = 0; d < depth; d += kDefaultCacheLineSize) {
      for (int i = 0; i < KernelRows; i++) {
        Prefetch(src_ptr + i * stride + d);
      }
    }

    const int AlignedDepth16 = RoundDown<16>(depth);
    if (AlignedDepth16) {
      // Fast inner loop for handling multiples of 16 levels of depth
      ScopedProfilingLabel label("optimized kernel");

      std::int32_t* rank_one_update_ptr =
          Base::packed_side_block()->rank_one_update() + start_width;

      std::uint8_t* dst_ptr = Base::packed_side_block()->current_data();
      const std::uint8_t* dst_end_ptr = dst_ptr + KernelRows * AlignedDepth16;

      __attribute__((aligned(32))) std::uint8_t buf[KernelRows * 16];
      __attribute__((aligned(16))) std::int32_t sumsbuf[12];
      asm volatile(
          "mov r4, %[src_ptr]\n"
          "mov r3, %[dst_ptr]\n"

          // We will accumulate the rank_one_update sums in q12--q14.
          "vmov.s32 q12, #0\n"
          "vmov.s32 q13, q12\n"
          "vmov.s32 q14, q12\n"

          // Main loop.
          "loop_PackSideBlockImplNEON4x3x2_%=:\n"

          // Load a 12x16 block into the 12 registers q0--q11.
          // So each of these 12 registers contains 16 entries of
          // one line, and there are 12 lines being processed.
          "mov r0, r4\n"
          "add r4, #16\n"
          "vld1.8 {d0,d1}, [r0], %[stride]\n"
          "vld1.8 {d2,d3}, [r0], %[stride]\n"
          "vld1.8 {d4,d5}, [r0], %[stride]\n"
          "vld1.8 {d6,d7}, [r0], %[stride]\n"
          "vld1.8 {d8,d9}, [r0], %[stride]\n"
          "vld1.8 {d10,d11}, [r0], %[stride]\n"
          "vld1.8 {d12,d13}, [r0], %[stride]\n"
          "vld1.8 {d14,d15}, [r0], %[stride]\n"
          "vld1.8 {d16,d17}, [r0], %[stride]\n"
          "vld1.8 {d18,d19}, [r0], %[stride]\n"
          "vld1.8 {d20,d21}, [r0], %[stride]\n"
          "vld1.8 {d22,d23}, [r0], %[stride]\n"

          // The CellOrder is the opposite of the MapOrder here
          // so we need to transpose the data in each cell.
          // We do so using an auxiliary buffer and the vst4 instruction,
          // which takes 4 registers and stores them interleaved.
          "mov r1, %[buf]\n"
          "vst4.8 {d0, d2, d4, d6}, [r1:256]!\n"
          "vst4.8 {d1, d3, d5, d7}, [r1:256]!\n"
          "vst4.8 {d8, d10, d12, d14}, [r1:256]!\n"
          "vst4.8 {d9, d11, d13, d15}, [r1:256]!\n"
          "vst4.8 {d16, d18, d20, d22}, [r1:256]!\n"
          "vst4.8 {d17, d19, d21, d23}, [r1:256]!\n"

          // Reload the data from our auxiliary buffer back into
          // the same regisers; it is now transposed so each of the
          // 24 d-registers from d0 to d23 now contains one cell,
          // just not yet in the right sequence. We will still have
          // to permute those cells.
          "mov r1, %[buf]\n"
          "vld1.8 {d0,d1,d2,d3}, [r1:256]!\n"
          "vld1.8 {d4,d5,d6,d7}, [r1:256]!\n"
          "vld1.8 {d8,d9,d10,d11}, [r1:256]!\n"
          "vld1.8 {d12,d13,d14,d15}, [r1:256]!\n"
          "vld1.8 {d16,d17,d18,d19}, [r1:256]!\n"
          "vld1.8 {d20,d21,d22,d23}, [r1:256]!\n"

          // Store these cells back to memory, now in the right
          // sequence.
          "vst1.8 d0, [r3:64]!\n"
          "vst1.8 d8, [r3:64]!\n"
          "vst1.8 d16, [r3:64]!\n"
          "vst1.8 d1, [r3:64]!\n"
          "vst1.8 d9, [r3:64]!\n"
          "vst1.8 d17, [r3:64]!\n"
          "vst1.8 d2, [r3:64]!\n"
          "vst1.8 d10, [r3:64]!\n"
          "vst1.8 d18, [r3:64]!\n"
          "vst1.8 d3, [r3:64]!\n"
          "vst1.8 d11, [r3:64]!\n"
          "vst1.8 d19, [r3:64]!\n"
          "vst1.8 d4, [r3:64]!\n"
          "vst1.8 d12, [r3:64]!\n"
          "vst1.8 d20, [r3:64]!\n"
          "vst1.8 d5, [r3:64]!\n"
          "vst1.8 d13, [r3:64]!\n"
          "vst1.8 d21, [r3:64]!\n"
          "vst1.8 d6, [r3:64]!\n"
          "vst1.8 d14, [r3:64]!\n"
          "vst1.8 d22, [r3:64]!\n"
          "vst1.8 d7, [r3:64]!\n"
          "vst1.8 d15, [r3:64]!\n"
          "vst1.8 d23, [r3:64]!\n"

          // Now we are done packing this 12x16 block. We still have
          // to accumulate the rank-one-update sums for it.

          // Add 8-bit values pair-wise into 16-bit values.
          "vaddl.u8 q0, d0, d1\n"
          "vaddl.u8 q1, d2, d3\n"
          "vaddl.u8 q2, d4, d5\n"
          "vaddl.u8 q3, d6, d7\n"
          "vaddl.u8 q4, d8, d9\n"
          "vaddl.u8 q5, d10, d11\n"
          "vaddl.u8 q6, d12, d13\n"
          "vaddl.u8 q7, d14, d15\n"
          "vaddl.u8 q8, d16, d17\n"
          "vaddl.u8 q9, d18, d19\n"
          "vaddl.u8 q10, d20, d21\n"
          "vaddl.u8 q11, d22, d23\n"

          // Add 16-bit values pair-wise into 32-bit values.
          "vaddl.u16 q0, d0, d1\n"
          "vaddl.u16 q1, d2, d3\n"
          "vaddl.u16 q2, d4, d5\n"
          "vaddl.u16 q3, d6, d7\n"
          "vaddl.u16 q4, d8, d9\n"
          "vaddl.u16 q5, d10, d11\n"
          "vaddl.u16 q6, d12, d13\n"
          "vaddl.u16 q7, d14, d15\n"
          "vaddl.u16 q8, d16, d17\n"
          "vaddl.u16 q9, d18, d19\n"
          "vaddl.u16 q10, d20, d21\n"
          "vaddl.u16 q11, d22, d23\n"

          // Accumulate the 32-bit sums into our accumulators q12--q14.
          "vadd.s32 q12, q12, q0\n"
          "vadd.s32 q13, q13, q4\n"
          "vadd.s32 q14, q14, q8\n"
          "vadd.s32 q12, q12, q1\n"
          "vadd.s32 q13, q13, q5\n"
          "vadd.s32 q14, q14, q9\n"
          "vadd.s32 q12, q12, q2\n"
          "vadd.s32 q13, q13, q6\n"
          "vadd.s32 q14, q14, q10\n"
          "vadd.s32 q12, q12, q3\n"
          "vadd.s32 q13, q13, q7\n"
          "vadd.s32 q14, q14, q11\n"

          // End of main loop.
          "cmp r3, %[dst_end_ptr]\n"
          "bne loop_PackSideBlockImplNEON4x3x2_%=\n"

          // Store our rank-one-update accumulator registers to the
          // sums buffer.
          "mov r0, %[sumsbuf]\n"
          "vst1.32 {d24, d25}, [r0:128]!\n"
          "vst1.32 {d26, d27}, [r0:128]!\n"
          "vst1.32 {d28, d29}, [r0:128]!\n"

          :  // no outputs
          :  // inputs
          [dst_ptr] "r"(dst_ptr), [src_ptr] "r"(src_ptr),
          [dst_end_ptr] "r"(dst_end_ptr), [stride] "r"(stride), [buf] "r"(buf),
          [sumsbuf] "r"(sumsbuf)
          :  // clobbers
          "cc", "memory", "r0", "r1", "r3", "r4",
          // note: someone on internet says that quad registers are
          // unsupported in the clobber list!
          "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
          "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
          "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
          "d31");

      // Accumulate the final rank_one_update vector.
      int32x4x3_t sums;
      sums.val[0] = vld1q_s32(sumsbuf);
      sums.val[1] = vld1q_s32(sumsbuf + 4);
      sums.val[2] = vld1q_s32(sumsbuf + 8);
      sums.val[0] = vmulq_n_s32(
          sums.val[0], Base::packed_side_block()->rank_one_update_multiplier());
      sums.val[1] = vmulq_n_s32(
          sums.val[1], Base::packed_side_block()->rank_one_update_multiplier());
      sums.val[2] = vmulq_n_s32(
          sums.val[2], Base::packed_side_block()->rank_one_update_multiplier());

      int32x4x3_t old_sums;
      old_sums.val[0] = vld1q_s32(rank_one_update_ptr + 0);
      old_sums.val[1] = vld1q_s32(rank_one_update_ptr + 4);
      old_sums.val[2] = vld1q_s32(rank_one_update_ptr + 8);
      sums.val[0] = vaddq_s32(sums.val[0], old_sums.val[0]);
      sums.val[1] = vaddq_s32(sums.val[1], old_sums.val[1]);
      sums.val[2] = vaddq_s32(sums.val[2], old_sums.val[2]);
      vst1q_s32(rank_one_update_ptr + 0, sums.val[0]);
      vst1q_s32(rank_one_update_ptr + 4, sums.val[1]);
      vst1q_s32(rank_one_update_ptr + 8, sums.val[2]);
    }

    // We are done handling groups of 16 levels of depth; there may be
    // a leftover for which we use the generic path.
    Base::packed_side_block()->seek_forward_n_cells(
        SideFormat::kCells * AlignedDepth16 / SideFormat::kDepth);
    Base::PackRun(start_width, width, start_depth + AlignedDepth16,
                  depth - AlignedDepth16);
  }
};

// Specialization for 5 Cells of width 4, depth 4.
// This is the LHS format used by NEONKernel20x1Depth4.
typedef KernelSideFormat<CellFormat<4, 4>, 5> SideFormat5Cells4x4;

template <>
class PackSideBlockImpl<WidthMajorUint8SideMap, SideFormat5Cells4x4>
    : public PackSideBlockImplGeneric<WidthMajorUint8SideMap,
                                      SideFormat5Cells4x4> {
 public:
  typedef SideFormat5Cells4x4 SideFormat;
  typedef WidthMajorUint8SideMap SrcMapType;
  typedef PackSideBlockImplGeneric<SrcMapType, SideFormat> Base;

  PackSideBlockImpl(PackedSideBlock<SideFormat>* packed_side_block,
                    const SrcMapType& src_map)
      : Base(packed_side_block, src_map) {}

 protected:
  virtual void PackRun(int start_width, int width, int start_depth, int depth) {
    // Fall back to generic path for packing too narrow runs.
    if (width < SideFormat::kWidth) {
      Base::PackRun(start_width, width, start_depth, depth);
      return;
    }

    const std::uint8_t* src_ptr =
        Base::src_map().data(start_width, start_depth);
    const int stride = Base::src_map().stride();
    assert(src_ptr + stride ==
           Base::src_map().data(start_width + 1, start_depth));
    assert(src_ptr + 1 == Base::src_map().data(start_width, start_depth + 1));

    // Prefetch data.
    for (int d = 0; d < depth; d += kDefaultCacheLineSize) {
      for (int i = 0; i < SideFormat::kWidth; i++) {
        Prefetch(src_ptr + i * stride + d);
      }
    }

    const int AlignedDepth8 = RoundDown<8>(depth);
    if (AlignedDepth8) {
      // Fast inner loop for handling multiples of 8 levels of depth
      ScopedProfilingLabel label("optimized kernel");

      std::int32_t* rank_one_update_ptr =
          Base::packed_side_block()->rank_one_update() + start_width;

      std::uint8_t* dst_ptr = Base::packed_side_block()->current_data();
      const std::uint8_t* dst_end_ptr =
          dst_ptr + SideFormat::kWidth * AlignedDepth8;

      __attribute__((aligned(32))) std::uint8_t buf[SideFormat::kWidth * 8];
      __attribute__((aligned(16))) std::int32_t sumsbuf[20];
      asm volatile(
          "mov r4, %[src_ptr]\n"
          "mov r3, %[dst_ptr]\n"

          // We will accumulate the rank_one_update sums in q10--q14.
          "vmov.s32 q10, #0\n"
          "vmov.s32 q11, q10\n"
          "vmov.s32 q12, q10\n"
          "vmov.s32 q13, q10\n"
          "vmov.s32 q14, q10\n"

          // Main loop.
          "loop_PackSideBlockImplNEON4x5x4_%=:\n"

          // Load a 20x8 block into the 20 registers d0--d19.
          // So each of these 20 registers contains 8 entries of
          // one line, and there are 20 lines being processed.
          "mov r0, r4\n"
          "add r4, #8\n"
          "vld1.8 d0, [r0], %[stride]\n"
          "vld1.8 d1, [r0], %[stride]\n"
          "vld1.8 d2, [r0], %[stride]\n"
          "vld1.8 d3, [r0], %[stride]\n"
          "vld1.8 d4, [r0], %[stride]\n"
          "vld1.8 d5, [r0], %[stride]\n"
          "vld1.8 d6, [r0], %[stride]\n"
          "vld1.8 d7, [r0], %[stride]\n"
          "vld1.8 d8, [r0], %[stride]\n"
          "vld1.8 d9, [r0], %[stride]\n"
          "vld1.8 d10, [r0], %[stride]\n"
          "vld1.8 d11, [r0], %[stride]\n"
          "vld1.8 d12, [r0], %[stride]\n"
          "vld1.8 d13, [r0], %[stride]\n"
          "vld1.8 d14, [r0], %[stride]\n"
          "vld1.8 d15, [r0], %[stride]\n"
          "vld1.8 d16, [r0], %[stride]\n"
          "vld1.8 d17, [r0], %[stride]\n"
          "vld1.8 d18, [r0], %[stride]\n"
          "vld1.8 d19, [r0], %[stride]\n"

          // The CellOrder is the opposite of the MapOrder here
          // so we need to transpose the data in each cell.
          // We do so using an auxiliary buffer and the vst4 instruction,
          // which takes 4 registers and stores them interleaved.
          "mov r1, %[buf]\n"
          "vst4.8 {d0, d1, d2, d3}, [r1:256]!\n"
          "vst4.8 {d4, d5, d6, d7}, [r1:256]!\n"
          "vst4.8 {d8, d9, d10, d11}, [r1:256]!\n"
          "vst4.8 {d12, d13, d14, d15}, [r1:256]!\n"
          "vst4.8 {d16, d17, d18, d19}, [r1:256]!\n"

          // Reload the data from our auxiliary buffer back into
          // the same regisers; it is now transposed so each of the
          // 20 d-registers from d0 to d19 now contains half of one cell,
          // just not yet in the right sequence. We will still have
          // to permute those cells.
          "mov r1, %[buf]\n"
          "vld1.8 {d0,d1,d2,d3}, [r1:256]!\n"
          "vld1.8 {d4,d5,d6,d7}, [r1:256]!\n"
          "vld1.8 {d8,d9,d10,d11}, [r1:256]!\n"
          "vld1.8 {d12,d13,d14,d15}, [r1:256]!\n"
          "vld1.8 {d16,d17,d18,d19}, [r1:256]!\n"

          // Store these cells back to memory, now in the right
          // sequence.
          "vst1.8 d0, [r3:64]!\n"
          "vst1.8 d1, [r3:64]!\n"
          "vst1.8 d4, [r3:64]!\n"
          "vst1.8 d5, [r3:64]!\n"
          "vst1.8 d8, [r3:64]!\n"
          "vst1.8 d9, [r3:64]!\n"
          "vst1.8 d12, [r3:64]!\n"
          "vst1.8 d13, [r3:64]!\n"
          "vst1.8 d16, [r3:64]!\n"
          "vst1.8 d17, [r3:64]!\n"
          "vst1.8 d2, [r3:64]!\n"
          "vst1.8 d3, [r3:64]!\n"
          "vst1.8 d6, [r3:64]!\n"
          "vst1.8 d7, [r3:64]!\n"
          "vst1.8 d10, [r3:64]!\n"
          "vst1.8 d11, [r3:64]!\n"
          "vst1.8 d14, [r3:64]!\n"
          "vst1.8 d15, [r3:64]!\n"
          "vst1.8 d18, [r3:64]!\n"
          "vst1.8 d19, [r3:64]!\n"

          // Now we are done packing this 20x8 block. We still have
          // to accumulate the rank-one-update sums for it.

          // Add 8-bit values pair-wise into 16-bit values.
          "vaddl.u8 q0, d0, d1\n"
          "vaddl.u8 q1, d2, d3\n"
          "vaddl.u8 q2, d4, d5\n"
          "vaddl.u8 q3, d6, d7\n"
          "vaddl.u8 q4, d8, d9\n"
          "vaddl.u8 q5, d10, d11\n"
          "vaddl.u8 q6, d12, d13\n"
          "vaddl.u8 q7, d14, d15\n"
          "vaddl.u8 q8, d16, d17\n"
          "vaddl.u8 q9, d18, d19\n"

          // Add 16-bit values pair-wise into 32-bit values.
          "vaddl.u16 q0, d0, d1\n"
          "vaddl.u16 q1, d2, d3\n"
          "vaddl.u16 q2, d4, d5\n"
          "vaddl.u16 q3, d6, d7\n"
          "vaddl.u16 q4, d8, d9\n"
          "vaddl.u16 q5, d10, d11\n"
          "vaddl.u16 q6, d12, d13\n"
          "vaddl.u16 q7, d14, d15\n"
          "vaddl.u16 q8, d16, d17\n"
          "vaddl.u16 q9, d18, d19\n"

          // Accumulate the 32-bit sums into our accumulators q10--q14.
          "vadd.s32 q10, q10, q0\n"
          "vadd.s32 q11, q11, q2\n"
          "vadd.s32 q12, q12, q4\n"
          "vadd.s32 q13, q13, q6\n"
          "vadd.s32 q14, q14, q8\n"
          "vadd.s32 q10, q10, q1\n"
          "vadd.s32 q11, q11, q3\n"
          "vadd.s32 q12, q12, q5\n"
          "vadd.s32 q13, q13, q7\n"
          "vadd.s32 q14, q14, q9\n"

          // End of main loop.
          "cmp r3, %[dst_end_ptr]\n"
          "bne loop_PackSideBlockImplNEON4x5x4_%=\n"

          // Store our rank-one-update accumulator registers to the
          // sums buffer.
          "mov r0, %[sumsbuf]\n"
          "vst1.32 {d20, d21}, [r0:128]!\n"
          "vst1.32 {d22, d23}, [r0:128]!\n"
          "vst1.32 {d24, d25}, [r0:128]!\n"
          "vst1.32 {d26, d27}, [r0:128]!\n"
          "vst1.32 {d28, d29}, [r0:128]!\n"
          :  // no outputs
          :  // inputs
          [dst_ptr] "r"(dst_ptr), [src_ptr] "r"(src_ptr),
          [dst_end_ptr] "r"(dst_end_ptr), [stride] "r"(stride), [buf] "r"(buf),
          [sumsbuf] "r"(sumsbuf)
          :  // clobbers
          "cc", "memory", "r0", "r1", "r3", "r4",
          // note: someone on internet says that quad registers are
          // unsupported in the clobber list!
          "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
          "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
          "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
          "d31");

      // Accumulate the final rank_one_update vector.
      int32x4_t sums[5];
      sums[0] = vld1q_s32(sumsbuf);
      sums[1] = vld1q_s32(sumsbuf + 4);
      sums[2] = vld1q_s32(sumsbuf + 8);
      sums[3] = vld1q_s32(sumsbuf + 12);
      sums[4] = vld1q_s32(sumsbuf + 16);

      sums[0] = vmulq_n_s32(
          sums[0], Base::packed_side_block()->rank_one_update_multiplier());
      sums[1] = vmulq_n_s32(
          sums[1], Base::packed_side_block()->rank_one_update_multiplier());
      sums[2] = vmulq_n_s32(
          sums[2], Base::packed_side_block()->rank_one_update_multiplier());
      sums[3] = vmulq_n_s32(
          sums[3], Base::packed_side_block()->rank_one_update_multiplier());
      sums[4] = vmulq_n_s32(
          sums[4], Base::packed_side_block()->rank_one_update_multiplier());

      int32x4_t old_sums[5];
      old_sums[0] = vld1q_s32(rank_one_update_ptr + 0);
      old_sums[1] = vld1q_s32(rank_one_update_ptr + 4);
      old_sums[2] = vld1q_s32(rank_one_update_ptr + 8);
      old_sums[3] = vld1q_s32(rank_one_update_ptr + 12);
      old_sums[4] = vld1q_s32(rank_one_update_ptr + 16);

      sums[0] = vaddq_s32(sums[0], old_sums[0]);
      sums[1] = vaddq_s32(sums[1], old_sums[1]);
      sums[2] = vaddq_s32(sums[2], old_sums[2]);
      sums[3] = vaddq_s32(sums[3], old_sums[3]);
      sums[4] = vaddq_s32(sums[4], old_sums[4]);

      vst1q_s32(rank_one_update_ptr + 0, sums[0]);
      vst1q_s32(rank_one_update_ptr + 4, sums[1]);
      vst1q_s32(rank_one_update_ptr + 8, sums[2]);
      vst1q_s32(rank_one_update_ptr + 12, sums[3]);
      vst1q_s32(rank_one_update_ptr + 16, sums[4]);
    }

    // We are done handling groups of 8 levels of depth; there may be
    // a leftover for which we use the generic path.
    Base::packed_side_block()->seek_forward_n_cells(
        SideFormat::kCells * AlignedDepth8 / SideFormat::kDepth);
    Base::PackRun(start_width, width, start_depth + AlignedDepth8,
                  depth - AlignedDepth8);
  }
};

// Specialization for 1 Cell of width 8, depth 4.
// This is the LHS format used by NEONKernel8x1Depth4.
typedef KernelSideFormat<CellFormat<8, 4>, 1> SideFormat1Cell8x4;

template <>
class PackSideBlockImpl<WidthMajorUint8SideMap, SideFormat1Cell8x4>
    : public PackSideBlockImplGeneric<WidthMajorUint8SideMap,
                                      SideFormat1Cell8x4> {
 public:
  typedef SideFormat1Cell8x4 SideFormat;
  typedef WidthMajorUint8SideMap SrcMapType;
  typedef PackSideBlockImplGeneric<SrcMapType, SideFormat> Base;

  PackSideBlockImpl(PackedSideBlock<SideFormat>* packed_side_block,
                    const SrcMapType& src_map)
      : Base(packed_side_block, src_map) {}

 protected:
  virtual void PackRun(int start_width, int width, int start_depth, int depth) {
    // Fall back to generic path for packing too narrow runs.
    if (width < SideFormat::kWidth) {
      Base::PackRun(start_width, width, start_depth, depth);
      return;
    }

    const std::uint8_t* src_ptr =
        Base::src_map().data(start_width, start_depth);
    const int stride = Base::src_map().stride();
    assert(src_ptr + stride ==
           Base::src_map().data(start_width + 1, start_depth));
    assert(src_ptr + 1 == Base::src_map().data(start_width, start_depth + 1));

    // Prefetch data.
    for (int d = 0; d < depth; d += kDefaultCacheLineSize) {
      for (int i = 0; i < SideFormat::kWidth; i++) {
        Prefetch(src_ptr + i * stride + d);
      }
    }

    const int AlignedDepth8 = RoundDown<8>(depth);
    if (AlignedDepth8) {
      // Fast inner loop for handling multiples of 8 levels of depth
      ScopedProfilingLabel label("optimized kernel");

      std::int32_t* rank_one_update_ptr =
          Base::packed_side_block()->rank_one_update() + start_width;

      std::uint8_t* dst_ptr = Base::packed_side_block()->current_data();
      const std::uint8_t* dst_end_ptr =
          dst_ptr + SideFormat::kWidth * AlignedDepth8;

      __attribute__((aligned(16))) std::int32_t sumsbuf[8];
      asm volatile(
          "mov r4, %[src_ptr]\n"
          "mov r3, %[dst_ptr]\n"

          // We will accumulate the rank_one_update sums in q10--q11.
          "vmov.s32 q10, #0\n"
          "vmov.s32 q11, q10\n"

          // Main loop.
          "loop_PackSideBlockImplNEON8x1x4_%=:\n"

          // Load a 8x8 block into the 8 registers d0--d7.
          // So each of these 8 registers contains 8 entries of
          // one line, and there are 8 lines being processed.
          "mov r0, r4\n"
          "add r4, #8\n"
          "vld1.8 d0, [r0], %[stride]\n"
          "vld1.8 d1, [r0], %[stride]\n"
          "vld1.8 d2, [r0], %[stride]\n"
          "vld1.8 d3, [r0], %[stride]\n"
          "vld1.8 d4, [r0], %[stride]\n"
          "vld1.8 d5, [r0], %[stride]\n"
          "vld1.8 d6, [r0], %[stride]\n"
          "vld1.8 d7, [r0], %[stride]\n"

          // The CellOrder is the opposite of the MapOrder here
          // so we need to transpose the data in each cell.
          // Fortunately in this case, since what we have to transpose
          // is a 8x8 block of 8-bit values, we can do so in-place with
          // vtrn instructions; no need for an auxiliary buffer here.
          "vtrn.8 d0, d1\n"
          "vtrn.8 d2, d3\n"
          "vtrn.8 d4, d5\n"
          "vtrn.8 d6, d7\n"
          "vtrn.16 q0, q1\n"
          "vtrn.16 q2, q3\n"
          "vtrn.32 q0, q2\n"
          "vtrn.32 q1, q3\n"

          // Store the packed data to memory.
          "vst1.8 {d0, d1, d2, d3}, [r3:64]!\n"
          "vst1.8 {d4, d5, d6, d7}, [r3:64]!\n"

          // Now we are done packing this 20x8 block. We still have
          // to accumulate the rank-one-update sums for it.

          // Add 8-bit values pair-wise into 16-bit values.
          "vaddl.u8 q0, d0, d1\n"
          "vaddl.u8 q1, d2, d3\n"
          "vaddl.u8 q2, d4, d5\n"
          "vaddl.u8 q3, d6, d7\n"

          // Add these 16-bit values into the 32-bit accumulators q10--q11.
          "vaddw.u16 q10, q10, d0\n"
          "vaddw.u16 q11, q11, d1\n"
          "vaddw.u16 q10, q10, d2\n"
          "vaddw.u16 q11, q11, d3\n"
          "vaddw.u16 q10, q10, d4\n"
          "vaddw.u16 q11, q11, d5\n"
          "vaddw.u16 q10, q10, d6\n"
          "vaddw.u16 q11, q11, d7\n"

          // End of main loop.
          "cmp r3, %[dst_end_ptr]\n"
          "bne loop_PackSideBlockImplNEON8x1x4_%=\n"

          // Store our rank-one-update accumulator registers to the
          // sums buffer.
          "mov r0, %[sumsbuf]\n"
          "vst1.32 {d20, d21}, [r0:128]!\n"
          "vst1.32 {d22, d23}, [r0:128]!\n"
          :  // no outputs
          :  // inputs
          [dst_ptr] "r"(dst_ptr), [src_ptr] "r"(src_ptr),
          [dst_end_ptr] "r"(dst_end_ptr), [stride] "r"(stride),
          [sumsbuf] "r"(sumsbuf)
          :  // clobbers
          "cc", "memory", "r0", "r3", "r4",
          // note: someone on internet says that quad registers are
          // unsupported in the clobber list!
          "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
          "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
          "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
          "d31");

      // Accumulate the final rank_one_update vector.
      int32x4_t sums[2];
      sums[0] = vld1q_s32(sumsbuf);
      sums[1] = vld1q_s32(sumsbuf + 4);

      sums[0] = vmulq_n_s32(
          sums[0], Base::packed_side_block()->rank_one_update_multiplier());
      sums[1] = vmulq_n_s32(
          sums[1], Base::packed_side_block()->rank_one_update_multiplier());

      int32x4_t old_sums[2];
      old_sums[0] = vld1q_s32(rank_one_update_ptr + 0);
      old_sums[1] = vld1q_s32(rank_one_update_ptr + 4);

      sums[0] = vaddq_s32(sums[0], old_sums[0]);
      sums[1] = vaddq_s32(sums[1], old_sums[1]);

      vst1q_s32(rank_one_update_ptr + 0, sums[0]);
      vst1q_s32(rank_one_update_ptr + 4, sums[1]);
    }

    // We are done handling groups of 8 levels of depth; there may be
    // a leftover for which we use the generic path.
    Base::packed_side_block()->seek_forward_n_cells(
        SideFormat::kCells * AlignedDepth8 / SideFormat::kDepth);
    Base::PackRun(start_width, width, start_depth + AlignedDepth8,
                  depth - AlignedDepth8);
  }
};

#endif  // GEMMLOWP_NEON32
// Paths below do not use assembly, so they should work on both
// 32bit and 64bit instruction sets.

// Partial specialization for 1 Cell of width 4, and any Depth.
// This is the RHS format used by NEONKernel12x4Depth2.
template <int Depth>
using SideFormat1Cell4xD = KernelSideFormat<CellFormat<4, Depth>, 1>;
template <int Depth>
class PackSideBlockImpl<WidthMajorUint8SideMap, SideFormat1Cell4xD<Depth>>
    : public PackSideBlockImplGeneric<WidthMajorUint8SideMap,
                                      SideFormat1Cell4xD<Depth>> {
 public:
  typedef SideFormat1Cell4xD<Depth> SideFormat;
  typedef WidthMajorUint8SideMap SrcMapType;
  typedef PackSideBlockImplGeneric<SrcMapType, SideFormat> Base;

  PackSideBlockImpl(PackedSideBlock<SideFormat>* packed_side_block,
                    const SrcMapType& src_map)
      : Base(packed_side_block, src_map) {}

 protected:
  virtual void PackRun(int start_width, int width, int start_depth, int depth) {
    ScopedProfilingLabel label("optimized kernel");

    std::int32_t* rank_one_update_ptr =
        Base::packed_side_block()->rank_one_update() + start_width;
    const int AlignedDepth16 = RoundDown<16>(depth);
    const std::uint8_t* src_line0ptr =
        width < 1 ? nullptr : Base::src_map().data(start_width, start_depth);
    const std::uint8_t* src_line1ptr =
        width < 2 ? nullptr
                  : Base::src_map().data(start_width + 1, start_depth);
    const std::uint8_t* src_line2ptr =
        width < 3 ? nullptr
                  : Base::src_map().data(start_width + 2, start_depth);
    const std::uint8_t* src_line3ptr =
        width < 4 ? nullptr
                  : Base::src_map().data(start_width + 3, start_depth);
    std::uint8_t* dst_ptr = Base::packed_side_block()->current_data();
    const std::uint8_t* dst_end_ptr =
        dst_ptr + SideFormat::kWidth * AlignedDepth16;
    int32x4x4_t local_col_sums;
    local_col_sums.val[0] = vdupq_n_s32(0);
    local_col_sums.val[1] = vdupq_n_s32(0);
    local_col_sums.val[2] = vdupq_n_s32(0);
    local_col_sums.val[3] = vdupq_n_s32(0);

    switch (width) {
      case 4:
        while (dst_ptr != dst_end_ptr) {
          uint8x16x4_t src_regs;
          src_regs.val[0] = vld1q_u8(src_line0ptr);
          src_regs.val[1] = vld1q_u8(src_line1ptr);
          src_regs.val[2] = vld1q_u8(src_line2ptr);
          src_regs.val[3] = vld1q_u8(src_line3ptr);
          uint16x8x4_t s;
          s.val[0] = vaddl_u8(vget_low_u8(src_regs.val[0]),
                              vget_high_u8(src_regs.val[0]));
          s.val[1] = vaddl_u8(vget_low_u8(src_regs.val[1]),
                              vget_high_u8(src_regs.val[1]));
          s.val[2] = vaddl_u8(vget_low_u8(src_regs.val[2]),
                              vget_high_u8(src_regs.val[2]));
          s.val[3] = vaddl_u8(vget_low_u8(src_regs.val[3]),
                              vget_high_u8(src_regs.val[3]));
          local_col_sums.val[0] =
              vaddq_s32(local_col_sums.val[0],
                        vreinterpretq_s32_u32(vaddl_u16(
                            vget_low_u16(s.val[0]), vget_high_u16(s.val[0]))));
          local_col_sums.val[1] =
              vaddq_s32(local_col_sums.val[1],
                        vreinterpretq_s32_u32(vaddl_u16(
                            vget_low_u16(s.val[1]), vget_high_u16(s.val[1]))));
          local_col_sums.val[2] =
              vaddq_s32(local_col_sums.val[2],
                        vreinterpretq_s32_u32(vaddl_u16(
                            vget_low_u16(s.val[2]), vget_high_u16(s.val[2]))));
          local_col_sums.val[3] =
              vaddq_s32(local_col_sums.val[3],
                        vreinterpretq_s32_u32(vaddl_u16(
                            vget_low_u16(s.val[3]), vget_high_u16(s.val[3]))));
          vst4q_u8(dst_ptr, src_regs);
          src_line0ptr += 16;
          src_line1ptr += 16;
          src_line2ptr += 16;
          src_line3ptr += 16;
          dst_ptr += SideFormat::kWidth * 16;
        }
        break;
      case 3:
        while (dst_ptr != dst_end_ptr) {
          uint8x16x4_t src_regs;
          src_regs.val[0] = vld1q_u8(src_line0ptr);
          src_regs.val[1] = vld1q_u8(src_line1ptr);
          src_regs.val[2] = vld1q_u8(src_line2ptr);
          src_regs.val[3] = vdupq_n_u8(0);
          uint16x8x3_t s;
          s.val[0] = vaddl_u8(vget_low_u8(src_regs.val[0]),
                              vget_high_u8(src_regs.val[0]));
          s.val[1] = vaddl_u8(vget_low_u8(src_regs.val[1]),
                              vget_high_u8(src_regs.val[1]));
          s.val[2] = vaddl_u8(vget_low_u8(src_regs.val[2]),
                              vget_high_u8(src_regs.val[2]));
          local_col_sums.val[0] =
              vaddq_s32(local_col_sums.val[0],
                        vreinterpretq_s32_u32(vaddl_u16(
                            vget_low_u16(s.val[0]), vget_high_u16(s.val[0]))));
          local_col_sums.val[1] =
              vaddq_s32(local_col_sums.val[1],
                        vreinterpretq_s32_u32(vaddl_u16(
                            vget_low_u16(s.val[1]), vget_high_u16(s.val[1]))));
          local_col_sums.val[2] =
              vaddq_s32(local_col_sums.val[2],
                        vreinterpretq_s32_u32(vaddl_u16(
                            vget_low_u16(s.val[2]), vget_high_u16(s.val[2]))));
          vst4q_u8(dst_ptr, src_regs);
          src_line0ptr += 16;
          src_line1ptr += 16;
          src_line2ptr += 16;
          dst_ptr += SideFormat::kWidth * 16;
        }
        break;
      case 2:
        while (dst_ptr != dst_end_ptr) {
          uint8x16x4_t src_regs;
          src_regs.val[0] = vld1q_u8(src_line0ptr);
          src_regs.val[1] = vld1q_u8(src_line1ptr);
          src_regs.val[2] = vdupq_n_u8(0);
          src_regs.val[3] = vdupq_n_u8(0);
          uint16x8x2_t s;
          s.val[0] = vaddl_u8(vget_low_u8(src_regs.val[0]),
                              vget_high_u8(src_regs.val[0]));
          s.val[1] = vaddl_u8(vget_low_u8(src_regs.val[1]),
                              vget_high_u8(src_regs.val[1]));
          local_col_sums.val[0] =
              vaddq_s32(local_col_sums.val[0],
                        vreinterpretq_s32_u32(vaddl_u16(
                            vget_low_u16(s.val[0]), vget_high_u16(s.val[0]))));
          local_col_sums.val[1] =
              vaddq_s32(local_col_sums.val[1],
                        vreinterpretq_s32_u32(vaddl_u16(
                            vget_low_u16(s.val[1]), vget_high_u16(s.val[1]))));
          vst4q_u8(dst_ptr, src_regs);
          src_line0ptr += 16;
          src_line1ptr += 16;
          dst_ptr += SideFormat::kWidth * 16;
        }
        break;
      case 1:
        while (dst_ptr != dst_end_ptr) {
          uint8x16x4_t src_regs;
          src_regs.val[0] = vld1q_u8(src_line0ptr);
          src_regs.val[1] = vdupq_n_u8(0);
          src_regs.val[2] = vdupq_n_u8(0);
          src_regs.val[3] = vdupq_n_u8(0);
          uint16x8x2_t s;
          s.val[0] = vaddl_u8(vget_low_u8(src_regs.val[0]),
                              vget_high_u8(src_regs.val[0]));
          local_col_sums.val[0] =
              vaddq_s32(local_col_sums.val[0],
                        vreinterpretq_s32_u32(vaddl_u16(
                            vget_low_u16(s.val[0]), vget_high_u16(s.val[0]))));
          vst4q_u8(dst_ptr, src_regs);
          src_line0ptr += 16;
          dst_ptr += SideFormat::kWidth * 16;
        }
        break;
      default:
        abort();
    }

    int extra = depth - AlignedDepth16;

    if (extra) {
      __attribute__((aligned(16))) std::uint8_t line0extra[16];
      __attribute__((aligned(16))) std::uint8_t line1extra[16];
      __attribute__((aligned(16))) std::uint8_t line2extra[16];
      __attribute__((aligned(16))) std::uint8_t line3extra[16];
      __attribute__((aligned(16))) std::uint8_t dstextra[64];

      for (int i = 0; i < extra; i++) {
        line0extra[i] = width >= 1 ? src_line0ptr[i] : 0;
        line1extra[i] = width >= 2 ? src_line1ptr[i] : 0;
        line2extra[i] = width >= 3 ? src_line2ptr[i] : 0;
        line3extra[i] = width >= 4 ? src_line3ptr[i] : 0;
      }
      for (int i = extra; i < 16; i++) {
        line0extra[i] = 0;
        line1extra[i] = 0;
        line2extra[i] = 0;
        line3extra[i] = 0;
      }
      src_line0ptr = line0extra;
      src_line1ptr = line1extra;
      src_line2ptr = line2extra;
      src_line3ptr = line3extra;

      {
        uint8x16x4_t src_regs;
        src_regs.val[0] = vld1q_u8(src_line0ptr);
        src_regs.val[1] = vld1q_u8(src_line1ptr);
        src_regs.val[2] = vld1q_u8(src_line2ptr);
        src_regs.val[3] = vld1q_u8(src_line3ptr);
        uint16x8x4_t s;
        s.val[0] = vaddl_u8(vget_low_u8(src_regs.val[0]),
                            vget_high_u8(src_regs.val[0]));
        s.val[1] = vaddl_u8(vget_low_u8(src_regs.val[1]),
                            vget_high_u8(src_regs.val[1]));
        s.val[2] = vaddl_u8(vget_low_u8(src_regs.val[2]),
                            vget_high_u8(src_regs.val[2]));
        s.val[3] = vaddl_u8(vget_low_u8(src_regs.val[3]),
                            vget_high_u8(src_regs.val[3]));
        local_col_sums.val[0] =
            vaddq_s32(local_col_sums.val[0],
                      vreinterpretq_s32_u32(vaddl_u16(
                          vget_low_u16(s.val[0]), vget_high_u16(s.val[0]))));
        local_col_sums.val[1] =
            vaddq_s32(local_col_sums.val[1],
                      vreinterpretq_s32_u32(vaddl_u16(
                          vget_low_u16(s.val[1]), vget_high_u16(s.val[1]))));
        local_col_sums.val[2] =
            vaddq_s32(local_col_sums.val[2],
                      vreinterpretq_s32_u32(vaddl_u16(
                          vget_low_u16(s.val[2]), vget_high_u16(s.val[2]))));
        local_col_sums.val[3] =
            vaddq_s32(local_col_sums.val[3],
                      vreinterpretq_s32_u32(vaddl_u16(
                          vget_low_u16(s.val[3]), vget_high_u16(s.val[3]))));
        vst4q_u8(dstextra, src_regs);
      }

      for (int i = 0; i < 4 * extra; i++) {
        dst_ptr[i] = dstextra[i];
      }
    }

    // Accumulate the final rank_one_update vector.
    std::int32_t r[4];
    if (width >= 1) {
      vst1q_s32(r, local_col_sums.val[0]);
      rank_one_update_ptr[0] +=
          Base::packed_side_block()->rank_one_update_multiplier() *
          (r[0] + r[1] + r[2] + r[3]);
    }
    if (width >= 2) {
      vst1q_s32(r, local_col_sums.val[1]);
      rank_one_update_ptr[1] +=
          Base::packed_side_block()->rank_one_update_multiplier() *
          (r[0] + r[1] + r[2] + r[3]);
    }
    if (width >= 3) {
      vst1q_s32(r, local_col_sums.val[2]);
      rank_one_update_ptr[2] +=
          Base::packed_side_block()->rank_one_update_multiplier() *
          (r[0] + r[1] + r[2] + r[3]);
    }
    if (width >= 4) {
      vst1q_s32(r, local_col_sums.val[3]);
      rank_one_update_ptr[3] +=
          Base::packed_side_block()->rank_one_update_multiplier() *
          (r[0] + r[1] + r[2] + r[3]);
    }
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_PACK_NEON_H_
