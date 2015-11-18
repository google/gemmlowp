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
//
// single_thread_gemm.h: programatically generated GEMM library header.

#ifndef GEMMLOWP_META_SINGLE_THREAD_GEMM_H_
#define GEMMLOWP_META_SINGLE_THREAD_GEMM_H_

#ifdef GEMMLOWP_NEON_32

#include <cassert>

namespace gemmlowp {
namespace meta {
namespace internal {

void zip_1x8_aligned(const std::uint8_t* source, std::int32_t count,
                     std::int32_t stride, std::uint8_t* destination,
                     std::int32_t multiplicative_offset,
                     std::int32_t additive_offset) {
  asm volatile(
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_1_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #1\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_2_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #2\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.16 {d0[0]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_3_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #3\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[2]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_4_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #4\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_5_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #5\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[4]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_6_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #6\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_7_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #7\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.8 {d0[6]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_2x8_aligned(const std::uint8_t* source, std::int32_t count,
                     std::int32_t stride, std::uint8_t* destination,
                     std::int32_t multiplicative_offset,
                     std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_1_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #1\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.8 {d0[0]}, [%[source]]\n"
      "vld1.8 {d1[0]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_2_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #2\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[source]]\n"
      "vld1.16 {d1[0]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_3_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #3\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d0[2]}, [%[source]]\n"
      "vld1.8 {d1[2]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_4_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #4\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]\n"
      "vld1.32 {d1[0]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_5_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #5\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d0[4]}, [%[source]]\n"
      "vld1.8 {d1[4]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_6_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #6\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d0[2]}, [%[source]]\n"
      "vld1.16 {d1[2]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_7_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #7\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d0[6]}, [%[source]]\n"
      "vld1.8 {d1[6]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_3x8_aligned(const std::uint8_t* source, std::int32_t count,
                     std::int32_t stride, std::uint8_t* destination,
                     std::int32_t multiplicative_offset,
                     std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_1_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #1\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.8 {d0[0]}, [%[source]]\n"
      "vld1.8 {d1[0]}, [r0]\n"
      "vld1.8 {d2[0]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_2_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #2\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.16 {d0[0]}, [%[source]]\n"
      "vld1.16 {d1[0]}, [r0]\n"
      "vld1.16 {d2[0]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_3_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #3\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d0[2]}, [%[source]]\n"
      "vld1.8 {d1[2]}, [r0]\n"
      "vld1.8 {d2[2]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_4_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #4\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]\n"
      "vld1.32 {d1[0]}, [r0]\n"
      "vld1.32 {d2[0]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_5_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #5\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d0[4]}, [%[source]]\n"
      "vld1.8 {d1[4]}, [r0]\n"
      "vld1.8 {d2[4]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_6_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #6\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d0[2]}, [%[source]]\n"
      "vld1.16 {d1[2]}, [r0]\n"
      "vld1.16 {d2[2]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_7_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #7\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d0[6]}, [%[source]]\n"
      "vld1.8 {d1[6]}, [r0]\n"
      "vld1.8 {d2[6]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_1x8(const std::uint8_t* source, std::int32_t count,
             std::int32_t stride, std::uint8_t* destination,
             std::int32_t multiplicative_offset, std::int32_t additive_offset) {
  asm volatile(
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_1(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #1\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_2(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #2\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.16 {d0[0]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_3(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #3\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[2]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_4(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #4\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_5(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #5\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[4]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_6(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #6\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_1x8_7(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #7\n"
      "vmov.i16 q2, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.8 {d0[6]}, [%[source]]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 q1, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpadd.u32 d6, d4, d5\n"
      "vpadd.u32 d8, d6, d6\n"
      "vmul.i32 q4, q4, d1[0]\n"
      "vadd.i32 q4, q4, q1\n"
      "vst1.32 {d8[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "cc", "memory");
}

void zip_2x8(const std::uint8_t* source, std::int32_t count,
             std::int32_t stride, std::uint8_t* destination,
             std::int32_t multiplicative_offset, std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_1(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #1\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.8 {d0[0]}, [%[source]]\n"
      "vld1.8 {d1[0]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_2(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #2\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[source]]\n"
      "vld1.16 {d1[0]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_3(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #3\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d0[2]}, [%[source]]\n"
      "vld1.8 {d1[2]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_4(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #4\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]\n"
      "vld1.32 {d1[0]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_5(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #5\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d0[4]}, [%[source]]\n"
      "vld1.8 {d1[4]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_6(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #6\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d0[2]}, [%[source]]\n"
      "vld1.16 {d1[2]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_2x8_7(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #7\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d0[6]}, [%[source]]\n"
      "vld1.8 {d1[6]}, [r0]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 q4, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpadd.u32 d3, d4, d5\n"
      "vpadd.u32 d10, d6, d7\n"
      "vpadd.u32 d12, d3, d10\n"
      "vmul.i32 q6, q6, d2[0]\n"
      "vadd.i32 q6, q6, q4\n"
      "vst1.32 {d12}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d12", "d13", "cc", "memory");
}

void zip_3x8(const std::uint8_t* source, std::int32_t count,
             std::int32_t stride, std::uint8_t* destination,
             std::int32_t multiplicative_offset, std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_1(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #1\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.8 {d0[0]}, [%[source]]\n"
      "vld1.8 {d1[0]}, [r0]\n"
      "vld1.8 {d2[0]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_2(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #2\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.16 {d0[0]}, [%[source]]\n"
      "vld1.16 {d1[0]}, [r0]\n"
      "vld1.16 {d2[0]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_3(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #3\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d0[2]}, [%[source]]\n"
      "vld1.8 {d1[2]}, [r0]\n"
      "vld1.8 {d2[2]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_4(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #4\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]\n"
      "vld1.32 {d1[0]}, [r0]\n"
      "vld1.32 {d2[0]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_5(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #5\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d0[4]}, [%[source]]\n"
      "vld1.8 {d1[4]}, [r0]\n"
      "vld1.8 {d2[4]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_6(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #6\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d0[2]}, [%[source]]\n"
      "vld1.16 {d1[2]}, [r0]\n"
      "vld1.16 {d2[2]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

void zip_3x8_7(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #7\n"
      "vmov.i16 q2, #0\n"
      "vmov.i16 q3, #0\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d0[6]}, [%[source]]\n"
      "vld1.8 {d1[6]}, [r0]\n"
      "vld1.8 {d2[6]}, [r1]\n"
      "vaddw.u8 q2, q2, d0\n"
      "vaddw.u8 q3, q3, d1\n"
      "vaddw.u8 q4, q4, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q5, %[additive_offset]\n"
      "vpaddl.u16 q2, q2\n"
      "vpaddl.u16 q3, q3\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d12, d4, d5\n"
      "vpadd.u32 d13, d6, d7\n"
      "vpadd.u32 d14, d8, d9\n"
      "vpadd.u32 d16, d12, d13\n"
      "vpadd.u32 d17, d14, d14\n"
      "vmul.i32 q8, q8, d3[0]\n"
      "vadd.i32 q8, q8, q5\n"
      "vst1.32 {d16}, [%[destination]:64]!\n"
      "vst1.32 {d17[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d16", "d17", "cc", "memory");
}

inline void mul_1x8_1x8_rhsadd(const std::uint8_t* left,
                               const std::uint8_t* right, std::int32_t count,
                               std::int32_t* results,
                               std::int32_t results_stride) {
  asm volatile(
      "pld [%[left]]\n"
      "pld [%[right]]\n"
      // Clear aggregators.
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d0}, [%[left]:64]!\n"
      "vld1.8 {d1}, [%[right]:64]!\n"
      "pld [%[left], #64]\n"
      "pld [%[right], #64]\n"
      "vmull.u8 q2, d1, d0\n"
      "vpadal.u16 q1, q2\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[right]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d2, d2, d3\n"

      // Reduce rows.
      "vpadd.u32 d0, d2, d2\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d0, d0, d8\n"

      // Store reduced (rhs added) rows.
      "vst1.32 {d0[0]}, [%[results]], %[results_stride]\n"
      : [count] "+r"(count), [right] "+r"(right),
        [results_stride] "+r"(results_stride), [results] "+r"(results),
        [left] "+r"(left)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d8", "cc", "memory");
}

inline void mul_1x8_2x8_rhsadd(const std::uint8_t* left,
                               const std::uint8_t* right, std::int32_t count,
                               std::int32_t* results,
                               std::int32_t results_stride) {
  asm volatile(
      "pld [%[left]]\n"
      "pld [%[right]]\n"
      // Clear aggregators.
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d0}, [%[left]:64]!\n"
      "vld1.8 {d1, d2}, [%[right]:64]!\n"
      "pld [%[left], #64]\n"
      "pld [%[right], #64]\n"
      "vmull.u8 q4, d1, d0\n"
      "vmull.u8 q5, d2, d0\n"
      "vpadal.u16 q2, q4\n"
      "vpadal.u16 q3, q5\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[right]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"

      // Reduce rows.
      "vpadd.u32 d0, d4, d6\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d0, d0, d8\n"

      // Store reduced (rhs added) rows.
      "vst1.32 {d0}, [%[results]], %[results_stride]\n"
      : [count] "+r"(count), [right] "+r"(right),
        [results_stride] "+r"(results_stride), [results] "+r"(results),
        [left] "+r"(left)
      :
      : "d0", "d1", "d2", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
        "cc", "memory");
}

inline void mul_1x8_3x8_rhsadd(const std::uint8_t* left,
                               const std::uint8_t* right, std::int32_t count,
                               std::int32_t* results,
                               std::int32_t results_stride) {
  asm volatile(
      "pld [%[left]]\n"
      "pld [%[right]]\n"
      // Clear aggregators.
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, #0\n"
      "vmov.i32 q4, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d0}, [%[left]:64]!\n"
      "vld1.8 {d1, d2, d3}, [%[right]:64]!\n"
      "pld [%[left], #64]\n"
      "pld [%[right], #64]\n"
      "vmull.u8 q5, d1, d0\n"
      "vmull.u8 q6, d2, d0\n"
      "vmull.u8 q7, d3, d0\n"
      "vpadal.u16 q2, q5\n"
      "vpadal.u16 q3, q6\n"
      "vpadal.u16 q4, q7\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {q5}, [%[right]:64]\n"

      // Change stride because storing in two ops.
      "sub %[results_stride], %[results_stride], #8\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"

      // Reduce rows.
      "vpadd.u32 d0, d4, d6\n"
      "vpadd.u32 d1, d8, d8\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q5\n"

      // Store reduced (rhs added) rows.
      "vst1.32 {d0}, [%[results]]!\n"
      "vst1.32 {d1[0]}, [%[results]], %[results_stride]\n"

      : [count] "+r"(count), [right] "+r"(right),
        [results_stride] "+r"(results_stride), [results] "+r"(results),
        [left] "+r"(left)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

inline void mul_2x8_1x8_rhsadd(const std::uint8_t* left,
                               const std::uint8_t* right, std::int32_t count,
                               std::int32_t* results,
                               std::int32_t results_stride) {
  asm volatile(
      "pld [%[left]]\n"
      "pld [%[right]]\n"
      // Clear aggregators.
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d0, d1}, [%[left]:64]!\n"
      "vld1.8 {d2}, [%[right]:64]!\n"
      "pld [%[left], #64]\n"
      "pld [%[right], #64]\n"
      "vmull.u8 q4, d2, d0\n"
      "vmull.u8 q5, d2, d1\n"
      "vpadal.u16 q2, q4\n"
      "vpadal.u16 q3, q5\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[right]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"

      // Reduce rows.
      "vpadd.u32 d0, d4, d4\n"
      "vpadd.u32 d1, d6, d6\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d0, d0, d8\n"
      "vadd.s32 d1, d1, d8\n"

      // Store reduced (rhs added) rows.
      "vst1.32 {d0[0]}, [%[results]], %[results_stride]\n"
      "vst1.32 {d1[0]}, [%[results]], %[results_stride]\n"
      : [count] "+r"(count), [right] "+r"(right),
        [results_stride] "+r"(results_stride), [results] "+r"(results),
        [left] "+r"(left)
      :
      : "d0", "d1", "d2", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
        "cc", "memory");
}

inline void mul_2x8_2x8_rhsadd(const std::uint8_t* left,
                               const std::uint8_t* right, std::int32_t count,
                               std::int32_t* results,
                               std::int32_t results_stride) {
  asm volatile(
      "pld [%[left]]\n"
      "pld [%[right]]\n"
      // Clear aggregators.
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, #0\n"
      "vmov.i32 q4, #0\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d0, d1}, [%[left]:64]!\n"
      "vld1.8 {d2, d3}, [%[right]:64]!\n"
      "pld [%[left], #64]\n"
      "pld [%[right], #64]\n"
      "vmull.u8 q6, d2, d0\n"
      "vmull.u8 q7, d3, d0\n"
      "vmull.u8 q8, d2, d1\n"
      "vmull.u8 q9, d3, d1\n"
      "vpadal.u16 q2, q6\n"
      "vpadal.u16 q3, q7\n"
      "vpadal.u16 q4, q8\n"
      "vpadal.u16 q5, q9\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d12}, [%[right]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"

      // Reduce rows.
      "vpadd.u32 d0, d4, d6\n"
      "vpadd.u32 d1, d8, d10\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d0, d0, d12\n"
      "vadd.s32 d1, d1, d12\n"

      // Store reduced (rhs added) rows.
      "vst1.32 {d0}, [%[results]], %[results_stride]\n"
      "vst1.32 {d1}, [%[results]], %[results_stride]\n"
      : [count] "+r"(count), [right] "+r"(right),
        [results_stride] "+r"(results_stride), [results] "+r"(results),
        [left] "+r"(left)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

inline void mul_2x8_3x8_rhsadd(const std::uint8_t* left,
                               const std::uint8_t* right, std::int32_t count,
                               std::int32_t* results,
                               std::int32_t results_stride) {
  asm volatile(
      "pld [%[left]]\n"
      "pld [%[right]]\n"
      // Clear aggregators.
      "vmov.i32 q3, #0\n"
      "vmov.i32 q4, #0\n"
      "vmov.i32 q5, #0\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"
      "vmov.i32 q8, q5\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d0, d1}, [%[left]:64]!\n"
      "vld1.8 {d2, d3, d4}, [%[right]:64]!\n"
      "pld [%[left], #64]\n"
      "pld [%[right], #64]\n"
      "vmull.u8 q9, d2, d0\n"
      "vmull.u8 q10, d3, d0\n"
      "vmull.u8 q11, d4, d0\n"
      "vmull.u8 q12, d2, d1\n"
      "vmull.u8 q13, d3, d1\n"
      "vmull.u8 q14, d4, d1\n"
      "vpadal.u16 q3, q9\n"
      "vpadal.u16 q4, q10\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"
      "vpadal.u16 q7, q13\n"
      "vpadal.u16 q8, q14\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {q9}, [%[right]:64]\n"

      // Change stride because storing in two ops.
      "sub %[results_stride], %[results_stride], #8\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d16, d17\n"

      // Reduce rows.
      "vpadd.u32 d0, d6, d8\n"
      "vpadd.u32 d1, d10, d10\n"
      "vpadd.u32 d2, d12, d14\n"
      "vpadd.u32 d3, d16, d16\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q1, q1, q9\n"

      // Store reduced (rhs added) rows.
      "vst1.32 {d0}, [%[results]]!\n"
      "vst1.32 {d1[0]}, [%[results]], %[results_stride]\n"

      "vst1.32 {d2}, [%[results]]!\n"
      "vst1.32 {d3[0]}, [%[results]], %[results_stride]\n"

      : [count] "+r"(count), [right] "+r"(right),
        [results_stride] "+r"(results_stride), [results] "+r"(results),
        [left] "+r"(left)
      :
      : "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9", "d10", "d11",
        "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

inline void mul_3x8_1x8_rhsadd(const std::uint8_t* left,
                               const std::uint8_t* right, std::int32_t count,
                               std::int32_t* results,
                               std::int32_t results_stride) {
  asm volatile(
      "pld [%[left]]\n"
      "pld [%[right]]\n"
      // Clear aggregators.
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, #0\n"
      "vmov.i32 q4, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d0, d1, d2}, [%[left]:64]!\n"
      "vld1.8 {d3}, [%[right]:64]!\n"
      "pld [%[left], #64]\n"
      "pld [%[right], #64]\n"
      "vmull.u8 q5, d3, d0\n"
      "vmull.u8 q6, d3, d1\n"
      "vmull.u8 q7, d3, d2\n"
      "vpadal.u16 q2, q5\n"
      "vpadal.u16 q3, q6\n"
      "vpadal.u16 q4, q7\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d10}, [%[right]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"

      // Reduce rows.
      "vpadd.u32 d0, d4, d4\n"
      "vpadd.u32 d1, d6, d6\n"
      "vpadd.u32 d2, d8, d8\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d0, d0, d10\n"
      "vadd.s32 d1, d1, d10\n"
      "vadd.s32 d2, d2, d10\n"

      // Store reduced (rhs added) rows.
      "vst1.32 {d0[0]}, [%[results]], %[results_stride]\n"
      "vst1.32 {d1[0]}, [%[results]], %[results_stride]\n"
      "vst1.32 {d2[0]}, [%[results]], %[results_stride]\n"
      : [count] "+r"(count), [right] "+r"(right),
        [results_stride] "+r"(results_stride), [results] "+r"(results),
        [left] "+r"(left)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

inline void mul_3x8_2x8_rhsadd(const std::uint8_t* left,
                               const std::uint8_t* right, std::int32_t count,
                               std::int32_t* results,
                               std::int32_t results_stride) {
  asm volatile(
      "pld [%[left]]\n"
      "pld [%[right]]\n"
      // Clear aggregators.
      "vmov.i32 q3, #0\n"
      "vmov.i32 q4, #0\n"
      "vmov.i32 q5, #0\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"
      "vmov.i32 q8, q5\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d0, d1, d2}, [%[left]:64]!\n"
      "vld1.8 {d3, d4}, [%[right]:64]!\n"
      "pld [%[left], #64]\n"
      "pld [%[right], #64]\n"
      "vmull.u8 q9, d3, d0\n"
      "vmull.u8 q10, d4, d0\n"
      "vmull.u8 q11, d3, d1\n"
      "vmull.u8 q12, d4, d1\n"
      "vmull.u8 q13, d3, d2\n"
      "vmull.u8 q14, d4, d2\n"
      "vpadal.u16 q3, q9\n"
      "vpadal.u16 q4, q10\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"
      "vpadal.u16 q7, q13\n"
      "vpadal.u16 q8, q14\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d18}, [%[right]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d16, d17\n"

      // Reduce rows.
      "vpadd.u32 d0, d6, d8\n"
      "vpadd.u32 d1, d10, d12\n"
      "vpadd.u32 d2, d14, d16\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d0, d0, d18\n"
      "vadd.s32 d1, d1, d18\n"
      "vadd.s32 d2, d2, d18\n"

      // Store reduced (rhs added) rows.
      "vst1.32 {d0}, [%[results]], %[results_stride]\n"
      "vst1.32 {d1}, [%[results]], %[results_stride]\n"
      "vst1.32 {d2}, [%[results]], %[results_stride]\n"
      : [count] "+r"(count), [right] "+r"(right),
        [results_stride] "+r"(results_stride), [results] "+r"(results),
        [left] "+r"(left)
      :
      : "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9", "d10", "d11",
        "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

inline void mul_3x8_3x8_rhsadd(const std::uint8_t* left,
                               const std::uint8_t* right, std::int32_t count,
                               std::int32_t* results,
                               std::int32_t results_stride) {
  asm volatile(
      "pld [%[left]]\n"
      "pld [%[right]]\n"
      // Clear aggregators.
      "vmov.i32 q3, #0\n"
      "vmov.i32 q4, #0\n"
      "vmov.i32 q5, #0\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"
      "vmov.i32 q8, q5\n"
      "vmov.i32 q9, q6\n"
      "vmov.i32 q10, q7\n"
      "vmov.i32 q11, q8\n"

      // 3x3 lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d0, d1, d2}, [%[left]:64]!\n"
      "vld1.8 {d3, d4, d5}, [%[right]:64]!\n"
      "pld [%[left], #64]\n"
      "pld [%[right], #64]\n"
      "vmull.u8 q12, d0, d3\n"
      "vmull.u8 q13, d0, d4\n"
      "vmull.u8 q14, d0, d5\n"
      "vmull.u8 q15, d1, d3\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"
      "vpadal.u16 q6, q15\n"
      "vmull.u8 q12, d1, d4\n"
      "vmull.u8 q13, d1, d5\n"
      "vmull.u8 q14, d2, d3\n"
      "vmull.u8 q15, d2, d4\n"
      "vmull.u8 q0, d2, d5\n"
      "vpadal.u16 q7, q12\n"
      "vpadal.u16 q8, q13\n"
      "vpadal.u16 q9, q14\n"
      "vpadal.u16 q10, q15\n"
      "vpadal.u16 q11, q0\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {q12}, [%[right]:64]\n"

      // Change stride because storing in two ops.
      "sub %[results_stride], %[results_stride], #8\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d18, d18, d19\n"
      "vpadd.u32 d20, d20, d21\n"
      "vpadd.u32 d22, d22, d23\n"

      // Reduce rows.
      "vpadd.u32 d0, d6, d8\n"
      "vpadd.u32 d1, d10, d10\n"
      "vpadd.u32 d2, d12, d14\n"
      "vpadd.u32 d3, d16, d16\n"
      "vpadd.u32 d4, d18, d20\n"
      "vpadd.u32 d5, d22, d22\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q12\n"
      "vadd.s32 q1, q1, q12\n"
      "vadd.s32 q2, q2, q12\n"

      // Store reduced (rhs added) rows.
      "vst1.32 {d0}, [%[results]]!\n"
      "vst1.32 {d1[0]}, [%[results]], %[results_stride]\n"

      "vst1.32 {d2}, [%[results]]!\n"
      "vst1.32 {d3[0]}, [%[results]], %[results_stride]\n"

      "vst1.32 {d4}, [%[results]]!\n"
      "vst1.32 {d5[0]}, [%[results]], %[results_stride]\n"

      : [count] "+r"(count), [right] "+r"(right),
        [results_stride] "+r"(results_stride), [results] "+r"(results),
        [left] "+r"(left)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31", "cc", "memory");
}

void qnt_1x8_aligned(const std::int32_t* source, std::int32_t count,
                     std::int32_t stride, const std::int32_t* offsets,
                     std::uint8_t* destination, std::int32_t destination_stride,
                     std::int32_t multiplicative_offset,
                     std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]:64]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_1_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8[0]}, [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_2_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8}, [%[source]:64]\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.16 {d12[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_3_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8}, [%[source]:64]!\n"
      "vld1.32 {d9[0]}, [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.16 {d12[0]}, [%[destination]]!\n"
      "vst1.8 {d12[2]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_4_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9}, [%[source]:64]\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.32 {d12[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_5_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9}, [%[source]:64]!\n"
      "vld1.32 {d10[0]}, [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.8 {d12[4]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_6_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9, d10}, [%[source]:64]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.16 {d12[2]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_7_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9, d10}, [%[source]:64]!\n"
      "vld1.32 {d11[0]}, [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.16 {d12[2]}, [%[destination]]!\n"
      "vst1.8 {d12[6]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_2x8_aligned(const std::int32_t* source, std::int32_t count,
                     std::int32_t stride, const std::int32_t* offsets,
                     std::uint8_t* destination, std::int32_t destination_stride,
                     std::int32_t multiplicative_offset,
                     std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]:64]!\n"
      "vst1.8 {d20}, [r1:64]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_1_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]:64]!\n"
      "vst1.8 {d20}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10[0]}, [%[source]]\n"
      "vld1.32 {d14[0]}, [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18[0]}, [%[destination]]\n"
      "vst1.8 {d20[0]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_2_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]:64]!\n"
      "vst1.8 {d20}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10}, [%[source]:64]\n"
      "vld1.32 {d14}, [r0:64]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.16 {d18[0]}, [%[destination]]\n"
      "vst1.16 {d20[0]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_3_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]:64]!\n"
      "vst1.8 {d20}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10}, [%[source]:64]!\n"
      "vld1.32 {d14}, [r0:64]!\n"
      "vld1.32 {d11[0]}, [%[source]]\n"
      "vld1.32 {d15[0]}, [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.16 {d18[0]}, [%[destination]]!\n"
      "vst1.16 {d20[0]}, [r1]!\n"
      "vst1.8 {d18[2]}, [%[destination]]\n"
      "vst1.8 {d20[2]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_4_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]:64]!\n"
      "vst1.8 {d20}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11}, [%[source]:64]\n"
      "vld1.32 {d14, d15}, [r0:64]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d18[0]}, [%[destination]]\n"
      "vst1.32 {d20[0]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_5_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]:64]!\n"
      "vst1.8 {d20}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11}, [%[source]:64]!\n"
      "vld1.32 {d14, d15}, [r0:64]!\n"
      "vld1.32 {d12[0]}, [%[source]]\n"
      "vld1.32 {d16[0]}, [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d18[0]}, [%[destination]]!\n"
      "vst1.32 {d20[0]}, [r1]!\n"
      "vst1.8 {d18[4]}, [%[destination]]\n"
      "vst1.8 {d20[4]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_6_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]:64]!\n"
      "vst1.8 {d20}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11, d12}, [%[source]:64]\n"
      "vld1.32 {d14, d15, d16}, [r0:64]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d18[0]}, [%[destination]]!\n"
      "vst1.32 {d20[0]}, [r1]!\n"
      "vst1.16 {d18[2]}, [%[destination]]\n"
      "vst1.16 {d20[2]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_7_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]:64]!\n"
      "vst1.8 {d20}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11, d12}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16}, [r0:64]!\n"
      "vld1.32 {d13[0]}, [%[source]]\n"
      "vld1.32 {d17[0]}, [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d18[0]}, [%[destination]]!\n"
      "vst1.32 {d20[0]}, [r1]!\n"
      "vst1.16 {d18[2]}, [%[destination]]!\n"
      "vst1.16 {d20[2]}, [r1]!\n"
      "vst1.8 {d18[6]}, [%[destination]]!\n"
      "vst1.8 {d20[6]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_3x8_aligned(const std::int32_t* source, std::int32_t count,
                     std::int32_t stride, const std::int32_t* offsets,
                     std::uint8_t* destination, std::int32_t destination_stride,
                     std::int32_t multiplicative_offset,
                     std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]:64]!\n"
      "vst1.8 {d26}, [r1:64]!\n"
      "vst1.8 {d28}, [r3:64]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_1_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]:64]!\n"
      "vst1.8 {d26}, [r1:64]!\n"
      "vst1.8 {d28}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12[0]}, [%[source]]\n"
      "vld1.32 {d16[0]}, [r0]\n"
      "vld1.32 {d20[0]}, [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24[0]}, [%[destination]]\n"
      "vst1.8 {d26[0]}, [r1]\n"
      "vst1.8 {d28[0]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_2_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]:64]!\n"
      "vst1.8 {d26}, [r1:64]!\n"
      "vst1.8 {d28}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12}, [%[source]:64]\n"
      "vld1.32 {d16}, [r0:64]\n"
      "vld1.32 {d20}, [r2:64]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.16 {d24[0]}, [%[destination]]\n"
      "vst1.16 {d26[0]}, [r1]\n"
      "vst1.16 {d28[0]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_3_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]:64]!\n"
      "vst1.8 {d26}, [r1:64]!\n"
      "vst1.8 {d28}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12}, [%[source]:64]!\n"
      "vld1.32 {d16}, [r0:64]!\n"
      "vld1.32 {d20}, [r2:64]!\n"
      "vld1.32 {d13[0]}, [%[source]]\n"
      "vld1.32 {d17[0]}, [r0]\n"
      "vld1.32 {d21[0]}, [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.16 {d24[0]}, [%[destination]]!\n"
      "vst1.16 {d26[0]}, [r1]!\n"
      "vst1.16 {d28[0]}, [r3]!\n"
      "vst1.8 {d24[2]}, [%[destination]]\n"
      "vst1.8 {d26[2]}, [r1]\n"
      "vst1.8 {d28[2]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_4_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]:64]!\n"
      "vst1.8 {d26}, [r1:64]!\n"
      "vst1.8 {d28}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13}, [%[source]:64]\n"
      "vld1.32 {d16, d17}, [r0:64]\n"
      "vld1.32 {d20, d21}, [r2:64]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.32 {d24[0]}, [%[destination]]\n"
      "vst1.32 {d26[0]}, [r1]\n"
      "vst1.32 {d28[0]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_5_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]:64]!\n"
      "vst1.8 {d26}, [r1:64]!\n"
      "vst1.8 {d28}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d16, d17}, [r0:64]!\n"
      "vld1.32 {d20, d21}, [r2:64]!\n"
      "vld1.32 {d14[0]}, [%[source]]\n"
      "vld1.32 {d18[0]}, [r0]\n"
      "vld1.32 {d22[0]}, [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.32 {d24[0]}, [%[destination]]!\n"
      "vst1.32 {d26[0]}, [r1]!\n"
      "vst1.32 {d28[0]}, [r3]!\n"
      "vst1.8 {d24[4]}, [%[destination]]\n"
      "vst1.8 {d26[4]}, [r1]\n"
      "vst1.8 {d28[4]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_6_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]:64]!\n"
      "vst1.8 {d26}, [r1:64]!\n"
      "vst1.8 {d28}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13, d14}, [%[source]:64]\n"
      "vld1.32 {d16, d17, d18}, [r0:64]\n"
      "vld1.32 {d20, d21, d22}, [r2:64]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.32 {d24[0]}, [%[destination]]!\n"
      "vst1.32 {d26[0]}, [r1]!\n"
      "vst1.32 {d28[0]}, [r3]!\n"
      "vst1.16 {d24[2]}, [%[destination]]\n"
      "vst1.16 {d26[2]}, [r1]\n"
      "vst1.16 {d28[2]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_7_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]:64]!\n"
      "vst1.8 {d26}, [r1:64]!\n"
      "vst1.8 {d28}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13, d14}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22}, [r2:64]!\n"
      "vld1.32 {d15[0]}, [%[source]]\n"
      "vld1.32 {d19[0]}, [r0]\n"
      "vld1.32 {d23[0]}, [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.32 {d24[0]}, [%[destination]]!\n"
      "vst1.32 {d26[0]}, [r1]!\n"
      "vst1.32 {d28[0]}, [r3]!\n"
      "vst1.16 {d24[2]}, [%[destination]]!\n"
      "vst1.16 {d26[2]}, [r1]!\n"
      "vst1.16 {d28[2]}, [r3]!\n"
      "vst1.8 {d24[6]}, [%[destination]]!\n"
      "vst1.8 {d26[6]}, [r1]!\n"
      "vst1.8 {d28[6]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_1x8(const std::int32_t* source, std::int32_t count,
             std::int32_t stride, const std::int32_t* offsets,
             std::uint8_t* destination, std::int32_t destination_stride,
             std::int32_t multiplicative_offset, std::int32_t rounding_offset,
             std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_1(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8[0]}, [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_2(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8}, [%[source]:64]\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.16 {d12[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_3(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8}, [%[source]:64]!\n"
      "vld1.32 {d9[0]}, [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.16 {d12[0]}, [%[destination]]!\n"
      "vst1.8 {d12[2]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_4(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9}, [%[source]:64]\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.32 {d12[0]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_5(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9}, [%[source]:64]!\n"
      "vld1.32 {d10[0]}, [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.8 {d12[4]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_6(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9, d10}, [%[source]:64]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.16 {d12[2]}, [%[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_1x8_7(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:64]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.8 {d12}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9, d10}, [%[source]:64]!\n"
      "vld1.32 {d11[0]}, [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d12, q4\n"
      "vqmovn.s32 d13, q5\n"
      "vqmovun.s16 d12, q6\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.16 {d12[2]}, [%[destination]]!\n"
      "vst1.8 {d12[6]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "cc", "memory");
}

void qnt_2x8(const std::int32_t* source, std::int32_t count,
             std::int32_t stride, const std::int32_t* offsets,
             std::uint8_t* destination, std::int32_t destination_stride,
             std::int32_t multiplicative_offset, std::int32_t rounding_offset,
             std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]]!\n"
      "vst1.8 {d20}, [r1]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_1(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]]!\n"
      "vst1.8 {d20}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10[0]}, [%[source]]\n"
      "vld1.32 {d14[0]}, [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18[0]}, [%[destination]]\n"
      "vst1.8 {d20[0]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_2(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]]!\n"
      "vst1.8 {d20}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10}, [%[source]:64]\n"
      "vld1.32 {d14}, [r0:64]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.16 {d18[0]}, [%[destination]]\n"
      "vst1.16 {d20[0]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_3(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]]!\n"
      "vst1.8 {d20}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10}, [%[source]:64]!\n"
      "vld1.32 {d14}, [r0:64]!\n"
      "vld1.32 {d11[0]}, [%[source]]\n"
      "vld1.32 {d15[0]}, [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.16 {d18[0]}, [%[destination]]!\n"
      "vst1.16 {d20[0]}, [r1]!\n"
      "vst1.8 {d18[2]}, [%[destination]]\n"
      "vst1.8 {d20[2]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_4(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]]!\n"
      "vst1.8 {d20}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11}, [%[source]:64]\n"
      "vld1.32 {d14, d15}, [r0:64]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d18[0]}, [%[destination]]\n"
      "vst1.32 {d20[0]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_5(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]]!\n"
      "vst1.8 {d20}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11}, [%[source]:64]!\n"
      "vld1.32 {d14, d15}, [r0:64]!\n"
      "vld1.32 {d12[0]}, [%[source]]\n"
      "vld1.32 {d16[0]}, [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d18[0]}, [%[destination]]!\n"
      "vst1.32 {d20[0]}, [r1]!\n"
      "vst1.8 {d18[4]}, [%[destination]]\n"
      "vst1.8 {d20[4]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_6(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]]!\n"
      "vst1.8 {d20}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11, d12}, [%[source]:64]\n"
      "vld1.32 {d14, d15, d16}, [r0:64]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d18[0]}, [%[destination]]!\n"
      "vst1.32 {d20[0]}, [r1]!\n"
      "vst1.16 {d18[2]}, [%[destination]]\n"
      "vst1.16 {d20[2]}, [r1]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_2x8_7(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d18}, [%[destination]]!\n"
      "vst1.8 {d20}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11, d12}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16}, [r0:64]!\n"
      "vld1.32 {d13[0]}, [%[source]]\n"
      "vld1.32 {d17[0]}, [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d18, q5\n"
      "vqmovn.s32 d19, q6\n"
      "vqmovn.s32 d20, q7\n"
      "vqmovn.s32 d21, q8\n"
      "vqmovun.s16 d18, q9\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d18[0]}, [%[destination]]!\n"
      "vst1.32 {d20[0]}, [r1]!\n"
      "vst1.16 {d18[2]}, [%[destination]]!\n"
      "vst1.16 {d20[2]}, [r1]!\n"
      "vst1.8 {d18[6]}, [%[destination]]!\n"
      "vst1.8 {d20[6]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
        "d20", "d21", "cc", "memory");
}

void qnt_3x8(const std::int32_t* source, std::int32_t count,
             std::int32_t stride, const std::int32_t* offsets,
             std::uint8_t* destination, std::int32_t destination_stride,
             std::int32_t multiplicative_offset, std::int32_t rounding_offset,
             std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]]!\n"
      "vst1.8 {d26}, [r1]!\n"
      "vst1.8 {d28}, [r3]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_1(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]]!\n"
      "vst1.8 {d26}, [r1]!\n"
      "vst1.8 {d28}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12[0]}, [%[source]]\n"
      "vld1.32 {d16[0]}, [r0]\n"
      "vld1.32 {d20[0]}, [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24[0]}, [%[destination]]\n"
      "vst1.8 {d26[0]}, [r1]\n"
      "vst1.8 {d28[0]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_2(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]]!\n"
      "vst1.8 {d26}, [r1]!\n"
      "vst1.8 {d28}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12}, [%[source]:64]\n"
      "vld1.32 {d16}, [r0:64]\n"
      "vld1.32 {d20}, [r2:64]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.16 {d24[0]}, [%[destination]]\n"
      "vst1.16 {d26[0]}, [r1]\n"
      "vst1.16 {d28[0]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_3(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]]!\n"
      "vst1.8 {d26}, [r1]!\n"
      "vst1.8 {d28}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12}, [%[source]:64]!\n"
      "vld1.32 {d16}, [r0:64]!\n"
      "vld1.32 {d20}, [r2:64]!\n"
      "vld1.32 {d13[0]}, [%[source]]\n"
      "vld1.32 {d17[0]}, [r0]\n"
      "vld1.32 {d21[0]}, [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.16 {d24[0]}, [%[destination]]!\n"
      "vst1.16 {d26[0]}, [r1]!\n"
      "vst1.16 {d28[0]}, [r3]!\n"
      "vst1.8 {d24[2]}, [%[destination]]\n"
      "vst1.8 {d26[2]}, [r1]\n"
      "vst1.8 {d28[2]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_4(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]]!\n"
      "vst1.8 {d26}, [r1]!\n"
      "vst1.8 {d28}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13}, [%[source]:64]\n"
      "vld1.32 {d16, d17}, [r0:64]\n"
      "vld1.32 {d20, d21}, [r2:64]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.32 {d24[0]}, [%[destination]]\n"
      "vst1.32 {d26[0]}, [r1]\n"
      "vst1.32 {d28[0]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_5(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]]!\n"
      "vst1.8 {d26}, [r1]!\n"
      "vst1.8 {d28}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d16, d17}, [r0:64]!\n"
      "vld1.32 {d20, d21}, [r2:64]!\n"
      "vld1.32 {d14[0]}, [%[source]]\n"
      "vld1.32 {d18[0]}, [r0]\n"
      "vld1.32 {d22[0]}, [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.32 {d24[0]}, [%[destination]]!\n"
      "vst1.32 {d26[0]}, [r1]!\n"
      "vst1.32 {d28[0]}, [r3]!\n"
      "vst1.8 {d24[4]}, [%[destination]]\n"
      "vst1.8 {d26[4]}, [r1]\n"
      "vst1.8 {d28[4]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_6(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]]!\n"
      "vst1.8 {d26}, [r1]!\n"
      "vst1.8 {d28}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13, d14}, [%[source]:64]\n"
      "vld1.32 {d16, d17, d18}, [r0:64]\n"
      "vld1.32 {d20, d21, d22}, [r2:64]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.32 {d24[0]}, [%[destination]]!\n"
      "vst1.32 {d26[0]}, [r1]!\n"
      "vst1.32 {d28[0]}, [r3]!\n"
      "vst1.16 {d24[2]}, [%[destination]]\n"
      "vst1.16 {d26[2]}, [r1]\n"
      "vst1.16 {d28[2]}, [r3]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void qnt_3x8_7(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:64]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.8 {d24}, [%[destination]]!\n"
      "vst1.8 {d26}, [r1]!\n"
      "vst1.8 {d28}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13, d14}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22}, [r2:64]!\n"
      "vld1.32 {d15[0]}, [%[source]]\n"
      "vld1.32 {d19[0]}, [r0]\n"
      "vld1.32 {d23[0]}, [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d24, q6\n"
      "vqmovn.s32 d25, q7\n"
      "vqmovn.s32 d26, q8\n"
      "vqmovn.s32 d27, q9\n"
      "vqmovn.s32 d28, q10\n"
      "vqmovn.s32 d29, q11\n"
      "vqmovun.s16 d24, q12\n"
      "vqmovun.s16 d26, q13\n"
      "vqmovun.s16 d28, q14\n"
      "vst1.32 {d24[0]}, [%[destination]]!\n"
      "vst1.32 {d26[0]}, [r1]!\n"
      "vst1.32 {d28[0]}, [r3]!\n"
      "vst1.16 {d24[2]}, [%[destination]]!\n"
      "vst1.16 {d26[2]}, [r1]!\n"
      "vst1.16 {d28[2]}, [r3]!\n"
      "vst1.8 {d24[6]}, [%[destination]]!\n"
      "vst1.8 {d26[6]}, [r1]!\n"
      "vst1.8 {d28[6]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "cc", "memory");
}

void multi_qnt_1x8_aligned(const std::int32_t* source, std::int32_t count,
                           std::int32_t stride, const std::int32_t* offsets,
                           std::uint8_t* destination,
                           std::int32_t destination_stride,
                           std::int32_t multiplicative_offset,
                           std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_1x8_aligned(source, count, stride, offsets, destination,
                      destination_stride, multiplicative_offset,
                      rounding_offset, shift);
      break;
    case 1:
      qnt_1x8_1_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 2:
      qnt_1x8_2_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 3:
      qnt_1x8_3_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 4:
      qnt_1x8_4_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 5:
      qnt_1x8_5_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 6:
      qnt_1x8_6_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 7:
      qnt_1x8_7_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
  }
}

void multi_qnt_2x8_aligned(const std::int32_t* source, std::int32_t count,
                           std::int32_t stride, const std::int32_t* offsets,
                           std::uint8_t* destination,
                           std::int32_t destination_stride,
                           std::int32_t multiplicative_offset,
                           std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_2x8_aligned(source, count, stride, offsets, destination,
                      destination_stride, multiplicative_offset,
                      rounding_offset, shift);
      break;
    case 1:
      qnt_2x8_1_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 2:
      qnt_2x8_2_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 3:
      qnt_2x8_3_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 4:
      qnt_2x8_4_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 5:
      qnt_2x8_5_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 6:
      qnt_2x8_6_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 7:
      qnt_2x8_7_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
  }
}

void multi_qnt_3x8_aligned(const std::int32_t* source, std::int32_t count,
                           std::int32_t stride, const std::int32_t* offsets,
                           std::uint8_t* destination,
                           std::int32_t destination_stride,
                           std::int32_t multiplicative_offset,
                           std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_3x8_aligned(source, count, stride, offsets, destination,
                      destination_stride, multiplicative_offset,
                      rounding_offset, shift);
      break;
    case 1:
      qnt_3x8_1_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 2:
      qnt_3x8_2_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 3:
      qnt_3x8_3_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 4:
      qnt_3x8_4_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 5:
      qnt_3x8_5_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 6:
      qnt_3x8_6_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 7:
      qnt_3x8_7_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
  }
}

void multi_qnt_1x8(const std::int32_t* source, std::int32_t count,
                   std::int32_t stride, const std::int32_t* offsets,
                   std::uint8_t* destination, std::int32_t destination_stride,
                   std::int32_t multiplicative_offset,
                   std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_1x8(source, count, stride, offsets, destination, destination_stride,
              multiplicative_offset, rounding_offset, shift);
      break;
    case 1:
      qnt_1x8_1(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 2:
      qnt_1x8_2(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 3:
      qnt_1x8_3(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 4:
      qnt_1x8_4(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 5:
      qnt_1x8_5(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 6:
      qnt_1x8_6(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 7:
      qnt_1x8_7(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
  }
}

void multi_qnt_2x8(const std::int32_t* source, std::int32_t count,
                   std::int32_t stride, const std::int32_t* offsets,
                   std::uint8_t* destination, std::int32_t destination_stride,
                   std::int32_t multiplicative_offset,
                   std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_2x8(source, count, stride, offsets, destination, destination_stride,
              multiplicative_offset, rounding_offset, shift);
      break;
    case 1:
      qnt_2x8_1(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 2:
      qnt_2x8_2(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 3:
      qnt_2x8_3(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 4:
      qnt_2x8_4(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 5:
      qnt_2x8_5(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 6:
      qnt_2x8_6(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 7:
      qnt_2x8_7(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
  }
}

void multi_qnt_3x8(const std::int32_t* source, std::int32_t count,
                   std::int32_t stride, const std::int32_t* offsets,
                   std::uint8_t* destination, std::int32_t destination_stride,
                   std::int32_t multiplicative_offset,
                   std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_3x8(source, count, stride, offsets, destination, destination_stride,
              multiplicative_offset, rounding_offset, shift);
      break;
    case 1:
      qnt_3x8_1(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 2:
      qnt_3x8_2(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 3:
      qnt_3x8_3(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 4:
      qnt_3x8_4(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 5:
      qnt_3x8_5(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 6:
      qnt_3x8_6(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 7:
      qnt_3x8_7(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
  }
}

void gemm_0_0_0_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_1_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_2_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_3_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_4_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_5_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_6_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_7_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_0_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_1_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_2_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_3_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_4_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_5_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_6_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_7_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_0_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_1_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_2_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_3_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_4_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_5_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_6_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_7_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_1_0_0_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_0_1_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_0_2_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_0_3_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_0_4_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_0_5_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_0_6_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_0_7_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_1_0_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_1_1_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_1_2_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_1_3_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_1_4_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_1_5_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_1_6_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_1_7_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_2_0_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_2_1_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_2_2_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_2_3_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_2_4_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_2_5_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_2_6_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_1_2_7_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_1_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_0_0_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_0_1_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_0_2_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_0_3_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_0_4_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_0_5_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_0_6_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_0_7_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_1_0_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_1_1_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_1_2_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_1_3_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_1_4_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_1_5_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_1_6_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_1_7_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_2_0_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_2_1_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_1_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_1_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_2_2_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_2_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_2_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_2_3_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_3_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_3_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_2_4_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_4_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_4_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_2_5_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_5_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_5_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_2_6_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_6_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_6_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_2_2_7_aligned(std::uint8_t* scratch, const std::uint8_t* lhs,
                        const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                        std::int32_t k, std::int32_t lhs_offset,
                        std::int32_t rhs_offset, std::int32_t result_offset,
                        std::int32_t multiplicative_offset, std::int32_t shift,
                        std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_7_aligned(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8_aligned(temp_result, m, temp_result_stride,
                          zipped_lhs_3_offsets, result_chunk, m,
                          multiplicative_offset, rounding_offset, -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_7_aligned(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8_aligned(temp_result, m, temp_result_stride,
                        zipped_lhs_2_offsets, result_chunk, m,
                        multiplicative_offset, rounding_offset, -shift);
}

void gemm_0_0_0(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_1(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_2(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_3(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_4(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_5(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_6(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_0_7(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_0(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_1(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_2(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_3(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_4(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_5(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_6(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_1_7(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_0(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_1(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_2(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_3(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_4(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_5(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_6(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_0_2_7(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }
}

void gemm_1_0_0(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_0_1(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_0_2(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_0_3(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_0_4(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_0_5(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_0_6(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_0_7(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_1_0(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_1_1(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_1_2(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_1_3(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_1_4(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_1_5(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_1_6(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_1_7(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_2_0(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_2_1(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_2_2(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_2_3(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_2_4(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_2_5(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_2_6(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_1_2_7(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_1_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 1);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_1x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_1x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_1x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_1x8(temp_result, m, temp_result_stride, zipped_lhs_1_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_0_0(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_0_1(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_0_2(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_0_3(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_0_4(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_0_5(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_0_6(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_0_7(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_1_0(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_1_1(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_1_2(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_1_3(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_1_4(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_1_5(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_1_6(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_1_7(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_1x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_1x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_2_0(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_2_1(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_1(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_1(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_2_2(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_2(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_2(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_2_3(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_3(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_3(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_2_4(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_4(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_4(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_2_5(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_5(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_5(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_2_6(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_6(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_6(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

void gemm_2_2_7(std::uint8_t* scratch, const std::uint8_t* lhs,
                const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
                std::int32_t k, std::int32_t lhs_offset,
                std::int32_t rhs_offset, std::int32_t result_offset,
                std::int32_t multiplicative_offset, std::int32_t shift,
                std::uint8_t* result) {
  const std::int32_t row_chunks = n / 3;
  const std::int32_t col_chunks = m / 3;
  const std::int32_t padded_k = ((k + 7) / 8) * 8;

  const std::int32_t chunk_size = k * 3;
  const std::int32_t zipped_chunk_size = (padded_k + 16) * 3;
  const std::int32_t zipped_rhs_size = (padded_k + 16) * m;
  const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8;
  const std::int32_t temp_result_size = 3 * temp_result_stride;
  const std::int32_t rounding_offset = (1 << (shift - 1));
  const std::int32_t result_chunk_size = m * 3;

  std::uint8_t* zipped_lhs = scratch;
  std::int32_t* zipped_lhs_3_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3);
  std::int32_t* zipped_lhs_2_offsets =
      reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 2);
  std::uint8_t* zipped_rhs = scratch + zipped_chunk_size;
  std::int32_t* temp_result = reinterpret_cast<std::int32_t*>(
      scratch + zipped_chunk_size + zipped_rhs_size);

  const std::uint8_t* lhs_chunk = lhs;
  const std::uint8_t* rhs_chunk = rhs;
  std::uint8_t* zipped_rhs_chunk = zipped_rhs;
  std::int32_t* temp_result_chunk = temp_result;
  std::uint8_t* result_chunk = result;

  const std::int32_t const_offset = lhs_offset * rhs_offset * k + result_offset;
  for (int i = 0; i < col_chunks; ++i) {
    zip_3x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);
    rhs_chunk += chunk_size;
    zipped_rhs_chunk += zipped_chunk_size;
  }
  zip_2x8_7(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0);

  for (int i = 0; i < row_chunks; ++i) {
    zip_3x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
    zipped_rhs_chunk = zipped_rhs;
    temp_result_chunk = temp_result;
    for (int j = 0; j < col_chunks; ++j) {
      mul_3x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                         temp_result_chunk, temp_result_stride);
      zipped_rhs_chunk += zipped_chunk_size;
      temp_result_chunk += 3;
    }
    mul_3x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    multi_qnt_3x8(temp_result, m, temp_result_stride, zipped_lhs_3_offsets,
                  result_chunk, m, multiplicative_offset, rounding_offset,
                  -shift);
    lhs_chunk += chunk_size;
    result_chunk += result_chunk_size;
  }

  zip_2x8_7(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset);
  zipped_rhs_chunk = zipped_rhs;
  temp_result_chunk = temp_result;
  for (int j = 0; j < col_chunks; ++j) {
    mul_2x8_3x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k,
                       temp_result_chunk, temp_result_stride);
    zipped_rhs_chunk += zipped_chunk_size;
    temp_result_chunk += 3;
  }
  mul_2x8_2x8_rhsadd(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk,
                     temp_result_stride);
  multi_qnt_2x8(temp_result, m, temp_result_stride, zipped_lhs_2_offsets,
                result_chunk, m, multiplicative_offset, rounding_offset,
                -shift);
}

}  // namespace internal

void gemm(std::uint8_t* scratch, const std::uint8_t* lhs,
          const std::uint8_t* rhs, std::int32_t n, std::int32_t m,
          std::int32_t k, std::int32_t lhs_offset, std::int32_t rhs_offset,
          std::int32_t result_offset, std::int32_t multiplicative_offset,
          std::int32_t shift, std::uint8_t* result) {
  const bool lhs_aligned = ((reinterpret_cast<std::uintptr_t>(lhs) % 8) == 0);
  const bool rhs_aligned = ((reinterpret_cast<std::uintptr_t>(rhs) % 8) == 0);
  const bool result_aligned =
      ((reinterpret_cast<std::uintptr_t>(result) % 8) == 0);
  const bool m_aligned = ((m % 8) == 0);
  const bool k_aligned = ((k % 8) == 0);
  const bool aligned =
      lhs_aligned && rhs_aligned && result_aligned && m_aligned && k_aligned;
  if (aligned) {
    switch (n % 3) {
      case 0:
        switch (m % 3) {
          case 0:
            switch (k % 8) {
              case 0:
                internal::gemm_0_0_0_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_0_0_1_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_0_0_2_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_0_0_3_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_0_0_4_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_0_0_5_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_0_0_6_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_0_0_7_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
            }
            break;
          case 1:
            switch (k % 8) {
              case 0:
                internal::gemm_0_1_0_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_0_1_1_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_0_1_2_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_0_1_3_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_0_1_4_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_0_1_5_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_0_1_6_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_0_1_7_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
            }
            break;
          case 2:
            switch (k % 8) {
              case 0:
                internal::gemm_0_2_0_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_0_2_1_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_0_2_2_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_0_2_3_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_0_2_4_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_0_2_5_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_0_2_6_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_0_2_7_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
            }
            break;
        }
        break;
      case 1:
        switch (m % 3) {
          case 0:
            switch (k % 8) {
              case 0:
                internal::gemm_1_0_0_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_1_0_1_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_1_0_2_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_1_0_3_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_1_0_4_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_1_0_5_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_1_0_6_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_1_0_7_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
            }
            break;
          case 1:
            switch (k % 8) {
              case 0:
                internal::gemm_1_1_0_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_1_1_1_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_1_1_2_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_1_1_3_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_1_1_4_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_1_1_5_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_1_1_6_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_1_1_7_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
            }
            break;
          case 2:
            switch (k % 8) {
              case 0:
                internal::gemm_1_2_0_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_1_2_1_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_1_2_2_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_1_2_3_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_1_2_4_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_1_2_5_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_1_2_6_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_1_2_7_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
            }
            break;
        }
        break;
      case 2:
        switch (m % 3) {
          case 0:
            switch (k % 8) {
              case 0:
                internal::gemm_2_0_0_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_2_0_1_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_2_0_2_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_2_0_3_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_2_0_4_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_2_0_5_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_2_0_6_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_2_0_7_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
            }
            break;
          case 1:
            switch (k % 8) {
              case 0:
                internal::gemm_2_1_0_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_2_1_1_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_2_1_2_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_2_1_3_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_2_1_4_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_2_1_5_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_2_1_6_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_2_1_7_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
            }
            break;
          case 2:
            switch (k % 8) {
              case 0:
                internal::gemm_2_2_0_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_2_2_1_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_2_2_2_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_2_2_3_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_2_2_4_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_2_2_5_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_2_2_6_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_2_2_7_aligned(
                    scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset,
                    result_offset, multiplicative_offset, shift, result);
                break;
            }
            break;
        }
        break;
    }
  } else {
    switch (n % 3) {
      case 0:
        switch (m % 3) {
          case 0:
            switch (k % 8) {
              case 0:
                internal::gemm_0_0_0(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_0_0_1(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_0_0_2(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_0_0_3(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_0_0_4(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_0_0_5(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_0_0_6(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_0_0_7(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
            }
            break;
          case 1:
            switch (k % 8) {
              case 0:
                internal::gemm_0_1_0(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_0_1_1(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_0_1_2(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_0_1_3(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_0_1_4(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_0_1_5(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_0_1_6(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_0_1_7(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
            }
            break;
          case 2:
            switch (k % 8) {
              case 0:
                internal::gemm_0_2_0(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_0_2_1(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_0_2_2(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_0_2_3(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_0_2_4(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_0_2_5(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_0_2_6(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_0_2_7(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
            }
            break;
        }
        break;
      case 1:
        switch (m % 3) {
          case 0:
            switch (k % 8) {
              case 0:
                internal::gemm_1_0_0(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_1_0_1(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_1_0_2(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_1_0_3(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_1_0_4(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_1_0_5(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_1_0_6(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_1_0_7(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
            }
            break;
          case 1:
            switch (k % 8) {
              case 0:
                internal::gemm_1_1_0(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_1_1_1(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_1_1_2(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_1_1_3(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_1_1_4(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_1_1_5(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_1_1_6(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_1_1_7(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
            }
            break;
          case 2:
            switch (k % 8) {
              case 0:
                internal::gemm_1_2_0(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_1_2_1(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_1_2_2(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_1_2_3(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_1_2_4(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_1_2_5(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_1_2_6(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_1_2_7(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
            }
            break;
        }
        break;
      case 2:
        switch (m % 3) {
          case 0:
            switch (k % 8) {
              case 0:
                internal::gemm_2_0_0(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_2_0_1(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_2_0_2(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_2_0_3(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_2_0_4(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_2_0_5(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_2_0_6(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_2_0_7(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
            }
            break;
          case 1:
            switch (k % 8) {
              case 0:
                internal::gemm_2_1_0(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_2_1_1(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_2_1_2(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_2_1_3(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_2_1_4(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_2_1_5(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_2_1_6(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_2_1_7(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
            }
            break;
          case 2:
            switch (k % 8) {
              case 0:
                internal::gemm_2_2_0(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 1:
                internal::gemm_2_2_1(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 2:
                internal::gemm_2_2_2(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 3:
                internal::gemm_2_2_3(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 4:
                internal::gemm_2_2_4(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 5:
                internal::gemm_2_2_5(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 6:
                internal::gemm_2_2_6(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
              case 7:
                internal::gemm_2_2_7(scratch, lhs, rhs, n, m, k, lhs_offset,
                                     rhs_offset, result_offset,
                                     multiplicative_offset, shift, result);
                break;
            }
            break;
        }
        break;
    }
  }
}

}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm fast-path requires GEMMLOWP_NEON_32!"
#endif

#endif  // GEMMLOWP_META_SINGLE_THREAD_GEMM_H_
