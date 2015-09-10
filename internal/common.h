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

// common.h: contains stuff that's used throughout gemmlowp
// and should always be available.

#ifndef GEMMLOWP_INTERNAL_COMMON_H_
#define GEMMLOWP_INTERNAL_COMMON_H_

#include <pthread.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include "../profiling/instrumentation.h"

#ifdef GEMMLOWP_PROFILING
#include <set>
#include <cstdio>
#include <cstring>
#endif

// Detect NEON. It's important to check for both tokens.
#if (defined __ARM_NEON) || (defined __ARM_NEON__)
#define GEMMLOWP_NEON
#ifdef __arm__
#define GEMMLOWP_NEON32
#endif
#ifdef __aarch64__
#define GEMMLOWP_NEON64
#endif
#endif

// Detect SSE.
#if defined __SSE4_2__  // at the moment, our SSE code assumes SSE 4.something
#define GEMMLOWP_SSE
#if defined(__i386__) || defined(_M_IX86) || defined(_X86_) || defined(__i386)
#define GEMMLOWP_SSE32
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64)
#define GEMMLOWP_SSE64
#endif
#endif

namespace gemmlowp {

// Standard cache line size. Useful to optimize alignment and
// prefetches. Ideally we would query this at runtime, however
// 64 byte cache lines are the vast majority, and even if it's
// wrong on some device, it will be wrong by no more than a 2x factor,
// which should be acceptable.
const int kDefaultCacheLineSize = 64;

// Default L1 and L2 data cache sizes. On x86, we should ideally query this at
// runtime. On ARM, the instruction to query this is privileged and
// Android kernels do not expose it to userspace. Fortunately, the majority
// of ARM devices have roughly comparable values:
//   Nexus 5: L1 16k, L2 1M
//   Android One: L1 32k, L2 512k
// The following values are equal to or somewhat lower than that, and were
// found to perform well on both the Nexus 5 and Android One.
// Of course, they would be too low for typical x86 CPUs where we would want
// to set the L2 value to (L3 cache size / number of cores) at least.
const int kDefaultL1CacheSize = 16 * 1024;
const int kDefaultL2CacheSize = 384 * 1024;

// The proportion of the cache that we intend to use for storing
// RHS blocks. This should be between 0 and 1, and typically closer to 1,
// as we typically want to use most of the L2 cache for storing a large
// RHS block.
// Note: with less-than-8-bit depth, requantization makes packing more
// expensive. We lowered this value from 0.9 to 0.75 with the introduction
// of expensive requantization; this results in much higher performance
// for 1000x1000 matrices; the exact reason for that is not understood.
// Anyway, clearly we will eventually need better heuristics than just
// those constant parameters here.
const float kDefaultL2RhsFactor = 0.75f;

// The number of bytes in a SIMD register. This is used to determine
// the dimensions of PackingRegisterBlock so that such blocks can
// be efficiently loaded into registers, so that packing code can
// work within registers as much as possible.
// In the non-SIMD generic fallback code, this is just a generic array
// size, so any size would work there. Different platforms may set this
// to different values but must ensure that their own optimized packing paths
// are consistent with this value.
const int kRegisterSize = 16;

// The threshold on the depth dimension at which we switch to
// probabilistic rounding instead of rounding-to-nearest when
// requantizing input data. Indeed, both statistical theory and
// empirical measurements show that for given input data and bit depth,
// probabilistic rounding gives more accurate results for large enough
// depth, while rounding-to-nearest does for smaller depth. This threshold
// is naively determined from some experiments with Inception at 7bit/5bit
// on a set of 10,000 images:
//
//   7 bit weights, 5 bit activations, switch at 64:   59.82% top-1 accuracy
//   7 bit weights, 5 bit activations, switch at 128:  59.58% top-1 accuracy
//   7 bit weights, 5 bit activations, switch at 192:  63.37% top-1 accuracy
//   7 bit weights, 5 bit activations, switch at 256:  63.47% top-1 accuracy
//   7 bit weights, 5 bit activations, switch at 320:  63.71% top-1 accuracy
//   7 bit weights, 5 bit activations, switch at 384:  63.71% top-1 accuracy
//   7 bit weights, 5 bit activations, switch at 448:  63.58% top-1 accuracy
//   7 bit weights, 5 bit activations, switch at 512:  64.10% top-1 accuracy
//   7 bit weights, 5 bit activations, switch at 640:  62.49% top-1 accuracy
//   7 bit weights, 5 bit activations, switch at 768:  62.49% top-1 accuracy
//   7 bit weights, 5 bit activations, switch at 1024: 58.96% top-1 accuracy
//
// So here, 384 looks comfortably in the middle of a plateau of good values,
// and it's a roundish number (3/2 * 256) so let's stick with that for now.
// It would be nice to work out the theory of this, and understand how this
// should depend on the distribution of inputs and the bit depth.
const int kProbabilisticRoundingThreshold = 384;

// Hints the CPU to prefetch the cache line containing ptr.
inline void Prefetch(const void* ptr) {
#ifdef __GNUC__  // Clang and GCC define __GNUC__ and have __builtin_prefetch.
  __builtin_prefetch(ptr);
#else
  (void)ptr;
#endif
}

// Returns the runtime argument rounded down to the nearest multiple of
// the fixed Modulus.
template <int Modulus>
int RoundDown(int i) {
  return i - (i % Modulus);
}

// Returns the runtime argument rounded up to the nearest multiple of
// the fixed Modulus.
template <int Modulus>
int RoundUp(int i) {
  return RoundDown<Modulus>(i + Modulus - 1);
}

// Returns the quotient a / b rounded up ('ceil') to the nearest integer.
template <typename Integer>
Integer CeilQuotient(Integer a, Integer b) {
  return (a + b - 1) / b;
}

// Returns the argument rounded up to the nearest power of two.
template <typename Integer>
Integer RoundUpToPowerOfTwo(Integer n) {
  Integer i = n - 1;
  i |= i >> 1;
  i |= i >> 2;
  i |= i >> 4;
  i |= i >> 8;
  i |= i >> 16;
  return i + 1;
}

template <int N>
struct IsPowerOfTwo {
  static const bool value = !(N & (N - 1));
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_COMMON_H_
