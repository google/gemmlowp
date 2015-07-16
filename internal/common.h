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
const int kDefaultL2CacheSize = 256 * 1024;

// The proportion of the cache that we intend to use for storing
// RHS blocks. This should be between 0 and 1, and typically closer to 1,
// as we typically want to use most of the L2 cache for storing a large
// RHS block.
const float kDefaultL2RhsFactor = 0.90f;

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
