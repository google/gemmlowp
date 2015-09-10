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

// bit_depth_util.h: helpers to handle BitDepthSetting's

#ifndef GEMMLOWP_INTERNAL_BIT_DEPTH_H_
#define GEMMLOWP_INTERNAL_BIT_DEPTH_H_

#include "../public/bit_depth.h"
#include "common.h"

namespace gemmlowp {

// A specific bit depth to requantize an operand (Lhs or Rhs) to.
// The case tBits==8 means no requantization, since at the moment
// we only accept 8-bit input data.
template <int tBits>
struct BitDepth {
  static const int kBits = tBits;
  static_assert(kBits >= 1 && kBits <= 8, "bad bit depth");
};

// A rounding mode to use when requantizing an operand.
// The requantizing operation is:
//   dst = (src * maxval + rounding_offset) / 255;
// Where dst and src are uint8, maxval is 2^(dstbits)-1,
// and the intermediate values are computed as uint16s
// so no overflow occurs.
// The rounding_offset in the above formula is what is
// determined by the RoundingMode, as follows:
enum class RoundingMode {
  Nearest,       // rounding_offset = 127
  Probabilistic  // rounding_offset = random in [0 ... 254].
};

// Chooses a rounding mode. See the comment on
// kProbabilisticRoundingThreshold. This heuristic is overly naive
// and could be improved with better understanding of the stats here.
template <typename BitDepth>
RoundingMode ChooseRoundingMode(int accumulation_depth) {
  if (BitDepth::kBits == 8 ||
      accumulation_depth < kProbabilisticRoundingThreshold) {
    return RoundingMode::Nearest;
  } else {
    return RoundingMode::Probabilistic;
  }
}

template <BitDepthSetting tBitDepthSetting>
struct LhsBitDepth {};

template <BitDepthSetting tBitDepthSetting>
struct RhsBitDepth {};

template <>
struct LhsBitDepth<BitDepthSetting::L8R8> : BitDepth<8> {};

template <>
struct RhsBitDepth<BitDepthSetting::L8R8> : BitDepth<8> {};

template <>
struct LhsBitDepth<BitDepthSetting::L7R5> : BitDepth<7> {};

template <>
struct RhsBitDepth<BitDepthSetting::L7R5> : BitDepth<5> {};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_BIT_DEPTH_H_
