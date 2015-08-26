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

namespace gemmlowp {

template <int tBits>
struct BitDepth
{
  static const int kBits = tBits;
};

template <BitDepthSetting tBitDepthSetting>
struct LhsBitDepth
{};

template <BitDepthSetting tBitDepthSetting>
struct RhsBitDepth
{};

template <>
struct LhsBitDepth<BitDepthSetting::L8R8>
  : BitDepth<8>
{};

template <>
struct RhsBitDepth<BitDepthSetting::L8R8>
  : BitDepth<8>
{};

template <>
struct LhsBitDepth<BitDepthSetting::L7R5>
  : BitDepth<7>
{};

template <>
struct RhsBitDepth<BitDepthSetting::L7R5>
  : BitDepth<5>
{};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_BIT_DEPTH_H_
