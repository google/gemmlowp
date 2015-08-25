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

// bit_depth.h: defines the BitDepthSetting enum

#ifndef GEMMLOWP_PUBLIC_BIT_DEPTH_H_
#define GEMMLOWP_PUBLIC_BIT_DEPTH_H_

namespace gemmlowp {

// The BitDepthSetting enum lists supported Lhs/Rhs bit-depth combinations.
enum class BitDepthSetting {
  L8R8,  // 8-bit Lhs, 8-bit Rhs
  L7R5   // 7-bit Lhs, 5-bit Rhs
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_BIT_DEPTH_H_
