// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
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

// bit_depth.h: defines the settins controlling LHS/RHS bit depth

#ifndef GEMMLOWP_PUBLIC_BIT_DEPTH_H_
#define GEMMLOWP_PUBLIC_BIT_DEPTH_H_

namespace gemmlowp {

// A specific bit depth of an operand (Lhs or Rhs).
template <int tBits>
struct BitDepth {
  static const int kBits = tBits;
  static_assert(kBits >= 1 && kBits <= 8, "bad bit depth");
};

// Default: LHS and RHS are 8bit.
struct DefaultL8R8BitDepthParams {
  typedef BitDepth<8> LhsBitDepth;
  typedef BitDepth<8> RhsBitDepth;
};

// Deprecated: when gemmlowp used to allow requantizing 8bit
// inputs to less-than-8-bit depths, the public setting allowing
// that was DefaultL7R5BitDepthParams. That requantization
// feature has been removed, but as the whole point of that
// requantization was to make less-than-8-bit an internal
// optimization without any impact on the API (other than lowering
// accuracy), we can temporarily support users who were using it
// by mapping it to the default 8bit behavior.
using DefaultL7R5BitDepthParams = DefaultL8R8BitDepthParams;

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_BIT_DEPTH_H_
