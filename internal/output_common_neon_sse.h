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

// output_common_neon_sse.h: common SIMD (NEON and SSE) code for output.h

#ifndef GEMMLOWP_INTERNAL_OUTPUT_COMMON_NEON_SSE_H_
#define GEMMLOWP_INTERNAL_OUTPUT_COMMON_NEON_SSE_H_

#include "output.h"

namespace gemmlowp {



template <typename SrcScalarType, int N>
struct LoadImpl<RegBlockInt32<4,N>, MatrixMap<SrcScalarType, MapOrder::ColMajor>>
{
  static RegBlockInt32<4,N> Run(const MatrixMap<SrcScalarType, MapOrder::ColMajor>& src, int row, int col) {
    RegBlockInt32<4,N> result;
    for (int i = 0; i < N; i++) {
      result.buf.reg[i] = LoadInt32x4(src.data(row, col + i));
    }
    return result;
  }
};

template <typename SrcScalarType, int N>
struct LoadImpl<RegBlockInt32<8,N>, MatrixMap<SrcScalarType, MapOrder::ColMajor>>
{
  static RegBlockInt32<8,N> Run(const MatrixMap<SrcScalarType, MapOrder::ColMajor>& src, int row, int col) {
    RegBlockInt32<8,N> result;
    for (int i = 0; i < N; i++) {
      result.buf.reg[2 * i + 0] = LoadInt32x4(src.data(row + 0, col + i));
      result.buf.reg[2 * i + 1] = LoadInt32x4(src.data(row + 4, col + i));
    }
    return result;
  }
};

template <typename SrcScalarType>
struct LoadImpl<RegBlockInt32<1,4>, MatrixMap<SrcScalarType, MapOrder::ColMajor>>
{
  static RegBlockInt32<1,4> Run(const MatrixMap<SrcScalarType, MapOrder::ColMajor>& src, int row, int col) {
    RegBlockInt32<1,4> result;
    std::int32_t buf[4];
    for (int i = 0; i < 4; i++) {
      buf[i] = src(row, col + i);
    }
    result.buf.reg[0] = LoadInt32x4(buf);
    return result;
  }
};

template <typename SrcScalarType>
struct LoadImpl<RegBlockInt32<1,8>, MatrixMap<SrcScalarType, MapOrder::ColMajor>>
{
  static RegBlockInt32<1,8> Run(const MatrixMap<SrcScalarType, MapOrder::ColMajor>& src, int row, int col) {
    RegBlockInt32<1,8> result;
    std::int32_t buf[8];
    for (int i = 0; i < 8; i++) {
      buf[i] = src(row, col + i);
    }
    result.buf.reg[0] = LoadInt32x4(buf);
    result.buf.reg[1] = LoadInt32x4(buf + 4);
    return result;
  }
};

template <typename SrcScalarType>
struct LoadImpl<RegBlockInt32<4,1>, VectorMap<SrcScalarType, VectorShape::Col>>
{
  static RegBlockInt32<4,1> Run(const VectorMap<SrcScalarType, VectorShape::Col>& src, int pos) {
    RegBlockInt32<4,1> result;
    result.buf.reg[0] = LoadInt32x4(src.data(pos));
    return result;
  }
};

template <typename SrcScalarType>
struct LoadImpl<RegBlockInt32<4,1>, VectorDup<SrcScalarType, VectorShape::Col>>
{
  static RegBlockInt32<4,1> Run(const VectorDup<SrcScalarType, VectorShape::Col>& src, int) {
    RegBlockInt32<4,1> result;
    result.buf.reg[0] = LoadInt32x4(src(0));
    return result;
  }
};

template <typename SrcScalarType, int N>
struct LoadForBroadcastingImpl<RegBlockInt32<4,N>, VectorMap<SrcScalarType, VectorShape::Col>>
{
  using SrcObjectType = VectorMap<SrcScalarType, VectorShape::Col>;
  using RegisterBlockType = RegBlockInt32<4,N>;
  using ResultBlockType = typename LoadForBroadcastingRegisterBlock<RegisterBlockType, SrcObjectType>::Type;

  static ResultBlockType Run(const SrcObjectType& src, int pos) {
    ResultBlockType result;
    static_assert(ResultBlockType::kRegisterCount == 1, "");
    result.buf.reg[0] = LoadInt32x4(src.data(pos));
    return result;
  }
};

template <typename SrcScalarType, int N>
struct LoadForBroadcastingImpl<RegBlockInt32<8,N>, VectorMap<SrcScalarType, VectorShape::Col>>
{
  using SrcObjectType = VectorMap<SrcScalarType, VectorShape::Col>;
  using RegisterBlockType = RegBlockInt32<8,N>;
  using ResultBlockType = typename LoadForBroadcastingRegisterBlock<RegisterBlockType, SrcObjectType>::Type;

  static ResultBlockType Run(const SrcObjectType& src, int pos) {
    ResultBlockType result;
    static_assert(ResultBlockType::kRegisterCount == 2, "");
    result.buf.reg[0] = LoadInt32x4(src.data(pos));
    result.buf.reg[1] = LoadInt32x4(src.data(pos + 4));
    return result;
  }
};

template <typename SrcScalarType>
struct LoadForBroadcastingImpl<RegBlockInt32<4,1>, VectorMap<SrcScalarType, VectorShape::Row>>
{
  using SrcObjectType = VectorMap<SrcScalarType, VectorShape::Row>;
  using RegisterBlockType = RegBlockInt32<4,1>;
  using ResultBlockType = typename LoadForBroadcastingRegisterBlock<RegisterBlockType, SrcObjectType>::Type;

  static ResultBlockType Run(const SrcObjectType& src, int pos) {
    ResultBlockType result;
    result.buf.reg[0] = src(pos);
    return result;
  }
};

template <typename SrcScalarType, int N>
struct LoadForBroadcastingImpl<RegBlockInt32<N,4>, VectorMap<SrcScalarType, VectorShape::Row>>
{
  using SrcObjectType = VectorMap<SrcScalarType, VectorShape::Row>;
  using RegisterBlockType = RegBlockInt32<N,4>;
  using ResultBlockType = typename LoadForBroadcastingRegisterBlock<RegisterBlockType, SrcObjectType>::Type;

  static ResultBlockType Run(const SrcObjectType& src, int pos) {
    ResultBlockType result;
    static_assert(ResultBlockType::kRegisterCount == 1, "");
    result.buf.reg[0] = LoadInt32x4(src.data(pos));
    return result;
  }
};

template <typename SrcScalarType, int N>
struct LoadForBroadcastingImpl<RegBlockInt32<N,8>, VectorMap<SrcScalarType, VectorShape::Row>>
{
  using SrcObjectType = VectorMap<SrcScalarType, VectorShape::Row>;
  using RegisterBlockType = RegBlockInt32<N,8>;
  using ResultBlockType = typename LoadForBroadcastingRegisterBlock<RegisterBlockType, SrcObjectType>::Type;

  static ResultBlockType Run(const SrcObjectType& src, int pos) {
    ResultBlockType result;
    static_assert(ResultBlockType::kRegisterCount == 2, "");
    result.buf.reg[0] = LoadInt32x4(src.data(pos));
    result.buf.reg[1] = LoadInt32x4(src.data(pos + 4));
    return result;
  }
};

// 4x1 := 4x1 + 1x1
template <>
struct BroadcastAddImpl<RegBlockInt32<4,1>, RegBlockInt32<1,1>> {
  static RegBlockInt32<4,1> Run(const RegBlockInt32<4,1>& lhs, const RegBlockInt32<1,1>& rhs) {
    RegBlockInt32<4,1> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 1x4 := 1x4 + 1x1
template <>
struct BroadcastAddImpl<RegBlockInt32<1,4>, RegBlockInt32<1,1>> {
  static RegBlockInt32<1,4> Run(const RegBlockInt32<1,4>& lhs, const RegBlockInt32<1,1>& rhs) {
    RegBlockInt32<1,4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x1 := 4x1 + 4x1
template <>
struct BroadcastAddImpl<RegBlockInt32<4,1>, RegBlockInt32<4,1>> {
  static RegBlockInt32<4,1> Run(const RegBlockInt32<4,1>& lhs, const RegBlockInt32<4,1>& rhs) {
    RegBlockInt32<4,1> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 1x4 := 1x4 + 1x4
template <>
struct BroadcastAddImpl<RegBlockInt32<1,4>, RegBlockInt32<1,4>> {
  static RegBlockInt32<1,4> Run(const RegBlockInt32<1,4>& lhs, const RegBlockInt32<1,4>& rhs) {
    RegBlockInt32<1,4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 4x4 := 4x4 + 1x4
template <>
struct BroadcastAddImpl<RegBlockInt32<4,4>, RegBlockInt32<1,4>> {
  static RegBlockInt32<4,4> Run(const RegBlockInt32<4,4>& lhs, const RegBlockInt32<1,4>& rhs) {
    RegBlockInt32<4,4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[1] = Add(lhs.buf.reg[1], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[2] = Add(lhs.buf.reg[2], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[3] = Add(lhs.buf.reg[3], DupLane<3>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x4 := 4x4 + 4x1
template <>
struct BroadcastAddImpl<RegBlockInt32<4,4>, RegBlockInt32<4,1>> {
  static RegBlockInt32<4,4> Run(const RegBlockInt32<4,4>& lhs, const RegBlockInt32<4,1>& rhs) {
    RegBlockInt32<4,4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = Add(lhs.buf.reg[1], rhs.buf.reg[0]);
    result.buf.reg[2] = Add(lhs.buf.reg[2], rhs.buf.reg[0]);
    result.buf.reg[3] = Add(lhs.buf.reg[3], rhs.buf.reg[0]);
    return result;
  }
};

// 4x4 := 4x4 + 4x4
template <>
struct BroadcastAddImpl<RegBlockInt32<4,4>, RegBlockInt32<4,4>> {
  static RegBlockInt32<4,4> Run(const RegBlockInt32<4,4>& lhs, const RegBlockInt32<4,4>& rhs) {
    RegBlockInt32<4,4> result;
    for (int i = 0; i < 4; i++) {
      result.buf.reg[i] = Add(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

// 8x4 := 8x4 + 8x4
template <>
struct BroadcastAddImpl<RegBlockInt32<8,4>, RegBlockInt32<8,4>> {
  static RegBlockInt32<8,4> Run(const RegBlockInt32<8,4>& lhs, const RegBlockInt32<8,4>& rhs) {
    RegBlockInt32<8,4> result;
    for (int i = 0; i < 8; i++) {
      result.buf.reg[i] = Add(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

// 4x8 := 4x8 + 4x8
template <>
struct BroadcastAddImpl<RegBlockInt32<4,8>, RegBlockInt32<4,8>> {
  static RegBlockInt32<4,8> Run(const RegBlockInt32<4,8>& lhs, const RegBlockInt32<4,8>& rhs) {
    RegBlockInt32<4,8> result;
    for (int i = 0; i < 8; i++) {
      result.buf.reg[i] = Add(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

// 8x8 := 8x8 + 8x8
template <>
struct BroadcastAddImpl<RegBlockInt32<8,8>, RegBlockInt32<8,8>> {
  static RegBlockInt32<8,8> Run(const RegBlockInt32<8,8>& lhs, const RegBlockInt32<8,8>& rhs) {
    RegBlockInt32<8,8> result;
    for (int i = 0; i < 16; i++) {
      result.buf.reg[i] = Add(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

// 8x1 := 8x1 + 1x1
template <>
struct BroadcastAddImpl<RegBlockInt32<8,1>, RegBlockInt32<1,1>> {
  static RegBlockInt32<8,1> Run(const RegBlockInt32<8,1>& lhs, const RegBlockInt32<1,1>& rhs) {
    RegBlockInt32<8,1> result;
    const Int32x4 p = Dup<Int32x4>(rhs.buf.reg[0]);
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = Add(lhs.buf.reg[i], p);
    }
    return result;
  }
};

// 8x1 := 8x1 + 8x1
template <>
struct BroadcastAddImpl<RegBlockInt32<8,1>, RegBlockInt32<8,1>> {
  static RegBlockInt32<8,1> Run(const RegBlockInt32<8,1>& lhs, const RegBlockInt32<8,1>& rhs) {
    RegBlockInt32<8,1> result;
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = Add(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

// 8x4 := 8x4 + 1x4
template <>
struct BroadcastAddImpl<RegBlockInt32<8,4>, RegBlockInt32<1,4>> {
  static RegBlockInt32<8,4> Run(const RegBlockInt32<8,4>& lhs, const RegBlockInt32<1,4>& rhs) {
    RegBlockInt32<8,4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[1] = Add(lhs.buf.reg[1], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[2] = Add(lhs.buf.reg[2], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[3] = Add(lhs.buf.reg[3], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[4] = Add(lhs.buf.reg[4], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[5] = Add(lhs.buf.reg[5], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[6] = Add(lhs.buf.reg[6], DupLane<3>(rhs.buf.reg[0]));
    result.buf.reg[7] = Add(lhs.buf.reg[7], DupLane<3>(rhs.buf.reg[0]));
    return result;
  }
};

// 8x4 := 8x4 + 8x1
template <>
struct BroadcastAddImpl<RegBlockInt32<8,4>, RegBlockInt32<8,1>> {
  static RegBlockInt32<8,4> Run(const RegBlockInt32<8,4>& lhs, const RegBlockInt32<8,1>& rhs) {
    RegBlockInt32<8,4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = Add(lhs.buf.reg[1], rhs.buf.reg[1]);
    result.buf.reg[2] = Add(lhs.buf.reg[2], rhs.buf.reg[0]);
    result.buf.reg[3] = Add(lhs.buf.reg[3], rhs.buf.reg[1]);
    result.buf.reg[4] = Add(lhs.buf.reg[4], rhs.buf.reg[0]);
    result.buf.reg[5] = Add(lhs.buf.reg[5], rhs.buf.reg[1]);
    result.buf.reg[6] = Add(lhs.buf.reg[6], rhs.buf.reg[0]);
    result.buf.reg[7] = Add(lhs.buf.reg[7], rhs.buf.reg[1]);
    return result;
  }
};

// 4x8 := 4x8 + 4x1
template <>
struct BroadcastAddImpl<RegBlockInt32<4,8>, RegBlockInt32<4,1>> {
  static RegBlockInt32<4,8> Run(const RegBlockInt32<4,8>& lhs, const RegBlockInt32<4,1>& rhs) {
    RegBlockInt32<4,8> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = Add(lhs.buf.reg[1], rhs.buf.reg[0]);
    result.buf.reg[2] = Add(lhs.buf.reg[2], rhs.buf.reg[0]);
    result.buf.reg[3] = Add(lhs.buf.reg[3], rhs.buf.reg[0]);
    result.buf.reg[4] = Add(lhs.buf.reg[4], rhs.buf.reg[0]);
    result.buf.reg[5] = Add(lhs.buf.reg[5], rhs.buf.reg[0]);
    result.buf.reg[6] = Add(lhs.buf.reg[6], rhs.buf.reg[0]);
    result.buf.reg[7] = Add(lhs.buf.reg[7], rhs.buf.reg[0]);
    return result;
  }
};

// 4x8 := 4x8 + 1x8
template <>
struct BroadcastAddImpl<RegBlockInt32<4,8>, RegBlockInt32<1,8>> {
  static RegBlockInt32<4,8> Run(const RegBlockInt32<4,8>& lhs, const RegBlockInt32<1,8>& rhs) {
    RegBlockInt32<4,8> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[1] = Add(lhs.buf.reg[1], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[2] = Add(lhs.buf.reg[2], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[3] = Add(lhs.buf.reg[3], DupLane<3>(rhs.buf.reg[0]));
    result.buf.reg[4] = Add(lhs.buf.reg[4], DupLane<0>(rhs.buf.reg[1]));
    result.buf.reg[5] = Add(lhs.buf.reg[5], DupLane<1>(rhs.buf.reg[1]));
    result.buf.reg[6] = Add(lhs.buf.reg[6], DupLane<2>(rhs.buf.reg[1]));
    result.buf.reg[7] = Add(lhs.buf.reg[7], DupLane<3>(rhs.buf.reg[1]));
    return result;
  }
};

// 1x8 := 1x8 + 1x8
template <>
struct BroadcastAddImpl<RegBlockInt32<1,8>, RegBlockInt32<1,8>> {
  static RegBlockInt32<1,8> Run(const RegBlockInt32<1,8>& lhs, const RegBlockInt32<1,8>& rhs) {
    RegBlockInt32<1,8> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = Add(lhs.buf.reg[1], rhs.buf.reg[1]);
    return result;
  }
};

// 1x8 := 1x8 + 1x1
template <>
struct BroadcastAddImpl<RegBlockInt32<1,8>, RegBlockInt32<1,1>> {
  static RegBlockInt32<1,8> Run(const RegBlockInt32<1,8>& lhs, const RegBlockInt32<1,1>& rhs) {
    RegBlockInt32<1,8> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    result.buf.reg[1] = Add(lhs.buf.reg[1], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 8x8 := 8x8 + 1x8
template <>
struct BroadcastAddImpl<RegBlockInt32<8,8>, RegBlockInt32<1,8>> {
  static RegBlockInt32<8,8> Run(const RegBlockInt32<8,8>& lhs, const RegBlockInt32<1,8>& rhs) {
    RegBlockInt32<8,8> result;
    const Int32x4 p0 = rhs.buf.reg[0];
    const Int32x4 p1 = rhs.buf.reg[1];
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        const int k = 8 * i + j;
        result.buf.reg[k + 0] = Add(lhs.buf.reg[k + 0], DupLane<0>(rhs.buf.reg[i]));
        result.buf.reg[k + 2] = Add(lhs.buf.reg[k + 2], DupLane<1>(rhs.buf.reg[i]));
        result.buf.reg[k + 4] = Add(lhs.buf.reg[k + 4], DupLane<2>(rhs.buf.reg[i]));
        result.buf.reg[k + 6] = Add(lhs.buf.reg[k + 6], DupLane<3>(rhs.buf.reg[i]));
      }
    }
    return result;
  }
};

// 8x8 := 8x8 + 8x1
template <>
struct BroadcastAddImpl<RegBlockInt32<8,8>, RegBlockInt32<8,1>> {
  static RegBlockInt32<8,8> Run(const RegBlockInt32<8,8>& lhs, const RegBlockInt32<8,1>& rhs) {
    RegBlockInt32<8,8> result;
    for (int i = 0; i < 8; i++) {
      result.buf.reg[2 * i + 0] = Add(lhs.buf.reg[2 * i + 0], rhs.buf.reg[0]);
      result.buf.reg[2 * i + 1] = Add(lhs.buf.reg[2 * i + 1], rhs.buf.reg[1]);
    }
    return result;
  }
};

// 4x1 := 4x1 * 1x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<4,1>, RegBlockInt32<1,1>, Multiplier> {
  static RegBlockInt32<4,1> Run(const RegBlockInt32<4,1>& lhs, const RegBlockInt32<1,1>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<4,1> result;
    result.buf.reg[0] = multiplier.Mul(Mul(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0])));
    return result;
  }
};

// 4x1 := 4x1 * 4x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<4,1>, RegBlockInt32<4,1>, Multiplier> {
  static RegBlockInt32<4,1> Run(const RegBlockInt32<4,1>& lhs, const RegBlockInt32<4,1>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<4,1> result;
    result.buf.reg[0] = multiplier.Mul(Mul(lhs.buf.reg[0], rhs.buf.reg[0]));
    return result;
  }
};

// 1x4 := 1x4 * 1x4
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<1,4>, RegBlockInt32<1,4>, Multiplier> {
  static RegBlockInt32<1,4> Run(const RegBlockInt32<1,4>& lhs, const RegBlockInt32<1,4>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<1,4> result;
    result.buf.reg[0] = multiplier.Mul(Mul(lhs.buf.reg[0], rhs.buf.reg[0]));
    return result;
  }
};

// 1x4 := 1x4 * 1x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<1,4>, RegBlockInt32<1,1>, Multiplier> {
  static RegBlockInt32<1,4> Run(const RegBlockInt32<1,4>& lhs, const RegBlockInt32<1,1>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<1,4> result;
    result.buf.reg[0] = multiplier.Mul(Mul(lhs.buf.reg[0], rhs.buf.reg[0]));
    return result;
  }
};

// 4x4 := 4x1 * 1x4
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<4,1>, RegBlockInt32<1,4>, Multiplier> {
  static RegBlockInt32<4,4> Run(const RegBlockInt32<4,1>& lhs, const RegBlockInt32<1,4>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<4,4> result;
    const Int32x4 p = multiplier.Mul(rhs.buf.reg[0]);
    result.buf.reg[0] = MulByRhsLane<0>(lhs.buf.reg[0], p);
    result.buf.reg[1] = MulByRhsLane<1>(lhs.buf.reg[0], p);
    result.buf.reg[2] = MulByRhsLane<2>(lhs.buf.reg[0], p);
    result.buf.reg[3] = MulByRhsLane<3>(lhs.buf.reg[0], p);
    return result;
  }
};

// 4x4 := 4x4 * 1x4
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<4,4>, RegBlockInt32<1,4>, Multiplier> {
  static RegBlockInt32<4,4> Run(const RegBlockInt32<4,4>& lhs, const RegBlockInt32<1,4>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<4,4> result;
    const Int32x4 p = multiplier.Mul(rhs.buf.reg[0]);
    result.buf.reg[0] = MulByRhsLane<0>(lhs.buf.reg[0], p);
    result.buf.reg[1] = MulByRhsLane<1>(lhs.buf.reg[1], p);
    result.buf.reg[2] = MulByRhsLane<2>(lhs.buf.reg[2], p);
    result.buf.reg[3] = MulByRhsLane<3>(lhs.buf.reg[3], p);
    return result;
  }
};

// 4x4 := 4x4 * 4x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<4,4>, RegBlockInt32<4,1>, Multiplier> {
  static RegBlockInt32<4,4> Run(const RegBlockInt32<4,4>& lhs, const RegBlockInt32<4,1>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<4,4> result;
    const Int32x4 p = multiplier.Mul(rhs.buf.reg[0]);
    result.buf.reg[0] = Mul(lhs.buf.reg[0], p);
    result.buf.reg[1] = Mul(lhs.buf.reg[1], p);
    result.buf.reg[2] = Mul(lhs.buf.reg[2], p);
    result.buf.reg[3] = Mul(lhs.buf.reg[3], p);
    return result;
  }
};

// 8x1 := 8x1 * 1x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<8,1>, RegBlockInt32<1,1>, Multiplier> {
  static RegBlockInt32<8,1> Run(const RegBlockInt32<8,1>& lhs, const RegBlockInt32<1,1>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<8,1> result;
    const std::int32_t p = multiplier.Mul(rhs.buf.reg[0]);
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = Mul(lhs.buf.reg[i], p);
    }
    return result;
  }
};

// 8x1 := 8x1 * 8x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<8,1>, RegBlockInt32<8,1>, Multiplier> {
  static RegBlockInt32<8,1> Run(const RegBlockInt32<8,1>& lhs, const RegBlockInt32<8,1>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<8,1> result;
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = Mul(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

// 8x4 := 8x1 * 1x4
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<8,1>, RegBlockInt32<1,4>, Multiplier> {
  static RegBlockInt32<8,4> Run(const RegBlockInt32<8,1>& lhs, const RegBlockInt32<1,4>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<8,4> result;
    const Int32x4 p = multiplier.Mul(rhs.buf.reg[0]);
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i + 0] = MulByRhsLane<0>(lhs.buf.reg[i], p);
      result.buf.reg[i + 2] = MulByRhsLane<1>(lhs.buf.reg[i], p);
      result.buf.reg[i + 4] = MulByRhsLane<2>(lhs.buf.reg[i], p);
      result.buf.reg[i + 6] = MulByRhsLane<3>(lhs.buf.reg[i], p);
    }
    return result;
  }
};

// 8x8 := 8x1 * 1x8
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<8,1>, RegBlockInt32<1,8>, Multiplier> {
  static RegBlockInt32<8,8> Run(const RegBlockInt32<8,1>& lhs, const RegBlockInt32<1,8>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<8,8> result;
    const Int32x4 p[2] {
      multiplier.Mul(rhs.buf.reg[0]),
      multiplier.Mul(rhs.buf.reg[1])
    };
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        result.buf.reg[8 * j + i + 0] = MulByRhsLane<0>(lhs.buf.reg[i], p[j]);
        result.buf.reg[8 * j + i + 2] = MulByRhsLane<1>(lhs.buf.reg[i], p[j]);
        result.buf.reg[8 * j + i + 4] = MulByRhsLane<2>(lhs.buf.reg[i], p[j]);
        result.buf.reg[8 * j + i + 6] = MulByRhsLane<3>(lhs.buf.reg[i], p[j]);
      }
    }
    return result;
  }
};

// 8x8 := 8x8 * 1x8
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<8,8>, RegBlockInt32<1,8>, Multiplier> {
  static RegBlockInt32<8,8> Run(const RegBlockInt32<8,8>& lhs, const RegBlockInt32<1,8>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<8,8> result;
    const Int32x4 p[2] {
      multiplier.Mul(rhs.buf.reg[0]),
      multiplier.Mul(rhs.buf.reg[1])
    };
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        const int k = 8 * j + i;
        result.buf.reg[k + 0] = MulByRhsLane<0>(lhs.buf.reg[k + 0], p[j]);
        result.buf.reg[k + 2] = MulByRhsLane<1>(lhs.buf.reg[k + 2], p[j]);
        result.buf.reg[k + 4] = MulByRhsLane<2>(lhs.buf.reg[k + 4], p[j]);
        result.buf.reg[k + 6] = MulByRhsLane<3>(lhs.buf.reg[k + 6], p[j]);
      }
    }
    return result;
  }
};

// 8x4 := 8x4 * 1x4
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<8,4>, RegBlockInt32<1,4>, Multiplier> {
  static RegBlockInt32<8,4> Run(const RegBlockInt32<8,4>& lhs, const RegBlockInt32<1,4>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<8,4> result;
    const Int32x4 p = multiplier.Mul(rhs.buf.reg[0]);
    for (int i = 0; i < 2; i++) {
        result.buf.reg[i + 0] = MulByRhsLane<0>(lhs.buf.reg[i + 0], p);
        result.buf.reg[i + 2] = MulByRhsLane<1>(lhs.buf.reg[i + 2], p);
        result.buf.reg[i + 4] = MulByRhsLane<2>(lhs.buf.reg[i + 4], p);
        result.buf.reg[i + 6] = MulByRhsLane<3>(lhs.buf.reg[i + 6], p);
    }
    return result;
  }
};

// 8x4 := 8x4 * 8x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<8,4>, RegBlockInt32<8,1>, Multiplier> {
  static RegBlockInt32<8,4> Run(const RegBlockInt32<8,4>& lhs, const RegBlockInt32<8,1>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<8,4> result;
    const Int32x4 p[2] {
      multiplier.Mul(rhs.buf.reg[0]),
      multiplier.Mul(rhs.buf.reg[1])
    };
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 2; j++) {
        const int k = j + 2 * i;
        result.buf.reg[k] = Mul(lhs.buf.reg[k], p[j]);
      }
    }
    return result;
  }
};

// 1x8 := 1x8 * 1x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<1,8>, RegBlockInt32<1,1>, Multiplier> {
  static RegBlockInt32<1,8> Run(const RegBlockInt32<1,8>& lhs, const RegBlockInt32<1,1>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<1,8> result;
    const std::int32_t p = multiplier.Mul(rhs.buf.reg[0]);
    result.buf.reg[0] = Mul(lhs.buf.reg[0], p);
    result.buf.reg[1] = Mul(lhs.buf.reg[1], p);
    return result;
  }
};

// 1x8 := 1x8 * 1x8
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<1,8>, RegBlockInt32<1,8>, Multiplier> {
  static RegBlockInt32<1,8> Run(const RegBlockInt32<1,8>& lhs, const RegBlockInt32<1,8>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<1,8> result;
    result.buf.reg[0] = multiplier.Mul(Mul(lhs.buf.reg[0], rhs.buf.reg[0]));
    result.buf.reg[1] = multiplier.Mul(Mul(lhs.buf.reg[1], rhs.buf.reg[1]));
    return result;
  }
};

// 4x8 := 1x8 * 4x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<1,8>, RegBlockInt32<4,1>, Multiplier> {
  static RegBlockInt32<4,8> Run(const RegBlockInt32<1,8>& lhs, const RegBlockInt32<4,1>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<4,8> result;
    const Int32x4 p = multiplier.Mul(rhs.buf.reg[0]);
    for (int i = 0; i < 2; i++) {
      result.buf.reg[4 * i + 0] = MulByRhsLane<0>(p, lhs.buf.reg[i]);
      result.buf.reg[4 * i + 1] = MulByRhsLane<1>(p, lhs.buf.reg[i]);
      result.buf.reg[4 * i + 2] = MulByRhsLane<2>(p, lhs.buf.reg[i]);
      result.buf.reg[4 * i + 3] = MulByRhsLane<3>(p, lhs.buf.reg[i]); 
    }
    return result;
  }
};

// 4x8 := 1x8 * 4x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<4,8>, RegBlockInt32<1,8>, Multiplier> {
  static RegBlockInt32<4,8> Run(const RegBlockInt32<4,8>& lhs, const RegBlockInt32<1,8>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<4,8> result;
    const Int32x4 p[2] {
      multiplier.Mul(rhs.buf.reg[0]),
      multiplier.Mul(rhs.buf.reg[1])
    };
    for (int i = 0; i < 2; i++) {
      const int k = 4 * i;
      result.buf.reg[k + 0] = MulByRhsLane<0>(lhs.buf.reg[k + 0], p[i]);
      result.buf.reg[k + 1] = MulByRhsLane<1>(lhs.buf.reg[k + 1], p[i]);
      result.buf.reg[k + 2] = MulByRhsLane<2>(lhs.buf.reg[k + 2], p[i]);
      result.buf.reg[k + 3] = MulByRhsLane<3>(lhs.buf.reg[k + 3], p[i]);      
    }
    return result;
  }
};

// 4x8 := 4x8 * 4x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<4,8>, RegBlockInt32<4,1>, Multiplier> {
  static RegBlockInt32<4,8> Run(const RegBlockInt32<4,8>& lhs, const RegBlockInt32<4,1>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<4,8> result;
    const Int32x4 p = multiplier.Mul(rhs.buf.reg[0]);
    for (int i = 0; i < 8; i++) {
      result.buf.reg[i] = Mul(lhs.buf.reg[i], p);
    }
    return result;
  }
};

// 8x8 := 8x8 * 8x1
template <typename Multiplier>
struct BroadcastMulImpl<RegBlockInt32<8,8>, RegBlockInt32<8,1>, Multiplier> {
  static RegBlockInt32<8,8> Run(const RegBlockInt32<8,8>& lhs, const RegBlockInt32<8,1>& rhs, const Multiplier& multiplier) {
    RegBlockInt32<8,8> result;
    const Int32x4 p0 = multiplier.Mul(rhs.buf.reg[0]);
    const Int32x4 p1 = multiplier.Mul(rhs.buf.reg[1]);
    for (int i = 0; i < 8; i++) {
      result.buf.reg[2 * i + 0] = Mul(lhs.buf.reg[2 * i + 0], p0);
      result.buf.reg[2 * i + 1] = Mul(lhs.buf.reg[2 * i + 1], p1);
    }
    return result;
  }
};

template<>
struct ConstantMultiplierInt32Impl<Int32x4>
{
  static Int32x4 Mul(std::int32_t c, Int32x4 x) {
    return gemmlowp::Mul(x, Dup<Int32x4>(c));
  }
};


}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_OUTPUT_COMMON_NEON_SSE_H_