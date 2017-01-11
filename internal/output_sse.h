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

// output_sse.h: optimized SSE4.2 specializations of the templates in output.h.

#ifndef GEMMLOWP_INTERNAL_OUTPUT_SSE4_H_
#define GEMMLOWP_INTERNAL_OUTPUT_SSE4_H_

#include <smmintrin.h>
#include "output.h"

namespace gemmlowp {

typedef struct _int32x16x1_t { __m128i val[4]; } int32x16x1_t;

// Definitions of Fragment types wrapping SSE4.2 vector types.
typedef Fragment<__m128i, 4, 1, MapOrder::ColMajor> SSE4FragmentInt32x4x1;
typedef Fragment<int32x16x1_t, 16, 1, MapOrder::ColMajor>
    SSE4FragmentInt32x16x1;
typedef Fragment<std::uint32_t, 4, 1, MapOrder::ColMajor> SSE4FragmentUint8x4x1;
typedef Fragment<__m128i, 16, 1, MapOrder::ColMajor> SSE4FragmentUint8x16x1;

template <typename OutputStageType>
struct OutputStageEvalImpl<OutputStageType, SSE4FragmentInt32x16x1> {
  typedef SSE4FragmentInt32x16x1 InputType;
  typedef SSE4FragmentInt32x16x1 OutputType;
  typedef OutputStageEvalImpl<OutputStageType, SSE4FragmentInt32x4x1>
      ImplInt32x4;
  OutputStageEvalImpl(const OutputStageType& s) : impl_int32x4(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    OutputType output;

    for (int i = 0; i < 4; i++) {
      output.data.val[i] =
          impl_int32x4.Eval(input.data.val[i], row + 4 * i, col);
    }
    return output;
  }

  ImplInt32x4 impl_int32x4;
};

// Implementation of OutputStageQuantizeDownInt32ToUint8Scale for
// SSE4FragmentInt32x4x1
template <>
struct OutputStageEvalImpl<OutputStageQuantizeDownInt32ToUint8Scale,
                           SSE4FragmentInt32x4x1> {
  typedef SSE4FragmentInt32x4x1 InputType;
  typedef SSE4FragmentInt32x4x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8Scale OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int, int) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const __m128i result_mult_int =
        _mm_set1_epi32(output_stage.result_mult_int);
    const __m128i result_offset = _mm_set1_epi32(output_stage.result_offset);
    const __m128i a =
        _mm_mullo_epi32(_mm_add_epi32(input, result_offset), result_mult_int);
    return RoundingDivideByPOT(a, result_shift);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageQuantizeDownInt32ToUint8ScalePC for
// SSE4FragmentInt32x4x1
template <>
struct OutputStageEvalImpl<
    OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Col>,
    SSE4FragmentInt32x4x1> {
  typedef SSE4FragmentInt32x4x1 InputType;
  typedef SSE4FragmentInt32x4x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Col>
      OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int /*col*/) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const __m128i result_mult_int =
        _mm_lddqu_si128(reinterpret_cast<const __m128i*>(
            output_stage.result_mult_int.data(row)));
    const __m128i result_offset = _mm_lddqu_si128(
        reinterpret_cast<const __m128i*>(output_stage.result_offset.data(row)));
    const __m128i a =
        _mm_mullo_epi32(_mm_add_epi32(input, result_offset), result_mult_int);
    return RoundingDivideByPOT(a, result_shift);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageQuantizeDownInt32ToUint8ScalePC for
// SSE4FragmentInt32x4x1
template <>
struct OutputStageEvalImpl<
    OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Row>,
    SSE4FragmentInt32x4x1> {
  typedef SSE4FragmentInt32x4x1 InputType;
  typedef SSE4FragmentInt32x4x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Row>
      OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const __m128i result_mult_int =
        _mm_lddqu_si128(reinterpret_cast<const __m128i*>(
            output_stage.result_mult_int.data(col)));
    const __m128i result_offset = _mm_lddqu_si128(
        reinterpret_cast<const __m128i*>(output_stage.result_offset.data(row)));
    const __m128i a =
        _mm_mullo_epi32(_mm_add_epi32(input, result_offset), result_mult_int);
    return RoundingDivideByPOT(a, result_shift);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint for
// SSE4FragmentInt32x4x1
template <>
struct OutputStageEvalImpl<OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint,
                           SSE4FragmentInt32x4x1> {
  typedef SSE4FragmentInt32x4x1 InputType;
  typedef SSE4FragmentInt32x4x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int, int) const {
    const __m128i mulhigh_val = SaturatingRoundingDoublingHighMul(
        input.data, _mm_set1_epi32(output_stage.result_fixedpoint_multiplier));
    const std::int32_t result_shift = output_stage.result_shift;
    const __m128i shifted_val = RoundingDivideByPOT(mulhigh_val, result_shift);
    return Add(shifted_val,
               _mm_set1_epi32(output_stage.result_offset_after_shift));
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageSaturatingCastToUint8 for SSE4FragmentInt32x4x1
template <>
struct OutputStageEvalImpl<OutputStageSaturatingCastToUint8,
                           SSE4FragmentInt32x4x1> {
  typedef SSE4FragmentInt32x4x1 InputType;
  typedef SSE4FragmentUint8x4x1 OutputType;
  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalImpl(const OutputStage&) {}

  OutputType Eval(InputType input, int, int) const {
    const __m128i zero = _mm_set1_epi32(0);
    __m128i res_16 = _mm_packs_epi32(input, zero);
    __m128i res_8 = _mm_packus_epi16(res_16, zero);
    return _mm_cvtsi128_si32(res_8);
  }
};

// In the case of OutputStageSaturatingCastToUint8, the handling of
// SSE4FragmentInt32x16x1 data can be made much more efficient by handling
// it all at once, instead of as 4 separate int32x4 values as in the above
// generic partial specialization. This also avoids the poor (50%) register
// utilization of FragmentUint8x4x1: by handling 16 scalar values at once,
// we are able to fill a uint8x16_t.
template <>
struct OutputStageEvalImpl<OutputStageSaturatingCastToUint8,
                           SSE4FragmentInt32x16x1> {
  typedef SSE4FragmentInt32x16x1 InputType;
  typedef SSE4FragmentUint8x16x1 OutputType;
  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalImpl(const OutputStage&) {}

  OutputType Eval(InputType input, int, int) const {
    __m128i q16[2];
    for (int i = 0; i < 2; i++) {
      q16[i] =
          _mm_packus_epi32(input.data.val[2 * i], input.data.val[2 * i + 1]);
    }
    return _mm_packus_epi16(q16[0], q16[1]);
  }
};

// Implementation of OutputStageBiasAddition for SSE4FragmentInt32x4x1
template <typename VectorType>
struct OutputStageEvalImpl<OutputStageBiasAddition<VectorType>,
                           SSE4FragmentInt32x4x1> {
  typedef SSE4FragmentInt32x4x1 InputType;
  typedef SSE4FragmentInt32x4x1 OutputType;
  typedef OutputStageBiasAddition<VectorType> OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    __m128i bias;
    if (VectorType::kShape == VectorShape::Row) {
      bias = _mm_set1_epi32(output_stage.bias_vector(col));
    } else {
      bias = _mm_lddqu_si128(
          reinterpret_cast<const __m128i*>(output_stage.bias_vector.data(row)));
    }
    return _mm_add_epi32(input, bias);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageClamp for SSE4FragmentInt32x4x1
template <>
struct OutputStageEvalImpl<OutputStageClamp, SSE4FragmentInt32x4x1> {
  typedef SSE4FragmentInt32x4x1 InputType;
  typedef SSE4FragmentInt32x4x1 OutputType;
  typedef OutputStageClamp OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int, int) const {
    const __m128i min = _mm_set1_epi32(output_stage.min);
    const __m128i max = _mm_set1_epi32(output_stage.max);
    return _mm_min_epi32(_mm_max_epi32(input, min), max);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageTanh for SSE4FragmentInt32x4x1
template <>
struct OutputStageEvalImpl<OutputStageTanh, SSE4FragmentInt32x4x1>
    : OutputStageTanhEvalImpl<SSE4FragmentInt32x4x1> {
  OutputStageEvalImpl(const OutputStageTanh& output_stage)
      : OutputStageTanhEvalImpl(output_stage) {}
};

// Specialization of StoreFinalOutput for SSE4FragmentUint8x4x1.
template <typename DstType>
inline void StoreFinalOutput(SSE4FragmentUint8x4x1 value, DstType* dst, int row,
                             int col) {
  unsigned char* tmp = dst->data(row, col);
  for (int i = 0; i < 4; i++) tmp[i] = (value >> (i * 8)) & 0xff;
}

// Specialization of StoreFinalOutput for SSE4FragmentUint8x16x1.
template <typename DstType>
inline void StoreFinalOutput(SSE4FragmentUint8x16x1 value, DstType* dst,
                             int row, int col) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst->data(row, col)), value);
}

// Specialization of StoreFinalOutput for SSE4FragmentInt32x4x1, storing into
// a int32 destination.
template <typename DstType>
inline void StoreFinalOutput(SSE4FragmentInt32x4x1 value, DstType* dst, int row,
                             int col) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst->data(row, col)), value);
}

// Specialization of StoreFinalOutput for SSE4FragmentInt32x16x1, storing into
// a int32 destination.
template <typename DstType>
inline void StoreFinalOutput(SSE4FragmentInt32x16x1 value, DstType* dst,
                             int row, int col) {
  for (int i = 0; i < 4; i++) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst->data(row + 4 * i, col)),
                     value.data.val[i]);
  }
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_OUTPUT_SSE4_H_
