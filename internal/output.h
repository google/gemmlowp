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

// output.h: processing the 32-bit accumulators output by the unpack
// stage, obtaining the final result matrix entries and storing them into
// the destination matrix.

#ifndef GEMMLOWP_INTERNAL_OUTPUT_H_
#define GEMMLOWP_INTERNAL_OUTPUT_H_

#include <tuple>
#include <type_traits>

#include "../public/output_stages.h"

namespace gemmlowp {

// EvalOutputStageImpl is the template that we specialize to provide
// implementations of each output stage for each type of input data.
//
// Each specialization provides a OutputType typedef and an Eval function
// returning OutputType. The OutputType typically depends on the InputType.
//
// There are two dimensions in which input data types can vary:
//   1. Different output stages may expect different data types. The
//      only hard constraint is that the first stage accepts int32, as
//      the unpack stage produces int32 accumulators.
//   2. For a given scalar data type such as int32, there is still the
//      possibility of having SIMD vector types such as NEON int32x4_t,
//      typically wrapped as "fragment" types, see struct Fragment.
//      Thus, there can be several EvalOutputStageImpl
//      specializations for a single OutputStageType, for different
//      InputType's.
template <typename OutputStageType, typename InputType>
struct EvalOutputStageImpl {
  // This generic template body should never be hit.
  static_assert(
      std::is_same<InputType, void>::value,
      "Unimplemented: missing implementation of this output pipeline stage "
      "for this data type. This would happen if some architecture-specific "
      "SIMD back-end (output_$arch.h) were incomplete.");
};

// Implementation of OutputStageQuantizeDownInt32ToUint8Scale for scalar data
template <>
struct EvalOutputStageImpl<OutputStageQuantizeDownInt32ToUint8Scale,
                           std::int32_t> {
  typedef std::int32_t InputType;
  typedef std::int32_t OutputType;

  static OutputType Eval(
      const OutputStageQuantizeDownInt32ToUint8Scale& output_stage,
      InputType input, int, int) {
    const std::int32_t result_shift = output_stage.result_shift;
    const std::int32_t result_mult_int = output_stage.result_mult_int;
    const std::int32_t result_offset = output_stage.result_offset;
    const std::int32_t kRoundingTerm =
        (result_shift < 1) ? 0 : (1 << (result_shift - 1));
    return ((input + result_offset) * result_mult_int + kRoundingTerm) >>
           result_shift;
  }
};

// Implementation of OutputStageSaturatingCastToUint8 for scalar data
template <>
struct EvalOutputStageImpl<OutputStageSaturatingCastToUint8, std::int32_t> {
  typedef std::int32_t InputType;
  typedef std::uint8_t OutputType;

  static OutputType Eval(const OutputStageSaturatingCastToUint8&,
                         InputType input, int, int) {
    return input > 255 ? 255 : input < 0 ? 0 : input;
  }
};

// Implementation of OutputStageBiasAddition for scalar data
template <typename VectorType>
struct EvalOutputStageImpl<OutputStageBiasAddition<VectorType>, std::int32_t> {
  typedef std::int32_t InputType;
  typedef std::int32_t OutputType;
  typedef OutputStageBiasAddition<VectorType> OutputStage;

  static OutputType Eval(const OutputStage& output_stage, InputType input,
                         int row, int col) {
    if (VectorType::kShape == VectorShape::Row) {
      return input + output_stage.bias_vector(col);
    } else {
      return input + output_stage.bias_vector(row);
    }
  }
};

// Implementation of OutputStageClamp for scalar data
template <>
struct EvalOutputStageImpl<OutputStageClamp, std::int32_t> {
  typedef std::int32_t InputType;
  typedef std::int32_t OutputType;

  static OutputType Eval(const OutputStageClamp& output_stage, InputType input,
                         int, int) {
    const std::int32_t min = output_stage.min;
    const std::int32_t max = output_stage.max;
    return std::min(std::max(input, min), max);
  }
};

// OutputPipelineOutputType is a helper to determine the output data type of a
// pipeline, for a
// given input data type. It is a recursive template; see the explanation on
// EvalOutputPipelineFromStageImpl below.
template <typename OutputPipelineType, int FirstStage, typename InputType,
          bool StopRecursion =
              FirstStage == std::tuple_size<OutputPipelineType>::value>
struct OutputPipelineOutputType {
  typedef typename std::tuple_element<FirstStage, OutputPipelineType>::type
      FirstStageType;
  typedef typename EvalOutputStageImpl<FirstStageType, InputType>::OutputType
      FirstStageOutputType;
  typedef typename OutputPipelineOutputType<OutputPipelineType, FirstStage + 1,
                                            FirstStageOutputType>::Type Type;
};

template <typename OutputPipelineType, int FirstStage, typename InputType>
struct OutputPipelineOutputType<OutputPipelineType, FirstStage, InputType,
                                true> {
  typedef InputType Type;
};

// EvalOutputPipelineFromStageImpl is a helper to implement the evaluation of
// the whole pipeline. It is a recursive template to implement compile-time
// unrolling of the loop over all pipeline stages. The 'FirstStage' parameter
// is how we implement recursion: each specialization implements only
// evaluation starting at 'FirstStage'. The StopRecursion parameter is just a
// helper to implement the termination of the recursion as a partial
// specialization below.
template <typename OutputPipelineType, int FirstStage, typename InputType,
          bool StopRecursion =
              FirstStage == std::tuple_size<OutputPipelineType>::value>
struct EvalOutputPipelineFromStageImpl {
  typedef typename std::tuple_element<FirstStage, OutputPipelineType>::type
      FirstStageType;
  typedef typename EvalOutputStageImpl<FirstStageType, InputType>::OutputType
      FirstStageOutputType;
  typedef typename OutputPipelineOutputType<OutputPipelineType, FirstStage,
                                            InputType>::Type OutputType;

  static OutputType Eval(const OutputPipelineType& output_pipeline,
                         InputType input, int row, int col) {
    // Evaluate the first stage.
    FirstStageOutputType first_stage_output =
        EvalOutputStageImpl<FirstStageType, InputType>::Eval(
            std::get<FirstStage>(output_pipeline), input, row, col);
    // Recurse into the remaining stages.
    return EvalOutputPipelineFromStageImpl<
        OutputPipelineType, FirstStage + 1,
        FirstStageOutputType>::Eval(output_pipeline, first_stage_output, row,
                                    col);
  }
};

// Specialization on 'StopRecursion' for terminating the recursion.
template <typename OutputPipelineType, int FirstStage, typename InputType>
struct EvalOutputPipelineFromStageImpl<OutputPipelineType, FirstStage,
                                       InputType, true> {
  static InputType Eval(const OutputPipelineType&, InputType input, int, int) {
    // Terminating the recursion.
    return input;
  }
};

// StoreFinalOutput takes the final value at the end of the output pipeline and
// stores it into the destination matrix. It can be specialized for different
// data types; the generic implementation here is typically used only for plain
// old scalar (not SIMD) types.
template <typename OutputType, typename DstType>
void StoreFinalOutput(OutputType value, DstType* dst, int row, int col) {
  *dst->data(row, col) = value;
}

// RunOutputPipeline is the entry point into the output pipeline evaluation
// code. It should be the only thing that unpack code calls. It takes the result
// of the unpack stage and stores it into the destination matrix.
template <typename OutputPipelineType, typename InputType, typename DstType>
void RunOutputPipeline(const OutputPipelineType& output_pipeline,
                       InputType input, DstType* dst, int row, int col) {
  // Statically assert that the output pipeline matches the given destination
  // matrix's scalar type.
  typedef
      typename OutputPipelineOutputType<OutputPipelineType, 0,
                                        std::int32_t>::Type ScalarOutputType;
  typedef typename DstType::Scalar ScalarDstType;
  static_assert(std::is_same<ScalarOutputType, ScalarDstType>::value,
                "mismatched destination scalar type and output pipeline");

  // Evaluate the output pipeline.
  auto output =
      EvalOutputPipelineFromStageImpl<OutputPipelineType, 0, InputType>::Eval(
          output_pipeline, input, row, col);
  // Store the result into the destination matrix.
  StoreFinalOutput(output, dst, row, col);
}

// A Fragment is a small fixed-size matrix typically stored in one or
// a few architecture-specific SIMD vectors. Besides plain old scalar types
// such as int32_t, Fragment types are what can be used as input/output data
// types for output pipeline stages.
//
// More details:
//
// In the generic scalar code in this file, we have only implemented
// evaluation of output stages for scalar inputs (e.g. plain int32_t values).
// Other files (e.g. output_neon.h) are to provide SIMD paths by implementing
// evaluation of output stages for SIMD vector types. However, this raises
// the question of how the different values ("lanes") in a SIMD vector
// correspond to different values in the whole matrices. For simple entry-wise
// output stages, this doesn't matter, but for other output stages depending
// on position within the whole matrix, this does matter. To solve this
// problem, rather than implementing evaluation of output stages for raw
// SIMD vector types, we wrap SIMD vector types in "fragment" structs that
// bring the additional structure of "shape" i.e. mapping SIMD lanes to
// matrix entries, and we specialize evaluation of output stage for such
// fragment types. The Fragment template struct here is how we generate
// all fragment structs. For example, in output_neon.h, it may be specialized
// with DataType=int32x4_t, Rows=4, Cols=1. MapOrder doesn't matter for
// vector shapes. While Fragment is only used for SIMD paths, we leave it
// here in this platform-generic file because this same template should
// cover the needs of any SIMD architectures.
template <typename DataType, int Rows, int Cols, MapOrder Order>
struct Fragment {
  Fragment() {}
  Fragment(const DataType& d) : data(d) {}
  operator DataType() const { return data; }

  DataType data;
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_OUTPUT_H_
