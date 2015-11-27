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

#include <iostream>
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
//      or even struct-like types grouping more scalar values into
//      bigger vectors. Thus, there can be several EvalOutputStageImpl
//      specializations for a single OutputStageType, for different
//      InputType's.
template <typename OutputStageType, typename InputType>
struct EvalOutputStageImpl {};

// Implementation of OutputStageQuantizeDownInt32ToUint8Scale for scalar data
template <>
struct EvalOutputStageImpl<OutputStageQuantizeDownInt32ToUint8Scale,
                           std::int32_t> {
  typedef std::int32_t InputType;
  typedef std::int32_t OutputType;

  static OutputType Eval(
      const OutputStageQuantizeDownInt32ToUint8Scale& output_stage,
      InputType input) {
    std::int32_t result_shift = output_stage.result_shift;
    std::int32_t result_mult_int = output_stage.result_mult_int;
    std::int32_t result_offset = output_stage.result_offset;
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

  static OutputType Eval(const OutputStageSaturatingCastToUint8& output_stage,
                         InputType input) {
    return input > 255 ? 255 : input < 0 ? 0 : input;
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
// the whole
// pipeline. It is a recursive template to implement compile-time unrolling of
// the loop
// over all pipeline stages. The 'FirstStage' parameter is how we implement
// recursion:
// each specialization implements only evaluation starting at 'FirstStage'.
// The StopRecursion parameter is just a helper to implement the termination of
// the
// recursion as a partial specialization below.
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
                         InputType input) {
    // Evaluate the first stage.
    FirstStageOutputType first_stage_output =
        EvalOutputStageImpl<FirstStageType, InputType>::Eval(
            std::get<FirstStage>(output_pipeline), input);
    // Recurse into the remaining stages.
    return EvalOutputPipelineFromStageImpl<
        OutputPipelineType, FirstStage + 1,
        FirstStageOutputType>::Eval(output_pipeline, first_stage_output);
  }
};

// Specialization on 'StopRecursion' for terminating the recursion.
template <typename OutputPipelineType, int FirstStage, typename InputType>
struct EvalOutputPipelineFromStageImpl<OutputPipelineType, FirstStage,
                                       InputType, true> {
  static InputType Eval(const OutputPipelineType&, InputType input) {
    // Terminating the recursion.
    return input;
  }
};

// StoreUnpackOutput takes the final value at the end of the output pipeline and
// stores it into the destination matrix. It can be specialized for different
// data types;
// the generic implementation here is typically used only for plain old scalar
// (not SIMD)
// types.
// The return value is the number of entries (of type DstType) written to dst.
template <typename OutputType, typename DstType>
std::size_t StoreUnpackOutput(OutputType value, DstType* dst) {
  *dst = value;
  return 1;
}

// RunOutputPipeline is the entry point into the output pipeline evaluation
// code.
// It should be the only thing that unpack code calls. It takes the result of
// the
// unpack stage and stores it into the destination matrix.
// The return value is the number of entries (of type DstType) written to dst.
template <typename OutputPipelineType, typename InputType, typename DstType>
std::size_t RunOutputPipeline(const OutputPipelineType& output_pipeline,
                              InputType input, DstType* dst) {
  // Evaluate the output pipeline.
  auto output =
      EvalOutputPipelineFromStageImpl<OutputPipelineType, 0, InputType>::Eval(
          output_pipeline, input);
  // Store the result into the destination matrix.
  // std::cerr << __PRETTY_FUNCTION__ << std::endl;
  return StoreUnpackOutput(output, dst);
}

}  // namespace gemmlowp

#ifdef GEMMLOWP_NEON
#include "output_neon.h"
#endif

#endif  // GEMMLOWP_INTERNAL_OUTPUT_H_
