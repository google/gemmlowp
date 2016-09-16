#ifndef GEMMLOWP_META_BASE_H_
#define GEMMLOWP_META_BASE_H_

#include <cassert>
#include <cstdint>

#include "../internal/common.h"

namespace gemmlowp {
namespace meta {

template <int align>
int AlignTo(int value) {
  return ((value + align - 1) / align) * align;
}

int AlignTo(int align, int value) {
  return ((value + align - 1) / align) * align;
}

template <typename Kernel_, typename OutputStream_>
struct FusedKernelParams {
 public:
  typedef Kernel_ Kernel;
  typedef OutputStream_ OutputStream;

  Kernel kernel;
  OutputStream output_stream;
};

template <typename InType_, typename OutType_, typename LeftStream_,
          typename RightStream_, typename Kernel_, typename OutputStream_>
struct GemmParams {
 public:
  typedef InType_ InType;
  typedef OutType_ OutType;
  typedef LeftStream_ LeftStream;
  typedef RightStream_ RightStream;
  typedef Kernel_ Kernel;
  typedef OutputStream_ OutputStream;

  typedef FusedKernelParams<Kernel, OutputStream> FusedKernel;

  // Common parameters.

  int m;
  int n;
  int k;

  const InType* lhs;
  const InType* rhs;
  OutType* result;
  uint8_t* scratch;

  // Specialized parameters.

  LeftStream left_stream;
  RightStream right_stream;
  FusedKernel fused_kernel;
};

template <typename InType, int lanes_count, int pack_size, int leftovers,
          typename StreamParams>
class Stream {
 public:
  static void Pack(const InType* in, const StreamParams& params, InType* out);

  static int UnpackedAdvance(const StreamParams& params);

  static int PackedAdvance(const StreamParams& params);

  static int UnpackedStride(const StreamParams& params);

  static int PackedStride(const StreamParams& params);
};

template <typename InType, typename StreamType>
class StreamUtil {
 public:
  static const InType* Offset(const StreamType& params, const InType* source,
                              int offset_stride, int offset_advance);

  static int Scratch(const StreamType& params, int lanes);
};

template <typename InType, typename OutType, typename Kernel,
          typename OutputStream, int kernel_m, int kernel_n, int pack_size>
class MulKernel {
 public:
  static void Multiply(const InType* lhs, const InType* rhs,
                       const FusedKernelParams<Kernel, OutputStream>& params,
                       OutType* result);
};

}  // namespace meta
}  // namespace gemmlowp

#endif  // GEMMLOWP_META_BASE_H_
