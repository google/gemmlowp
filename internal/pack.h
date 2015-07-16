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

// pack.h: packing blocks of the LHS and RHS into the data layout
// that is expected by compute.h and eventually by kernels.
// Because this data layout depends on the kernel format, code here
// is templated in KernelLhsFormat/KernelRhsFormat.
//
// Readers note: an important theme around here is that we try hard
// to handle both Lhs and Rhs with a single piece of code. We indifferently
// refer to the Lhs and Rhs as a 'Side'. Instead of addressing matrices
// by (row, column) indices, we address them by (width, depth), as explained
// in kernel.h. This allows us to handle both Lhs and Rhs on an equal footing,
// at once.

#ifndef GEMMLOWP_INTERNAL_PACK_H_
#define GEMMLOWP_INTERNAL_PACK_H_

#include <cstring>

#include "block_params.h"
#include "kernel.h"
#include "common.h"
#include "allocator.h"

namespace gemmlowp {

// A PackedSideBlock instance is a packed block of either the LHS or RHS
// (whence the generic 'Side' name).
//
// 'Packed' means that it is laid out in the storage order that
// is expected by the specified kernel format. From a block of the input
// LHS or RHS matrix, one obtains a PackedSideBlock by calling PackLhs()
// or PackRhs().
template <typename KernelSideFormat>
class PackedSideBlock {
 public:
  PackedSideBlock(Side side, Allocator* allocator,
                  const BlockParams& block_params,
                  int rank_one_update_multiplier)
      : allocator_(allocator),
        rank_one_update_multiplier_(rank_one_update_multiplier),
        pos_(0) {
    GetSideBlockParams(side, &params_, block_params);
    data_handle_ =
        allocator_->Reserve<std::uint8_t>(params_.l2_width * params_.l2_depth);
    rank_one_update_handle_ =
        allocator_->Reserve<std::int32_t>(params_.l2_width);
  }

  ~PackedSideBlock() {}

  void seek_run(int start_width, int start_depth) const {
    int kernel_run_depth =
        std::min<int>(params_.l1_depth, params_.l2_depth - start_depth);
    pos_ = params_.l2_width * start_depth + start_width * kernel_run_depth;
  }

  void seek_next_cell() const { pos_ += KernelSideFormat::Cell::kSize; }

  void seek_forward_n_cells(int n) const {
    pos_ += n * KernelSideFormat::Cell::kSize;
  }

  const std::uint8_t* current_data() const {
    return allocator_->GetPointer<std::uint8_t>(data_handle_) + pos_;
  }

  std::uint8_t* current_data() {
    return allocator_->GetPointer<std::uint8_t>(data_handle_) + pos_;
  }

  std::int32_t* rank_one_update() {
    return allocator_->GetPointer<std::int32_t>(rank_one_update_handle_);
  }

  const std::int32_t* rank_one_update() const {
    return allocator_->GetPointer<const std::int32_t>(rank_one_update_handle_);
  }

  std::int32_t rank_one_update_multiplier() const {
    return rank_one_update_multiplier_;
  }

  const SideBlockParams& params() const { return params_; }

 private:
  // The block size parameters that this PackedSizeBlock follows.
  // The L2 parameters determine its overall size, while the L1 parameters,
  // together with the kernel format template parameter, determine
  // the fine details of the storage/traversal order.
  SideBlockParams params_;

  // Pointer to the allocator provided by the caller. Not owned.
  // The Allocator is assumed to outlive the PackedSideBlock.
  Allocator* const allocator_;

  // Handle on the buffer backing this packed block. Owned.
  Allocator::Handle data_handle_;

  // Handle on the additional buffer backing the rank-one-update vector
  // associated with this block. Owned.
  Allocator::Handle rank_one_update_handle_;

  // The constant multiplier of the rank one update vector.
  std::int32_t rank_one_update_multiplier_;

  // pos_ is the current position in the buffer, which we access
  // sequentially, like a file.
  // The idea is that we pack data in the same order as it is
  // going to be traversed during the computation, which for
  // cache-friendliness reasons is complicated to random-access,
  // as the offsets calculations would be intricate. So we
  // give up random-access addressing, and instead content ourselves
  // with sequential access.
  //
  // pos_ is mutable because during the computation we will want to
  // be able to iterate on the data in a const PackedSideBlock.
  mutable int pos_;
};

// WidthMajor and DepthMajor are custom phrases modelled after the
// standard terminology 'row-major' and 'column-major'. Their meaning
// should be transparent once one has read the explanation in kernel.h:
// for example, in the Lhs, the 'width' dimension is the rows dimension,
// so there WidthMajor means RowMajor, while in the Rhs it is the opposite.
// Another way to put it: WidthMajor means that contiguous storage is used
// for entries having the same 'width' index.
enum class SideMapOrder { WidthMajor, DepthMajor };

// Similar to MatrixMap from map.h, but in terms of width/depth instead of
// rows/columns. Used to address blocks of the input LHS/RHS matrices when
// packing them.
template <typename tScalar, SideMapOrder tOrder>
class SideMap {
 public:
  typedef tScalar Scalar;
  static const SideMapOrder kOrder = tOrder;

  SideMap(Scalar* data, int width, int depth, int stride)
      : data_(data), width_(width), depth_(depth), stride_(stride) {}

  SideMap(const SideMap& other)
      : data_(other.data_),
        width_(other.width_),
        depth_(other.depth_),
        stride_(other.stride_) {}

  int width() const { return width_; }
  int depth() const { return depth_; }
  int stride() const { return stride_; }
  int width_stride() const {
    return kOrder == SideMapOrder::DepthMajor ? 1 : stride_;
  }
  int depth_stride() const {
    return kOrder == SideMapOrder::WidthMajor ? 1 : stride_;
  }
  Scalar* data() const { return data_; }
  Scalar* data(int w, int d) const {
    return data_ + w * width_stride() + d * depth_stride();
  }
  Scalar operator()(int w, int d) const { return *data(w, d); }
  Scalar& operator()(int w, int d) { return *data(w, d); }

  SideMap block(int start_width, int start_depth, int block_width,
                int block_depth) const {
    assert(start_width >= 0);
    assert(start_width + block_width <= width_);
    assert(start_depth >= 0);
    assert(start_depth + block_depth <= depth_);

    return SideMap(data(start_width, start_depth), block_width, block_depth,
                   stride_);
  }

 private:
  Scalar* data_;  // not owned.
  int width_, depth_, stride_;
};

// Generic (slow) packing code.
template <typename SrcMapType, typename KernelSideFormat>
class PackSideBlockImplGeneric {
 public:
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kLhsCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kLhsCells;
  static const int kCellDepth = CellFormat::kDepth;

  virtual ~PackSideBlockImplGeneric() {}

  PackSideBlockImplGeneric(PackedSideBlock<KernelSideFormat>* packed_side_block,
                           const SrcMapType& src_map)
      : packed_side_block_(packed_side_block), src_map_(src_map) {}

  // The public entry point to pack a block.
  void PackL2() {
    memset(packed_side_block_->rank_one_update(), 0,
           sizeof(std::int32_t) * packed_side_block_->params().l2_width);
    for (int d = 0; d < src_map_.depth();
         d += packed_side_block_->params().l1_depth) {
      int ds = std::min<int>(packed_side_block_->params().l1_depth,
                             src_map_.depth() - d);

      for (int w = 0; w < src_map_.width();
           w += packed_side_block_->params().l1_width) {
        int ws = std::min<int>(packed_side_block_->params().l1_width,
                               src_map_.width() - w);

        PackL1(w, ws, d, ds);
      }
    }
  }

  PackedSideBlock<KernelSideFormat>* packed_side_block() const {
    return packed_side_block_;
  }

  const SrcMapType& src_map() const { return src_map_; }

 protected:
  // PackRun packs only a run i.e. is the inner loop in the depth dimension.
  // This is what subclasses may override to provide optimized code paths.
  // Optimized implementations may still fall back to this generic code
  // to handle unaligned boundaries.
  virtual void PackRun(int start_width, int width, int start_depth, int depth) {
    for (int d = 0; d < depth; d += kDefaultCacheLineSize) {
      for (int w = 0; w < width; w++) {
        Prefetch(src_map_.data(start_width + w, start_depth + d));
      }
    }
    for (int d = 0; d < depth; d += kCellDepth) {
      // The next loop's boundary is kKernelWidth, not width,
      // because we always pack whole kernels so that the
      // compute stage doesn't need to worry about unaligned kernel sizes.
      for (int w = 0; w < +kKernelWidth; w += kCellWidth) {
        PackUnalignedCell(start_width + w, start_depth + d);
      }
    }
  }

 private:
  // The intermediate-level loops, between PackL2 and PackRun.
  void PackL1(int start_width, int width, int start_depth, int depth) {
    for (int w = 0; w < width; w += kKernelWidth) {
      int ws = std::min(+kKernelWidth, width - w);
      packed_side_block_->seek_run(start_width + w, start_depth);
      PackRun(start_width + w, ws, start_depth, depth);
    }
  }

  // Reference un-optimized implementation of the packing of a cell;
  // also serves as a fallback to handle unaligned edges.
  void PackUnalignedCell(int start_width, int start_depth) {
    std::uint8_t* dst_ptr = packed_side_block_->current_data();
    std::int32_t* dst_rank_one_update =
        packed_side_block_->rank_one_update() + start_width;
    std::int32_t dst_rank_one_update_multiplier =
        packed_side_block_->rank_one_update_multiplier();

    memset(dst_ptr, 0, sizeof(std::uint8_t) * CellFormat::kSize);

    if (start_width < src_map_.width() && start_depth < src_map_.depth()) {
      int width = std::min<int>(+kCellWidth, src_map_.width() - start_width);
      int depth = std::min<int>(+kCellDepth, src_map_.depth() - start_depth);
      auto src_block = src_map_.block(start_width, start_depth, width, depth);

      for (int w = 0; w < width; w++) {
        for (int d = 0; d < depth; d++) {
          std::uint8_t s = src_block(w, d);
          dst_ptr[OffsetIntoCell<CellFormat>(w, d)] = s;
          dst_rank_one_update[w] += s * dst_rank_one_update_multiplier;
        }
      }
    }

    packed_side_block_->seek_next_cell();
  }

  // The PackedSideBlock being packed, i.e. the 'destination'.
  PackedSideBlock<KernelSideFormat>* const packed_side_block_;

  // A map on the block of the original matrix block being packed,
  // i.e. the 'source'.
  const SrcMapType& src_map_;
};

// The packing code that we actually use. Defaults to using the above
// generic code; optimized paths can be inserted by specializing this
// template. See e.g. pack_neon.h.
template <typename SrcMapType, typename KernelSideFormat>
class PackSideBlockImpl
    : public PackSideBlockImplGeneric<SrcMapType, KernelSideFormat> {
 public:
  typedef PackSideBlockImplGeneric<SrcMapType, KernelSideFormat> Base;

  PackSideBlockImpl(PackedSideBlock<KernelSideFormat>* packed_side_block,
                    const SrcMapType& src_map)
      : Base(packed_side_block, src_map) {}
};

// Packs a block of the input LHS matrix, into a PackedSideBlock
template <typename KernelSideFormat, typename MatrixMapType>
void PackLhs(PackedSideBlock<KernelSideFormat>* dst, const MatrixMapType& src) {
  ScopedProfilingLabel label("pack LHS");
  static const SideMapOrder kSideMapOrder =
      MatrixMapType::kOrder == MapOrder::RowMajor ? SideMapOrder::WidthMajor
                                                  : SideMapOrder::DepthMajor;
  typedef typename MatrixMapType::Scalar Scalar;
  typedef SideMap<Scalar, kSideMapOrder> SideMapType;
  SideMapType src_side_map(src.data(), src.rows(), src.cols(), src.stride());
  typedef PackSideBlockImpl<SideMapType, KernelSideFormat> ImplType;
  ImplType impl(dst, src_side_map);
  impl.PackL2();
}

// Packs a block of the input RHS matrix, into a PackedSideBlock
template <typename KernelSideFormat, typename MatrixMapType>
void PackRhs(PackedSideBlock<KernelSideFormat>* dst, const MatrixMapType& src) {
  ScopedProfilingLabel label("pack RHS");
  static const SideMapOrder kSideMapOrder =
      MatrixMapType::kOrder == MapOrder::ColMajor ? SideMapOrder::WidthMajor
                                                  : SideMapOrder::DepthMajor;
  typedef typename MatrixMapType::Scalar Scalar;
  typedef SideMap<Scalar, kSideMapOrder> SideMapType;
  SideMapType src_side_map(src.data(), src.cols(), src.rows(), src.stride());
  typedef PackSideBlockImpl<SideMapType, KernelSideFormat> ImplType;
  ImplType impl(dst, src_side_map);
  impl.PackL2();
}

}  // namespace gemmlowp

#ifdef GEMMLOWP_NEON
#include "pack_neon.h"
#endif

#endif  // GEMMLOWP_INTERNAL_PACK_H_
