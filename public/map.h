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

// map.h: a minimalist view-existing-buffer-as-a-matrix class,
// which is how gemmlowp interfaces with external matrix data.

#ifndef GEMMLOWP_PUBLIC_MAP_H_
#define GEMMLOWP_PUBLIC_MAP_H_

#include "../internal/common.h"

namespace gemmlowp {

// The two storage orders allowed to map buffers as matrices: ColMajor
// means column-major, RowMajor means row-major.
enum class MapOrder { ColMajor, RowMajor };

// A MatrixMap is a view of an existing buffer as a matrix. It does not own
// the buffer.
template <typename tScalar, MapOrder tOrder>
class MatrixMap {
 public:
  typedef tScalar Scalar;
  static const MapOrder kOrder = tOrder;

 protected:
  Scalar* data_;  // not owned.
  int rows_, cols_, stride_;

 public:
  MatrixMap(Scalar* data, int rows, int cols, int stride)
      : data_(data), rows_(rows), cols_(cols), stride_(stride) {}

  MatrixMap(const MatrixMap& other)
      : data_(other.data_),
        rows_(other.rows_),
        cols_(other.cols_),
        stride_(other.stride_) {}

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int stride() const { return stride_; }
  int rows_stride() const { return kOrder == MapOrder::ColMajor ? 1 : stride_; }
  int cols_stride() const { return kOrder == MapOrder::RowMajor ? 1 : stride_; }
  Scalar* data() const { return data_; }
  Scalar* data(int row, int col) const {
    return data_ + row * rows_stride() + col * cols_stride();
  }
  Scalar operator()(int row, int col) const { return *data(row, col); }
  Scalar& operator()(int row, int col) { return *data(row, col); }

  MatrixMap block(int start_row, int start_col, int block_rows,
                  int block_cols) const {
    assert(start_row >= 0);
    assert(start_row + block_rows <= rows_);
    assert(start_col >= 0);
    assert(start_col + block_cols <= cols_);

    return MatrixMap(data(start_row, start_col), block_rows, block_cols,
                     stride_);
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_MAP_H_
