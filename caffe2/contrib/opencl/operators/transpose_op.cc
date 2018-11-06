
#include <algorithm>
#include <vector>

#include "caffe2/contrib/opencl/operator.h"
#include "caffe2/contrib/opencl/context.h"
#include "caffe2/operators/transpose_op.h"
#include "caffe2/contrib/opencl/math_opencl.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
  template <>
  bool TransposeOp<OpenCLContext>::RunOnDevice() {
    const Tensor& X = Input(0);
    Tensor* Y = Output(0);
    const int ndim = X.ndim();

    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.rbegin(), axes_.rend(), 0);
    }

    CAFFE_ENFORCE_EQ(ndim, axes_.size());

    const std::vector<int> X_dims(X.dims().cbegin(), X.dims().cend());
    std::vector<int> Y_dims(ndim);
    for (int i = 0; i < ndim; ++i) {
      Y_dims[i] = X_dims[axes_[i]];
    }
    Y->Resize(Y_dims);

    math::Transpose<cl::Buffer, OpenCLContext>(
        X_dims.size(),
        X_dims.data(),
        axes_.data(),
        reinterpret_cast<const cl::Buffer*>(X.data<float>()),
        reinterpret_cast<cl::Buffer*>(Y->mutable_data<float>()),
        &context_);

    return true;
  }

  REGISTER_OPENCL_OPERATOR(Transpose, TransposeOp<OpenCLContext>);
}
