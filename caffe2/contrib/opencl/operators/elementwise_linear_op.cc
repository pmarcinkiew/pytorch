/**
 * Copyright (c) 2018-present, Samsung Electronics
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/operators/elementwise_linear_op.h"
#include "caffe2/core/tensor.h"

#include "context.h"
#include "operator.h"

namespace caffe2 {

static constexpr const char* kElementwiseLinearOp = R"CLC(
kernel void K(
const global float* X_in,
const global float* a,
const global float* b,
global float* Y_out
) {
  const int idx_d = get_global_id(0);
  const int idx_n = get_global_id(1);
  const int D = get_global_size(0);

  Y_out[idx_n * D + idx_d] = a[idx_d] * X_in[idx_n * D + idx_d] + b[idx_d];
}
)CLC";

template<>
bool ElementwiseLinearOp<float, OpenCLContext>::RunOnDevice() {

  auto& ctx = context_.GetSingleton();
  auto kernel = context_.BuildKernelCached(kElementwiseLinearOp);

  const auto& X = Input(0);
  const auto& a = Input(1);
  const auto& b = Input(2);
  auto* Y = Output(0);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);

  CAFFE_ENFORCE_EQ(a.ndim(), 1, a.ndim());
  CAFFE_ENFORCE_EQ(a.dim(0), D, a.ndim());
  CAFFE_ENFORCE_EQ(b.ndim(), 1, b.ndim());
  CAFFE_ENFORCE_EQ(b.dim(0), D, b.ndim());

  Y->ResizeLike(X);

  cl::Buffer* xBuffer = (cl::Buffer*)X.data<float>();
  cl::Buffer* aBuffer = (cl::Buffer*)a.data<float>();
  cl::Buffer* bBuffer = (cl::Buffer*)b.data<float>();

  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();

  OPENCL_CHECK(kernel.setArg(0, *xBuffer));
  OPENCL_CHECK(kernel.setArg(1, *aBuffer));
  OPENCL_CHECK(kernel.setArg(2, *bBuffer));
  OPENCL_CHECK(kernel.setArg(3, *yBuffer));

  cl::Event event;
  OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
    kernel,
    cl::NullRange,
    cl::NDRange(D, N),
    cl::NullRange,
    NULL,
    &event));
  event.wait();

  return true;
}

REGISTER_OPENCL_OPERATOR(
  ElementwiseLinear,
  ElementwiseLinearOp<float, OpenCLContext>);

}  // namespace caffe2
