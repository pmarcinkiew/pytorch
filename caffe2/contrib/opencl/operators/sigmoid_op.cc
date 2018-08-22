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

#include "caffe2/contrib/opencl/operators/sigmoid_op.h"
#include "caffe2/core/tensor.h"

#include "context.h"
#include "operator.h"

namespace caffe2 {

static constexpr const char* kSigmoidOp = R"CLC(
  kernel void K(
    const global float* X_in,
    global float* Y_out
  ) {
    const int idx_n = get_global_id(0);

    Y_out[idx_n] = 1.0 / (1.0 + exp(-X_in[idx_n]));

  }
)CLC";

template <typename T>
bool SigmoidOp<T>::RunOnDevice()
{
  auto& ctx = context_.GetSingleton();
  auto kernel = context_.BuildKernelCached(kSigmoidOp);

  const Tensor<OpenCLContext>& X = Input(0);
  Tensor<OpenCLContext>* Y = Output(0);

  const int N = X.size();

  CAFFE_ENFORCE_GE(N, 1, "N must be >= 1 ");

  Y->ResizeLike(X);

  cl::Buffer* xBuffer = (cl::Buffer*)X.data<float>();
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();

  OPENCL_CHECK(kernel.setArg(0, *xBuffer));
  OPENCL_CHECK(kernel.setArg(1, *yBuffer));

  cl::Event event;
  OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
    kernel,
    cl::NullRange,
    cl::NDRange(N),
    cl::NullRange,
    NULL,
    &event));
  event.wait();

  return true;
}

REGISTER_OPENCL_OPERATOR(
    SigmoidCL,
    SigmoidOp<float>);

// Input: X, output: Y
OPERATOR_SCHEMA(SigmoidCL)
  .NumInputs(1)
  .NumOutputs(1)
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC")
  .Input(0, "X", "1D input tensor")
  .Output(0, "Y", "1D output tensor");

}  // namespace caffe2
