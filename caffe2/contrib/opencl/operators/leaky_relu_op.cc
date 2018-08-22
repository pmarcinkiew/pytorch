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

#include "caffe2/operators/leaky_relu_op.h"

#include "context.h"
#include "operator.h"

namespace caffe2 {
static constexpr const char* kLeakyRelu = R"CLC(
kernel void K(
  const global float* X,
  global float* Y,
  const float alpha
) {
    int ix = get_global_id(0);
    float x = X[ix];

    if (x < 0.0)
     x *= alpha;

    Y[ix] = x;
  }
)CLC";

template <>
bool LeakyReluOp<float, OpenCLContext>::RunOnDevice() {

  const auto& X = Input(0);
  const int N = X.dim32(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

  auto& ctx = context_.GetSingleton();
  auto kernel = context_.BuildKernelCached(kLeakyRelu);

  cl::Buffer* xBuffer = (cl::Buffer*)X.data<float>();
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();

  OPENCL_CHECK(kernel.setArg(0, *xBuffer));
  OPENCL_CHECK(kernel.setArg(1, *yBuffer));
  OPENCL_CHECK(kernel.setArg(2, alpha_));

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

REGISTER_OPENCL_OPERATOR(LeakyRelu, LeakyReluOp<float, OpenCLContext>);

}
