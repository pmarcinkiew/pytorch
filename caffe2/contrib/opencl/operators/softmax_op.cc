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

#include "caffe2/operators/softmax_op.h"
#include "caffe2/core/tensor.h"

#include "context.h"
#include "operator.h"
#include <cstdio>
#include <sys/time.h>

namespace caffe2 {
static constexpr const char* kSoftmaxExp = R"CLC(
kernel void K(
  global float* x,
  global float* z_exp,
  global float* softmax,
  const int N,
  const int D
){
  int idx = get_global_id(0);
  if (idx < N * D) {
    int dim = idx / D * D;
    // row-wise max
    float max = x[idx];
    for (int i = 0; i < D; i++)
        if (x[dim+i] > max)
            max = x[dim+i];
    barrier(CLK_GLOBAL_MEM_FENCE);
    // Subtract the max (for numerical reasons) & Exponentiation
    // TODO similar to cpu version, but why subtracting?
    z_exp[idx] = exp(x[idx] - max);
    barrier(CLK_GLOBAL_MEM_FENCE);
    float sum_z_exp = 0;
    for (int i = 0; i < D; i++)
        sum_z_exp += z_exp[dim + i];
    softmax[idx] = z_exp[idx] / sum_z_exp;
  }
}
)CLC";

template <>
bool SoftmaxOp<float, OpenCLContext>::RunOnDevice() {
  auto kernel = context_.BuildKernelCached(kSoftmaxExp);
  auto& X = Input(0);
  Tensor<OpenCLContext> S;
  S.ResizeLike(X);
  auto* Y = Output(0);
  int axis_ = OperatorBase::GetSingleArgument<int>("axis", 1);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);
  Y->ResizeLike(X);
  auto xBuffer = (cl::Buffer*)X.data<float>();
  auto sBuffer = (cl::Buffer*)S.mutable_data<float>();
  auto yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  OPENCL_CHECK(kernel.setArg(0, *xBuffer));
  OPENCL_CHECK(kernel.setArg(1, *sBuffer));
  OPENCL_CHECK(kernel.setArg(2, *yBuffer));
  OPENCL_CHECK(kernel.setArg(3, N));
  OPENCL_CHECK(kernel.setArg(4, D));
  auto& ctx = context_.GetSingleton();
  
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long start = tv.tv_sec * 1000000 + tv.tv_usec;

  cl::Event event;
  OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
    kernel,
    cl::NullRange,
    cl::NDRange(X.size()),
    cl::NullRange,
    NULL,
    &event));
  event.wait();

  gettimeofday(&tv, NULL);
  long long end = tv.tv_sec * 1000000 + tv.tv_usec;

  std::stringstream outstr;
  outstr << "SoftmaxOp_c2 "<< end << " cpu time delta: " << end - start;
  outstr << "N: " << N << " inputsize: " << D;
  context_.LogProfilingInfo(event, outstr.str());

  return true;
}

REGISTER_OPENCL_OPERATOR(Softmax, SoftmaxOp<float, OpenCLContext>);

}  // namespace caffe2
