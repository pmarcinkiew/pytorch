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

#include "caffe2/operators/relu_op.h"

#include "caffe2/utils/math.h"

#include "context.h"
#include "operator.h"
#include <sys/time.h>

namespace caffe2 {

static constexpr const char* kRelu = R"CLC(
kernel void K(
  global float* a,
  global float* y,
  int N
){
  int index = get_global_id(0);
  if (index < N) {
    y[index] = fmax(a[index], 0);
  }
}
)CLC";

static constexpr const char* kRelu2 = R"CLC(
kernel void K(
  global float* a,
  global float* y,
  int N,
  int batch_id
){
  //int batch_id = 4;    
  int index = get_global_id(0);
  if (index < N) {
    for(int i=0; i<batch_id; i++)
      y[index + (N * i)] = fmax(a[index+ (N * i)], 0);
  }
}
)CLC";

template <>
bool ReluOp<float, OpenCLContext>::RunOnDevice() {
  auto kernel = context_.BuildKernelCached(kRelu2);
  auto& X = Input(0);
  auto* Y = Output(0);
  int batch_size = X.dim32(0);
  auto xBuffer = (cl::Buffer*)X.data<float>();
  Y->ResizeLike(X);
  auto yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  OPENCL_CHECK(kernel.setArg(0, *xBuffer));
  OPENCL_CHECK(kernel.setArg(1, *yBuffer));
  OPENCL_CHECK(kernel.setArg(2, (int)X.size()/batch_size));
  OPENCL_CHECK(kernel.setArg(3, batch_size));
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
  outstr << "ReluOp " << end << " cpu time delta: " << end - start;
  outstr << "X " << X.size() << " Y " << Y->size() << " batch " << batch_size;
  context_.LogProfilingInfo(event, outstr.str());

  return true;
}

REGISTER_OPENCL_OPERATOR(Relu, ReluOp<float, OpenCLContext>);

} // namespace caffe2

