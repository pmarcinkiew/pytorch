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

#ifndef CAFFE2_CONTRIB_OPENCL_OPERATORS_UPSAMPLE_OP_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATORS_UPSAMPLE_OP_H_

#include "caffe2/contrib/opencl/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/types.h"
#include "caffe2/core/workspace.h"
#include <vector>
#include <cmath>

namespace caffe2 {
namespace {

static constexpr const char* kUpsampleOp = R"CLC(
  kernel void K(
    const global float* X_in,
    constant int *params,
    global float* Y_out
  ) {

    int y_ix = get_global_id(0); //this tells which output pixel is computed
    int c_ix = get_global_id(1); //this tells which image channel is being processed
    int n_ix = get_global_id(2); //this tells which image is being processed

    const int channels_count = params[0];
    const int X_h = params[1];
    const int X_w = params[2];
    const int Y_h = params[3];
    const int Y_w = params[4];
    const int stride = params[5];

    int y_col = y_ix % Y_w;
    int y_row = y_ix / Y_h;

    int in_index = n_ix * X_w * X_h * channels_count + c_ix * X_w * X_h + (y_row / stride) * X_w + y_col / stride;
    int out_index = n_ix * Y_w * Y_h * channels_count + c_ix * Y_w * Y_h + y_ix;

    Y_out[out_index] = X_in[in_index];
  }
  )CLC";

template <typename T>
class UpsampleOp final : public Operator<OpenCLContext> {
 public:
  UpsampleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<OpenCLContext>(operator_def, ws),
        order_(StringToStorageOrder(
                          OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
        stride_(OperatorBase::GetSingleArgument<int>("stride", 1)) {

        CAFFE_ENFORCE(!HasArgument("order") || order_ == NCHW, "NCHW only supported");
        CAFFE_ENFORCE(!HasArgument("stride") || stride_ >= 1, "Stride must be >= 1");

  }
  ~UpsampleOp() {}

  virtual bool RunOnDevice() override;

 private:
  StorageOrder order_; //for now squared NCHW only
  int stride_; //kernel window step
  // Input: X
  // Output: Y
  INPUT_TAGS(IN);
  OUTPUT_TAGS(OUT);
};

template <typename T>
bool UpsampleOp<T>::RunOnDevice()
{//NCHW order only

  auto& ctx = context_.GetSingleton();
  auto kernel = context_.BuildKernelCached(kUpsampleOp);

  const Tensor<OpenCLContext>& X = Input(IN);
  Tensor<OpenCLContext>* Y = Output(OUT);

  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);

  CAFFE_ENFORCE_GE(H, 1, "Input Height must be >= 1 ");
  CAFFE_ENFORCE_GE(W, 1, "Input Width must be >= 1 ");
  CAFFE_ENFORCE_GE(N, 1, "N must be >= 1 ");
  CAFFE_ENFORCE_GE(C, 1, "Channels count must be >= 1");
  CAFFE_ENFORCE_LE(X.ndim(), 4, "Input size must be at most 2D (H x W)");

  int output_h = H * stride_;
  int output_w = W * stride_;

  CAFFE_ENFORCE_GE(output_h, 1, "Image height is too small. Calculated output height: ", output_h);
  CAFFE_ENFORCE_GE(output_w, 1, "Image width is too small. Calculated output width: ", output_w);

  Y->Resize(std::vector<int>({N, C, output_h, output_w}));

  cl::Buffer* xBuffer = (cl::Buffer*)X.data<float>();
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();


  OPENCL_CHECK(kernel.setArg(0, *xBuffer));

  int params[] = {C, H, W, output_h, output_w, stride_};
  cl::Buffer params_cl(ctx.context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY , sizeof(params), params);
  OPENCL_CHECK(kernel.setArg(1, params_cl));
  OPENCL_CHECK(kernel.setArg(2, *yBuffer));

  cl::Event event;
  OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
    kernel,
    cl::NullRange,
    cl::NDRange(output_h * output_w, C, N), //global dim is the output size x number of channels x number of images
    cl::NullRange,
    NULL,
    &event));
  event.wait();

  return true;
}


}
}
#endif /* CAFFE2_CONTRIB_OPENCL_OPERATORS_UPSAMPLE_OP_H_ */
