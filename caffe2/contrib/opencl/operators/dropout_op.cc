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

#include "caffe2/contrib/opencl/context.h"
#include "caffe2/contrib/opencl/operator.h"
#include "caffe2/operators/dropout_op.h"

namespace caffe2 {

template <>
bool DropoutOp<float, OpenCLContext>::RunOnDevice() {

  const Tensor<OpenCLContext>& X = Input(0);
  Tensor<OpenCLContext>* Y = Output(0);
  Y->ResizeLike(X);

  if (is_test_) {
    if (Y != &X) {

      cl::Buffer* xBuffer = (cl::Buffer*)X.data<float>();
      cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();

      context_.CopyCL(xBuffer, yBuffer, X.size() * sizeof(float));
    }
    return true;
  } else {
    CAFFE_THROW("Training mode not implemented.");
  }
}

REGISTER_OPENCL_OPERATOR(Dropout, DropoutOp<float, OpenCLContext>);
}
