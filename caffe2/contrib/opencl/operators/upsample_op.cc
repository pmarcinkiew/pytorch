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

#include "caffe2/contrib/opencl/operator.h"
#include "caffe2/contrib/opencl/operators/upsample_op.h"

namespace caffe2 {

REGISTER_OPENCL_OPERATOR(Upsample, UpsampleOp<float>);

OPERATOR_SCHEMA(Upsample)
    .NumInputs(1)
    .Input(0, "X", "The input data N x C x H x W")
    .NumOutputs(1)
    .Output(0, "Y", "The output N x C x Yh x Yw")
    .Arg("stride", "The output images will be stride times larger (in each direction) than the input. Default is 1.")
    .SetDoc(R"DOC(Upsample operator. Does not support output scaling.)DOC");

} // namespace caffe2
