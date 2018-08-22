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
//#include "context.h"
#include "operator.h"
#include "caffe2/operators/filler_op.h"

namespace caffe2 {
REGISTER_OPENCL_OPERATOR(ConstantFill, ConstantFillOp<OpenCLContext>);
}
