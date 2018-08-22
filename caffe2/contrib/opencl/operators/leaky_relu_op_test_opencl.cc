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

#include <gtest/gtest.h>
#include "caffe2/operators/leaky_relu_op.h"
#include "copy_ops.h"
#include "../context.h"

namespace caffe2 {

TEST(OpenCL, leakyReluTest) {

  const int expected_size = 4;
  DeviceOption option;
  OpenCLContext context(option);
  Workspace ws;
  std::vector<float> input({-1.0, -2.5, 0.0, 1.0});
  std::vector<float> expected_output({-0.5, -1.25, 0.0, 1.0});
  std::vector<long int> expected_shape({expected_size});
  //X:
  {
    caffe2::CPUContext cpuContext;
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    auto x_src_data_tensor = TensorCPU(expected_shape, input, &cpuContext);
    t->CopyFrom(x_src_data_tensor, &context);

    OperatorDef def_copy1;
    def_copy1.set_name("CopyToOpenCL");
    def_copy1.add_input("X_cpu");
    def_copy1.add_output("X");
    def_copy1.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op1(
       new CopyToOpenCLOp<OpenCLContext>(def_copy1, &ws));
    EXPECT_TRUE(op1->Run());
  }

  OperatorDef leakydef;
  leakydef.mutable_device_option()->set_device_type(OPENCL);
  leakydef.set_name("Conv2");
  leakydef.add_input("X");
  leakydef.add_output("Y");
  Argument *arg = leakydef.add_arg();
  arg->set_name("alpha");
  arg->set_f(0.5);
  unique_ptr<OperatorBase> leakyop(new LeakyReluOp<float, OpenCLContext>(leakydef, &ws));
  EXPECT_TRUE(leakyop->Run());

  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto Y = Yblob->GetMutable<Tensor<OpenCLContext>>();

  EXPECT_EQ(Y->size(), expected_size);

  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());
  context.CopyBytes<OpenCLContext, CPUContext>(Y->size() * sizeof(float),
                                              (void*)yBuffer, (void*)&data[0]);

  for (int i = 0; i < Y->size(); ++i) {
    EXPECT_NEAR(data[i], expected_output[i], 0.001);
  }

}

}
