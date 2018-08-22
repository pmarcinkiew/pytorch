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
#include "caffe2/operators/elementwise_linear_op.h"
#include "copy_ops.h"
#include "../context.h"

namespace caffe2 {

TEST(OpenCL, ElemwiseLinearOpTest) {

  DeviceOption option;
  OpenCLContext context(option);
  Workspace ws;
  std::vector<long int> in_shape({2, 9});
  std::vector<float> input({//data0:
                            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,

                            //data1:
                            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0
                          });
  std::vector<float> a({
                            2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0
                      });
  std::vector<float> b({
                            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
                      });
  std::vector<float> expected_output({
                            //data0:
                            2.1, 0.2, 6.3, 0.4, 10.5, 0.6, 14.7, 0.8, 18.9,

                            //data1:
                            20.1, 0.2, 24.3, 0.4, 28.5, 0.6, 32.7, 0.8, 36.9
  });
  std::vector<long int> expected_shape({2, 9});
  //X:
  {
    caffe2::CPUContext cpuContext;
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    auto x_src_data_tensor = TensorCPU(in_shape, input, &cpuContext);
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

  //a:
  {
    caffe2::CPUContext cpuContext;
    auto* t = ws.CreateBlob("a_cpu")->GetMutable<TensorCPU>();
    auto a_src_data_tensor = TensorCPU({9}, a, &cpuContext);
    t->CopyFrom(a_src_data_tensor, &context);

    OperatorDef def_copy1;
    def_copy1.set_name("CopyToOpenCL");
    def_copy1.add_input("a_cpu");
    def_copy1.add_output("a");
    def_copy1.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op1(
       new CopyToOpenCLOp<OpenCLContext>(def_copy1, &ws));
    EXPECT_TRUE(op1->Run());
  }

  //b:
  {
    caffe2::CPUContext cpuContext;
    auto* t = ws.CreateBlob("b_cpu")->GetMutable<TensorCPU>();
    auto b_src_data_tensor = TensorCPU({9}, b, &cpuContext);
    t->CopyFrom(b_src_data_tensor, &context);

    OperatorDef def_copy1;
    def_copy1.set_name("CopyToOpenCL");
    def_copy1.add_input("b_cpu");
    def_copy1.add_output("b");
    def_copy1.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op1(
       new CopyToOpenCLOp<OpenCLContext>(def_copy1, &ws));
    EXPECT_TRUE(op1->Run());
  }

  OperatorDef elemwisedef;
  elemwisedef.mutable_device_option()->set_device_type(OPENCL);
  elemwisedef.set_name("upsample");
  elemwisedef.add_input("X");
  elemwisedef.add_input("a");
  elemwisedef.add_input("b");
  elemwisedef.add_output("Y");
  unique_ptr<OperatorBase> elemwiseop(new ElementwiseLinearOp<float, OpenCLContext>(elemwisedef, &ws));
  EXPECT_TRUE(elemwiseop->Run());

  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto Y = Yblob->GetMutable<Tensor<OpenCLContext>>();

  EXPECT_EQ(Y->size(), 2 * 9);

  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());
  context.CopyBytes<OpenCLContext, CPUContext>(Y->size() * sizeof(float),
                                              (void*)yBuffer, (void*)&data[0]);

  for (int i = 0; i < Y->size(); ++i) {
    EXPECT_NEAR(data[i], expected_output[i], 0.001);
  }


}

}//caffe2
