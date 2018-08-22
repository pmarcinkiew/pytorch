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
#include "upsample_op.h"
#include "copy_ops.h"
#include "../context.h"

namespace caffe2 {

TEST(OpenCL, leakyReluTest) {

  DeviceOption option;
  OpenCLContext context(option);
  Workspace ws;
  std::vector<long int> in_shape({2, 2, 3, 3});
  std::vector<float> input({//im0c0:
                            1.0, 2.0, 3.0,
                            4.0, 5.0, 6.0,
                            7.0, 8.0, 9.0,
                            //im0c1:
                            10.0, 11.0, 12.0,
                            13.0, 14.0, 15.0,
                            16.0, 17.0, 18.0,

                            //im1c0:
                            19.0, 20.0, 21.0,
                            22.0, 23.0, 24.0,
                            25.0, 26.0, 27.0,
                            //im1c1:
                            28.0, 29.0, 30.0,
                            31.0, 32.0, 33.0,
                            34.0, 35.0, 36.0
                          });
  std::vector<float> expected_output({//im0c0:
                            1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
                            1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
                            4.0, 4.0, 5.0, 5.0, 6.0, 6.0,
                            4.0, 4.0, 5.0, 5.0, 6.0, 6.0,
                            7.0, 7.0, 8.0, 8.0, 9.0, 9.0,
                            7.0, 7.0, 8.0, 8.0, 9.0, 9.0,
                            //im0c1:
                            10.0, 10.0, 11.0, 11.0, 12.0, 12.0,
                            10.0, 10.0, 11.0, 11.0, 12.0, 12.0,
                            13.0, 13.0, 14.0, 14.0, 15.0, 15.0,
                            13.0, 13.0, 14.0, 14.0, 15.0, 15.0,
                            16.0, 16.0, 17.0, 17.0, 18.0, 18.0,
                            16.0, 16.0, 17.0, 17.0, 18.0, 18.0,

                            //im1c0:
                            19.0, 19.0, 20.0, 20.0, 21.0, 21.0,
                            19.0, 19.0, 20.0, 20.0, 21.0, 21.0,
                            22.0, 22.0, 23.0, 23.0, 24.0, 24.0,
                            22.0, 22.0, 23.0, 23.0, 24.0, 24.0,
                            25.0, 25.0, 26.0, 26.0, 27.0, 27.0,
                            25.0, 25.0, 26.0, 26.0, 27.0, 27.0,
                            //im1c1:
                            28.0, 28.0, 29.0, 29.0, 30.0, 30.0,
                            28.0, 28.0, 29.0, 29.0, 30.0, 30.0,
                            31.0, 31.0, 32.0, 32.0, 33.0, 33.0,
                            31.0, 31.0, 32.0, 32.0, 33.0, 33.0,
                            34.0, 34.0, 35.0, 35.0, 36.0, 36.0,
                            34.0, 34.0, 35.0, 35.0, 36.0, 36.0
                        });
  std::vector<long int> expected_shape({2, 2, 6, 6});
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

  OperatorDef upsampledef;
  upsampledef.mutable_device_option()->set_device_type(OPENCL);
  upsampledef.set_name("upsample");
  upsampledef.add_input("X");
  upsampledef.add_output("Y");
  Argument *arg = upsampledef.add_arg();
  arg->set_name("stride");
  arg->set_i(2);
  unique_ptr<OperatorBase> upsampleop(new UpsampleOp<float>(upsampledef, &ws));
  EXPECT_TRUE(upsampleop->Run());

  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto Y = Yblob->GetMutable<Tensor<OpenCLContext>>();

  EXPECT_EQ(Y->size(), 2 * 2 * 6 * 6);

  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());
  context.CopyBytes<OpenCLContext, CPUContext>(Y->size() * sizeof(float),
                                              (void*)yBuffer, (void*)&data[0]);

  for (int i = 0; i < Y->size(); ++i) {
    EXPECT_NEAR(data[i], expected_output[i], 0.001);
  }

}

}
