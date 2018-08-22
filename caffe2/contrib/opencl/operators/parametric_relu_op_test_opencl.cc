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


#include "caffe2/core/blob.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/common.h"
#include "caffe2/core/init.h"
#include "caffe2/core/workspace.h"
#include "caffe2/core/logging.h"
#include "../context.h"
#include "copy_ops.h"
#include "caffe2/operators/prelu_op.h"

#include <gtest/gtest.h>


namespace caffe2 {


void runTest(const int expected_size, std::vector<float>& input, std::vector<float>& w,
             std::vector<float>& expected_output,
             std::vector<long int>& expected_shape, std::vector<long int>& w_shape
              ) {
  DeviceOption option;
  OpenCLContext context(option);
  Workspace ws;

  //X:
  {
    caffe2::CPUContext cpuContext;
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    auto x_src_data_tensor = TensorCPU(expected_shape, input, &cpuContext);
    t->CopyFrom(x_src_data_tensor, &context);

    OperatorDef def_copy;
    def_copy.set_name("CopyToOpenCL");
    def_copy.add_input("X_cpu");
    def_copy.add_output("X");
    def_copy.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op(
       new CopyToOpenCLOp<OpenCLContext>(def_copy, &ws));
    EXPECT_TRUE(op->Run());
  }
  //W:
  {
    caffe2::CPUContext cpuContext;
    auto* t = ws.CreateBlob("W_cpu")->GetMutable<TensorCPU>();
    auto w_src_data_tensor = TensorCPU(w_shape, w, &cpuContext);
    t->CopyFrom(w_src_data_tensor, &context);

    OperatorDef def_copy;
    def_copy.set_name("CopyToOpenCL2");
    def_copy.add_input("W_cpu");
    def_copy.add_output("W");
    def_copy.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op(
       new CopyToOpenCLOp<OpenCLContext>(def_copy, &ws));
    EXPECT_TRUE(op->Run());
  }

  OperatorDef preludef;
  preludef.mutable_device_option()->set_device_type(OPENCL);
  preludef.set_name("PRelu");
  preludef.add_input("X");
  preludef.add_input("W");
  preludef.add_output("Y");
  unique_ptr<OperatorBase> preluop(new PReluOp<float, OpenCLContext>(preludef, &ws));
  EXPECT_TRUE(preluop->Run());

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

TEST(OpenCL, PRelu_test1) {
  const int expected_size = 4;
  std::vector<float> input({-1.0, -2.5, 0.0, 1.1});
  std::vector<float> w({2.0});
  std::vector<long int> w_shape({1});
  std::vector<float> expected_output({-2.0, -5.0, 0.0, 1.1});
  std::vector<long int> expected_shape({1, 1, 1, 4}); //1 image, 1 channel , h 1, w 4

  runTest(expected_size, input, w, expected_output, expected_shape, w_shape);
}

TEST(OpenCL, PRelu_test2) {
  const int expected_size = 4;
  std::vector<float> input({-1.0, -2.5, 0.1, -1.1});
  std::vector<float> w({2.0, 3.0});
  std::vector<long int> w_shape({2});
  std::vector<float> expected_output({-2.0, -5.0, 0.1, -3.3});  //1 image, 2 channels , h 1, w 2
  std::vector<long int> expected_shape({1, 2, 1, 2});

  runTest(expected_size, input, w, expected_output, expected_shape, w_shape);
}

TEST(OpenCL, PRelu_test3) {//shared weight:
  const int expected_size = 12;
  //2 image, 2 channels , h 3, w 1:
  std::vector<float> input({
                              //im0:
                              -1.0, -2.5, 1.5,
                              47.0, 0.0, -1.1,
                              //im1:
                              -2.0, -5.0, 3.0,
                              94.0, 0.0, -2.2,
                            }
                          );
  std::vector<float> w({2.0});
  std::vector<long int> w_shape({1});
  std::vector<float> expected_output({
                                        //im0:
                                        -2.0, -5.0, 1.5,
                                        47.0, 0.0, -2.2,
                                        //im1:
                                        -4.0, -10.0, 3.0,
                                        94.0, 0.0, -4.4,

                                     });
  std::vector<long int> expected_shape({2, 2, 3, 1});

  runTest(expected_size, input, w, expected_output, expected_shape, w_shape);
}

TEST(OpenCL, PRelu_test4) {//non shared weights:
  const int expected_size = 12;
  //2 image, 2 channels , h 3, w 1:
  std::vector<float> input({
                              //im0:
                              -1.0, -2.5, 1.5,
                              47.0, 0.0, -1.1,
                              //im1:
                              -2.0, -5.0, 3.0,
                              94.0, 0.0, -2.2,
                            }
                          );
  std::vector<float> w({2.0, -1.0});
  std::vector<long int> w_shape({2});
  std::vector<float> expected_output({
                                        //im0:
                                        -2.0, -5.0, 1.5,
                                        47.0, 0.0, 1.1,
                                        //im1:
                                        -4.0, -10.0, 3.0,
                                        94.0, 0.0, 2.2,

                                     });
  std::vector<long int> expected_shape({2, 2, 3, 1});

  runTest(expected_size, input, w, expected_output, expected_shape, w_shape);
}

}
