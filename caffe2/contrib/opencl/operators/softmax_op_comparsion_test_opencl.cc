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

#include "caffe2/core/flags.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/softmax_op.h"
#include "caffe2/core/workspace.h"
#include <gtest/gtest.h>
#include "../context.h"
#include "copy_ops.h"

#include <gtest/gtest.h>
#include <math.h>

#include <cstdio>

CAFFE2_DECLARE_string(caffe_test_root);

#define TEST_DATA_SIZE 4352

namespace caffe2 {

TEST(OpenCL, SoftmaxOpTest) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);
  Workspace ws, ws2;
  {
    auto* t = ws.CreateBlob("input_cpu")->GetMutable<TensorCPU>();
    //std::vector<float> cpuData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<float> cpuData(TEST_DATA_SIZE, 1.0);
    std::vector<long int> shape{1, 2, 34, 64};
    caffe2::CPUContext cpuContext;
    auto src = TensorCPU(shape, cpuData, &cpuContext);
    t->Resize(shape);
    t->CopyFrom(src, &context);
    OperatorDef def;
    def.set_name("CopyToOpenCL");
    def.add_input("input_cpu");
    def.add_output("X");
    def.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op(
        new CopyToOpenCLOp<OpenCLContext>(def, &ws));
    EXPECT_TRUE(op->Run());
  }

  OperatorDef def;
  def.set_name("softmax");
  def.add_input("X");
  def.add_output("Y");
  def.mutable_device_option()->set_device_type(OPENCL);
  unique_ptr<OperatorBase> op(
      new SoftmaxOp<float, OpenCLContext>(def, &ws));
  EXPECT_NE(nullptr, op.get());;


  OperatorDef def2;
  def2.set_name("softmaxcpu");
  def2.add_input("X");
  def2.add_output("Y");
  def2.mutable_device_option()->set_device_type(CPU);
  unique_ptr<OperatorBase> op2(
      new SoftmaxOp<float, CPUContext>(def2, &ws));
  EXPECT_NE(nullptr, op2.get());;


  EXPECT_TRUE(op->Run());
  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto Y = Yblob->GetMutable<Tensor<OpenCLContext>>();
  EXPECT_EQ(Y->size(), TEST_DATA_SIZE);
  EXPECT_EQ(Y->dim32(0), 1);
  EXPECT_EQ(Y->dim32(1), 2);
  EXPECT_EQ(Y->dim32(2), 34);
  EXPECT_EQ(Y->dim32(3), 64);
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());
  context.CopyBytes<OpenCLContext, CPUContext>(Y->size() * sizeof(float),
                                               (void*)yBuffer, (void*)&data[0]);

  //EXPECT_TRUE(op2->Run());
  //Blob* Yblob2 = ws2.GetBlob("Y");
  //EXPECT_NE(nullptr, Yblob);



  //std::vector<float> ans = {0.034, 0.091, 0.249, 0.676, 0.032, 0.087, 0.237, 0.645};
  std::vector<float> ans(TEST_DATA_SIZE, 1.0);
  for (int i = 0; i < Y->size(); ++i) {
    EXPECT_NEAR(data[i], ans[i], 0.9998);
  }
}

}  // namespace caffe2
