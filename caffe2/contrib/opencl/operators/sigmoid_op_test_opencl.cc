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
#include "sigmoid_op.h"
#include "caffe2/core/workspace.h"
#include <gtest/gtest.h>
#include "../context.h"
#include "copy_ops.h"

#include <gtest/gtest.h>

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {

TEST(OpenCL, SigmoidOpTest) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);
  std::vector<float> cpuData{0.0, 1.0, 2.0, -3.0, -4.0};
  std::vector<long int> shape{5};
  std::vector<float> expected_output({
                              0.5, 0.731, 0.880, 0.047, 0.017
                          });
  Workspace ws;
  {
    auto* t = ws.CreateBlob("input_cpu")->GetMutable<TensorCPU>();
    caffe2::CPUContext cpuContext;
    auto src = TensorCPU(shape, cpuData, &cpuContext);
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
  def.set_name("sigmoid");
  def.add_input("X");
  def.add_output("Y");
  def.mutable_device_option()->set_device_type(OPENCL);
  unique_ptr<OperatorBase> op(new SigmoidOp<float>(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());

  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto Y = Yblob->GetMutable<Tensor<OpenCLContext>>();
  EXPECT_EQ(Y->size(), 5);
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());
  context.CopyBytes<OpenCLContext, CPUContext>(Y->size() * sizeof(float),
                                               (void*)yBuffer, (void*)&data[0]);

  for (int i = 0; i < Y->size(); ++i) {
    EXPECT_NEAR(data[i], expected_output[i], 0.001);
  }
}

}  // namespace caffe2
