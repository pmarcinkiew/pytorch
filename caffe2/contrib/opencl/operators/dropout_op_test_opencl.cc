
#include "caffe2/core/blob.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/common.h"
#include "caffe2/core/init.h"
#include "caffe2/core/workspace.h"
#include "caffe2/core/logging.h"
#include "../context.h"
#include "copy_ops.h"
#include "caffe2/operators/dropout_op.h"

#include <gtest/gtest.h>


namespace caffe2 {


void runTest(const int expected_size, std::vector<float>& input,
             std::vector<float>& expected_output
              ) {
  DeviceOption option;
  OpenCLContext context(option);
  Workspace ws;

  //X:
  {
    caffe2::CPUContext cpuContext;
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    auto x_src_data_tensor = TensorCPU({expected_size}, input, &cpuContext);
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

  OperatorDef dropoutdef;
  dropoutdef.mutable_device_option()->set_device_type(OPENCL);
  dropoutdef.set_name("Dropout");
  dropoutdef.add_input("X");
  dropoutdef.add_output("Y");
  Argument *arg = dropoutdef.add_arg();
  arg->set_name("is_test");
  arg->set_i(true);
  unique_ptr<OperatorBase> dropout(new DropoutOp<float, OpenCLContext>(dropoutdef, &ws));
  EXPECT_TRUE(dropout->Run());

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

TEST(OpenCL, Dropout_test1) {
  const int expected_size = 7;
  std::vector<float> input({0.0, 1.0, 3.0, 5.0, 10.1, 7.0, -1.0});

  runTest(expected_size, input, input);
}

}
