
#include "caffe2/contrib/opencl/operator.h"
#include "caffe2/operators/utility_ops.h"
#include <gtest/gtest.h>
#include "copy_ops.h"
#include "../context.h"
#include <string>

namespace caffe2 {


void test_helper(const std::vector<std::vector<float>> &inputs,
                const std::vector<long int> &shape,
                const std::vector<float> &expected_output
                )
{
  DeviceOption option;
  caffe2::OpenCLContext context(option);
  Workspace ws;

  OperatorDef elemwise_def;
  elemwise_def.set_name("elemwisesum");
  elemwise_def.add_output("Y");
  elemwise_def.mutable_device_option()->set_device_type(OPENCL);

  for(int i = 0; i < inputs.size(); i++) {
    auto* t = ws.CreateBlob("input_cpu" + std::to_string(i))->GetMutable<TensorCPU>();
    CPUContext cpuContext;
    auto src = TensorCPU(shape, inputs[i], &cpuContext);
    t->CopyFrom(src, &context);
    OperatorDef def;
    def.set_name("CopyToOpenCL");
    def.add_input("input_cpu" + std::to_string(i));
    def.add_output("X" + std::to_string(i));
    def.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op(
        new CopyToOpenCLOp<OpenCLContext>(def, &ws));
    EXPECT_TRUE(op->Run());

    elemwise_def.add_input("X" + std::to_string(i));
  }

  unique_ptr<OperatorBase> op(new SumOp<OpenCLContext>(elemwise_def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());

  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto Y = Yblob->GetMutable<Tensor<OpenCLContext>>();
  EXPECT_EQ(Y->size(), expected_output.size());
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());
  context.CopyBytes<OpenCLContext, CPUContext>(Y->size() * sizeof(float),
                                               (void*)yBuffer, (void*)&data[0]);

  for (int i = 0; i < Y->size(); ++i) {
    EXPECT_NEAR(data[i], expected_output[i], 0.001);
  }
}

TEST(OpenCL, ElemwiseSumOpTest1) {
  std::vector<float> cpuData1{0.0, 1.0, 2.0, -3.0, -4.0};
  std::vector<long int> shape{5};
  std::vector<float> expected_output{0.0, 1.0, 2.0, -3.0, -4.0};
  const std::vector<std::vector<float>> inputs{cpuData1};

  test_helper(inputs, shape,expected_output);
}

TEST(OpenCL, ElemwiseSumOpTest2) {
  std::vector<float> cpuData1{0.0, 1.0, 2.0, -3.0, -4.0};
  std::vector<float> cpuData2{0.0, 1.0, 2.0, -3.0, -4.0};
  std::vector<long int> shape{5};
  std::vector<float> expected_output{0.0, 2.0, 4.0, -6.0, -8.0};
  const std::vector<std::vector<float>> inputs{cpuData1, cpuData2};

  test_helper(inputs, shape,expected_output);

}

TEST(OpenCL, ElemwiseSumOpTest3) {

  std::vector<float> cpuData1{0.0, 1.0, 2.0, -3.0, -4.0};
  std::vector<float> cpuData2{0.0, 1.0, 2.0, -3.0, -4.0};
  std::vector<float> cpuData3{0.0, 1.0, 2.0, -3.0, -4.0};
  std::vector<long int> shape{5};
  std::vector<float> expected_output{0.0, 3.0, 6.0, -9.0, -12.0};
  const std::vector<std::vector<float>> inputs{cpuData1, cpuData2, cpuData3};

  test_helper(inputs, shape,expected_output);
}

}
