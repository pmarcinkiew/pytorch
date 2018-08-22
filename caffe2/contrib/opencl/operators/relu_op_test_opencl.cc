#include "caffe2/core/flags.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/relu_op.h"
#include "caffe2/core/workspace.h"
#include <gtest/gtest.h>
#include "../context.h"
#include "copy_ops.h"

#include <gtest/gtest.h>

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {

TEST(OpenCL, ReluOpTest) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);
  Workspace ws;
  {
    auto* t = ws.CreateBlob("input_cpu")->GetMutable<TensorCPU>();
    std::vector<float> cpuData(20, 1);
    std::vector<long int> shape{4, 5};
    caffe2::CPUContext cpuContext;
    auto src = TensorCPU(shape, cpuData, &cpuContext);
    t->Resize(5);
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
  def.set_name("relu");
  def.add_input("X");
  def.add_output("Y");
  def.mutable_device_option()->set_device_type(OPENCL);
  unique_ptr<OperatorBase> op(
                              new UnaryElementwiseOp<TensorTypes<float>,
                                OpenCLContext,
                                ReluFunctor<OpenCLContext>>(def, &ws)
                              );
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());
  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto Y = Yblob->GetMutable<Tensor<OpenCLContext>>();
  EXPECT_EQ(Y->size(), 20);
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());
  context.CopyBytes<OpenCLContext, CPUContext>(Y->size() * sizeof(float),
                                               (void*)yBuffer, (void*)&data[0]);

  for (int i = 0; i < Y->size(); ++i) {
    EXPECT_EQ(data[i], 1);
    printf("%d - %f\n", i, data[i]);
  }
}

}  // namespace caffe2
