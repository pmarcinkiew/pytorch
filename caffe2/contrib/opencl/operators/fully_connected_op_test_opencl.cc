
#include "caffe2/core/flags.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/fully_connected_op.h"
#include "caffe2/core/workspace.h"
#include <gtest/gtest.h>
#include "../context.h"
#include "copy_ops.h"

#include <gtest/gtest.h>

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {

TEST(OpenCL, FullyConnectedOpTest) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);
  Workspace ws;
  {
    auto* t = ws.CreateBlob("input_cpu_X")->GetMutable<TensorCPU>();
    std::vector<float> cpuData(4, 1);
    cpuData[0] = 0;
    std::vector<long int> shape{2, 2};
    caffe2::CPUContext cpuContext;
    auto src = TensorCPU(shape, cpuData, &cpuContext);
    t->Resize(shape);
    t->CopyFrom(src, &context);
    OperatorDef def;
    def.set_name("CopyToOpenCL");
    def.add_input("input_cpu_X");
    def.add_output("X");
    def.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op(
        new CopyToOpenCLOp<OpenCLContext>(def, &ws));
    EXPECT_TRUE(op->Run());
  }
  {
    auto* t = ws.CreateBlob("input_cpu_W")->GetMutable<TensorCPU>();
    std::vector<float> cpuData(4, 1);
    std::vector<long int> shape{2 ,2};
    caffe2::CPUContext cpuContext;
    auto src = TensorCPU(shape, cpuData, &cpuContext);
    t->Resize(shape);
    t->CopyFrom(src, &context);
    OperatorDef def;
    def.set_name("CopyToOpenCL");
    def.add_input("input_cpu_W");
    def.add_output("W");
    def.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op(
        new CopyToOpenCLOp<OpenCLContext>(def, &ws));
    EXPECT_TRUE(op->Run());
  }
  {
    auto* t = ws.CreateBlob("input_cpu_b")->GetMutable<TensorCPU>();
    std::vector<float> cpuData(2, 1);
    std::vector<long int> shape{2};
    caffe2::CPUContext cpuContext;
    auto src = TensorCPU(shape, cpuData, &cpuContext);
    t->Resize(shape);
    t->CopyFrom(src, &context);
    OperatorDef def;
    def.set_name("CopyToOpenCL");
    def.add_input("input_cpu_b");
    def.add_output("b");
    def.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op(
        new CopyToOpenCLOp<OpenCLContext>(def, &ws));
    EXPECT_TRUE(op->Run());
  }

  OperatorDef def;
  def.set_name("fc");
  def.add_input("X");
  def.add_input("W");
  def.add_input("b");
  def.add_output("Y");
  def.mutable_device_option()->set_device_type(OPENCL);
  unique_ptr<OperatorBase> op(
      new FullyConnectedOp<OpenCLContext>(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());
  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto Y = Yblob->GetMutable<Tensor<OpenCLContext>>();
  EXPECT_EQ(Y->size(), 4);
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());
  context.CopyBytes<OpenCLContext, CPUContext>(Y->size() * sizeof(float),
                                               (void*)yBuffer, (void*)&data[0]);
  EXPECT_EQ(data[0], 2);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);
  EXPECT_EQ(data[3], 3);
}

TEST(OpenCL, FullyConnectedBatchedOpTest2) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);
  Workspace ws;
  {
    auto* t = ws.CreateBlob("input_cpu_X")->GetMutable<TensorCPU>();
    std::vector<float> cpuData(18, 1);
    cpuData[0] = 1; cpuData[1] = 2; cpuData[2] = 3;
    cpuData[3] = 2; cpuData[4] = 8; cpuData[5] = 1;
    cpuData[6] = 5; cpuData[7] = 2; cpuData[8] = 4;

    cpuData[9] = 1; cpuData[10] = 2; cpuData[11] = 3;
    cpuData[12] = 2; cpuData[13] = 8; cpuData[14] = 1;
    cpuData[15] = 5; cpuData[16] = 2; cpuData[17] = 4;

    std::vector<long int> shape{2,3,3};
    caffe2::CPUContext cpuContext;
    auto src = TensorCPU(shape, cpuData, &cpuContext);
    t->Resize(shape);
    t->CopyFrom(src, &context);
    OperatorDef def;
    def.set_name("CopyToOpenCL");
    def.add_input("input_cpu_X");
    def.add_output("X");
    def.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op(
        new CopyToOpenCLOp<OpenCLContext>(def, &ws));
    EXPECT_TRUE(op->Run());
  }
  {
    auto* t = ws.CreateBlob("input_cpu_W")->GetMutable<TensorCPU>();
    std::vector<float> cpuData(18, 1);
    cpuData[0] = 2; cpuData[1] = 2; cpuData[2] = 2;
    cpuData[3] = 2; cpuData[4] = 2; cpuData[5] = 2;
    cpuData[6] = 2; cpuData[7] = 2; cpuData[8] = 2;

    cpuData[9] = 2; cpuData[10] = 2; cpuData[11] = 2;
    cpuData[12] = 2; cpuData[13] = 2; cpuData[14] = 2;
    cpuData[15] = 2; cpuData[16] = 2; cpuData[17] = 2;

    std::vector<long int> shape{2 ,3, 3};
    caffe2::CPUContext cpuContext;
    auto src = TensorCPU(shape, cpuData, &cpuContext);
    t->Resize(shape);
    t->CopyFrom(src, &context);
    OperatorDef def;
    def.set_name("CopyToOpenCL");
    def.add_input("input_cpu_W");
    def.add_output("W");
    def.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op(
        new CopyToOpenCLOp<OpenCLContext>(def, &ws));
    EXPECT_TRUE(op->Run());
  }
  {
    auto* t = ws.CreateBlob("input_cpu_b")->GetMutable<TensorCPU>();
    std::vector<float> cpuData(6, 1);
    std::vector<long int> shape{2,3};
    caffe2::CPUContext cpuContext;
    auto src = TensorCPU(shape, cpuData, &cpuContext);
    t->Resize(shape);
    t->CopyFrom(src, &context);
    OperatorDef def;
    def.set_name("CopyToOpenCL");
    def.add_input("input_cpu_b");
    def.add_output("b");
    def.mutable_device_option()->set_device_type(OPENCL);
    unique_ptr<OperatorBase> op(
        new CopyToOpenCLOp<OpenCLContext>(def, &ws));
    EXPECT_TRUE(op->Run());
  }

  OperatorDef def;
  def.set_name("fc");
  def.add_input("X");
  def.add_input("W");
  def.add_input("b");
  def.add_output("Y");
  def.mutable_device_option()->set_device_type(OPENCL);
  unique_ptr<OperatorBase> op(
      new FullyConnectedBatchedOp<OpenCLContext>(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());

  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto Y = Yblob->GetMutable<Tensor<OpenCLContext>>();
  EXPECT_EQ(Y->size(), 18);
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(18);
  context.CopyBytes<OpenCLContext, CPUContext>(18 * sizeof(float),
                                               (void*)yBuffer, (void*)data.data());
  EXPECT_EQ(data[0], 13);
  EXPECT_EQ(data[1], 13);
  EXPECT_EQ(data[2], 13);

  EXPECT_EQ(data[3], 23);
  EXPECT_EQ(data[4], 23);
  EXPECT_EQ(data[5], 23);

  EXPECT_EQ(data[9], 13);
  EXPECT_EQ(data[10], 13);
  EXPECT_EQ(data[11], 13);

  EXPECT_EQ(data[12], 23);
  EXPECT_EQ(data[13], 23);
  EXPECT_EQ(data[14], 23);
}

}  // namespace caffe2
