
#include <caffe2/core/workspace.h>
#include <caffe2/core/init.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/core/net.h>
#include <gtest/gtest.h>
#include <vector>
#include "context.h"

const std::string init_net = R"(
name: "TestNetInit"
op {
  output: "data"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
    ints: 1
    ints: 5
    ints: 5
  }
  arg {
    name: "values"
    floats: 1.0
    floats: 2.5
    floats: 3.0
    floats: 4.5
    floats: 5.0
    floats: 6.5
    floats: 7.0
    floats: 8.5
    floats: 9.5
    floats: 10.0
    floats: 11.5
    floats: 12.0
    floats: 13.5
    floats: 14.0
    floats: 15.5
    floats: 16.0
    floats: 17.5
    floats: 18.0
    floats: 19.5
    floats: 20.0
    floats: 21.5
    floats: 22.0
    floats: 23.5
    floats: 24.0
    floats: 25.5
  }
}
op {
  output: "conv_w"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
    ints: 1
    ints: 3
    ints: 3
  }
  arg {
    name: "values"
    floats: 1.0
    floats: 2.0
    floats: 3.0
    floats: 4.0
    floats: 5.0
    floats: 6.0
    floats: 7.0
    floats: 8.0
    floats: 9.0
  }
}
op {
  output: "conv_b"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
  }
  arg {
    name: "values"
    floats: 0.5
  }
}

op {
  output: "fc_w"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
    ints: 1
    ints: 2
    ints: 2
  }
  arg {
    name: "values"
    floats: 666.0
    floats: 555.0
    floats: 444.0
    floats: 333.0
  }
}
op {
  output: "fc_b"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
  }
  arg {
    name: "values"
    floats: 111.0
  }
}

op {
  output: "prelu_w"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
  }
  arg {
    name: "values"
    floats: 10.0
  }
}

op {
  output: "sum_1"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
    ints: 1
  }
  arg {
    name: "values"
    floats: -1.0
  }
}
op {
  output: "sum_2"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
    ints: 1
  }
  arg {
    name: "values"
    floats: 2.0
  }
}

op {
  output: "a"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
  }
  arg {
    name: "values"
    floats: 1.0
  }
}
op {
  output: "b"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
  }
  arg {
    name: "values"
    floats: 8.0
  }
}

)";

const std::string predict_net = R"(
name: "TestNetPredict"
op {
  input: "data"
  input: "conv_w"
  input: "conv_b"
  output: "conv"
  type: "Conv"
  arg {
    name: "stride"
    i: 1
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "kernel"
    i: 3
  }
}

op {
  input: "conv"
  output: "pool"
  type: "MaxPool"
  arg {
    name: "stride"
    i: 2
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "kernel"
    i: 2
  }
  arg {
    name: "order"
    s: "NCHW"
  }
  arg {
    name: "legacy_pad"
    i: 3
  }
}

op {
  input: "pool"
  input: "fc_w"
  input: "fc_b"
  output: "fc"
  type: "FC"
}

op {
  input: "fc"
  output: "softmax"
  type: "Softmax"
}

op {
  input: "softmax"
  output: "relu"
  type: "Relu"
}

op {
  input: "relu"
  input: "prelu_w"
  output: "prelu"
  type: "PRelu"
}

op {
  input: "prelu"
  output: "lrelu"
  type: "LeakyRelu"
  arg {
    name: "alpha"
    f: -3.0
  }
}

op {
  input: "lrelu"
  output: "sigmoid"
  type: "Sigmoid"
}

op {
  input: "sigmoid"
  input: "sum_1"
  input: "sum_2"
  output: "sum"
  type: "Sum"
}

op {
  input: "sum"
  input: "a"
  input: "b"
  output: "linear"
  type: "ElementwiseLinear"
}

)";


namespace caffe2 {

void mnist_load_n_run(Workspace &ws, const DeviceType dt)
{
  NetDef init, predict;

  caffe2::TextFormat::ParseFromString(init_net, &init);
  caffe2::TextFormat::ParseFromString(predict_net, &predict);

  init.mutable_device_option()->set_device_type(dt);
  predict.mutable_device_option()->set_device_type(dt);

  //init net:
  NetBase *res = ws.CreateNet(init);
  EXPECT_NE(res, nullptr);
  ws.RunNet(res->Name());

  EXPECT_TRUE(ws.HasBlob("data"));
  EXPECT_TRUE(ws.HasBlob("conv_w"));
  EXPECT_TRUE(ws.HasBlob("conv_b"));
  EXPECT_TRUE(ws.HasBlob("fc_w"));
  EXPECT_TRUE(ws.HasBlob("fc_b"));
  EXPECT_TRUE(ws.HasBlob("prelu_w"));
  EXPECT_TRUE(ws.HasBlob("sum_1"));
  EXPECT_TRUE(ws.HasBlob("sum_2"));
  EXPECT_TRUE(ws.HasBlob("a"));
  EXPECT_TRUE(ws.HasBlob("b"));

  //predict net:
  res = ws.CreateNet(predict);
  EXPECT_NE(res, nullptr);
  ws.RunNet(res->Name());

  EXPECT_TRUE(ws.HasBlob("conv"));
  EXPECT_TRUE(ws.HasBlob("pool"));
  EXPECT_TRUE(ws.HasBlob("fc"));
  EXPECT_TRUE(ws.HasBlob("softmax"));
  EXPECT_TRUE(ws.HasBlob("relu"));
  EXPECT_TRUE(ws.HasBlob("prelu"));
  EXPECT_TRUE(ws.HasBlob("lrelu"));
  EXPECT_TRUE(ws.HasBlob("sigmoid"));
  EXPECT_TRUE(ws.HasBlob("sum"));
  EXPECT_TRUE(ws.HasBlob("linear"));

}

TEST(OpenCL, net_loading) {
#ifdef CAFFE2_USE_LITE_PROTO
  return;
#endif

  caffe2::GlobalInit();

  DeviceOption option;
  OpenCLContext context(option);
  Workspace ws_cpu, ws_opencl;

  mnist_load_n_run(ws_cpu, CPU);
  mnist_load_n_run(ws_opencl, OPENCL);

  auto cpu_output_tensor = ws_cpu.GetBlob("linear")->Get<TensorCPU>().Clone();
  const auto &data_cpu = cpu_output_tensor.data<float>();

  auto cl_output_tensor = ws_opencl.GetBlob("linear")->GetMutable<TensorCL>();
  cl::Buffer* cl_out_buffer = (cl::Buffer*)cl_output_tensor->mutable_data<float>();
  std::vector<float> data_cl(cl_output_tensor->size());

  context.CopyBytes<OpenCLContext, CPUContext>(cl_output_tensor->size() * sizeof(float),
                                              (void*)cl_out_buffer, (void*)&data_cl[0]);

  //outputs from both workspaces must have equal size...
  EXPECT_EQ(cpu_output_tensor.size(), cl_output_tensor->size());

  for(int i = 0; i < cpu_output_tensor.size(); i++)
  {//and must have similar values:
    EXPECT_NEAR(data_cpu[i], data_cl[i], 0.001);
  }
}

}
