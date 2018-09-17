#include <gtest/gtest.h>
#include "caffe2/contrib/opencl/operator.h"
#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include <google/protobuf/text_format.h>
#include "context.h"
#include "caffe2/contrib/opencl/net_async_dag_opencl.h"
#include "caffe2/contrib/opencl/math_opencl.h"
#include <iostream>

namespace caffe2 {

/*
                                        [d0]
                                          |
                                          \/
[data_cpu_0]->{CopyToOpenCL}->[data_0]->{Sum0}-----------------------------|       {CopyFromOpenCL}->[res_0]
                                                                           |       /\
                                        [d1]    [d3]                       |       |
                                          |       |                        |       |
                                          \/      \/                       \/      |
[data_cpu_1]->{CopyToOpenCL}->[data_1]->{Sum1}->{Sum3}-------------------->{Sum6}--|
                                                                           /\      |
                                                                           |       |
                                        [d2]    [d4]    [d5]               |       |
                                          |       |       |                |       |
                                          \/      \/      \/               |       \/
[data_cpu_2]->{CopyToOpenCL}->[data_2]->{Sum2}->{Sum4}->{Sum5}-------------|       {Sum7}->{CopyFromOpenCL}->[res_1]
                                                                                      /\
                                                                                      |
                                                                                    [d6]
*/


const auto multichain_spec = R"DOC(
      name: "example"
      type: "async_dag_opencl"

      op {
        output: "d0"
        type: "GivenTensorFill"
        arg {
          name: "shape"
          ints: 4
        }
        arg {
          name: "values"
          floats: 0.0
          floats: 0.0
          floats: 0.0
          floats: 0.0
        }
      }
      op {
        output: "d1"
        type: "GivenTensorFill"
        arg {
          name: "shape"
          ints: 4
        }
        arg {
          name: "values"
          floats: 1.0
          floats: 1.0
          floats: 1.0
          floats: 1.0
        }
      }
      op {
        output: "d2"
        type: "GivenTensorFill"
        arg {
          name: "shape"
          ints: 4
        }
        arg {
          name: "values"
          floats: 2.0
          floats: 2.0
          floats: 2.0
          floats: 2.0
        }
      }
      op {
        output: "d3"
        type: "GivenTensorFill"
        arg {
          name: "shape"
          ints: 4
        }
        arg {
          name: "values"
          floats: 3.0
          floats: 3.0
          floats: 3.0
          floats: 3.0
        }
      }
      op {
        output: "d4"
        type: "GivenTensorFill"
        arg {
          name: "shape"
          ints: 4
        }
        arg {
          name: "values"
          floats: 4.0
          floats: 4.0
          floats: 4.0
          floats: 4.0
        }
      }
      op {
        output: "d5"
        type: "GivenTensorFill"
        arg {
          name: "shape"
          ints: 4
        }
        arg {
          name: "values"
          floats: 5.0
          floats: 5.0
          floats: 5.0
          floats: 5.0
        }
      }
      op {
        output: "d6"
        type: "GivenTensorFill"
        arg {
          name: "shape"
          ints: 4
        }
        arg {
          name: "values"
          floats: -1.0
          floats: -1.0
          floats: -1.0
          floats: -1.0
        }
      }

      op {
        input: "data_cpu_0"
        output: "data_0"
        type: "CopyToOpenCL"
      }
      op {
        input: "data_cpu_1"
        output: "data_1"
        type: "CopyToOpenCL"
      }
      op {
        input: "data_cpu_2"
        output: "data_2"
        type: "CopyToOpenCL"
      }

      op {
        input: "data_0"
        input: "d0"
        output: "sum0"
        type: "Sum"
      }
      op {
        input: "data_1"
        input: "d1"
        output: "sum1"
        type: "Sum"
      }
      op {
        input: "data_2"
        input: "d2"
        output: "sum2"
        type: "Sum"
      }

      op {
        input: "sum1"
        input: "d3"
        output: "sum3"
        type: "Sum"
      }
      op {
        input: "sum2"
        input: "d4"
        output: "sum4"
        type: "Sum"
      }

      op {
        input: "sum4"
        input: "d5"
        output: "sum5"
        type: "Sum"
      }

      op {
        input: "sum0"
        input: "sum3"
        input: "sum5"
        output: "sum6"
        type: "Sum"
      }

      op {
        input: "sum6"
        output: "res_0"
        type: "CopyFromOpenCL"
      }
      op {
        input: "sum6"
        input: "d6"
        output: "sum7"
        type: "Sum"
      }

      op {
        input: "sum7"
        output: "res_1"
        type: "CopyFromOpenCL"
      }

      external_input: "data_cpu_0"
      external_input: "data_cpu_1"
      external_input: "data_cpu_2"
      external_output: "res_0"
      external_output: "res_1"
    )DOC";

TEST(NetTest, multichain) {

  Workspace ws;
  ws.CreateBlob("data_cpu_0")->GetMutable<TensorCPU>()->Resize(4);
  float *tmp = ws.GetBlob("data_cpu_0")->GetMutable<TensorCPU>()->mutable_data<float>();
  for(int i = 0; i < 4; i++)
    tmp[i] = 1.0;

  ws.CreateBlob("data_cpu_1")->GetMutable<TensorCPU>()->Resize(4);
  tmp = ws.GetBlob("data_cpu_1")->GetMutable<TensorCPU>()->mutable_data<float>();
  for(int i = 0; i < 4; i++)
    tmp[i] = 2.0;

  ws.CreateBlob("data_cpu_2")->GetMutable<TensorCPU>()->Resize(4);
  tmp = ws.GetBlob("data_cpu_2")->GetMutable<TensorCPU>()->mutable_data<float>();
  for(int i = 0; i < 4; i++)
    tmp[i] = 3.0;

  NetDef net_def;

  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(multichain_spec, &net_def));

  {
    net_def.mutable_device_option()->set_device_type(OPENCL);
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    ASSERT_TRUE(net);
    ASSERT_TRUE(net->Run());
  }

  {
    Tensor<CPUContext> * res_0 = ws.GetBlob("res_0")->GetMutable<TensorCPU>();
    Tensor<CPUContext> * res_1 = ws.GetBlob("res_1")->GetMutable<TensorCPU>();

    const float * res_0_data = res_0->data<float>();
    const float * res_1_data = res_1->data<float>();

    for(unsigned int i = 0;i < 4; i++) {
      EXPECT_EQ(res_0_data[i], 21.0);
      EXPECT_EQ(res_1_data[i], 20.0);
    }
  }
}


/*

[data_cpu_0]->{CopyToOpenCL}->[data_0]->{Sum0}-----------------------------|       {CopyFromOpenCL}->[res_0]
                                                                           |       /\
                                                                           |       |
                                                                           |       |
                                                                           \/      |
[data_cpu_1]->{CopyToOpenCL}->[data_1]->{Sum1}->{Sum3}-------------------->{Sum6}--|
                                                                           /\      |
                                                                           |       |
                                                                           |       |
                                                                           |       |
                                                                           |       \/
[data_cpu_2]->{CopyToOpenCL}->[data_2]->{Sum2}->{Sum4}->{Sum5}-------------|       {Sum7}->{CopyFromOpenCL}->[res_1]
                                                                                      /\
                                                                                      |
                                                                                    [d6]
*/


const auto multichain2_spec = R"DOC(
      name: "example"
      type: "async_dag_opencl"

      op {
        output: "d6"
        type: "GivenTensorFill"
        arg {
          name: "shape"
          ints: 4
        }
        arg {
          name: "values"
          floats: -1.0
          floats: -1.0
          floats: -1.0
          floats: -1.0
        }
      }

      op {
        input: "data_cpu_0"
        output: "data_0"
        type: "CopyToOpenCL"
      }
      op {
        input: "data_cpu_1"
        output: "data_1"
        type: "CopyToOpenCL"
      }
      op {
        input: "data_cpu_2"
        output: "data_2"
        type: "CopyToOpenCL"
      }

      op {
        input: "data_0"
        output: "sum0"
        type: "Sum"
      }
      op {
        input: "data_1"
        output: "sum1"
        type: "Sum"
      }
      op {
        input: "data_2"
        output: "sum2"
        type: "Sum"
      }

      op {
        input: "sum1"
        output: "sum3"
        type: "Sum"
      }
      op {
        input: "sum2"
        output: "sum4"
        type: "Sum"
      }

      op {
        input: "sum4"
        output: "sum5"
        type: "Sum"
      }

      op {
        input: "sum0"
        input: "sum3"
        input: "sum5"
        output: "sum6"
        type: "Sum"
      }

      op {
        input: "sum6"
        output: "res_0"
        type: "CopyFromOpenCL"
      }
      op {
        input: "sum6"
        input: "d6"
        output: "sum7"
        type: "Sum"
      }

      op {
        input: "sum7"
        output: "res_1"
        type: "CopyFromOpenCL"
      }

      external_input: "data_cpu_0"
      external_input: "data_cpu_1"
      external_input: "data_cpu_2"
      external_output: "res_0"
      external_output: "res_1"
    )DOC";

TEST(NetTest, multichain2) {

  Workspace ws;
  ws.CreateBlob("data_cpu_0")->GetMutable<TensorCPU>()->Resize(4);
  float *tmp = ws.GetBlob("data_cpu_0")->GetMutable<TensorCPU>()->mutable_data<float>();
  for(int i = 0; i < 4; i++)
    tmp[i] = 1.0;

  ws.CreateBlob("data_cpu_1")->GetMutable<TensorCPU>()->Resize(4);
  tmp = ws.GetBlob("data_cpu_1")->GetMutable<TensorCPU>()->mutable_data<float>();
  for(int i = 0; i < 4; i++)
    tmp[i] = 2.0;

  ws.CreateBlob("data_cpu_2")->GetMutable<TensorCPU>()->Resize(4);
  tmp = ws.GetBlob("data_cpu_2")->GetMutable<TensorCPU>()->mutable_data<float>();
  for(int i = 0; i < 4; i++)
    tmp[i] = 3.0;

  NetDef net_def;

  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(multichain2_spec, &net_def));

  {
    net_def.mutable_device_option()->set_device_type(OPENCL);
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    ASSERT_TRUE(net);
    ASSERT_TRUE(net->Run());

  }

  {
    Tensor<CPUContext> * res_0 = ws.GetBlob("res_0")->GetMutable<TensorCPU>();
    Tensor<CPUContext> * res_1 = ws.GetBlob("res_1")->GetMutable<TensorCPU>();

    const float * res_0_data = res_0->data<float>();
    const float * res_1_data = res_1->data<float>();

    for(unsigned int i = 0;i < 4; i++) {
      EXPECT_EQ(res_0_data[i], 6.0);
      EXPECT_EQ(res_1_data[i], 5.0);
    }

  }
}

/*

[data_cpu_0]->{CopyToOpenCL}->[data_0]->{Sum1}->{CopyFromOpenCL}->[res_0]

[data_cpu_1]->{CopyToOpenCL}->[data_1]->{Sum2}->{CopyFromOpenCL}->[res_1]

[data_cpu_2]->{CopyToOpenCL}->[data_2]->{Sum3}
[data_cpu_3]->{CopyToOpenCL}->[data_3]->{Sum4}
[data_cpu_4]->{CopyToOpenCL}->[data_4]->{Sum5}

*/

const auto fivechain_spec = R"DOC(
      name: "example"
      type: "async_dag_opencl"

      op {
        input: "data_cpu_0"
        output: "data_0"
        type: "CopyToOpenCL"
      }
      op {
        input: "data_cpu_1"
        output: "data_1"
        type: "CopyToOpenCL"
      }
      op {
        input: "data_cpu_2"
        output: "data_2"
        type: "CopyToOpenCL"
      }
      op {
        input: "data_cpu_3"
        output: "data_3"
        type: "CopyToOpenCL"
      }
      op {
        input: "data_cpu_4"
        output: "data_4"
        type: "CopyToOpenCL"
      }

      op {
        input: "data_0"
        output: "sum1"
        type: "Sum"
      }

      op {
        input: "data_1"
        output: "sum2"
        type: "Sum"
      }
      op {
        input: "data_2"
        output: "sum3"
        type: "Sum"
      }
      op {
        input: "data_3"
        output: "sum4"
        type: "Sum"
      }
      op {
        input: "data_4"
        output: "sum5"
        type: "Sum"
      }


      op {
        input: "sum1"
        output: "res_0"
        type: "CopyFromOpenCL"
      }
      op {
        input: "sum2"
        output: "res_1"
        type: "CopyFromOpenCL"
      }


      external_input: "data_cpu_0"
      external_input: "data_cpu_1"
      external_input: "data_cpu_2"
      external_input: "data_cpu_3"
      external_input: "data_cpu_4"
      external_output: "res_0"
      external_output: "res_1"
    )DOC";

TEST(NetTest, fivechain) {

  caffe2::GlobalInit();
  Workspace ws;

  ws.CreateBlob("data_cpu_0")->GetMutable<TensorCPU>()->Resize(4);
  float *tmp = ws.GetBlob("data_cpu_0")->GetMutable<TensorCPU>()->mutable_data<float>();
  for(int i = 0; i < 4; i++)
    tmp[i] = 1.0;

  ws.CreateBlob("data_cpu_1")->GetMutable<TensorCPU>()->Resize(4);
  tmp = ws.GetBlob("data_cpu_1")->GetMutable<TensorCPU>()->mutable_data<float>();
  for(int i = 0; i < 4; i++)
    tmp[i] = 2.0;

  ws.CreateBlob("data_cpu_2")->GetMutable<TensorCPU>()->Resize(4);
  tmp = ws.GetBlob("data_cpu_2")->GetMutable<TensorCPU>()->mutable_data<float>();
  for(int i = 0; i < 4; i++)
    tmp[i] = 3.0;

  ws.CreateBlob("data_cpu_3")->GetMutable<TensorCPU>()->Resize(4);
  tmp = ws.GetBlob("data_cpu_3")->GetMutable<TensorCPU>()->mutable_data<float>();
  for(int i = 0; i < 4; i++)
    tmp[i] = 4.0;

  ws.CreateBlob("data_cpu_4")->GetMutable<TensorCPU>()->Resize(4);
  tmp = ws.GetBlob("data_cpu_4")->GetMutable<TensorCPU>()->mutable_data<float>();
  for(int i = 0; i < 4; i++)
    tmp[i] = 5.0;

  NetDef net_def;

  CAFFE_ENFORCE(
      ::google::protobuf::TextFormat::ParseFromString(fivechain_spec, &net_def));

  {
    net_def.mutable_device_option()->set_device_type(OPENCL);
    std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    ASSERT_TRUE(net);
    ASSERT_TRUE(net->Run());

  }

  {
    Tensor<CPUContext> * res_0 = ws.GetBlob("res_0")->GetMutable<TensorCPU>();
    const float * res_0_data = res_0->data<float>();

    Tensor<CPUContext> * res_1 = ws.GetBlob("res_1")->GetMutable<TensorCPU>();
    const float * res_1_data = res_1->data<float>();

    for(unsigned int i = 0;i < 4; i++) {
      EXPECT_EQ(res_0_data[i], 1.0);
      EXPECT_EQ(res_1_data[i], 2.0);
    }

  }
}

}
