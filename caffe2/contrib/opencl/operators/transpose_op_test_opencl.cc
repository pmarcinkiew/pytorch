
#include <caffe2/core/workspace.h>
#include <caffe2/core/init.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/core/net.h>
#include <gtest/gtest.h>
#include <vector>
#include "context.h"

const std::string init_net = R"(
name: "transpose_test_model_init"
op {
  output: "data_nchw1"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 3
    ints: 3
    ints: 2
    ints: 4
  }
  arg {
    name: "values"
  floats: 1
  floats: 2
  floats: 3
  floats: 4
  floats: 5
  floats: 6
  floats: 7
  floats: 8
  floats: 1.333
  floats: 2.333
  floats: 3.333
  floats: 4.333
  floats: 5.333
  floats: 6.333
  floats: 7.333
  floats: 8.333
  floats: 1.666
  floats: 2.666
  floats: 3.666
  floats: 4.666
  floats: 5.666
  floats: 6.666
  floats: 7.666
  floats: 8.666
  floats: 9
  floats: 10
  floats: 11
  floats: 12
  floats: 13
  floats: 14
  floats: 15
  floats: 16
  floats: 9.333
  floats: 10.333
  floats: 11.333
  floats: 12.333
  floats: 13.333
  floats: 14.333
  floats: 15.333
  floats: 16.333
  floats: 9.666
  floats: 10.666
  floats: 11.666
  floats: 12.666
  floats: 13.666
  floats: 14.666
  floats: 15.666
  floats: 16.666
  floats: 17
  floats: 18
  floats: 19
  floats: 20
  floats: 21
  floats: 22
  floats: 23
  floats: 24
  floats: 17.333
  floats: 18.333
  floats: 19.333
  floats: 20.333
  floats: 21.333
  floats: 22.333
  floats: 23.333
  floats: 24.333
  floats: 17.666
  floats: 18.666
  floats: 19.666
  floats: 20.666
  floats: 21.666
  floats: 22.666
  floats: 23.666
  floats: 24.666
  }
}
op {
  output: "expected_data_nhwc1"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 3
    ints: 2
    ints: 4
    ints: 3
  }
  arg {
    name: "values"
  floats: 1
  floats: 1.333
  floats: 1.666
  floats: 2
  floats: 2.333
  floats: 2.666
  floats: 3
  floats: 3.333
  floats: 3.666
  floats: 4
  floats: 4.333
  floats: 4.666
  floats: 5
  floats: 5.333
  floats: 5.666
  floats: 6
  floats: 6.333
  floats: 6.666
  floats: 7
  floats: 7.333
  floats: 7.666
  floats: 8
  floats: 8.333
  floats: 8.666
  floats: 9
  floats: 9.333
  floats: 9.666
  floats: 10
  floats: 10.333
  floats: 10.666
  floats: 11
  floats: 11.333
  floats: 11.666
  floats: 12
  floats: 12.333
  floats: 12.666
  floats: 13
  floats: 13.333
  floats: 13.666
  floats: 14
  floats: 14.333
  floats: 14.666
  floats: 15
  floats: 15.333
  floats: 15.666
  floats: 16
  floats: 16.333
  floats: 16.666
  floats: 17
  floats: 17.333
  floats: 17.666
  floats: 18
  floats: 18.333
  floats: 18.666
  floats: 19
  floats: 19.333
  floats: 19.666
  floats: 20
  floats: 20.333
  floats: 20.666
  floats: 21
  floats: 21.333
  floats: 21.666
  floats: 22
  floats: 22.333
  floats: 22.666
  floats: 23
  floats: 23.333
  floats: 23.666
  floats: 24
  floats: 24.333
  floats: 24.666
  }
}
op {
  output: "data_nhwc1"
  type: "ConstantFill"
  arg {
    name: "shape"
    ints: 3
    ints: 2
    ints: 4
    ints: 3
  }
}
op {
  output: "data_nchw2"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
    ints: 1
    ints: 1
    ints: 1
  }
  arg {
    name: "values"
  floats: 666.6

  }
}
op {
  output: "expected_data_nhwc2"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
    ints: 1
    ints: 1
    ints: 1
  }
  arg {
    name: "values"
  floats: 666.6
  }
}
op {
  output: "data_nhwc2"
  type: "ConstantFill"
  arg {
    name: "shape"
  floats: 1
  floats: 3
  floats: 4
  floats: 2
  }
}
op {
  output: "data_nchw3"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
    ints: 2
    ints: 2
    ints: 2
  }
  arg {
    name: "values"
  floats: 1
  floats: 2
  floats: 3
  floats: 4
  floats: 5
  floats: 6
  floats: 7
  floats: 8
  }
}
op {
  output: "expected_data_nhwc3"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 1
    ints: 2
    ints: 2
    ints: 2
  }
  arg {
    name: "values"
  floats: 1
  floats: 5
  floats: 2
  floats: 6
  floats: 3
  floats: 7
  floats: 4
  floats: 8
  }
}
op {
  output: "data_nhwc3"
  type: "ConstantFill"
  arg {
    name: "shape"
    ints: 1
  }
}
op {
  output: "data_nchw4"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 2
    ints: 1
    ints: 2
    ints: 2
  }
  arg {
    name: "values"
  floats: 1
  floats: 2
  floats: 3
  floats: 4
  floats: 5
  floats: 6
  floats: 7
  floats: 8
  }
}
op {
  output: "expected_data_nhwc4"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 2
    ints: 2
    ints: 2
    ints: 1
  }
  arg {
    name: "values"
  floats: 1
  floats: 2
  floats: 3
  floats: 4
  floats: 5
  floats: 6
  floats: 7
  floats: 8
  }
}
op {
  output: "data_nhwc4"
  type: "ConstantFill"
  arg {
    name: "shape"
    ints: 1
  }
}
op {
  output: "data_nchw5"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 2
    ints: 2
    ints: 1
    ints: 2
  }
  arg {
    name: "values"
  floats: 1
  floats: 2
  floats: 1.1
  floats: 2.1
  floats: 3
  floats: 4
  floats: 3.1
  floats: 4.1
  }
}
op {
  output: "expected_data_nhwc5"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 2
    ints: 1
    ints: 2
    ints: 2
  }
  arg {
    name: "values"
  floats: 1
  floats: 1.1
  floats: 2
  floats: 2.1
  floats: 3
  floats: 3.1
  floats: 4
  floats: 4.1
  }
}
op {
  output: "data_nhwc5"
  type: "ConstantFill"
  arg {
    name: "shape"
    ints: 1
  }
}
op {
  output: "data_nchw6"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 2
    ints: 2
    ints: 2
    ints: 1
  }
  arg {
    name: "values"
  floats: 1
  floats: 2
  floats: 1.1
  floats: 2.1
  floats: 3
  floats: 4
  floats: 3.1
  floats: 4.1
  }
}
op {
  output: "expected_data_nhwc6"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 2
    ints: 2
    ints: 1
    ints: 2
  }
  arg {
    name: "values"
  floats: 1
  floats: 1.1
  floats: 2
  floats: 2.1
  floats: 3
  floats: 3.1
  floats: 4
  floats: 4.1
  }
}
op {
  output: "data_nhwc6"
  type: "ConstantFill"
  arg {
    name: "shape"
    ints: 1
  }
}
op {
  output: "data_nchw7"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 2
    ints: 3
    ints: 4
    ints: 5
  }
  arg {
    name: "values"
  floats: 1
  floats: 2
  floats: 3
  floats: 4
  floats: 5
  floats: 6
  floats: 7
  floats: 8
  floats: 9
  floats: 10
  floats: 11
  floats: 12
  floats: 13
  floats: 14
  floats: 15
  floats: 16
  floats: 17
  floats: 18
  floats: 19
  floats: 20
  floats: 1.1
  floats: 2.1
  floats: 3.1
  floats: 4.1
  floats: 5.1
  floats: 6.1
  floats: 7.1
  floats: 8.1
  floats: 9.1
  floats: 10.1
  floats: 11.1
  floats: 12.1
  floats: 13.1
  floats: 14.1
  floats: 15.1
  floats: 16.1
  floats: 17.1
  floats: 18.1
  floats: 19.1
  floats: 20.1
  floats: 1.2
  floats: 2.2
  floats: 3.2
  floats: 4.2
  floats: 5.2
  floats: 6.2
  floats: 7.2
  floats: 8.2
  floats: 9.2
  floats: 10.2
  floats: 11.2
  floats: 12.2
  floats: 13.2
  floats: 14.2
  floats: 15.2
  floats: 16.2
  floats: 17.2
  floats: 18.2
  floats: 19.2
  floats: 20.2
  floats: 21
  floats: 22
  floats: 23
  floats: 24
  floats: 25
  floats: 26
  floats: 27
  floats: 28
  floats: 29
  floats: 30
  floats: 31
  floats: 32
  floats: 33
  floats: 34
  floats: 35
  floats: 36
  floats: 37
  floats: 38
  floats: 39
  floats: 40
  floats: 21.1
  floats: 22.1
  floats: 23.1
  floats: 24.1
  floats: 25.1
  floats: 26.1
  floats: 27.1
  floats: 28.1
  floats: 29.1
  floats: 30.1
  floats: 31.1
  floats: 32.1
  floats: 33.1
  floats: 34.1
  floats: 35.1
  floats: 36.1
  floats: 37.1
  floats: 38.1
  floats: 39.1
  floats: 40.1
  floats: 21.2
  floats: 22.2
  floats: 23.2
  floats: 24.2
  floats: 25.2
  floats: 26.2
  floats: 27.2
  floats: 28.2
  floats: 29.2
  floats: 30.2
  floats: 31.2
  floats: 32.2
  floats: 33.2
  floats: 34.2
  floats: 35.2
  floats: 36.2
  floats: 37.2
  floats: 38.2
  floats: 39.2
  floats: 40.2
  }
}
op {
  output: "expected_data_nhwc7"
  type: "GivenTensorFill"
  arg {
    name: "shape"
    ints: 2
    ints: 4
    ints: 5
    ints: 3
  }
  arg {
    name: "values"
  floats: 1
  floats: 1.1
  floats: 1.2
  floats: 2
  floats: 2.1
  floats: 2.2
  floats: 3
  floats: 3.1
  floats: 3.2
  floats: 4
  floats: 4.1
  floats: 4.2
  floats: 5
  floats: 5.1
  floats: 5.2
  floats: 6
  floats: 6.1
  floats: 6.2
  floats: 7
  floats: 7.1
  floats: 7.2
  floats: 8
  floats: 8.1
  floats: 8.2
  floats: 9
  floats: 9.1
  floats: 9.2
  floats: 10
  floats: 10.1
  floats: 10.2
  floats: 11
  floats: 11.1
  floats: 11.2
  floats: 12
  floats: 12.1
  floats: 12.2
  floats: 13
  floats: 13.1
  floats: 13.2
  floats: 14
  floats: 14.1
  floats: 14.2
  floats: 15
  floats: 15.1
  floats: 15.2
  floats: 16
  floats: 16.1
  floats: 16.2
  floats: 17
  floats: 17.1
  floats: 17.2
  floats: 18
  floats: 18.1
  floats: 18.2
  floats: 19
  floats: 19.1
  floats: 19.2
  floats: 20
  floats: 20.1
  floats: 20.2
  floats: 21
  floats: 21.1
  floats: 21.2
  floats: 22
  floats: 22.1
  floats: 22.2
  floats: 23
  floats: 23.1
  floats: 23.2
  floats: 24
  floats: 24.1
  floats: 24.2
  floats: 25
  floats: 25.1
  floats: 25.2
  floats: 26
  floats: 26.1
  floats: 26.2
  floats: 27
  floats: 27.1
  floats: 27.2
  floats: 28
  floats: 28.1
  floats: 28.2
  floats: 29
  floats: 29.1
  floats: 29.2
  floats: 30
  floats: 30.1
  floats: 30.2
  floats: 31
  floats: 31.1
  floats: 31.2
  floats: 32
  floats: 32.1
  floats: 32.2
  floats: 33
  floats: 33.1
  floats: 33.2
  floats: 34
  floats: 34.1
  floats: 34.2
  floats: 35
  floats: 35.1
  floats: 35.2
  floats: 36
  floats: 36.1
  floats: 36.2
  floats: 37
  floats: 37.1
  floats: 37.2
  floats: 38
  floats: 38.1
  floats: 38.2
  floats: 39
  floats: 39.1
  floats: 39.2
  floats: 40
  floats: 40.1
  floats: 40.2
  }
}
op {
  output: "data_nhwc7"
  type: "ConstantFill"
  arg {
    name: "shape"
    ints: 1
  }
}
)";

const std::string predict_net = R"(
name: "transpose_test_model_predict"
op {
  input: "data_nchw1"
  output: "data_nhwc1"
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "data_nchw2"
  output: "data_nhwc2"
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "data_nchw3"
  output: "data_nhwc3"
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "data_nchw4"
  output: "data_nhwc4"
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "data_nchw5"
  output: "data_nhwc5"
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "data_nchw6"
  output: "data_nhwc6"
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "data_nchw7"
  output: "data_nhwc7"
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}

external_input: "data_nchw1"
external_output: "data_nhwc1"
external_input: "data_nchw2"
external_output: "data_nhwc2"
external_input: "data_nchw3"
external_output: "data_nhwc3"
external_input: "data_nchw4"
external_output: "data_nhwc4"
external_input: "data_nchw5"
external_output: "data_nhwc5"
external_input: "data_nchw6"
external_output: "data_nhwc6"
external_input: "data_nchw7"
external_output: "data_nhwc7"
)";

namespace caffe2 {

void load_n_run(Workspace &ws)
{
  NetDef init, predict;

  caffe2::TextFormat::ParseFromString(init_net, &init);
  caffe2::TextFormat::ParseFromString(predict_net, &predict);

  init.mutable_device_option()->set_device_type(PROTO_OPENCL);
  predict.mutable_device_option()->set_device_type(PROTO_OPENCL);

  //init net:
  NetBase *res = ws.CreateNet(init);
  EXPECT_NE(res, nullptr);
  ws.RunNet(res->Name());

  //predict net:
  res = ws.CreateNet(predict);
  EXPECT_NE(res, nullptr);
  ws.RunNet(res->Name());
}


void compare_tensors_by_name(const std::string &t_expected, const std::string &t_result, Workspace &ws_opencl, OpenCLContext &context) {

  auto cl_expected_tensor = ws_opencl.GetBlob(t_expected)->Get<TensorCL>();
  auto cl_expected_out_buffer = reinterpret_cast<const cl::Buffer*>(cl_expected_tensor.data<float>());
  std::vector<float> expected_data_cl(cl_expected_tensor.size());

  context.CopyBytes<OpenCLContext, CPUContext>(cl_expected_tensor.size() * sizeof(float), cl_expected_out_buffer, &expected_data_cl[0]);

  auto cl_output_tensor = ws_opencl.GetBlob(t_result)->Get<TensorCL>();
  auto cl_out_buffer = reinterpret_cast<const cl::Buffer*>(cl_output_tensor.data<float>());
  std::vector<float> data_cl(cl_output_tensor.size());

  context.CopyBytes<OpenCLContext, CPUContext>(cl_output_tensor.size() * sizeof(float), cl_out_buffer, &data_cl[0]);

  EXPECT_EQ(cl_expected_tensor.size(), cl_output_tensor.size());

  EXPECT_EQ(cl_expected_tensor.dim(0), cl_output_tensor.dim(0));
  EXPECT_EQ(cl_expected_tensor.dim(1), cl_output_tensor.dim(1));
  EXPECT_EQ(cl_expected_tensor.dim(2), cl_output_tensor.dim(2));
  EXPECT_EQ(cl_expected_tensor.dim(3), cl_output_tensor.dim(3));

  for(int i = 0; i < cl_output_tensor.size(); i++)
  {//must have similar values:
    //EXPECT_NEAR(expected_data_cl[i], data_cl[i], 0.001);
    EXPECT_EQ(expected_data_cl[i], data_cl[i]);
  }

}


TEST(OpenCL, Transpose_op_comparison_test) {
#ifdef CAFFE2_USE_LITE_PROTO
  return;
#endif

  caffe2::GlobalInit();

  DeviceOption option;
  OpenCLContext context(option);
  Workspace ws_opencl;

  load_n_run(ws_opencl);

  compare_tensors_by_name("expected_data_nhwc1", "data_nhwc1", ws_opencl, context);
  compare_tensors_by_name("expected_data_nhwc2", "data_nhwc2", ws_opencl, context);
  compare_tensors_by_name("expected_data_nhwc3", "data_nhwc3", ws_opencl, context);
  compare_tensors_by_name("expected_data_nhwc4", "data_nhwc4", ws_opencl, context);
  compare_tensors_by_name("expected_data_nhwc5", "data_nhwc5", ws_opencl, context);
  compare_tensors_by_name("expected_data_nhwc6", "data_nhwc6", ws_opencl, context);
  compare_tensors_by_name("expected_data_nhwc7", "data_nhwc7", ws_opencl, context);
}

}
