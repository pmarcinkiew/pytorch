
#include "caffe2/core/flags.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/common.h"
#include "caffe2/core/init.h"
#include "caffe2/core/workspace.h"
#include "caffe2/core/logging.h"
#include "../context.h"
#include "conv_op.h"
#include "copy_ops.h"
#include <gtest/gtest.h>


namespace caffe2 {

static void run_test_helper(std::vector<float> &x_cpu_data,
                     std::vector<long int> &x_cpu_data_shape,
                     std::vector<float> &kernel_cpu_data,
                     std::vector<long int> &kernel_cpu_data_shape,
                     std::vector<float> &bias,
                     std::vector<float> &norm_mean,
                     std::vector<float> &norm_variance,
                     std::vector<float> &scale_bias,
                     std::vector<float> &expected_output,
                     std::vector<long int> &expected_output_shape,
                     const int kernel_size,
                     const int kernels_count,
                     const int pad,
                     const int stride
                    )
{
  DeviceOption option;
  OpenCLContext context(option);
  Workspace ws;

  {
   caffe2::CPUContext cpuContext;

   //X, NCHW:
   auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
   auto x_src_data_tensor = TensorCPU(x_cpu_data_shape, x_cpu_data, &cpuContext);
   t->CopyFrom(x_src_data_tensor, &context);

   OperatorDef def_copy1;
   def_copy1.set_name("CopyToOpenCL_IN1");
   def_copy1.add_input("X_cpu");
   def_copy1.add_output("X");
   def_copy1.mutable_device_option()->set_device_type(OPENCL);
   unique_ptr<OperatorBase> op1(
       new CopyToOpenCLOp<OpenCLContext>(def_copy1, &ws));

   //Kernel, NCHW:
   auto* tkern = ws.CreateBlob("Kernel_cpu")->GetMutable<TensorCPU>();
   auto kernel_src_data_tensor = TensorCPU(kernel_cpu_data_shape, kernel_cpu_data, &cpuContext);
   tkern->CopyFrom(kernel_src_data_tensor, &context);

   OperatorDef def_copy_kernel;
   def_copy_kernel.set_name("CopyToOpenCL_IN2");
   def_copy_kernel.add_input("Kernel_cpu");
   def_copy_kernel.add_output("Kernel");
   def_copy_kernel.mutable_device_option()->set_device_type(OPENCL);
   unique_ptr<OperatorBase> op_kernel(
       new CopyToOpenCLOp<OpenCLContext>(def_copy_kernel, &ws));

   //bias:
   {
     auto* tbias = ws.CreateBlob("Bias_cpu")->GetMutable<TensorCPU>();
     auto bias_src_data_tensor = TensorCPU({kernels_count}, bias, &cpuContext);
     tbias->CopyFrom(bias_src_data_tensor, &context);

     OperatorDef def_copy_bias;
     def_copy_bias.set_name("CopyToOpenCL_IN3");
     def_copy_bias.add_input("Bias_cpu");
     def_copy_bias.add_output("Bias");
     def_copy_bias.mutable_device_option()->set_device_type(OPENCL);
     unique_ptr<OperatorBase> op_bias(
         new CopyToOpenCLOp<OpenCLContext>(def_copy_bias, &ws));
     EXPECT_TRUE(op_bias->Run());
   }

   //bias scale:
   if(scale_bias.size() > 0) {
     auto* tsbias = ws.CreateBlob("Scale_Bias_cpu")->GetMutable<TensorCPU>();
     auto scale_bias_src_data_tensor = TensorCPU({kernels_count}, scale_bias, &cpuContext);
     tsbias->CopyFrom(scale_bias_src_data_tensor, &context);

     OperatorDef def_copy_scale_bias;
     def_copy_scale_bias.set_name("CopyToOpenCL_IN4");
     def_copy_scale_bias.add_input("Scale_Bias_cpu");
     def_copy_scale_bias.add_output("Scale_Bias");
     def_copy_scale_bias.mutable_device_option()->set_device_type(OPENCL);
     unique_ptr<OperatorBase> op_scale_bias(
         new CopyToOpenCLOp<OpenCLContext>(def_copy_scale_bias, &ws));
     EXPECT_TRUE(op_scale_bias->Run());
   }

   //normalization:
   if(norm_mean.size() > 0 && norm_variance.size() > 0) {
     auto* tmean = ws.CreateBlob("Mean_cpu")->GetMutable<TensorCPU>();
     auto mean_src_data_tensor = TensorCPU({kernels_count}, norm_mean, &cpuContext);
     tmean->CopyFrom(mean_src_data_tensor, &context);

     OperatorDef def_copy_mean;
     def_copy_mean.set_name("CopyToOpenCL_IN5");
     def_copy_mean.add_input("Mean_cpu");
     def_copy_mean.add_output("Mean");
     def_copy_mean.mutable_device_option()->set_device_type(OPENCL);
     unique_ptr<OperatorBase> op_mean(
         new CopyToOpenCLOp<OpenCLContext>(def_copy_mean, &ws));
     EXPECT_TRUE(op_mean->Run());

     auto* tvariance = ws.CreateBlob("Variance_cpu")->GetMutable<TensorCPU>();
     auto variance_src_data_tensor = TensorCPU({kernels_count}, norm_variance, &cpuContext);
     tvariance->CopyFrom(variance_src_data_tensor, &context);

     OperatorDef def_copy_variance;
     def_copy_variance.set_name("CopyToOpenCL_IN6");
     def_copy_variance.add_input("Variance_cpu");
     def_copy_variance.add_output("Variance");
     def_copy_variance.mutable_device_option()->set_device_type(OPENCL);
     unique_ptr<OperatorBase> op_variance(
         new CopyToOpenCLOp<OpenCLContext>(def_copy_variance, &ws));
     EXPECT_TRUE(op_variance->Run());
   }

   EXPECT_TRUE(op1->Run());
   EXPECT_TRUE(op_kernel->Run());
  }

  OperatorDef convdef;
  convdef.set_name("Conv");
  convdef.add_input("X");
  convdef.add_input("Kernel");
  convdef.add_input("Bias");
  if(scale_bias.size() > 0) {
    convdef.add_input("Scale_Bias");
    Argument *arg = convdef.add_arg();
    arg->set_name("use_scale");
    arg->set_i(true);
  }

  if(norm_mean.size() > 0 && norm_variance.size() > 0) {
    convdef.add_input("Mean");
    convdef.add_input("Variance");
    Argument *arg = convdef.add_arg();
    arg->set_name("use_normalization");
    arg->set_i(true);
  }

  convdef.add_output("Y");
  Argument *arg = convdef.add_arg();
  arg->set_name("kernel");
  arg->set_i(kernel_size);
  arg = convdef.add_arg();
  arg->set_name("pad");
  arg->set_i(pad);
  arg = convdef.add_arg();
  arg->set_name("stride");
  arg->set_i(stride);

  convdef.mutable_device_option()->set_device_type(OPENCL);
  unique_ptr<OperatorBase> convop(new ConvOp<float>(convdef, &ws));

  EXPECT_NE(nullptr, convop.get());
  EXPECT_TRUE(convop->Run());
  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto Y = Yblob->GetMutable<Tensor<OpenCLContext>>();

  long int expected_size = std::accumulate
                (
                  expected_output_shape.begin(),
                  expected_output_shape.end(),
                  1,
                  [](long int e1, long int e2)->long int{ return e1 * e2;}
                );

  EXPECT_EQ(Y->size(), expected_size);

  for(int d = 0; d < Y->ndim(); d++)
    EXPECT_EQ(Y->dim32(d), expected_output_shape[d]);

  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  std::vector<float> data(Y->size());
  context.CopyBytes<OpenCLContext, CPUContext>(Y->size() * sizeof(float),
                                              (void*)yBuffer, (void*)&data[0]);

  for (int i = 0; i < Y->size(); ++i) {
   EXPECT_NEAR(data[i], expected_output[i], 0.001);
  }
}

TEST(OpenCL, Convolution_simple1) {
  std::vector<float> x_cpu_data = {
                                  0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 1.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0
                              };
  std::vector<long int> x_cpu_data_shape{1, 1, 5, 5};

  std::vector<float> kernel_cpu_data = {
                                  2.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 1, 1, 1};

  std::vector<float> bias = {0.0};
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 2.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0
  };
  std::vector<long int> expected_output_shape = {1, 1, 5, 5};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  1, 1, 0, 1
                  );

}

TEST(OpenCL, Convolution_test1) {
  std::vector<float> x_cpu_data = {
                                  0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 1.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0
                              };
  std::vector<long int> x_cpu_data_shape{1, 1, 5, 5};

  std::vector<float> kernel_cpu_data = {
                                  0.0, 0.0, 0.0,
                                  0.0, 2.5, 0.0,
                                  0.0, 0.0, 0.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 1, 3, 3};

  std::vector<float> bias = {0.0};
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  0.0, 0.0, 0.0,
                                  0.0, 2.5, 0.0,
                                  0.0, 0.0, 0.0
                              };
  std::vector<long int> expected_output_shape = {1, 1, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 1, 0, 1
                  );

}

TEST(OpenCL, Convolution_test2) {
  std::vector<float> x_cpu_data = {
                                  1.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 1.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 1.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 1.0
                              };
  std::vector<long int> x_cpu_data_shape{1, 1, 5, 5};

  std::vector<float> kernel_cpu_data = {
                                  1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 1, 3, 3};
  std::vector<float> bias = {0.0};
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  3.0, 0.0, 0.0,
                                  0.0, 3.0, 0.0,
                                  0.0, 0.0, 3.0
                              };
  std::vector<long int> expected_output_shape = {1, 1, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 1, 0, 1
                  );
}

TEST(OpenCL, Convolution_test3) {
  std::vector<float> x_cpu_data = {
                                  1.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                                  2.0, 11.0, 0.0, 0.0, 0.0, 0.0,
                                  3.0, 12.0, 0.0, 0.0, 0.0, 0.0,
                                  4.0, 13.0, 0.0, 0.0, 0.0, 0.0,
                                  5.0, 14.0, 0.0, 0.0, 0.0, 0.0,
                                  6.0, 15.0, 0.0, 0.0, 0.0, 0.0,
                                  7.0, 16.0, 0.0, 0.0, 0.0, 0.0,
                                  8.0, 17.0, 0.0, 0.0, 0.0, 0.0,
                                  9.0, 18.0, 0.0, 0.0, 0.0, 0.0
                              };
  std::vector<long int> x_cpu_data_shape{1, 1, 9, 6};

  std::vector<float> kernel_cpu_data = {
                                  1.0, 0.0, 0.0, 0.0, 1.0,
                                  1.0, 0.0, 0.0, 0.0, 1.0,
                                  1.0, 0.0, 0.0, 0.0, 1.0,
                                  1.0, 0.0, 0.0, 0.0, 1.0,
                                  1.0, 0.0, 0.0, 0.0, 1.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 1, 5, 5};
  std::vector<float> bias = {0.0};
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  15.0, 60.0,
                                  20.0, 65.0,
                                  25.0, 70.0,
                                  30.0, 75.0,
                                  35.0, 80.0
                              };
  std::vector<long int> expected_output_shape = {1, 1, 5, 2};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  5, 1, 0, 1
                  );
}

TEST(OpenCL, Convolution_test4) {
  std::vector<float> x_cpu_data = {
                                  1.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                                  2.0, 11.0, 0.0, 0.0, 0.0, 0.0,
                                  3.0, 12.0, 0.0, 0.0, 0.0, 0.0,
                                  4.0, 13.0, 0.0, 0.0, 0.0, 0.0,
                                  5.0, 14.0, 0.0, 0.0, 0.0, 0.0,
                                  6.0, 15.0, 0.0, 0.0, 0.0, 0.0,
                                  7.0, 16.0, 0.0, 0.0, 0.0, 0.0,
                                  8.0, 17.0, 0.0, 0.0, 0.0, 0.0,
                                  9.0, 18.0, 0.0, 0.0, 0.0, 2.0
                              };
  std::vector<long int> x_cpu_data_shape{1, 1, 9, 6};

  std::vector<float> kernel_cpu_data = {
                                  1.0, 0.0, 0.0, 0.0, 1.0,
                                  1.0, 0.0, 0.0, 0.0, 1.0,
                                  1.0, 0.0, 0.0, 0.0, 1.0,
                                  1.0, 0.0, 0.0, 0.0, 1.0,
                                  1.0, 0.0, 0.0, 0.0, 1.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 1, 5, 5};
  std::vector<float> bias({1.5});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  16.5, 61.5,
                                  21.5, 66.5,
                                  26.5, 71.5,
                                  31.5, 76.5,
                                  36.5, 83.5
                              };
  std::vector<long int> expected_output_shape = {1, 1, 5, 2};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  5, 1, 0, 1
                  );
}

TEST(OpenCL, Convolution_test5) {
  //3 channel 5x5 image:
  std::vector<float> x_cpu_data = {
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                              };
  std::vector<long int> x_cpu_data_shape{1, 3, 5, 5};

  //3 channel 3x3 kernel
  std::vector<float> kernel_cpu_data = {
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0,

                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 3, 3, 3};

  std::vector<float> bias = {0.0};
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  54.0, 54.0, 54.0,
                                  54.0, 54.0, 54.0,
                                  54.0, 54.0, 54.0
                              };
  std::vector<long int> expected_output_shape = {1, 1, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 1, 0, 1
                  );
}

TEST(OpenCL, Convolution_test6) {
  //2 channel 5x5 image:
  std::vector<float> x_cpu_data = {
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0
                              };
  std::vector<long int> x_cpu_data_shape{1, 2, 5, 5};

  //4 kernels, 2 channel 3x3 each
  std::vector<float> kernel_cpu_data = {//k1
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0,

                                  //k2
                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0,

                                  4.0, 4.0, 4.0,
                                  4.0, 4.0, 4.0,
                                  4.0, 4.0, 4.0,

                                  //k3
                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0,

                                  5.0, 5.0, 5.0,
                                  5.0, 5.0, 5.0,
                                  5.0, 5.0, 5.0,

                                  //k4
                                  4.0, 4.0, 4.0,
                                  4.0, 4.0, 4.0,
                                  4.0, 4.0, 4.0,

                                  6.0, 6.0, 6.0,
                                  6.0, 6.0, 6.0,
                                  6.0, 6.0, 6.0

                              };
  std::vector<long int> kernel_cpu_data_shape{4, 2, 3, 3};

  std::vector<float> bias({0.1, 0.2, 0.3, 0.4});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  36.1, 36.1, 36.1,
                                  36.1, 36.1, 36.1,
                                  36.1, 36.1, 36.1,

                                  54.2, 54.2, 54.2,
                                  54.2, 54.2, 54.2,
                                  54.2, 54.2, 54.2,

                                  72.3, 72.3, 72.3,
                                  72.3, 72.3, 72.3,
                                  72.3, 72.3, 72.3,

                                  90.4, 90.4, 90.4,
                                  90.4, 90.4, 90.4,
                                  90.4, 90.4, 90.4
                              };
  std::vector<long int> expected_output_shape = {1, 4, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 4, 0, 1
                  );
}

TEST(OpenCL, Convolution_test7) {
  //2 channel 3x3 image:
  std::vector<float> x_cpu_data = {
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0

                              };
  std::vector<long int> x_cpu_data_shape{1, 2, 3, 3};

  //3 kernels, 2 channel 3x3 each
  std::vector<float> kernel_cpu_data = {//k1
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  //k2
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  //k3
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0

                              };
  std::vector<long int> kernel_cpu_data_shape{3, 2, 3, 3};

  std::vector<float> bias({0.1, 0.2, 0.3});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  27.1,
                                  27.2,
                                  27.3
                              };
  std::vector<long int> expected_output_shape = {1, 3, 1, 1};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 3, 0, 1
                  );
}

TEST(OpenCL, Convolution_test8) {

  //2 images each with 2 channels 3x3 :
  std::vector<float> x_cpu_data = {//im1:
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0,
                                  //im2:
                                  4.0, 4.0, 4.0,
                                  4.0, 4.0, 4.0,
                                  4.0, 4.0, 4.0,

                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0
                              };
  std::vector<long int> x_cpu_data_shape{2, 2, 3, 3};

  //3 kernels, 2 channel 3x3 each
  std::vector<float> kernel_cpu_data = {//k1
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  //k2
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  //k3
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0

                              };
  std::vector<long int> kernel_cpu_data_shape{3, 2, 3, 3};
  std::vector<float> bias({0.1, 0.2, 0.3});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  27.1,
                                  27.2,
                                  27.3,
                                  63.1,
                                  63.2,
                                  63.3
                              };
  std::vector<long int> expected_output_shape = {2, 3, 1, 1};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 3, 0, 1
                  );

}

TEST(OpenCL, Convolution_test9) {

  //3 images 1 channel 5x5 :
  std::vector<float> x_cpu_data = {
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0,
                              };
  std::vector<long int> x_cpu_data_shape{3, 1, 5, 5};

  //3 kernels 1 channel 3x3
  std::vector<float> kernel_cpu_data = {//k1:
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  //k2:
                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0,
                                  //k3:
                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0
                              };
  std::vector<long int> kernel_cpu_data_shape{3, 1, 3, 3};

  std::vector<float> bias({0.1, 0.2, 0.3});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  //im1 k1:
                                  9.1, 9.1, 9.1,
                                  9.1, 9.1, 9.1,
                                  9.1, 9.1, 9.1,
                                  //im1 k2:
                                  18.2, 18.2, 18.2,
                                  18.2, 18.2, 18.2,
                                  18.2, 18.2, 18.2,
                                  //im1 k3:
                                  27.3, 27.3, 27.3,
                                  27.3, 27.3, 27.3,
                                  27.3, 27.3, 27.3,
                                  //im2 k1:
                                  9.1, 9.1, 9.1,
                                  9.1, 9.1, 9.1,
                                  9.1, 9.1, 9.1,
                                  //im2 k2:
                                  18.2, 18.2, 18.2,
                                  18.2, 18.2, 18.2,
                                  18.2, 18.2, 18.2,
                                  //im2 k3:
                                  27.3, 27.3, 27.3,
                                  27.3, 27.3, 27.3,
                                  27.3, 27.3, 27.3,
                                  //im3 k1:
                                  9.1, 9.1, 9.1,
                                  9.1, 9.1, 9.1,
                                  9.1, 9.1, 9.1,
                                  //im3 k2:
                                  18.2, 18.2, 18.2,
                                  18.2, 18.2, 18.2,
                                  18.2, 18.2, 18.2,
                                  //im3 k3:
                                  27.3, 27.3, 27.3,
                                  27.3, 27.3, 27.3,
                                  27.3, 27.3, 27.3,
                              };
  std::vector<long int> expected_output_shape = {3, 3, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 3, 0, 1
                  );
}

//Stride and padding:
TEST(OpenCL, Convolution_test10) {

  //1 image 1 channel 3x3 :
  std::vector<float> x_cpu_data = {//im1:
                                  1.0, 1.0, 0.0,
                                  1.0, 1.0, 0.0,
                                  0.0, 0.0, 0.0,

                              };
  std::vector<long int> x_cpu_data_shape{1, 1, 3, 3};

  //3 kernels, 2 channel 3x3 each
  std::vector<float> kernel_cpu_data = {//k1
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 1, 3, 3};
  std::vector<float> bias({0.1});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  4.1, 4.1, 2.1,
                                  4.1, 4.1, 2.1,
                                  2.1, 2.1, 1.1
                              };
  std::vector<long int> expected_output_shape = {1, 1, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 1, 1, 1
                  );

}

TEST(OpenCL, Convolution_test11) {

  //2 images each with 2 channels 3x3 :
  std::vector<float> x_cpu_data = {//im1:
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0,
                                  //im2:
                                  4.0, 4.0, 4.0,
                                  4.0, 4.0, 4.0,
                                  4.0, 4.0, 4.0,

                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0,
                                  3.0, 3.0, 3.0
                              };
  std::vector<long int> x_cpu_data_shape{2, 2, 3, 3};

  //3 kernels, 2 channel 3x3 each
  std::vector<float> kernel_cpu_data = {//k1
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  //k2
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  //k3
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0

                              };
  std::vector<long int> kernel_cpu_data_shape{3, 2, 3, 3};
  std::vector<float> bias({0.1, 0.2, 0.3});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                  12.1, 18.1, 12.1,
                                  18.1, 27.1, 18.1,
                                  12.1, 18.1, 12.1,

                                  12.2, 18.2, 12.2,
                                  18.2, 27.2, 18.2,
                                  12.2, 18.2, 12.2,

                                  12.3, 18.3, 12.3,
                                  18.3, 27.3, 18.3,
                                  12.3, 18.3, 12.3,


                                  28.1, 42.1, 28.1,
                                  42.1, 63.1, 42.1,
                                  28.1, 42.1, 28.1,

                                  28.2, 42.2, 28.2,
                                  42.2, 63.2, 42.2,
                                  28.2, 42.2, 28.2,

                                  28.3, 42.3, 28.3,
                                  42.3, 63.3, 42.3,
                                  28.3, 42.3, 28.3
                              };
  std::vector<long int> expected_output_shape = {2, 3, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 3, 1, 1
                  );

}

TEST(OpenCL, Convolution_test12) {

  //1 image 1 channel 3x3 :
  std::vector<float> x_cpu_data = {//im1:
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                              };
  std::vector<long int> x_cpu_data_shape{1, 1, 3, 3};

  //1 kernel, 1 channel 3x3
  std::vector<float> kernel_cpu_data = {//k1
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 1, 3, 3};
  std::vector<float> bias({0.1});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                             1.1, 2.1, 3.1, 2.1, 1.1,
                             2.1, 4.1, 6.1, 4.1, 2.1,
                             3.1, 6.1, 9.1, 6.1, 3.1,
                             2.1, 4.1, 6.1, 4.1, 2.1,
                             1.1, 2.1, 3.1, 2.1, 1.1
                              };
  std::vector<long int> expected_output_shape = {1, 1, 5, 5};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 1, 2, 1
                  );

}

TEST(OpenCL, Convolution_test13) {

  //1 image 1 channel 3x3 :
  std::vector<float> x_cpu_data = {//im1:
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                              };
  std::vector<long int> x_cpu_data_shape{1, 1, 3, 3};

  //1 kernel, 1 channel 3x3
  std::vector<float> kernel_cpu_data = {//k1
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 1, 3, 3};
  std::vector<float> bias({0.1});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                             1.1, 3.1, 1.1,
                             3.1, 9.1, 3.1,
                             1.1, 3.1, 1.1
                              };
  std::vector<long int> expected_output_shape = {1, 1, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 1, 2, 2
                  );

}

TEST(OpenCL, Convolution_test14) {

  //1 image 2 channel 3x3 :
  std::vector<float> x_cpu_data = {//im1:
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0
                              };
  std::vector<long int> x_cpu_data_shape{1, 2, 3, 3};

  //1 kernel, 2 channel 3x3
  std::vector<float> kernel_cpu_data = {//k1
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,

                                  2.0, 1.0, 1.0,
                                  1.0, 2.0, 1.0,
                                  1.0, 1.0, 2.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 2, 3, 3};
  std::vector<float> bias({0.1});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                             0.1, 0.1,  0.1,
                             0.1, 10.1, 4.1,
                             0.1, 4.1,  3.1
                              };
  std::vector<long int> expected_output_shape = {1, 1, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 1, 4, 3
                  );

}

TEST(OpenCL, Convolution_test15) {

  //1 image 2 channel 1x1 :
  std::vector<float> x_cpu_data = {//im1:
                                    1.0,
                                    2.0
                              };
  std::vector<long int> x_cpu_data_shape{1, 2, 1, 1};

  //1 kernel, 2 channel 1x1
  std::vector<float> kernel_cpu_data = {//k1
                                    -1.0,
                                    -2.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 2, 1, 1};
  std::vector<float> bias({0.1});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                     0.1, 0.1,
                                     0.1, 0.1
                              };
  std::vector<long int> expected_output_shape = {1, 1, 2, 2};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  1, 1, 1, 2
                  );

}

TEST(OpenCL, Convolution_test16) {

  //2 images 2 channel 6x6 :
  std::vector<float> x_cpu_data = {//im1:
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                    4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                    5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                    6.0, 6.0, 6.0, 6.0, 6.0, 6.0,

                                    7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
                                    8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                    9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
                                    10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                    11.0, 11.0, 11.0, 11.0, 11.0, 11.0,
                                    12.0, 12.0, 12.0, 12.0, 12.0, 12.0,

                                    //im2:
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                    4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                    5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                    6.0, 6.0, 6.0, 6.0, 6.0, 6.0,

                                    7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
                                    8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                    9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
                                    10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                    11.0, 11.0, 11.0, 11.0, 11.0, 11.0,
                                    12.0, 12.0, 12.0, 12.0, 12.0, 12.0
                              };
  std::vector<long int> x_cpu_data_shape{2, 2, 6, 6};

  //1 kernel, 2 channel 5x5
  std::vector<float> kernel_cpu_data = {//k1
                                    1.0, 2.0, 3.0, 4.0, 5.0,
                                    1.0, 2.0, 3.0, 4.0, 5.0,
                                    1.0, 2.0, 3.0, 4.0, 5.0,
                                    1.0, 2.0, 3.0, 4.0, 5.0,
                                    1.0, 2.0, 3.0, 4.0, 5.0,

                                    6.0, 7.0, 8.0, 9.0, 10.0,
                                    6.0, 7.0, 8.0, 9.0, 10.0,
                                    6.0, 7.0, 8.0, 9.0, 10.0,
                                    6.0, 7.0, 8.0, 9.0, 10.0,
                                    6.0, 7.0, 8.0, 9.0, 10.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 2, 5, 5};
  std::vector<float> bias({0.1});
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;

  std::vector<float> expected_output = {
                                     0.1, 0.1, 0.1,
                                     0.1, 1296.1, 472.1,
                                     0.1, 936.1, 332.1,

                                     0.1, 0.1, 0.1,
                                     0.1, 1296.1, 472.1,
                                     0.1, 936.1, 332.1,
                              };
  std::vector<long int> expected_output_shape = {2, 1, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  5, 1, 6, 5
                  );

}

TEST(OpenCL, Convolution_test17a) {

  //1 images 1 channel 5x5 :
  std::vector<float> x_cpu_data = {//im1:
                                    0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 1.0, 1.0, 1.0, 0.0,
                                    0.0, 1.0, 1.0, 1.0, 0.0,
                                    0.0, 1.0, 1.0, 1.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0
                              };
  std::vector<long int> x_cpu_data_shape{1, 1, 5, 5};

  //1 kernel, 1 channel 3x3
  std::vector<float> kernel_cpu_data = {//k1
                                    1.0, 2.0, 3.0,
                                    1.0, 2.0, 3.0,
                                    1.0, 2.0, 3.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 1, 3, 3};

  std::vector<float> bias({-1.0});
  std::vector<float> norm_mean({2.0});
  std::vector<float> norm_variance({0.666});
  std::vector<float> scale_bias;


  //Y = (([result of convolution] - norm_mean) / (sqrt(norm_variance) + .000001f)) * scale_bias + bias

  std::vector<float> expected_output = {
                                     8.8028, 11.2535, 3.9014,
                                     14.9296, 18.6056, 7.5774,
                                     8.8028, 11.2535, 3.9014
                              };
  std::vector<long int> expected_output_shape = {1, 1, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 1, 0, 1
                  );

}


TEST(OpenCL, Convolution_test17) {

  //1 images 1 channel 5x5 :
  std::vector<float> x_cpu_data = {//im1:
                                    0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 1.0, 1.0, 1.0, 0.0,
                                    0.0, 1.0, 1.0, 1.0, 0.0,
                                    0.0, 1.0, 1.0, 1.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0
                              };
  std::vector<long int> x_cpu_data_shape{1, 1, 5, 5};

  //1 kernel, 1 channel 3x3
  std::vector<float> kernel_cpu_data = {//k1
                                    1.0, 2.0, 3.0,
                                    1.0, 2.0, 3.0,
                                    1.0, 2.0, 3.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 1, 3, 3};

  std::vector<float> bias({-1.0});
  std::vector<float> norm_mean({2.0});
  std::vector<float> norm_variance({0.666});
  std::vector<float> scale_bias({10.0});


  //Y = (([result of convolution] - norm_mean) / (sqrt(norm_variance) + .000001f)) * scale_bias + bias

  std::vector<float> expected_output = {
                                     97.028, 121.535, 48.014,
                                     158.296, 195.056, 84.774,
                                     97.028, 121.535, 48.014
                              };
  std::vector<long int> expected_output_shape = {1, 1, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 1, 0, 1
                  );

}

TEST(OpenCL, Convolution_test18) {

  //1 images 1 channel 5x5 :
  std::vector<float> x_cpu_data = {//im1:
                                    0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 1.0, 1.0, 1.0, 0.0,
                                    0.0, 1.0, 1.0, 1.0, 0.0,
                                    0.0, 1.0, 1.0, 1.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0
                              };
  std::vector<long int> x_cpu_data_shape{1, 1, 5, 5};

  //1 kernel, 1 channel 3x3
  std::vector<float> kernel_cpu_data = {//k1
                                    1.0, 2.0, 3.0,
                                    1.0, 2.0, 3.0,
                                    1.0, 2.0, 3.0
                              };
  std::vector<long int> kernel_cpu_data_shape{1, 1, 3, 3};

  std::vector<float> bias({-1.0});
  std::vector<float> norm_mean({2.0});
  std::vector<float> norm_variance({0.666});
  std::vector<float> scale_bias;


  //Y = (([result of convolution] - norm_mean) / (sqrt(norm_variance) + .000001f)) + bias

  std::vector<float> expected_output = {
                                8.802, 11.253, 3.901,
                                14.929, 18.605, 7.577,
                                8.802, 11.253, 3.901
                              };
  std::vector<long int> expected_output_shape = {1, 1, 3, 3};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  3, 1, 0, 1
                  );

}

TEST(OpenCL, Convolution_perf_test) {

  //1 images 1 channel 1920 x 1080
  std::vector<float> x_cpu_data(1920 * 1080 * 3, 0.0);
  std::vector<long int> x_cpu_data_shape{1, 3, 1920, 1080};

  //1 kernel, 3 channel 5x5
  std::vector<float> kernel_cpu_data = {
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,

                                    2.0, 2.0, 2.0, 2.0, 2.0,
                                    2.0, 2.0, 2.0, 2.0, 2.0,
                                    2.0, 2.0, 2.0, 2.0, 2.0,
                                    2.0, 2.0, 2.0, 2.0, 2.0,
                                    2.0, 2.0, 2.0, 2.0, 2.0,

                                    3.0, 3.0, 3.0, 3.0, 3.0,
                                    3.0, 3.0, 3.0, 3.0, 3.0,
                                    3.0, 3.0, 3.0, 3.0, 3.0,
                                    3.0, 3.0, 3.0, 3.0, 3.0,
                                    3.0, 3.0, 3.0, 3.0, 3.0,



                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0,

                                    2.0, 2.0, 2.0, 2.0, 2.0,
                                    2.0, 2.0, 2.0, 2.0, 2.0,
                                    2.0, 2.0, 2.0, 2.0, 2.0,
                                    2.0, 2.0, 2.0, 2.0, 2.0,
                                    2.0, 2.0, 2.0, 2.0, 2.0,

                                    3.0, 3.0, 3.0, 3.0, 3.0,
                                    3.0, 3.0, 3.0, 3.0, 3.0,
                                    3.0, 3.0, 3.0, 3.0, 3.0,
                                    3.0, 3.0, 3.0, 3.0, 3.0,
                                    3.0, 3.0, 3.0, 3.0, 3.0

                              };
  std::vector<long int> kernel_cpu_data_shape{2, 3, 5, 5};

  std::vector<float> bias(2, 0.0);
  std::vector<float> norm_mean;
  std::vector<float> norm_variance;
  std::vector<float> scale_bias;


  //Y = (([result of convolution] - norm_mean) / (sqrt(norm_variance) + .000001f)) + bias

  std::vector<float> expected_output(1 * 2 * 1916 * 1076);
  std::vector<long int> expected_output_shape = {1, 2, 1916, 1076};

  run_test_helper(x_cpu_data, x_cpu_data_shape,
                  kernel_cpu_data, kernel_cpu_data_shape,
                  bias, norm_mean, norm_variance, scale_bias,
                  expected_output, expected_output_shape,
                  5, 2, 0, 1
                  );

}

}
