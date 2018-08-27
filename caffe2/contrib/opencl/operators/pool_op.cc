#include "caffe2/core/context.h"
#include "caffe2/contrib/opencl/context.h"
#include "caffe2/contrib/opencl/operator.h"

#include "caffe2/contrib/opencl/kernels/utils.h"
#include "caffe2/utils/math.h"

#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/pool_op.h"

#include <sys/time.h>

namespace caffe2{

static const char *kernelMaxPool1DForwardNCHW = R"CLC(
    __kernel void kernelMaxPool1DForwardNCHW(
                    const __global float *input, __global float *pooled,
                    const int n_threads, const int channels,
                    const int input_height,
                    const int pooled_height,
                    const int kernel_h,
                    const int stride_h,
                    const int pad_t) {

            int n_thread = get_global_id(0);

            if(n_thread < n_threads) {
              int ph = n_thread % pooled_height;
              int c = ( n_thread / pooled_height) % channels;
              int n = n_thread / pooled_height / channels;

              int hstart = ph * stride_h - pad_t;
              hstart = max(hstart, 0);

              int hend = min(hstart + kernel_h, input_height);
                            
              __global const float *input_offset = input + n * channels * input_height;
              
              float max_value = -FLT_MAX;
              int idx = -1;
              
              for(int h = hstart; h < hend; ++h) {
                idx = c * input_height + h;
                max_value = max(max_value, input_offset[idx]);
              }
              pooled[n_thread] = max_value;
           }
       }
)CLC";

static const char *kernelMaxPool2DForwardNCHW = R"CLC(
   __kernel void kernelMaxPool2DForwardNCHW(
                   const __global float *input, __global float *pooled,
                   const int n_threads, const int channels, 
                   const int input_width, const int input_height,
                   const int pooled_width,  const int pooled_height,
                   const int kernel_w, const int kernel_h,
                   const int stride_w, const int stride_h,
                   const int pad_t, const int pad_l ) {


          //for (int n_thread  = get_global_id(0); n_thread < n_threads; n_thread+=get_global_size(0)){
          int n_thread = get_global_id(0);

          if(n_thread < n_threads) {
            int pw = n_thread % pooled_width;
            int ph = (n_thread / pooled_width) % pooled_height;

            int c = (n_thread / pooled_width / pooled_height) % channels;
            int n = n_thread / pooled_width / pooled_height / channels;

            int hstart = ph * stride_h - pad_t;
            int wstart = pw * stride_w - pad_l;

            int hend = min(hstart + kernel_h, input_height);
            int wend = min(wstart + kernel_w, input_width);

            hstart = max(hstart, 0);
            wstart = max(wstart, 0);


            __global const float *input_offset = input + n * channels * input_height * input_width;

            float max_value = -FLT_MAX;
            int idx = -1;

            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                idx = c * input_height * input_width + h * input_width + w;
                max_value = max( max_value, input_offset[idx] );
              }
            }
            pooled[n_thread] = max_value;
         }
      }
)CLC";

static const char *kernelMaxPool3DForwardNCHW = R"CLC(

    __kernel void kernelMaxPool3DForwardNCHW(
                    const __global float *input, __global float *pooled,
                    const int n_threads, const int channels,
                    const int input_width, const int input_height, const int input_depth,
                    const int pooled_width, const int pooled_height, const int pooled_depth,
                    const int kernel_w, const int kernel_h, const int kernel_d,
                    const int stride_w, const int stride_h, const int stride_d,
                    const int pad_t, const int pad_l, const int pad_f) {


          int n_thread = get_global_id(0);
          if(n_thread < n_threads) {

            int pd = n_thread % pooled_depth;
            int pw = (n_thread / pooled_depth) % pooled_width;
            int ph = (n_thread / pooled_depth / pooled_width) % pooled_height;

            int c = (n_thread / pooled_depth / pooled_width / pooled_height) % channels;
            int n = n_thread / pooled_depth / pooled_width / pooled_height / channels;

            int hstart = ph * stride_h - pad_t;
            int wstart = pw * stride_w - pad_l;

            int hend = min(hstart + kernel_h, input_height);
            int wend = min(wstart + kernel_w, input_width);

            int dstart = pd * stride_d - pad_f;

            int dend = min(dstart + kernel_d, input_depth);

            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            dstart = max(dstart, 0);

            float max_value = -FLT_MAX;

            __global const float *input_offset = input + n * channels * input_height * input_width * input_depth;

            int idx = -1;
            for(int h = hstart; h < hend; ++h) {
              for(int w = wstart; w < wend; ++w) {
                for(int d = dstart; d < dend; ++d) {
                  idx = ((c * input_height + h) * input_width + w ) * input_depth + d;
                  if(input_offset[idx] > max_value) {
                    max_value = input_offset[idx];
                  }
                  //max_value = max( max_value, input_offset[idx] );
                }
              }
            }
            pooled[n_thread] = max_value;
         }
      }
)CLC";

class MaxPool{};


template<> 
bool PoolOp<float, OpenCLContext, MaxPool>::RunOnDeviceWithOrderNCHW() {

  auto& X = Input(0);
  auto* Y = Output(0);

  int inputDimSize = X.dims().size()-2;
  int kernels_size = kernel_.size();
  int strides_size = stride_.size();

  int samples = X.dim32(0);
  CAFFE_ENFORCE_GE(samples, 1, "Number of samples must be >= 1.");

  int channels = X.dim32(1);
  CAFFE_ENFORCE_GE(channels, 1, "Number of channels must be >= 1.");

  CAFFE_ENFORCE_EQ(kernels_size, inputDimSize, "The size of kernels should be the same.");
  CAFFE_ENFORCE_EQ(strides_size, inputDimSize);
  CAFFE_ENFORCE_EQ(strides_size, kernels_size);

  ConvPoolOpBase<OpenCLContext>::SetOutputSize(X, Y, X.dim32(1));

  int height = X.dim32(2);
  int width = kernel_.size() > 1 ? X.dim32(3) : 1;
  int depth = kernel_.size() > 2 ? X.dim32(4) : 1;

  int pooled_height = Y->dim32(2);
  int pooled_width = kernel_.size() > 1 ? Y->dim32(3) : 1;
  int pooled_depth = kernel_.size() > 2 ? Y->dim32(4) : 1;

  int kernel_width = kernel_w();
  int kernel_height = kernel_h();
  int kernel_depth = 0;  

  int stride_width = stride_w();
  int stride_height = stride_h();
  int stride_depth = 0;

  int pad_top = pad_t();
  int pad_low  = pad_l();
  int pad_depth = 0;

  cl::Buffer* bufferX = (cl::Buffer*)X.template data<float>();
  cl::Buffer* bufferY = (cl::Buffer*)Y->template mutable_data<float>();

  int output_size = Y->size();
  int kernel_size = kernel_.size();


  cl::Kernel kernel;

  switch(kernel_.size()) {
    case 1:
      kernel = context_.BuildKernelCached(kernelMaxPool1DForwardNCHW,
                                          "", "kernelMaxPool1DForwardNCHW");

      OPENCL_CHECK(kernel.setArg(0, *bufferX));
      OPENCL_CHECK(kernel.setArg(1, *bufferY));

      OPENCL_CHECK(kernel.setArg(2, output_size));
      OPENCL_CHECK(kernel.setArg(3, channels));

      OPENCL_CHECK(kernel.setArg(4, height));
      OPENCL_CHECK(kernel.setArg(5, pooled_height));
      OPENCL_CHECK(kernel.setArg(6, kernel_height));
      OPENCL_CHECK(kernel.setArg(7, stride_height));
      OPENCL_CHECK(kernel.setArg(8, pad_top));
      break;

    case 2:
      kernel = context_.BuildKernelCached(kernelMaxPool2DForwardNCHW,
                                          "", "kernelMaxPool2DForwardNCHW");

      OPENCL_CHECK(kernel.setArg(0, *bufferX));
      OPENCL_CHECK(kernel.setArg(1, *bufferY));

      OPENCL_CHECK(kernel.setArg(2, output_size));
      OPENCL_CHECK(kernel.setArg(3, channels));

      OPENCL_CHECK(kernel.setArg(4, width));
      OPENCL_CHECK(kernel.setArg(5, height));

      OPENCL_CHECK(kernel.setArg(6, pooled_width));
      OPENCL_CHECK(kernel.setArg(7, pooled_height));

      OPENCL_CHECK(kernel.setArg(8, kernel_width));
      OPENCL_CHECK(kernel.setArg(9, kernel_height));

      OPENCL_CHECK(kernel.setArg(10, stride_width));
      OPENCL_CHECK(kernel.setArg(11, stride_height));

      OPENCL_CHECK(kernel.setArg(12, pad_top));
      OPENCL_CHECK(kernel.setArg(13, pad_low));
      break;

    case 3:
      kernel_depth = kernel_[2];  
      stride_depth = stride_[2];
      pad_depth = pads_[2];

      kernel = context_.BuildKernelCached(kernelMaxPool3DForwardNCHW,
                                          "", "kernelMaxPool3DForwardNCHW");

      OPENCL_CHECK(kernel.setArg(0, *bufferX));
      OPENCL_CHECK(kernel.setArg(1, *bufferY));

      OPENCL_CHECK(kernel.setArg(2, output_size));
      OPENCL_CHECK(kernel.setArg(3, channels));

      OPENCL_CHECK(kernel.setArg(4, width));
      OPENCL_CHECK(kernel.setArg(5, height));
      OPENCL_CHECK(kernel.setArg(6, depth));

      OPENCL_CHECK(kernel.setArg(7, pooled_width));
      OPENCL_CHECK(kernel.setArg(8, pooled_height));
      OPENCL_CHECK(kernel.setArg(9, pooled_depth));

      OPENCL_CHECK(kernel.setArg(10, kernel_width));
      OPENCL_CHECK(kernel.setArg(11, kernel_height));
      OPENCL_CHECK(kernel.setArg(12, kernel_depth));

      OPENCL_CHECK(kernel.setArg(13, stride_width));
      OPENCL_CHECK(kernel.setArg(14, stride_height));
      OPENCL_CHECK(kernel.setArg(15, stride_depth));

      OPENCL_CHECK(kernel.setArg(16, pad_top));
      OPENCL_CHECK(kernel.setArg(17, pad_low));
      OPENCL_CHECK(kernel.setArg(18, pad_depth));
      break;

    default:
      CAFFE_THROW("Unsupported pooling size: ", kernel_.size());
      return false;
  }

  struct timeval tv;

  gettimeofday(&tv, NULL);
  long long start = tv.tv_sec * 1000000 + tv.tv_usec;

  cl::Event event;
  context_.enqueue(
                kernel,
                cl::NullRange,
                cl::NDRange(output_size),
                cl::NullRange,
                NULL,
                &event);

  gettimeofday(&tv, NULL);
  long long end = tv.tv_sec * 1000000 + tv.tv_usec;

  std::stringstream outstr;
  outstr << "PoolOp " << end << " cpu time delta: " << end - start;
  outstr << " N: " << samples << " C: " << channels;
  outstr << " k_size: " << kernel_.size() << " outputsize: " << output_size;
  outstr << " X " << X.size() << " Y " << Y->size();
  context_.LogProfilingInfo(event, outstr.str());

  return true;
}


template<>
bool PoolOp<float, OpenCLContext, MaxPool>::RunOnDeviceWithOrderNHWC(){

  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<OpenCLContext>::SetOutputSize(X, Y, X.dim32(1));

  return true;
}


REGISTER_OPENCL_OPERATOR(MaxPool, PoolOp<float, OpenCLContext, MaxPool>);
}//namespace caffe2

