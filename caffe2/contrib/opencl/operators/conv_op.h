#ifndef CAFFE2_CONTRIB_OPENCL_OPERATORS_CONV_OP_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATORS_CONV_OP_H_

#include "caffe2/operators/conv_pool_op_base.h"

#include "context.h"
#include "kernels/utils.h"

#include <sys/time.h>

namespace caffe2 {
namespace {

static constexpr const char* kConvOp = R"CLC(

#define X_NCHW(n,c,y,x) X[(n)*X_C*X_H*X_W+(c)*X_H*X_W+(y)*X_W+(x)]
#define Y_NCHW(n,c,y,x) Y[(n)*Y_C*Y_H*Y_W+(c)*Y_H*Y_W+(y)*Y_W+(x)]
#define K_NCHW(n,c,y,x) K[(n)*K_C*K_H*K_W+(c)*K_H*K_W+(y)*K_W+(x)]

  kernel void kConvOp(
    const global float* X,
    const global float* K,
    const global float* bias,
    global float* Y,
    const global float* norm_means,
    const global float* norm_variances,
    const global float* scale_bias
  ) {

    int y = get_global_id(0) / Y_W;
    int x = get_global_id(0) % Y_W;
    int k = get_global_id(1); //this tells which kernel is being used
    int n = get_global_id(2); //this tells which image out of the whole batch is being processed

    float out = 0.0f;

    for (int c = 0; c < K_C; c++) {
        for (int j = 0; j < K_H; j++) {
            for (int i = 0; i < K_W; i++) {
                int X_y = y * STRIDE_H - PAD_T + j;
                int X_x = x * STRIDE_W - PAD_L + i;
                if (X_x >= 0 && X_y >= 0 && X_x < X_W && X_y < X_H) {
                  out += K_NCHW(k, c, j, i) * X_NCHW(n, c, X_y, X_x);
                }
            }
        }
    }

#ifdef USE_NORMALIZATION
    out = (out - norm_means[k]) / (sqrt(norm_variances[k]) + 0.000001f);
#endif

#ifdef USE_SCALE_BIAS
    out *= scale_bias[k];
#endif

    out += bias[k];

    Y_NCHW(n, k, y, x) = out;
  }


  kernel void kConvOp_noStride_noPad(
    const global float* X,
    const global float* K,
    const global float* bias,
    global float* Y,
    const global float* norm_means,
    const global float* norm_variances,
    const global float* scale_bias
  ) {

    int y = get_global_id(0)/Y_W_fn;
    int x = get_global_id(0)%Y_W_fn;
    x *= fn;

    int k = get_global_id(1); //this tells which kernel is being used
    int n = get_global_id(2); //this tells which image out of the whole batch is being processed

    float4 out = 0.0f;

    for(int c = 0; c < K_C; c++) {
        for(int j = 0; j < K_H; j++) {
            float4 xv_lo, xv_hi, kv;
            for(int i = 0; i < K_W; i++) {
                if ((i % 4) == 0) {
                    xv_lo = vload4(0, &X_NCHW(n,c,y+j,x+i));
                    xv_hi = vload4(0, &X_NCHW(n,c,y+j,x+i+4));
                    kv    = vload4(0, &K_NCHW(k,c,j,i));
                }
                out += xv_lo*kv.x;
                //rotate left
                xv_lo = xv_lo.s1230;
                xv_lo.s3 = xv_hi.s0;
                xv_hi = xv_hi.s1230;
                kv = kv.s1230;
            }
        }
    }

#ifdef USE_NORMALIZATION
    out = (out - norm_means[k]) / (sqrt(norm_variances[k]) + 0.000001f);
#endif

#ifdef USE_SCALE_BIAS
    out *= scale_bias[k];
#endif

    out += bias[k];

    if (Y_W-x >= 4)      vstore4(out, 0, &Y_NCHW(n,k,y,x));
    else if (Y_W-x >= 3) vstore3(out.xyz, 0, &Y_NCHW(n,k,y,x));
    else if (Y_W-x >= 2) vstore2(out.xy, 0, &Y_NCHW(n,k,y,x));
    else if (Y_W-x >= 1) Y_NCHW(n,k,y,x) = out.x;
  }

)CLC";


template <typename T> // only float is supported
class ConvOp final : public ConvPoolOpBase<OpenCLContext> {
 public:
  ConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<OpenCLContext>(operator_def, ws),
        use_scale_bias_(OperatorBase::GetSingleArgument<bool>("use_scale", false)),
        use_normalization_(OperatorBase::GetSingleArgument<bool>("use_normalization", false)) {
  }

  bool RunOnDeviceWithOrderNCHW() override {

    const Tensor<OpenCLContext>& X = Input(INPUT);
    const Tensor<OpenCLContext>& filter = Input(FILTER);
    const Tensor<OpenCLContext>& bias = Input(BIAS);

    Tensor<OpenCLContext>* Y = Output(OUTPUT);

    const int X_N = X.dim32(0);
    const int X_C = X.dim32(1);
    const int X_H = X.dim32(2);
    const int X_W = X.dim32(3);

    CAFFE_ENFORCE_GE(X_H, 1, "Input Height must be >= 1 ");
    CAFFE_ENFORCE_GE(X_W, 1, "Input Width must be >= 1 ");
    CAFFE_ENFORCE_GE(X_H, kernel_h(), "Input H must be >= kernel_h ", kernel_h());
    CAFFE_ENFORCE_GE(X_W, kernel_w(), "Input W must be >= kernel_w ", kernel_w());
    CAFFE_ENFORCE_GE(X_N, 1, "N must be >= 1 ");
    CAFFE_ENFORCE_GE(X_C, 1, "Channels count must be >= 1");
    CAFFE_ENFORCE_LE(X.ndim(), 4, "Input size must be at most 2D (H x W)");
    CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim(),
        "Number of dimensions in kernel must match the input dimensions");
    CAFFE_ENFORCE_EQ(X_C, filter.dim32(1),
        "Number of channels must match the input channel number");
    CAFFE_ENFORCE_EQ(kernel_h(), filter.dim32(2),
        "Filter dim(2) must be equal to kernel_h size ", kernel_h());
    CAFFE_ENFORCE_EQ(kernel_w(), filter.dim32(3),
        "Filter dim(3) must be equal to kernel_w size ", kernel_w());
    CAFFE_ENFORCE_LE(dilation_h(), 1, "OpenCL currently only supports dilation_h equal to 1");
    CAFFE_ENFORCE_LE(dilation_w(), 1, "OpenCL currently only supports dilation_w equal to 1");

    const int K_count = filter.dim32(0);
    CAFFE_ENFORCE_GE(K_count, 1, "Number of filters must be >= 1");

    const int Y_C = K_count;

    SetOutputSize(X, Y, Y_C);

    const int Y_N = Y->dim32(0);
    const int Y_H = Y->dim32(2);
    const int Y_W = Y->dim32(3);

    CAFFE_ENFORCE_GE(Y_H, 1, "Image height is too small. Calculated output height: ", Y_H);
    CAFFE_ENFORCE_GE(Y_W, 1, "Image width is too small. Calculated output width: ", Y_W);

    std::vector<std::pair<std::string, std::string>> args;
    args.emplace_back("X_N",        std::to_string(X_N));
    args.emplace_back("X_C",        std::to_string(X_C));
    args.emplace_back("X_H",        std::to_string(X_H));
    args.emplace_back("X_W",        std::to_string(X_W));

    args.emplace_back("Y_N",        std::to_string(Y_N));
    args.emplace_back("Y_C",        std::to_string(Y_C));
    args.emplace_back("Y_H",        std::to_string(Y_H));
    args.emplace_back("Y_W",        std::to_string(Y_W));

    args.emplace_back("K_N",        std::to_string(K_count));
    args.emplace_back("K_C",        std::to_string(X_C));
    args.emplace_back("K_H",        std::to_string(kernel_h()));
    args.emplace_back("K_W",        std::to_string(kernel_w()));

    args.emplace_back("PAD_T",      std::to_string(pad_t()));
    args.emplace_back("PAD_B",      std::to_string(pad_b()));
    args.emplace_back("PAD_L",      std::to_string(pad_l()));
    args.emplace_back("PAD_R",      std::to_string(pad_r()));
    args.emplace_back("STRIDE_H",   std::to_string(stride_h()));
    args.emplace_back("STRIDE_W",   std::to_string(stride_w()));

    const int fn = 4;
    const int Y_W_fn = (Y_W + 3) / fn;
    args.emplace_back("fn",       std::to_string(fn));
    args.emplace_back("Y_W_fn",   std::to_string(Y_W_fn));

    if (use_scale_bias_) {
      args.emplace_back("USE_SCALE_BIAS", "1");
    }
    if (use_normalization_) {
      args.emplace_back("USE_NORMALIZATION", "1");
    }

    const char * kernel_name;
    cl::NDRange global_NDRange;
    if (pad_t() == 0 && pad_b() == 0 && pad_l() == 0 && pad_l() == 0 && pad_r() == 0
        && stride_w() == 1 && stride_h() == 1) {
      kernel_name = "kConvOp_noStride_noPad";
      global_NDRange = cl::NDRange(Y_H * Y_W_fn, K_count, X_N);
    } else {
      kernel_name = "kConvOp";
      global_NDRange = cl::NDRange(Y_H * Y_W, K_count, X_N);
    }

    std::string arg_list = context_.BuildArgumentList(args);
    cl::Kernel kernel = context_.BuildKernelCached(kConvOp, arg_list, kernel_name);

    cl::Buffer* xBuffer = (cl::Buffer*)X.data<float>();
    cl::Buffer* wBuffer = (cl::Buffer*)filter.data<float>();
    cl::Buffer* bBuffer = (cl::Buffer*)bias.data<float>();
    cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();

    OPENCL_CHECK(kernel.setArg(0, *xBuffer));
    OPENCL_CHECK(kernel.setArg(1, *wBuffer));
    OPENCL_CHECK(kernel.setArg(2, *bBuffer));
    OPENCL_CHECK(kernel.setArg(3, *yBuffer));

    int next_input = BIAS + 1;

    if (use_scale_bias_) {
      const Tensor<OpenCLContext> &scaleBias = Input(next_input++);

      CAFFE_ENFORCE_EQ(scaleBias.ndim(), K_count, "Size of scale bias vector must be equal to number of kernels");

      cl::Buffer* scaleBiasBuffer = (cl::Buffer*)scaleBias.data<float>();
      OPENCL_CHECK(kernel.setArg(6, *scaleBiasBuffer));
    } else {
      OPENCL_CHECK(kernel.setArg(6, nullptr));
    }

    if (use_normalization_) {
      const Tensor<OpenCLContext> &normMean = Input(next_input++);
      const Tensor<OpenCLContext> &normVariance = Input(next_input++);

      CAFFE_ENFORCE_EQ(normMean.ndim(), K_count, "Size of normalization means vector must be equal to number of kernels");
      CAFFE_ENFORCE_EQ(normVariance.ndim(), K_count, "Size of normalization variances vector must be equal to number of kernels");

      cl::Buffer* normMeanBuffer = (cl::Buffer*)normMean.data<float>();
      cl::Buffer* normVarBuffer = (cl::Buffer*)normVariance.data<float>();

      OPENCL_CHECK(kernel.setArg(4, *normMeanBuffer));
      OPENCL_CHECK(kernel.setArg(5, *normVarBuffer));
    } else {
      OPENCL_CHECK(kernel.setArg(4, nullptr));
      OPENCL_CHECK(kernel.setArg(5, nullptr));
    }

    struct timeval tv;

    gettimeofday(&tv, NULL);
    long long start = tv.tv_sec * 1000000 + tv.tv_usec;

    auto& ctx = context_.GetSingleton();
    cl::Event event;
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_NDRange, //global dim is the output size x number of kernels x number of images
      cl::NullRange, //each work group services whole kernel per one output pixel
      NULL,
      &event));

    gettimeofday(&tv, NULL);
    long long end = tv.tv_sec * 1000000 + tv.tv_usec;

    std::stringstream outstr;
    outstr << "ConvOp " << end << " cpu time delta: " << end - start << " N: " << X_N << " C: " << X_C << " H: " << X_H
           << " W: " << X_W << " k_count: " << K_count << " k_size: " << kernel_w() << ", " << kernel_h();
    context_.LogProfilingInfo(event, outstr.str());

    return true;
  }


 private:
  bool use_scale_bias_;
  bool use_normalization_;

  INPUT_TAGS(INPUT, FILTER, BIAS);
  OUTPUT_TAGS(OUTPUT);
};

} // namespace
} // namespace caffe2
#endif // CAFFE2_CONTRIB_OPENCL_OPERATORS_CONV_OP_H_
