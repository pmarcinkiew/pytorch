
#include "caffe2/contrib/opencl/operator.h"
#include "caffe2/contrib/opencl/context.h"
#include "caffe2/operators/prelu_op.h"

#include <sys/time.h>

namespace caffe2 {

static constexpr const char* kParametricReluOpSharedWeights = R"CLC(
  kernel void kParametricReluOpSharedWeights(
    const global float* X_in,
    const global float* weights,
    global float* Y_out
  ) {
      const float w = weights[0];
      const int ec_ix = get_global_id(0);
      const int c = get_global_id(1);
      const int n = get_global_id(2);
      const int EC = get_global_size(0);
      const int C = get_global_size(1);

      float x = X_in[n * C * EC + c * EC + ec_ix];

      Y_out[n * C * EC + c * EC + ec_ix] = max(x, 0.0) + min(x, 0.0) * w;
  }
  )CLC";

static constexpr const char* kParametricReluOp = R"CLC(
  kernel void kParametricReluOp(
    const global float* X_in,
    const global float* weights,
    global float* Y_out
  ) {
      const int ec_ix = get_global_id(0);
      const int c = get_global_id(1);
      const int n = get_global_id(2);
      const int EC = get_global_size(0);
      const int C = get_global_size(1);

      float x = X_in[n * C * EC + c * EC + ec_ix];

      Y_out[n * C * EC + c * EC + ec_ix] = max(x, 0.0) + min(x, 0.0) * weights[c];
  }
  )CLC";

template <>
bool PReluOp<float, OpenCLContext>::RunOnDevice() {

  const Tensor<OpenCLContext>& X = Input(0);
  const Tensor<OpenCLContext>& W = Input(1);

  Tensor<OpenCLContext>* Y = Output(0);
  Y->ResizeLike(X);

  if (order_ != StorageOrder::NCHW)
    CAFFE_THROW("Only NCHW order is supported for now");

  const bool C_shared = (W.size() == 1);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int elements_count = X.size_from_dim(2);

  if (!C_shared)
    CAFFE_ENFORCE_EQ(C, W.size());

  cl::Buffer* xBuffer = (cl::Buffer*)X.data<float>();
  cl::Buffer* wBuffer = (cl::Buffer*)W.data<float>();

  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();

  cl::Kernel kernel;
  if (C_shared) {
    kernel = context_.BuildKernelCached(kParametricReluOpSharedWeights,
                                        "", "kParametricReluOpSharedWeights");
  } else {
    kernel = context_.BuildKernelCached(kParametricReluOp,
                                        "", "kParametricReluOp");
  }

  OPENCL_CHECK(kernel.setArg(0, *xBuffer));
  OPENCL_CHECK(kernel.setArg(1, *wBuffer));
  OPENCL_CHECK(kernel.setArg(2, *yBuffer));

  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long start = tv.tv_sec * 1000000 + tv.tv_usec;

  cl::Event event;
  context_.enqueue(
    kernel,
    cl::NullRange,
    cl::NDRange(elements_count, C, N), //global dim is the image pixels count x number of image channels x number of images
    cl::NullRange,
    NULL,
    &event);

  gettimeofday(&tv, NULL);
  long long end = tv.tv_sec * 1000000 + tv.tv_usec;

  std::stringstream outstr;
  outstr << "PReluOp " << end << " cpu time delta: " << end - start;
  outstr <<" N: " << N << " C: " << C << " elem_count: " << elements_count;
  context_.LogProfilingInfo(event, outstr.str());

  return true;
}

REGISTER_OPENCL_OPERATOR(PRelu, PReluOp<float, OpenCLContext>);

} // namespace caffe2

