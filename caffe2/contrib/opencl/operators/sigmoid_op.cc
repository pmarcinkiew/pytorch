
#include "caffe2/contrib/opencl/operators/sigmoid_op.h"
#include "caffe2/core/tensor.h"

#include "context.h"
#include "operator.h"

namespace caffe2 {

static constexpr const char* kSigmoidOp = R"CLC(
  kernel void kSigmoidOp(
    const global float* X_in,
    global float* Y_out
  ) {
    const int idx_n = get_global_id(0);

    Y_out[idx_n] = 1.0 / (1.0 + exp(-X_in[idx_n]));

  }
)CLC";

template <typename T>
bool SigmoidOp<T>::RunOnDevice()
{
  auto kernel = context_.BuildKernelCached(kSigmoidOp, "", "kSigmoidOp");

  const Tensor<OpenCLContext>& X = Input(0);
  Tensor<OpenCLContext>* Y = Output(0);

  const int N = X.size();

  CAFFE_ENFORCE_GE(N, 1, "N must be >= 1 ");

  Y->ResizeLike(X);

  cl::Buffer* xBuffer = (cl::Buffer*)X.data<float>();
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();

  OPENCL_CHECK(kernel.setArg(0, *xBuffer));
  OPENCL_CHECK(kernel.setArg(1, *yBuffer));

  cl::Event event;
  context_.enqueue(
    kernel,
    cl::NullRange,
    cl::NDRange(N),
    cl::NullRange,
    NULL,
    &event);
  return true;
}

REGISTER_OPENCL_OPERATOR(
    Sigmoid,
    SigmoidOp<float>);

}  // namespace caffe2

