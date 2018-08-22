
#include "caffe2/operators/leaky_relu_op.h"

#include "context.h"
#include "operator.h"

namespace caffe2 {
static constexpr const char* kLeakyRelu = R"CLC(
kernel void kLeakyRelu(
  const global float* X,
  global float* Y,
  const float alpha
) {
    int ix = get_global_id(0);
    float x = X[ix];

    if (x < 0.0)
     x *= alpha;

    Y[ix] = x;
  }
)CLC";

template <>
bool LeakyReluOp<float, OpenCLContext>::RunOnDevice() {

  const auto& X = Input(0);
  const int N = X.dim32(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

  auto kernel = context_.BuildKernelCached(kLeakyRelu, "", "kLeakyRelu");

  cl::Buffer* xBuffer = (cl::Buffer*)X.data<float>();
  cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();

  OPENCL_CHECK(kernel.setArg(0, *xBuffer));
  OPENCL_CHECK(kernel.setArg(1, *yBuffer));
  OPENCL_CHECK(kernel.setArg(2, alpha_));

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

REGISTER_OPENCL_OPERATOR(LeakyRelu, LeakyReluOp<float, OpenCLContext>);

}
