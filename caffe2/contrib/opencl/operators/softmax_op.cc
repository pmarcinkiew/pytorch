
#include "caffe2/operators/softmax_op.h"
#include "caffe2/core/tensor.h"

#include "context.h"
#include "operator.h"
#include <cstdio>
#include <sys/time.h>

namespace caffe2 {
static constexpr const char* kSoftmaxExp = R"CLC(
kernel void kSoftmaxExp(
  global float* x,
  global float* softmax,
  const int N,
  const int D
){
  int idx = get_global_id(0);
  if (idx < N * D) {
    int dim = idx / D * D;
    // row-wise max
    float max = x[idx];
    for (int i = 0; i < D; i++)
        if (x[dim+i] > max)
            max = x[dim+i];
    barrier(CLK_GLOBAL_MEM_FENCE);
    // Subtract the max (for numerical reasons) & Exponentiation
    // TODO similar to cpu version, but why subtracting?
    barrier(CLK_GLOBAL_MEM_FENCE);
    float sum_z_exp = 0;
    for (int i = 0; i < D; i++)
        sum_z_exp += exp(x[dim+i] - max);
    softmax[idx] = exp(x[idx] - max) / sum_z_exp;
  }
}
)CLC";


static constexpr const char* kVelociraptor = R"CLC(
kernel void kVelociraptor(
  global float* Xdata,
  global float* Ydata,
  const int o,
  const int c
){

    int oi = get_global_id(0);
    // NDRange loop
    //for (int oi = 0; oi<o; ++oi)
    {
      float sum = 0;
      for (int ci = 0; ci<c; ++ci)
      {
        sum+=exp(Xdata[ci * o + oi]);
      }

      for (int ci = 0; ci<c; ++ci)
      {
        Ydata[ci * o + oi] = exp(Xdata[ci * o + oi])/ sum;
      }

    }
}
)CLC";

template <>
bool SoftmaxOp<float, OpenCLContext>::RunOnDevice() {


  {

    auto kernel = context_.BuildKernelCached(kVelociraptor, "", "kVelociraptor");
    auto& X = Input(0);
    auto* Y = Output(0);
    auto shape = X.dims();


    if (shape.size() != 4 || shape.at(0) != 1 || shape.at(1)!= 2)
      goto velociraptor;



    CAFFE_ENFORCE_EQ(shape.size(), 4);
    CAFFE_ENFORCE_EQ(shape.at(0), 1);
    CAFFE_ENFORCE_EQ(shape.at(1), 2);


    int h = shape.at(2);
    int w = shape.at(3);
    int c = 2;
    int o = h*w;

    Y->ResizeLike(X);
  
    auto xBuffer = (cl::Buffer*)X.data<float>();
    auto yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  
    //const float* Xdata = X.data<float>();
    //float* Ydata = Y->mutable_data<float>();

    OPENCL_CHECK(kernel.setArg(0, *xBuffer));
    OPENCL_CHECK(kernel.setArg(1, *yBuffer));
    OPENCL_CHECK(kernel.setArg(2, o));
    OPENCL_CHECK(kernel.setArg(3, c));
  
  
  struct timeval tv;

  gettimeofday(&tv, NULL);
  long long start = tv.tv_sec * 1000000 + tv.tv_usec;
  
  cl::Event event;
  context_.enqueue(
    kernel,
    cl::NullRange,
    cl::NDRange(o),
    cl::NullRange,
    NULL,
    &event);

  gettimeofday(&tv, NULL);
  long long end = tv.tv_sec * 1000000 + tv.tv_usec;

  std::stringstream outstr;
  outstr << "SoftmaxOp_Velo "<< end << " cpu time delta: " << end - start;
  outstr << "X " << X.size() << " Y " << Y->size();
  outstr << " o " << o;
  outstr <<  " c " << c;
  context_.LogProfilingInfo(event, outstr.str());


    return true;
  }

  velociraptor:
{
  auto kernel = context_.BuildKernelCached(kSoftmaxExp, "", "kSoftmaxExp");
  auto& X = Input(0);
  auto* Y = Output(0);
  int axis_ = OperatorBase::GetSingleArgument<int>("axis", 1);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);
  Y->ResizeLike(X);
  auto xBuffer = (cl::Buffer*)X.data<float>();
  auto yBuffer = (cl::Buffer*)Y->mutable_data<float>();
  OPENCL_CHECK(kernel.setArg(0, *xBuffer));
  OPENCL_CHECK(kernel.setArg(1, *yBuffer));
  OPENCL_CHECK(kernel.setArg(2, N));
  OPENCL_CHECK(kernel.setArg(3, D));
  
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long start = tv.tv_sec * 1000000 + tv.tv_usec;

  cl::Event event;
  context_.enqueue(
    kernel,
    cl::NullRange,
    cl::NDRange(X.size()),
    cl::NullRange,
    NULL,
    &event);

  gettimeofday(&tv, NULL);
  long long end = tv.tv_sec * 1000000 + tv.tv_usec;

  std::stringstream outstr;
  outstr << "SoftmaxOp_c2 "<< end << " cpu time delta: " << end - start;
  outstr << "X " << X.size() << " Y " << Y->size();
  context_.LogProfilingInfo(event, outstr.str());

  return true;
  }
}

REGISTER_OPENCL_OPERATOR(Softmax, SoftmaxOp<float, OpenCLContext>);

}  // namespace caffe2
