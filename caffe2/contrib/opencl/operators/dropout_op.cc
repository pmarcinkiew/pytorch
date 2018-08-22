#include "caffe2/contrib/opencl/context.h"
#include "caffe2/contrib/opencl/operator.h"
#include "caffe2/operators/dropout_op.h"

namespace caffe2 {

template <>
bool DropoutOp<float, OpenCLContext>::RunOnDevice() {

  const Tensor<OpenCLContext>& X = Input(0);
  Tensor<OpenCLContext>* Y = Output(0);
  Y->ResizeLike(X);

  if (is_test_) {
    if (Y != &X) {

      cl::Buffer* xBuffer = (cl::Buffer*)X.data<float>();
      cl::Buffer* yBuffer = (cl::Buffer*)Y->mutable_data<float>();

      context_.CopyCL(xBuffer, yBuffer, X.size() * sizeof(float));
    }
    return true;
  } else {
    CAFFE_THROW("Training mode not implemented.");
  }
}

REGISTER_OPENCL_OPERATOR(Dropout, DropoutOp<float, OpenCLContext>);
}
