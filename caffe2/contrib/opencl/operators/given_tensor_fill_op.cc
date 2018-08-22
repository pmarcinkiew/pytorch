
#include "context.h"
#include "operator.h"
#include "caffe2/operators/given_tensor_fill_op.h"

namespace caffe2 {

REGISTER_OPENCL_OPERATOR(GivenTensorFill, GivenTensorFillOp<float, OpenCLContext>);
REGISTER_OPENCL_OPERATOR(
    GivenTensorDoubleFill,
    GivenTensorFillOp<double, OpenCLContext>);
REGISTER_OPENCL_OPERATOR(GivenTensorBoolFill, GivenTensorFillOp<bool, OpenCLContext>);
REGISTER_OPENCL_OPERATOR(GivenTensorIntFill, GivenTensorFillOp<int, OpenCLContext>);
REGISTER_OPENCL_OPERATOR(
    GivenTensorInt64Fill,
    GivenTensorFillOp<int64_t, OpenCLContext>);
REGISTER_OPENCL_OPERATOR(
    GivenTensorStringFill,
    GivenTensorFillOp<std::string, OpenCLContext>);
}
