
#include "caffe2/contrib/opencl/operator.h"
#include "caffe2/operators/utility_ops.h"
#include "caffe2/contrib/opencl/context.h"

namespace caffe2 {
REGISTER_OPENCL_OPERATOR(Sum, SumOp<OpenCLContext>);
}
