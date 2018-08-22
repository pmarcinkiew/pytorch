
#include "caffe2/contrib/opencl/context.h"
#include "caffe2/contrib/opencl/operator.h"
#include "operator.h"
#include "caffe2/operators/filler_op.h"

namespace caffe2 {
REGISTER_OPENCL_OPERATOR(ConstantFill, ConstantFillOp<OpenCLContext>);
}
