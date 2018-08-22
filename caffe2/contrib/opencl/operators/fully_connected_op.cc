
#include "context.h"
#include "operator.h"

#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {

REGISTER_OPENCL_OPERATOR(FC, FullyConnectedOp<OpenCLContext>);

} // namespace caffe2
