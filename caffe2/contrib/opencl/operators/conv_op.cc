#include "conv_op.h"
#include "operator.h"

namespace caffe2 {
namespace {

REGISTER_OPENCL_OPERATOR(Conv, ConvOp<float>);

} // namespace
} // namespace caffe2
