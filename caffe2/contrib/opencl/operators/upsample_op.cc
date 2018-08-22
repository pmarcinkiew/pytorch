
#include "caffe2/contrib/opencl/operator.h"
#include "caffe2/contrib/opencl/operators/upsample_op.h"

namespace caffe2 {

REGISTER_OPENCL_OPERATOR(Upsample, UpsampleOp<float>);

OPERATOR_SCHEMA(Upsample)
    .NumInputs(1)
    .Input(0, "X", "The input data N x C x H x W")
    .NumOutputs(1)
    .Output(0, "Y", "The output N x C x Yh x Yw")
    .Arg("stride", "The output images will be stride times larger (in each direction) than the input. Default is 1.")
    .SetDoc(R"DOC(Upsample operator. Does not support output scaling.)DOC");

} // namespace caffe2
