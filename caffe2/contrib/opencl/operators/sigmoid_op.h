
#ifndef CAFFE2_CONTRIB_OPENCL_OPERATORS_SIGMOID_OP_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATORS_SIGMOID_OP_H_

#include "context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {

template <typename T>
class SigmoidOp final : public Operator<OpenCLContext> {
 public:
  SigmoidOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<OpenCLContext>(operator_def, ws) {}

  bool RunOnDevice() override;
};

} //caffe2

#endif /* CAFFE2_CONTRIB_OPENCL_OPERATORS_SIGMOID_OP_H_ */
