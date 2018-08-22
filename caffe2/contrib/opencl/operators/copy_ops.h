#ifndef CAFFE2_CONTRIB_OPENCL_OPERATORS_COPV_OPS_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATORS_COPY_OPS_H_

#include "context.h"

namespace caffe2 {

template <class Context>
class CopyToOpenCLOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(CopyToOpenCLOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
};

} // namespace caffe2
#endif // CAFFE2_CONTRIB_OPENCL_OPERATORS_COPY_OPS_H_
