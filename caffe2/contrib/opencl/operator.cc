#include "operator.h"

//#include "caffe2/core/common.h"
//#include "caffe2/core/registry.h"
#include "caffe2/core/operator.h"
//#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/operator_c10wrapper.h"
namespace caffe2 {

C10_DEFINE_REGISTRY(
    OpenCLOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
CAFFE_REGISTER_DEVICE_TYPE(OPENCL, OpenCLOperatorRegistry);

} // namespace caffe2
