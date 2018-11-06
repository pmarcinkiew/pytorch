#ifndef CAFFE2_CONTRIB_OPENCL_NET_ASYNC_DAG_OPENCL_H_
#define CAFFE2_CONTRIB_OPENCL_NET_ASYNC_DAG_OPENCL_H_

#include "c10/util/Registry.h"
#include "caffe2/core/common.h"
#include "caffe2/core/net_dag.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

class AsyncDAGOpenCLNet : public DAGNetBase {
 public:
  AsyncDAGOpenCLNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  bool SupportsAsync() override {
    return true;
  }
  bool RunAt(int queue_id, const std::vector<int>& chain) override;

 protected:
  bool DoRunAsync() override;

  C10_DISABLE_COPY_AND_ASSIGN(AsyncDAGOpenCLNet);
};

} // namespace caffe2



#endif /* CAFFE2_CONTRIB_OPENCL_NET_ASYNC_DAG_OPENCL_H_ */
