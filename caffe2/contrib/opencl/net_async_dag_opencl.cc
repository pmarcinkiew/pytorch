#include "caffe2/contrib/opencl/net_async_dag_opencl.h"
#include "caffe2/contrib/opencl/context.h"

namespace caffe2 {

AsyncDAGOpenCLNet::AsyncDAGOpenCLNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : DAGNetBase(net_def, ws) {
  VLOG(1) << "Constructing AsyncDAGOpenCLNet " << net_def->name();

  // For all chains, their tail should consist the list of events that we are
  // needing for synchronization in the Run() interface, unless there are other
  // chains depending on it.
  events_.reserve(execution_chains_.size());
  for (const auto& chain : execution_chains_) {
    const int tail_op_idx = chain.second.back();
    if (operator_nodes_[tail_op_idx].children_.empty()) {
      events_.push_back(&operator_nodes_[tail_op_idx].operator_->event());
    }
  }

  VLOG(1) << "Total " << execution_chains_.size()
          << " chains, final waiting on " << events_.size() << " events";
}

bool AsyncDAGOpenCLNet::DoRunAsync() {
  StartAllObservers();

  success_ = true;
  // Initialize the runtime parent count.
  for (auto& node : operator_nodes_) {
    node.runtime_parent_count_ = node.parents_.size();
  }

  // First, set up job queue.
  remaining_ops_ = operator_nodes_.size();
  std::queue<int> jobs;

  // Kickstart the job queue.
  for (auto& value : initial_frontier_) {
    jobs.push(value);
  }


  // An infinite loop until there are no more jobs to run.
  while (true) {
    int idx = 0;

    if (jobs.empty())
      break;

    idx = jobs.front();
    jobs.pop();

    VLOG(1) << "Running chain starting at operator #" << idx << " "
            << operator_nodes_[idx].operator_->debug_def().name() << "("
            << operator_nodes_[idx].operator_->debug_def().type() << ").";
    CAFFE_ENFORCE(
        execution_chains_.find(idx) != execution_chains_.end(),
        "Can't find chain ",
        idx,
        ".");
    bool this_success = false;
    try {
      this_success = RunAt(idx, execution_chains_[idx]);

      if (!this_success) {
        // If an exception was thrown, the operator def will get printed
        // by Operator::Run[Async], but if no exception occurs we print it here.
        LOG(ERROR) << "Operator chain failed starting at: "
                   << ProtoDebugString(
                          operator_nodes_[idx].operator_->debug_def());
      }
    } catch (std::exception& e) {
      //std::string exception_str = GetExceptionString(e);
      //HandleException(idx, exception_str);
    } catch (...) {
      std::string exception_str = "Unknown exception";
      HandleException(idx, exception_str);
    }

    // Do book-keeping
    std::vector<int> chains_to_queue;
    const auto& chain = execution_chains_[idx];
    for (const auto idx : chain) {
      for (const auto child : operator_nodes_[idx].children_) {
        const int count = --operator_nodes_[child].runtime_parent_count_;

        if (count != 0) {
          continue;
        }

        if (operator_nodes_[child].is_chain_start_) {
          VLOG(2) << "Pushing chain #" << child << " to queue.";
          chains_to_queue.push_back(child);
        }
      }
    }

    {

      remaining_ops_ -= chain.size();
      CAFFE_ENFORCE(remaining_ops_ >= 0);
      success_ &= this_success;

      // Break if this or any other operator chain failed.
      if (!success_) {
        break;
      }

      // Queue follow up operator chains.
      for (const auto idx : chains_to_queue) {
        jobs.push(idx);
      }
    }

    VLOG(2) << "Finished executing operator #" << idx;
  }

  VLOG(2) << "All ops finished running.";

  StopAllObservers();
  // If the above while loop finished, we know that the current run finished.
  return success_;
}

bool AsyncDAGOpenCLNet::RunAt(int queue_id, const std::vector<int>& chain) {
  CAFFE_ENFORCE(!chain.empty(), "Chain should not be empty.");

  const auto source_idx = chain.front();
  const auto& parents = operator_nodes_[source_idx].parents_;

  std::vector<const Event*> parent_events;
  parent_events.reserve(operator_nodes_[source_idx].parents_.size());
  for (auto source_parent_idx : operator_nodes_[source_idx].parents_) {
    parent_events.push_back(
        &operator_nodes_[source_parent_idx].operator_->event());
    LOG(INFO) << " waiting on events from parent: " << source_parent_idx;
  }

  operator_nodes_[source_idx].operator_->WaitEvents(parent_events, queue_id);

  // We've waited on all our parent indices.
  bool success = true;

  for (auto idx : chain) {

      Operator<OpenCLContext> *op_ptr =
          dynamic_cast<Operator<OpenCLContext> *>(operator_nodes_[idx].operator_.get());
       
      OpenCLContext *ctx = const_cast<OpenCLContext *>(op_ptr->getContext());
      /*op_ptr->SwitchToStream(queue_id);*/
      ctx->SwitchToDevice(queue_id);
      auto result = op_ptr->RunOnDevice();

  }

  const auto& last_idx = chain.back();
  Operator<OpenCLContext> *last_op_in_chain_ptr =
            dynamic_cast<Operator<OpenCLContext> *>(operator_nodes_[last_idx].operator_.get());
  //last_op_in_chain_ptr->RecordEvent();


  return success;
}


REGISTER_NET(async_dag_opencl, AsyncDAGOpenCLNet);

} // namespace caffe2
