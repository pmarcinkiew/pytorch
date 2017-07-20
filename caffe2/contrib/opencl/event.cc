#include "context.h"
#include "caffe2/core/event_cpu.h"

#include <atomic>

namespace caffe2 {

struct OpenCLEventWrapper {
  explicit OpenCLEventWrapper(const DeviceOption& option)
      : status_(EventStatus::EVENT_INITIALIZED) {
    CAFFE_ENFORCE(option.device_type(), OPENCL);
  }
  ~OpenCLEventWrapper() {
  }

  cl::Event cl_event_;
  cl::CommandQueue::cl_type cl_queue_ptr_ = nullptr;

  std::atomic<int> status_;
  std::mutex mutex_recorded_;
  std::condition_variable cv_recorded_;
  std::string err_msg_;
};

namespace {
const std::string kNoError = "No error";
}


void EventCreateOPENCL(const DeviceOption& option, Event* event) {
  event->event_ = std::make_shared<OpenCLEventWrapper>(option);
}

void EventRecordOPENCL(Event* event, const void* ctx_ptr, const char* err_msg) {
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
    auto context = static_cast<const OpenCLContext*>(ctx_ptr);

    // Possible state changes:
    //  INITIALIZED -> SCHEDULED/FAILED
    //  SCHEDULED -> SUCCESS/FAILED
    //  SUCCESS/FAILED - terminal
    //
    // No further changes to cl_event_ and cl_queue_ptr_ after transitioning
    // from INITIALIZED
    // No further changes to err_msg_ after transitioning into FAILED

    CAFFE_ENFORCE_EQ(
        wrapper->status_,
        EventStatus::EVENT_INITIALIZED,
        "Calling Record multiple times");

    if (!err_msg) {
      context->queue().enqueueMarker(&wrapper->cl_event_);
      wrapper->cl_queue_ptr_ = context->queue()();
      wrapper->status_ = EventStatus::EVENT_SCHEDULED;
    } else {
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::EVENT_FAILED;
    }
  }
  wrapper->cv_recorded_.notify_all();
}

void EventFinishOPENCL(const Event* event) {
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
    while (wrapper->status_ == EventStatus::EVENT_INITIALIZED) {
      wrapper->cv_recorded_.wait(lock);
    }
  }

  if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
    auto status = wrapper->cl_event_.wait();
    if (status == CL_SUCCESS) {
      wrapper->status_ = EventStatus::EVENT_SUCCESS;
    } else {
      std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
      wrapper->err_msg_ = "OpenCL event wait error: " + std::to_string(status);
      wrapper->status_ = EventStatus::EVENT_FAILED;
    }
  }
}

void EventWaitOPENCLOPENCL(const Event* event, void* ctx_ptr) {
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
    while (wrapper->status_ == EventStatus::EVENT_INITIALIZED) {
      wrapper->cv_recorded_.wait(lock);
    }
  }

  if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
    auto context = static_cast<const OpenCLContext*>(ctx_ptr);
    if (context->queue()() != wrapper->cl_queue_ptr_) {
      std::vector<cl::Event> events = {wrapper->cl_event_};
      context->queue().enqueueBarrierWithWaitList(&events);
    }
  }
}

void EventWaitCPUOPENCL(const Event* event, void* context) {
  EventFinishOPENCL(event);
}

void EventWaitOPENCLCPU(const Event* event, void* context) {
  event->Finish(); // calls EventFinishCPU
}

EventStatus EventQueryOPENCL(const Event* event) {
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
    cl_int err;
    auto status = wrapper->cl_event_.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>(&err);
    if (err != CL_SUCCESS) {
      std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
      wrapper->err_msg_ = "OpenCL event query error: " + std::to_string(err);
      wrapper->status_ = EventStatus::EVENT_FAILED;
    } else if (status == CL_COMPLETE) {
      wrapper->status_ = EventStatus::EVENT_SUCCESS;
    }
  }
  return static_cast<EventStatus>(wrapper->status_.load());
}

const std::string& EventErrorMessageOPENCL(const Event* event) {
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  // supposed to be called after EventQueryCUDA to update status first
  if (wrapper->status_ == EventStatus::EVENT_FAILED) {
    return wrapper->err_msg_;
  } else {
    return kNoError;
  }
}

void EventSetFinishedOPENCL(const Event* event, const char* err_msg) {
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);

    CAFFE_ENFORCE_EQ(
        wrapper->status_,
        EventStatus::EVENT_INITIALIZED,
        "Calling SetFinished on recorded OpenCL event");

    if (!err_msg) {
      wrapper->status_ = EventStatus::EVENT_SUCCESS;
    } else {
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::EVENT_FAILED;
    }
  }
  wrapper->cv_recorded_.notify_all();
}

void EventResetOPENCL(Event* event) {
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
  wrapper->status_ = EventStatus::EVENT_INITIALIZED;
  wrapper->err_msg_ = "";
}


REGISTER_EVENT_CREATE_FUNCTION(OPENCL, EventCreateOPENCL);
REGISTER_EVENT_RECORD_FUNCTION(OPENCL, EventRecordOPENCL);
REGISTER_EVENT_WAIT_FUNCTION(OPENCL, OPENCL, EventWaitOPENCLOPENCL);
REGISTER_EVENT_WAIT_FUNCTION(CPU, OPENCL, EventWaitCPUOPENCL);
REGISTER_EVENT_WAIT_FUNCTION(OPENCL, CPU, EventWaitOPENCLCPU);

REGISTER_EVENT_FINISH_FUNCTION(OPENCL, EventFinishOPENCL);
REGISTER_EVENT_RESET_FUNCTION(OPENCL, EventResetOPENCL);

REGISTER_EVENT_QUERY_FUNCTION(OPENCL, EventQueryOPENCL);
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(OPENCL, EventErrorMessageOPENCL);
REGISTER_EVENT_SET_FINISHED_FUNCTION(OPENCL, EventSetFinishedOPENCL);


} // namespace caffe2
