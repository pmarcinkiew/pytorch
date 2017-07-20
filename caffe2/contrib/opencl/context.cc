#include "context.h"

#include "caffe2/core/allocator.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/context.h"
#include "caffe2/contrib/opencl/cl_utils.h"

#include <cstdlib>

namespace caffe2 {

CAFFE_KNOWN_TYPE(Tensor<OpenCLContext>);

OpenCLContextSingleton::OpenCLContextSingleton() {
  const auto platform_id = 0;
  const auto device_id = 0;

  profiling_info_enabled_ = (std::getenv("PROFILING_INFO") != nullptr);

  auto platforms = std::vector<cl::Platform>();
  OPENCL_CHECK(cl::Platform::get(&platforms));
  if (platforms.size() == 0 || platform_id >= platforms.size()) {
    CAFFE_THROW("Cannot find platform for OpenCL.");
  }
  platform = platforms[platform_id];

  devices = std::vector<cl::Device>();
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (devices.size() == 0 || device_id >= devices.size()) {
    CAFFE_THROW("Cannot find OpenCL compatible device.");
  }
  device = devices[device_id];

  context  = cl::Context({device});

  for (int i = 0; i < 32; i++) {
    available_queues.push_back(cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE));
  }
  queue = available_queues[0];
}

OpenCLContextSingleton& OpenCLContextSingleton::getInstance() {
  static OpenCLContextSingleton* instance;
  if (instance == nullptr) {
    instance = new OpenCLContextSingleton();
  }
  return *instance;
}

void OpenCLContextSingleton::PrintProfilingLogs()
{
  for(auto p: events_profiling_log)
    cl_utils_log_kernel_times(p.second, p.first);
  events_profiling_log.clear();
}

void OpenCLContextSingleton::LogProfilingInfo(const cl::Event& ev, const std::string& str)
{
  if (!profiling_info_enabled_)
    return;

  events_profiling_log.push_back(std::make_pair(ev, str));
}

std::pair<void*, MemoryDeleter> OpenCLContext::New(size_t nbytes) {
  auto& ctx = GetSingleton();
  cl_int err = 0;

  cl::Buffer* buffer = new cl::Buffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      nbytes, nullptr, &err);
  OPENCL_CHECK(err);
  // TODO(bwasti): use host ptr if possible to make CopyBytes free
  return std::make_pair((void *)buffer, OpenCLContext::Delete);
}

template <>
void OpenCLContext::CopyBytes<OpenCLContext, CPUContext>(size_t nbytes, const void *src, void *dst) {
  auto& ctx = GetSingleton();
  cl::copy(ctx.queue, *((cl::Buffer*)src), static_cast<char*>(dst), static_cast<char*>(dst) + nbytes);
}

template <>
void OpenCLContext::enqueueCopyBytes<OpenCLContext, CPUContext>(size_t nbytes, const void *src, void *dst) {
  auto& ctx = GetSingleton();
  ctx.queue.enqueueReadBuffer(*((cl::Buffer*)src), CL_FALSE, 0, nbytes, static_cast<char*>(dst));
}

template <>
void OpenCLContext::enqueueCopyBytes<CPUContext, OpenCLContext>(size_t nbytes, const void *src, void *dst) {
  auto& ctx = GetSingleton();
  ctx.queue.enqueueWriteBuffer(*((cl::Buffer*)(dst)), CL_FALSE, 0, nbytes, static_cast<const char*>(src));
}


template <>
void OpenCLContext::CopyBytes<CPUContext, OpenCLContext>(size_t nbytes, const void *src, void *dst) {
  auto& ctx = GetSingleton();
  char * data = new char[nbytes];
  std::memcpy(data, src, nbytes);
  cl::Event event;
  ctx.queue.enqueueWriteBuffer(*((cl::Buffer*)(dst)), CL_FALSE, 0, nbytes, static_cast<const char*>(data), nullptr, &event);
  event.setCallback(CL_COMPLETE, [](cl_event, cl_int, void *dataptr) {delete[] (char*)dataptr;}, data);
}

template <>
void OpenCLContext::CopyBytes<OpenCLContext, OpenCLContext>(size_t nbytes, const void *src, void *dst) {
  vector<char> tmp(nbytes);
  CopyBytes<OpenCLContext, CPUContext>(nbytes, src, (void*)&tmp[0]);
  CopyBytes<CPUContext, OpenCLContext>(nbytes, (void*)&tmp[0], dst);
}
template <>
void OpenCLContext::CopyBytes<CPUContext, CPUContext>(size_t nbytes, const void *src, void *dst) {
  memcpy(dst, src, nbytes);
}

void OpenCLContext::LogProfilingInfo(const cl::Event& ev, const std::string& str)
{
  auto& ctx = GetSingleton();
  ctx.LogProfilingInfo(ev, str);
}

void OpenCLContext::Delete(void *ptr) {
  delete (cl::Buffer *)ptr;
}

struct OpenCLContextSingleton& OpenCLContext::GetSingleton() {
  return OpenCLContextSingleton::getInstance();
}

cl::Kernel OpenCLContext::BuildKernel(const char* src, std::string additional_options, const char* fn_name) {
  auto& ctx = GetSingleton();

  cl::Program::Sources source(1,
      std::make_pair(src, strlen(src)));

  cl_int err = CL_SUCCESS;
  cl::Program p = cl::Program(ctx.context, source, &err);
  OPENCL_CHECK(err);

  std::string options = "-cl-std=CL1.1 -cl-fast-relaxed-math -cl-single-precision-constant ";
  options += additional_options;

  // TODO support more than one device
  // this will involve checking a compiler exists on each device
  vector<cl::Device> devices_{ctx.device};
  err = p.build(devices_, options.c_str());
  cl_build_status build_status = p.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(ctx.device);
  if (err != CL_SUCCESS || build_status != CL_BUILD_SUCCESS) {
    auto str = p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device);
    LOG(ERROR) << "Error code: " << err << " Build status: " << build_status;
    CAFFE_THROW(str);
  }

  auto kernel = cl::Kernel(p, fn_name, &err);
  OPENCL_CHECK(err);
  return kernel;
}

cl::Kernel OpenCLContext::BuildKernelCached(const char *src, std::string compile_options, const char *kernel_function_name) {
  std::stringstream cacheId;
  cacheId << (void*)src << "|" << compile_options << "|" << kernel_function_name;
  return BuildKernelCachedId(cacheId.str(), src, compile_options, kernel_function_name);
}

cl::Kernel OpenCLContext::BuildKernelCachedId(const std::string &cacheId, const char *src,
                                              std::string compile_options, const char *kernel_function_name) {

  auto& kernel_cache_ = GetSingleton().kernel_cache_;
  auto kernelIt = kernel_cache_.find(cacheId);
  if (kernelIt == kernel_cache_.end()) {
    if (!kernel_function_name)
      kernelIt = kernel_cache_.emplace(cacheId, BuildKernel(src, compile_options)).first;
    else
      kernelIt = kernel_cache_.emplace(cacheId, BuildKernel(src, compile_options, kernel_function_name)).first;
  }
  return kernelIt->second;
}

std::string OpenCLContext::BuildArgumentList(std::vector<std::pair<std::string, std::string>> args) {
  std::string out = " "; // There may be args before this
  for (auto arg : args) {
    out += "-D " + arg.first + "=" + arg.second + " ";
  }
  return out;
}

void EventCreateOPENCL(const DeviceOption& /* unused */, Event* /* unused */) {}
void EventRecordOPENCL(
    Event* /* unused */,
    const void* /* unused */,
    const char* /* unused */) {}
void EventWaitOPENCL(const Event* /* unused */, void* /* unused */) {}
void EventFinishOPENCL(const Event* /* unused */) {}
void EventResetOPENCL(Event* /* unused */) {}

REGISTER_EVENT_CREATE_FUNCTION(OPENCL, EventCreateOPENCL);
REGISTER_EVENT_RECORD_FUNCTION(OPENCL, EventRecordOPENCL);
REGISTER_EVENT_WAIT_FUNCTION(OPENCL, OPENCL, EventWaitOPENCL);
REGISTER_EVENT_FINISH_FUNCTION(OPENCL, EventFinishOPENCL);
REGISTER_EVENT_RESET_FUNCTION(OPENCL, EventResetOPENCL);

} // namespace caffe2
