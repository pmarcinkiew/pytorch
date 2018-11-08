#include "context.h"

#include "caffe2/core/allocator.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/context.h"
#include "caffe2/contrib/opencl/cl_utils.h"

#include <cstdlib>

namespace caffe2 {

//CAFFE_KNOWN_TYPE(Tensor<OpenCLContext>);

thread_local ThreadLocalOpenCLObjects OpenCLContext::openCL_objects_;
cl::Context OpenCLContext::context;
cl::Platform OpenCLContext::platform;
cl::Device OpenCLContext::computing_device;
std::vector<cl::Device> OpenCLContext::devices;

bool OpenCLContext::initialized = false;

class OpenCLKernelCache {
  std::mutex mutex_;
  std::unordered_map<std::string, cl::Kernel> kernel_cache_;
public:
  cl::Kernel GetKernel(const std::string& cacheId) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto kernelIt = kernel_cache_.find(cacheId);
    if (kernelIt == kernel_cache_.end()) {
      return cl::Kernel();
    }
    return kernelIt->second;
  }
  void SetKernel(const std::string& cacheId, cl::Kernel kernel) {
    std::unique_lock<std::mutex> lock(mutex_);
    kernel_cache_.emplace(cacheId, kernel);
  }
};

ThreadLocalOpenCLObjects::ThreadLocalOpenCLObjects() {
  profiling_info_enabled_ = (std::getenv("PROFILING_INFO") != nullptr);
}

cl::CommandQueue& ThreadLocalOpenCLObjects::GetQueue(int queue_id)
{
  if (queues.size() <= (unsigned)queue_id) {
    queues.resize(queue_id + 1);
  }
  if (queues[queue_id]() == nullptr) {
    queues[queue_id] = cl::CommandQueue(OpenCLContext::context, OpenCLContext::computing_device, profiling_info_enabled_ ? CL_QUEUE_PROFILING_ENABLE : 0);
  }
  return queues[queue_id];
}

void ThreadLocalOpenCLObjects::PrintProfilingLogs()
{
  for(auto p: events_profiling_log)
    cl_utils_log_kernel_times(p.second, p.first);
  events_profiling_log.clear();
}

void ThreadLocalOpenCLObjects::LogProfilingInfo(const cl::Event& ev, const std::string& str)
{
  if (!profiling_info_enabled_)
    return;

  events_profiling_log.push_back(std::make_pair(ev, str));
}

std::pair<void*, MemoryDeleter> OpenCLContext::New(size_t nbytes) {
  cl_int err = 0;

  cl::Buffer* buffer = new cl::Buffer(OpenCLContext::context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      nbytes, nullptr, &err);
  OPENCL_CHECK(err);
  // TODO(bwasti): use host ptr if possible to make CopyBytes free
  return std::make_pair((void *)buffer, OpenCLContext::Delete);
}

template <>
void OpenCLContext::CopyBytes<OpenCLContext, CPUContext>(size_t nbytes, const void *src, void *dst) {
  //TODO: p.stawicki@samsung.com: This is blocking and should be removed in the future
  queue().enqueueReadBuffer(*((cl::Buffer*)src), CL_TRUE, 0, nbytes, static_cast<char*>(dst));
}

template <>
void OpenCLContext::CopyBytes<CPUContext, OpenCLContext>(size_t nbytes, const void *src, void *dst) {
  queue().enqueueWriteBuffer(*((cl::Buffer*)(dst)), CL_FALSE, 0, nbytes, static_cast<const char*>(src));
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

template <>
void OpenCLContext::enqueueCopyBytes<OpenCLContext, CPUContext>(size_t nbytes, const void *src, void *dst) {
  queue().enqueueReadBuffer(*((cl::Buffer*)src), CL_FALSE, 0, nbytes, static_cast<char*>(dst));
}

template <>
void OpenCLContext::enqueueCopyBytes<CPUContext, OpenCLContext>(size_t nbytes, const void *src, void *dst) {
  queue().enqueueWriteBuffer(*((cl::Buffer*)(dst)), CL_FALSE, 0, nbytes, static_cast<const char*>(src));
}

/*

template <>
inline void CopyItems<OpenCLContext, CPUContext>(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    CAFFE_ENFORCE(!meta.copy(), "OpenCLContext requires fundamental types.");
    CopyBytes<OpenCLContext, CPUContext>(n * meta.itemsize(), src, dst);
}

template <>
inline void CopyItems<CPUContext, OpenCLContext>(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    CAFFE_ENFORCE(!meta.copy(), "OpenCLContext requires fundamental types.");
    CopyBytes<CPUContext, OpenCLContext>(n * meta.itemsize(), src, dst);
}

template <>
inline void CopyItems<CPUContext, CPUContext>(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    CAFFE_ENFORCE(!meta.copy(), "OpenCLContext requires fundamental types.");
    CopyBytes<CPUContext, CPUContext>(n * meta.itemsize(), src, dst);
}

template <>
inline void CopyItems<OpenCLContext, OpenCLContext>(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    CAFFE_ENFORCE(!meta.copy(), "OpenCLContext requires fundamental types.");
    CopyBytes<OpenCLContext, OpenCLContext>(n * meta.itemsize(), src, dst);
}
*/
void OpenCLContext::LogProfilingInfo(const cl::Event& ev, const std::string& str)
{
  openCL_objects_.LogProfilingInfo(ev, str);
}

void OpenCLContext::Delete(void *ptr) {
  delete (cl::Buffer *)ptr;
}

cl::Kernel OpenCLContext::BuildKernel(const char* src, std::string additional_options, const char* fn_name) {
  cl::Program::Sources source(1,
      std::make_pair(src, strlen(src)));

  cl_int err = CL_SUCCESS;
  cl::Program p = cl::Program(clContext(), source, &err);
  OPENCL_CHECK(err);

  std::string options = "-cl-std=CL1.1 -cl-fast-relaxed-math -cl-single-precision-constant ";
  options += additional_options;

  // TODO support more than one device
  // this will involve checking a compiler exists on each device
  vector<cl::Device> devices{OpenCLContext::computing_device};
  err = p.build(devices, options.c_str());
  cl_build_status build_status = p.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(OpenCLContext::computing_device);
  if (err != CL_SUCCESS || build_status != CL_BUILD_SUCCESS) {
    auto str = p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(OpenCLContext::computing_device);
    LOG(ERROR) << "Error code: " << err << " Build status: " << build_status;
    CAFFE_THROW(str);
  }

  auto kernel = cl::Kernel(p, fn_name, &err);
  OPENCL_CHECK(err);
  return kernel;
}

cl::Kernel OpenCLContext::BuildKernelCached(const char *src, std::string compile_options, const char *function_name) {
  std::stringstream cacheId;
  cacheId << src << "|" << compile_options << "|" << function_name;
  return BuildKernelCachedId(cacheId.str(), src, compile_options, function_name);
}

cl::Kernel OpenCLContext::BuildKernelCachedId(const std::string &cacheId, const char *src,
                                              std::string compile_options, const char *function_name) {
  thread_local OpenCLKernelCache cache;
  auto kernel = cache.GetKernel(cacheId);
  if (kernel() == nullptr) {
    kernel = BuildKernel(src, compile_options, function_name);
    cache.SetKernel(cacheId, kernel);
  }
  return kernel;
}

std::string OpenCLContext::BuildArgumentList(std::vector<std::pair<std::string, std::string>> args) {
  std::string out = " "; // There may be args before this
  for (auto arg : args) {
    out += "-D " + arg.first + "=" + arg.second + " ";
  }
  return out;
}

struct DefaultOpenCLAllocator final : public at::Allocator {
  DefaultOpenCLAllocator() {}
  ~DefaultOpenCLAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    cl_int err = 0;
    cl::Buffer* buffer = new cl::Buffer(OpenCLContext::context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, nbytes, nullptr, &err);
    OPENCL_CHECK(err);
    return {buffer, buffer, &Delete, at::Device(OPENCL)};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &Delete;
  }

 private:
  static void Delete(void* ptr) {
    OpenCLContext::Delete(ptr);
  }
};

static DefaultOpenCLAllocator g_opencl_alloc;
REGISTER_ALLOCATOR(OPENCL, &g_opencl_alloc);

} // namespace caffe2
