#ifndef CAFFE2_OPENCL_CONTEXT_H_
#define CAFFE2_OPENCL_CONTEXT_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

#define CL_HPP_ENABLE_EXCEPTIONS 1
#define CL_HPP_CL_1_2_DEFAULT_BUILD 1
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include "libopencl.h"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif


#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "caffe2_opencl_test-jni"
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#else
#define ALOGI(...) printf(__VA_ARGS__)
#endif

#include <mutex>

#define OPENCL_CHECK(v) do {\
    cl_int _err = (v); \
    if (_err != CL_SUCCESS) { \
      CAFFE_THROW("OpenCL Error:", _err, " on line ", __LINE__);\
    }\
  } while(0)

namespace caffe2 {

struct OpenCLContextSingleton {
 private:
  OpenCLContextSingleton();
  OpenCLContextSingleton(const OpenCLContextSingleton &) = delete;
  OpenCLContextSingleton(OpenCLContextSingleton&&) = delete;
  bool profiling_info_enabled_;
 public:
  void LogProfilingInfo(const cl::Event& ev, const std::string& str);
  void PrintProfilingLogs();
  static OpenCLContextSingleton& getInstance();
  cl::Platform platform;
  cl::Device device;
  std::vector<cl::Device> devices;
  cl::Context context;
  cl::CommandQueue queue;
  std::vector<cl::CommandQueue> available_queues;
  std::unordered_map<std::string, cl::Kernel> kernel_cache_;
  std::vector<std::pair<cl::Event, std::string>> events_profiling_log;
};

class OpenCLContext final {
 public:
  explicit OpenCLContext() {}
  explicit OpenCLContext(const DeviceOption& option) {
    // TODO: Investigate why this assert was introduced
    //
    // It is not clear why this assert was introduced
    // it fails during initialization in opencl conv operator
    // test: conv_op_test.cc:193
    // It seams that copy operator can contain
    // OpenCLContext but transfer data from/to CPU context.
    //
    // DCHECK_EQ(option.device_type(), OPENCL);
    OpenCLContext();
  }
  ~OpenCLContext() {}

  /*
   * Everything below is basically boiler plate for Context classes
   */
  static std::pair<void*, MemoryDeleter> New(size_t nbytes);

  static void Delete(void* data);

  void LogProfilingInfo(const cl::Event& ev, const std::string& str);

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(int n, const T* src, T* dst) {
    CopyBytes<SrcContext, DstContext>(
        n * sizeof(T), static_cast<const void*>(src), static_cast<void*>(dst));
  }

  inline void CopyCL(const cl::Buffer* src, cl::Buffer* dst, size_t size_bytes) {
    auto& ctx = GetSingleton();
    cl::Event event;
    OPENCL_CHECK(
        ctx.queue.enqueueCopyBuffer(*src, *dst, 0, 0, size_bytes,
        NULL,
        &event)
    );
  }

  template <class SrcContext, class DstContext>
  inline void
  CopyItems(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    CAFFE_ENFORCE(!meta.copy(), "OpenCLContext requires fundamental types.");
    CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
  }

  void SwitchToDevice(int a, ...) {
  }

  void SwitchToDevice2(int a, ...) {
    auto& ctx = GetSingleton();
    //CAFFE_ENFORCE(a < ctx.devices.size());
    //ctx.device = ctx.devices[a];
    //std::cout << "radek SwitchToDevice:" << a << "\n";
    CAFFE_ENFORCE(a < ctx.available_queues.size());
    ctx.queue = ctx.available_queues[a];
  }

  void SwitchToDevice() {
    SwitchToDevice(0);
  }

  inline void WaitEvent(const Event& ev) {
    ev.Wait(OPENCL, this);
  }

  inline void Record(Event* ev, const char* err_msg = nullptr) const {
    CAFFE_ENFORCE(ev, "Event must not be null.");
    ev->Record(OPENCL, this, err_msg);
  }

  bool FinishDeviceComputation() {
    return true;
  }

  bool FinishDeviceComputation2() {
    auto& ctx = GetSingleton();
    ctx.queue.finish();
    ctx.PrintProfilingLogs();
    return true;
  }

  cl::Kernel BuildKernel(const char* src, std::string additional_options = "", const char* fn_name = "K");

  cl::Kernel BuildKernelCachedId(const std::string &cacheId, const char *src, std::string compile_options = "", const char *kernel_function_name = nullptr);
  cl::Kernel BuildKernelCached(const char *src, std::string compile_options = "", const char *kernel_function_name = nullptr);

  template <class SrcContext, class DstContext>
  void CopyBytes(size_t nbytes, const void *src, void *dst);

  template <class SrcContext, class DstContext>
  void enqueueCopyBytes(size_t nbytes, const void *src, void *dst);

  // Disabled for PyTorch
  // It causes ambiguous concretization of give tensor fill operator.
  // long int was cast both to size_t and int so
  // it was not possible to select one of Copy overloaded versions.
  //
  //
  // For compatibility with old style copy
  //template <typename T, class SrcContext, class DstContext>
  //inline void Copy(int n, const T* src, T* dst) {
  //  if (std::is_fundamental<T>::value) {
  //    CopyBytes<SrcContext, DstContext>(n * sizeof(T),
  //                                   static_cast<const void*>(src),
  //                                   static_cast<void*>(dst));
  //  } else {
  //    for (int i = 0; i < n; ++i) {
  //      dst[i] = src[i];
  //    }
  //  }
  //}

  template <typename T_in, typename T_out, class SrcContext, class DstContext>
  inline void Copy(const Tensor<SrcContext>& src, Tensor<DstContext>& dst) {
    dst.Resize(src.dims());
    size_t n = src.size();
    if (std::is_same<T_in, T_out>::value) {
      if (std::is_fundamental<T_in>::value) {
        CopyBytes<SrcContext, DstContext>(n * sizeof(T_in),
                                       static_cast<const void*>(src.template data<T_in>()),
                                       static_cast<void*>(dst.template mutable_data<T_out>()));
      } else {
        for (int i = 0; i < n; ++i) {
          dst.template mutable_data<T_out>()[i] = src.template data<T_in>()[i];
        }
      }
    } else {
      CAFFE_THROW("This Copy requires specialization.");
    }
  }

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(const Tensor<SrcContext>& src, Tensor<DstContext>& dst) {
    Copy<T, T>(src, dst);
  }

  // By default CUDA operators have async device parts
  static bool HasAsyncPartDefault() {
    return true;
  }

  static bool SupportsAsyncScheduling() {
    return true;
  }

  static bool IsStreamFree(const DeviceOption& option, int stream_id) {
    // FIXME: Not implemented
    //       Is it necessary for OPENCL?
    CAFFE_THROW("IsStreamFree not implemented for OpenCL");
    // This is implementation from CUDA
    // auto stream = CUDAContext::cuda_stream(option.cuda_gpu_id(), stream_id);
    // return cudaStreamQuery(stream) == cudaSuccess;
  }
std::string BuildArgumentList(std::vector<std::pair<std::string, std::string>> args);

  static struct OpenCLContextSingleton& GetSingleton();
};

typedef Tensor<OpenCLContext> TensorCL;

} // namespace caffe2

#endif // ifndef CAFFE2_OPENCL_CONTEXT_H_

