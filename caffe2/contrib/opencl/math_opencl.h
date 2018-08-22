
#include "caffe2/utils/math.h"
#include "caffe2/core/context.h"
#include "context.h"

//Enable to obtain more logs
#define DEBUGGING false

namespace caffe2 {
namespace math {

#define DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(Funcname)                \
  template <typename T>                                                       \
  void Funcname(                                                              \
      const int N, const T* a, const T* b, T* y,   \
      OpenCLContext* context);

DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(Add);
DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(Sub);
DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(Mul);
DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(Div);
#undef DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION

#define DELEGATE_SIMPLE_UNARY_FUNCTION(Funcname)                              \
  template <typename T>                                                       \
  void Funcname(                                                              \
      const int N, const cl::Buffer* a, cl::Buffer* y, OpenCLContext* context);

DELEGATE_SIMPLE_UNARY_FUNCTION(Log)
DELEGATE_SIMPLE_UNARY_FUNCTION(Cos)
DELEGATE_SIMPLE_UNARY_FUNCTION(Sin)
DELEGATE_SIMPLE_UNARY_FUNCTION(Sqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(Rsqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(Not)

#undef DELEGATE_SIMPLE_UNARY_FUNCTION

template <typename T>
void Gemm(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const T alpha,
    const cl::Buffer* A,
    const cl::Buffer* B,
    const T beta,
    cl::Buffer* C,
    OpenCLContext* context);

template <typename T>
void GemmBatched(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const T alpha,
    const cl::Buffer* A,
    const cl::Buffer* B,
    const T beta,
    cl::Buffer* C,
    OpenCLContext* context);

} // namespace math
} // namespace caffe2

