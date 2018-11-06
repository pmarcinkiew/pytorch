#include <cfloat>
#include <sys/time.h>

#include "caffe2/utils/math.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "context.h"
#include "math_opencl.h"

//Enable to obtain more logs
#define DEBUGGING false

namespace caffe2 {
namespace math {

// TODO header guard
static constexpr const char* kTwoOperator = R"CLC(
kernel void kMathTwoOperator(
  const global %s* a,
  const global %s* b,
  global %s* y
){
  int index = get_global_id(0);
  y[index] = a[index] %s b[index];
}
)CLC";

static void runPureGPUKernel(const int N, const cl::Buffer* a,
                             const cl::Buffer* b, cl::Buffer* y,
                             OpenCLContext* context, const char* kernelSrc){
  auto kernel = context->BuildKernelCached(kernelSrc, "", "kMathTwoOperator");
  OPENCL_CHECK(kernel.setArg(0, *a));
  OPENCL_CHECK(kernel.setArg(1, *b));
  OPENCL_CHECK(kernel.setArg(2, *y));
  cl::Event event;
  context->enqueue(
    kernel,
    cl::NullRange,
    cl::NDRange(N),
    cl::NullRange,
    NULL,
    &event);
}

#if DEBUGGING
  #define DEBUGPRINTF(format, ...)                                            \
    do {                                                                      \
        printf("[%s:%s]:" format, ,__FILE__, __LINE__,  __VA_ARGS__);         \
    } while (false);
#else
  #define DEBUGPRINTF(format, ...) do {} while (false);
#endif

#define MAXKERNELSRCSIZE 1024

#define DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(T, Funcname, expr)       \
  template <>                                                                 \
  void Funcname<T>(                                                           \
      const int N, const T* a, const T* b, T* y,   \
      OpenCLContext* context) {                                               \
        char kernelsrc[MAXKERNELSRCSIZE];                                     \
        snprintf(kernelsrc, sizeof(kernelsrc), kTwoOperator,                  \
                 #T, #T, #T, #expr);                                          \
        DEBUGPRINTF("KERNEL %s\n", kernelsrc);                                \
        runPureGPUKernel(N, reinterpret_cast<const cl::Buffer*>(a),           \
                        reinterpret_cast<const cl::Buffer*>(b),               \
                        reinterpret_cast<cl::Buffer*>(y),                     \
                        context, kernelsrc);                                  \
  }

DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(float, Add, +);
DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(int, Add, +);
DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(float, Sub, -);
DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(float, Mul, *);
DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(float, Div, /);
#undef DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION

#define DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(T, Funcname)             \
  template <>                                                                 \
  void Funcname<T, OpenCLContext>(                                            \
      const int N, const T* a, const T* b, T* y, OpenCLContext* context) {    \
      Funcname(N, a, b, y, context);                                          \
  }

DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(float, Add);
DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(int, Add);
DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(float, Sub);
DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(float, Mul);
DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION(float, Div);

#undef DELEGATE_SIMPLE_OPENCL_BINARY_INFIX_FUNCTION


static constexpr const char* kOneOperator = R"CLC(
bool Not(bool x) {
    return !x;
}
kernel void kMathOneOperator(
  global %s* a,
  global %s* y,
  int N
){
  int index = get_global_id(0);
  if (index < N) {
    y[index] = %s(a[index]);
  }
}
)CLC";

static void runOnePureGPUKernel(const int N, const cl::Buffer* a, cl::Buffer* y,
    OpenCLContext* context, const char* kernelSrc){
  //kernel is created dynamically
  //FIXME: It should be cached. If cache is used here,
  //       the kernel fails to produce correct computations results
  auto kernel = context->BuildKernelCached(kernelSrc, "", "kMathOneOperator");

  OPENCL_CHECK(kernel.setArg(0, *a));
  OPENCL_CHECK(kernel.setArg(1, *y));
  OPENCL_CHECK(kernel.setArg(2, N));
  cl::Event event;
  context->enqueue(
    kernel,
    cl::NullRange,
    cl::NDRange(N),
    cl::NullRange,
    NULL,
    &event);
}

#define DELEGATE_SIMPLE_UNARY_FUNCTION(T, Funcname, expr)                     \
  template <>                                                                 \
  void Funcname<T>(                                                           \
      const int N, const cl::Buffer* a, cl::Buffer* y,                        \
      OpenCLContext* context) {                                               \
        char kernelsrc[MAXKERNELSRCSIZE];                                     \
        snprintf(kernelsrc, sizeof(kernelsrc), kOneOperator,                  \
                 #T, #T, #expr);                                              \
        DEBUGPRINTF("KERNEL %s\n", kernelsrc);                                \
        runOnePureGPUKernel(N, a, y, context, kernelsrc);                     \
  }

DELEGATE_SIMPLE_UNARY_FUNCTION(float, Log, log)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Cos, cos)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sin, sin)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqrt, sqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, InvSqrt, rsqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(bool, Not, Not)

#undef DELEGATE_SIMPLE_UNARY_FUNCTION

static constexpr const char* kFillWithNaN = R"CLC(
kernel void kMathFillWithNaN(
  __global float* out_mat,
  const int Dim
  )
{
  int col_ix = get_global_id(0);
  if (col_ix >= Dim)
  {
    return;
  }
  out_mat[col_ix] = NAN;
}
)CLC";


static void runFillWithNaN(
    cl::Buffer &buff,
    const int dim,
    OpenCLContext* context
    )
{
  auto kernel = context->BuildKernelCached(kFillWithNaN, "", "kMathFillWithNaN");
  OPENCL_CHECK(kernel.setArg(0, buff));
  OPENCL_CHECK(kernel.setArg(1, dim));

  cl::Event event;

  context->enqueue(
        kernel,
        cl::NullRange,
        cl::NDRange(dim), //The dimension of work is the number of matrix rows
        cl::NullRange,
        NULL,
        &event);
}

static constexpr const char* kRowwiseMax = R"CLC(
kernel void kMathRowwiseMax(
  const __global float* in_mat,
  __global float* out_mat,
  const int N_rows,
  const int D_cols)
{
  int n_ix = get_global_id(0);

  if (n_ix >= N_rows)
  {
    return;
  }

  for (int ix = n_ix * D_cols; ix < (n_ix + 1) * D_cols; ix++)
  {
    if (isnan(out_mat[n_ix]) || in_mat[ix] > out_mat[n_ix])
      out_mat[n_ix] = in_mat[ix];
  }
}
)CLC";

static void runRowwiseMaxKernel(
  const int N_rows,
  const int D_cols,
  const void* x_in,
  void* y_out,
  OpenCLContext* context,
  size_t type_size)
{
  //At first initialize output items with special value (output will be used in computation),
  //to mark the maxima were not computed yet:
  auto kernel1 = context->BuildKernelCached(kFillWithNaN, "", "kMathFillWithNaN");

  size_t out_mem_size = N_rows * type_size;
  auto mat_y_out = context->New(out_mem_size);

  OPENCL_CHECK(kernel1.setArg(0, *(cl::Buffer*)mat_y_out.first));
  OPENCL_CHECK(kernel1.setArg(1, N_rows));

  cl::Event event;
  context->enqueue(
        kernel1,
        cl::NullRange,
        cl::NDRange(N_rows), //The dimension of work is the number of matrix rows
        cl::NullRange,
        NULL,
        &event);

  //Now, as all output values were initialized, we can begin to search for Max per column:
  auto kernel2 = context->BuildKernelCached(kRowwiseMax, "", "kMathRowwiseMax");
  size_t in_mem_size = N_rows * D_cols * type_size;
  auto mat_x_in = context->New(in_mem_size);

  context->CopyBytes<CPUContext, OpenCLContext>(in_mem_size, x_in, mat_x_in.first);

  OPENCL_CHECK(kernel2.setArg(0, *(cl::Buffer*)mat_x_in.first));
  OPENCL_CHECK(kernel2.setArg(1, *(cl::Buffer*)mat_y_out.first));
  OPENCL_CHECK(kernel2.setArg(2, N_rows));
  OPENCL_CHECK(kernel2.setArg(3, D_cols));

  context->enqueue(
        kernel2,
        cl::NullRange,
        cl::NDRange(N_rows), //The dimension of work is the number of matrix rows
        cl::NullRange,
        NULL,
        &event);
  event.wait();

  context->CopyBytes<OpenCLContext, CPUContext>(out_mem_size, mat_y_out.first, y_out);
  context->Delete(mat_x_in.first);
  context->Delete(mat_y_out.first);
}

#define CAFFE2_SPECIALIZED_ROWWISEMAX(T)            \
template <>                                         \
void RowwiseMax<T, OpenCLContext>(                  \
  const int N,                                      \
  const int D,                                      \
  const T* x,                                       \
  T* y,                                             \
  OpenCLContext* ctx)                               \
{                                                   \
  runRowwiseMaxKernel(N, D, x, y, ctx, sizeof(T));  \
}
CAFFE2_SPECIALIZED_ROWWISEMAX(float)
#undef CAFFE2_SPECIALIZED_ROWWISEMAX

typedef std::function<void(const int N, const cl::Buffer* a, cl::Buffer* y,
                           OpenCLContext* context)> UnaryInfixOperation;

static void runOneKernel(const int N, const void* a, void* y,
    OpenCLContext* context, size_t size, UnaryInfixOperation operation){
  size_t memSize = N * size;
  auto ab = context->New(memSize);
  auto yb = context->New(memSize);
  context->CopyBytes<CPUContext, OpenCLContext>(memSize, a, ab.first);
  operation(N, (cl::Buffer*)ab.first, (cl::Buffer*)yb.first, context);
  context->CopyBytes<OpenCLContext, CPUContext>(memSize, yb.first, y);
  context->Delete(ab.first);
  context->Delete(yb.first);
}

// This is wrong - OpenCL math should used just OpenCL data structures
// no buffers in RAM
// It doesn't work for anything other than int yet.
#define DELEGATE_SIMPLE_UNARY_FUNCTION(T, Funcname)                           \
  template <>                                                                 \
  void Funcname<T, OpenCLContext>(                                            \
      const int N, const T* a, T* y, OpenCLContext* context) {                \
        runOneKernel(N, (void*)a, (void*)y, context,                          \
                  sizeof(T), caffe2::math::Funcname<T>);                      \
  }

DELEGATE_SIMPLE_UNARY_FUNCTION(float, Log)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Cos)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sin)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqrt)
//DELEGATE_SIMPLE_UNARY_FUNCTION(float, InvSqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(bool, Not)

#undef DELEGATE_SIMPLE_UNARY_FUNCTION

//https://cnugteren.github.io/tutorial/pages/page3.html

static constexpr const char* kMatrixMulOperatorANBNCN = R"CLC(
  __kernel void kMathMatrixMulOperatorANBNCN(
                  const int M, const int N, const int K,
                  const float alpha, const float beta,
                  const __global float* A,
                  const __global float* B,
                  __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    if (globalRow < M && globalCol < N) {

      // Compute a single element (loop over K)
      float acc = 0.0f;
      for (int k = 0; k < K; k++) {
        acc += A[globalRow * K + k] * B[k * N + globalCol] * alpha;
      }

      // Store the result
      C[globalRow * N + globalCol] = acc + C[globalRow * N + globalCol] * beta;
    }
}
)CLC";

static constexpr const char* kMatrixMulOperatorATBNCN = R"CLC(
  __kernel void kMathMatrixMulOperatorATBNCN(
                  const int M, const int N, const int K,
                  const float alpha, const float beta,
                  const __global float* A,
                  const __global float* B,
                  __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    if (globalRow < M && globalCol < N) {

      // Compute a single element (loop over K)
      float acc = 0.0f;
      for (int k = 0; k < K; k++) {
        acc += A[k * M + globalRow] * B[k * N + globalCol] * alpha;
      }

      // Store the result
      C[globalRow * N + globalCol] = acc + C[globalRow * N + globalCol] * beta;
    }
}
)CLC";

static constexpr const char* kMatrixMulOperatorANBTCN = R"CLC(
  __kernel void kMathMatrixMulOperatorANBTCN(
                  const int M, const int N, const int K,
                  const float alpha, const float beta,
                  const __global float* A,
                  const __global float* B,
                  __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    if (globalRow < M && globalCol < N) {

      // Compute a single element (loop over K)
      float acc = 0.0f;
      for (int k = 0; k < K; k++) {
        acc += A[globalRow * K + k] * B[globalCol * K + k] * alpha;
      }

      // Store the result
      C[globalRow * N + globalCol] = acc + C[globalRow * N + globalCol] * beta;
    }
}
)CLC";

static constexpr const char* kMatrixMulOperatorATBTCN = R"CLC(
  __kernel void kMathMatrixMulOperatorATBTCN(
                  const int M, const int N, const int K,
                  const float alpha, const float beta,
                  const __global float* A,
                  const __global float* B,
                  __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    if (globalRow < M && globalCol < N) {

      // Compute a single element (loop over K)
      float acc = 0.0f;
      for (int k = 0; k < K; k++) {
        acc += A[k * M + globalRow] * B[globalCol * K + k] * alpha;
      }

      // Store the result
      C[globalRow * N + globalCol] = acc + C[globalRow * N + globalCol] * beta;
    }
}
)CLC";

//Let's keep it here for reference
//static constexpr const char* kMatrixMulAccesANorm = "A[globalRow * K + k]";
//static constexpr const char* kMatrixMulAccesATrans = "A[k * M + globalRow]";
//static constexpr const char* kMatrixMulAccesBNorm = "B[k * N + globalCol]";
//static constexpr const char* kMatrixMulAccesBTrans = "B[globalCol * K + k]";
//static constexpr const char* kMatrixMulAccesCNorm = "C[globalRow * N + globalCol]";
//static constexpr const char* kMatrixMulAccesCTrans = "C[globalCol * M + globalRow]";

template <>
void Gemm<float>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const cl::Buffer* A,
    const cl::Buffer* B,
    const float beta,
    cl::Buffer* C,
    OpenCLContext* context)
{
  const char* kernel_src;
  const char* kernel_name;
  if (TransA == CblasNoTrans) {
    if (TransB == CblasNoTrans) {
      kernel_src = kMatrixMulOperatorANBNCN;
      kernel_name = "kMathMatrixMulOperatorANBNCN";
    } else {
      kernel_src = kMatrixMulOperatorANBTCN;
      kernel_name = "kMathMatrixMulOperatorANBTCN";
    }
  } else {
    if (TransB == CblasNoTrans) {
      kernel_src = kMatrixMulOperatorATBNCN;
      kernel_name = "kMathMatrixMulOperatorATBNCN";
    } else {
      kernel_src = kMatrixMulOperatorATBTCN;
      kernel_name = "kMathMatrixMulOperatorATBTCN";
    }
  }

  auto kernel = context->BuildKernelCached(kernel_src, "", kernel_name);
  OPENCL_CHECK(kernel.setArg(0, M));
  OPENCL_CHECK(kernel.setArg(1, N));
  OPENCL_CHECK(kernel.setArg(2, K));
  OPENCL_CHECK(kernel.setArg(3, alpha));
  OPENCL_CHECK(kernel.setArg(4, beta));
  OPENCL_CHECK(kernel.setArg(5, *A));
  OPENCL_CHECK(kernel.setArg(6, *B));
  OPENCL_CHECK(kernel.setArg(7, *C));

  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long start = tv.tv_sec * 1000000 + tv.tv_usec;

  cl::Event event;
  context->enqueue(
    kernel,
    cl::NullRange,
    cl::NDRange(M, N),
    cl::NullRange,
    NULL,
    &event);

  gettimeofday(&tv, NULL);
  long long end = tv.tv_sec * 1000000 + tv.tv_usec;

  std::stringstream outstr;
  outstr << "math Gemm " << end << " cpu time delta: " << end - start;
  outstr << " M: " << M << " N: " << N << " K: " << K;
  context->LogProfilingInfo(event, outstr.str());
}

template <>
void Gemm<float, OpenCLContext, DefaultEngine>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    OpenCLContext* context,
    TensorProto::DataType math_type)
{
  Gemm<float>(TransA,
              TransB,
              M,
              N,
              K,
              alpha,
              (cl::Buffer*)A,
              (cl::Buffer*)B,
              beta,
              (cl::Buffer*)C,
              context);
}

static constexpr const char* kMatrixMulOperatorBatchedANBNCN = R"CLC(
  __kernel void kMathMatrixMulOperatorBatchedANBNCN(const int batch_size,
                  const int M, const int N, const int K,
                  const float alpha, const float beta,
                  const __global float* A,
                  const __global float* B,
                  __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    if (globalRow < M && globalCol < N) {
      int batch_A, batch_B, batch_C;

      for(int i = 0; i < batch_size; i++){
        batch_A = M * K * i;
        batch_B = K * N * i;
        batch_C = M * N * i;

        // Compute a single element (loop over K)
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
          acc += A[globalRow * K + k + batch_A] * B[k * N + globalCol + batch_B] * alpha;
        }

        // Store the result
        C[globalRow * N + globalCol + batch_C] = acc + C[globalRow * N + globalCol + batch_C] * beta;
      }
    }
}
)CLC";

static constexpr const char* kMatrixMulOperatorBatchedATBNCN = R"CLC(
  __kernel void kMathMatrixMulOperatorBatchedATBNCN(
                  const int batch_size,
                  const int M, const int N, const int K,
                  const float alpha, const float beta,
                  const __global float* A,
                  const __global float* B,
                  __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    if (globalRow < M && globalCol < N) {
      int batch_A, batch_B, batch_C;
      for(int i = 0; i < batch_size; i++){
        batch_A = M * K * i;
        batch_B = K * N * i;
        batch_C = M * N * i;

        // Compute a single element (loop over K)
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
          acc += A[k * M + globalRow + batch_A] * B[k * N + globalCol + batch_B] * alpha;
        }

        // Store the result
        C[globalRow * N + globalCol + batch_C] = acc + C[globalRow * N + globalCol + batch_C] * beta;
      }
    }
}
)CLC";

static constexpr const char* kMatrixMulOperatorBatchedANBTCN = R"CLC(
  __kernel void kMathMatrixMulOperatorBatchedANBTCN(
                  const int batch_size,
                  const int M, const int N, const int K,
                  const float alpha, const float beta,
                  const __global float* A,
                  const __global float* B,
                  __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    if (globalRow < M && globalCol < N) {
      int batch_A, batch_B, batch_C;
      for(int i = 0; i < batch_size; i++){
        batch_A = M * K * i;
        batch_B = K * N * i;
        batch_C = M * N * i;

        // Compute a single element (loop over K)
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
          acc += A[globalRow * K + k + batch_A] * B[globalCol * K + k + batch_B] * alpha;
        }

        // Store the result
        C[globalRow * N + globalCol + batch_C] = acc + C[globalRow * N + globalCol + batch_C] * beta;
      }
    }
}
)CLC";

static constexpr const char* kMatrixMulOperatorBatchedATBTCN = R"CLC(
  __kernel void kMathMatrixMulOperatorBatchedATBTCN(
                  const int batch_size,
                  const int M, const int N, const int K,
                  const float alpha, const float beta,
                  const __global float* A,
                  const __global float* B,
                  N__global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    if (globalRow < M && globalCol < N) {
      int batch_A, batch_B, batch_C;
      for(int i = 0; i < batch_size; i++){
        batch_A = M * K * i;
        batch_B = K * N * i;
        batch_C = M * N * i;

        // Compute a single element (loop over K)
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
          acc += A[k * M + globalRow + batch_A] * B[globalCol * K + k + batch_B] * alpha;
        }

        // Store the result
        C[globalRow * N + globalCol + batch_C] = acc + C[globalRow * N + globalCol + batch_C] * beta;
      }
    }
}
)CLC";

//Let's keep it here for reference
//static constexpr const char* kMatrixMulAccesANorm = "A[globalRow * K + k]";
//static constexpr const char* kMatrixMulAccesATrans = "A[k * M + globalRow]";
//static constexpr const char* kMatrixMulAccesBNorm = "B[k * N + globalCol]";
//static constexpr const char* kMatrixMulAccesBTrans = "B[globalCol * K + k]";
//static constexpr const char* kMatrixMulAccesCNorm = "C[globalRow * N + globalCol]";
//static constexpr const char* kMatrixMulAccesCTrans = "C[globalCol * M + globalRow]";

template <>
void GemmBatched<float>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const cl::Buffer* A,
    const cl::Buffer* B,
    const float beta,
    cl::Buffer* C,
    OpenCLContext* context)
{
  const char* kernel_src;
  const char* kernel_name;
  if (TransA == CblasNoTrans) {
    if (TransB == CblasNoTrans) {
      kernel_src = kMatrixMulOperatorBatchedANBNCN;
      kernel_name = "kMathMatrixMulOperatorBatchedANBNCN";
    } else {
      kernel_src = kMatrixMulOperatorBatchedANBTCN;
      kernel_name = "kMathMatrixMulOperatorBatchedANBTCN";
    }
  } else {
    if (TransB == CblasNoTrans) {
      kernel_src = kMatrixMulOperatorBatchedATBNCN;
      kernel_name = "kMathMatrixMulOperatorBatchedATBNCN";
    } else {
      kernel_src = kMatrixMulOperatorBatchedATBTCN;
      kernel_name = "kMathMatrixMulOperatorBatchedATBTCN";
    }
  }

  auto kernel = context->BuildKernelCached(kernel_src, "", kernel_name);
  OPENCL_CHECK(kernel.setArg(0, batch_size));
  OPENCL_CHECK(kernel.setArg(1, M));
  OPENCL_CHECK(kernel.setArg(2, N));
  OPENCL_CHECK(kernel.setArg(3, K));
  OPENCL_CHECK(kernel.setArg(4, alpha));
  OPENCL_CHECK(kernel.setArg(5, beta));
  OPENCL_CHECK(kernel.setArg(6, *A));
  OPENCL_CHECK(kernel.setArg(7, *B));
  OPENCL_CHECK(kernel.setArg(8, *C));

  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long start = tv.tv_sec * 1000000 + tv.tv_usec;

  cl::Event event;
  context->enqueue(
    kernel,
    cl::NullRange,
    cl::NDRange(M, N),
    cl::NullRange,
    NULL,
    &event);

  gettimeofday(&tv, NULL);
  long long end = tv.tv_sec * 1000000 + tv.tv_usec;

  std::stringstream outstr;
  outstr << "math GemmBatched ";
  outstr <<  " M: " << M << " N: " << N << " K: " << K;
  context->LogProfilingInfo(event, outstr.str());
}
/*template <typename T, class Context, class Engine = DefaultEngine>
CAFFE2_API void GemmBatched(
    const float alpha,
    const T** A,
    const T** B,
    const float beta,
    T** C,
    Context* context,
    TensorProto::DataType math_type = TensorProto_DataType_FLOAT);
  */  
template <>
void GemmBatched<float, OpenCLContext, DefaultEngine>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float** A,
    const float** B,
    const float beta,
    float** C,
    OpenCLContext* context,
    TensorProto::DataType math_type)
{
  GemmBatched<float>(TransA,
              TransB,
              batch_size,
              M,
              N,
              K,
              alpha,
              (cl::Buffer*)*A,
              (cl::Buffer*)*B,
              beta,
              (cl::Buffer*)*C,
              context);
}

static constexpr const char* kSetKernel = R"CLC(
kernel void kMathSetKernel(
  __global float* out_mat,
  const int Dim,
  const float value
  )
{
  int col_ix = get_global_id(0);
  if (col_ix < Dim)
  {
    out_mat[col_ix] = value;
  }
}
)CLC";

template<>
void Set<float, caffe2::OpenCLContext>(unsigned long size, float value,
                                       float* X,
                                       caffe2::OpenCLContext* context) {
  auto kernel1 = context->BuildKernelCached(kSetKernel, "", "kMathSetKernel");
  cl::Buffer* xBuf = (cl::Buffer*)X;
  OPENCL_CHECK(kernel1.setArg(0, *(cl::Buffer*)xBuf));
  OPENCL_CHECK(kernel1.setArg(1, (int)size));
  OPENCL_CHECK(kernel1.setArg(2, value));
  cl::Event event;

  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long start = tv.tv_sec * 1000000 + tv.tv_usec;

  context->enqueue(
        kernel1,
        cl::NullRange,
        cl::NDRange(size), //The dimension of work is the number of matrix rows
        cl::NullRange,
        NULL,
        &event);

  gettimeofday(&tv, NULL);
  long long end = tv.tv_sec * 1000000 + tv.tv_usec;

  std::stringstream outstr;
  outstr << "math Set ";
  outstr << " size: " << size;
  context->LogProfilingInfo(event, outstr.str());
}

template<>
void Set<short, caffe2::OpenCLContext>(unsigned long size, short value,
                                       short* X,
                                       caffe2::OpenCLContext* context) {
  CAFFE_THROW("Not implemented.");
}

template<>
void Set<unsigned short, caffe2::OpenCLContext>(unsigned long size, unsigned short value,
                                                unsigned short* X,
                                                caffe2::OpenCLContext* context) {
  CAFFE_THROW("Not implemented.");
}

template<>
void Set<bool, caffe2::OpenCLContext>(unsigned long size, bool value,
                                      bool* X,
                                      caffe2::OpenCLContext* context) {
  CAFFE_THROW("Not implemented.");
}

template<>
void Set<signed char, caffe2::OpenCLContext>(unsigned long size, signed char value,
                                             signed char* X,
                                             caffe2::OpenCLContext* context) {
  CAFFE_THROW("Not implemented.");
}

template<>
void Set<unsigned char, caffe2::OpenCLContext>(unsigned long size, unsigned char value,
                                             unsigned char* X,
                                             caffe2::OpenCLContext* context) {
  CAFFE_THROW("Not implemented.");
}

template<>
void Set<int, caffe2::OpenCLContext>(unsigned long size, int value,
                                     int* X,
                                     caffe2::OpenCLContext* context) {
  CAFFE_THROW("Not implemented.");
}

template<>
void Set<long, caffe2::OpenCLContext>(unsigned long size, long value,
                                      long* X,
                                      caffe2::OpenCLContext* context) {
  CAFFE_THROW("Not implemented.");
}

template<>
void Set<double, caffe2::OpenCLContext>(unsigned long size, double value,
                                        double* X,
                                        caffe2::OpenCLContext* context) {
  CAFFE_THROW("Not implemented.");
}

static constexpr const char* kColwiseMax = R"CLC(
kernel void kMathColwiseMax(
  const __global float* in_mat,
  __global float* out_mat,
  const int N_rows,
  const int D_cols)
{
  int c_ix = get_global_id(0);

  if (c_ix >= D_cols)
  {
    return;
  }

  for (int ix = c_ix; ix < N_rows * D_cols; ix += D_cols)
  {
    if (isnan(out_mat[c_ix]) || in_mat[ix] > out_mat[c_ix])
      out_mat[c_ix] = in_mat[ix];
  }
}
)CLC";

static void runColwiseMaxKernel(
  const int N_rows,
  const int D_cols,
  const void* x_in,
  void* y_out,
  OpenCLContext* context,
  size_t type_size)
{
  size_t out_mem_size = D_cols * type_size;
  auto mat_y_out = context->New(out_mem_size);
  runFillWithNaN(*(cl::Buffer*)mat_y_out.first, D_cols, context);

  //Now, as all output values were initialized, we can begin to search for Max per column:
  auto kernel = context->BuildKernelCached(kColwiseMax, "", "kMathColwiseMax");
  size_t in_mem_size = N_rows * D_cols * type_size;
  auto mat_x_in = context->New(in_mem_size);

  context->CopyBytes<CPUContext, OpenCLContext>(in_mem_size, x_in, mat_x_in.first);

  OPENCL_CHECK(kernel.setArg(0, *(cl::Buffer*)mat_x_in.first));
  OPENCL_CHECK(kernel.setArg(1, *(cl::Buffer*)mat_y_out.first));
  OPENCL_CHECK(kernel.setArg(2, N_rows));
  OPENCL_CHECK(kernel.setArg(3, D_cols));

  cl::Event event;

  context->enqueue(
        kernel,
        cl::NullRange,
        cl::NDRange(D_cols), //The dimension of work is the number of matrix columns
        cl::NullRange,
        NULL,
        &event);
  event.wait();

  context->CopyBytes<OpenCLContext, CPUContext>(out_mem_size, mat_y_out.first, y_out);
  context->Delete(mat_x_in.first);
  context->Delete(mat_y_out.first);
}

#define CAFFE2_SPECIALIZED_COLWISEMAX(T)            \
template <>                                         \
void ColwiseMax<T, OpenCLContext>(                  \
  const int N,                                      \
  const int D,                                      \
  const T* x,                                       \
  T* y,                                             \
  OpenCLContext* ctx)                               \
{                                                   \
  runColwiseMaxKernel(N, D, x, y, ctx, sizeof(T));  \
}
CAFFE2_SPECIALIZED_COLWISEMAX(float)
#undef CAFFE2_SPECIALIZED_COLWISEMAX


static constexpr const char* kTranspose1D = R"CLC(
kernel void kMathTranspose1D(
  const global float* X,
  global float* Y) {

  int idx = get_global_id(0);

  Y[idx] = X[idx];
}
)CLC";

static constexpr const char* kTranspose2D = R"CLC(
kernel void kMathTranspose2D(
  constant int* x_dims,
  constant int* y_dims,
  constant int* axes,
  const global float* X,
  global float* Y) {

  int idx_row = get_global_id(0);
  int idx_col = get_global_id(1);
  int new_idx_row = idx_row;
  int new_idx_col = idx_col;

  /* Assume both *_dims[] arrays contain Rows count at[0] and Columns count at[1]: */

  /* This means there is a coordinates permutation and matrix must be transposed: */
  if (axes[0] == 1 && axes[1] == 0) {
    new_idx_row = idx_col;
    new_idx_col = idx_row;
  }

  Y[new_idx_row * y_dims[1] + new_idx_col] = X[idx_row * x_dims[1] + idx_col];
}
)CLC";

static constexpr const char* kTranspose3D = R"CLC(
kernel void kMathTranspose3D(
  constant int* x_dims,
  constant int* y_dims,
  constant int* axes,
  const global float* X,
  global float* Y) {

  int idx[3];
  int new_idx[3];
  idx[0] = get_global_id(0);
  idx[1] = get_global_id(1);
  idx[2] = get_global_id(2);

  /* Assume both *_dims[] arrays contain Layers count at[0],
     Rows count at[1], Columns count at[2]: */

  new_idx[0] = idx[axes[0]];
  new_idx[1] = idx[axes[1]];
  new_idx[2] = idx[axes[2]];

  Y[new_idx[0] * y_dims[2] * y_dims[1] + new_idx[1] * y_dims[2] + new_idx[2]] =
    X[idx[0] * x_dims[2] * x_dims[1] + idx[1] * x_dims[2] + idx[2]];
}
)CLC";

static constexpr const char* kTranspose4D = R"CLC(
kernel void kMathTranspose4D(
  constant int* x_dims,
  constant int* y_dims,
  constant int* axes,
  const global float* X,
  global float* Y) {

  int idx[4];
  int new_idx[4];

  idx[0] = get_global_id(0) / x_dims[1];
  idx[1] = get_global_id(0) - idx[0] * x_dims[1];
  idx[2] = get_global_id(1);
  idx[3] = get_global_id(2);

  new_idx[0] = idx[axes[0]];
  new_idx[1] = idx[axes[1]];
  new_idx[2] = idx[axes[2]];
  new_idx[3] = idx[axes[3]];

  Y[new_idx[0] * y_dims[3] * y_dims[2] * y_dims[1] + new_idx[1] * y_dims[3] * y_dims[2] + new_idx[2] * y_dims[3] + new_idx[3]] =
    X[idx[0] * x_dims[3] * x_dims[2] * x_dims[1] + idx[1] * x_dims[3] * x_dims[2] + idx[2] * x_dims[3] + idx[3]];
}
)CLC";


template <>
void Transpose<cl::Buffer, OpenCLContext>(
    const int ndim,
    const int* x_dims,
    const int* axes,
    const cl::Buffer* X,
    cl::Buffer* Y,
    OpenCLContext* context) {

  CAFFE_ENFORCE_GE(ndim, 1, "Number of dimensions must be >= 1, %d given.", ndim);
  CAFFE_ENFORCE_LE(ndim, 4, "Number of dimensions must be <= 4, %d given.", ndim);

  cl::NDRange problem_range;
  cl::Kernel kernel;
  switch (ndim) {
  case 1:
    problem_range = cl::NDRange(x_dims[0]);
    kernel = context->BuildKernelCached(kTranspose1D, "", "kMathTranspose1D");
    break;
  case 2:
    problem_range = cl::NDRange(x_dims[0], x_dims[1]);
    kernel = context->BuildKernelCached(kTranspose2D, "", "kMathTranspose2D");
    break;
  case 3:
    problem_range = cl::NDRange(x_dims[0], x_dims[1], x_dims[2]);
    kernel = context->BuildKernelCached(kTranspose3D, "", "kMathTranspose3D");
    break;
  case 4:
    problem_range = cl::NDRange(x_dims[0] * x_dims[1], x_dims[2], x_dims[3]);
    kernel = context->BuildKernelCached(kTranspose4D, "", "kMathTranspose4D");
    break;
  default:
    CAFFE_THROW("Unsupported number of axes ndim ");
  }

  int y_dims[ndim] = {0,};
  auto x_dims_buff = context->New(ndim * sizeof(int));
  auto y_dims_buff = context->New(ndim * sizeof(int));
  auto axes_buff = context->New(ndim * sizeof(int));

  for (int i = 0; i < ndim; ++i)
    y_dims[i] = x_dims[axes[i]];

  context->CopyBytes<CPUContext, OpenCLContext>(ndim * sizeof(int), x_dims, x_dims_buff.first);
  context->CopyBytes<CPUContext, OpenCLContext>(ndim * sizeof(int), y_dims, y_dims_buff.first);
  context->CopyBytes<CPUContext, OpenCLContext>(ndim * sizeof(int), axes, axes_buff.first);

  switch (ndim) {
  case 1:
    OPENCL_CHECK(kernel.setArg(0, *X));
    OPENCL_CHECK(kernel.setArg(1, *Y));
    break;
  case 2:
  case 3:
  case 4:
    OPENCL_CHECK(kernel.setArg(0, *reinterpret_cast<cl::Buffer*>(x_dims_buff.first)));
    OPENCL_CHECK(kernel.setArg(1, *reinterpret_cast<cl::Buffer*>(y_dims_buff.first)));
    OPENCL_CHECK(kernel.setArg(2, *reinterpret_cast<cl::Buffer*>(axes_buff.first)));
    OPENCL_CHECK(kernel.setArg(3, *X));
    OPENCL_CHECK(kernel.setArg(4, *Y));
    break;
  default:
    CAFFE_THROW("Unsupported number of axes ndim ");
  }

  cl::Event event;

  context->enqueue(
        kernel,
        cl::NullRange,
        problem_range,
        cl::NullRange,
        NULL,
        &event);
  event.wait();

  context->Delete(x_dims_buff.first);
  context->Delete(y_dims_buff.first);
  context->Delete(axes_buff.first);
}

static constexpr const char* kSum = R"CLC(
kernel void kMathSum(const __global float* in_mat,
              const int N_rows,
              __global float* out)
{
  *out = 0;
  for (int i = 0; i < N_rows; i++){
    *out += in_mat[i];
  }
}
)CLC";

static void runSumKernel(
  const int N_rows,
  const void* x_in,
  float *out,
  OpenCLContext* context,
  size_t type_size)
{
  auto kernel = context->BuildKernelCached(kSum, "", "kMathSum");
  size_t in_mem_size = N_rows * type_size;
  auto mat_x_in = context->New(in_mem_size);
  auto out_bf = context->New(sizeof(float));

  context->CopyBytes<CPUContext, OpenCLContext>(in_mem_size, x_in, mat_x_in.first);

  OPENCL_CHECK(kernel.setArg(0, *(cl::Buffer*)mat_x_in.first));
  OPENCL_CHECK(kernel.setArg(1, N_rows));
  OPENCL_CHECK(kernel.setArg(2, *(cl::Buffer*)out_bf.first));

  cl::Event event;

  context->enqueue(
        kernel,
        cl::NullRange,
        cl::NDRange(1),
        cl::NullRange,
        NULL,
        &event);
  event.wait();

  context->CopyBytes<OpenCLContext, CPUContext>(sizeof(float), out_bf.first, out);
  context->Delete(mat_x_in.first);
  context->Delete(out_bf.first);
}

#define CAFFE2_SPECIALIZED_SUM(T)                     \
  template <>                                         \
  void Sum<T, OpenCLContext>(                         \
    const int N,                                      \
    const T* x,                                       \
    T* y,                                             \
    OpenCLContext* ctx,                               \
    Tensor* /* unused */)                             \
  {                                                   \
  runSumKernel(N, x, y, ctx, sizeof(T));              \
  }
CAFFE2_SPECIALIZED_SUM(float)
#undef CAFFE2_SPECIALIZED_SUM

static constexpr const char* kReduce = R"CLC(
#ifndef FUNCTION
#define FUNCTION mul
#endif

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef INITIAL_OUTPUT_VALUE
#define INITIAL_OUTPUT_VALUE 0
#endif

DATA_TYPE sum(DATA_TYPE A, DATA_TYPE B)
{
    return A + B;
}

DATA_TYPE mul(DATA_TYPE A, DATA_TYPE B)
{
    return A * B;
}

 /**
  * It is not easy to synchronize work-items in different work-groups.
  * Because of that, we're using a trivial unparallelized implementation for the
  * time being.
  * Source:
  * https://stackoverflow.com/questions/20613013/opencl-float-sum-reduction
  */
kernel void kMathReduce( int size, global DATA_TYPE *x, global DATA_TYPE *y)
{
    if (get_global_id(0) == 0) {

        // Store the partial result in private memory for optimization
        DATA_TYPE y_tmp = INITIAL_OUTPUT_VALUE;
        for (int i = 0; i < size; ++i) {
            y_tmp = FUNCTION(y_tmp, x[i]);
        }

        *y = y_tmp;
    }
}
)CLC";

template <typename T> static void runReductionKernel(const int N, const T* x, T* y,
                               OpenCLContext* context, cl::Kernel* kernel)
{
  cl::Buffer *x_buf = (cl::Buffer*)context->New(N * sizeof(T)).first;
  cl::Buffer *y_buf = (cl::Buffer*)context->New(sizeof(T)).first;
  OPENCL_CHECK(kernel->setArg(0, N));
  OPENCL_CHECK(kernel->setArg(1, *x_buf));
  OPENCL_CHECK(kernel->setArg(2, *y_buf));

  cl::Event event;

  context->CopyBytes<CPUContext, OpenCLContext>(sizeof(T) * N, x, x_buf);
  context->CopyBytes<CPUContext, OpenCLContext>(sizeof(T), y, y_buf);

  context->enqueue(
        *kernel,
        cl::NullRange,
        cl::NDRange(1),
        cl::NullRange,
        NULL,
        &event);
  event.wait();

  context->CopyBytes<OpenCLContext, CPUContext>(sizeof(T), y_buf, y);

  context->Delete(x_buf);
  context->Delete(y_buf);
}

#define DELEGATE_REDUCTION_FUNCTION(T, Funcname, kName, initial_output_value)   \
template <>                                                                     \
void Funcname<T, OpenCLContext>(                                                \
  const int N,                                                                  \
  const T* x,                                                                   \
  T *y,                                                                         \
  Tensor* scratch_ptr,                                                          \
  OpenCLContext* context                                                        \
  ) {                                                                           \
  cl::Kernel* kernel = new cl::Kernel(                                          \
      context->BuildKernelCached(                                               \
        kReduce,                                                                \
        " -DDATA_TYPE=" #T                                                      \
        " -DFUNCTION=" #kName                                                   \
        " -DINITIAL_OUTPUT_VALUE=" #initial_output_value,                       \
        "kMathReduce"                                                           \
        )                                                                       \
      );                                                                        \
    runReductionKernel(N, x, y, context, kernel);                               \
  delete kernel;                                                                \
  }
DELEGATE_REDUCTION_FUNCTION(float, ReduceMin, min, FLT_MAX)
DELEGATE_REDUCTION_FUNCTION(float, ReduceMax, max, -FLT_MAX)
DELEGATE_REDUCTION_FUNCTION(int, ReduceMin, min, INT_MAX)
DELEGATE_REDUCTION_FUNCTION(int, ReduceMax, max, INT_MIN)
DELEGATE_REDUCTION_FUNCTION(long, ReduceMin, min, LONG_MAX)
DELEGATE_REDUCTION_FUNCTION(long, ReduceMax, max, LONG_MIN)
#undef DELEGATE_REDUCTION_FUNCTION

#define DELEGATE_ELEMWISE_FUNCTION(T, Funcname, ReduceKind)                     \
template <>                                                                     \
void Funcname<T, OpenCLContext>(                                                \
  const int N,                                                                  \
  const T* x,                                                                   \
  const T* y,                                                                   \
  T* z,                                                                         \
  OpenCLContext* context                                                        \
  ) {                                                                           \
    ReduceKind(N, x, z, static_cast<Tensor *>(nullptr), context);               \
  }

DELEGATE_ELEMWISE_FUNCTION(float, ElemwiseMax, ReduceMax)
DELEGATE_ELEMWISE_FUNCTION(int, ElemwiseMax, ReduceMax)
DELEGATE_ELEMWISE_FUNCTION(long, ElemwiseMax, ReduceMax)

#undef DELEGATE_ELEMWISE_FUNCTION

static constexpr const char* kDot = R"CLC(
kernel void kMathDot(const int N_rows,
              const __global float* A,
              const __global float* B,
              __global float* out)
{
  *out = 0;
  for (int i = 0; i < N_rows; i++){
    *out += A[i] * B[i];
  }
}
)CLC";

static void runDotKernel(
  const int N_rows,
  const void* a_in,
  const void* b_in,
  float *out,
  OpenCLContext* context,
  size_t type_size)
{
  auto kernel = context->BuildKernelCached(kDot, "", "kMathDot");
  size_t in_mem_size = N_rows * type_size;
  auto mat_a_in = context->New(in_mem_size);
  auto mat_b_in = context->New(in_mem_size);
  auto out_bf = context->New(sizeof(float));

  context->CopyBytes<CPUContext, OpenCLContext>(in_mem_size, a_in, mat_a_in.first);
  context->CopyBytes<CPUContext, OpenCLContext>(in_mem_size, b_in, mat_b_in.first);

  OPENCL_CHECK(kernel.setArg(0, N_rows));
  OPENCL_CHECK(kernel.setArg(1, *(cl::Buffer*)mat_a_in.first));
  OPENCL_CHECK(kernel.setArg(2, *(cl::Buffer*)mat_b_in.first));
  OPENCL_CHECK(kernel.setArg(3, *(cl::Buffer*)out_bf.first));

  cl::Event event;

  context->enqueue(
        kernel,
        cl::NullRange,
        cl::NDRange(1),
        cl::NullRange,
        NULL,
        &event);
  event.wait();

  context->CopyBytes<OpenCLContext, CPUContext>(sizeof(float), out_bf.first, out);
  context->Delete(mat_a_in.first);
  context->Delete(mat_b_in.first);
  context->Delete(out_bf.first);
}

#define CAFFE2_SPECIALIZED_DOT(T)                     \
template<>                                            \
void Dot<T, OpenCLContext>(                           \
    const int N, const T* a, const T* b, T* y,        \
    OpenCLContext* context) {                         \
  runDotKernel(N, a, b, y, context, sizeof(T));       \
}
CAFFE2_SPECIALIZED_DOT(float)
#undef CAFFE2_SPECIALIZED_DOT

static constexpr const char* kScale = R"CLC(
kernel void kMathScale(const int N_rows,
              const float alpha,
              const __global float* A,
              __global float* out)
{
  int col_ix = get_global_id(0);
  if (col_ix >= N_rows)
    return;
  out[col_ix] = A[col_ix] * alpha;
}
)CLC";

static void runScaleKernel(
  const int N_rows,
  const float alpha,
  const void* a_in,
  void* out,
  OpenCLContext* context,
  size_t type_size)
{
  auto kernel = context->BuildKernelCached(kScale, "", "kMathScale");
  size_t matrix_mem_size = N_rows * type_size;
  auto mat_a_in = context->New(matrix_mem_size);
  auto out_bf = context->New(matrix_mem_size);

  context->CopyBytes<CPUContext, OpenCLContext>(matrix_mem_size, a_in, mat_a_in.first);

  OPENCL_CHECK(kernel.setArg(0, N_rows));
  OPENCL_CHECK(kernel.setArg(1, alpha));
  OPENCL_CHECK(kernel.setArg(2, *(cl::Buffer*)mat_a_in.first));
  OPENCL_CHECK(kernel.setArg(3, *(cl::Buffer*)out_bf.first));

  cl::Event event;

  context->enqueue(
        kernel,
        cl::NullRange,
        cl::NDRange(N_rows),  //The dimension of work is the number of matrix rows
        cl::NullRange,
        NULL,
        &event);
  event.wait();

  context->CopyBytes<OpenCLContext, CPUContext>(matrix_mem_size, out_bf.first, out);
  context->Delete(mat_a_in.first);
  context->Delete(out_bf.first);
}

#define CAFFE2_SPECIALIZED_SCALE(T)                                            \
  template <>                                                                  \
  void Scale<T, T, OpenCLContext>(                                             \
      const int N,                                                             \
      const float alpha,                                                       \
      const T* x,                                                              \
      T* out,                                                                  \
      OpenCLContext* context) {                                                \
    runScaleKernel(N, alpha, x, out, context, sizeof(T));                      \
  }
CAFFE2_SPECIALIZED_SCALE(float)
#undef CAFFE2_SPECIALIZED_SCALE

} // math
} // caffe2
