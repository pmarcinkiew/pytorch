/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Implements the math functions for CPU.
// The implementation in this file allows us to route the underlying numerical
// computation library to different backends. Notably:
// (1) For all BLAS-related functions, one can explicitly request a BLAS backend
//     such as MKL, openblas or Atlas. To see the set of supported backends
//     currently provided, check //third_party/blas/.
// (2) If one chooses to link against MKL, we utilize MKL's vector math library
//     (VML) for a few functions such as Exp and Log.
// (3) Fallback implementations are provided in Eigen for cross-platform
//     support. Since Eigen is a header-only library and supports a number of
//     platforms, it allows one to quickly port Caffe2 to different platforms
//     where BLAS may not be present.

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
DELEGATE_SIMPLE_UNARY_FUNCTION(InvSqrt)
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

