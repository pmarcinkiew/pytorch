#include "caffe2/core/common.h"

#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/timer.h"
#include "caffe2/core/workspace.h"
#include "caffe2/operators/filler_op.h"
#include "caffe2/utils/proto_utils.h"
#include <gtest/gtest.h>
#include "../context.h"

#include <CL/cl.hpp>

#ifdef __ANDROID__
#include <android/log.h>
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#else
#define ALOGI(...) printf(__VA_ARGS__)
#endif

namespace caffe2 {

#define EVILFLOATTEST(val1, val2, val3, context, op)                                                \
do{                                                                                                 \
  float u = val1, v = val2, z = val3;                                                                \
  size_t size = sizeof(float);                                                                      \
  auto ub = context.New(size);                                                                      \
  auto vb = context.New(size);                                                                      \
  auto zb = context.New(size);                                                                      \
  context.CopyBytes<CPUContext, OpenCLContext>(size, &u, ub.first);                                 \
  context.CopyBytes<CPUContext, OpenCLContext>(size, &v, vb.first);                                 \
  caffe2::math::op(1, (float *)ub.first, (float *)vb.first, (float *)zb.first, &context);           \
  float z_tmp;                                                                                      \
  context.CopyBytes<OpenCLContext, CPUContext>(size, zb.first, &z_tmp);                             \
  EXPECT_FLOAT_EQ(z_tmp, z);                                                                        \
  context.Delete(ub.first);                                                                         \
  context.Delete(vb.first);                                                                         \
  context.Delete(zb.first);                                                                         \
}while (false);


#define EVILONEFLOATTEST(a, y, op)                                            \
do{                                                                           \
  float ab = a, yb;                                                           \
  caffe2::math::op(1, &ab, &yb, &context);                                    \
  EXPECT_FLOAT_EQ(yb, y);                                                     \
}while (false);

#define EVILONEFLOATTESTEPS(a, y, op, eps)                                    \
do{                                                                           \
  float ab = a, yb;                                                           \
  caffe2::math::op(1, &ab, &yb, &context);                                    \
  EXPECT_TRUE(fabs(yb - y) < eps);                                            \
}while (false);

#define EVILONEDOUBLETEST(a, y, op)                                           \
do{                                                                           \
  double ab = a, yb;                                                          \
  caffe2::math::op(1, &ab, &yb, &context);                                    \
  EXPECT_DOUBLE_EQ(yb, y);                                                    \
}while (false);

TEST(OpenCL, AddOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  //integers:
  int a[5] = {5, 4, 3, 2, 1},
      b[5] = {-1, -2, -3, -4, -5},
      y[5];

  auto ab = context.New(5 * sizeof(int));
  auto bb = context.New(5 * sizeof(int));
  auto yb = context.New(5 * sizeof(int));
  context.CopyBytes<CPUContext, OpenCLContext>(5 * sizeof(int), a, ab.first);
  context.CopyBytes<CPUContext, OpenCLContext>(5 * sizeof(int), b, bb.first);
  caffe2::math::Add(5, (int *)ab.first, (int *)bb.first, (int *)yb.first, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(5 * sizeof(int), yb.first, y);
  EXPECT_EQ(y[0], 4);
  EXPECT_EQ(y[1], 2);
  EXPECT_EQ(y[2], 0);
  EXPECT_EQ(y[3], -2);
  EXPECT_EQ(y[4], -4);
  context.Delete(ab.first);
  context.Delete(bb.first);
  context.Delete(yb.first);

  //floats:
  EVILFLOATTEST(1, 2, 3, context, Add);
}

TEST(OpenCL, SubOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  EVILFLOATTEST(3, 2, 1, context, Sub);
}

TEST(OpenCL, MulOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  EVILFLOATTEST(2, 2, 4, context, Mul);
}

TEST(OpenCL, DivOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  EVILFLOATTEST(4, 2, 2, context, Div);
}

TEST(OpenCL, LogOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  EVILONEFLOATTEST(1, 0, Log);
}

TEST(OpenCL, CosOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  EVILONEFLOATTEST(0, 1, Cos);
}

TEST(OpenCL, SinOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  EVILONEFLOATTEST(0, 0, Sin);
}

TEST(OpenCL, SqrtOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  EVILONEFLOATTEST(4, 2, Sqrt);
}

TEST(OpenCL, RsqrtOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  EVILONEFLOATTESTEPS(0.25, 2, Rsqrt, 0.01);
}

TEST(OpenCL, NotOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  bool ab = false, yb;
  caffe2::math::Not(1, &ab, &yb, &context);
  EXPECT_TRUE(yb);
}

TEST(OpenCL, RowwiseMax) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  //Test 1: 4 rows, 3 columns

  float matrix_in[] = {
        1.1,  2.1,  3.1,
       17.2,  8.2,  1.2,
        5.3, 22.3,  4.3,
      -89.4, 99.4, -9.4
  };

  float matrix_out[4];
  caffe2::math::RowwiseMax(4, 3, matrix_in, matrix_out, &context);

  EXPECT_FLOAT_EQ(matrix_out[0], 3.1);
  EXPECT_FLOAT_EQ(matrix_out[1], 17.2);
  EXPECT_FLOAT_EQ(matrix_out[2], 22.3);
  EXPECT_FLOAT_EQ(matrix_out[3], 99.4);

  //Test 2: 1 row, 1 column
  float matrix_in2[] = {1.1};
  float matrix_out2[1];
  caffe2::math::RowwiseMax(1, 1, matrix_in2, matrix_out2, &context);
  EXPECT_FLOAT_EQ(matrix_out2[0], 1.1);

  //Test 3: 1 row, 5 columns
  float matrix_in3[] = {1.1, -2.0, 4.8, 4.79, -0.5};
  float matrix_out3[1];
  caffe2::math::RowwiseMax(1, 5, matrix_in3, matrix_out3, &context);
  EXPECT_FLOAT_EQ(matrix_out3[0], 4.8);

  //Test 4: 5 rows, 1 column
  float matrix_in4[] = {1.1,
                       -2.0,
                        4.8,
                        4.79,
                       -0.5};
  float matrix_out4[5];
  caffe2::math::RowwiseMax(5, 1, matrix_in4, matrix_out4, &context);
  EXPECT_FLOAT_EQ(matrix_out4[0], 1.1);
  EXPECT_FLOAT_EQ(matrix_out4[1], -2.0);
  EXPECT_FLOAT_EQ(matrix_out4[2], 4.8);
  EXPECT_FLOAT_EQ(matrix_out4[3], 4.79);
  EXPECT_FLOAT_EQ(matrix_out4[4], -0.5);

}

TEST(OpenCL, RowwiseMax_GPU) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  float *matrix_in = new float[10000*10000];
  float *matrix_out = new float[10000];
  caffe2::math::RowwiseMax(10000, 10000, matrix_in, matrix_out, &context);
  delete[] matrix_in;
  delete[] matrix_out;
}

TEST(OpenCL, RowwiseMax_CPU) {
  DeviceOption option;
  caffe2::CPUContext context(option);

  float *matrix_in = new float[10000*10000];
  float *matrix_out = new float[10000];
  caffe2::math::RowwiseMax(10000, 10000, matrix_in, matrix_out, &context);
  delete[] matrix_in;
  delete[] matrix_out;
}

void compareGemm(size_t M, size_t N, float* a, float* b) {
  for (int i = 0; i < M; ++i) for(int j = 0; j < N; ++j) {
    if (a[i * N + j] != b[i * N + j]) {
      printf("ERROR detected a[%d][%d](%f) != b[%d][%d](%f)\n",
             i, j, a[i * N + j], i, j, b[i * N + j]);
      EXPECT_TRUE(false && "Gemm is not identical");
    }
  }
}
void compareGemm(size_t M, size_t N, volatile float* a, float* b) {
  for (int i = 0; i < M; ++i) for(int j = 0; j < N; ++j) {
    if (a[i * N + j] != b[i * N + j]) {
      printf("ERROR detected a[%d][%d](%f) != b[%d][%d](%f)\n",
             i, j, a[i * N + j], i, j, b[i * N + j]);
      EXPECT_TRUE(false && "Gemm is not identical");
    }
  }
}

TEST(OpenCL, GemmBatched) {

    DeviceOption option;
    caffe2::OpenCLContext context(option);

    int M = 3, N = 3, K = 3;
    int batch_size = 2;
    size_t size = sizeof(float);
    size_t memSizeA = M * K * size * batch_size;
    size_t memSizeB = K * N * size * batch_size;
    size_t memSizeC = M * N * size * batch_size;
    auto ab = context.New(memSizeA);
    auto bb = context.New(memSizeB);
    auto cb = context.New(memSizeC);

    float A[] = {
        1.0, 2.0, 3.0,
        2.0, 8.0, 1.0,
        5.0, 2.0, 4.0,
        1.0, 2.0, 3.0,
        2.0, 8.0, 1.0,
        5.0, 2.0, 4.0
    };

    float B[] = {
        2.0, 2.0, 2.0,
        2.0, 2.0, 2.0,
        2.0, 2.0, 2.0,
        2.0, 2.0, 2.0,
        2.0, 2.0, 2.0,
        2.0, 2.0, 2.0
    };

    float C[] = {
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0,
        0,  0,  0
    };

    float result[] = {
        12.0, 12.0, 12.0,
        22.0, 22.0, 22.0,
        22.0, 22.0, 22.0,
        12.0, 12.0, 12.0,
        22.0, 22.0, 22.0,
        22.0, 22.0, 22.0
    };

    context.CopyBytes<CPUContext, OpenCLContext>(memSizeA, A, ab.first);
    context.CopyBytes<CPUContext, OpenCLContext>(memSizeB, B, bb.first);
    context.CopyBytes<CPUContext, OpenCLContext>(memSizeC, (float *)C, cb.first);

    caffe2::math::GemmBatched(
        CblasNoTrans,
        CblasNoTrans,
        batch_size,
        M,
        N,
        K,
        1,
        (float*)ab.first,
        (float*)bb.first,
        0,
        (float*)cb.first,
        &context);

    context.CopyBytes<OpenCLContext, CPUContext>(memSizeC, cb.first, (float *)C);
    context.Delete(ab.first);
    context.Delete(bb.first);
    context.Delete(cb.first);

    EXPECT_FLOAT_EQ(result[0], C[0]);
    EXPECT_FLOAT_EQ(result[3], C[3]);
    EXPECT_FLOAT_EQ(result[8], C[8]);
    EXPECT_FLOAT_EQ(result[12], C[12]);
    EXPECT_FLOAT_EQ(result[17], C[17]);
}

void gemmWithCPUBuffers(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                        size_t M, size_t N, size_t K, float alpha, float* A,
                        float* B, float beta,
                        volatile float* C, caffe2::OpenCLContext* context) {
  size_t size = sizeof(float);
  size_t memSizeA = M * K * size;
  size_t memSizeB = K * N * size;
  size_t memSizeY = M * N * size;
  auto ab = context->New(memSizeA);
  auto bb = context->New(memSizeB);
  auto yb = context->New(memSizeY);
  context->CopyBytes<CPUContext, OpenCLContext>(memSizeA, A, ab.first);
  context->CopyBytes<CPUContext, OpenCLContext>(memSizeB, B, bb.first);
  context->CopyBytes<CPUContext, OpenCLContext>(memSizeY, (float *)C, yb.first);
  caffe2::math::Gemm<float>(
    TransA,
    TransB,
    M,
    N,
    K,
    alpha,
    (float*)ab.first,
    (float*)bb.first,
    beta,
    (float*)yb.first,
    context);
  context->CopyBytes<OpenCLContext, CPUContext>(memSizeY, yb.first, (float *)C);
  context->Delete(ab.first);
  context->Delete(bb.first);
  context->Delete(yb.first);
}

void matrixTestGemm(size_t M, size_t N, size_t K, float* A, float* B,
                    float* R, caffe2::OpenCLContext& context)
{
  volatile float C[M * N] = {0};
  gemmWithCPUBuffers(
    CblasNoTrans,
    CblasNoTrans,
    M,
    N,
    K,
    1,
    A,
    B,
    0,
    C,
    &context);

  compareGemm(M, N, C, R);
}

void matrixTestGemmTrans(size_t M, size_t N, size_t K, float* A, float* B,
                    float* R, caffe2::OpenCLContext& context)
{
  volatile float C[M * N] = {0};
  gemmWithCPUBuffers(
    CblasTrans,
    CblasTrans,
    M,
    N,
    K,
    1,
    A,
    B,
    0,
    C,
    &context);

  compareGemm(M, N, C, R);
}
void matrixTestGemmTransA(size_t M, size_t N, size_t K, float* A, float* B,
                    float* R, caffe2::OpenCLContext& context)
{
  volatile float C[M * N] = {0};
  gemmWithCPUBuffers(
    CblasTrans,
    CblasNoTrans,
    M,
    N,
    K,
    1,
    A,
    B,
    0,
    C,
    &context);

  compareGemm(M, N, C, R);
}
void matrixTestGemmTransB(size_t M, size_t N, size_t K, float* A, float* B,
                    float* R, caffe2::OpenCLContext& context)
{
  volatile float C[M * N] = {0};
  gemmWithCPUBuffers(
    CblasNoTrans,
    CblasTrans,
    M,
    N,
    K,
    1,
    A,
    B,
    0,
    C,
    &context);

  compareGemm(M, N, C, R);
}


void matrixTestGemmBeta(size_t M, size_t N, size_t K, float* A, float* B,
                    float* R, caffe2::OpenCLContext& context)
{
  volatile float C[M * N] = {0};
  memcpy((float *)C, R, sizeof(C));

  gemmWithCPUBuffers(
    CblasNoTrans,
    CblasTrans,
    M,
    N,
    K,
    0,
    A,
    B,
    1,
    C,
    &context);

  compareGemm(M, N, C, R);
}

TEST(OpenCL, Gemm213OpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  float
        A[2 * 3] = { 1, 2, 3,
                    4, 5, 6},
        B[3 * 1] = { 1,
                    2,
                    3},
        R[2 * 1] = {14,
                    32};
  matrixTestGemm(2, 1, 3, A, B, R, context);
}

TEST(OpenCL, Gemm223NNOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  float
        AN[2 * 3] = { 1, 2, 3,
                      4, 5, 6},
        BN[3 * 2] = { 1, 2,
                      3, 4,
                      5, 6},
        R[2 * 2] = {22, 28,
                    49, 64};
  matrixTestGemm(2, 2, 3, AN, BN, R, context);
}
TEST(OpenCL, Gemm223TTOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  float
        AT[2 * 3] = { 1, 4,
                      2, 5,
                      3, 6},
        BT[3 * 2] = { 1, 3, 5,
                      2, 4, 6},
        R[2 * 2] = {22, 28,
                    49, 64};
  matrixTestGemmTrans( 2, 2, 3, AT, BT, R, context);
}
TEST(OpenCL, Gemm223TNOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  float
        AT[2 * 3] = { 1, 4,
                      2, 5,
                      3, 6},
        BN[3 * 2] = { 1, 2,
                      3, 4,
                      5, 6},
        R[2 * 2] = {22, 28,
                    49, 64};
  matrixTestGemmTransA(2, 2, 3, AT, BN, R, context);
}
TEST(OpenCL, Gemm223NTOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  float
        AN[2 * 3] = { 1, 2, 3,
                      4, 5, 6},
        BT[3 * 2] = { 1, 3, 5,
                      2, 4, 6},
        R[2 * 2] = {22, 28,
                    49, 64};
    matrixTestGemmTransB(2, 2, 3, AN, BT, R, context);
}

TEST(OpenCL, Gemm223BetaOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  float
        AN[2 * 3] = { 1, 2, 3,
                      4, 5, 6},
        BN[3 * 2] = { 1, 2,
                      3, 4,
                      5, 6},
        R[2 * 2] = {22, 28,
                    49, 64};
  matrixTestGemmBeta(2, 2, 3, AN, BN, R, context);
}


// This test is very resource consuming. It takes more than 30 seconds
// at GT730 with i7 CPU.
// #define LARGETEST

#ifdef LARGETEST

TEST(OpenCL, GemmLargeOpenCL) {
  DeviceOption option;
  caffe2::OpenCLContext  context(option);
  float*     A = new float[2000 * 3000],
            *B = new float[3000 * 1000],
            *C = new float[2000 * 1000];
  gemmWithCPUBuffers(
      CblasNoTrans,
      CblasNoTrans,
      2000,
      1000,
      3000,
      1,
      A,
      B,
      0,
      C,
      &context);
  delete[] A;
  delete[] B;
  delete[] C;
}

TEST(OpenCL, GemmLargeMulCPU) {
  DeviceOption option;
  caffe2::CPUContext  context(option);
  float*     A = new float[2000 * 3000],
            *B = new float[3000 * 1000],
            *C = new float[2000 * 1000];

  caffe2::math::Gemm(
      CblasNoTrans,
      CblasNoTrans,
      2000,
      1000,
      3000,
      1,
      A,
      B,
      0,
      C,
      &context);
  delete[] A;
  delete[] B;
  delete[] C;
}

#endif //LARGETEST

TEST(OpenCL, ColwiseMax) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  //Test 1: 4 rows, 3 columns
  float matrix_in[] = {
        1.1,  2.1,  3.1,
       17.2,  8.2,  1.2,
        5.3, 22.3,  4.3,
      -89.4, 99.4, -9.4
  };

  float matrix_out[3];
  caffe2::math::ColwiseMax(4, 3, matrix_in, matrix_out, &context);

  EXPECT_FLOAT_EQ(matrix_out[0], 17.2);
  EXPECT_FLOAT_EQ(matrix_out[1], 99.4);
  EXPECT_FLOAT_EQ(matrix_out[2], 4.3);

  //Test 2: 1 row, 1 column
  float matrix_in2[] = {1.1};
  float matrix_out2[1];
  caffe2::math::ColwiseMax(1, 1, matrix_in2, matrix_out2, &context);
  EXPECT_FLOAT_EQ(matrix_out2[0], 1.1);

  //Test 3: 1 row, 5 columns
  float matrix_in3[] = {1.1, -2.0, 4.8, 4.79, -0.5};
  float matrix_out3[5];
  caffe2::math::ColwiseMax(1, 5, matrix_in3, matrix_out3, &context);
  EXPECT_FLOAT_EQ(matrix_out3[0], 1.1);
  EXPECT_FLOAT_EQ(matrix_out3[1], -2.0);
  EXPECT_FLOAT_EQ(matrix_out3[2], 4.8);
  EXPECT_FLOAT_EQ(matrix_out3[3], 4.79);
  EXPECT_FLOAT_EQ(matrix_out3[4], -0.5);

  //Test 4: 5 rows, 1 column
  float matrix_in4[] = {1.1,
                       -2.0,
                        4.8,
                        4.79,
                       -0.5};
  float matrix_out4[1];
  caffe2::math::ColwiseMax(5, 1, matrix_in4, matrix_out4, &context);
  EXPECT_FLOAT_EQ(matrix_out4[0], 4.8);
}

TEST(OpenCL, ColwiseMax_GPU) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  float *matrix_in = new float[10000*10000];
  float *matrix_out = new float[10000];
  caffe2::math::ColwiseMax(10000, 10000, matrix_in, matrix_out, &context);
  delete[] matrix_in;
  delete[] matrix_out;
}

TEST(OpenCL, ColwiseMax_CPU) {
  DeviceOption option;
  caffe2::CPUContext context(option);

  float *matrix_in = new float[10000*10000];
  float *matrix_out = new float[10000];
  caffe2::math::ColwiseMax(10000, 10000, matrix_in, matrix_out, &context);
  delete[] matrix_in;
  delete[] matrix_out;
}

TEST(OpenCL, Dot) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  //Test 1: 1 row, 5 columns
  float matrix_in1[] = {1.0, -2.0, 3.0, 4.0, -1.0};
  float matrix_in2[] = {-1.0, 4.0, 3.0, -2.0, 1.0};
  float matrix_out[1];
  caffe2::math::Dot(5, matrix_in1, matrix_in2, matrix_out, &context);
  EXPECT_FLOAT_EQ(matrix_out[0], -9.0);

}

TEST(OpenCL, Sum) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  //Test 1: 1 row, 5 columns
  float matrix_in3[] = {1.0, -2.0, 3.0, 4.0, -1.0};
  float matrix_out3[1];
  caffe2::math::Sum(5, matrix_in3, matrix_out3, &context);
  EXPECT_FLOAT_EQ(matrix_out3[0], 5.0);

}

TEST(OpenCL, Scale) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  //Test 1: 1 row, 5 columns
  float matrix_in1[] = {1.0, -2.0, 3.0, 4.0, -1.0};
  float matrix_out1[5]= {0.0};
  float result1[] = {2.0, -4.0, 6.0, 8.0, -2.0};
  caffe2::math::Scale(5, 2.0 , matrix_in1, matrix_out1, &context);
  EXPECT_FLOAT_EQ(matrix_out1[0], result1[0]);
  EXPECT_FLOAT_EQ(matrix_out1[1], result1[1]);
  EXPECT_FLOAT_EQ(matrix_out1[2], result1[2]);
  EXPECT_FLOAT_EQ(matrix_out1[3], result1[3]);
  EXPECT_FLOAT_EQ(matrix_out1[4], result1[4]);

  //Test 2: 3 rows, 3 columns
  float matrix_in2[] = {
        1.1,  2.1,  3.1,
       17.2,  8.2,  1.2,
        5.3, 22.3,  4.3
  };

  float matrix_out2[9]= {0.0};
  float matrix_result2[] = {
        2.2,  4.2,  6.2,
       34.4,  16.4,  2.4,
        10.6, 44.6,  8.6
  };
  caffe2::math::Scale(9, 2.0 , matrix_in2, matrix_out2, &context);
  EXPECT_FLOAT_EQ(matrix_out2[0], matrix_result2[0]);
  EXPECT_FLOAT_EQ(matrix_out2[1], matrix_result2[1]);
  EXPECT_FLOAT_EQ(matrix_out2[2], matrix_result2[2]);
  EXPECT_FLOAT_EQ(matrix_out2[3], matrix_result2[3]);
  EXPECT_FLOAT_EQ(matrix_out2[4], matrix_result2[4]);
  EXPECT_FLOAT_EQ(matrix_out2[5], matrix_result2[5]);
  EXPECT_FLOAT_EQ(matrix_out2[6], matrix_result2[6]);
  EXPECT_FLOAT_EQ(matrix_out2[7], matrix_result2[7]);
  EXPECT_FLOAT_EQ(matrix_out2[8], matrix_result2[8]);

  //Test 3: 4 rows, 3 columns
  float matrix_in3[] = {
        1.1,  2.1,  3.1,
       17.2,  8.2,  1.2,
        5.3, 22.3,  4.3,
      -89.4, 99.4, -9.4
  };

  float matrix_out3[12]= {0.0};
  float matrix_result3[] = {
        2.2,  4.2,  6.2,
       34.4,  16.4,  2.4,
        10.6, 44.6,  8.6,
      -178.8, 198.8, -18.8
  };
  caffe2::math::Scale(12, 2.0 , matrix_in3, matrix_out3, &context);
  EXPECT_FLOAT_EQ(matrix_out3[0], matrix_result3[0]);
  EXPECT_FLOAT_EQ(matrix_out3[1], matrix_result3[1]);
  EXPECT_FLOAT_EQ(matrix_out3[2], matrix_result3[2]);
  EXPECT_FLOAT_EQ(matrix_out3[3], matrix_result3[3]);
  EXPECT_FLOAT_EQ(matrix_out3[4], matrix_result3[4]);
  EXPECT_FLOAT_EQ(matrix_out3[5], matrix_result3[5]);
  EXPECT_FLOAT_EQ(matrix_out3[6], matrix_result3[6]);
  EXPECT_FLOAT_EQ(matrix_out3[7], matrix_result3[7]);
  EXPECT_FLOAT_EQ(matrix_out3[8], matrix_result3[8]);
  EXPECT_FLOAT_EQ(matrix_out3[9], matrix_result3[9]);
  EXPECT_FLOAT_EQ(matrix_out3[10], matrix_result3[10]);
  EXPECT_FLOAT_EQ(matrix_out3[11], matrix_result3[11]);
}

TEST(OpenCL, Transpose1D) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  float matrix_in1[1] = {1};
  float matrix_out1[1] = {-666.6};
  int dims[] = {1};
  int axes[] = {0};

  cl::Buffer cl_x(context.clContext(), CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(matrix_in1), matrix_in1);
  cl::Buffer cl_y(context.clContext(), CL_MEM_WRITE_ONLY, sizeof(matrix_out1));

  caffe2::math::Transpose(1, &dims[0], &axes[0], &cl_x, &cl_y, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out1), &cl_y, &matrix_out1[0]);
  EXPECT_FLOAT_EQ(matrix_out1[0], 1);

  float matrix_in2[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  float matrix_out2[10] = {};
  int dims2[] = {10};
  int axes2[] = {0};

  cl::Buffer cl_x2(context.clContext(), CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(matrix_in2), matrix_in2);
  cl::Buffer cl_y2(context.clContext(), CL_MEM_WRITE_ONLY, sizeof(matrix_out2));

  caffe2::math::Transpose(1, &dims2[0], &axes2[0], &cl_x2, &cl_y2, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out2), &cl_y2, &matrix_out2[0]);

  for (int i = 0; i < 10; i++)
    EXPECT_FLOAT_EQ(matrix_out2[i], matrix_in2[i]);
}

TEST(OpenCL, Transpose2D) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  const int num_axes = 2;

  /*Squared: 2x2*/
  const int rows_count1 = 2;
  const int cols_count1 = 2;
  float matrix_in1[rows_count1][cols_count1] = {            /*2 columns*/
                                               /*2 rows*/   {0, 1},
                                                            {2, 3}
                                   };
  float matrix_out1[rows_count1][cols_count1] = {};
  int dims_x1[num_axes]= {rows_count1, cols_count1};

  /*No axis permutations->no transposition:*/
  int axes1_no_transpose[num_axes]= {0, 1};

  cl::Buffer cl_x1(context.clContext(), CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(matrix_in1), &matrix_in1[0][0]);
  cl::Buffer cl_y1(context.clContext(), CL_MEM_WRITE_ONLY, sizeof(matrix_out1));
  caffe2::math::Transpose(num_axes, &dims_x1[0], &axes1_no_transpose[0], &cl_x1, &cl_y1, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out1), &cl_y1, &matrix_out1[0][0]);

  for (int r = 0; r < rows_count1; r++)
    for (int c = 0; c < cols_count1; c++) {
      EXPECT_FLOAT_EQ(matrix_out1[r][c], matrix_in1[r][c]);
    }

  /*Permutation:*/
  int axes1_transpose[num_axes]= {1, 0};

  caffe2::math::Transpose(num_axes, &dims_x1[0], &axes1_transpose[0], &cl_x1, &cl_y1, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out1), &cl_y1, &matrix_out1[0][0]);

  for (int r = 0; r < rows_count1; r++)
    for (int c = 0; c < cols_count1; c++) {
      EXPECT_FLOAT_EQ(matrix_out1[c][r], matrix_in1[r][c]);
    }

  /*Rect 3x5:*/
  const int rows_count2 = 3;
  const int cols_count2 = 5;
  float matrix_in2[rows_count2][cols_count2] = {           /* 5 columns */
                                         /* 3 rows */    { 0,  1,  2,  3,  4},
                                                         { 5,  6,  7,  8,  9},
                                                         {10, 11, 12, 13, 14}
                                            };
  float matrix_out2[cols_count2][rows_count2] = {};
  int dims_x2[num_axes]= {rows_count2, cols_count2};
  /*Permutation:*/
  int axes2_transpose[num_axes]= {1, 0};

  cl::Buffer cl_x2(context.clContext(), CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(matrix_in2), &matrix_in2[0][0]);
  cl::Buffer cl_y2(context.clContext(), CL_MEM_WRITE_ONLY, sizeof(matrix_out2));
  caffe2::math::Transpose(num_axes, &dims_x2[0], &axes2_transpose[0], &cl_x2, &cl_y2, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out2), &cl_y2, &matrix_out2[0][0]);

  for (int r = 0; r < rows_count2; r++)
    for (int c = 0; c < cols_count2; c++) {
      EXPECT_FLOAT_EQ(matrix_out2[c][r], matrix_in2[r][c]);
    }
}

TEST(OpenCL, Transpose3D) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  const int num_axes = 3;

  /* Squared: 2x2x2 */
  const int layers_count1 = 2;
  const int rows_count1 = 2;
  const int cols_count1 = 2;
  float matrix_in1[layers_count1][rows_count1][cols_count1] = {
                                                          {
                                                                      /*2 columns*/
                                                            /*2 rows*/{0, 1},
                                                                      {2, 3},
                                              /*2 layers*/},
                                                          {
                                                                     /*2 columns*/
                                                            /*2 rows*/{4, 5},
                                                                      {6, 7}
                                                          }
                                                 };
  float matrix_out1[layers_count1][rows_count1][cols_count1] = {};
  int dims_x1[num_axes]= {layers_count1, rows_count1, cols_count1};

  /*No axis permutations->no transposition:*/
  int axes1_no_transpose[num_axes]= {0, 1, 2};

  cl::Buffer cl_x1(context.clContext(), CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(matrix_in1), &matrix_in1[0][0]);
  cl::Buffer cl_y1(context.clContext(), CL_MEM_WRITE_ONLY, sizeof(matrix_out1));
  caffe2::math::Transpose(num_axes, &dims_x1[0], &axes1_no_transpose[0], &cl_x1, &cl_y1, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out1), &cl_y1, &matrix_out1[0][0][0]);

  for (int l = 0; l < layers_count1; l++)
    for (int r = 0; r < rows_count1; r++)
      for (int c = 0; c < cols_count1; c++) {
        EXPECT_FLOAT_EQ(matrix_out1[l][r][c], matrix_in1[l][r][c]);
      }

  /*layer and row index swap:*/
  int axes1_transpose_type1[num_axes]= {1, 0, 2};


  caffe2::math::Transpose(num_axes, &dims_x1[0], &axes1_transpose_type1[0], &cl_x1, &cl_y1, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out1), &cl_y1, &matrix_out1[0][0][0]);

  for (int l = 0; l < layers_count1; l++)
    for (int r = 0; r < rows_count1; r++)
      for (int c = 0; c < cols_count1; c++) {
        EXPECT_FLOAT_EQ(matrix_out1[r][l][c], matrix_in1[l][r][c]);
      }


  /*Rect 3x3x5*/
  const int layers_count2 = 3;
  const int rows_count2 = 4;
  const int cols_count2 = 5;
  float matrix_in2[layers_count2][rows_count2][cols_count2] = {
                                /*3 layers*/{           /*5 cols*/
                                              /*4 rows*/{0, 1, 2, 3, 4},
                                                        {5, 6, 7, 8, 9},
                                                        {10, 11, 12, 13, 14},
                                                        {15, 16, 17, 18, 19}
                                            },
                                            {           /*5 cols*/
                                              /*4 rows*/{20, 21, 22, 23, 24},
                                                        {25, 26, 27, 28, 29},
                                                        {30, 31, 32, 33, 34},
                                                        {35, 36, 37, 38, 39}
                                            },
                                            {           /*5 cols*/
                                              /*4 rows*/{40, 41, 42, 43, 44},
                                                        {45, 46, 47, 48, 49},
                                                        {50, 51, 52, 53, 54},
                                                        {55, 56, 57, 58, 59}
                                            }
                                      };

  /*Indexes of layers become columns, rows become layers and columns become rows:*/
  int axes2_transpose[num_axes]= {2, 0, 1};
  float matrix_out2[cols_count2][layers_count2][rows_count2] = {};
  int dims_x2[num_axes]= {layers_count2, rows_count2, cols_count2};

  cl::Buffer cl_x2(context.clContext(), CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(matrix_in2), &matrix_in2[0][0][0]);
  cl::Buffer cl_y2(context.clContext(), CL_MEM_WRITE_ONLY, sizeof(matrix_out2));
  caffe2::math::Transpose(num_axes, &dims_x2[0], &axes2_transpose[0], &cl_x2, &cl_y2, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out2), &cl_y2, &matrix_out2[0][0][0]);

  for (int l = 0; l < layers_count1; l++)
    for (int r = 0; r < rows_count2; r++)
      for (int c = 0; c < cols_count2; c++) {
        EXPECT_FLOAT_EQ(matrix_out2[c][l][r], matrix_in2[l][r][c]);
      }
}


TEST(OpenCL, Transpose3D_GPU) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  const int num_axes = 3;

  const int cols = 1000;
  const int rows = 100;
  const int layers = 1000;
  int axes_transpose[num_axes]= {2, 0, 1};
  int dims_x[num_axes]= {layers, rows, cols};

  std::unique_ptr<int[]>mat_in(new int[layers * rows * cols]);
  std::unique_ptr<int[]>mat_out(new int[layers * rows * cols]);

  cl::Buffer cl_x1(context.clContext(), CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, layers * rows * cols * sizeof(int), mat_in.get());
  cl::Buffer cl_y1(context.clContext(), CL_MEM_WRITE_ONLY, layers * rows * cols * sizeof(int));
  caffe2::math::Transpose(num_axes, &dims_x[0], &axes_transpose[0], &cl_x1, &cl_y1, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(layers * rows * cols * sizeof(int), &cl_y1, mat_out.get());
}

TEST(OpenCL, Transpose3D_CPU) {
  DeviceOption option;
  caffe2::CPUContext context(option);

  const int num_axes = 3;

  const int cols = 1000;
  const int rows = 100;
  const int layers = 1000;
  int axes_transpose[num_axes]= {2, 0, 1};
  int dims_x[num_axes]= {layers, rows, cols};
  int dims_y[num_axes]= {cols, layers, rows};
  int *mat_in = new int[layers * rows * cols];
  int *mat_out = new int[layers * rows * cols];

  caffe2::math::Transpose(num_axes, &dims_x[0], &axes_transpose[0], mat_in, mat_out, &context);

  delete []mat_in;
  delete []mat_out;
}

TEST(OpenCL, Transpose3D_CHW_2_HWC_and_back) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  const int num_axes = 3;

  /* Squared chw: 3x2x4 */
  const int layers_count = 3;
  const int rows_count = 2;
  const int cols_count = 4;
  float matrix_in[layers_count][rows_count][cols_count] = {
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{1, 2, 3, 4},
                                                                        {5, 6, 7, 8},
                                              /*3 layers*/},
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{1.333, 2.333, 3.333, 4.333},
                                                                        {5.333, 6.333, 7.333, 8.333},
                                                          },
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{1.666, 2.666, 3.666, 4.666},
                                                                        {5.666, 6.666, 7.666, 8.666},
                                                          }
                                                 };
  float matrix_out[rows_count][cols_count][layers_count] = {};
  int dims_x[num_axes]= {layers_count, rows_count, cols_count};

  /*from chw to hwc:*/
  int axes_chw_2_hwc[num_axes]= {1, 2, 0};

  cl::Buffer cl_x(context.clContext(), CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(matrix_in), matrix_in);
  cl::Buffer cl_y(context.clContext(), CL_MEM_WRITE_ONLY, sizeof(matrix_out));
  caffe2::math::Transpose(num_axes, &dims_x[0], &axes_chw_2_hwc[0], &cl_x, &cl_y, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out), &cl_y, &matrix_out[0][0][0]);


  float expected_array1[] = {1, 1.333, 1.666, 2, 2.333, 2.666, 3, 3.333, 3.666, 4, 4.333, 4.666,
                             5, 5.333, 5.666, 6, 6.333, 6.666, 7, 7.333, 7.666, 8, 8.333, 8.666
                            };

  int eix = 0;
  for (int r = 0; r < rows_count; r++)
    for (int c = 0; c < cols_count; c++)
      for (int l = 0; l < layers_count; l++) {
        EXPECT_FLOAT_EQ(matrix_out[r][c][l], expected_array1[eix++]);
      }


  /*and hwc back to chw:*/
  float matrix_out2[layers_count][rows_count][cols_count] = {};
  int dims_y[num_axes]= {rows_count, cols_count, layers_count};
  int axes_hwc_2_chw[num_axes]= {2, 0, 1};

  cl::Buffer cl_y2(context.clContext(), CL_MEM_WRITE_ONLY, sizeof(matrix_out2));
  caffe2::math::Transpose(num_axes, &dims_y[0], &axes_hwc_2_chw[0], &cl_y, &cl_y2, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out2), &cl_y2, &matrix_out2[0][0][0]);

  for (int l = 0; l < layers_count; l++)
    for (int r = 0; r < rows_count; r++)
      for (int c = 0; c < cols_count; c++) {
        EXPECT_FLOAT_EQ(matrix_out2[l][r][c], matrix_in[l][r][c]);
      }
}

TEST(OpenCL, Transpose4D_cl_comparison_with_cpu) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);
  caffe2::CPUContext cpuContext;

  const int num_axes = 4;
  /* nchw: 3x3x2x4 */
  const int n = 3;
  const int c = 3;
  const int h = 2;
  const int w = 4;
  float matrix_in[n][c][h][w] = {
                           /* 3 images */
                                        {
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{1, 2, 3, 4},
                                                                        {5, 6, 7, 8}
                                              /*3 layers*/},
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{1.333, 2.333, 3.333, 4.333},
                                                                        {5.333, 6.333, 7.333, 8.333}
                                                          },
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{1.666, 2.666, 3.666, 4.666},
                                                                        {5.666, 6.666, 7.666, 8.666}
                                                          }
                                        },

                                        {
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{9, 10, 11, 12},
                                                                        {13, 14, 15, 16}
                                              /*3 layers*/},
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{9.333, 10.333, 11.333, 12.333},
                                                                        {13.333, 14.333, 15.333, 16.333}
                                                          },
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{9.666, 10.666, 11.666, 12.666},
                                                                        {13.666, 14.666, 15.666, 16.666}
                                                          }
                                        },

                                        {
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{17, 18, 19, 20},
                                                                        {21, 22, 23, 24},
                                              /*3 layers*/},
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{17.333, 18.333, 19.333, 20.333},
                                                                        {21.333, 22.333, 23.333, 24.333}
                                                          },
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{17.666, 18.666, 19.666, 20.666},
                                                                        {21.666, 22.666, 23.666, 24.666}
                                                          }
                                        }

                                                 };
  int dims_x[num_axes]= {n, c, h, w};
  float matrix_out_cl[n][h][w][c];
  float matrix_out_cpu[n][h][w][c];

  int axes_nchw_2_nhwc[num_axes]= {0, 2, 3, 1};

  cl::Buffer cl_x(context.clContext(), CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(matrix_in), matrix_in);
  cl::Buffer cl_y(context.clContext(), CL_MEM_WRITE_ONLY, sizeof(matrix_out_cl));
  caffe2::math::Transpose(num_axes, &dims_x[0], &axes_nchw_2_nhwc[0], &cl_x, &cl_y, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out_cl), &cl_y, &matrix_out_cl[0][0][0]);

  caffe2::math::Transpose(num_axes, &dims_x[0], &axes_nchw_2_nhwc[0], &matrix_in[0][0][0][0], &matrix_out_cpu[0][0][0][0], &cpuContext);

  for (int ni = 0; ni < n; ni++)
    for (int hi = 0; hi < h; hi++)
      for (int wi = 0; wi < w; wi++)
        for (int ci = 0; ci < c; ci++) {
          EXPECT_FLOAT_EQ(matrix_out_cpu[ni][hi][wi][ci], matrix_out_cl[ni][hi][wi][ci]);
      }

  int axes_nhwc_2_nchw[num_axes]= {0, 3, 1, 2};

  int dims_y[num_axes]= {n, h, w, c};
  float matrix_out_cl2[n][c][h][w];
  float matrix_out_cpu2[n][c][h][w];
  cl::Buffer cl_y2(context.clContext(), CL_MEM_WRITE_ONLY, sizeof(matrix_out_cl2));
  caffe2::math::Transpose(num_axes, &dims_y[0], &axes_nhwc_2_nchw[0], &cl_y, &cl_y2, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out_cl2), &cl_y2, &matrix_out_cl2[0][0][0]);

  caffe2::math::Transpose(num_axes, &dims_y[0], &axes_nhwc_2_nchw[0], &matrix_out_cpu[0][0][0][0], &matrix_out_cpu2[0][0][0][0], &cpuContext);

  for (int ni = 0; ni < n; ni++)
    for (int ci = 0; ci < c; ci++)
      for (int hi = 0; hi < h; hi++)
        for (int wi = 0; wi < w; wi++) {
          EXPECT_FLOAT_EQ(matrix_out_cpu2[ni][ci][hi][wi], matrix_out_cl2[ni][ci][hi][wi]);
      }
}

TEST(OpenCL, Transpose4D_cl_comparison_with_cpu2) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);
  caffe2::CPUContext cpuContext;

  const int num_axes = 4;
  /* nchw: 3x3x2x4 */
  const int n = 3;
  const int c = 3;
  const int h = 2;
  const int w = 4;
  float matrix_in[n][c][h][w] = {
                           /* 3 images */
                                        {
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{1, 2, 3, 4},
                                                                        {5, 6, 7, 8}
                                              /*3 layers*/},
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{1.333, 2.333, 3.333, 4.333},
                                                                        {5.333, 6.333, 7.333, 8.333}
                                                          },
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{1.666, 2.666, 3.666, 4.666},
                                                                        {5.666, 6.666, 7.666, 8.666}
                                                          }
                                        },

                                        {
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{9, 10, 11, 12},
                                                                        {13, 14, 15, 16}
                                              /*3 layers*/},
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{9.333, 10.333, 11.333, 12.333},
                                                                        {13.333, 14.333, 15.333, 16.333}
                                                          },
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{9.666, 10.666, 11.666, 12.666},
                                                                        {13.666, 14.666, 15.666, 16.666}
                                                          }
                                        },

                                        {
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{17, 18, 19, 20},
                                                                        {21, 22, 23, 24},
                                              /*3 layers*/},
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{17.333, 18.333, 19.333, 20.333},
                                                                        {21.333, 22.333, 23.333, 24.333}
                                                          },
                                                          {
                                                                        /*4 columns*/
                                                              /*2 rows*/{17.666, 18.666, 19.666, 20.666},
                                                                        {21.666, 22.666, 23.666, 24.666}
                                                          }
                                        }

                                                 };
  int dims_x[num_axes]= {n, c, h, w};
  float matrix_out_cl[h][c][w][n];
  float matrix_out_cpu[h][c][w][n];

  int axes_perm[num_axes]= {2, 1, 3, 0};

  cl::Buffer cl_x(context.clContext(), CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(matrix_in), matrix_in);
  cl::Buffer cl_y(context.clContext(), CL_MEM_WRITE_ONLY, sizeof(matrix_out_cl));
  caffe2::math::Transpose(num_axes, &dims_x[0], &axes_perm[0], &cl_x, &cl_y, &context);
  context.CopyBytes<OpenCLContext, CPUContext>(sizeof(matrix_out_cl), &cl_y, &matrix_out_cl[0][0][0]);

  caffe2::math::Transpose(num_axes, &dims_x[0], &axes_perm[0], &matrix_in[0][0][0][0], &matrix_out_cpu[0][0][0][0], &cpuContext);

  for (int hi = 0; hi < h; hi++)
    for (int ci = 0; ci < c; ci++)
      for (int wi = 0; wi < w; wi++)
        for (int ni = 0; ni < n; ni++) {
          EXPECT_FLOAT_EQ(matrix_out_cpu[hi][ci][wi][ni], matrix_out_cl[hi][ci][wi][ni]);
      }
}

TEST(OpenCL, Reduce_GPU) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  float X_float[] = {100.001, 100, 5, 2, 1.99, 45.301, 45.3};
  float y_float;
  size_t X_float_size = sizeof(X_float) / sizeof(X_float[0]);

  // Minimum on floats
  caffe2::math::ReduceMin(X_float_size, X_float, &y_float,
      static_cast<Tensor<OpenCLContext> *>(nullptr), &context);
  EXPECT_FLOAT_EQ(X_float[4], y_float);

  // Maximum on floats
  caffe2::math::ReduceMax(X_float_size, X_float, &y_float,
      static_cast<Tensor<OpenCLContext> *>(nullptr), &context);
  EXPECT_FLOAT_EQ(X_float[0], y_float);

  int X_int[] = {-150, 1, INT_MAX, 2, 15348, INT_MIN};
  int y_int;
  size_t X_int_size = sizeof(X_int) / sizeof(X_int[0]);

  // Minimum on ints
  caffe2::math::ReduceMin(X_int_size, X_int, &y_int,
      static_cast<Tensor<OpenCLContext> *>(nullptr), &context);
  EXPECT_EQ(INT_MIN, y_int);

  // Maximum on ints
  caffe2::math::ReduceMax(X_int_size, X_int, &y_int,
      static_cast<Tensor<OpenCLContext> *>(nullptr), &context);
  EXPECT_EQ(INT_MAX, y_int);

  long X_long[] = {5, 3, -5048575943, LONG_MAX, 125, 21, LONG_MIN};
  long y_long;
  // Don't use the '\0' at the end
  size_t X_long_size = sizeof(X_long) / sizeof(X_long[0]);

  // Minimum on longs
  caffe2::math::ReduceMin(X_long_size, X_long, &y_long,
      static_cast<Tensor<OpenCLContext> *>(nullptr), &context);
  EXPECT_EQ(LONG_MIN, y_long);

  // Maximum on longs
  caffe2::math::ReduceMax(X_long_size, X_long, &y_long,
      static_cast<Tensor<OpenCLContext> *>(nullptr), &context);
  EXPECT_EQ(LONG_MAX, y_long);
}

TEST(OpenCL, Elemwise_GPU) {
  DeviceOption option;
  caffe2::OpenCLContext context(option);

  float X_float[][7] = {
    {100.001, 100, 5, 2, 1.99, 45.301, 45.3},
    {1, 5, 5, 2, 0, 45.301, 45.3},
    {100.001, 100, 5, 200, 1.99, 45.301, 45.3},
    {100.001, 100, 5, 2, 1.99, 45.301, FLT_MAX},
  };
  float y_float;
  size_t X_float_size = (sizeof(X_float) / sizeof(X_float[0])) * (sizeof(X_float[0]) / sizeof(X_float[0][0]));

  // Maximum on floats
  caffe2::math::ElemwiseMax(X_float_size, (const float *)X_float,
      static_cast<const float *>(nullptr), &y_float, &context);
  EXPECT_FLOAT_EQ(FLT_MAX, y_float);

  int X_int[][7] = {
    {100, 100, 5, 2, 1, 45, 45},
    {1, 5, 5, 2, 0, 45, 45},
    {100, 100000, 5, 200, 1, 45, 45},
    {100, 100, 5, 2, 1, 45, 0},
  };
  int y_int;
  size_t X_int_size = (sizeof(X_int) / sizeof(X_int[0])) * (sizeof(X_int[0]) / sizeof(X_int[0][0]));

  // Maximum on ints
  caffe2::math::ElemwiseMax(X_int_size, (const int *)X_int,
      static_cast<const int *>(nullptr), &y_int, &context);
  EXPECT_EQ(100000, y_int);

  long X_long[][7] = {
    {100, 100, 5, 2, 1, 45, 45},
    {1, 5, 5, 2, 0, 45, 45},
    {100, 100000, 5, 200, 1, 45, 45},
    {100, 100, 5, LONG_MAX, 1, 45, 0},
  };
  long y_long;
  size_t X_long_size = (sizeof(X_long) / sizeof(X_long[0])) * (sizeof(X_long[0]) / sizeof(X_long[0][0]));

  // Maximum on longs
  caffe2::math::ElemwiseMax(X_long_size, (const long *)X_long,
      static_cast<const long *>(nullptr), &y_long, &context);
  EXPECT_EQ(LONG_MAX, y_long);
}
} // namespace caffe2

