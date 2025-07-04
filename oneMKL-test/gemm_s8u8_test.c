#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
#ifndef LOOP_COUNT
#define LOOP_COUNT 10
#endif
#define ROUNDUP(x, y) (((x) + (y) - 1) / (y) * (y))

int num_core = 1;
const int MKL_MEM_ALIGNMENT = 64;
FILE *file;

// 测试 cblas_gemm_s8u8s32 矩阵乘法的性能，A,B矩阵都是行优先存储
void test_performance(int m, int n, int k) {
  MKL_INT8 *A;
  MKL_UINT8 *B;
  MKL_INT32 *C;
  MKL_INT8 oa, ob;
  MKL_INT32 oc;
  float alpha, beta;
  int i, r;

  oa = 0;
  ob = 0;
  oc = 0;
  alpha = 1.0;
  beta = 1.0;

  // Allocating memory for matrices aligned on 64-byte boundary for better
  // performance
  A = (MKL_INT8 *)mkl_malloc(m * k * sizeof(MKL_INT8), MKL_MEM_ALIGNMENT);
  B = (MKL_UINT8 *)mkl_malloc(k * n * sizeof(MKL_UINT8), MKL_MEM_ALIGNMENT);
  C = (MKL_INT32 *)mkl_malloc(m * n * sizeof(MKL_INT32), MKL_MEM_ALIGNMENT);
  if (A == NULL || B == NULL || C == NULL) {
    printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    return;
  }

  // Initializing matrix data
  for (i = 0; i < (m * k); i++) {
    A[i] = (MKL_INT8)(i);
  }
  for (i = 0; i < (k * n); i++) {
    B[i] = (MKL_UINT8)(i);
  }
  for (i = 0; i < (m * n); i++) {
    C[i] = (MKL_INT32)(i);
  }

  // Pack matrix A and B to get better performance
  size_t packed_sz_A = cblas_gemm_s8u8s32_pack_get_size(CblasAMatrix, m, n, k);
  void *packed_A = mkl_malloc(packed_sz_A, MKL_MEM_ALIGNMENT);
  cblas_gemm_s8u8s32_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, A,
                          k, packed_A);
  size_t packed_sz_B = cblas_gemm_s8u8s32_pack_get_size(CblasBMatrix, m, n, k);
  void *packed_B = mkl_malloc(packed_sz_B, MKL_MEM_ALIGNMENT);
  cblas_gemm_s8u8s32_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, m, n, k, B,
                          n, packed_B);

  // Ignore the Time Required for the First Call
  // cblas_gemm_s8u8s32(CblasRowMajor,  // 矩阵存储为行优先
  //                    CblasNoTrans,   // A 不转置
  //                    CblasNoTrans,   // B 不转置
  //                    CblasFixOffset, // offsetc, 决定 oc 的形式
  //                    m, n, k,        // 矩阵维度
  //                    alpha,          // 标量 α
  //                    A, k, oa,       // 矩阵 A 和其主轴长度，偏移量
  //                    B, n, ob,       // 矩阵 B 和其主轴长度，偏移量
  //                    beta,           // 标量 β
  //                    C, n, &oc       // 输出矩阵 C 和其主轴长度，偏移量
  // );                                 // C := alpha*op(A) *op(B) + beta*C

  cblas_gemm_s8u8s32_compute(
      CblasRowMajor,   // 矩阵存储为行优先
      CblasPacked,     // A 使用 packed 格式
      CblasPacked,     // B 使用 packed 格式
      CblasFixOffset,  // offsetc, 决定 oc 的形式
      m, n, k,         // 矩阵维度
      alpha,           // 标量 α
      packed_A, k, oa, // packed 格式的矩阵A，主轴长度，偏移量
      packed_B, n, ob, // packed 格式的矩阵B，主轴长度，偏移量
      beta,            // 标量 β
      C, n, &oc        // 输出矩阵 C 和其主轴长度，偏移量
  );

  dsecnd(); // dummy call to improve accuracy of timing
  double time_st = dsecnd();
  for (r = 0; r < LOOP_COUNT; r++) {
    // cblas_gemm_s8u8s32(CblasRowMajor,  // 矩阵存储为行优先
    //                    CblasNoTrans,   // A 不转置
    //                    CblasNoTrans,   // B 不转置
    //                    CblasFixOffset, // offsetc, 决定 oc 的形式
    //                    m, n, k,        // 矩阵维度
    //                    alpha,          // 标量 α
    //                    A, k, oa,       // 矩阵 A 和其主轴长度，偏移量
    //                    B, n, ob,       // 矩阵 B 和其主轴长度，偏移量
    //                    beta,           // 标量 β
    //                    C, n, &oc       // 输出矩阵 C 和其主轴长度，偏移量
    // );                                 // C := alpha*op(A) *op(B) + beta*C

    // C := alpha*op(A) *op(B) + beta*C
    cblas_gemm_s8u8s32_compute(
        CblasRowMajor,   // 矩阵存储为行优先
        CblasPacked,     // A 使用 packed 格式
        CblasPacked,     // B 使用 packed 格式
        CblasFixOffset,  // offsetc, 决定 oc 的形式
        m, n, k,         // 矩阵维度
        alpha,           // 标量 α
        packed_A, k, oa, // packed 格式的矩阵A，主轴长度，偏移量
        packed_B, n, ob, // packed 格式的矩阵B，主轴长度，偏移量
        beta,            // 标量 β
        C, n, &oc        // 输出矩阵 C 和其主轴长度，偏移量
    );
  }
  double time_end = dsecnd();
  double elapsed_time = time_end - time_st;
  uint64_t mac_count = (uint64_t)m * n * k * LOOP_COUNT;
  uint64_t ideal_mac_per_cycle = 1024 * num_core;
  double frequency = 2.3e9;

  double utilization =
      ((double)mac_count / elapsed_time) / ideal_mac_per_cycle / frequency;
  double TOPS = (double)mac_count * 2 * 1e-12 / elapsed_time;

  printf("M N K = %5d %5d %5d, Elapsed time = %4.6f s, "
         "Performance = %4.2f TOPS, Utilization = %3.2f%%\n",
         m, n, k, elapsed_time, TOPS, utilization * 100);
  fprintf(file,
          "M N K = %5d %5d %5d, Elapsed time = %4.6f s, "
          "Performance = %4.2f TOPS, Utilization = %3.2f%%\n",
          m, n, k, elapsed_time, TOPS, utilization * 100);

  // Deallocating memory
  mkl_free(A);
  mkl_free(B);
  mkl_free(C);

  mkl_free(packed_A);
  mkl_free(packed_B);
}

int main(int argc, char *argv[]) {
  if (argc >= 2) {
    num_core = atoi(argv[1]);
  }
  printf("Running GEMM test with %d cores!\n", num_core);

  char file_name[64];
  sprintf(file_name, "./gemm-i8-%dcore.txt", num_core);
  file = fopen(file_name, "a");
  if (file == NULL) {
    perror("cannot open file");
  } else {
    for (int i = 256; i <= 32768; i += 256) {
      int m = 6144;
      int n = 5120;
      int k = i;
      // int m = ROUNDUP(i, 512);
      // int n = ROUNDUP(i, 512);
      // int k = ROUNDUP(i, 1472);
      test_performance(m, n, k);
    }

    // test_performance(16128, 16128, 16128);
  }
  fclose(file);

  // test_performance(MSIZE_M, MSIZE_N, MSIZE_K);
  return 0;
}
