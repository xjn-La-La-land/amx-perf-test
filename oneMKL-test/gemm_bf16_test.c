#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
#define LOOP_COUNT 10

int num_core = 1;
const int MKL_MEM_ALIGNMENT = 64;
FILE *file;

void test_performance(int m, int n, int k) {
  MKL_BF16 *A;
  MKL_BF16 *B;
  float *C;
  float alpha, beta;
  int i, r;

  alpha = 1.0;
  beta = 1.0;

  // Allocating memory for matrices aligned on 64-byte boundary for better
  // performance
  A = (MKL_BF16 *)mkl_malloc(m * k * sizeof(MKL_BF16), MKL_MEM_ALIGNMENT);
  B = (MKL_BF16 *)mkl_malloc(k * n * sizeof(MKL_BF16), MKL_MEM_ALIGNMENT);
  C = (float *)mkl_malloc(m * n * sizeof(float), MKL_MEM_ALIGNMENT);
  if (A == NULL || B == NULL || C == NULL) {
    printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    return;
  }

  // Initializing matrix data
  srand(time(NULL));
  for (i = 0; i < (m * k); i++) {
    A[i] = (MKL_BF16)(i);
  }
  for (i = 0; i < (k * n); i++) {
    B[i] = (MKL_BF16)(i);
  }
  for (i = 0; i < (m * n); i++) {
    C[i] = (float)(i);
  }

  // Pack matrix A and B to get better performance
  size_t packed_sz_A =
      cblas_gemm_bf16bf16f32_pack_get_size(CblasAMatrix, m, n, k);
  MKL_BF16 *packed_A = (MKL_BF16 *)mkl_malloc(packed_sz_A, MKL_MEM_ALIGNMENT);
  cblas_gemm_bf16bf16f32_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n,
                              k, A, k, packed_A);
  size_t packed_sz_B =
      cblas_gemm_bf16bf16f32_pack_get_size(CblasBMatrix, m, n, k);
  MKL_BF16 *packed_B = (MKL_BF16 *)mkl_malloc(packed_sz_B, MKL_MEM_ALIGNMENT);
  cblas_gemm_bf16bf16f32_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, m, n,
                              k, B, n, packed_B);

  // Warm up
  cblas_gemm_bf16bf16f32_compute(CblasRowMajor, // 矩阵存储为行优先
                                 CblasPacked,   // A 使用 packed 格式
                                 CblasPacked,   // B 使用 packed 格式
                                 m, n, k,       // 矩阵维度
                                 alpha,         // 标量 α
                                 packed_A, k,   // packed 格式的矩阵A，主轴长度
                                 packed_B, n,   // packed 格式的矩阵B，主轴长度
                                 beta,          // 标量 β
                                 C, n           // 输出矩阵 C 和其主轴长度
  );

  dsecnd(); // dummy call to improve accuracy of timing
  double time_st = dsecnd();
  for (r = 0; r < LOOP_COUNT; r++) {
    // cblas_gemm_bf16bf16f32(CblasRowMajor, // 矩阵存储为行优先
    //                        CblasNoTrans,  // A 不转置
    //                        CblasNoTrans,  // B 不转置
    //                        m, n, k,       // 矩阵维度
    //                        alpha,         // 标量 α
    //                        A, k,          // 矩阵 A 和其主轴长度
    //                        B, n,          // 矩阵 B 和其主轴长度
    //                        beta,          // 标量 β
    //                        C, n           // 输出矩阵 C 和其主轴长度
    // );

    // C := alpha* A* B + beta*C
    cblas_gemm_bf16bf16f32_compute(CblasRowMajor, // 矩阵存储为行优先
                                   CblasPacked,   // A 使用 packed 格式
                                   CblasPacked,   // B 使用 packed 格式
                                   m, n, k,       // 矩阵维度
                                   alpha,         // 标量 α
                                   packed_A, k, // packed 格式的矩阵A，主轴长度
                                   packed_B, n, // packed 格式的矩阵B，主轴长度
                                   beta,        // 标量 β
                                   C, n         // 输出矩阵 C 和其主轴长度
    );
  }
  double time_end = dsecnd();
  double elapsed_time = time_end - time_st;
  uint64_t mac_count = (uint64_t)m * n * k * LOOP_COUNT;
  uint64_t ideal_mac_per_cycle = 512 * num_core;
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
}

int main(int argc, char *argv[]) {
  if (argc >= 2) {
    num_core = atoi(argv[1]);
  }
  printf("Running GEMM test with %d cores!\n", num_core);

  char file_name[100];
  sprintf(file_name, "./build/gemm-bf16-%dcore.txt", num_core);
  file = fopen(file_name, "a");

  if (file == NULL) {
    perror("cannot open file");
  } else {
    for (int i = 32; i <= 16384; i += 256) {
      test_performance(i, i, i);
    }
  }
  fclose(file);

  // test_performance(MSIZE_M, MSIZE_N, MSIZE_K);
  return 0;
}
