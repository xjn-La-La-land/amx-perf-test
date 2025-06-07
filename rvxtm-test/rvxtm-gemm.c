#include "rvxtm-gemm.h"

// input:
//     A: [M, K] array
//     B: [K/KPACK, N*KPACK] array, where KPACK = (4/sizeof(type_t))
// output:
//     C: [M, N] array
void cpu_gemm_i8i8i32(int8_t *__restrict__ A, int8_t *__restrict__ B,
                      int32_t *__restrict__ C, const int M, const int N,
                      const int K, const int lda, const int ldb,
                      const int ldc) {

  assert(M > 0 && N > 0 && K > 0);
  assert(lda >= K && ldb >= N && ldc >= N);

  const int KPACK = KPACK_b8;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int32_t sum = C[OFFSET2D(i, j, ldc)];
      for (int k = 0; k < K; k++) {
        sum += A[OFFSET2D(i, k, lda)] *
               B[OFFSET3D(k / KPACK, j, k % KPACK, ldb, KPACK)];
      }
      C[OFFSET2D(i, j, N)] = sum;
    }
  }
}

void xtm_gemm_i8i8i32_naive(int8_t *__restrict__ A, int8_t *__restrict__ B,
                            int32_t *__restrict__ C, const int M, const int N,
                            const int K, const int lda, const int ldb,
                            const int ldc);
void amx_gemm_i8i8i32_l0_tiling_2A2B(int8_t *__restrict__ A,
                                     int8_t *__restrict__ B,
                                     int32_t *__restrict__ C, const int M,
                                     const int N, const int K, const int lda,
                                     const int ldb, const int ldc);

void xtm_gemm_i8i8i32(int8_t *__restrict__ A, int8_t *__restrict__ B,
                      int32_t *__restrict__ C, const int M, const int N,
                      const int K, const int lda, const int ldb,
                      const int ldc) {

  assert(M > 0 && N > 0 && K > 0);
  assert(lda >= K && ldb >= N && ldc >= N);

  xtm_gemm_i8i8i32_naive(A, B, C, M, N, K, lda, ldb, ldc);
}

void xtm_gemm_i8i8i32_naive(int8_t *__restrict__ A, int8_t *__restrict__ B,
                            int32_t *__restrict__ C, const int M, const int N,
                            const int K, const int lda, const int ldb,
                            const int ldc) {

  assert(M > 0 && N > 0 && K > 0);
  assert(lda >= K && ldb >= N && ldc >= N);

  for (int i = 0; i < M; i += MAX_ROWS) {
    for (int j = 0; j < N; j += MAX_ROWS) {
      xtm_tileload_L2C(0, C, i, j, ldc);
      for (int k = 0; k < K; k += MAX_COLS) {
        xtm_tileload_L2A(1, A, i, k, lda);
        xtm_tileload_L2B(2, B, k, j, ldb);
        TDPBSSD(0, 1, 2);
      }
      xtm_tilestore_L1C(0, C, i, j, ldc);
    }
  }
}

void xtm_gemm_i8i8i32_l0_tiling_2A2B(int8_t *__restrict__ A,
                                     int8_t *__restrict__ B,
                                     int32_t *__restrict__ C, const int M,
                                     const int N, const int K, const int lda,
                                     const int ldb, const int ldc) {

  assert(M > 0 && N > 0 && K > 0);
  assert(lda >= K && ldb >= N && ldc >= N);

  for (int i = 0; i < M; i += MAX_ROWS * 2) {
    for (int j = 0; j < N; j += MAX_ROWS * 2) {
      xtm_tileload_L2C(0, C, i, j, ldc);
      xtm_tileload_L2C(1, C, i, j + MAX_ROWS, ldc);
      xtm_tileload_L2C(2, C, i + MAX_ROWS, j, ldc);
      xtm_tileload_L2C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
#pragma unroll 32
      for (int k = 0; k < K; k += MAX_COLS) {
        xtm_tileload_L2B(6, B, k, j, ldb);
        xtm_tileload_L2A(4, A, i, k,
                         lda); // should be xtm_tileload_L1A here, but
                               // tileload_L1 not supported yet :(
        xtm_tileload_L2B(7, B, k, j + MAX_ROWS, ldb);
        TDPBSSD(0, 4, 6);
        xtm_tileload_L2A(5, A, i + MAX_ROWS, k, lda);
        TDPBSSD(1, 4, 7);
        TDPBSSD(2, 5, 6);
        TDPBSSD(3, 5, 7);
      }
      xtm_tilestore_L1C(0, C, i, j, ldc);
      xtm_tilestore_L1C(1, C, i, j + MAX_ROWS, ldc);
      xtm_tilestore_L1C(2, C, i + MAX_ROWS, j, ldc);
      xtm_tilestore_L1C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
    }
  }
}

void test_correctness(const int M, const int N, const int K,
                      const size_t mem_align) {

  assert(M > 0 && N > 0 && K > 0);

  int8_t *_A = (int8_t *)malloc((size_t)M * K * sizeof(int8_t) + mem_align);
  int8_t *_B = (int8_t *)malloc((size_t)K * N * sizeof(int8_t) + mem_align);
  int32_t *_C_xtm =
      (int32_t *)malloc((size_t)M * N * sizeof(int32_t) + mem_align);
  int32_t *_C_cpu =
      (int32_t *)malloc((size_t)M * N * sizeof(int32_t) + mem_align);

  int8_t *A = (int8_t *)(((size_t)_A + mem_align) & ~(mem_align - 1));
  int8_t *B = (int8_t *)(((size_t)_B + mem_align) & ~(mem_align - 1));
  int32_t *C_xtm = (int32_t *)(((size_t)_C_xtm + mem_align) & ~(mem_align - 1));
  int32_t *C_cpu = (int32_t *)(((size_t)_C_cpu + mem_align) & ~(mem_align - 1));

  for (int i = 0; i < M * K; i++)
    A[i] = rand() % 256;
  for (int i = 0; i < K * N; i++)
    B[i] = rand() % 256;
  for (int i = 0; i < M * N; i++)
    C_xtm[i] = C_cpu[i] = 0xffffffff;

  cpu_gemm_i8i8i32(A, B, C_cpu, M, N, K, K, N, N);
  xtm_gemm_i8i8i32(A, B, C_xtm, M, N, K, K, N, N);

  int correct = 0 == memcmp(C_cpu, C_xtm, M * N * sizeof(int32_t));
  if (!correct) {
    for (int i = 0; i < M * N; i++) {
      if (C_cpu[i] != C_xtm[i]) {
        printf("Test Failed: M N K = %5d %5d %5d, "
               "Mismatch at Index %d: %d != %d\n",
               M, N, K, i, C_cpu[i], C_xtm[i]);
        break;
      }
    }
  } else {
    printf("Test passed: M N K = %5d %5d %5d\n", M, N, K);
  }
  free(_A);
  free(_B);
  free(_C_xtm);
  free(_C_cpu);
}

void test_performance(const int M, const int N, const int K,
                      const size_t mem_align, const int num_repeats) {

  assert(M > 0 && N > 0 && K > 0);

  int8_t *_A = (int8_t *)malloc((size_t)M * K * sizeof(int8_t) + mem_align);
  int8_t *_B = (int8_t *)malloc((size_t)K * N * sizeof(int8_t) + mem_align);
  int32_t *_C = (int32_t *)malloc((size_t)M * N * sizeof(int32_t) + mem_align);

  int8_t *A = (int8_t *)(((size_t)_A + mem_align) & ~(mem_align - 1));
  int8_t *B = (int8_t *)(((size_t)_B + mem_align) & ~(mem_align - 1));
  int32_t *C = (int32_t *)(((size_t)_C + mem_align) & ~(mem_align - 1));

  memset(A, 1, M * K * sizeof(int8_t));
  memset(B, 1, K * N * sizeof(int8_t));
  memset(C, 1, M * N * sizeof(int32_t));

  xtm_gemm_i8i8i32(A, B, C, M, N, K, K, N, N); // warm up

  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (uint32_t i = 0; i < num_repeats; i++)
    xtm_gemm_i8i8i32(A, B, C, M, N, K, K, N, N);
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  uint64_t mac_count = (uint64_t)M * N * K * num_repeats;
  uint64_t ideal_mac_per_cycle = 1024;
  double frequency = 2.3e9;

  uint64_t nanoseconds = (end_time.tv_sec - start_time.tv_sec) * 1e9 +
                         (end_time.tv_nsec - start_time.tv_nsec);
  double elapsed_time = (double)nanoseconds / 1e9;

  double utilization =
      ((double)mac_count / elapsed_time) / ideal_mac_per_cycle / frequency;
  double TOPS = (double)mac_count * 2 / 1e12 / elapsed_time;

  printf("M N K = %5d %5d %5d, Elapsed time = %4.6f s, "
         "Performance = %4.2f TOPS, Utilization = %3.2f%%\n",
         M, N, K, elapsed_time, TOPS, utilization * 100);

  free(_A);
  free(_B);
  free(_C);
}

int main() {
  // paramters
  size_t mem_align = 4096;
  // const int num_repeats = 10;

  // test correctness
  srand(time(0));
  for (int i = 0; i < 10; i++) {
    const int M_align = 32;
    const int N_align = 32;
    const int K_align = 64;
    int M = (rand() % 1024 + M_align) / M_align * M_align;
    int N = (rand() % 1024 + N_align) / N_align * N_align;
    int K = (rand() % 1024 + K_align) / K_align * K_align;
    test_correctness(M, N, K, mem_align);
  }
  // test performance
  // for (int i = 256; i <= 8192; i *= 2) {
  //   for (int j = 256; j <= 8192; j *= 2) {
  //     test_performance(i, i, j, mem_align, num_repeats);
  //   }
  // }

  return 0;
}
