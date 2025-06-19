#include "amx-gemm.h"
#include <stdint.h>

#ifndef LOOP_COUNT
#define LOOP_COUNT 10
#endif
#define LINE "------------------------------------------------------------\n"

// core configuations
int num_core = 1;
double frequency = 2.3e9; // default frequency in Hz
const int MKL_MEM_ALIGNMENT = 64;
FILE *file;
bool packA, packB; // A/B packed or not

// gemm API to test
void amx_gemm_i8i8i32(GEMM_PARAMS) {
  if (!packA && !packB) {
    // amx_gemm_i8i8i32_l0_tiling(A, B, C, M, N, K, lda, ldb, ldc);
    amx_gemm_i8i8i32_l2_tiling(A, B, C, M, N, K, lda, ldb, ldc);
  } else if (!packA && packB) {
    amx_gemm_i8i8i32_l2_tiling_packedB(A, B, C, M, N, K, lda, ldb, ldc);
  } else if (packA && packB) {
    amx_gemm_i8i8i32_l2_tiling_packedAB(A, B, C, M, N, K, lda, ldb, ldc);
  }
}

void test_correctness(const size_t M, const size_t N, const size_t K,
                      const size_t mem_align) {

  int8_t *_A = (int8_t *)malloc(M * K * sizeof(int8_t) + mem_align);
  int8_t *_B = (int8_t *)malloc(K * N * sizeof(int8_t) + mem_align);
  int32_t *_C_amx = (int32_t *)malloc(M * N * sizeof(int32_t) + mem_align);
  int32_t *_C_cpu = (int32_t *)malloc(M * N * sizeof(int32_t) + mem_align);

  int8_t *A = (int8_t *)(((size_t)_A + mem_align) & ~(mem_align - 1));
  int8_t *B = (int8_t *)(((size_t)_B + mem_align) & ~(mem_align - 1));
  int32_t *C_amx = (int32_t *)(((size_t)_C_amx + mem_align) & ~(mem_align - 1));
  int32_t *C_cpu = (int32_t *)(((size_t)_C_cpu + mem_align) & ~(mem_align - 1));

  for (size_t i = 0; i < M * K; i++)
    A[i] = rand() % 256;
  for (size_t i = 0; i < K * N; i++)
    B[i] = rand() % 256;
  for (size_t i = 0; i < M * N; i++)
    C_amx[i] = C_cpu[i] = 0xffffffff;

  int8_t *_A1 = (int8_t *)malloc(M * K * sizeof(int8_t) + mem_align);
  int8_t *_B1 = (int8_t *)malloc(K * N * sizeof(int8_t) + mem_align);
  int8_t *A1 =
      (int8_t *)(((size_t)_A1 + mem_align) & ~(mem_align - 1)); // packed A
  int8_t *B1 =
      (int8_t *)(((size_t)_B1 + mem_align) & ~(mem_align - 1)); // packed B
  amx_packA_i8i8i32(A, A1, M, K);
  amx_packB_i8i8i32(B, B1, N, K);

  cpu_gemm_i8i8i32(A, B, C_cpu, M, N, K, K, N, N);
  amx_gemm_i8i8i32(A1, B1, C_amx, M, N, K, K, N, N);

  int correct = 0 == memcmp(C_cpu, C_amx, M * N * sizeof(int32_t));
  if (!correct) {
    for (size_t i = 0; i < M * N; i++) {
      if (C_cpu[i] != C_amx[i]) {
        printf("Test Failed: M N K = %5ld %5ld %5ld, "
               "Mismatch at Index %ld: %d != %d\n",
               M, N, K, i, C_cpu[i], C_amx[i]);
        break;
      }
    }
  } else {
    printf("Test passed: M N K = %5ld %5ld %5ld\n", M, N, K);
  }
  free(_A);
  free(_B);
  free(_C_amx);
  free(_C_cpu);
  free(_A1);
  free(_B1);
}

void test_performance(const size_t M, const size_t N, const size_t K,
                      const size_t mem_align, const int num_repeats) {

  int8_t *_A = (int8_t *)malloc(M * K * sizeof(int8_t) + mem_align);
  int8_t *_B = (int8_t *)malloc(K * N * sizeof(int8_t) + mem_align);
  int32_t *_C = (int32_t *)malloc(M * N * sizeof(int32_t) + mem_align);

  int8_t *A = (int8_t *)(((size_t)_A + mem_align) & ~(mem_align - 1));
  int8_t *B = (int8_t *)(((size_t)_B + mem_align) & ~(mem_align - 1));
  int32_t *C = (int32_t *)(((size_t)_C + mem_align) & ~(mem_align - 1));

  memset(A, 1, M * K * sizeof(int8_t));
  memset(B, 1, N * K * sizeof(int8_t));
  memset(C, 1, M * N * sizeof(int32_t));

  int8_t *_A1 = (int8_t *)malloc(M * K * sizeof(int8_t) + mem_align);
  int8_t *_B1 = (int8_t *)malloc(K * N * sizeof(int8_t) + mem_align);
  int8_t *A1 = NULL;
  int8_t *B1 = NULL;
  if (packA) {
    A1 = (int8_t *)(((size_t)_A1 + mem_align) & ~(mem_align - 1)); // packed A
    amx_packA_i8i8i32(A, A1, M, K);
  } else {
    A1 = A;
  }
  if (packB) {
    B1 = (int8_t *)(((size_t)_B1 + mem_align) & ~(mem_align - 1)); // packed B
    amx_packB_i8i8i32(B, B1, N, K);
  } else {
    B1 = B;
  }

  amx_gemm_i8i8i32(A1, B1, C, M, N, K, K, N, N); // warm up

  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (uint32_t i = 0; i < num_repeats; i++)
    amx_gemm_i8i8i32(A1, B1, C, M, N, K, K, N, N);
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  uint64_t mac_count = (uint64_t)M * N * K * num_repeats;
  uint64_t ideal_mac_per_cycle = 1024;

  uint64_t nanoseconds = (end_time.tv_sec - start_time.tv_sec) * 1e9 +
                         (end_time.tv_nsec - start_time.tv_nsec);
  double elapsed_time = (double)nanoseconds / 1e9;

  double utilization =
      ((double)mac_count / elapsed_time) / ideal_mac_per_cycle / frequency;
  double TOPS = (double)mac_count * 2 / 1e12 / elapsed_time;

  printf("M N K = %5ld %5ld %5ld, "
         "Performance = %6.2f TOPS, Utilization = %5.2f%%\n",
         M, N, K, TOPS, utilization * 100);
  // fprintf(file,
  //         "M N K = %5ld %5ld %5ld, Elapsed time = %10.6f s, "
  //         "Performance = %6.2f TOPS, Utilization = %5.2f%%\n",
  //         M, N, K, elapsed_time, TOPS, utilization * 100);

  free(_A);
  free(_B);
  free(_C);
  free(_A1);
  free(_B1);
}

int main(int argc, char *argv[]) {
  int opt;
  while ((opt = getopt(argc, argv, "c:f:")) != -1) {
    switch (opt) {
    case 'c':
      num_core = atoi(optarg);
      break;
    case 'f':
      frequency = atof(optarg) * 1000;
      break;
    default:
      fprintf(stderr, "Usage: %s -c NUM_CORE -f FREQ\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  packA = true;
  packB = true;
  printf(LINE);
  printf("Running GEMM test with %d cores, at %.2f GHz!\n", num_core,
         frequency / 1e9);
  printf("Cache Block Size: TM=%d, TN=%d, TK=%d\n", TM, TN, TK);
  printf("Matrix Layout: A - %s, B - %s\n", packA ? "packed" : "normal",
         packB ? "packed" : "normal");
  printf("Running %d rounds!\n", LOOP_COUNT);
  printf(LINE);

  char file_name[100];
  sprintf(file_name, "./gemm-i8-%dcore.txt", num_core);
  file = fopen(file_name, "a");
  if (file == NULL) {
    perror("cannot open file");
  } else {
    amx_init();

    // test_correctness(1024, 2048, 2048, MKL_MEM_ALIGNMENT);

    // general test
    for (int i = 64; i <= 16384; i += 64) {
      int m = i;
      int n = ROUNDUP(i, 512);
      int k = ROUNDUP(i, 1472);
      // int n = 512;
      // int k = 1472;
      test_performance(m, n, k, MKL_MEM_ALIGNMENT, LOOP_COUNT);
      if (m * k + n * k + 4 * m * n >= 300 * 1024 * 1024) // 200MB
        break;
    }
    // test_performance(32, 32, 1472, MKL_MEM_ALIGNMENT, LOOP_COUNT);

    // printf(LINE "test K!\n");
    // for (int k = 64; k <= 16384; k += 64) {
    //   test_performance(256, 256, k, MKL_MEM_ALIGNMENT, LOOP_COUNT);
    // }
    // printf(LINE "test M!\n");
    // for (int m = 64; m <= 8192; m += 64) {
    //   test_performance(m, 512, 1472, MKL_MEM_ALIGNMENT, LOOP_COUNT);
    // }
    // printf(LINE "test N!\n");
    // for (int n = 512; n <= 16384; n += 512) {
    //   test_performance(2048, n, 1472, MKL_MEM_ALIGNMENT, LOOP_COUNT);
    // }

    // test_performance(256, 512, 1472, MKL_MEM_ALIGNMENT, LOOP_COUNT);
    // test_performance(2048, 512, 14720, MKL_MEM_ALIGNMENT, LOOP_COUNT);
    // test_performance(32, 512, 1472, MKL_MEM_ALIGNMENT, LOOP_COUNT);
  }
  fclose(file);
  return 0;
}
