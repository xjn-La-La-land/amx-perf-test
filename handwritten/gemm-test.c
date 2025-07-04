#include "amx-gemm.h"
#include <stdint.h>

#define LINE "------------------------------------------------------------\n"

gemm_config_t gemm_config = DEFAULT_GEMM_CONFIG;
FILE *file;

void test_correctness(const size_t M, const size_t N, const size_t K) {

  int8_t *A = (int8_t *)aligned_alloc(MEM_ALIGNMENT, M * K * sizeof(int8_t));
  int8_t *B = (int8_t *)aligned_alloc(MEM_ALIGNMENT, K * N * sizeof(int8_t));
  int32_t *C_amx =
      (int32_t *)aligned_alloc(MEM_ALIGNMENT, M * N * sizeof(int32_t));
  int32_t *C_cpu =
      (int32_t *)aligned_alloc(MEM_ALIGNMENT, M * N * sizeof(int32_t));

  for (size_t i = 0; i < M * K; i++)
    A[i] = rand() % 256;
  for (size_t i = 0; i < K * N; i++)
    B[i] = rand() % 256;
  for (size_t i = 0; i < M * N; i++)
    C_amx[i] = C_cpu[i] = 0xffffffff;

  cpu_gemm_i8(A, B, C_cpu, M, N, K, K, N, N);
  amx_gemm_i8(A, B, C_amx, M, N, K, K, N, N);

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
  free(A);
  free(B);
  free(C_amx);
  free(C_cpu);
}

void test_performance(const size_t M, const size_t N, const size_t K) {

  void *A = aligned_alloc(MEM_ALIGNMENT, M * K * sizeof(int8_t));
  void *B = aligned_alloc(MEM_ALIGNMENT, K * N * sizeof(int8_t));
  void *C = aligned_alloc(MEM_ALIGNMENT, M * N * sizeof(int32_t));

  // initialize matrices A, B, C
  memset(A, 1, M * K * sizeof(int8_t));
  memset(B, 1, N * K * sizeof(int8_t));
  memset(C, 1, M * N * sizeof(int32_t));

  void *A1 = NULL;
  if (gemm_config.packA)
    A1 = amx_packA_i8(A, M, K);
  else
    A1 = A; // use original A if not packing
  void *B1 = NULL;
  if (gemm_config.packB)
    B1 = amx_packB_i8(B, N, K);
  else
    B1 = B; // use original B if not packing
  void *C1 = NULL;
  if (gemm_config.use_numa)
    C1 = amx_reallocC_i8(C, M,
                         N); // re-allocate C for NUMA-aware parallelization
  else
    C1 = C; // use original C if not NUMA-aware

  int32_t C_ori = ((int32_t *)C)[0];
  amx_gemm_i8(A1, B1, C1, M, N, K, K, N, N); // warm up

  // if (gemm_config.use_numa) {
  //   amx_copyC_i8(C, C1, M, N); // copy packed C back to original C
  // }
  // for (int i = 0; i < M * N; i++) {
  //   if (((int32_t *)C)[i] != K + C_ori) {
  //     printf("C[%d][%d] = %d, expected %d\n", i / N, i % N, ((int32_t
  //     *)C)[i],
  //            K + C_ori);
  //   }
  // }

  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (uint32_t i = 0; i < gemm_config.loop_count; i++)
    amx_gemm_i8(A1, B1, C1, M, N, K, K, N, N);
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  uint64_t mac_count = (uint64_t)M * N * K * gemm_config.loop_count;
  uint64_t ideal_mac_per_cycle =
      1024 * gemm_config.num_core; // 1024 MACs per cycle per core

  uint64_t nanoseconds = (end_time.tv_sec - start_time.tv_sec) * 1e9 +
                         (end_time.tv_nsec - start_time.tv_nsec);
  double elapsed_time = (double)nanoseconds / 1e9;

  double utilization = ((double)mac_count / elapsed_time) /
                       ideal_mac_per_cycle / gemm_config.frequency;
  double TOPS = (double)mac_count * 2 / 1e12 / elapsed_time;

  printf("M N K = %5ld %5ld %5ld, "
         "Performance = %6.2f TOPS, Utilization = %5.2f%%\n",
         M, N, K, TOPS, utilization * 100);
  // fprintf(file,
  //         "M N K = %5ld %5ld %5ld, Elapsed time = %10.6f s, "
  //         "Performance = %6.2f TOPS, Utilization = %5.2f%%\n",
  //         M, N, K, elapsed_time, TOPS, utilization * 100);

  free(A);
  free(B);
  free(C);
  if (gemm_config.packA)
    amx_packA_free(A1, M, K);
  if (gemm_config.packB)
    amx_packB_free(B1, N, K);
  if (gemm_config.use_numa)
    amx_reallocC_free(C1, M, N);
}

int main(int argc, char *argv[]) {
  int opt;
  while ((opt = getopt(argc, argv, "n:c:f:r:")) != -1) {
    switch (opt) {
    case 'n':
      gemm_config.num_node = atoi(optarg);
      gemm_config.use_numa = true;
      gemm_config.num_core = gemm_config.num_node * NUM_CORE_PER_NODE;
      break;
    case 'c':
      gemm_config.num_core = atoi(optarg);
      break;
    case 'f':
      gemm_config.frequency = atof(optarg) * 1000;
      break;
    case 'r':
      gemm_config.loop_count = atoi(optarg);
      break;
    default:
      fprintf(stderr, "Usage: %s -n NUM_NODE -c NUM_CORE -f FREQ -r LOOP\n",
              argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  // gemm_config.omp_parallel = OMP_AUTO;

  char file_name[64];
  sprintf(file_name, "./gemm-i8-%dcore.txt", gemm_config.num_core);
  file = fopen(file_name, "a");
  if (file == NULL) {
    perror("cannot open file");
  }

  // set ups
  amx_init();
  if (gemm_config.num_core > 1) {
    omp_set_num_threads(gemm_config.num_core);
  }
  if (gemm_config.use_numa) {
    if (numa_available() < 0) {
      fprintf(stderr, "NUMA is not available on this system.\n");
      exit(EXIT_FAILURE);
    }
    assert(gemm_config.num_node > 0 &&
           gemm_config.num_node <= numa_max_node() + 1);
  }

  // print configurations
  printf(LINE);
  printf("Running GEMM test with %d CPU Cores, at %.2f GHz!\n",
         gemm_config.num_core, gemm_config.frequency / 1e9);

  if (gemm_config.use_numa) {
    printf("Using NUMA-aware parallelization with %d NUMA nodes.\n",
           gemm_config.num_node);
  }
  printf("Cache Block Size: TM=%d, TN=%d, TK=%d\n", TM, TN, TK);
  printf("Matrix Layout: A - %s, B - %s\n",
         gemm_config.packA ? "packed" : "normal",
         gemm_config.packB ? "packed" : "normal");
  printf("Running %d rounds!\n", gemm_config.loop_count);
  printf(LINE);

  // test_correctness(1024, 2048, 2048);

  // general test
  for (int i = 512; i <= 32768; i += 512) {
    int m = ROUNDUP(i, TM);
    int n = ROUNDUP(i, TN);
    int k = ROUNDUP(i, TK);
    // int m = 12288;
    // int n = 10240;
    // int k = i;
    test_performance(m, n, k);
    // if (m * k + n * k + 4 * m * n >= 300 * 1024 * 1024) // 300MB L3
    //   break;
  }

  // printf(LINE "test K!\n");
  // for (int k = 64; k <= 16384; k += 64) {
  //   test_performance(256, 256, k);
  // }
  // printf(LINE "test M!\n");
  // for (int m = 64; m <= 8192; m += 64) {
  //   test_performance(m, 512, 1472);
  // }
  // printf(LINE "test N!\n");
  // for (int n = 512; n <= 16384; n += 512) {
  //   test_performance(2048, n, 1472);
  // }

  fclose(file);
  return 0;
}
