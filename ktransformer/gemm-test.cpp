#include "amx.hpp"
#include <getopt.h>

FILE *file;
int MEM_ALIGN = 4096;
#define ROUNDUP(a, b) (((a) + (b) - 1) / (b) * (b))
int num_threads = 1;
double frequency = 2.3e9;
int loop_count = 10;

void test_performance(const int M, const int N, const int K) {
  ggml_bf16_t *A =
      (ggml_bf16_t *)aligned_alloc(MEM_ALIGN, M * K * sizeof(ggml_bf16_t));
  ggml_bf16_t *B =
      (ggml_bf16_t *)aligned_alloc(MEM_ALIGN, K * N * sizeof(ggml_bf16_t));
  ggml_bf16_t *C =
      (ggml_bf16_t *)aligned_alloc(MEM_ALIGN, M * N * sizeof(ggml_bf16_t));

  memset(A, 1, M * K * sizeof(ggml_bf16_t));
  memset(B, 1, K * N * sizeof(ggml_bf16_t));
  memset(C, 1, M * N * sizeof(ggml_bf16_t));

  auto gemm =
      std::make_shared<amx::GemmHandwrittenBF16>(A, B, C, M, N, K, num_threads);
  gemm->pack_input();

  gemm->compute(); // run once to warm up

  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (int i = 0; i < loop_count; i++) {
    gemm->compute();
  }
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  gemm->unpack_output();
  gemm->gemm_free();

  uint64_t mac_count = (uint64_t)M * N * K * loop_count;
  uint64_t ideal_mac_per_cycle =
      512 * num_threads; // 512 MACs per cycle per core for bf16
  uint64_t nanoseconds = (end_time.tv_sec - start_time.tv_sec) * 1e9 +
                         (end_time.tv_nsec - start_time.tv_nsec);
  double elapsed_time = (double)nanoseconds / 1e9;

  double utilization =
      ((double)mac_count / elapsed_time) / ideal_mac_per_cycle / frequency;
  double TOPS = (double)mac_count * 2 / 1e12 / elapsed_time;

  printf("M N K = %5d %5d %5d, Elapsed time = %4.6f s, "
         "Performance = %4.2f TOPS, Utilization = %3.2f%%\n",
         M, N, K, elapsed_time, TOPS, utilization * 100);
  fprintf(file,
          "M N K = %5d %5d %5d, Elapsed time = %4.6f s, "
          "Performance = %4.2f TOPS, Utilization = %3.2f%%\n",
          M, N, K, elapsed_time, TOPS, utilization * 100);

  free(A);
  free(B);
  free(C);
}

int main(int argc, char *argv[]) {
  int opt;
  while ((opt = getopt(argc, argv, "t:f:r:")) != -1) {
    switch (opt) {
    case 't':
      num_threads = atoi(optarg);
      break;
    case 'f':
      frequency = atof(optarg) * 1000;
      break;
    case 'r':
      loop_count = atoi(optarg);
      break;
    default:
      fprintf(stderr, "Usage: %s -t NUM_THREADS -f FREQ -r LOOP\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  printf("Running GEMM test with %d threads!\n", num_threads);

  char file_name[100];
  sprintf(file_name, "./gemm-bf16-%dthreads.txt", num_threads);
  file = fopen(file_name, "a");

  if (file == nullptr) {
    perror("cannot open file");
  } else {
    for (int i = 256; i <= 32768; i += 256) {
      int m = i;
      int n = i;
      int k = ROUNDUP(i, 1792);
      test_performance(m, n, k);
    }
  }
  fclose(file);

  return 0;
}
