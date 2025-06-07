#include "amx.hpp"

FILE *file;
int num_core = 1;

void test_performance(const int M, const int N, const int K,
                      const size_t mem_align, const int num_repeats) {
  ggml_bf16_t *A =
      (ggml_bf16_t *)aligned_alloc(mem_align, M * K * sizeof(ggml_bf16_t));
  ggml_bf16_t *B =
      (ggml_bf16_t *)aligned_alloc(mem_align, K * N * sizeof(ggml_bf16_t));
  ggml_bf16_t *C =
      (ggml_bf16_t *)aligned_alloc(mem_align, M * N * sizeof(ggml_bf16_t));

  memset(A, 1, M * K * sizeof(ggml_bf16_t));
  memset(B, 1, K * N * sizeof(ggml_bf16_t));
  memset(C, 1, M * N * sizeof(ggml_bf16_t));

  auto gemm = std::make_shared<amx::GemmHandwrittenBF16>(A, B, C, M, N, K);
  gemm->pack_input();

  gemm->compute(); // run once to warm up

  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (int i = 0; i < num_repeats; i++) {
    gemm->compute();
  }
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  gemm->unpack_output();
  gemm->gemm_free();

  uint64_t mac_count = (uint64_t)M * N * K * num_repeats;
  uint64_t ideal_mac_per_cycle =
      512 * num_core; // 512 MACs per cycle per core for bf16
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
  fprintf(file,
          "M N K = %5d %5d %5d, Elapsed time = %4.6f s, "
          "Performance = %4.2f TOPS, Utilization = %3.2f%%\n",
          M, N, K, elapsed_time, TOPS, utilization * 100);

  free(A);
  free(B);
  free(C);
}

int main(int argc, char *argv[]) {
  if (argc >= 2) {
    num_core = atoi(argv[1]);
  }
  printf("Running GEMM test with %d cores!\n", num_core);

  char file_name[100];
  sprintf(file_name, "./build/gemm-bf16-%dcore.txt", num_core);
  file = fopen(file_name, "a");

  if (file == nullptr) {
    perror("cannot open file");
  } else {
    int mem_align = 4096;
    int num_repeats = 10;
    for (int i = 64; i <= 16384; i += 256) {
      test_performance(i, i, i, mem_align, num_repeats);
    }
  }
  fclose(file);

  return 0;
}
