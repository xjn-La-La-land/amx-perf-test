#include "amx_tile.hpp"
#include <stdio.h>

type_t A_mem[M][K];              // A matrix
type_t B_mem[K/KPACK][N][KPACK]; // B matrix
res_type_t C_mem[M][N];          // C matrix

type_t B_mem_orig[K][N];
res_type_t C_mem_expected[M][N];

static int8_t next_int8(int8_t i)
{
	if (i == 127)
		return -128;
	return i + 1;
}

static void init_sources()
{
	int8_t counter = -128;

	memset(C_mem, 0, sizeof(C_mem));

	for (size_t m = 0; m < M; m++) {
		for (size_t k = 0; k < K; k++) {
			A_mem[m][k] = counter;
			counter = next_int8(counter);
		}
	}

	for (size_t k = 0; k < K; k++) {
		for (size_t n = 0; n < N; n++) {
			B_mem_orig[k][n] = counter;
			counter = next_int8(counter);
		}
	}

	for (size_t m = 0; m < M; m++)
		for (size_t n = 0; n < N; n++)
			for (size_t k = 0; k < K; k++)
				C_mem_expected[m][n] += ((int32_t) A_mem[m][k]) * ((int32_t) B_mem_orig[k][n]);
}

static void amx_gemm_int8_test()
{
	// B matrix Re-layout
	for (int k = 0; k < K; ++k)
		for (int n = 0; n < N; ++n)
			B_mem[k / KPACK][n][k % KPACK] = B_mem_orig[k][n];

  // compute C += A*B
	amx_gemm_int8();

  // print the results 
  for (size_t m = 0; m < M; ++m) {
    printf("[");
    for (size_t n = 0; n < N; ++n) {
      printf("%6d", C_mem_expected[m][n]);
    }
    printf("]\n");
    printf("[");
    for (size_t n = 0; n < N; ++n) {
      printf("%6d", C_mem[m][n]);
    }
    printf("]\n");
  }
}


int main()
{
	init_sources();

	amx_gemm_int8_test();

  return 0;
}