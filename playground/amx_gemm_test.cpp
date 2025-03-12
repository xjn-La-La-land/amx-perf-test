#include "amx_gemm_example.hpp"
#include <stdio.h>
#include <cassert>
#include <mkl.h>

#ifndef MSIZE_M
#define MSIZE_M 1024
#endif
#ifndef MSIZE_N
#define MSIZE_N 1024
#endif
#ifndef MSIZE_K
#define MSIZE_K 1024
#endif
#define LOOP_COUNT 100

#define M MSIZE_M       // Number of rows in the A or C matrices
#define K MSIZE_K       // Number of columns in the A or rows in the B matrices
#define N MSIZE_N       // Number of columns in the B or C matrices
uint8_t A_mem[M][K];    // A matrix
uint8_t B_mem[K][N];    // B matrix
int32_t C_mem[M][N];    // C matrix

int32_t C_mem_expected[M][N];


static void init_sources()
{
	uint8_t counter = 0;

	for (size_t m = 0; m < M; m++) {
		for (size_t k = 0; k < K; k++) {
			A_mem[m][k] = counter++;
		}
	}

	for (size_t k = 0; k < K; k++) {
		for (size_t n = 0; n < N; n++) {
			B_mem[k][n] = counter++;
		}
	}

}

// check the correctness
static void amx_gemm_int8_test()
{
    for (size_t m = 0; m < M; m++)
		for (size_t n = 0; n < N; n++)
			for (size_t k = 0; k < K; k++)
				C_mem_expected[m][n] += ((int32_t) (int8_t) A_mem[m][k]) * ((int32_t) B_mem[k][n]);
    
    // compute C += A*B
	amx_gemm_s8u8s32_example(A_mem, B_mem, C_mem, M, N, K);

    // check the results 
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            // printf("C_mem_expected[%d][%d] = %d, C_mem[%d][%d] = %d\n", m, n, C_mem_expected[m][n], m, n, C_mem[m][n]);
            assert(C_mem_expected[m][n] == C_mem[m][n]);
        }
    }
    printf("test pass!\n");
}


int main()
{
	init_sources();
	if (!set_tiledata_use())
      exit(-1);

    // Ignore the Time Required for the First Call
    amx_gemm_s8u8s32_example(A_mem, B_mem, C_mem, M, N, K);
    
    dsecnd(); // dummy call to improve accuracy of timing
    double time_st = dsecnd();
    for (int r = 0; r < LOOP_COUNT; ++r) {
        amx_gemm_s8u8s32_example(A_mem, B_mem, C_mem, M, N, K);
    }
    double time_end = dsecnd();
    double time_avg = (time_end - time_st) / LOOP_COUNT;
    double top = (2.0*M*N*K)*1E-12;

    printf("Matrix_size(mkn) %5d %5d %5d Average_time(ms) %.5f tops %.5f\n", M, K, N, (time_avg * 1000), top/time_avg);

  return 0;
}
