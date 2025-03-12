#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define LOOP_COUNT 1000

int main() 
{
    MKL_BF16 *A;
    MKL_BF16 *B;
    float *C;
    int m, n, k;
    float alpha, beta;
    int i, r;

    FILE *file = fopen("./build/log.txt", "a");
    if (file == NULL) {
        perror("cannot open file");
        return 1;
    }

    m = MSIZE_M; k = MSIZE_K; n = MSIZE_N;
    alpha = 1.0; beta = 1.0;

    // Allocating memory for matrices aligned on 64-byte boundary for better performance
    A = (MKL_BF16 *)mkl_malloc( m*k*sizeof( MKL_BF16 ), 64 );
    B = (MKL_BF16 *)mkl_malloc( k*n*sizeof( MKL_BF16 ), 64 );
    C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
        printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }

    // Initializing matrix data
    srand(time(NULL));
    for (i = 0; i < (m*k); i++) {
        A[i] = (MKL_BF16)(rand());
    }
    for (i = 0; i < (k*n); i++) {
        B[i] = (MKL_BF16)(rand());
    }
    for (i = 0; i < (m*n); i++) {
        C[i] = (float)(rand());
    }

    // Ignore the Time Required for the First Call
    cblas_gemm_bf16bf16f32(
        CblasRowMajor,  // 矩阵存储为行优先
        CblasNoTrans,   // A 不转置
        CblasNoTrans,   // B 不转置
        m, n, k,        // 矩阵维度
        alpha,          // 标量 α
        A, k,           // 矩阵 A 和其主轴长度
        B, n,           // 矩阵 B 和其主轴长度
        beta,           // 标量 β
        C, n            // 输出矩阵 C 和其主轴长度
    ); // C := alpha*op(A) *op(B) + beta*C

    dsecnd(); // dummy call to improve accuracy of timing
    double time_st = dsecnd();
    for (r = 0; r < LOOP_COUNT; r++) {
        cblas_gemm_bf16bf16f32(
            CblasRowMajor,  // 矩阵存储为行优先
            CblasNoTrans,   // A 不转置
            CblasNoTrans,   // B 不转置
            m, n, k,        // 矩阵维度
            alpha,          // 标量 α
            A, k,           // 矩阵 A 和其主轴长度
            B, n,           // 矩阵 B 和其主轴长度
            beta,           // 标量 β
            C, n            // 输出矩阵 C 和其主轴长度
        );
    }
    double time_end = dsecnd();
    double time_avg = (time_end - time_st) / LOOP_COUNT;
    double gflop = (2.0*m*n*k)*1E-9;
    
    printf("Matrix_size(mkn) %5d %5d %5d Average_time(ms) %.5f gflops %.5f\n", m, k, n, (time_avg * 1000), gflop/time_avg);
    fprintf(file, "Matrix_size(mkn) %5d %5d %5d Average_time(ms) %.5f gflops %.5f\n", m, k, n, (time_avg * 1000), gflop/time_avg);

    // Deallocating memory
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);


    fclose(file);
    return 0;
}
