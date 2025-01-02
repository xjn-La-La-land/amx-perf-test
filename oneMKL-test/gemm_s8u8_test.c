#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define LOOP_COUNT 1000
#ifndef MSIZE_M
#define MSIZE_M 1000
#endif
#ifndef MSIZE_N
#define MSIZE_N 1000
#endif
#ifndef MSIZE_K
#define MSIZE_K 200
#endif

int main() 
{
    uint8_t *A;
    int8_t  *B;
    int32_t *C;
    int m, n, k;
    float alpha, beta;
    int i, r;

    FILE *file = fopen("./results.txt", "a");
    if (file == NULL) {
        perror("cannot open file");
        return 1;
    }

    m = MSIZE_M; k = MSIZE_K; n = MSIZE_N;
    alpha = 1.0; beta = 1.0;

    // Allocating memory for matrices aligned on 64-byte boundary for better performance
    A = (uint8_t *)mkl_malloc( m*k*sizeof( uint8_t ), 64 );
    B = (int8_t  *)mkl_malloc( k*n*sizeof(  int8_t ), 64 );
    C = (int32_t *)mkl_malloc( m*n*sizeof( int32_t ), 64 );
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
        A[i] = (uint8_t)(rand());
    }
    for (i = 0; i < (k*n); i++) {
        B[i] = (int8_t)(rand());
    }
    for (i = 0; i < (m*n); i++) {
        C[i] = (int32_t)(rand());
    }

    // Ignore the Time Required for the First Call
    cblas_gemm_s8u8s32(
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
        cblas_gemm_s8u8s32(
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
    double top = (2.0*m*n*k)*1E-12;
    
    printf("Matrix_size(mkn) %5d %5d %5d Average_time(ms) %.5f tops %.5f\n", m, k, n, (time_avg * 1000), top/time_avg);
    fprintf(file, "Matrix_size(mkn) %5d %5d %5d Average_time(ms) %.5f tops %.5f\n", m, k, n, (time_avg * 1000), top/time_avg);

    // Deallocating memory
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);


    fclose(file);
    return 0;
}