#include "amx-gemm.h"

// 3 nested loops with no amx
void cpu_gemm_i8i8i32(GEMM_PARAMS) {

  const int KPACK = KPACK_b8;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int32_t sum = C[OFFSET2D(i, j, ldc)];
      for (int k = 0; k < K; k++) {
        sum += A[OFFSET2D(i, k, lda)] * B[OFFSET2D(k, j, ldb)];
      }
      C[OFFSET2D(i, j, N)] = sum;
    }
  }
}

// dummy implementation
void amx_gemm_i8i8i32_naive(GEMM_PARAMS) {

  for (int i = 0; i < M; i += MAX_ROWS) {
    for (int j = 0; j < N; j += MAX_ROWS) {
      amx_tile_load_L2C(0, C, i, j, N);
      for (int k = 0; k < K; k += MAX_COLS) {
        amx_tile_load_L1A(1, A, i, k, K);
        amx_tile_load_L1B(2, B, k, j, N);
        _tile_dpbssd(0, 1, 2);
      }
      amx_tile_store_L1C(0, C, i, j, N);
    }
  }
}

// 2A2B4C tiling
void amx_gemm_i8i8i32_l0_tiling(GEMM_PARAMS) {

  for (int i = 0; i < M; i += M_STEP) {
    for (int j = 0; j < N; j += N_STEP) {
      amx_tile_load_L2C(0, C, i, j, ldc);
      amx_tile_load_L2C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_load_L2C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_load_L2C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
// GEMM core loop
// amx_tile_load_L2B(6, B, 0, j, ldb);                  // tileload B0
#pragma unroll 32
      for (int k = 0; k < K; k += K_STEP) {
        amx_tile_load_L2B(6, B, k, j, ldb);            // tileload B0
        amx_tile_load_L1A(4, A, i, k, lda);            // tileload A0
        amx_tile_load_L2B(7, B, k, j + MAX_ROWS, ldb); // tileload B1
        amx_tile_load_L1A(5, A, i + MAX_ROWS, k, lda); // tileload A1
        _tile_dpbssd(0, 4, 6);                         // tdp A0, B0
        _tile_dpbssd(2, 5, 6);                         // tdp A1, B0
        _tile_dpbssd(1, 4, 7);                         // tdp A0, B1
        // if(likely(k + K_STEP < K)) {
        //     amx_tile_load_L2B(6, B, k + K_STEP, j, ldb); // tileload B0'
        // }
        _tile_dpbssd(3, 5, 7); // tdp A1, B1
      }
      amx_tile_store_L1C(0, C, i, j, ldc);
      amx_tile_store_L1C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_store_L1C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_store_L1C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
    }
  }
}

// L2 tiling
void amx_gemm_i8i8i32_l2_tiling(GEMM_PARAMS) {
  // for (int tm = 0; tm < M; tm += TM) {
  //   for (int tn = 0; tn < N; tn += TN) {
  //     for (int tk = 0; tk < K; tk += TK) {
  //       amx_gemm_i8i8i32_l0_tiling(
  //           &A[OFFSET2D(tm, tk, lda)],
  //           &B[OFFSET2D(tk / KPACK_b8, tn * KPACK_b8, ldb * KPACK_b8)],
  //           &C[OFFSET2D(tm, tn, ldc)], MIN(M - tm, TM), MIN(N - tn, TN),
  //           MIN(K - tk, TK), lda, ldb, ldc);
  //     }
  //   }
  // }

  // 对 M 似乎不需要分块？
  for (int tn = 0; tn < N; tn += TN) {
    for (int tk = 0; tk < K; tk += TK) {
      amx_gemm_i8i8i32_l0_tiling(
          &A[OFFSET2D(0, tk, lda)],
          &B[OFFSET2D(tk / KPACK_b8, tn * KPACK_b8, ldb * KPACK_b8)],
          &C[OFFSET2D(0, tn, ldc)], M, MIN(N - tn, TN), MIN(K - tk, TK), lda,
          ldb, ldc);
    }
  }
}

// l0 tiling with packed B
void amx_gemm_i8i8i32_l0_tiling_packedB(GEMM_PARAMS) {
  for (int i = 0; i < M; i += M_STEP) {
    int8_t *pB = B; // packed B block address
    for (int j = 0; j < N; j += N_STEP) {
      amx_tile_load_L2C(0, C, i, j, ldc);
      amx_tile_load_L2C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_load_L2C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_load_L2C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
// GEMM core loop
#pragma unroll 32
      for (int k = 0; k < K; k += K_STEP) {
        _tile_stream_loadd(6, pB, MIN_STRIDE);
        pB += TILE_SIZE;                    // tileload B0
        amx_tile_load_L1A(4, A, i, k, lda); // tileload A0
        _tile_stream_loadd(7, pB, MIN_STRIDE);
        pB += TILE_SIZE;                               // tileload B1
        amx_tile_load_L1A(5, A, i + MAX_ROWS, k, lda); // tileload A1
        _tile_dpbssd(0, 4, 6);                         // tdp A0, B0
        _tile_dpbssd(2, 5, 6);                         // tdp A1, B0
        _tile_dpbssd(1, 4, 7);                         // tdp A0, B1
        _tile_dpbssd(3, 5, 7);                         // tdp A1, B1
      }
      amx_tile_store_L1C(0, C, i, j, ldc);
      amx_tile_store_L1C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_store_L1C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_store_L1C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
    }
  }
}

// l2 tiling with packed B
void amx_gemm_i8i8i32_l2_tiling_packedB(GEMM_PARAMS) {
  // for (int tm = 0; tm < M; tm += TM) {
  //   for (int tn = 0; tn < N; tn += TN) {
  //     for (int tk = 0; tk < K; tk += TK) {
  //       amx_gemm_i8i8i32_l0_tiling_packedB(
  //           &A[OFFSET2D(tm, tk, lda)],
  //           B + tn * K + tk * MIN(N - tn, TN), // packedB block address
  //           &C[OFFSET2D(tm, tn, ldc)], MIN(M - tm, TM), MIN(N - tn, TN),
  //           MIN(K - tk, TK), lda, ldb, ldc);
  //     }
  //   }
  // }

  for (int tn = 0; tn < N; tn += TN) {
    for (int tk = 0; tk < K; tk += TK) {
      amx_gemm_i8i8i32_l0_tiling_packedB(
          &A[OFFSET2D(0, tk, lda)],
          B + tn * K + tk * MIN(N - tn, TN), // packedB block address
          &C[OFFSET2D(0, tn, ldc)], M, MIN(N - tn, TN), MIN(K - tk, TK), lda,
          ldb, ldc);
    }
  }
}

// l0 tiling with packed A & B
void amx_gemm_i8i8i32_l0_tiling_packedAB(GEMM_PARAMS) {
  for (int i = 0; i < M; i += M_STEP) {
    int8_t *pB = B; // packed B block address
    for (int j = 0; j < N; j += N_STEP) {
      amx_tile_load_L2C(0, C, i, j, ldc);
      amx_tile_load_L2C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_load_L2C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_load_L2C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);

      int8_t *pA = A + i * K; // packed A block address
// GEMM core loop
#pragma unroll 32
      for (int k = 0; k < K; k += K_STEP) {
        _tile_stream_loadd(6, pB, MIN_STRIDE);
        pB += TILE_SIZE; // tileload B0
        _tile_loadd(4, pA, MIN_STRIDE);
        pA += TILE_SIZE; // tileload A0
        _tile_stream_loadd(7, pB, MIN_STRIDE);
        pB += TILE_SIZE; // tileload B1
        _tile_loadd(5, pA, MIN_STRIDE);
        pA += TILE_SIZE;       // tileload A1
        _tile_dpbssd(0, 4, 6); // tdp A0, B0
        _tile_dpbssd(2, 5, 6); // tdp A1, B0
        _tile_dpbssd(1, 4, 7); // tdp A0, B1
        _tile_dpbssd(3, 5, 7); // tdp A1, B1
      }
      amx_tile_store_L1C(0, C, i, j, ldc);
      amx_tile_store_L1C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_store_L1C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_store_L1C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
    }
  }
}

// l2 tiling with packed A & B
void amx_gemm_i8i8i32_l2_tiling_packedAB(GEMM_PARAMS) {
  // for (int tm = 0; tm < M; tm += TM) {
  //   for (int tn = 0; tn < N; tn += TN) {
  //     for (int tk = 0; tk < K; tk += TK) {
  //       amx_gemm_i8i8i32_l0_tiling_packedAB(
  //           A + tm * K + tk * MIN(M - tm, TM), // packedA block address
  //           B + tn * K + tk * MIN(N - tn, TN), // packedB block address
  //           &C[OFFSET2D(tm, tn, ldc)], MIN(M - tm, TM), MIN(N - tn, TN),
  //           MIN(K - tk, TK), lda, ldb, ldc);
  //     }
  //   }
  // }

  // M 方向不分块
  // for (int tn = 0; tn < N; tn += TN) {
  //   for (int tk = 0; tk < K; tk += TK) {
  //     amx_gemm_i8i8i32_l0_tiling_packedAB(
  //         A + tk * M,                        // packedA block address
  //         B + tn * K + tk * MIN(N - tn, TN), // packedB block address
  //         &C[OFFSET2D(0, tn, ldc)], M, MIN(N - tn, TN), MIN(K - tk, TK), lda,
  //         ldb, ldc);
  //   }
  // }

  for (int tn = 0; tn < N; tn += TN) {
    for (int tk = 0; tk < K; tk += TK) {
      amx_gemm_i8i8i32_l0_tiling_prefetchAC(
          A + tk * M,                        // packedA block address
          B + tn * K + tk * MIN(N - tn, TN), // packedB block address
          &C[OFFSET2D(0, tn, ldc)], M, MIN(N - tn, TN), MIN(K - tk, TK), lda,
          ldb, ldc);
    }
  }
}

#define CACHELINE 64
// 2A2B4C tiling with prefetch
void amx_gemm_i8i8i32_l0_tiling_prefetchA(GEMM_PARAMS) {

  for (int i = 0; i < M; i += M_STEP) {
    int8_t *pB = B;                       // packed B block address
    int8_t *pA_pf = A + (i + M_STEP) * K; // prefetch address for A
    for (int j = 0; j < N; j += N_STEP) {
      amx_tile_load_L2C(0, C, i, j, ldc);
      amx_tile_load_L2C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_load_L2C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_load_L2C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);

      int8_t *pA = A + i * K; // packed A block address
// GEMM core loop
#pragma unroll 32
      for (int k = 0; k < K; k += K_STEP) {
        _tile_stream_loadd(6, pB, MIN_STRIDE);
        pB += TILE_SIZE; // tileload B0
        _tile_loadd(4, pA, MIN_STRIDE);
        pA += TILE_SIZE; // tileload A0
        _tile_stream_loadd(7, pB, MIN_STRIDE);
        pB += TILE_SIZE; // tileload B1
        _tile_loadd(5, pA, MIN_STRIDE);
        pA += TILE_SIZE;       // tileload A1
        _tile_dpbssd(0, 4, 6); // tdp A0, B0
        _tile_dpbssd(1, 4, 7); // tdp A0, B1
        _tile_dpbssd(2, 5, 6); // tdp A1, B0
        _tile_dpbssd(3, 5, 7); // tdp A1, B1
        // prefetch A
        _mm_prefetch(pA_pf, _MM_HINT_T1);
        pA_pf += CACHELINE;
        _mm_prefetch(pA_pf, _MM_HINT_T1);
        pA_pf += CACHELINE;
      }
      amx_tile_store_L1C(0, C, i, j, ldc);
      amx_tile_store_L1C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_store_L1C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_store_L1C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
    }
  }
}

void amx_gemm_i8i8i32_l0_tiling_prefetchAC(GEMM_PARAMS) {

  int8_t *pB_pf = B + N * K; // prefetch address for B
  for (int i = 0; i < M; i += M_STEP) {
    int8_t *pB = B;                       // packed B block address
    int8_t *pA_pf = A + (i + M_STEP) * K; // prefetch address for A
    for (int j = 0; j < N; j += N_STEP) {
      amx_tile_load_L2C(0, C, i, j, ldc);
      amx_tile_load_L2C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_load_L2C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_load_L2C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);

      int8_t *pA = A + i * K; // packed A block address
      int8_t *pC_pf =
          (int8_t *)((j + N_STEP <= N)
                         ? &C[OFFSET2D(i, j + N_STEP, ldc)]
                         : &C[OFFSET2D(i + M_STEP, 0,
                                       ldc)]); // prefetch address for C
      int pfC_cnt = 0;

// GEMM core loop
#pragma unroll 32
      for (int k = 0; k < K; k += K_STEP) {
        _tile_stream_loadd(6, pB, MIN_STRIDE);
        pB += TILE_SIZE; // tileload B0
        _tile_loadd(4, pA, MIN_STRIDE);
        pA += TILE_SIZE; // tileload A0
        _tile_stream_loadd(7, pB, MIN_STRIDE);
        pB += TILE_SIZE; // tileload B1
        _tile_loadd(5, pA, MIN_STRIDE);
        pA += TILE_SIZE;       // tileload A1
        _tile_dpbssd(0, 4, 6); // tdp A0, B0
        _tile_dpbssd(1, 4, 7); // tdp A0, B1
        _tile_dpbssd(2, 5, 6); // tdp A1, B0
        _tile_dpbssd(3, 5, 7); // tdp A1, B1
        // prefetch A
        _mm_prefetch(pA_pf, _MM_HINT_T1);
        pA_pf += CACHELINE;
        _mm_prefetch(pA_pf, _MM_HINT_T1);
        pA_pf += CACHELINE;
        // prefetch C
        if (likely(pfC_cnt < 4096 / CACHELINE)) { //  prefetch C
          _mm_prefetch(pC_pf, _MM_HINT_T1);
          _mm_prefetch(pC_pf + CACHELINE, _MM_HINT_T1);
          pC_pf += ldc * sizeof(int32_t); // move to next row
          _mm_prefetch(pC_pf, _MM_HINT_T1);
          _mm_prefetch(pC_pf + CACHELINE, _MM_HINT_T1);
          pC_pf += ldc * sizeof(int32_t); // move to next row
          pfC_cnt += 4;
        }
      }
      amx_tile_store_L1C(0, C, i, j, ldc);
      amx_tile_store_L1C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_store_L1C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_store_L1C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
    }
  }
}

void amx_gemm_i8i8i32_l0_tiling_prefetchABC(GEMM_PARAMS) {

#define DO_PREFETCH_B (pB_pf < B + 2 * N * K)
  int8_t *pB_pf = B + N * K; // prefetch address for B
  for (int i = 0; i < M; i += M_STEP) {
    int8_t *pB = B;                       // packed B block address
    int8_t *pA_pf = A + (i + M_STEP) * K; // prefetch address for A
    for (int j = 0; j < N; j += N_STEP) {
      amx_tile_load_L2C(0, C, i, j, ldc);
      amx_tile_load_L2C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_load_L2C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_load_L2C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);

      int8_t *pA = A + i * K; // packed A block address
      int8_t *pC_pf =
          (int8_t *)((j + N_STEP <= N)
                         ? &C[OFFSET2D(i, j + N_STEP, ldc)]
                         : &C[OFFSET2D(i + M_STEP, 0,
                                       ldc)]); // prefetch address for C
      int pfC_cnt = 0;

// GEMM core loop
#pragma unroll 32
      for (int k = 0; k < K; k += K_STEP) {
        _tile_stream_loadd(6, pB, MIN_STRIDE);
        pB += TILE_SIZE; // tileload B0
        _tile_loadd(4, pA, MIN_STRIDE);
        pA += TILE_SIZE; // tileload A0
        _tile_stream_loadd(7, pB, MIN_STRIDE);
        pB += TILE_SIZE; // tileload B1
        _tile_loadd(5, pA, MIN_STRIDE);
        pA += TILE_SIZE;       // tileload A1
        _tile_dpbssd(0, 4, 6); // tdp A0, B0
        _tile_dpbssd(1, 4, 7); // tdp A0, B1
        _tile_dpbssd(2, 5, 6); // tdp A1, B0
        _tile_dpbssd(3, 5, 7); // tdp A1, B1
        // prefetch A
        _mm_prefetch(pA_pf, _MM_HINT_T1);
        pA_pf += CACHELINE;
        _mm_prefetch(pA_pf, _MM_HINT_T1);
        pA_pf += CACHELINE;
        // prefetch B
        if (likely(pB_pf < B + 2 * N * K)) {
          _mm_prefetch(pB_pf, _MM_HINT_T2);
          pB_pf += CACHELINE;
          _mm_prefetch(pB_pf, _MM_HINT_T2);
          pB_pf += CACHELINE;
        }
        // prefetch C
        if (likely(pfC_cnt < 4096 / CACHELINE)) { //  prefetch C
          _mm_prefetch(pC_pf, _MM_HINT_T1);
          _mm_prefetch(pC_pf + CACHELINE, _MM_HINT_T1);
          pC_pf += ldc * sizeof(int32_t); // move to next row
          _mm_prefetch(pC_pf, _MM_HINT_T1);
          _mm_prefetch(pC_pf + CACHELINE, _MM_HINT_T1);
          pC_pf += ldc * sizeof(int32_t); // move to next row
          pfC_cnt += 4;
        }
      }
      amx_tile_store_L1C(0, C, i, j, ldc);
      amx_tile_store_L1C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_store_L1C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_store_L1C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
    }
  }
}

// misc
void amx_init() {
  __tilecfg tile_data = {0};
  // Request permission to linux kernel to run AMX
  if (!set_tiledata_use())
    exit(-1);
  init_tile_config(&tile_data);
}

void amx_packBtile_i8i8i32(int8_t *pB0, int8_t *pB1, int ldb) {

  const int kpack = KPACK_b8;
  for (int k = 0; k < MAX_COLS; k++) {
    for (int n = 0; n < MAX_ROWS; n++) {
      pB1[OFFSET3D(k / kpack, n, k % kpack, MAX_ROWS, kpack)] =
          pB0[OFFSET2D(k, n, ldb)];
    }
  }
}

void amx_packB_i8i8i32(int8_t *__restrict__ B, int8_t *__restrict__ B_packed,
                       const int N, const int K) {

  assert(N > 0 && K > 0);
  assert(N % N_STEP == 0 && K % K_STEP == 0);

  int8_t *pB1 = B_packed;
  for (int tn = 0; tn < N; tn += TN) {
    for (int tk = 0; tk < K; tk += TK) {
      int8_t *pB0 = &B[OFFSET2D(tk, tn, N)];

      for (int n = 0; n < MIN(TN, N - tn); n += N_STEP) {
        for (int k = 0; k < MIN(TK, K - tk); k += K_STEP) {
          // pack 2 tiles of B
          amx_packBtile_i8i8i32(&pB0[OFFSET2D(k, n, N)], pB1, N);
          pB1 += TILE_SIZE;
          amx_packBtile_i8i8i32(&pB0[OFFSET2D(k, n + MAX_ROWS, N)], pB1, N);
          pB1 += TILE_SIZE;
        }
      }
    }
  }
}

void amx_packAtile_i8i8i32(int8_t *pA0, int8_t *pA1, int lda) {
  for (int m = 0; m < MAX_ROWS; m++) {
    memcpy(pA1, pA0, MIN_STRIDE);
    pA1 += MIN_STRIDE;
    pA0 += lda; // move to next row
  }
}

void amx_packA_i8i8i32(int8_t *__restrict__ A, int8_t *__restrict__ A_packed,
                       const int M, const int K) {

  assert(M > 0 && K > 0);
  assert(M % M_STEP == 0 && K % K_STEP == 0);

  int8_t *pA1 = A_packed;
  for (int tk = 0; tk < K; tk += TK) {
    int8_t *pA0 = &A[OFFSET2D(0, tk, M)];

    for (int m = 0; m < M; m += M_STEP) {
      for (int k = 0; k < MIN(TK, K - tk); k += K_STEP) {
        // pack 2 tiles of A
        amx_packAtile_i8i8i32(&pA0[OFFSET2D(m, k, K)], pA1, K);
        pA1 += TILE_SIZE;
        amx_packAtile_i8i8i32(&pA0[OFFSET2D(m + MAX_ROWS, k, K)], pA1, K);
        pA1 += TILE_SIZE;
      }
    }
  }
}
