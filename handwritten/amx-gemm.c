#include "amx-gemm.h"

// external control variables
extern gemm_config_t gemm_config;

// 3 nested loops with no amx
void cpu_gemm_i8(GEMM_PARAMS_I8) {

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
void amx_gemm_i8_naive(GEMM_PARAMS_I8) {

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
void amx_gemm_i8_l0_tiling(GEMM_PARAMS_I8) {
  assert(M % M_STEP == 0 && N % N_STEP == 0 && K % K_STEP == 0);

  for (int i = 0; i < M; i += M_STEP) {
    for (int j = 0; j < N; j += N_STEP) {
      amx_tile_load_L2C(0, C, i, j, ldc);
      amx_tile_load_L2C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_load_L2C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_load_L2C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);

#pragma unroll 32 // GEMM core loop
      for (int k = 0; k < K; k += K_STEP) {
        amx_tile_load_L2B(6, B, k, j, ldb);            // tileload B0
        amx_tile_load_L1A(4, A, i, k, lda);            // tileload A0
        amx_tile_load_L2B(7, B, k, j + MAX_ROWS, ldb); // tileload B1
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

// l0 tiling with packed B
void amx_gemm_i8_l0_tiling_packedB(GEMM_PARAMS_I8) {
  assert(M % M_STEP == 0 && N % N_STEP == 0 && K % K_STEP == 0);

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



#define CACHELINE 64

// l0 tiling with packed A & B, prefetch options: A/C data => l1d, B => l2
void amx_gemm_i8_l0_tiling_packedAB(GEMM_PARAMS_I8) {
  assert(M % M_STEP == 0 && N % N_STEP == 0 && K % K_STEP == 0);

#if defined(PFETCH_B)
  int8_t *pB_pf = B + N * K; // prefetch address for B
#endif
  for (int i = 0; i < M; i += M_STEP) {
    int8_t *pB = B;                       // packed B block address
#if defined(PFETCH_A)
    int8_t *pA_pf = A + (i + M_STEP) * K; // prefetch address for A
#endif
    for (int j = 0; j < N; j += N_STEP) {
      amx_tile_load_L2C(0, C, i, j, ldc);
      amx_tile_load_L2C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_load_L2C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_load_L2C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);

      int8_t *pA = A + i * K; // packed A block address
#if defined(PFETCH_C)
      int8_t *pC_pf =
          (int8_t *)((j + N_STEP <= N)
                         ? &C[OFFSET2D(i, j + N_STEP, ldc)]
                         : &C[OFFSET2D(i + M_STEP, 0,
                                       ldc)]); // prefetch address for C
      int pfC_cnt = 0;
#endif

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

#if defined(PFETCH_A) // prefetch A
        _mm_prefetch(pA_pf, _MM_HINT_T1);
        pA_pf += CACHELINE;
        _mm_prefetch(pA_pf, _MM_HINT_T1);
        pA_pf += CACHELINE;
#endif
#if defined(PFETCH_B) // prefetch B
        if (likely(pB_pf < B + 2 * N * K)) {
          _mm_prefetch(pB_pf, _MM_HINT_T2);
          pB_pf += CACHELINE;
          _mm_prefetch(pB_pf, _MM_HINT_T2);
          pB_pf += CACHELINE;
        }
#endif
#if defined(PFETCH_C) // prefetch C
        if (likely(pfC_cnt < 4096 / CACHELINE)) { //  prefetch C
          _mm_prefetch(pC_pf, _MM_HINT_T1);
          _mm_prefetch(pC_pf + CACHELINE, _MM_HINT_T1);
          pC_pf += ldc * sizeof(int32_t); // move to next row
          _mm_prefetch(pC_pf, _MM_HINT_T1);
          _mm_prefetch(pC_pf + CACHELINE, _MM_HINT_T1);
          pC_pf += ldc * sizeof(int32_t); // move to next row
          pfC_cnt += 4;
        }
#endif
      }
      amx_tile_store_L1C(0, C, i, j, ldc);
      amx_tile_store_L1C(1, C, i, j + MAX_ROWS, ldc);
      amx_tile_store_L1C(2, C, i + MAX_ROWS, j, ldc);
      amx_tile_store_L1C(3, C, i + MAX_ROWS, j + MAX_ROWS, ldc);
    }
  }
}

// L2 blocking
void amx_gemm_i8_l2_blocking(GEMM_PARAMS_I8) {
  // 在 M 方向不需要分块
  for (int tn = 0; tn < N; tn += TN) {
    for (int tk = 0; tk < K; tk += TK) {
      amx_gemm_i8_l0_tiling(
          &A[OFFSET2D(0, tk, lda)],
          &B[OFFSET2D(tk / KPACK_b8, tn * KPACK_b8, ldb * KPACK_b8)],
          &C[OFFSET2D(0, tn, ldc)], M, MIN(N - tn, TN), MIN(K - tk, TK), lda,
          ldb, ldc);
    }
  }
}

// l2 blocking with packed B
void amx_gemm_i8_l2_blocking_packedB(GEMM_PARAMS_I8) {

  for (int tn = 0; tn < N; tn += TN) {
    for (int tk = 0; tk < K; tk += TK) {
      amx_gemm_i8_l0_tiling_packedB(
          &A[OFFSET2D(0, tk, lda)],
          B + tn * K + tk * MIN(N - tn, TN), // packedB block address
          &C[OFFSET2D(0, tn, ldc)], M, MIN(N - tn, TN), MIN(K - tk, TK), lda,
          ldb, ldc);
    }
  }
}

// l2 blocking with packed A & B
void amx_gemm_i8_l2_blocking_packedAB(GEMM_PARAMS_I8) {

  for (int tn = 0; tn < N; tn += TN) {
    for (int tk = 0; tk < K; tk += TK) {
      amx_gemm_i8_l0_tiling_packedAB(A + tk * M,                        // packedA block address
           B + tn * K + tk * MIN(N - tn, TN), // packedB block address
           &C[OFFSET2D(0, tn, ldc)], 
           M, MIN(N - tn, TN), MIN(K - tk, TK), 
           lda, ldb, ldc);
    }
  }
}


void amx_gemm_i8_single_thread(GEMM_PARAMS_I8) {

  // choose the appropriate L2 blocking function based on data layout
  void (*gemm_compute)(GEMM_PARAMS_I8) = NULL;
  
  if (gemm_config.packA && gemm_config.packB) {
    gemm_compute = amx_gemm_i8_l2_blocking_packedAB;
  } else if (gemm_config.packB) {
    gemm_compute = amx_gemm_i8_l2_blocking_packedB;
  } else {
    gemm_compute = amx_gemm_i8_l2_blocking;
  }

  gemm_compute(A, B, C, M, N, K, lda, ldb, ldc);
}

////////////////////////////////////////////////////////////////////////////////
// multi-threaded AMX GEMM
////////////////////////////////////////////////////////////////////////////////

// Split `num_core` into `core_m x core_n`, and return `core_m`.
// We do our best to balance the workload across each core.
int split_cores(int num_core, int M, int N) {
  int num_block_m = CEIL(M, TM);
  int num_block_n = CEIL(N, TN);

  int num_block_max =
      num_block_m * num_block_n; // max number of blocks per core
  int core_m = 1;
  for (int i = 2; i <= num_core; i++) {
    if (num_core % i == 0) {      // i is a divisor of num_block_max
      int core_m1 = i;            // candidate for core_m
      int core_n1 = num_core / i; // candidate for core_n
      int num_block_max1 =
          CEIL(num_block_m, core_m1) * CEIL(num_block_n, core_n1);
      if (num_block_max1 < num_block_max) {
        num_block_max = num_block_max1;
        core_m = core_m1; // update core_m
      }
    }
  }

  return core_m;
}

// amx_gemm_i8_l2_blocking with OpenMP parallelization
// Divide into `(m_core*n_core)` tasks in both M and N dimensions
void amx_gemm_i8_l2_blocking_omp(GEMM_PARAMS_I8) {
  // printf("We can use %d threads for OpenMP parallelization here.\n",
  // omp_get_max_threads());

  if (gemm_config.omp_parallel == OMP_AUTO) { 
    // Lets have OpenMP split the tasks automatically
    #pragma omp parallel for collapse(2)
    for (int tm = 0; tm < M; tm += TM) {
      for (int tn = 0; tn < N; tn += TN) {
        for (int tk = 0; tk < K; tk += TK) {
          amx_gemm_i8_l0_tiling_packedAB(
              A + tm * K + tk * MIN(M - tm, TM), // packedA block address
              B + tn * K + tk * MIN(N - tn, TN), // packedB block address
              &C[OFFSET2D(tm, tn, ldc)], MIN(M - tm, TM), MIN(N - tn, TN),
              MIN(K - tk, TK), lda, ldb, ldc);
        }
      }
    }
  }

  else if (gemm_config.omp_parallel == OMP_MANUAL) { 
    // We split the tasks manually
    int core_m = split_cores(gemm_config.num_core, M,
                             N); // split num_core into core_m x core_n
    int core_n = gemm_config.num_core / core_m;
    #pragma omp parallel num_threads(gemm_config.num_core)
    {
      int thread_id = omp_get_thread_num();

      // Scheme 1: round robin

      int m_start = thread_id / core_n * TM;
      int n_start = thread_id % core_n * TN;

      for (int tm = m_start; tm < M; tm += core_m * TM) {
        for (int tn = n_start; tn < N; tn += core_n * TN) {
          for (int tk = 0; tk < K; tk += TK) {
            amx_gemm_i8_l0_tiling_packedAB(
                A + tm * K + tk * MIN(M - tm, TM), // packedA block address
                B + tn * K + tk * MIN(N - tn, TN), // packedB block address
                &C[OFFSET2D(tm, tn, ldc)], MIN(M - tm, TM), MIN(N - tn, TN),
                MIN(K - tk, TK), lda, ldb, ldc);
          }
        }
      }

      // Scheme 2: contiguous blocks

      // int m_task_sz = ROUNDUP(M / core_m, TM);
      // int n_task_sz = ROUNDUP(N / core_n, TN);
      // int m_start = (thread_id / core_n) * m_task_sz;
      // int n_start = (thread_id % core_n) * n_task_sz;
      // int m_task_sz1 = MIN(M - m_start, m_task_sz); // the actual task size
      // for this thread int n_task_sz1 = MIN(N - n_start, n_task_sz);

      // if(m_task_sz1 > 0 && n_task_sz1 > 0) { // ensure we have work

      //   for(int tn = n_start; tn < n_start + n_task_sz1; tn += TN) {
      //     for(int tk = 0; tk < K; tk += TK) {
      //       amx_gemm_i8_l0_tiling_prefetchAC(
      //         A + m_start * K + tk * m_task_sz1, // packedA block address
      //         B + tn * K + tk * MIN(N - tn, TN), // packedB block address
      //         &C[OFFSET2D(m_start, tn, ldc)],
      //         m_task_sz1, MIN(N - tn, TN), MIN(K - tk, TK),
      //         lda, ldb, ldc);
      //     }
      //   }
      // }
    }
  }
}

// amx_gemm_i8_l2_blocking_omp on separate numa nodes
void amx_gemm_i8_l2_blocking_numa(GEMM_PARAMS_I8_NUMA) {

  // we divide the tasks on the M dimension
  // and dispatch the tasks to each numa node
  int M_div = M / gemm_config.num_node;
  int core_m = split_cores(NUM_CORE_PER_NODE, M_div, N);
  int core_n = NUM_CORE_PER_NODE / core_m;

#pragma omp parallel num_threads(gemm_config.num_core)
  {
    int thread_id = omp_get_thread_num();
    int node_id = thread_id / NUM_CORE_PER_NODE;
    bind_thread_to_cpu(thread_id);

    // M_div * N * K for each numa node
    int8_t *A = A_nodes[node_id];
    int8_t *B = B_nodes[node_id];
    int32_t *C = C_nodes[node_id];

    // round robin for each core
    int thread_id_in_node = thread_id % NUM_CORE_PER_NODE;
    int m_start = thread_id_in_node / core_n * TM;
    int n_start = thread_id_in_node % core_n * TN;

    for (int tm = m_start; tm < M_div; tm += core_m * TM) {
      for (int tn = n_start; tn < N; tn += core_n * TN) {
        for (int tk = 0; tk < K; tk += TK) {
          amx_gemm_i8_l0_tiling_packedAB(
              A + tm * K + tk * MIN(M_div - tm, TM), // packedA block address
              B + tn * K + tk * MIN(N - tn, TN),     // packedB block address
              &C[OFFSET2D(tm, tn, ldc)], MIN(M_div - tm, TM), MIN(N - tn, TN),
              MIN(K - tk, TK), lda, ldb, ldc);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// data pre-processing functions
////////////////////////////////////////////////////////////////////////////////

void amx_packBtile(int8_t *pB0, int8_t *pB1, int ldb) {
  const int kpack = KPACK_b8;
  for (int k = 0; k < MAX_COLS; k++) {
    for (int n = 0; n < MAX_ROWS; n++) {
      pB1[OFFSET3D(k / kpack, n, k % kpack, MAX_ROWS, kpack)] =
          pB0[OFFSET2D(k, n, ldb)];
    }
  }
}

void amx_packAtile(int8_t *pA0, int8_t *pA1, int lda) {
  for (int m = 0; m < MAX_ROWS; m++) {
    memcpy(pA1, pA0, MIN_STRIDE);
    pA1 += MIN_STRIDE;
    pA0 += lda; // move to next row
  }
}

void amx_packB_data(int8_t *__restrict__ B, int8_t *__restrict__ B_packed,
                    const int N, const int K, int ldb) {

#pragma omp parallel for collapse(2)
  for (int tn = 0; tn < N; tn += TN) {
    for (int tk = 0; tk < K; tk += TK) {
      int8_t *pB0 = &B[OFFSET2D(tk, tn, ldb)];
      int8_t *pB1 =
          B_packed + tn * K + tk * MIN(TN, N - tn); // packed B start address

      for (int n = 0; n < MIN(TN, N - tn); n += N_STEP) {
        for (int k = 0; k < MIN(TK, K - tk); k += K_STEP) {
          // pack 2 tiles of B
          amx_packBtile(&pB0[OFFSET2D(k, n, ldb)], pB1, ldb);
          pB1 += TILE_SIZE;
          amx_packBtile(&pB0[OFFSET2D(k, n + MAX_ROWS, ldb)], pB1, ldb);
          pB1 += TILE_SIZE;
        }
      }
    }
  }
}

void amx_packA_data(int8_t *__restrict__ A, int8_t *__restrict__ A_packed,
                    const int M, const int K, int lda) {

#pragma omp parallel for collapse(2)
  for (int tm = 0; tm < M; tm += TM) {
    for (int tk = 0; tk < K; tk += TK) {
      int8_t *pA0 = &A[OFFSET2D(tm, tk, lda)];
      int8_t *pA1 =
          A_packed + tm * K + tk * MIN(TM, M - tm); // packed A start address

      for (int m = 0; m < MIN(TM, M - tm); m += M_STEP) {
        for (int k = 0; k < MIN(TK, K - tk); k += K_STEP) {
          // pack 2 tiles of A
          amx_packAtile(&pA0[OFFSET2D(m, k, lda)], pA1, lda);
          pA1 += TILE_SIZE;
          amx_packAtile(&pA0[OFFSET2D(m + MAX_ROWS, k, lda)], pA1, lda);
          pA1 += TILE_SIZE;
        }
      }
    }
  }
}

////////////////////////////////////////////////////
// TOP LEVEL AMX GEMM API
////////////////////////////////////////////////////

void amx_init() {
  __tilecfg tile_data = {0};
  // Request permission to linux kernel to run AMX
  if (!set_tiledata_use())
    exit(EXIT_FAILURE);
  init_tile_config(&tile_data);
}

// Allocate memory for packed A, and pack A into it
void *amx_packA_i8(int8_t *__restrict__ A, const int M, const int K) {

  if (gemm_config.use_numa) {
    // We need to divide A into `num_node` blocks on the M dimension
    int M_div = M / gemm_config.num_node;
    int8_t **A_nodes =
        (int8_t **)malloc(gemm_config.num_node * sizeof(int8_t *));
    for (int node_id = 0; node_id < gemm_config.num_node; node_id++) {
      A_nodes[node_id] =
          (int8_t *)numa_alloc_onnode(M_div * K * sizeof(int8_t), node_id);
      int m_start = node_id * M_div;
      amx_packA_data(&A[OFFSET2D(m_start, 0, K)], A_nodes[node_id], M_div, K,
                     K);
    }
    return (void *)A_nodes;
  } else if (gemm_config.packA) {
    void *A_packed = aligned_alloc(MEM_ALIGNMENT, M * K * sizeof(int8_t));
    amx_packA_data(A, A_packed, M, K, K);
    return A_packed;
  }

  perror("AMX GEMM: A packing is not supported without packedA = true.");
  return NULL;
}

// Allocate memory for packed B, and pack B into it
void *amx_packB_i8(int8_t *__restrict__ B, const int N, const int K) {

  if (gemm_config.use_numa) {
    int8_t **B_nodes =
        (int8_t **)malloc(gemm_config.num_node * sizeof(int8_t *));
    // We make `num_node` cpyies of B, one for each numa node
    for (int node_id = 0; node_id < gemm_config.num_node; node_id++) {
      B_nodes[node_id] =
          (int8_t *)numa_alloc_onnode(N * K * sizeof(int8_t), node_id);
      amx_packB_data(B, B_nodes[node_id], N, K, N);
    }
    return (void *)B_nodes;
  } else if (gemm_config.packB) {
    void *B_packed = aligned_alloc(MEM_ALIGNMENT, N * K * sizeof(int8_t));
    amx_packB_data(B, B_packed, N, K, N);
    return B_packed;
  }

  perror("AMX GEMM: B packing is not supported without packedB = true.");
  return NULL;
}

// re-allocate memory for C, and copy the original C into it
void *amx_reallocC_i8(int32_t *__restrict__ C, const int M, const int N) {

  if (gemm_config.use_numa) {
    // We need to divide C into `num_node` blocks on the M dimension
    int M_div = M / gemm_config.num_node;
    int32_t **C_nodes =
        (int32_t **)malloc(gemm_config.num_node * sizeof(int32_t *));
    for (int node_id = 0; node_id < gemm_config.num_node; node_id++) {
      int m_start = node_id * M_div;
      C_nodes[node_id] =
          (int32_t *)numa_alloc_onnode(M_div * N * sizeof(int32_t), node_id);
      memcpy(C_nodes[node_id], &C[OFFSET2D(m_start, 0, N)],
             M_div * N * sizeof(int32_t));
    }
    return (void *)C_nodes;
  }

  perror(
      "AMX GEMM: C reallocation is not supported without NUMA configuration.");
  return NULL;
}

// Copy the packed C data into the original C
void amx_copyC_i8(int32_t *__restrict__ C, void *C1, const int M, const int N) {
  int M_div = M / gemm_config.num_node;
  int32_t **C_nodes = (int32_t **)C1; // cast to int32_t pointer array
  for (int node_id = 0; node_id < gemm_config.num_node; node_id++) {
    int m_start = node_id * M_div;
    memcpy(&C[OFFSET2D(m_start, 0, N)], C_nodes[node_id],
           M_div * N * sizeof(int32_t));
  }
}



// Top level AMX GEMM function for integer 8-bit matrix multiplication
// It will call the appropriate tiling function based on the configuration
void amx_gemm_i8(GEMM_PARAMS) {

  if (gemm_config.use_numa) {
    amx_gemm_i8_l2_blocking_numa(A, B, C, M, N, K, lda, ldb, ldc);
  } else if (gemm_config.num_core > 1) {
    amx_gemm_i8_l2_blocking_omp(A, B, C, M, N, K, lda, ldb, ldc);
  } else {
    amx_gemm_i8_single_thread(A, B, C, M, N, K, lda, ldb, ldc);
  }
}

// Free the allocated memory for packed A
void amx_packA_free(void *A_packed, int M, int K) {
  if (gemm_config.use_numa) {
    int M_div = M / gemm_config.num_node;
    int8_t **A_nodes = (int8_t **)A_packed;
    for (int node_id = 0; node_id < gemm_config.num_node; node_id++) {
      numa_free(A_nodes[node_id], M_div * K * sizeof(int8_t));
    }
  }
  free(A_packed);
}

// Free the allocated memory for packed B
void amx_packB_free(void *B_packed, int N, int K) {
  if (gemm_config.use_numa) {
    int8_t **B_nodes = (int8_t **)B_packed;
    for (int node_id = 0; node_id < gemm_config.num_node; node_id++) {
      numa_free(B_nodes[node_id], N * K * sizeof(int8_t));
    }
  }
  free(B_packed);
}

// Free the re-allocated memory for C
void amx_reallocC_free(void *C, int M, int N) {
  if (gemm_config.use_numa) {
    int M_div = M / gemm_config.num_node;
    int32_t **C_nodes = (int32_t **)C;
    for (int node_id = 0; node_id < gemm_config.num_node; node_id++) {
      numa_free(C_nodes[node_id], M_div * N * sizeof(int32_t));
    }
  }
  free(C);
}
