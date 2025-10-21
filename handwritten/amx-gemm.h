#define _GNU_SOURCE
#include <assert.h>
#include <immintrin.h>
#include <numa.h>
#include <omp.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define MAX_ROWS 16
#define MAX_COLS 64
#define TILE_SIZE (MAX_ROWS * MAX_COLS * sizeof(int8_t))
#define MIN_STRIDE (MAX_COLS * sizeof(int8_t))

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

#define MEM_ALIGNMENT 64

#define OFFSET2D(x, y, ld) ((x) * (ld) + (y))
#define OFFSET3D(x, y, z, ld1, ld2) ((x) * (ld1) * (ld2) + (y) * (ld2) + (z))

#define KPACK_b8 4
#define KPACK_b16 2
#define KPACK_b32 1

// tile blocking
#define M_STEP (MAX_ROWS * 2)
#define N_STEP (MAX_ROWS * 2)
#define K_STEP MAX_COLS

// cache blocking
#ifndef TM
#define TM 512
#endif
#ifndef TN
#define TN 512
#endif
#ifndef TK
#define TK 1472
#endif

#if !defined(likely)
#define likely(cond) __builtin_expect(cond, 1)
#define unlikely(cond) __builtin_expect(cond, 0)
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define ROUNDUP(x, y) (((x) + (y) - 1) / (y) * (y))
#define ROUNDDOWN(x, y) ((x) / (y) * (y))

// GEMM configurations
typedef struct {
  bool use_numa; // whether to use NUMA-aware parallelization
  int num_node;  // number of NUMA nodes
  int num_core;  // number of CPU cores

  bool packA;       // whether to pack matrix A
  bool packB;       // whether to pack matrix B
  int omp_parallel; // OpenMP parallelization ctrl

  double frequency; // CPU frequency in Hz
  int loop_count;   // number of loops for performance test
} gemm_config_t;

// number of CPU cores per numa node (physical core)
#define NUM_CORE_PER_NODE 30

// prefetch strategy
#define PFETCH_A
// #define PFETCH_B
#define PFETCH_C

#define OMP_AUTO 0
#define OMP_MANUAL 1

#define DEFAULT_GEMM_CONFIG                                                    \
  {                                                                            \
      .use_numa = false,                                                       \
      .packA = true,                                                           \
      .packB = true,                                                           \
      .omp_parallel = OMP_MANUAL,                                              \
      .frequency = 2.3e9, /* default frequency 2.3GHz */                       \
      .loop_count = 10,                                                        \
  }

// amx tile load/store L1
#define amx_tile_load_L1A(dst, arr, row, col, ld)                              \
  _tile_loadd(dst, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int8_t))
#define amx_tile_load_L1B(dst, arr, row, col, ld)                              \
  _tile_loadd(                                                                 \
      dst,                                                                     \
      &arr[OFFSET2D((row) / KPACK_b8, (col) * KPACK_b8, (ld) * KPACK_b8)],     \
      (ld) * KPACK_b8 * sizeof(int8_t))
#define amx_tile_load_L1C(dst, arr, row, col, ld)                              \
  _tile_loadd(dst, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int32_t))
#define amx_tile_store_L1C(src, arr, row, col, ld)                             \
  _tile_stored(src, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int32_t))

// amx tile load L2
#define amx_tile_load_L2A(dst, arr, row, col, ld)                              \
  _tile_stream_loadd(dst, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int8_t))
#define amx_tile_load_L2B(dst, arr, row, col, ld)                              \
  _tile_stream_loadd(                                                          \
      dst,                                                                     \
      &arr[OFFSET2D((row) / KPACK_b8, (col) * KPACK_b8, (ld) * KPACK_b8)],     \
      (ld) * KPACK_b8 * sizeof(int8_t))
#define amx_tile_load_L2C(dst, arr, row, col, ld)                              \
  _tile_stream_loadd(dst, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int32_t))

// amx tile prefetch L1
#define amx_tile_prefetch_L1A(arr, row, col, ld)                               \
  _mm_prefetch((const char *)&arr[OFFSET2D(row, col, ld)], _MM_HINT_T0)
#define amx_tile_prefetch_L1B(arr, row, col, ld)                               \
  _mm_prefetch((const char *)&arr[OFFSET2D((row) / KPACK_b8, (col) * KPACK_b8, \
                                           (ld) * KPACK_b8)],                  \
               _MM_HINT_T0)

// amx tile prefetch L2
#define amx_tile_prefetch_L2A(arr, row, col, ld)                               \
  _mm_prefetch((const char *)&arr[OFFSET2D(row, col, ld)], _MM_HINT_T1)
#define amx_tile_prefetch_L2B(arr, row, col, ld)                               \
  _mm_prefetch((const char *)&arr[OFFSET2D((row) / KPACK_b8, (col) * KPACK_b8, \
                                           (ld) * KPACK_b8)],                  \
               _MM_HINT_T1)

// Define tile config data structure
typedef struct __tile_config {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __tilecfg;

/* Initialize tile config */
static void init_tile_config(__tilecfg *tileinfo) {
  int i;
  tileinfo->palette_id = 1;
  tileinfo->start_row = 0;

  for (i = 0; i < 8; ++i) {
    tileinfo->colsb[i] = MAX_COLS;
    tileinfo->rows[i] = MAX_ROWS;
  }

  _tile_loadconfig(tileinfo);
}

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use() {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    printf("Fail to do XFEATURE_XTILEDATA\n");
    return false;
  } else {
    printf("TILE DATA USE SET - OK \n");
    return true;
  }

  return true;
}

// bind the current thread to a specific CPU core
// `cpu_id`: the CPU core ID to bind to
static void bind_thread_to_cpu(int cpu_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);

  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
    perror("sched_setaffinity");
  }
}

// bind the current thread to a specific NUMA node
// `node_id`: the NUMA node ID to bind to
static void bind_thread_to_numa_node(int node_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  for (int i = 0; i < NUM_CORE_PER_NODE; i++) {
    CPU_SET(node_id * NUM_CORE_PER_NODE + i, &cpuset);
  }

  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
    perror("sched_setaffinity");
  }
}

// amx_gemm APIs

// input:
//     A: [M, K] array
//     B: [K/KPACK, N*KPACK] array, where KPACK = (4/sizeof(type_t))
// output:
//     C: [M, N] array
#define GEMM_PARAMS_I8                                                         \
  int8_t *__restrict__ A, int8_t *__restrict__ B, int32_t *__restrict__ C,     \
      const int M, const int N, const int K, const int lda, const int ldb,     \
      const int ldc
void cpu_gemm_i8(GEMM_PARAMS_I8);           // 3 nested loops with no amx
void amx_gemm_i8_naive(GEMM_PARAMS_I8);     // dummy implementation
void amx_gemm_i8_l0_tiling(GEMM_PARAMS_I8); // 2A2B4C tiling
void amx_gemm_i8_l2_tiling(GEMM_PARAMS_I8); // L2 tiling

void amx_gemm_i8_l0_tiling_packedB(
    GEMM_PARAMS_I8); // 2A2B4C tiling with packed B
void amx_gemm_i8_l2_tiling_packedB(GEMM_PARAMS_I8); // L2 tiling with packed B

void amx_gemm_i8_l0_tiling_packedAB(
    GEMM_PARAMS_I8); // 2A2B4C tiling with packed A and B
void amx_gemm_i8_l2_tiling_packedAB(
    GEMM_PARAMS_I8); // L2 tiling with packed A and B

void amx_gemm_i8_l0_tiling_prefetchA(
    GEMM_PARAMS_I8); // 2A2B4C tiling with prefetch A
void amx_gemm_i8_l0_tiling_prefetchAC(
    GEMM_PARAMS_I8); // 2A2B4C tiling with prefetch A & C
void amx_gemm_i8_l0_tiling_prefetchABC(
    GEMM_PARAMS_I8); // 2A2B4C tiling with prefetch A & B & C

void amx_gemm_i8_l2_tiling_omp(
    GEMM_PARAMS_I8); // L2 tiling with OpenMP parallelization

#define GEMM_PARAMS_I8_NUMA                                                    \
  int8_t **__restrict__ A_nodes, int8_t **__restrict__ B_nodes,                \
      int32_t **__restrict__ C_nodes, const int M, const int N, const int K,   \
      const int lda, const int ldb, const int ldc

void amx_gemm_i8_l2_tiling_numa(
    GEMM_PARAMS_I8_NUMA); // L2 tiling with NUMA-aware parallelization

// Top level APIs

#define GEMM_PARAMS                                                            \
  void *__restrict__ A, void *__restrict__ B, void *__restrict__ C,            \
      const int M, const int N, const int K, const int lda, const int ldb,     \
      const int ldc

void amx_init();
void *amx_packA_i8(int8_t *__restrict__ A, const int M, const int K);
void *amx_packB_i8(int8_t *__restrict__ B, const int N, const int K);
void *amx_reallocC_i8(int32_t *__restrict__ C, int M, int N);
void amx_copyC_i8(int32_t *__restrict__ C, void *C1, const int M, const int N);
void amx_gemm_i8(GEMM_PARAMS);
void amx_packA_free(void *A_packed, int M, int K);
void amx_packB_free(void *B_packed, int N, int K);
void amx_reallocC_free(void *C_nodes, int M, int N);
