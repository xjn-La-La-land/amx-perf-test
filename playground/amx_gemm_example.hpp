#ifndef AMX_GEMM_EXAMPLE_H
#define AMX_GEMM_EXAMPLE_H

#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>

//Define tile config data structure 
typedef struct __tile_config
{
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __tilecfg;

#define TILE_M 16
#define TILE_N 16
#define TILE_K 64
#define KPACK 4
#define N_ACC 2
#define M_ACC 2
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

#define tA_idx(i)    i // 0~1
#define tB_idx(j)    (2+j) // 2~3
#define tC_idx(i, j) (4+i*N_ACC+j) // 4~7
// tileload, tilestore, tdp 内嵌函数的 tile index 参数必须是字面量，不能是表达式
#define tA_idx_0 0
#define tA_idx_1 1
#define tB_idx_0 2
#define tB_idx_1 3
#define tC_idx_00 4
#define tC_idx_01 5
#define tC_idx_10 6
#define tC_idx_11 7

#define min(a,b) ((a)<(b)? (a):(b))

// matrix data structure
template <typename T>
class matrix {
public:
  int rows;    // 行数
  int cols;    // 列数
  T* data;     // 数据

  int stride = cols * sizeof(T); // 行跨度

  // 构造函数：分配内存
  matrix(int r, int c) : rows(r), cols(c) {
    data = static_cast<T*>(malloc(rows * cols * sizeof(T)));
  }

  matrix(int r, int c, void *data_ptr): rows(r), cols(c), data(static_cast<T*>(data_ptr)) {}

  // operator[]: 返回矩阵第i行的起始地址
  T* operator[](size_t i) {
    return data + i * cols;
  }

};



/* Initialize tile config */
static void init_tile_config (__tilecfg *tileinfo)
{
  tileinfo->palette_id = 1;
  tileinfo->start_row  = 0;

  int i, j;
  for (i = 0; i < M_ACC; ++i) {
    tileinfo->colsb[tA_idx(i)] = TILE_K * sizeof(uint8_t);
    tileinfo->rows[tA_idx(i)]  = TILE_M;
  }
  
  for (j = 0; j < N_ACC; ++j) {
    tileinfo->colsb[tB_idx(j)] = TILE_N * sizeof(uint8_t) * KPACK;
    tileinfo->rows[tB_idx(j)]  = TILE_K / KPACK;
  }

  for (i = 0; i < M_ACC; ++i) {
    for (j = 0; j < N_ACC; ++j) {
      tileinfo->colsb[tC_idx(i,j)] = TILE_N * sizeof(int32_t);
      tileinfo->rows[tC_idx(i,j)]  = TILE_M;
    }
  }

  _tile_loadconfig(tileinfo);
}

static inline void refresh_tile_config_m (__tilecfg *tileinfo, int batch_sz_m)
{
  static int batch_sz_m_r = M_ACC * TILE_M;
  if (batch_sz_m != batch_sz_m_r) {
    batch_sz_m_r = batch_sz_m;
    int i = 0;
    while (batch_sz_m > 0) {
      int tile_sz_m = min(batch_sz_m, TILE_M);
      tileinfo->rows[tA_idx(i)]   = tile_sz_m;
      tileinfo->rows[tC_idx(i,0)] = tile_sz_m;
      tileinfo->rows[tC_idx(i,1)] = tile_sz_m;
      batch_sz_m -= TILE_M;
      i++;
    }
    _tile_loadconfig(tileinfo);
  }
}

static void refresh_tile_config_n (__tilecfg *tileinfo, int batch_sz_n)
{
  static int batch_sz_n_r = N_ACC * TILE_N;
  if (batch_sz_n != batch_sz_n_r) {
    batch_sz_n_r = batch_sz_n;
    int j = 0;
    while (batch_sz_n > 0) {
      int tile_sz_n = min(batch_sz_n, TILE_N);
      tileinfo->colsb[tB_idx(j)]   = tile_sz_n * sizeof(uint8_t);
      tileinfo->colsb[tC_idx(0,j)] = tile_sz_n * sizeof(int32_t);
      tileinfo->colsb[tC_idx(1,j)] = tile_sz_n * sizeof(int32_t);
      batch_sz_n -= TILE_N;
      j++;
    }
    _tile_loadconfig(tileinfo);
  }
}

static void refresh_tile_config_k (__tilecfg *tileinfo, int tile_sz_k)
{
  static int tile_sz_k_r = TILE_K;
  if (tile_sz_k != tile_sz_k_r) {
    tile_sz_k_r = tile_sz_k;
    tileinfo->colsb[tA_idx(0)] = tile_sz_k * sizeof(uint8_t);
    tileinfo->colsb[tA_idx(1)] = tile_sz_k * sizeof(uint8_t);
    tileinfo->rows[tB_idx(0)] = tile_sz_k;
    tileinfo->rows[tB_idx(1)] = tile_sz_k;
    _tile_loadconfig(tileinfo);
  }
}



/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use()
{
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) 
  {
    printf("Fail to do XFEATURE_XTILEDATA\n");
    return false;
  }
  else
  {
    printf("TILE DATA USE SET - OK\n");
    return true;
  }
}


void print_config(__tilecfg *tileinfo) {
  printf("tileinfo->palette_id = %d\n", tileinfo->palette_id);
  printf("tileinfo->start_row = %d\n", tileinfo->start_row);

  for (int i = 0; i < 8; ++i) {
    printf("tmm[%d]: rows = %d, colsb = %d\n", i, tileinfo->rows[i], tileinfo->colsb[i]);
  }
}


matrix<uint8_t> matrixB_relayout(matrix<uint8_t> &mB) {
  matrix<uint8_t> mB_relayout(mB.rows / KPACK, mB.cols * KPACK);
  for (int r = 0; r < mB.rows; ++r) {
    for (int c = 0; c < mB.cols; ++c) {
      mB_relayout[r / KPACK][c * KPACK + r % KPACK] = mB[r][c];
    }
  }
  return mB_relayout;
}

#define A_tile(i,j) (&mA[i*TILE_M][j*TILE_K])
#define B_tile(i,j) (&mB[i*TILE_K/KPACK][j*TILE_N*KPACK])
#define C_tile(i,j) (&mC[i*TILE_M][j*TILE_N])

// 用于记录矩阵分块的索引
typedef struct {
  int matrix_id;
  int i, j;
} tile_index_t;

#define MAT_A 0
#define MAT_B 1
#define MAT_C 2
#define LOOK_AHEAD 100
tile_index_t *acc_seq;
int acc_seq_len;
int current_tile_idx;

void init_acc_seq(int M, int N, int K) {
  int nr_tile_A = (M/TILE_M) * (K/TILE_K) * (N/(N_ACC*TILE_N));
  int nr_tile_B = (K/TILE_K) * (N/TILE_N) * (M/(M_ACC*TILE_M));
  int nr_tile_C = (M/TILE_M) * (N/TILE_N);
  acc_seq_len = nr_tile_A + nr_tile_B + nr_tile_C;
  acc_seq = (tile_index_t*)malloc(acc_seq_len * sizeof(tile_index_t));
  int t = 0;
  for (int j = 0; j < (N/TILE_N); j+=N_ACC) {
    for (int i = 0; i < (M/TILE_M); i+=M_ACC) {
      acc_seq[t++] = {MAT_C, i, j};
      acc_seq[t++] = {MAT_C, i, j+1};
      acc_seq[t++] = {MAT_C, i+1, j};
      acc_seq[t++] = {MAT_C, i+1, j+1};
      for (int k = 0; k < (K/TILE_K); ++k) {
        acc_seq[t++] = {MAT_A, i, k};
        acc_seq[t++] = {MAT_B, k, j};
        acc_seq[t++] = {MAT_A, i+1, k};
        acc_seq[t++] = {MAT_B, k, j+1};
      }
    }
  }
  current_tile_idx = 0;
}

// prefetch the following tiles
void tile_prefetch(int nr_tiles, matrix<uint8_t> &mA, matrix<uint8_t> &mB, matrix<int32_t> &mC) {
  int prefetch_tile_idx = current_tile_idx + LOOK_AHEAD;
  for (int i = 0; i < nr_tiles; ++i) {
    if (prefetch_tile_idx + i >= acc_seq_len) {
      break;
    }
    switch (acc_seq[prefetch_tile_idx + i].matrix_id) {
      case MAT_A:
        _mm_prefetch(A_tile(acc_seq[prefetch_tile_idx + i].i, acc_seq[prefetch_tile_idx + i].j), _MM_HINT_T1);
        break;
      case MAT_B:
        _mm_prefetch(B_tile(acc_seq[prefetch_tile_idx + i].i, acc_seq[prefetch_tile_idx + i].j), _MM_HINT_T1);
        break;
      case MAT_C:
        _mm_prefetch(C_tile(acc_seq[prefetch_tile_idx + i].i, acc_seq[prefetch_tile_idx + i].j), _MM_HINT_T1);
        break;
      default:
        printf("Error: unknown matrix id\n");
        break;
    }
  }
  current_tile_idx += nr_tiles;
}



// compute C += A*B
void amx_gemm_s8u8s32_example(
  void *A, void *B, void *C,
  int M, int N, int K
  )
{
  // initialization
  __tilecfg tile_data = {0};
  init_tile_config(&tile_data);
  matrix <uint8_t> mA(M, K, A);
  matrix <uint8_t> mB(K, N, B);
  matrix <int32_t> mC(M, N, C);
  // init_acc_seq(M, N, K);

  // relayout matrix B
  matrix <uint8_t> mB_re = matrixB_relayout(mB);

  for (int n = 0; n < N; n += N_ACC * TILE_N) {
    int batch_sz_n = min(N_ACC * TILE_N, N-n);
    // refresh_tile_config_n(&tile_data, batch_sz_n);

    for (int m = 0; m < M; m += M_ACC * TILE_M) {
      int batch_sz_m = min(M_ACC * TILE_M, M-m);
      // refresh_tile_config_m(&tile_data, batch_sz_m);

      _tile_loadd (tC_idx_00, &mC[m][n],                   mC.stride);
      _tile_loadd (tC_idx_01, &mC[m][n + TILE_N],          mC.stride);
      _tile_loadd (tC_idx_10, &mC[m + TILE_M][n],          mC.stride);
      _tile_loadd (tC_idx_11, &mC[m + TILE_M][n + TILE_N], mC.stride);
      // tile_prefetch(M_ACC * N_ACC, mA, mB_re, mC);

      // 计算C中的对应分块: 将 M_ACC 个 a_block 与 N_ACC 个 b_block 两两相乘，重复 K/TILE_K 次
      for (int k = 0; k < K; k += TILE_K) {
        // refresh_tile_config_k(&tile_data, min(TILE_K, K-k));

        // print_config(&tile_data);
        _tile_loadd (tB_idx_0, &mB_re[k/KPACK][n*KPACK], mB_re.stride);
        _tile_loadd (tA_idx_0, &mA[m][k], mA.stride);
        _tile_dpbsud(tC_idx_00, tA_idx_0, tB_idx_0);
        if (k >= K-TILE_K) {
          _tile_stored(tC_idx_00, &mC[m][n], mC.stride);
        }

        if(batch_sz_n > TILE_N) {
          _tile_loadd (tB_idx_1, &mB_re[k/KPACK][(n+TILE_N)*KPACK], mB_re.stride);
          _tile_dpbsud(tC_idx_01, tA_idx_0, tB_idx_1);
          if (k >= K-TILE_K) {
            _tile_stored(tC_idx_01, &mC[m][n + TILE_N], mC.stride);
          }
        }

        if(batch_sz_m > TILE_M) {
          _tile_loadd (tA_idx_1, &mA[m + TILE_M][k], mA.stride);
          _tile_dpbsud(tC_idx_10, tA_idx_1, tB_idx_0);
          if (k >= K-TILE_K) {
            _tile_stored(tC_idx_10, &mC[m + TILE_M][n], mC.stride);
            _tile_zero(tC_idx_10);
          }
        }

        if(batch_sz_m > TILE_M && batch_sz_n > TILE_N) {
          _tile_dpbsud(tC_idx_11, tA_idx_1, tB_idx_1);
          if (k >= K-TILE_K) {
            _tile_stored(tC_idx_11, &mC[m + TILE_M][n + TILE_N], mC.stride);
            _tile_zero(tC_idx_11);
          }
        }

        // tile_prefetch(M_ACC + N_ACC, mA, mB_re, mC);
      }
    }
  }

  // Release the tile configuration to return to the init state, 
  // which releases all storage it currently holds
  _tile_release();
  free(acc_seq);
} 



#endif // !AMX_GEMM_EXAMPLE_H
