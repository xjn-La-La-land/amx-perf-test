#ifndef AMX_TILE_HPP
#define AMX_TILE_HPP

#include <stdint.h>
#include <string.h>

#define M 64            // Number of rows in the A or C matrices
#define K 64            // Number of columns in the A or rows in the B matrices
#define N 32            // Number of columns in the B or C matrices
#define M_ACC 4         // Number of C accumulators spanning the M dimension
#define N_ACC 2         // Number of C accumulators spanning the N dimension
#define TILE_M 16       // Number of rows in an A or C tile
#define TILE_K 64       // Number of columns in an A tile or rows in a B tile
#define TILE_N 16       // Number of columns in a B or C tile

typedef int8_t type_t;     // The type of the data being operated on
typedef int32_t res_type_t; // The data type of the result

#define KPACK (4/sizeof(type_t)) // Vertical K packing into Dword

// tile data structure
template<size_t rows, size_t bytes_cols> class tile {
public:
  friend void tilezero(tile &t) { memset(t.v, 0, sizeof(v)); }
  friend void tileload(tile &t, void *src, size_t bytes_stride) 
  {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t bcol = 0; bcol < bytes_cols; ++bcol) {
        t.v[row][bcol] = static_cast<int8_t *>(src)[row * bytes_stride + bcol];
      }
    }
  }
  friend void tilestore(tile &t, void *dst, size_t bytes_stride) 
  {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t bcol = 0; bcol < bytes_cols; ++bcol) {
        static_cast<int8_t *>(dst)[row * bytes_stride + bcol] = t.v[row][bcol];
      }
    }
  }

  template <class TC, class TA, class TB>
	friend void tdp(TC &tC, TA &tA, TB &tB);

private:
  int8_t v[rows][bytes_cols];
};

template <class TC, class TA, class TB> void tdp(TC &tC, TA &tA, TB &tB) {
  int32_t v;
  for (size_t m = 0; m < TILE_M; m++)
    for (size_t k = 0; k < TILE_K / KPACK; k++)
      for (size_t n = 0; n < TILE_N; n++) {
        memcpy(&v, &tC.v[m][n * sizeof(res_type_t)], sizeof(v));
        v += tA.v[m][k * 4] * tB.v[k][n * 4];
        v += tA.v[m][k * 4 + 1] * tB.v[k][n * 4 + 1];
        v += tA.v[m][k * 4 + 2] * tB.v[k][n * 4 + 2];
        v += tA.v[m][k * 4 + 3] * tB.v[k][n * 4 + 3];
        memcpy(&tC.v[m][n * sizeof(res_type_t)], &v, sizeof(v));
      }
}

extern type_t A_mem[M][K];              // A matrix
extern type_t B_mem[K/KPACK][N][KPACK]; // B matrix
extern res_type_t C_mem[M][N];          // C matrix

void amx_gemm_int8();

#endif