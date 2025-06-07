#ifndef RVXTM_GEMM_H
#define RVXTM_GEMM_H

#include "rvxtm.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64

#define OFFSET2D(x, y, ld) ((x) * (ld) + (y))
#define OFFSET3D(x, y, z, ld1, ld2) ((x) * (ld1) * (ld2) + (y) * (ld2) + (z))

#define KPACK_b8 4
#define KPACK_b16 2
#define KPACK_b32 1

// Xtm tile load/store L1
#define xtm_tileload_L1A(tmm, arr, row, col, ld)                               \
  TILELOADD(tmm, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int8_t))
#define xtm_tileload_L1B(tmm, arr, row, col, ld)                               \
  TILELOADD(                                                                   \
      tmm,                                                                     \
      &arr[OFFSET2D((row) / KPACK_b8, (col) * KPACK_b8, (ld) * KPACK_b8)],     \
      (ld) * KPACK_b8 * sizeof(int8_t))
#define xtm_tileload_L1C(tmm, arr, row, col, ld)                               \
  TILELOADD(tmm, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int32_t))
#define xtm_tilestore_L1C(tmm, arr, row, col, ld)                              \
  TILESTORED(tmm, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int32_t))

// Xtm tile load L2
#define xtm_tileload_L2A(tmm, arr, row, col, ld)                               \
  TILELOADDT1(tmm, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int8_t))
#define xtm_tileload_L2B(tmm, arr, row, col, ld)                               \
  TILELOADDT1(                                                                 \
      tmm,                                                                     \
      &arr[OFFSET2D((row) / KPACK_b8, (col) * KPACK_b8, (ld) * KPACK_b8)],     \
      (ld) * KPACK_b8 * sizeof(int8_t))
#define xtm_tileload_L2C(tmm, arr, row, col, ld)                               \
  TILELOADDT1(tmm, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int32_t))

#endif // RVXTM_GEMM_H
