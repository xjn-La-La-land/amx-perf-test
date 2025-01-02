#include "amx_tile.hpp"

// compute C += A*B
void amx_gemm_int8() 
{
    for (int n = 0; n < N; n += N_ACC * TILE_N) {
        for (int m = 0; m < M; m += M_ACC * TILE_M) {
            // 分配tiles
            tile<TILE_M, TILE_N * sizeof(res_type_t)> tC[M_ACC][N_ACC];
            tile<TILE_M, TILE_K * sizeof(type_t)> tA[M_ACC];
            tile<TILE_K/KPACK, TILE_N*KPACK> tB;

            // 清空tC
            for (int m_acc = 0; m_acc < M_ACC; ++m_acc) {
                for (int n_acc = 0; n_acc < N_ACC; ++n_acc) {
                    tilezero(tC[m_acc][n_acc]);
                }
            }

            // 计算C中的对应分块
            for (int k = 0; k < K; k += TILE_K) {
                for (int n_acc = 0; n_acc < N_ACC; ++n_acc) {
                    tileload(tB, &B_mem[k / KPACK][n + n_acc * TILE_N], N * sizeof(type_t) * KPACK);
                    for (int m_acc = 0; m_acc < M_ACC; ++m_acc) {
                        if (n_acc == 0) { // 使用时再 load，避免连续的 tileload
                            tileload(tA[m_acc], &A_mem[m + m_acc * TILE_M][k], K * sizeof(type_t));
                        }
                        tdp(tC[m_acc][n_acc], tA[m_acc], tB);
                        if (k == K-TILE_K) {
                            int mc = m + m_acc * TILE_M;
                            int nc = n + n_acc * TILE_N;
                            tilestore(tC[m_acc][n_acc], &C_mem[mc][nc], N * sizeof(res_type_t));
                        }
                    }
                }
            }
        }
    }
}