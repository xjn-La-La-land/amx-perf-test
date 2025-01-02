## some strategy to optimize `TMUL` operations
(from the manual: Intel 64 and IA-32 Architectures Optimization Reference Manual)
1. Minimizing Tile Loads
* keep the K-dimension loop outside the M_ACC and N_ACC loops(减少 `tile_load` 的调用次数);
* Pre-Loading Innermost Loop Tiles(空间换时间，将acc_m和acc_n中外层循环的tile存储起来，避免重复从内存中加载);
* using 2D accumulator arrays is recommended. (Select dimensions close to square);
2. Software Pipelining of Tile Loads and Stores
* 避免连续的 `tile_load` 和 `tile_store`。将 `tile_load` 和 `tile_store` 插入到循环体中。