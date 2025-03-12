## 测试 intel oneMKL 库矩阵乘法算子性能

1. 安装 oneMKL 库

   ```bash
   wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/79153e0f-74d7-45af-b8c2-258941adf58a/intel-onemkl-2025.0.0.940.sh
   sh ./intel-onemkl-2025.0.0.940.sh
   source $HOME/intel/oneapi/setvars.sh # 设置环境变量
   ```

2. 运行 gemm-test，测试矩阵乘法算子 `cblas_gemm_s8u8s32`/`cblas_gemm_bf16bf16f32` 的性能

   ```bash
   # 不同矩阵大小下运行 gemm-test
   $ ./test-msize [-c num_cores]
   ```

3. 安装 intel vtune 分析器

   ```bash
   wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/e7797b12-ce87-4df0-aa09-df4a272fc5d9/intel-vtune-2025.0.0.1130.sh
   sh ./intel-vtune-2025.0.0.1130.sh
   source $HOME/intel/oneapi/setvars.sh # 设置环境变量
   ```

4. 使用 intel vtune 分析器统计AMX功能部件的利用率

   ```bash
   vtune -collect uarch-exploration -knob sampling-interval=0.5 -knob pmu-collection-mode=summary -r ./perf/ -- ./test-msize.sh [-c num_cores]
   ```

   在打印的总结报告中，可以找到 **AMX Busy: ?% of Clockticks**.

5. **Tip 1**: Set `KMP_AFFINITY` to Avoid Thread Migration
   > 为了在具有多核处理器的系统上实现最佳性能，要求线程不要在核心之间迁移。为此，请通过为线程设置关联掩码将线程绑定到 CPU 核心。对于 GEMM 性能测试，KMP_AFFINITY 环境变量很有用：

   ```bash
   export KMP_AFFINITY=compact,1,0,granularity=fine
   ```

6. **Tip 2**: Align the Data
    > 在Intel处理器上，缓存行通常是64字节，数据读/写与缓存行对齐。要提高调用 oneMKL 的应用程序的性能，请在 64 字节边界上对齐测试数组，并使用 `mkl_malloc` 和 `mkl_free` 来分配和释放对齐的内存。

  ```C
    A = (float*) mkl_malloc(sizeof(float)*lda*m, MKL_MEM_ALIGNMENT);
  ```

7. **Tip 3**: Avoid Leading Dimensions that are Multiples of 256
    > 为了避免缓存冲突，前导维度(lda, ldb)应该是缓存行的倍数，但不能是 2 的幂!
    > 也就是说，lda, ldb 应该是 64 的倍数，但不能是 128,256等。
    > example:

   ```
   Matrix_size(mkn)  4096  4096  4096 Average_time(ms) 1.66052 tops 82.76886
   Matrix_size(mkn)  4160  4160  4160 Average_time(ms) 1.43643 tops 100.23611
   ```

8. **Tip 4**: Use Variations of GEMM
    > oneMKL 通常提供并行高性能 GEMM 实现，以使用现代多核架构支持的并发线程。此策略在乘以大型矩阵时效果很好，因为所有核心都得到了有效利用。
    > 然而，在乘以小矩阵或特殊用途的场景时，经典的 GEMM 调用可能无法最佳地使用所有核心。

    > 使用 `cblas_gemm_?_pack` 函数预处理（packing）输入矩阵，将其转换为一种优化的内部格式，然后调用 `cblas_gemm_?_compute` 接口计算，与直接调用矩阵乘法接口 `cblas_gemm_?` 相比，可以提高性能。

   ```
   Matrix_size(mkn)  4160  4160  4160 Average_time(ms) 1.43075 tops 100.63456
   Matrix_size(mkn)  4160  4160  4160 Average_time(ms) 0.72548 tops 198.46510
   ```
