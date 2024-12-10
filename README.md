## 测试 intel oneMKL 库矩阵乘法算子性能

1. 安装 oneMKL 库

   ```bash
   $ wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/79153e0f-74d7-45af-b8c2-258941adf58a/intel-onemkl-2025.0.0.940.sh
   $ sh ./intel-onemkl-2025.0.0.940.sh
   ```

2. 运行 gemm-test，测试矩阵乘法算子 `cblas_gemm_bf16bf16f32` 的性能

   ```bash
   $ ./test.sh
   ```

3. 可能出现报错：error while loading shared libraries: libmkl_intel_lp64.so.1: cannot open shared object file: No such file or directory，原因是 Intel MKL 库的文件名和版本号不符合。解决方法：

   ```bash
   ln -s ~/intel/oneapi/mkl/latest/lib/libmkl_core.so.2 ~/intel/oneapi/mkl/latest/lib/libmkl_core.so.1
   ln -s ~/intel/oneapi/mkl/latest/lib/libmkl_intel_ilp64.so.2 ~/intel/oneapi/mkl/latest/lib/libmkl_intel_lp64.so.1
   ln -s ~/intel/oneapi/mkl/latest/lib/libmkl_gnu_thread.so.2 ~/intel/oneapi/mkl/latest/lib/libmkl_gnu_thread.so.1
   ```