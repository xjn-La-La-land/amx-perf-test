## 测试 intel oneMKL 库矩阵乘法算子性能

1. 安装 oneMKL 库

   ```bash
   $ wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/79153e0f-74d7-45af-b8c2-258941adf58a/intel-onemkl-2025.0.0.940.sh
   $ sh ./intel-onemkl-2025.0.0.940.sh
   $ source $HOME/intel/oneapi/setvars.sh # 设置环境变量
   ```

2. 运行 gemm-test，测试矩阵乘法算子 `cblas_gemm_bf16bf16f32` 的性能

   ```bash
   # 不同矩阵大小下运行 gemm-test
   $ ./test-msize [-c num_cores]
   ```

3. 安装 intel vtune 分析器

   ```bash
   $ wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/e7797b12-ce87-4df0-aa09-df4a272fc5d9/intel-vtune-2025.0.0.1130.sh
   $ sh ./intel-vtune-2025.0.0.1130.sh
   $ source $HOME/intel/oneapi/setvars.sh # 设置环境变量
   ```

4. 使用 intel vtune 分析器统计AMX功能部件的利用率
   ```bash
   $ vtune -collect uarch-exploration -knob sampling-interval=0.5 -knob pmu-collection-mode=summary -r ./perf/ -- ./test-msize.sh [-c num_cores]
   ```
   在打印的总结报告中，可以找到 **AMX Busy: ?% of Clockticks**.