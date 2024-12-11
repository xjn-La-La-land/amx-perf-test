import matplotlib.pyplot as plt

# 1. 读取文件并解析数据
matrix_sizes = []
times = []
gflops = []

with open("results.txt", "r") as file:
    for line in file:
        parts = line.strip().split()  # 按空格分割行
        matrix_sizes.append(int(parts[1]) * int(parts[2]) * int(parts[3]))
        times.append(float(parts[5]))     # 提取 time 值
        gflops.append(float(parts[-1]))    # 提取 gflops 值



# 2. 绘制图表
plt.figure(figsize=(8, 6))  # 设置图表尺寸
plt.plot(matrix_sizes, times, marker='o', markersize=2, label="Time vs Matrix Size")

# 3. 添加标题和标签
plt.title("time vs matrix size for cblas_gemm_bf16bf16f32", fontsize=14)
plt.xlabel("Matrix Size(m*n*k)", fontsize=12)
plt.ylabel("Average Runtime(milliseconds)", fontsize=12)

# 4. 显示网格和图例
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# 5. 保存图表或显示
plt.savefig("cblas_gemm_bf16bf16f32.png", dpi=300)  # 保存图表
plt.show()  # 显示图表