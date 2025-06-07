import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 1. 读取文件并解析数据
matrix_sizes = []
tops = []
amx_usage = []
num_cores = [1, 30, 60, 120, 240]
log_files = ["./build/gemm-i8-1core.txt", "./build/gemm-i8-30core.txt", "./build/gemm-i8-60core.txt", "./build/gemm-i8-120core.txt", "./build/gemm-i8-240core.txt"]
colors = ['b', 'g', 'r', 'c', 'm']

with open(log_files[0], "r") as file:
    for line in file:
        parts = line.strip().split()  # 按空格分割行
        matrix_sizes.append(int(parts[4])) # M=N=K

for log_file in log_files:
    with open(log_file, "r") as file:
        tops_temp = []
        amx_usage_temp = []
        for line in file:
            parts = line.strip().split()  # 按空格分割行
            tops_temp.append(float(parts[14]))
            amx_usage_temp.append(float(parts[-1].strip('%')))
    tops.append(tops_temp)
    amx_usage.append(amx_usage_temp)


# 2. 绘制图表
# font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
# my_font = fm.FontProperties(fname=font_path)
# plt.rcParams['font.sans-serif'] = my_font.get_name()
# plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 6))  # 设置图表尺寸
for i, core in enumerate(num_cores):
    plt.plot(matrix_sizes, amx_usage[i], marker='o', markersize=2, linestyle='-', color=colors[i], label=f'Cores: {core}')

# 3. 添加标题和标签
plt.title("Intel Xeon 8580 GEMM AMX utilization", fontsize=14)
plt.xlabel("Matrix size(M=N=K)", fontsize=12)
plt.ylabel("AMX Utilization(% of Clockticks)", fontsize=12)

# 4. 显示网格和图例
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# 5. 保存图表或显示
plt.savefig("gemm-amx-utilization.png", dpi=300)  # 保存图表
# plt.show()  # 显示图表


plt.figure(figsize=(8, 6))
for i, core in enumerate(num_cores):
    plt.plot(matrix_sizes, tops[i], marker='o', markersize=2, linestyle='-', color=colors[i], label=f'Cores: {core}')

# 3. 添加标题和标签
plt.title("Intel Xeon 8580 GEMM AMX performance", fontsize=14)
plt.xlabel("Matrix size(M=N=K)", fontsize=12)
plt.ylabel("TOPS", fontsize=12)

# 4. 显示网格和图例
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# 5. 保存图表或显示
plt.savefig("gemm-amx-performance.png", dpi=300)  # 保存图表
# plt.show()  # 显示图表

