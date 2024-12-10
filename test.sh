#!/bin/bash

# 输出文件
output_file="./results.txt"
> $output_file

# 参数范围和步长
m=2048
n=2048

k_start=100
k_end=3000
k_step=100

# 循环遍历所有参数组合
for k in $(seq $k_start $k_step $k_end); do
    echo "Running with m=$m, n=$n, k=$k..."
    make run m=$m n=$n k=$k -B
done

echo "All tests completed. Results are saved in $output_file."
