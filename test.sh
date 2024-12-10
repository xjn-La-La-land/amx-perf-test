#!/bin/bash

# 输出文件
output_file="./results.txt"
> $output_file

# 参数范围和步长
m=4096
n=4096

k_start=100
k_end=4000
k_step=100

# 循环遍历所有参数组合
for k in $(seq $k_start $k_step $k_end); do
    echo "Running with m=$m, n=$n, k=$k..."
    if [ "$1" = "single" ]; then
        make run-single m=$m n=$n k=$k -B
    elif [ "$1" = "multi" ]; then
        make run m=$m n=$n k=$k -B
    else
        echo "Usage: ./test.sh [single|multi]"
        exit 1
    fi
done

echo "All tests completed. Results are saved in $output_file."
