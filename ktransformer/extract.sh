#!/bin/bash

# 检查是否传入文件名
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

input_file="$1"

# 判断文件是否存在
if [ ! -f "$input_file" ]; then
  echo "Error: File '$input_file' not found."
  exit 1
fi

matrix_sz=()
utils=()

# 逐行读取文件
while read -r line; do
  if [[ "$line" == M\ N\ K* ]]; then
    # 提取 M（假设 M=N=K）
    m=$(echo "$line" | awk '{print $5}')
    # 提取 Utilization 百分比
    util=$(echo "$line" | grep -o 'Utilization = [0-9.]\+%' | awk '{print $3}' | tr -d '%')

    matrix_sz+=($m)
    utils+=($util)
  fi
done <"$input_file"

# 输出 Python 格式数组
echo -n "matrix_sz = ["
printf "%s" "${matrix_sz[0]}"
for ((i = 1; i < ${#matrix_sz[@]}; i++)); do
  printf ", %s" "${matrix_sz[i]}"
done
echo "]"

echo -n "utilizations = ["
printf "%s" "${utils[0]}"
for ((i = 1; i < ${#utils[@]}; i++)); do
  printf ", %s" "${utils[i]}"
done
echo "]"
