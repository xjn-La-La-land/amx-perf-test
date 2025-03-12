#!/bin/bash

# 输出文件
output_file="./build/result.txt"
mkdir -p build/
touch $output_file

# 参数范围和步长
m=4096
n=4096

k_start=100
k_end=4000
k_step=100

if [ "$#" = 0 ]; then
  echo "Running on all cores..."
  num_cores=-1
elif [ "$#" = 2 ] && [ "$1" = "-c" ]; then
  echo "Running on $2 cores..."
  num_cores=$2
else
  echo "Usage: ./test.sh [-c num_cores]"
  exit 1
fi

# 循环遍历所有参数组合
for k in $(seq $k_start $k_step $k_end); do
  echo "Running with m=$m, n=$n, k=$k..."
  echo -n "Matrix_Size(mnk) $m $n $k " >>$output_file
  rm -rf ./perf
  if [ "$num_cores" = -1 ]; then
    # make run m=$m n=i$n k=$k -B
    vtune -collect uarch-exploration -knob sampling-interval=0.5 -knob pmu-collection-mode=summary -r ./perf/ -- make run m=$m n=$n k=$k -B
  else
    # make run-setcore m=$m n=$n k=$k NUM_CORES=$num_cores -B
    vtune -collect uarch-exploration -knob sampling-interval=0.5 -knob pmu-collection-mode=summary -r ./perf/ -- make run-setcore m=$m n=$n k=$k core=0-$num_cores -B
  fi
  vtune -report summary -r ./perf/ | grep "AMX Busy" >>$output_file
done

echo "All tests completed. Results are saved in $output_file."
