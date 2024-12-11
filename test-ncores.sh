#!/bin/bash

# 输出文件
output_file="./log.txt"

nc_start=5
nc_end=$(nproc)
nc_step=5

echo "max cores: $nc_end"

for nc in $(seq $nc_start $nc_step $nc_end); do
    echo "Running with $nc cores..."
    echo -n "num_core $nc" >> $output_file
    rm -rf ./perf
    vtune -collect uarch-exploration -knob sampling-interval=0.5 -knob pmu-collection-mode=summary -r ./perf/ -- make run-setcore m=4096 n=4096 k=4096 NUM_CORES=$nc -B
    vtune -report summary -r ./perf/ | grep "AMX Busy" >> $output_file
done

echo "All tests completed. Results are saved in $output_file."