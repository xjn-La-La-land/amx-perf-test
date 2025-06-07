SRC = $(wildcard *.c)
BUILD_DIR = build

BIN = $(BUILD_DIR)/gemm-test

CC = mpic++
# CC = gcc
# CFLAGS = -O3 -march=native -fno-strict-aliasing -fopenmp
CFLAGS = -m64 -march=native -fno-strict-aliasing -O2 -DMKL_ALIGN -DMKL_LD
LDFLAGS = -m64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

NUM_CORE ?= 240
CORE_LIST = $(shell seq 0 $(shell expr $(NUM_CORE) - 1))
# FREQ ?= 2500000 # 2.5 GHz
FREQ ?= 2300000 # 2.3 GHz
# FREQ ?= 2000000 # 2.0 GHz
# FREQ ?= 1800000 # 1.8 GHz
# FREQ ?= 1500000 # 1.5 GHz
ifeq ($(NUM_CORE),1)
	core = 0
else
	core = 0-$(shell expr $(NUM_CORE) - 1)
endif

# open-hyper-threading!
export KMP_AFFINITY=compact,1,0,granularity=fine

PERFFLAGS = -e exe.amx_busy \
			-e cpu-cycles

# PERFFLAGS = -e L1-dcache-loads,L1-dcache-load-misses \
#             -e l2_request.all,l2_request.miss \
# 			-e LLC-loads,LLC-load-misses \

lockfreq:
	for i in $(CORE_LIST); do \
		echo userspace | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_governor > /dev/null; \
		echo $(FREQ) | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_setspeed > /dev/null; \
	done

unlockfreq:
	for i in $(CORE_LIST); do \
		echo ondemand | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_governor > /dev/null; \
	done

MSR_ADDR = 0x1a4
MSR_BIT = 4
disable-l1d-prefetch:
	for i in $(CORE_LIST); do \
		cur_val=$$(sudo rdmsr -p $$i $(MSR_ADDR)); \
		new_val=$$((cur_val | (1 << $(MSR_BIT)))); \
		echo "Original MSR: $$cur_val (0x$$(printf '%x' $$cur_val)), New: $$new_val (0x$$(printf '%x' $$new_val))"; \
		sudo wrmsr -p $$i $(MSR_ADDR) $$new_val; \
	done

enable-l1d-prefetch:
	for i in $(CORE_LIST); do \
		cur_val=$$(sudo rdmsr -p $$i $(MSR_ADDR)); \
		restored_val=$$((cur_val & ~(1 << $(MSR_BIT)))); \
		echo "Restored MSR: $$restored_val (0x$$(printf '%x' $$restored_val))"; \
		sudo wrmsr -p $$i $(MSR_ADDR) $$restored_val; \
	done

$(BIN): $(SRC)
	@ mkdir -p $(BUILD_DIR)
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@ 

run: $(BIN)
	make lockfreq
	taskset -c $(core) ./$(BIN) $(NUM_CORE)
	make unlockfreq

perf: $(BIN)
	make lockfreq
	taskset -c $(core) sudo perf stat $(PERFFLAGS) ./$(BIN) $(NUM_CORE)
	make unlockfreq
	
clean:
	rm -rf $(BUILD_DIR)

default: $(BIN)
