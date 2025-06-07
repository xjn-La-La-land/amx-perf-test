SRC = $(wildcard *.cpp)
BUILD_DIR = ./build

BIN = $(BUILD_DIR)/amx_gemm-test

CC = mpic++
CFLAGS = -m64 -march=native -fno-strict-aliasing -O2 -DMKL_ALIGN -DMKL_LD
LDFLAGS = -m64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

NUM_CORE ?= 1
CORE_LIST = $(shell seq 0 $(shell expr $(NUM_CORE) - 1))
FREQ ?= 2300000 # 2.3 GHz
ifeq ($(NUM_CORE),1)
	core = 0
else
	core = 0-$(shell expr $(NUM_CORE) - 1)
endif

# open-hyper-threading!
export KMP_AFFINITY=compact,1,0,granularity=fine



$(BIN): $(SRC) $(wildcard *.hpp)
	@ mkdir -p $(BUILD_DIR)
	@ $(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@ 

lockfreq:
	for i in $(CORE_LIST); do \
		echo userspace | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_governor > /dev/null; \
		echo $(FREQ) | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_setspeed > /dev/null; \
	done

unlockfreq:
	for i in $(CORE_LIST); do \
		echo ondemand | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_governor > /dev/null; \
	done


run: $(BIN)
	make lockfreq
	taskset -c $(core) ./$(BIN) $(NUM_CORE)
	make unlockfreq

gdb: $(BIN)
	@ gdb $<

clean:
	rm -rf $(BUILD_DIR)

default: $(BIN)
