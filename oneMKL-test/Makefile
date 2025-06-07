# DATA_TYP = s8u8
DATA_TYP = bf16
SRC = gemm_$(DATA_TYP)_test.c
BUILD_DIR = build

BIN = $(BUILD_DIR)/gemm-$(DATA_TYP)-test
LOG = $(BUILD_DIR)/results.txt

# CC = mpic++
# CFLAGS = -DMKL_ILP64  -m64  -I"${MKLROOT}/include"
# LDFLAGS = -m64  -L${MKLROOT}/lib -lmkl_scalapack_ilp64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_ilp64 -liomp5 -lpthread -lm -ldl
CC = icpx
CFLAGS = -DMKL_ILP64 -fsycl -I"${MKLROOT}/include"
LDFLAGS = ${MKLROOT}/lib/libmkl_scalapack_ilp64.a -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_intel_thread.a ${MKLROOT}/lib/libmkl_core.a ${MKLROOT}/lib/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

NUM_CORE ?= 240
CORE_LIST = $(shell seq 0 $(shell expr $(NUM_CORE) - 1))
FREQ ?= 2300000 # 2.3 GHz
ifeq ($(NUM_CORE),1)
	core = 0
else
	core = 0-$(shell expr $(NUM_CORE) - 1)
endif

# open-hyper-threading!
export KMP_AFFINITY=compact,1,0,granularity=fine

lockfreq:
	for i in $(CORE_LIST); do \
		echo userspace | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_governor > /dev/null; \
		echo $(FREQ) | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_setspeed > /dev/null; \
	done

unlockfreq:
	for i in $(CORE_LIST); do \
		echo ondemand | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_governor > /dev/null; \
	done


$(BIN): $(SRC)
	@ mkdir -p $(BUILD_DIR)
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@ 

run: $(BIN)
	make lockfreq
	taskset -c $(core) ./$(BIN) $(NUM_CORE)
	make unlockfreq
	
clean:
	rm -rf $(BUILD_DIR)

default: $(BIN)
