SRC = $(wildcard *.c)
OBJ = $(SRC:%.c=$(BUILD_DIR)/%.o)
DEP = $(OBJ:.o=.d)
BUILD_DIR = build
BIN = $(BUILD_DIR)/gemm-test

CC = mpic++
# CC = gcc
# CFLAGS = -O3 -march=native -fno-strict-aliasing -fopenmp
CFLAGS = -m64 -march=native -fno-strict-aliasing -O2 -DMKL_ALIGN -DMKL_LD
LDFLAGS = -m64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
CFLAGS += -MMD -MP

ifneq ($(LOOP_COUNT),)
	CFLAGS += -DLOOP_COUNT=$(LOOP_COUNT)
endif
ifneq ($(TM),)
	CFLAGS += -DTM=$(TM)
endif
ifneq ($(TN),)
	CFLAGS += -DTN=$(TN)
endif
ifneq ($(TK),)
	CFLAGS += -DTK=$(TK)
endif

NUM_CORE ?= 240
ifeq ($(NUM_CORE),1)
	CORE_LIST = 1
	core = 1
else
	CORE_LIST = $(shell seq 0 $(shell expr $(NUM_CORE) - 1))
	core = 0-$(shell expr $(NUM_CORE) - 1)
endif
# FREQ ?= 2500000 # 2.5 GHz
FREQ ?= 2300000 # 2.3 GHz
# FREQ ?= 2000000 # 2.0 GHz
# FREQ ?= 1800000 # 1.8 GHz
# FREQ ?= 1500000 # 1.5 GHz

# open-hyper-threading!
export KMP_AFFINITY=compact,1,0,granularity=fine

# PERFFLAGS = -e exe.amx_busy -e cpu-cycles

PERFFLAGS += -e L1-dcache-loads -e L1-dcache-load-misses
PERFFLAGS += -e l2_request.all,l2_request.miss
PERFFLAGS += -e LLC-loads,LLC-load-misses

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

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJ)
	@mkdir -p $(dir $@)
	$(CC) $^ $(LDFLAGS) -o $@

-include $(DEP)

run: $(BIN)
	make lockfreq
	taskset -c $(core) ./$(BIN) -c $(NUM_CORE) -f $(FREQ)
	make unlockfreq

perf: $(BIN)
	make lockfreq
	# taskset -c $(core) sudo perf stat $(PERFFLAGS) ./$(BIN) -c $(NUM_CORE) -f $(FREQ)
	taskset -c $(core) sudo perf stat -d -d -d ./$(BIN) -c $(NUM_CORE) -f $(FREQ)
	make unlockfreq
	
clean:
	rm -rf $(BUILD_DIR)

default: $(BIN)
