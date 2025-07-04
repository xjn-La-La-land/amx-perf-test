SRC = $(wildcard *.c)
OBJ = $(SRC:%.c=$(BUILD_DIR)/%.o)
DEP = $(OBJ:.o=.d)
BUILD_DIR = build
BIN = $(BUILD_DIR)/gemm-test

# CC = mpic++
# CFLAGS = -m64 -march=native -fno-strict-aliasing -O2 -DMKL_ALIGN -DMKL_LD
# LDFLAGS = -m64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
CC = gcc
CFLAGS = -O3 -march=native -fopenmp -fno-strict-aliasing
CFLAGS += -MMD -MP
LDFLAGS = -fopenmp -lnuma

# cache block size
ifneq ($(TM),)
	CFLAGS += -DTM=$(TM)
endif
ifneq ($(TN),)
	CFLAGS += -DTN=$(TN)
endif
ifneq ($(TK),)
	CFLAGS += -DTK=$(TK)
endif

core ?= 1

# core = 0,1,2 => NUM_CORE = 3
# core = 0-29  => NUM_CORE = 30
NUM_CORE := $(shell echo $(core) | awk '\
BEGIN{n=0} \
{ gsub(/,/," ",$$0); \
	for(i=1;i<=NF;i++){ \
		if ($$i ~ /-/) { \
		split($$i,a,"-"); n += a[2]-a[1]+1 \
		} else {n++} \
	} \
	print n \
}')

# core = 0,1,2 => CORE_LIST = 0 1 2
# core = 0-29  => CORE_LIST = 0 1 2 ... 29
CORE_LIST := $(shell echo $(core) | awk '\
BEGIN {OFS=" "} \
{ gsub(/,/," ",$$0); \
	for(i=1;i<=NF;i++){ \
		if ($$i ~ /-/) { \
			split($$i,a,"-"); \
			for(j=a[1];j<=a[2];j++) printf j" " \
		} else { printf $$i" " } \
	} \
}')


FREQ ?= 2300000 # 2.3 GHz
LOOP ?= 10      # Number of rounds to run


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
	taskset -c $(core) ./$(BIN) -c $(NUM_CORE) -f $(FREQ) -r $(LOOP)
	make unlockfreq

perf: $(BIN)
	make lockfreq
	# taskset -c $(core) sudo perf stat $(PERFFLAGS) ./$(BIN) -c $(NUM_CORE) -f $(FREQ) -r $(LOOP)
	taskset -c $(core) sudo perf stat -d -d -d ./$(BIN) -c $(NUM_CORE) -f $(FREQ) -r $(LOOP)
	make unlockfreq


run-1-node: $(BIN)
	# node 0
	make lockfreq core="0-29"
	numactl --cpunodebind=0 --membind=0 ./$(BIN) \
	-n 1 -f $(FREQ) -r $(LOOP)
	make unlockfreq core="0-29"

run-2-node: $(BIN)
	# node 0,1
	make lockfreq core="0-59"
	numactl --cpunodebind=0,1 --membind=0,1 ./$(BIN) \
	-n 2 -f $(FREQ) -r $(LOOP)
	make unlockfreq core="0-59"
	
run-4-node: $(BIN)
	# node = 0,1,2,3
	make lockfreq core="0-119"
	numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 ./$(BIN) \
	-n 4 -f $(FREQ) -r $(LOOP)
	make unlockfreq core="0-119"
	
clean:
	rm -rf $(BUILD_DIR)

default: $(BIN)
