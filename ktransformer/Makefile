SRC = $(wildcard *.cpp)
BUILD_DIR = ./build

BIN = $(BUILD_DIR)/amx_gemm-test

# CC = mpic++
# CFLAGS = -m64 -march=native -fno-strict-aliasing -O2 -DMKL_ALIGN -DMKL_LD
# LDFLAGS = -m64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
CC = g++
CFLAGS = -O3 -march=native -fopenmp -fno-strict-aliasing
CFLAGS += -MMD -MP
LDFLAGS = -fopenmp -lnuma


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

PERFFLAGS = -e exe.amx_busy -e cpu-cycles

# PERFFLAGS += -e L1-dcache-loads -e L1-dcache-load-misses
# PERFFLAGS += -e l2_request.all,l2_request.miss
# PERFFLAGS += -e LLC-loads,LLC-load-misses


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
	taskset -c $(core) ./$(BIN) -t $(NUM_CORE) -f $(FREQ) -r $(LOOP)
	make unlockfreq

gdb: $(BIN)
	@ gdb $<

perf: $(BIN)
	make lockfreq
	taskset -c $(core) sudo perf stat $(PERFFLAGS) ./$(BIN) -t $(NUM_CORE) -f $(FREQ) -r $(LOOP)
	make unlockfreq

clean:
	rm -rf $(BUILD_DIR)

default: $(BIN)

