SRC = $(wildcard *.cpp)
BUILD_DIR = ./build

BIN = $(BUILD_DIR)/amx_gemm-test
LOG = $(BUILD_DIR)/results.txt

CC = mpic++
CFLAGS = -m64 -march=native -fno-strict-aliasing -O2 -DMKL_ALIGN -DMKL_LD
LDFLAGS = -m64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

m ?= 4160
n ?= 4160
k ?= 4160
core ?= 0-119
CFLAGS_MSIZE += -DMSIZE_M=$(m)
CFLAGS_MSIZE += -DMSIZE_N=$(n)
CFLAGS_MSIZE += -DMSIZE_K=$(k)
CFLAGS += $(CFLAGS_MSIZE)


$(BIN): $(SRC) $(wildcard *.hpp)
	@ mkdir -p $(BUILD_DIR)
	@ $(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@ 

run: $(BIN)
	@ ./$(BIN)

run-setcore: $(BIN)
	@ taskset -c $(core) ./$(BIN)
	
clean:
	rm -rf $(BUILD_DIR)

default: $(BIN)
