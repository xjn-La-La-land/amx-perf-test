DATA_TYP = s8u8
SRC = gemm_$(DATA_TYP)_test.c
BUILD_DIR = ./build

BIN = $(BUILD_DIR)/gemm-test
LOG = $(BUILD_DIR)/results.txt

CC = mpic++
CFLAGS = -m64 -DMKL_ALIGN -DMKL_LD
LDFLAGS = -m64 -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_cdft_core -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_openmpi_lp64 -lgomp -lpthread -lm -ldl

m ?= 4160
n ?= 4160
k ?= 4160
CFLAGS_MSIZE += -DMSIZE_M=$(m)
CFLAGS_MSIZE += -DMSIZE_N=$(n)
CFLAGS_MSIZE += -DMSIZE_K=$(k)
CFLAGS += $(CFLAGS_MSIZE)


$(BIN): $(SRC)
	@ mkdir -p $(BUILD_DIR)
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@ 

run: $(BIN)
	@ ./$(BIN)

run-setcore: $(BIN)
	@ taskset -c $(core) ./$(BIN)
	
clean:
	rm -rf $(BUILD_DIR)

default: $(BIN)
