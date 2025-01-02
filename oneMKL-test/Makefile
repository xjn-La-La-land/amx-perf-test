DATA_TYP = s8u8
SRC = gemm_$(DATA_TYP)_test.c
BUILD_DIR = ./build

BIN = $(BUILD_DIR)/gemm-test
LOG = $(BUILD_DIR)/results.txt

IPATH = $(HOME)/miniconda/include
LPATH = $(HOME)/miniconda/lib

CC = gcc
CFLAGS = -m64 -I$(IPATH) 
LDFLAGS = -m64 -L$(LPATH) -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
CFLAGS_MSIZE += -DMSIZE_M=$(m)
CFLAGS_MSIZE += -DMSIZE_N=$(n)
CFLAGS_MSIZE += -DMSIZE_K=$(k)
CFLAGS += $(CFLAGS_MSIZE)


$(BIN): $(SRC)
	@ $(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@ 

run: $(BIN)
	@ ./$(BIN)

run-setcore: $(BIN)
	@ taskset -c 0-$(shell expr $(NUM_CORES) - 1) ./$(BIN)
	
clean:
	rm -f $(BUILD_DIR)

default: $(BIN)