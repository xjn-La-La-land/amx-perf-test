CROSS_COMPILE = riscv64-linux-gnu-
CC        = $(CROSS_COMPILE)gcc
LD        = $(CROSS_COMPILE)ld
OBJDUMP   = $(CROSS_COMPILE)objdump
OBJCOPY   = $(CROSS_COMPILE)objcopy
READELF   = $(CROSS_COMPILE)readelf

COMMON_CFLAGS = -fno-pic -mcmodel=medany -mstrict-align
COMMON_CFLAGS += -march=rv64g

CFLAGS = -O2 -MMD -Wall -Werror
CFLAGS += -fdata-sections -ffunction-sections
CFLAGS += $(COMMON_CFLAGS) -static

LDFLAGS = -z noexecstack
LDFLAGS += --gc-sections
LDFLAGS += -melf64lriscv

BUILD_DIR = build
CSRCS = rvxtm-gemm.c
BIN_TGT = $(BUILD_DIR)/rvxtm-gemm
OBJ_TGT = $(BIN_TGT).o
ELF_TGT = $(BIN_TGT).elf
TXT_TGT = $(BIN_TGT).txt

all: compile objdump

# $(OBJ_TGT): $(CSRCS)
# 	@echo "Building $(OBJ_TGT)"
# 	@mkdir -p $(BUILD_DIR)
# 	$(CC) $(CFLAGS) -c -o $(OBJ_TGT) $(CSRCS)
#
# $(ELF_TGT): $(OBJ_TGT)
# 	@echo "Building $(ELF_TGT)"
# 	$(CC) $(LDFLAGS) -o $(ELF_TGT) $(OBJ_TGT)
#
# $(TXT_TGT): $(ELF_TGT)
# 	@echo "Dumping $(ELF_TGT) to $(TXT_TGT)"
# 	@$(OBJDUMP) -d $(ELF_TGT) > $(TXT_TGT)objdump
#
# $(BIN_TGT): $(ELF_TGT)
# 	@echo "Copying $(ELF_TGT) to $(BIN_TGT)"
# 	@$(OBJCOPY) -S -O binary -j .text $(ELF_TGT) $(BIN_TGT)


compile: $(CSRCS)
	@echo "Building $(BIN_TGT)"
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $(BIN_TGT) $(CSRCS)

objdump: $(BIN_TGT)
	@echo "Dumping $(BIN_TGT)"
	$(OBJDUMP) -d $(BIN_TGT) > $(BIN_TGT).s

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all compile objdump clean
