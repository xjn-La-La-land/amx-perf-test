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
LDFLAGS += --gc-sections -e _start
LDFLAGS += -melf64lriscv

BUILD_DIR = build
OBJ_TGT = $(BUILD_DIR)/instr-test.o
ELF_TGT = $(BUILD_DIR)/instr-test.elf
TXT_TGT = $(BUILD_DIR)/instr-test.txt
BIN_TGT = $(BUILD_DIR)/instr-test.bin

instr-test: $(OBJ_TGT) $(TXT_TGT) $(BIN_TGT)

$(OBJ_TGT): instr-test.c
	@echo "Building $(OBJ_TGT)"
	@$(CC) $(CFLAGS) -c -o $(OBJ_TGT) instr-test.c

$(ELF_TGT): $(OBJ_TGT)
	@echo "Building $(ELF_TGT)"
	@$(LD) $(LDFLAGS) -o $(ELF_TGT) $(OBJ_TGT)

$(TXT_TGT): $(ELF_TGT)
	@echo "Dumping $(ELF_TGT) to $(TXT_TGT)"
	@$(OBJDUMP) -d $(ELF_TGT) > $(TXT_TGT)

$(BIN_TGT): $(ELF_TGT)
	@echo "Copying $(ELF_TGT) to $(BIN_TGT)"
	@$(OBJCOPY) -S -O binary -j .text $(ELF_TGT) $(BIN_TGT)

clean:
	rm -f $(BUILD_DIR)/*

.PHONY: instr-test clean
