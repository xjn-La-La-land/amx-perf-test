#include <array>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>

#define ILEN 32

#define CUSTOM0_OPCODE 0b0001011
#define XTM_OPCODE CUSTOM0_OPCODE

#define TILELD_FUNCT3 0b100
#define TILEST_FUNCT3 0b110
#define TDPBSSD_FUNCT3 0b011
#define TDPBSUD_FUNCT3 0b010
#define TDPBUSD_FUNCT3 0b001
#define TDPBUUD_FUNCT3 0b000

class XtmInstr {
public:
  static constexpr uint8_t opcode = XTM_OPCODE;

  virtual uint32_t encode() const = 0;           // 生成指令的 bit 码
  std::array<uint8_t, 4> encodeToBytes() const { // 生成小端序的字节数组
    uint32_t instr = encode();
    return {static_cast<uint8_t>(instr & 0xFF),
            static_cast<uint8_t>((instr >> 8) & 0xFF),
            static_cast<uint8_t>((instr >> 16) & 0xFF),
            static_cast<uint8_t>((instr >> 24) & 0xFF)};
  }
  // 打印指令汇编
  virtual void printInfo() const = 0;
  // 打印出字节序列
  void printBytes() const {
    auto iBytes = encodeToBytes();
    std::cout << "[";
    for (int i = 0; i < 4; ++i) {
      std::cout << "0x" << std::hex << std::setw(2) << std::setfill('0')
                << (uint32_t)iBytes[i];
      if (i != 3) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  // 输出指令的 32 位十六进制表示
  void printWord() const {
    std::cout << "0x" << std::hex << std::setw(8) << std::setfill('0')
              << encode() << std::endl;
  }
  // 输出内联汇编代码
  void printInlineCode() const {
    std::cout << "\".insn " << std::hex << std::setw(8) << std::setfill('0') << encode() << "\\n\\t\"";
    std::cout << " // ";
    printInfo();
  }
};

class TILELOADD : public XtmInstr {
public:
  int tmm1;
  int rs1, rs2;
  int imm;
  static constexpr uint8_t funct3 = TILELD_FUNCT3;

  // 构造函数
  TILELOADD(int tmm1, int rs1, int rs2, int imm)
      : tmm1(tmm1 & 0x7), rs1(rs1 & 0x1F), rs2(rs2 & 0x1F), imm(imm & 0xFFF) {}

  // 生成指令的 bit 码
  uint32_t encode() const override {
    uint32_t instr = 0;
    instr |= (imm & 0xFE0) << (25 - 5); // imm[11:5]
    instr |= rs2 << 20;                 // rs2
    instr |= rs1 << 15;                 // rs1
    instr |= funct3 << 12;              // funct3
    instr |= (imm & 0x18) << (10 - 3);  // imm[4:3]
    instr |= tmm1 << 7;                 // tmm1
    instr |= opcode;                    // opcode
    return instr;
  }

  void printInfo() const override {
    std::cout << "tileloadd tmm" << tmm1 << ", " << imm << "(x" << rs1 << "), x"
              << rs2 << std::endl;
  }
};

class TILESTORED : public XtmInstr {
public:
  // 指令字段
  int tmm1;
  int rs1, rs2;
  int imm;
  static constexpr uint8_t funct3 = TILEST_FUNCT3;

  // 构造函数
  TILESTORED(int tmm1, int rs1, int rs2, int imm)
      : tmm1(tmm1 & 0x7), rs1(rs1 & 0x1F), rs2(rs2 & 0x1F), imm(imm & 0xFFF) {}

  // 生成指令的 bit 码
  uint32_t encode() const override {
    uint32_t instr = 0;
    instr |= (imm & 0xFE0) << (25 - 5); // imm[11:5]
    instr |= rs2 << 20;                 // rs2
    instr |= rs1 << 15;                 // rs1
    instr |= funct3 << 12;              // funct3
    instr |= (imm & 0x18) << (10 - 3);  // imm[4:3]
    instr |= tmm1 << 7;                 // tmm1
    instr |= opcode;                    // opcode
    return instr;
  }

  void printInfo() const override {
    std::cout << "tilestored tmm" << tmm1 << ", " << imm << "(x" << rs1
              << "), x" << rs2 << std::endl;
  }
};

class TDPBSSD : public XtmInstr {
public:
  // 指令字段
  int tmm1, tmm2, tmm3;
  static constexpr uint8_t funct3 = TDPBSSD_FUNCT3;

  // 构造函数
  TDPBSSD(int tmm1, int tmm2, int tmm3)
      : tmm1(tmm1 & 0x7), tmm2(tmm2 & 0x7), tmm3(tmm3 & 0x7) {}

  // 生成指令的 bit 码
  uint32_t encode() const override {
    uint32_t instr = 0;
    instr |= tmm3 << 26;         // tmm3
    instr |= (tmm2 & 0x4) << 25; // tmm2[2]
    instr |= funct3 << 12;       // funct3
    instr |= (tmm2 & 0x3) << 10; // tmm2[1:0]
    instr |= tmm1 << 7;          // tmm1
    instr |= opcode;             // opcode
    return instr;
  }

  void printInfo() const override {
    std::cout << "tdpbssd tmm" << tmm1 << ", tmm" << tmm2 << ", tmm" << tmm3
              << std::endl;
  }
};

class TDPBSUD : public XtmInstr {
public:
  // 指令字段
  int tmm1, tmm2, tmm3;
  static constexpr uint8_t funct3 = TDPBSUD_FUNCT3;

  // 构造函数
  TDPBSUD(int tmm1, int tmm2, int tmm3)
      : tmm1(tmm1 & 0x7), tmm2(tmm2 & 0x7), tmm3(tmm3 & 0x7) {}

  // 生成指令的 bit 码
  uint32_t encode() const override {
    uint32_t instr = 0;
    instr |= tmm3 << 26;         // tmm3
    instr |= (tmm2 & 0x4) << 25; // tmm2[2]
    instr |= funct3 << 12;       // funct3
    instr |= (tmm2 & 0x3) << 10; // tmm2[1:0]
    instr |= tmm1 << 7;          // tmm1
    instr |= opcode;             // opcode
    return instr;
  }

  void printInfo() const override {
    std::cout << "tdpbsud tmm" << tmm1 << ", tmm" << tmm2 << ", tmm" << tmm3
              << std::endl;
  }
};

class TDPBUSD : public XtmInstr {
public:
  // 指令字段
  int tmm1, tmm2, tmm3;
  static constexpr uint8_t funct3 = TDPBUSD_FUNCT3;

  // 构造函数
  TDPBUSD(int tmm1, int tmm2, int tmm3)
      : tmm1(tmm1 & 0x7), tmm2(tmm2 & 0x7), tmm3(tmm3 & 0x7) {}

  // 生成指令的 bit 码
  uint32_t encode() const override {
    uint32_t instr = 0;
    instr |= tmm3 << 26;         // tmm3
    instr |= (tmm2 & 0x4) << 25; // tmm2[2]
    instr |= funct3 << 12;       // funct3
    instr |= (tmm2 & 0x3) << 10; // tmm2[1:0]
    instr |= tmm1 << 7;          // tmm1
    instr |= opcode;             // opcode
    return instr;
  }

  void printInfo() const override {
    std::cout << "tdpbusd tmm" << tmm1 << ", tmm" << tmm2 << ", tmm" << tmm3
              << std::endl;
  }
};

class TDPBUUD : public XtmInstr {
public:
  // 指令字段
  int tmm1, tmm2, tmm3;
  static constexpr uint8_t funct3 = TDPBUUD_FUNCT3;

  // 构造函数
  TDPBUUD(int tmm1, int tmm2, int tmm3)
      : tmm1(tmm1 & 0x7), tmm2(tmm2 & 0x7), tmm3(tmm3 & 0x7) {}

  // 生成指令的 bit 码
  uint32_t encode() const override {
    uint32_t instr = 0;
    instr |= tmm3 << 26;         // tmm3
    instr |= (tmm2 & 0x4) << 25; // tmm2[2]
    instr |= funct3 << 12;       // funct3
    instr |= (tmm2 & 0x3) << 10; // tmm2[1:0]
    instr |= tmm1 << 7;          // tmm1
    instr |= opcode;             // opcode
    return instr;
  }

  void printInfo() const override {
    std::cout << "tdpbuud tmm" << tmm1 << ", tmm" << tmm2 << ", tmm" << tmm3
              << std::endl;
  }
};

/* ------------------------------------------------------------------------ */
#include <random>

// generate some XTM instructions
void genXTM_demo() {
  std::random_device rd;                       // 获取硬件随机数种子
  std::mt19937 gen(rd());                      // 使用梅森旋转算法生成随机数
  std::uniform_int_distribution<> dis(1, 100); // 创建一个均匀分布的随机数引擎
  for (int r = 0; r < 2; ++r) {
    int tmm1 = dis(gen) % 8;
    int tmm2 = dis(gen) % 8;
    int tmm3 = dis(gen) % 8;
    int rs1 = dis(gen) % 32;
    int rs2 = dis(gen) % 32;
    int imm = 0;

    TILELOADD tileld(tmm1, rs1, rs2, imm);
    TILESTORED tilest(tmm1, rs1, rs2, imm);
    TDPBSSD tdpbssd(tmm1, tmm2, tmm3);

    tileld.printInlineCode();
    tilest.printInlineCode();
    tdpbssd.printInlineCode();

  }
}

int main() {
  genXTM_demo();
  return 0;
}
