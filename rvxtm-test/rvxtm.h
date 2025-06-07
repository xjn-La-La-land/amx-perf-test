#define ILEN 32

#define GET_VALUE1(x) #x
#define GET_VALUE(x) GET_VALUE1(x)

#define CUSTOM0_OPCODE 0b0001011
#define XTM_OPCODE CUSTOM0_OPCODE

#define TILELD_FUNCT3 (0b100 << 12)
#define TILELDT1_FUNCT3 (0b101 << 12)
#define TILEST_FUNCT3 (0b110 << 12)
#define TDPBSSD_FUNCT3 (0b011 << 12)
#define TDPBSUD_FUNCT3 (0b010 << 12)
#define TDPBUSD_FUNCT3 (0b001 << 12)
#define TDPBUUD_FUNCT3 (0b000 << 12)

#define REG(x) ((x) & 0x1F)
#define RS1(x) (REG(x) << 15)
#define RS2(x) (REG(x) << 20)

#define TMM(x) ((x) & 0x7)
#define TMM_C(x) (TMM(x) << 7)
#define TMM_A(x) ((TMM(x) & 0b100) << 25) | ((TMM(x) & 0b011) << 10)
#define TMM_B(x) (TMM(x) << 26)
#define TMM_SRC(x) TMM_C(x)
#define TMM_DST(x) TMM_C(x)

#define IMM_S(x) ((((x) & 0xFE0) << (25 - 5)) | (((x) & 0x18) << (10 - 3)))

#define TILELOADD(tmm, base, stride)                                           \
  {                                                                            \
    __asm__ __volatile__("sd t1, -16(sp)\n\t"                                  \
                         "sd t2,  -8(sp)\n\t"                                  \
                         "add t1, zero, %0\n\t"                                \
                         "add t2, zero, %1\n\t"                                \
                         ".word " GET_VALUE(XTM_OPCODE | TILELD_FUNCT3 |       \
                                            TMM_DST(tmm) | RS1(6) |            \
                                            RS2(7)) "\n\t"                     \
                                                    "ld t1, -16(sp)\n\t"       \
                                                    "ld t2,  -8(sp)\n\t"       \
                         :                                                     \
                         : "r"(base), "r"(stride));                            \
  }

#define TILELOADDT1(tmm, base, stride)                                         \
  {                                                                            \
    __asm__ __volatile__("sd t1, -16(sp)\n\t"                                  \
                         "sd t2,  -8(sp)\n\t"                                  \
                         "add t1, zero, %0\n\t"                                \
                         "add t2, zero, %1\n\t"                                \
                         ".word " GET_VALUE(XTM_OPCODE | TILELDT1_FUNCT3 |     \
                                            TMM_DST(tmm) | RS1(6) |            \
                                            RS2(7)) "\n\t"                     \
                                                    "ld t1, -16(sp)\n\t"       \
                                                    "ld t2,  -8(sp)\n\t"       \
                         :                                                     \
                         : "r"(base), "r"(stride));                            \
  }

#define TILESTORED(tmm, base, stride)                                          \
  {                                                                            \
    __asm__ __volatile__("sd t1, -16(sp)\n\t"                                  \
                         "sd t2,  -8(sp)\n\t"                                  \
                         "add t1, zero, %0\n\t"                                \
                         "add t2, zero, %1\n\t"                                \
                         ".word " GET_VALUE(XTM_OPCODE | TILEST_FUNCT3 |       \
                                            TMM_SRC(tmm) | RS1(6) |            \
                                            RS2(7)) "\n\t"                     \
                                                    "ld t1, -16(sp)\n\t"       \
                                                    "ld t2,  -8(sp)\n\t"       \
                         :                                                     \
                         : "r"(base), "r"(stride));                            \
  }

#define TDPBSSD(tmmC, tmmA, tmmB)                                              \
  {                                                                            \
    __asm__ __volatile__(".word " GET_VALUE(XTM_OPCODE | TDPBSSD_FUNCT3 |      \
                                            TMM_C(tmmC) | TMM_A(tmmA) |        \
                                            TMM_B(tmmB)) "\n\t");              \
  }

#define TDPBSSUD(tmmC, tmmA, tmmB)                                             \
  {                                                                            \
    __asm__ __volatile__(".word " GET_VALUE(XTM_OPCODE | TDPBSUD_FUNCT3 |      \
                                            TMM_C(tmmC) | TMM_A(tmmA) |        \
                                            TMM_B(tmmB)) "\n\t");              \
  }

#define TDPBUSD(tmmC, tmmA, tmmB)                                              \
  {                                                                            \
    __asm__ __volatile__(".word " GET_VALUE(XTM_OPCODE | TDPBUSD_FUNCT3 |      \
                                            TMM_C(tmmC) | TMM_A(tmmA) |        \
                                            TMM_B(tmmB)) "\n\t");              \
  }

#define TDPBUUD(tmmC, tmmA, tmmB)                                              \
  {                                                                            \
    __asm__ __volatile__(".word " GET_VALUE(XTM_OPCODE | TDPBUUD_FUNCT3 |      \
                                            TMM_C(tmmC) | TMM_A(tmmA) |        \
                                            TMM_B(tmmB)) "\n\t");              \
  }
