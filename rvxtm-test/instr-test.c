#include <stdint.h>

int mem_data = 0x12345678;
// uint32_t Xtm_insts[] = {
//   0x012fc08b,
//   0x012fd08b,
//   0x0800088b,
//   0x01f3420b,
//   0x01f3520b,
//   0x0c000a0b
// };

__attribute__((optimize("O0"))) void test_wrapper() {
  asm volatile("li t0, 0x9abc\n\t" // some regular riscv instrs
               "sub t1, t0, t0\n\t"
               "lw s1, 48(%0)\n\t"
               "xor t2, t1, t0\n\t"
               "or t3, t2, t0\n\t"
               "and t4, t3, t0\n\t"
               "sw s0, 48(%0)\n\t"
               "sll t5, t4, 2\n\t"
               "srl t6, t5, 1\n\t"

               ".insn 0x0039428b\n\t" // tileloadd tmm5, 0(x12), x3

               "li t0, 0x1234\n\t"
               "sub t2, t1, t0\n\t"
               "lw s1, 16(%0)\n\t"
               "xor t3, t2, t0\n\t"
               "or t4, t3, t0\n\t"
               "and t5, t4, t0\n\t"
               "sw s0, 16(%0)\n\t"
               "sll t6, t5, 2\n\t"
               "addi t1, t0, -1\n\t"

               ".insn 0x0039628b\n\t" // tilestored tmm5, 0(x12), x3

               "li t0, 0x5678\n\t"
               "add t1, t0, t0\n\t"
               "lw s1, 32(%0)\n\t"
               "xor t2, t1, t0\n\t"
               "or t3, t2, t0\n\t"
               "and t4, t3, t0\n\t"
               "sw s0, 32(%0)\n\t"
               "sll t5, t4, 2\n\t"
               "srl t6, t5, 1\n\t"

               ".insn 0x08003e8b\n\t" // tdpbssd tmm5, tmm7, tmm0
               :
               : "r"(&mem_data)
               : "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
}

int main() {
  test_wrapper();
  return 0;
}
