## some strategy to optimize `TMUL` operations
(from the manual: Intel 64 and IA-32 Architectures Optimization Reference Manual)
1. Minimizing Tile Loads
* keep the K-dimension loop outside the M_ACC and N_ACC loops(减少 `tile_load` 的调用次数);
* Pre-Loading Innermost Loop Tiles(空间换时间，将acc_m和acc_n中外层循环的tile存储起来，避免重复从内存中加载);
* using 2D accumulator arrays is recommended. (Select dimensions close to square);
2. Software Pipelining of Tile Loads and Stores
* 避免连续的 `tile_load` 和 `tile_store`。将 `tile_load` 和 `tile_store` 插入到循环体中。

3. cpuid 查看AMX的硬件参数
```
CPU Info:
------------------
  vendor_str : `GenuineIntel'
  vendor id  : 0
  brand_str  : `INTEL(R) XEON(R) PLATINUM 8580'
  family     : 6 (06h)
  model      : 15 (0Fh)
  stepping   : 2 (02h)
  ext_family : 6 (06h)
  ext_model  : 207 (CFh)
  num_cores  : 60
  num_logical: 120
  tot_logical: 240
  L1 D cache : 48 KB
  L1 I cache : 32 KB
  L2 cache   : 2048 KB
  L3 cache   : 307200 KB
  L4 cache   : -1 KB
  L1D assoc. : 12-way
  L1I assoc. : 8-way
  L2 assoc.  : 16-way
  L3 assoc.  : 20-way
  L4 assoc.  : -1-way
  L1D line sz: 64 bytes
  L1I line sz: 64 bytes
  L2 line sz : 64 bytes
  L3 line sz : 64 bytes
  L4 line sz : -1 bytes
  SSE units  : 128 bits (non-authoritative)
  code name  : `Xeon (Conroe/2M)'
  features   : fpu vme de pse tsc msr pae mce cx8 apic mtrr sep pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe pni pclmul dts64 ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm dca sse4_1 sse4_2 syscall xd movbe popcnt aes xsave osxsave avx rdtscp lm lahf_lm abm constant_tsc fma3 f16c rdrand x2apic avx2 bmi1 bmi2 avx512f avx512dq avx512cd sha_ni avx512bw avx512vl sgx rdseed adx avx512vnni avx512vbmi avx512vbmi2
```