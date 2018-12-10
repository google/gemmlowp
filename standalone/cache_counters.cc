#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#ifndef __aarch64__
#error This program is for 64-bit ARM only.
#endif

struct PerfEvent {
  perf_event_attr pe;
  int fd = -1;

  PerfEvent(std::uint32_t type, std::uint64_t config) {
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.type = type;
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (fd == -1) {
      fprintf(stderr, "perf_event_open failed for config 0x%lx\n", config);
      abort();
    }
  }

  void Start() {
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  }

  std::int64_t Stop() {
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    std::int64_t count = 0;
    read(fd, &count, sizeof(count));
    return count;
  }

  ~PerfEvent() { close(fd); }
};

struct ArmPmuEvent : PerfEvent {
  static constexpr std::uint16_t L1I_CACHE_REFILL = 0x01;
  static constexpr std::uint16_t L1I_TLB_REFILL = 0x02;
  static constexpr std::uint16_t L1D_CACHE_REFILL = 0x03;
  static constexpr std::uint16_t L1D_CACHE = 0x04;
  static constexpr std::uint16_t L1D_TLB_REFILL = 0x05;
  static constexpr std::uint16_t LD_RETIRED = 0x06;
  static constexpr std::uint16_t ST_RETIRED = 0x07;
  static constexpr std::uint16_t INST_RETIRED = 0x08;
  static constexpr std::uint16_t EXC_TAKEN = 0x09;
  static constexpr std::uint16_t EXC_RETURN = 0x0A;
  static constexpr std::uint16_t CID_WRITE_RETIRED = 0x0B;
  static constexpr std::uint16_t PC_WRITE_RETIRED = 0x0C;
  static constexpr std::uint16_t BR_IMMED_RETIRED = 0x0D;
  static constexpr std::uint16_t BR_RETURN_RETIRED = 0x0E;
  static constexpr std::uint16_t UNALIGNED_LDST_RETIRED = 0x0F;
  static constexpr std::uint16_t BR_MIS_PRED = 0x10;
  static constexpr std::uint16_t CPU_CYCLES = 0x11;
  static constexpr std::uint16_t BR_PRED = 0x12;
  static constexpr std::uint16_t MEM_ACCESS = 0x13;
  static constexpr std::uint16_t L1I_CACHE = 0x14;
  static constexpr std::uint16_t L1D_CACHE_WB = 0x15;
  static constexpr std::uint16_t L2D_CACHE = 0x16;
  static constexpr std::uint16_t L2D_CACHE_REFILL = 0x17;
  static constexpr std::uint16_t L2D_CACHE_WB = 0x18;
  static constexpr std::uint16_t BUS_ACCESS = 0x19;
  static constexpr std::uint16_t MEMORY_ERROR = 0x1A;
  static constexpr std::uint16_t INST_SPEC = 0x1B;
  static constexpr std::uint16_t TTBR_WRITE_RETIRED = 0x1C;
  static constexpr std::uint16_t BUS_CYCLES = 0x1D;
  static constexpr std::uint16_t CHAIN = 0x1E;
  static constexpr std::uint16_t L1D_CACHE_ALLOCATE = 0x1F;
  static constexpr std::uint16_t L2D_CACHE_ALLOCATE = 0x20;
  static constexpr std::uint16_t BR_RETIRED = 0x21;
  static constexpr std::uint16_t BR_MIS_PRED_RETIRED = 0x22;
  static constexpr std::uint16_t STALL_FRONTEND = 0x23;
  static constexpr std::uint16_t STALL_BACKEND = 0x24;
  static constexpr std::uint16_t L1D_TLB = 0x25;
  static constexpr std::uint16_t L1I_TLB = 0x26;
  static constexpr std::uint16_t L2I_CACHE = 0x27;
  static constexpr std::uint16_t L2I_CACHE_REFILL = 0x28;
  static constexpr std::uint16_t L3D_CACHE_ALLOCATE = 0x29;
  static constexpr std::uint16_t L3D_CACHE_REFILL = 0x2A;
  static constexpr std::uint16_t L3D_CACHE = 0x2B;
  static constexpr std::uint16_t L3D_CACHE_WB = 0x2C;
  static constexpr std::uint16_t L2D_TLB_REFILL = 0x2D;
  static constexpr std::uint16_t L2I_TLB_REFILL = 0x2E;
  static constexpr std::uint16_t L2D_TLB = 0x2F;
  static constexpr std::uint16_t L2I_TLB = 0x30;
  static constexpr std::uint16_t LL_CACHE = 0x32;
  static constexpr std::uint16_t LL_CACHE_MISS = 0x33;
  static constexpr std::uint16_t DTLB_WALK = 0x34;
  static constexpr std::uint16_t LL_CACHE_RD = 0x36;
  static constexpr std::uint16_t LL_CACHE_MISS_RD = 0x37;
  static constexpr std::uint16_t L1D_CACHE_RD = 0x40;
  static constexpr std::uint16_t L1D_CACHE_REFILL_RD = 0x42;
  static constexpr std::uint16_t L1D_TLB_REFILL_RD = 0x4C;
  static constexpr std::uint16_t L1D_TLB_RD = 0x4E;
  static constexpr std::uint16_t L2D_CACHE_RD = 0x50;
  static constexpr std::uint16_t L2D_CACHE_REFILL_RD = 0x52;
  static constexpr std::uint16_t BUS_ACCESS_RD = 0x60;
  static constexpr std::uint16_t MEM_ACCESS_RD = 0x66;
  static constexpr std::uint16_t L3D_CACHE_RD = 0xA0;
  static constexpr std::uint16_t L3D_CACHE_REFILL_RD = 0xA2;
  ArmPmuEvent(std::uint16_t number) : PerfEvent(PERF_TYPE_RAW, number) {}
};

struct CacheCounts {
  int ld_retired = 0;
  int mem_access = 0;
  int ll_cache = 0;
  int ll_cache_miss = 0;
  int l1d_cache = 0;
  int l1d_cache_refill = 0;
  int l2d_cache = 0;
  int l2d_cache_refill = 0;
  int l3d_cache = 0;
  int l3d_cache_refill = 0;
};

void PrintCacheCounts(const CacheCounts& cache_counts) {
  printf("ld_retired = %d\n", cache_counts.ld_retired);
  printf("mem_access = %d\n", cache_counts.mem_access);
  printf("ll_cache = %d\n", cache_counts.ll_cache);
  printf("ll_cache_miss = %d\n", cache_counts.ll_cache_miss);
  printf("l1d_cache = %d\n", cache_counts.l1d_cache);
  printf("l1d_cache_refill = %d\n", cache_counts.l1d_cache_refill);
  printf("l2d_cache = %d\n", cache_counts.l2d_cache);
  printf("l2d_cache_refill = %d\n", cache_counts.l2d_cache_refill);
  printf("l3d_cache = %d\n", cache_counts.l3d_cache);
  printf("l3d_cache_refill = %d\n", cache_counts.l3d_cache_refill);
}

void Workload(int accesses, int size, std::uint8_t* buf) {
  // The main reason to do this in assembly is an attempt to make sense
  // of instruction count counters, such as LD_RETIRED.
  // Also, if we did this in C++, we would need to be watchful of the compiler
  // optimizing away operations whose result isn't consumed.
  //
  // Note that TWO separate tricks are needed here to prevent Cortex-A76
  // speculative execution om prefetching data from future loop iterations:
  //   1. A data-dependency whereby the pointers being dereferenced at the
  //      next loop iteration depend on values loaded at the current iteration.
  //      That is the role of 'dummy'.
  //   2. A pseudo-random sequence. This is the role of register w0,
  //      where we implement a simple xorshift pseudorandom generator.
  // BOTH of these tricks are needed: if we disable just one of them,
  // Cortex-A76 successfully speculates some addresses, resulting in different
  // L3 / DRAM hit percentages on large sizes.
  std::uint64_t dummy = 123456789;
  asm volatile(
      // w0 := xorshift RNG state. Must be nonzero.
      "mov w0, #1\n"
      "1:\n"
      // xorshift RNG iteration: update w0 with the next pseudorandom value
      // in [1 .. 2^32-1].
      // This pseudorandomness is crucial to preventing speculative execution
      // on Cortex-A76 from prefetching data from future loop iterations.
      "eor w0, w0, w0, lsl #13\n"
      "eor w0, w0, w0, lsr #17\n"
      "eor w0, w0, w0, lsl #5\n"
      // w1 := size - 1 = size mask (size is required to be power-of-two).
      "sub w1, %w[size], #1\n"
      // w2 := (pseudorandom value w0) xor (data-dependent sum).
      "eor w2, w0, %w[dummy]\n"
      // w1 := w2 modulo size
      "and w1, w2, w1\n"
      // align w1
      "and w1, w1, #-64\n"
      // load at offset w1, again using x1 as destination.
      "ldr x1, [%[buf], w1, uxtw]\n"
      // Update our dummy so it depends on the value we have just loaded.
      // This data-dependency is key to preventing speculative execution on
      // Cortex-A76 from prefetching data from future loop iterations.
      "add %[dummy], %[dummy], w1, uxtw\n"
      // loop back.
      "subs %w[accesses], %w[accesses], #1\n"
      "bne 1b\n"
      : [ accesses ] "+r"(accesses), [ dummy ] "+r"(dummy)
      : [ size ] "r"(size), [ buf ] "r"(buf)
      : "memory", "cc", "x0", "x1", "x2");
}

void MeasureCacheCounts(int accesses, int size, std::uint8_t* buf,
                        CacheCounts* cache_counts) {
  const bool only_reads = getenv("ONLY_READS");
  ArmPmuEvent ld_retired(ArmPmuEvent::LD_RETIRED);
  ArmPmuEvent mem_access(only_reads ? ArmPmuEvent::MEM_ACCESS_RD
                                    : ArmPmuEvent::MEM_ACCESS);
  ArmPmuEvent ll_cache(only_reads ? ArmPmuEvent::LL_CACHE_RD
                                  : ArmPmuEvent::LL_CACHE);
  ArmPmuEvent ll_cache_miss(only_reads ? ArmPmuEvent::LL_CACHE_MISS_RD
                                       : ArmPmuEvent::LL_CACHE_MISS);
  ArmPmuEvent l1d_cache(only_reads ? ArmPmuEvent::L1D_CACHE_RD
                                   : ArmPmuEvent::L1D_CACHE);
  ArmPmuEvent l1d_cache_refill(only_reads ? ArmPmuEvent::L1D_CACHE_REFILL_RD
                                          : ArmPmuEvent::L1D_CACHE_REFILL);
  ArmPmuEvent l2d_cache(only_reads ? ArmPmuEvent::L2D_CACHE_RD
                                   : ArmPmuEvent::L2D_CACHE);
  ArmPmuEvent l2d_cache_refill(only_reads ? ArmPmuEvent::L2D_CACHE_REFILL_RD
                                          : ArmPmuEvent::L2D_CACHE_REFILL);
  ArmPmuEvent l3d_cache(only_reads ? ArmPmuEvent::L3D_CACHE_RD
                                   : ArmPmuEvent::L3D_CACHE);
  ArmPmuEvent l3d_cache_refill(only_reads ? ArmPmuEvent::L3D_CACHE_REFILL_RD
                                          : ArmPmuEvent::L3D_CACHE_REFILL);

  ld_retired.Start();
  mem_access.Start();
  ll_cache.Start();
  ll_cache_miss.Start();
  l1d_cache.Start();
  l1d_cache_refill.Start();
  l2d_cache.Start();
  l2d_cache_refill.Start();
  l3d_cache.Start();
  l3d_cache_refill.Start();

  Workload(accesses, size, buf);

  cache_counts->ld_retired = ld_retired.Stop();
  cache_counts->mem_access = mem_access.Stop();
  cache_counts->ll_cache = ll_cache.Stop();
  cache_counts->ll_cache_miss = ll_cache_miss.Stop();
  cache_counts->l1d_cache = l1d_cache.Stop();
  cache_counts->l1d_cache_refill = l1d_cache_refill.Stop();
  cache_counts->l2d_cache = l2d_cache.Stop();
  cache_counts->l2d_cache_refill = l2d_cache_refill.Stop();
  cache_counts->l3d_cache = l3d_cache.Stop();
  cache_counts->l3d_cache_refill = l3d_cache_refill.Stop();
}

struct PieChart {
  // How many accesses were recorded, total? The other fields must sum to that.
  int total;
  // How many accesses were serviced with the typical cost of a L1 cache hit?
  int l1_hits;
  // How many accesses were serviced with the typical cost of a L2 cache hit?
  int l2_hits;
  // How many accesses were serviced with the typical cost of a L3 cache hit?
  int l3_hits;
  // How many accesses were serviced with the typical cost of a DRAM access?
  int dram_hits;

  ~PieChart() {
    // Consistency check
    if (total != l1_hits + l2_hits + l3_hits + dram_hits) {
      fprintf(stderr, "inconsistent pie-chart\n");
      abort();
    }
  }
};

struct Hypothesis {
  virtual ~Hypothesis() {}
  virtual const char* Name() const = 0;
  virtual void Analyze(const CacheCounts& cache_counts,
                       PieChart* pie) const = 0;
};

struct Hypothesis1 : Hypothesis {
  const char* Name() const override { return "Hypothesis1"; }
  void Analyze(const CacheCounts& cache_counts, PieChart* pie) const override {
    pie->total = cache_counts.l1d_cache + cache_counts.l1d_cache_refill;
    pie->l1_hits = cache_counts.l1d_cache - cache_counts.l2d_cache_refill -
                   cache_counts.l3d_cache_refill;
    pie->l2_hits = cache_counts.l1d_cache_refill;
    pie->l3_hits = cache_counts.l2d_cache_refill;
    pie->dram_hits = cache_counts.l3d_cache_refill;
  }
};

struct Hypothesis2 : Hypothesis {
  const char* Name() const override { return "Hypothesis2"; }
  void Analyze(const CacheCounts& cache_counts, PieChart* pie) const override {
    pie->total = cache_counts.l1d_cache;
    pie->l1_hits = cache_counts.l1d_cache - cache_counts.l2d_cache;
    pie->l2_hits = cache_counts.l2d_cache - cache_counts.l3d_cache;
    pie->l3_hits = cache_counts.l3d_cache - cache_counts.l3d_cache_refill;
    pie->dram_hits = cache_counts.l3d_cache_refill;
  }
};

struct Hypothesis3 : Hypothesis {
  const char* Name() const override { return "Hypothesis3"; }
  void Analyze(const CacheCounts& cache_counts, PieChart* pie) const override {
    pie->total = cache_counts.l1d_cache;
    int corrected_l2 = std::min(cache_counts.l2d_cache, cache_counts.l1d_cache);
    int corrected_l3 = std::min(cache_counts.l3d_cache, corrected_l2);
    pie->l1_hits = cache_counts.l1d_cache - corrected_l2;
    pie->l2_hits = corrected_l2 - corrected_l3;
    pie->l3_hits = corrected_l3 - cache_counts.l3d_cache_refill;
    pie->dram_hits = cache_counts.l3d_cache_refill;
  }
};

struct Hypothesis4 : Hypothesis {
  const char* Name() const override { return "Hypothesis4"; }
  void Analyze(const CacheCounts& cache_counts, PieChart* pie) const override {
    pie->total = cache_counts.l1d_cache;
    pie->l1_hits = cache_counts.l1d_cache - cache_counts.l1d_cache_refill;
    pie->l2_hits =
        cache_counts.l1d_cache_refill - cache_counts.l2d_cache_refill;
    pie->l3_hits =
        cache_counts.l2d_cache_refill - cache_counts.l3d_cache_refill;
    pie->dram_hits = cache_counts.l3d_cache_refill;
  }
};

struct Hypothesis5 : Hypothesis {
  const char* Name() const override { return "Hypothesis5"; }
  void Analyze(const CacheCounts& cache_counts, PieChart* pie) const override {
    pie->l1_hits =
        std::max(0, cache_counts.l1d_cache - cache_counts.l1d_cache_refill);
    pie->l2_hits = std::max(
        0, cache_counts.l1d_cache_refill - cache_counts.l2d_cache_refill);
    const int l3_misses =
        std::max(cache_counts.ll_cache_miss, cache_counts.l3d_cache_refill);
    pie->l3_hits = std::max(0, cache_counts.l2d_cache_refill - l3_misses);
    pie->dram_hits = l3_misses;
    pie->total = pie->l1_hits + pie->l2_hits + pie->l3_hits + pie->dram_hits;
  }
};

void PrintPieChart(const PieChart& pie) {
  printf("total accesses: %d\n", pie.total);
  double l1_hits_pct = 100. * pie.l1_hits / pie.total;
  double l2_hits_pct = 100. * pie.l2_hits / pie.total;
  double l3_hits_pct = 100. * pie.l3_hits / pie.total;
  double dram_hits_pct = 100. * pie.dram_hits / pie.total;
  printf("L1 hits: %.2f%%\n", l1_hits_pct);
  printf("L2 hits: %.2f%%\n", l2_hits_pct);
  printf("L1/2 hits: %.2f%%\n", l1_hits_pct + l2_hits_pct);
  printf("L3 hits: %.2f%%\n", l3_hits_pct);
  printf("L1/2/3 hits: %.2f%%\n", l1_hits_pct + l2_hits_pct + l3_hits_pct);
  printf("DRAM hits: %.2f%%\n", dram_hits_pct);
}

void PrintPieChartCsvNoNewline(const PieChart& pie) {
  double l1_hits_pct = 100. * pie.l1_hits / pie.total;
  double l2_hits_pct = 100. * pie.l2_hits / pie.total;
  double l3_hits_pct = 100. * pie.l3_hits / pie.total;
  double dram_hits_pct = 100. * pie.dram_hits / pie.total;
  printf("%.2f,%.2f,%.2f,%.2f", l1_hits_pct, l2_hits_pct, l3_hits_pct,
         dram_hits_pct);
}

void Study(int accesses, int size, std::uint8_t* buf) {
  CacheCounts cache_counts;
  MeasureCacheCounts(accesses, size, buf, &cache_counts);
  const Hypothesis* hypotheses[] = {
      new Hypothesis5, new Hypothesis4, new Hypothesis3,
      new Hypothesis2, new Hypothesis1,
  };
  if (getenv("DUMP_CSV")) {
    printf("%d", size);
    for (const Hypothesis* hypothesis : hypotheses) {
      printf(",");
      PieChart pie;
      hypothesis->Analyze(cache_counts, &pie);
      PrintPieChartCsvNoNewline(pie);
    }
    printf("\n");
  } else {
    printf("\n\n\naccesses=%d, size=%d:\n", accesses, size);
    printf("\nCache counts:\n");
    PrintCacheCounts(cache_counts);
    for (const Hypothesis* hypothesis : hypotheses) {
      printf("\n%s:\n", hypothesis->Name());
      PieChart pie;
      hypothesis->Analyze(cache_counts, &pie);
      PrintPieChart(pie);
    }
  }
  fflush(stdout);
  for (const Hypothesis* hypothesis : hypotheses) {
    delete hypothesis;
  }
}

int main() {
  const int kMinSize = 1 << 12;
  const int kMaxSize = 1 << 24;
  const int kAccesses = 1e8;
  void* buf_void = nullptr;
  posix_memalign(&buf_void, 64, kMaxSize);
  std::uint8_t* buf = static_cast<std::uint8_t*>(buf_void);
  std::default_random_engine random_engine;
  for (int i = 0; i < kMaxSize; i++) {
    buf[i] = random_engine();
  }
  for (int size = kMinSize; size <= kMaxSize; size *= 2) {
    Study(kAccesses, size, buf);
  }
  delete[] buf;
}
