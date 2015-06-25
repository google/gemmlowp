gemmlowp: a small self-contained low-precision GEMM library
===========================================================

This is not a full linear algebra library, only a GEMM library: it only does
general matrix multiplication ("GEMM").


Disclaimer
==========

This is not an official Google product (experimental or otherwise), it is just
code that happens to be owned by Google.


Portability, target platforms/architectures
===========================================

Should be portable to any platform with some C++11 and POSIX support,
while we have optional optimized code paths for specifica architectures.

Required:
  C++11 (a small conservative subset of it)

Required for some features:
  * Some POSIX interfaces:
    * pthreads (for multi-threaded operation and for profiling).
    * sysconf (for multi-threaded operation to detect number of cores;
               may be bypassed).

Optional optimized code paths:
At the moment, we only have optimized code paths for ARM NEON SIMD.
Some are written in inline assembly, some are written in C++ using
intrinsics. Both GCC and Clang are supported.


Public interfaces
=================

1. gemmlowp public interface
----------------------------

  gemmlowp's main public interface is in the public/ subdirectory. The
  header to include is
      public/gemmlowp.h.
  This is a headers-only library, so there is nothing to link to.

2. EightBitIntGemm standard interface
-------------------------------------

  Additionally, the eight_bit_int_gemm/ subdirectory provides an
  implementation of the standard EightBitIntGemm interface. The header
  to include is
    eight_bit_int_gemm/eight_bit_int_gemm.h
  This is *NOT* a headers-only library, users need to link to
    eight_bit_int_gemm/eight_bit_int_gemm.cc.


Testing
=======

The test/ directory contains unit tests. The primary unit test is
  test/test.cc
Since it covers also the EightBitIntGemm interface, it needs to be
linked against
  eight_bit_int_gemm/eight_bit_int_gemm.cc

The scripts/ directory contains a script to build and run a program
on an Android device:
  scripts/test-android.sh

It expects the CXX environment variable to point to an Android toolchain's
C++ compiler, and expects source files (and optionally, cflags) as
command-line parameters. To build and run the above-mentioned main unit test,
first set CXX e.g.:

$ export CXX=/some/toolchains/arm-linux-androideabi-4.8/bin/arm-linux-androideabi-g++

Then run:

$ ./scripts/test-android.sh test/test.cc eight_bit_int_gemm/eight_bit_int_gemm.cc


Profiling
=========

The profiling/ subdirectory offers a very simple non-interrupting sampling
profiler that only requires pthreads (no signals).

It relies on source code being instrumented with pseudo-stack labels.
See profiling/instrumentation.h.
A full example of using this profiler is given in profiling/profiler.h.


Low-precision?
==============

"Low-precision" means that the input and output matrix entries are integers
on at most 8 bits. The scalar type is uint8_t.

This isn't the same as just doing plain matrix arithmetic over uint8_t,
because that would overflow. To avoid overflow, we internally accumulate
results on more than 8 bits, and at the end we keep only some significant
8 bits. This relies on the caller providing suitable offset/multiplier/shift
parameters, which effectively govern how we extract some significant 8 bit
from our more-than-8bit temporary accumulators. See the extra function
parameters taken by Gemm() in public/gemmlowp.h or by EightBitIntGemm() in
eight_bit_int_gemm/eight_bit_int_gemm.h.


Performance goals
============================

Our performance goals differ from typical GEMM performance goals in the
following ways:

1. We care not only about speed, but also about minimizing power usage.
   We specifically care about charge usage in mobile/embedded devices.
   This implies that we care doubly about minimizing memory bandwidth usage:
   we care about it, like any GEMM, because of the impact on speed, and we
   also care about it because it is a key factor of power usage.

2. Most GEMMs are optimized primarily for large dense matrix sizes (>= 1000).
   We do care about large sizes, but we also care specifically about the
   typically smaller matrix sizes encountered in various mobile applications.
   This means that we have to optimize for all sizes, not just for large enough
   sizes.
