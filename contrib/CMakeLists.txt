# Gemmlowp CMake file written for Debian.
# Copyright © 2016 Zhou Mo <cdluminate@gmail.com>
# Licence Apache-2.0
cmake_minimum_required(VERSION 3.7)

# Project
project(gemmlowp C CXX)

# Set C++11 as default standard
set(CMAKE_CXX_STANDARD 11)
# Enabling SIMD is recommended for modern x86 machines
#set(CMAKE_CXX_FLAGS "-msse4")
# However here we take the best compatibility
add_definitions(-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK)

# Glob header files
file(GLOB gemmlowp_private_headers "fixedpoint/*.h" "internal/*.h")
file(GLOB gemmlowp_public_headers "meta/*.h" "public/*.h" "profiling/*.h")
list(APPEND gemmlowp_headers ${gemmlowp_private_headers} ${gemmlowp_public_headers})

file(GLOB eight_bit_int_gemm_headers "eight_bit_int_gemm/*.h")
list(APPEND eight_bit_int_gemm_public_headers ${eight_bit_int_gemm_headers} ${gemmlowp_public_headers})
file(GLOB eight_bit_int_gemm_sources_with_no_headers "eight_bit_int_gemm/*.cc")

list(APPEND eight_bit_int_gemm_sources
            ${eight_bit_int_gemm_headers}
            ${eight_bit_int_gemm_sources_with_no_headers}
            ${gemmlowp_headers})

file(GLOB gemmlowp_test_headers "test/*.h")
list(APPEND gemmlowp_test_headers ${gemmlowp_headers})

file(GLOB fixedpoint_private_headers "fixedpoint/*.h")
list(APPEND fixedpoint_private_headers "internal/common.h")

# Eight bit int gemm library
add_library("eight_bit_int_gemm" SHARED ${eight_bit_int_gemm_sources_with_no_headers})
target_link_libraries("eight_bit_int_gemm" "pthread")

# Gemmlowp test
add_executable("test_gemmlowp" "test/test.cc" "test/test_data.cc" ${gemmlowp_test_headers})
target_link_libraries("test_gemmlowp" "eight_bit_int_gemm")

# Math helpers test
add_executable("test_math_helpers" "test/test_math_helpers.cc" ${gemmlowp_test_headers})

# BlockingCounter test
add_executable("test_blocking_counter" "test/test_blocking_counter.cc" ${gemmlowp_test_headers})
target_link_libraries("test_blocking_counter" "pthread")

# Allocator test
add_executable("test_allocator" "test/test_allocator.cc" ${gemmlowp_test_headers})

# FixedPoint test
add_executable("test_fixedpoint" "test/test_fixedpoint.cc" ${gemmlowp_test_headers})

# Add tests
enable_testing()
foreach(testname "test_math_helpers" "test_blocking_counter" "test_allocator" "test_fixedpoint" "test_gemmlowp")
  add_test(NAME ${testname} COMMAND "${testname}")
endforeach(testname)
