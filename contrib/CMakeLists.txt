# Gemmlowp CMake file written for Debian.
# Copyright © 2016 Zhou Mo <cdluminate@gmail.com>
# Licence Apache-2.0

cmake_minimum_required(VERSION 3.7)

# Project
project(gemmlowp C CXX)

include(CTest) # option(BUILD_TESTING). ON by default.
include(GNUInstallDirs)

# Set C++11 as default standard
set(CMAKE_CXX_STANDARD 11)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

get_filename_component(gemmlowp_src ${gemmlowp_SOURCE_DIR} PATH)

if(WIN32)
  # one can enable simd from the cmake command line, ie -DCMAKE_CXX_FLAGS="/arch:AVX2
  add_definitions(-DNOMINMAX -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI)
  add_definitions(/bigobj /nologo /EHsc /GF /MP /Gm- /wd4800 /wd4805 /wd4244)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # if we compile for windows with clang, allow inline asm
    add_definitions(-DGEMMLOWP_ALLOW_INLINE_ASM)
  endif()
else()
  set(EXTERNAL_LIBRARIES Threads::Threads)
endif()

# Glob header files
file(GLOB gemmlowp_private_headers "${gemmlowp_src}/fixedpoint/*.h" "${gemmlowp_src}/internal/*.h")
file(GLOB gemmlowp_public_headers "${gemmlowp_src}/meta/*.h" "${gemmlowp_src}/public/*.h" "${gemmlowp_src}/profiling/*.h")
list(APPEND gemmlowp_headers ${gemmlowp_private_headers} ${gemmlowp_public_headers})

file(GLOB eight_bit_int_gemm_headers "${gemmlowp_src}/eight_bit_int_gemm/*.h")
list(APPEND eight_bit_int_gemm_public_headers ${eight_bit_int_gemm_headers} ${gemmlowp_public_headers})
file(GLOB eight_bit_int_gemm_sources_with_no_headers "${gemmlowp_src}/eight_bit_int_gemm/*.cc")

list(APPEND eight_bit_int_gemm_sources
            ${eight_bit_int_gemm_headers}
            ${eight_bit_int_gemm_sources_with_no_headers}
            ${gemmlowp_headers})

file(GLOB gemmlowp_test_headers "${gemmlowp_src}/test/*.h")
list(APPEND gemmlowp_test_headers ${gemmlowp_headers})

file(GLOB fixedpoint_private_headers "${gemmlowp_src}/fixedpoint/*.h")
list(APPEND fixedpoint_private_headers "${gemmlowp_src}/internal/common.h")

add_library(eight_bit_int_gemm ${eight_bit_int_gemm_sources_with_no_headers})
set_target_properties(eight_bit_int_gemm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_link_libraries(eight_bit_int_gemm ${EXTERNAL_LIBRARIES})

# INTERFACE target to help header include
add_library(gemmlowp INTERFACE)
target_include_directories(gemmlowp INTERFACE
    $<BUILD_INTERFACE:${gemmlowp_src}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/gemmlowp>)
target_link_libraries(gemmlowp INTERFACE eight_bit_int_gemm)

install(FILES ${eight_bit_int_gemm_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gemmlowp/eight_bit_int_gemm)
file(GLOB meta_headers "${gemmlowp_src}/meta/*.h")
install(FILES ${meta_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gemmlowp/meta)
file(GLOB public_headers "${gemmlowp_src}/public/*.h")
install(FILES ${public_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gemmlowp/public)
file(GLOB profile_headers "${gemmlowp_src}/profiling/*.h")
install(FILES ${profile_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gemmlowp/profiling)
file(GLOB internal_headers "${gemmlowp_src}/internal/*.h")
install(FILES ${internal_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gemmlowp/internal)
file(GLOB fixedpoint_headers "${gemmlowp_src}/fixedpoint/*.h")
install(FILES ${fixedpoint_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gemmlowp/fixedpoint)

install(TARGETS gemmlowp eight_bit_int_gemm
        EXPORT  gemmlowp-config # support find_package(gemmlowp CONFIG)
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT  gemmlowp-config  # export gemmlowp::gemmlowp
        NAMESPACE gemmlowp::     #        gemmlowp::eight_bit_int_gemm
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gemmlowp)

if(BUILD_TESTING)
    # Benchmarks
    add_executable(benchmark
        "${gemmlowp_src}/test/benchmark.cc" ${gemmlowp_test_headers})
    target_link_libraries(benchmark ${EXTERNAL_LIBRARIES})
    
    add_executable(benchmark_all_sizes
        "${gemmlowp_src}/test/benchmark_all_sizes.cc" ${gemmlowp_test_headers})
    target_compile_options(benchmark_all_sizes PRIVATE -DBENCHMARK_8bit -DBENCHMARK_QUICK)
    target_link_libraries(benchmark_all_sizes ${EXTERNAL_LIBRARIES})
    
    # Gemmlowp test
    add_executable(test_gemmlowp
        "${gemmlowp_src}/test/test.cc" "${gemmlowp_src}/test/test_data.cc" ${gemmlowp_test_headers})
    target_link_libraries(test_gemmlowp eight_bit_int_gemm)
    
    # Math helpers test
    add_executable(test_math_helpers
        "${gemmlowp_src}/test/test_math_helpers.cc" ${gemmlowp_test_headers})
    
    # BlockingCounter test
    add_executable(test_blocking_counter
        "${gemmlowp_src}/test/test_blocking_counter.cc" ${gemmlowp_test_headers})
    target_link_libraries(test_blocking_counter ${EXTERNAL_LIBRARIES})
    
    # Allocator test
    add_executable(test_allocator
        "${gemmlowp_src}/test/test_allocator.cc" ${gemmlowp_test_headers})
    
    # FixedPoint test
    add_executable(test_fixedpoint
        "${gemmlowp_src}/test/test_fixedpoint.cc" ${gemmlowp_test_headers})
    
    # Add tests
    enable_testing()
    foreach(testname "test_math_helpers" "test_blocking_counter" "test_allocator" "test_fixedpoint" "test_gemmlowp")
        add_test(NAME ${testname} COMMAND "${testname}")
    endforeach(testname)
endif()
