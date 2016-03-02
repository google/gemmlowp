#
# Description:
#   gemmlowp is a small self-contained low-precision GEMM library.
#   https://github.com/google/gemmlowp

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("flags", "LIB_LINKOPTS", "BIN_LINKOPTS")

gemmlowp_private_headers = glob([
    "internal/*.h",
    "meta/*.h",
])

gemmlowp_public_headers = glob([
    "public/*.h",
    "profiling/*.h",
])

filegroup(
    name = "gemmlowp_private_headers",
    srcs = gemmlowp_private_headers,
    visibility = ["//visibility:private"],
)

filegroup(
    name = "gemmlowp_public_headers",
    srcs = gemmlowp_public_headers,
    visibility = ["//visibility:public"],
)

gemmlowp_headers = gemmlowp_private_headers + gemmlowp_public_headers

filegroup(
    name = "gemmlowp_headers",
    srcs = gemmlowp_headers,
    visibility = ["//visibility:private"],
)

eight_bit_int_gemm_headers = glob([
    "eight_bit_int_gemm/*.h",
])

eight_bit_int_gemm_public_headers = eight_bit_int_gemm_headers + gemmlowp_public_headers

filegroup(
    name = "eight_bit_int_gemm_public_headers",
    srcs = eight_bit_int_gemm_public_headers,
    visibility = ["//visibility:public"],
)

eight_bit_int_gemm_sources_with_no_headers = glob([
    "eight_bit_int_gemm/*.cc",
])

eight_bit_int_gemm_sources_and_headers = eight_bit_int_gemm_sources_with_no_headers + eight_bit_int_gemm_headers + gemmlowp_headers

filegroup(
    name = "eight_bit_int_gemm_sources_with_no_headers",
    srcs = eight_bit_int_gemm_sources_with_no_headers,
    visibility = ["//visibility:private"],
)

filegroup(
    name = "eight_bit_int_gemm_sources",
    srcs = eight_bit_int_gemm_sources_and_headers,
    visibility = ["//visibility:public"],
)

gemmlowp_test_headers = glob([
    "test/*.h",
]) + gemmlowp_headers

filegroup(
    name = "gemmlowp_test_headers",
    srcs = gemmlowp_test_headers,
    visibility = ["//visibility:private"],
)

cc_library(
    name = "gemmlowp",
    srcs = [
        ":gemmlowp_headers",
    ],
    hdrs = [
        ":gemmlowp_public_headers",
    ],
    linkopts = LIB_LINKOPTS,
    # Blaze warning:
    # "setting 'linkstatic=1' is recommended if there are no object files."
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "eight_bit_int_gemm",
    srcs = [
        ":eight_bit_int_gemm_sources_with_no_headers",
    ],
    hdrs = [
        ":eight_bit_int_gemm_public_headers",
        ":gemmlowp_headers",
    ],
    linkopts = LIB_LINKOPTS,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":gemmlowp",
    ],
)

# The main gemmlowp unit test
cc_test(
    name = "test",
    size = "medium",
    srcs = [
        "test/test.cc",
        "test/test_data.cc",
        ":gemmlowp_test_headers",
    ],
    copts = [
        "-O3",
    ],
    deps = [
        ":eight_bit_int_gemm",
    ],
)

# Math helpers test
cc_test(
    name = "test_math_helpers",
    size = "small",
    srcs = [
        "test/test_math_helpers.cc",
        ":gemmlowp_test_headers",
    ],
)

# BlockingCounter test
cc_test(
    name = "test_blocking_counter",
    size = "medium",
    srcs = [
        "test/test_blocking_counter.cc",
        ":gemmlowp_test_headers",
    ],
    linkopts = BIN_LINKOPTS,
)

# Allocator test
cc_test(
    name = "test_allocator",
    size = "small",
    srcs = [
        "test/test_allocator.cc",
        ":gemmlowp_test_headers",
    ],
)

# FixedPoint test
cc_test(
    name = "test_fixedpoint",
    size = "small",
    srcs = [
        "test/test_fixedpoint.cc",
        ":gemmlowp_test_headers",
    ],
)

# Benchmark
cc_binary(
    name = "benchmark",
    srcs = [
        "test/benchmark.cc",
        ":gemmlowp_test_headers",
    ],
    copts = [
        "-O3",
        "-DNDEBUG",
    ],
    linkopts = BIN_LINKOPTS,
)

# Benchmark
cc_binary(
    name = "benchmark_profile",
    srcs = [
        "test/benchmark.cc",
        ":gemmlowp_test_headers",
    ],
    copts = [
        "-O3",
        "-DNDEBUG",
        "-DGEMMLOWP_TEST_PROFILE",
    ],
    linkopts = BIN_LINKOPTS,
)
