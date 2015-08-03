#ifndef GEMMLOWP_TEST_TEST_DATA_H_
#define GEMMLOWP_TEST_TEST_DATA_H_

namespace test_data {

extern const bool is_a_transposed;
extern const bool is_b_transposed;
extern const bool is_c_transposed;
extern const int m;
extern const int n;
extern const int k;
extern const int a_offset;
extern const int b_offset;
extern const int c_shift;
extern const int c_mult_int;
extern const int c_shift;
extern const int c_offset;

extern const int a_count;
extern const int b_count;
extern const int c_count;

extern unsigned char a_data[];
extern unsigned char b_data[];
extern unsigned char expected_c_data[];

}  // namespace test_data

#endif  // GEMMLOWP_TEST_TEST_DATA_H
