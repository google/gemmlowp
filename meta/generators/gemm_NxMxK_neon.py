"""Generates the whole gemm header.

"""

import cc_emitter
import mul_Nx8_Mx8_neon
import neon_emitter
import qnt_Nx8_neon
import zip_Nx8_neon

_HEADER_COPYRIGHT = """// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// single_thread_gemm.h: programatically generated GEMM library header.
"""


def GenerateTempsCountersAndConsts(emitter, rows):
  """Emits constants and variables declarations."""
  emitter.EmitCode('const std::int32_t row_chunks = n / 3')
  emitter.EmitCode('const std::int32_t col_chunks = m / 3')
  emitter.EmitCode('const std::int32_t padded_k = ((k + 7) / 8) * 8')
  emitter.EmitNewline()

  emitter.EmitCode('const std::int32_t chunk_size = k * 3')
  emitter.EmitCode('const std::int32_t zipped_chunk_size = (padded_k + 16) * 3')
  emitter.EmitCode('const std::int32_t zipped_rhs_size = (padded_k + 16) * m')
  emitter.EmitCode(
      'const std::int32_t temp_result_stride = ((m * 4 + 7) / 8) * 8')
  emitter.EmitCode(
      'const std::int32_t temp_result_size = 3 * temp_result_stride')
  emitter.EmitCode('const std::int32_t rounding_offset = (1 << (shift - 1))')
  emitter.EmitCode('const std::int32_t result_chunk_stride = result_stride * 3')
  emitter.EmitNewline()

  emitter.EmitCode('std::uint8_t* zipped_lhs = scratch')
  emitter.EmitCode('std::int32_t* zipped_lhs_3_offsets = '
                   'reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3)')
  leftover_rows = rows % 3
  if leftover_rows:
    emitter.EmitCode(
        ('std::int32_t* zipped_lhs_%d_offsets = '
         'reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * %d)') % (
             leftover_rows, leftover_rows))

  emitter.EmitCode('std::uint8_t* zipped_rhs = scratch + zipped_chunk_size')
  emitter.EmitCode(
      'std::int32_t* temp_result = reinterpret_cast<std::int32_t*>('
      'scratch + zipped_chunk_size + zipped_rhs_size)')
  emitter.EmitNewline()

  emitter.EmitCode('const std::uint8_t* lhs_chunk = lhs')
  emitter.EmitCode('const std::uint8_t* rhs_chunk = rhs')
  emitter.EmitCode('std::uint8_t* zipped_rhs_chunk = zipped_rhs')
  emitter.EmitCode('std::int32_t* temp_result_chunk = temp_result')
  emitter.EmitCode('std::uint8_t* result_chunk = result')
  emitter.EmitNewline()
  emitter.EmitCode('const std::int32_t const_offset = '
                   'lhs_offset * rhs_offset * k + result_offset')


def ZipName(rows, leftovers, aligned):
  return zip_Nx8_neon.BuildName(rows, leftovers, aligned)


def GenerateZipRhs(emitter, aligned, cols, leftovers):
  """Emits the code responsible for zipping the rhs matrix."""
  emitter.EmitOpenBracket('for (int i = 0; i < col_chunks; ++i)')
  emitter.EmitCode(
      '%s(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0)' % ZipName(
          3, leftovers, aligned))
  emitter.EmitCode('rhs_chunk += chunk_size')
  emitter.EmitCode('zipped_rhs_chunk += zipped_chunk_size')
  emitter.EmitCloseBracket()

  leftover_cols = cols % 3

  if leftover_cols:
    emitter.EmitCode(
        '%s(rhs_chunk, k, k, zipped_rhs_chunk, lhs_offset, 0)' % ZipName(
            leftover_cols, leftovers, aligned))
  emitter.EmitNewline()


def MulName(rows, cols):
  return mul_Nx8_Mx8_neon.BuildName(rows, cols)


def GenerateMulRows(emitter, aligned, rows, cols, leftover):
  """Emits code responsible for multiplication of one horizontal lhs strip."""
  emitter.EmitCode(
      '%s(lhs_chunk, k, k, zipped_lhs, rhs_offset, const_offset)' % ZipName(
          rows, leftover, aligned))

  emitter.EmitCode('zipped_rhs_chunk = zipped_rhs')
  emitter.EmitCode('temp_result_chunk = temp_result')
  emitter.EmitOpenBracket('for (int j = 0; j < col_chunks; ++j)')

  emitter.EmitCode(
      ('%s(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk, '
       'temp_result_stride)') % MulName(rows, 3))
  emitter.EmitCode('zipped_rhs_chunk += zipped_chunk_size')
  emitter.EmitCode('temp_result_chunk += 3')

  emitter.EmitCloseBracket()

  leftover_cols = cols % 3
  if leftover_cols:
    emitter.EmitCode(
        ('%s(zipped_lhs, zipped_rhs_chunk, padded_k, temp_result_chunk, '
         'temp_result_stride)') % MulName(rows, leftover_cols))

  emitter.EmitCode(
      ('%s(temp_result, m, temp_result_stride, zipped_lhs_%d_offsets, '
       'result_chunk, result_stride, multiplicative_offset, rounding_offset, '
       '-shift)') % (BuildMultiQuantizeName(aligned, rows), rows))


def GenerateMul(emitter, aligned, rows, cols, leftover):
  """Emits code for all horizontal lhs strips, plus leftover rows."""
  emitter.EmitOpenBracket('for (int i = 0; i < row_chunks; ++i)')

  GenerateMulRows(emitter, aligned, 3, cols, leftover)
  emitter.EmitCode('lhs_chunk += chunk_size')
  emitter.EmitCode('result_chunk += result_chunk_stride')

  emitter.EmitCloseBracket()
  emitter.EmitNewline()

  leftover_rows = rows % 3
  if leftover_rows:
    GenerateMulRows(emitter, aligned, leftover_rows, cols, leftover)


def BuildName(aligned, rows, cols, leftover):
  name = 'gemm_%d_%d_%d' % (rows, cols, leftover)
  if aligned:
    name = '%s_aligned' % name
  return name


def GenerateGemm(emitter, aligned, rows, cols, leftover):
  """Build one gemm function for given row, col, and depth leftovers."""
  name = BuildName(aligned, rows, cols, leftover)

  emitter.EmitFunctionBeginA(name,
                             [['std::uint8_t*', 'scratch'],
                              ['const std::uint8_t*', 'lhs'],
                              ['const std::uint8_t*', 'rhs'],
                              ['std::int32_t', 'n'],
                              ['std::int32_t', 'm'],
                              ['std::int32_t', 'k'],
                              ['std::int32_t', 'lhs_offset'],
                              ['std::int32_t', 'rhs_offset'],
                              ['std::int32_t', 'result_offset'],
                              ['std::int32_t', 'multiplicative_offset'],
                              ['std::int32_t', 'shift'],
                              ['std::uint8_t*', 'result'],
                              ['std::int32_t', 'result_stride']],
                             'void')
  emitter.EmitAssert('n %% 3 == %d' % rows)
  emitter.EmitAssert('m %% 3 == %d' % cols)
  emitter.EmitAssert('k %% 8 == %d' % leftover)

  GenerateTempsCountersAndConsts(emitter, rows)
  GenerateZipRhs(emitter, aligned, cols, leftover)
  GenerateMul(emitter, aligned, rows, cols, leftover)
  emitter.EmitFunctionEnd()


def BuildMultiQuantizeName(aligned, rows):
  name = 'multi_qnt_%dx8' % rows
  if aligned:
    name = '%s_aligned' % name
  return name


def GenerateMultiQuantize(emitter, aligned, rows):
  """Emit main quantization code that switches between optimized versions."""
  name = BuildMultiQuantizeName(aligned, rows)
  emitter.EmitFunctionBeginA(name,
                             [['const std::int32_t*', 'source'],
                              ['std::int32_t', 'count'],
                              ['std::int32_t', 'stride'],
                              ['const std::int32_t*', 'offsets'],
                              ['std::uint8_t*', 'destination'],
                              ['std::int32_t', 'destination_stride'],
                              ['std::int32_t', 'multiplicative_offset'],
                              ['std::int32_t', 'rounding_offset'],
                              ['std::int32_t', 'shift']],
                             'void')
  emitter.EmitSwitch('count % 8')

  for i in range(0, 8):
    emitter.EmitCase(i)
    emitter.PushIndent()

    called_name = qnt_Nx8_neon.BuildName(rows, i, aligned)
    emitter.EmitCode(
        ('%s(source, count, stride, offsets, destination, destination_stride, '
         'multiplicative_offset, rounding_offset, shift)') % called_name)
    emitter.EmitBreak()
    emitter.PopIndent()

  emitter.EmitSwitchEnd()
  emitter.EmitFunctionEnd()


def GenerateGemmSwitch3(emitter, aligned, n_mod, m_mod):
  """Third level of main switch, choose optimized version on depth leftover."""
  emitter.EmitSwitch('k % 8')

  for i in range(0, 8):
    emitter.EmitCase(i)
    emitter.PushIndent()
    emitter.EmitCode(
        ('internal::%s(scratch, lhs, rhs, n, m, k, lhs_offset, rhs_offset, '
         'result_offset, multiplicative_offset, shift, result, result_stride)') % (BuildName(aligned, n_mod, m_mod, i)))
    emitter.EmitBreak()
    emitter.PopIndent()

  emitter.EmitSwitchEnd()


def GenerateGemmSwitch2(emitter, aligned, n_mod):
  """Second level of main switch, choose optimized version on cols leftover."""
  emitter.EmitSwitch('m % 3')

  for i in range(0, 3):
    emitter.EmitCase(i)
    emitter.PushIndent()
    GenerateGemmSwitch3(emitter, aligned, n_mod, i)
    emitter.EmitBreak()
    emitter.PopIndent()

  emitter.EmitSwitchEnd()


def GenerateGemmSwitch1(emitter, aligned):
  """First level of main switch, choose optimized version on rows leftover."""
  emitter.EmitSwitch('n % 3')

  for i in range(0, 3):
    emitter.EmitCase(i)
    emitter.PushIndent()
    GenerateGemmSwitch2(emitter, aligned, i)
    emitter.EmitBreak()
    emitter.PopIndent()

  emitter.EmitSwitchEnd()


def GetCommonGemmParameters():
  return [['std::uint8_t*', 'scratch'],
          ['const std::uint8_t*', 'lhs'],
          ['const std::uint8_t*', 'rhs'],
          ['std::int32_t', 'n'],
          ['std::int32_t', 'm'],
          ['std::int32_t', 'k'],
          ['std::int32_t', 'lhs_offset'],
          ['std::int32_t', 'rhs_offset'],
          ['std::int32_t', 'result_offset'],
          ['std::int32_t', 'multiplicative_offset'],
          ['std::int32_t', 'shift'],
          ['std::uint8_t*', 'result']]


def GenerateMainGemmFunction(emitter):
  """Emit high level gemm function that switches between optimized versions."""
  params = GetCommonGemmParameters()
  params.append(['std::int32_t', 'result_stride'])

  emitter.EmitFunctionBeginA('gemm_strided', params, 'void')

  emitter.EmitCode('const bool lhs_aligned = '
                   '((reinterpret_cast<std::uintptr_t>(lhs) % 8) == 0)')
  emitter.EmitCode('const bool rhs_aligned = '
                   '((reinterpret_cast<std::uintptr_t>(rhs) % 8) == 0)')
  emitter.EmitCode('const bool result_aligned = '
                   '((reinterpret_cast<std::uintptr_t>(result) % 8) == 0)')
  emitter.EmitCode('const bool k_aligned = ((k % 8) == 0)')
  emitter.EmitCode(
      'const bool result_stride_aligned = ((result_stride % 8) == 0)')
  emitter.EmitCode(
      'const bool aligned = lhs_aligned && rhs_aligned && result_aligned && '
      'k_aligned && result_stride_aligned')

  emitter.EmitIf('aligned')
  GenerateGemmSwitch1(emitter, True)
  emitter.EmitElse()
  GenerateGemmSwitch1(emitter, False)
  emitter.EmitEndif()
  emitter.EmitFunctionEnd()


def GenerateWrapperGemmFunctions(emitter):
  params = GetCommonGemmParameters()
  emitter.EmitFunctionBeginA('gemm', params, 'void')
  emitter.EmitCode('gemm_strided(scratch, lhs, rhs, n, m, k, lhs_offset, '
                   'rhs_offset, result_offset, multiplicative_offset, shift, '
                   'result, m)')
  emitter.EmitFunctionEnd()


def GenerateFunctions(emitter):

  for aligned in [True, False]:
    for rows in range(1, 4):
      GenerateMultiQuantize(emitter, aligned, rows)
      emitter.EmitNewline()

  for aligned in [True, False]:
    for rows in range(0, 3):
      for cols in range(0, 3):
        for leftover in range(0, 8):
          GenerateGemm(emitter, aligned, rows, cols, leftover)
          emitter.EmitNewline()


def Main():
  emitter = cc_emitter.CCEmitter()

  emitter.EmitCodeNoSemicolon(_HEADER_COPYRIGHT)
  emitter.EmitHeaderBegin('gemmlowp_meta_single_thread_gemm')

  emitter.EmitPreprocessor1('ifdef', 'GEMMLOWP_NEON_32')
  emitter.EmitNewline()

  emitter.EmitInclude('<cassert>')
  emitter.EmitNewline()

  emitter.EmitNamespaceBegin('gemmlowp')
  emitter.EmitNamespaceBegin('meta')
  emitter.EmitNamespaceBegin('internal')
  emitter.EmitNewline()

  zip_Nx8_neon.GenerateFunctions(neon_emitter.NeonEmitter())
  emitter.EmitNewline()

  mul_Nx8_Mx8_neon.GenerateFunctions(neon_emitter.NeonEmitter())
  emitter.EmitNewline()

  qnt_Nx8_neon.GenerateFunctions(neon_emitter.NeonEmitter())
  emitter.EmitNewline()

  GenerateFunctions(emitter)

  emitter.EmitNewline()
  emitter.EmitNamespaceEnd()
  emitter.EmitNewline()

  GenerateMainGemmFunction(emitter)
  emitter.EmitNewline()

  GenerateWrapperGemmFunctions(emitter)
  emitter.EmitNewline()

  emitter.EmitNamespaceEnd()
  emitter.EmitNamespaceEnd()
  emitter.EmitNewline()

  emitter.EmitPreprocessor('else')
  emitter.EmitPreprocessor1('warning',
                            '"Meta gemm fast-path requires GEMMLOWP_NEON_32!"')
  emitter.EmitPreprocessor('endif')
  emitter.EmitNewline()

  emitter.EmitHeaderEnd()


if __name__ == '__main__':
  Main()
