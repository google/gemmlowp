"""Generates the meta gemm/gemv library header."""

import cc_emitter
import gemm_NxMxK_neon
import gemv_1xMxK_neon
import mul_1x8_Mx8_neon
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


def GenerateInternalFunctions(emitter):
  """Generate all the functions hidden in the internal namespace."""
  zip_Nx8_neon.GenerateFunctions(neon_emitter.NeonEmitter())
  emitter.EmitNewline()

  mul_Nx8_Mx8_neon.GenerateFunctions(neon_emitter.NeonEmitter(), 'int32', False,
                                     True)
  emitter.EmitNewline()

  mul_Nx8_Mx8_neon.GenerateFunctions(neon_emitter.NeonEmitter(), 'int32', True,
                                     True)
  emitter.EmitNewline()

  mul_Nx8_Mx8_neon.GenerateFunctions(neon_emitter.NeonEmitter(), 'float', True,
                                     True)
  emitter.EmitNewline()

  mul_1x8_Mx8_neon.GenerateFunctions(neon_emitter.NeonEmitter(), 'int32', False,
                                     True)
  emitter.EmitNewline()

  mul_1x8_Mx8_neon.GenerateFunctions(neon_emitter.NeonEmitter(), 'int32', True,
                                     True)
  emitter.EmitNewline()

  mul_1x8_Mx8_neon.GenerateFunctions(neon_emitter.NeonEmitter(), 'float', True,
                                     True)
  emitter.EmitNewline()

  qnt_Nx8_neon.GenerateFunctions(neon_emitter.NeonEmitter(), emitter)
  emitter.EmitNewline()

  gemm_NxMxK_neon.GenerateInternalFunctions(emitter)
  emitter.EmitNewline()

  gemv_1xMxK_neon.GenerateInternalFunctions(emitter)
  emitter.EmitNewline()


def GeneratePublicFunctions(emitter):
  gemm_NxMxK_neon.GeneratePublicFunctions(emitter)
  emitter.EmitNewline()

  gemv_1xMxK_neon.GeneratePublicFunctions(emitter)
  emitter.EmitNewline()


def Main():
  """Generate the single threaded meta gemm library."""
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

  GenerateInternalFunctions(emitter)

  emitter.EmitNamespaceEnd()
  emitter.EmitNewline()

  GeneratePublicFunctions(emitter)

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
