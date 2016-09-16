"""Generates the arm32 headers used by the gemm/gemv lib."""

import cc_emitter
import common
import neon_emitter
import streams_common


def Main():
  """."""
  cc = cc_emitter.CCEmitter()
  common.GenerateHeader(cc, 'gemmlowp_meta_streams_arm_32', 'GEMMLOWP_NEON_32')

  cc.EmitNamespaceBegin('gemmlowp')
  cc.EmitNamespaceBegin('meta')
  cc.EmitNewline()

  streams_common.GenerateUInt8x8Streams(cc, neon_emitter.NeonEmitter(), 8)

  cc.EmitNamespaceEnd()
  cc.EmitNamespaceEnd()
  cc.EmitNewline()

  common.GenerateFooter(cc, 'Meta gemm for arm32 requires: GEMMLOWP_NEON_32!')


if __name__ == '__main__':
  Main()
