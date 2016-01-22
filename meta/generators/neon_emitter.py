"""ARM/NEON assembly emitter.

Used by code generators to produce ARM assembly with NEON simd code.
Provides tools for easier register management: named register variable
allocation/deallocation, and offers a more procedural/structured approach
to generating assembly.

TODO: right now neon emitter prints out assembly instructions immediately,
it might be beneficial to keep the whole structure and emit the assembly after
applying some optimizations like: instruction reordering or register reuse.

TODO: NeonRegister object assigns explicit registers at allocation time.
Similarily to emiting code, register mapping and reuse can be performed and
optimized lazily.
"""


class Error(Exception):
  """Module level error."""


class RegisterAllocationError(Error):
  """Cannot alocate registers."""


class LaneError(Error):
  """Wrong lane number."""


def Low(register):
  assert register[0] == 'q'
  num = int(register[1:])
  return 'd%d' % (num * 2)


def High(register):
  assert register[0] == 'q'
  num = int(register[1:])
  return 'd%d' % (num * 2 + 1)


class NeonRegisters(object):
  """Utility that keeps track of used ARM/NEON registers."""

  def __init__(self):
    self.double = set()
    self.double_ever = set()
    self.general = set()
    self.general_ever = set()
    self.parameters = set()

  def MapParameter(self, parameter):
    self.parameters.add(parameter)
    return '%%[%s]' % parameter

  def DoubleRegister(self, min_val=0):
    for i in range(min_val, 32):
      if i not in self.double:
        self.double.add(i)
        self.double_ever.add(i)
        return 'd%d' % i
    raise RegisterAllocationError('Not enough double registers.')

  def QuadRegister(self, min_val=0):
    for i in range(min_val, 16):
      if ((i * 2) not in self.double) and ((i * 2 + 1) not in self.double):
        self.double.add(i * 2)
        self.double.add(i * 2 + 1)
        self.double_ever.add(i * 2)
        self.double_ever.add(i * 2 + 1)
        return 'q%d' % i
    raise RegisterAllocationError('Not enough quad registers.')

  def GeneralRegister(self):
    for i in range(0, 16):
      if i not in self.general:
        self.general.add(i)
        self.general_ever.add(i)
        return 'r%d' % i
    raise RegisterAllocationError('Not enough general registers.')

  def MappedParameters(self):
    return [x for x in self.parameters]

  def Clobbers(self):
    return (['r%d' % i
             for i in self.general_ever] + ['d%d' % i
                                            for i in self.DoubleClobbers()])

  def DoubleClobbers(self):
    return sorted(self.double_ever)

  def Low(self, register):
    return Low(register)

  def High(self, register):
    return High(register)

  def FreeRegister(self, register):
    assert len(register) > 1
    num = int(register[1:])

    if register[0] == 'r':
      assert num in self.general
      self.general.remove(num)
    elif register[0] == 'd':
      assert num in self.double
      self.double.remove(num)
    elif register[0] == 'q':
      assert num * 2 in self.double
      assert num * 2 + 1 in self.double
      self.double.remove(num * 2)
      self.double.remove(num * 2 + 1)
    else:
      raise RegisterDeallocationError('Register not allocated: %s' % register)


class NeonEmitter(object):
  """Emits ARM/NEON assembly opcodes."""

  def __init__(self, debug=False):
    self.ops = {}
    self.indent = ''
    self.debug = debug

  def PushIndent(self):
    self.indent += '  '

  def PopIndent(self):
    self.indent = self.indent[:-2]

  def EmitIndented(self, what):
    print self.indent + what

  def PushOp(self, op):
    if op in self.ops.keys():
      self.ops[op] += 1
    else:
      self.ops[op] = 1

  def ClearCounters(self):
    self.ops.clear()

  def EmitNewline(self):
    print ''

  def EmitPreprocessor1(self, op, param):
    print '#%s %s' % (op, param)

  def EmitPreprocessor(self, op):
    print '#%s' % op

  def EmitInclude(self, include):
    self.EmitPreprocessor1('include', include)

  def EmitCall1(self, function, param):
    self.EmitIndented('%s(%s);' % (function, param))

  def EmitAssert(self, assert_expression):
    if self.debug:
      self.EmitCall1('assert', assert_expression)

  def EmitHeaderBegin(self, header_name, includes):
    self.EmitPreprocessor1('ifndef', (header_name + '_H_').upper())
    self.EmitPreprocessor1('define', (header_name + '_H_').upper())
    self.EmitNewline()
    if includes:
      for include in includes:
        self.EmitInclude(include)
      self.EmitNewline()

  def EmitHeaderEnd(self):
    self.EmitPreprocessor('endif')

  def EmitCode(self, code):
    self.EmitIndented('%s;' % code)

  def EmitFunctionBeginA(self, function_name, params, return_type):
    self.EmitIndented('%s %s(%s) {' %
                      (return_type, function_name,
                       ', '.join(['%s %s' % (t, n) for (t, n) in params])))
    self.PushIndent()

  def EmitFunctionEnd(self):
    self.PopIndent()
    self.EmitIndented('}')

  def EmitAsmBegin(self):
    self.EmitIndented('asm volatile(')
    self.PushIndent()

  def EmitAsmMapping(self, elements, modifier):
    if elements:
      self.EmitIndented(': ' + ', '.join(['[%s] "%s"(%s)' % (d, modifier, d)
                                          for d in elements]))
    else:
      self.EmitIndented(':')

  def EmitClobbers(self, elements):
    if elements:
      self.EmitIndented(': ' + ', '.join(['"%s"' % c for c in elements]))
    else:
      self.EmitIndented(':')

  def EmitAsmEnd(self, outputs, inputs, clobbers):
    self.EmitAsmMapping(outputs, '+r')
    self.EmitAsmMapping(inputs, 'r')
    self.EmitClobbers(clobbers)
    self.PopIndent()
    self.EmitIndented(');')

  def EmitComment(self, comment):
    self.EmitIndented('// ' + comment)

  def EmitNumericalLabel(self, label):
    self.EmitIndented('"%d:"' % label)

  def EmitOp1(self, op, param1):
    self.PushOp(op)
    self.EmitIndented('"%s %s\\n"' % (op, param1))

  def EmitOp2(self, op, param1, param2):
    self.PushOp(op)
    self.EmitIndented('"%s %s, %s\\n"' % (op, param1, param2))

  def EmitOp3(self, op, param1, param2, param3):
    self.PushOp(op)
    self.EmitIndented('"%s %s, %s, %s\\n"' % (op, param1, param2, param3))

  def EmitZip(self, size, param1, param2):
    self.EmitOp2('vzip.%d' % size, param1, param2)

  def EmitZip8(self, param1, param2):
    self.EmitZip(8, param1, param2)

  def EmitZip16(self, param1, param2):
    self.EmitZip(16, param1, param2)

  def EmitZip32(self, param1, param2):
    self.EmitZip(32, param1, param2)

  def EmitAdd(self, destination, source, param):
    self.EmitOp3('add', destination, source, param)

  def EmitSubs(self, destination, source, param):
    self.EmitOp3('subs', destination, source, param)

  def EmitSub(self, destination, source, param):
    self.EmitOp3('sub', destination, source, param)

  def EmitMul(self, destination, source, param):
    self.EmitOp3('mul', destination, source, param)

  def EmitMov(self, param1, param2):
    self.EmitOp2('mov', param1, param2)

  def EmitSkip(self, register, skip, stride):
    self.EmitOp3('add', register, register, '#%d' % (skip * stride))

  def EmitBeqBack(self, label):
    self.EmitOp1('beq', '%db' % label)

  def EmitBeqFront(self, label):
    self.EmitOp1('beq', '%df' % label)

  def EmitBneBack(self, label):
    self.EmitOp1('bne', '%db' % label)

  def EmitBneFront(self, label):
    self.EmitOp1('bne', '%df' % label)

  def EmitVAdd(self, add_type, destination, source_1, source_2):
    self.EmitOp3('vadd.%s' % add_type, destination, source_1, source_2)

  def EmitVAddw(self, add_type, destination, source_1, source_2):
    self.EmitOp3('vaddw.%s' % add_type, destination, source_1, source_2)

  def EmitVCvt(self, cvt_to, cvt_from, destination, source):
    self.EmitOp2('vcvt.%s.%s' % (cvt_to, cvt_from), destination, source)

  def EmitVDup(self, dup_type, destination, source):
    self.EmitOp2('vdup.%s' % dup_type, destination, source)

  def EmitVMov(self, mov_type, destination, source):
    self.EmitOp2('vmov.%s' % mov_type, destination, source)

  def EmitVQmovn(self, mov_type, destination, source):
    self.EmitOp2('vqmovn.%s' % mov_type, destination, source)

  def EmitVQmovun(self, mov_type, destination, source):
    self.EmitOp2('vqmovun.%s' % mov_type, destination, source)

  def EmitVMul(self, mul_type, destination, source_1, source_2):
    self.EmitOp3('vmul.%s' % mul_type, destination, source_1, source_2)

  def EmitVMull(self, mul_type, destination, source_1, source_2):
    self.EmitOp3('vmull.%s' % mul_type, destination, source_1, source_2)

  def EmitVPadd(self, add_type, destination, source_1, source_2):
    self.EmitOp3('vpadd.%s' % add_type, destination, source_1, source_2)

  def EmitVPaddl(self, add_type, destination, source):
    self.EmitOp2('vpaddl.%s' % add_type, destination, source)

  def EmitVPadal(self, add_type, destination, source):
    self.EmitOp2('vpadal.%s' % add_type, destination, source)

  def EmitVLoad(self, load_type, destination, source):
    self.EmitOp2('vld%s' % load_type, '{%s}' % destination, '%s' % source)

  def EmitVLoadA(self, load_type, destinations, source):
    self.EmitVLoad(load_type, ', '.join(destinations), source)

  def EmitPld(self, load_address_register):
    self.EmitOp1('pld', '[%s]' % load_address_register)

  def EmitPldOffset(self, load_address_register, offset):
    self.EmitOp1('pld', '[%s, %s]' % (load_address_register, offset))

  def EmitInstructionPreload(self, label):
    self.EmitOp1('pli', label)

  def EmitVShl(self, shift_type, destination, source, shift):
    self.EmitOp3('vshl.%s' % shift_type, destination, source, shift)

  def EmitVStore(self, store_type, source, destination):
    self.EmitOp2('vst%s' % store_type, '{%s}' % source, destination)

  def EmitVStoreA(self, store_type, sources, destination):
    self.EmitVStore(store_type, ', '.join(sources), destination)

  def EmitVStoreOffset(self, store_type, source, destination, offset):
    self.EmitOp3('vst%s' % store_type, '{%s}' % source, destination, offset)

  def Dereference(self, value, alignment):
    if alignment:
      return '[%s:%d]' % (value, alignment)
    else:
      return '[%s]' % value

  def DereferenceIncrement(self, value, alignment):
    return '%s!' % self.Dereference(value, alignment)

  def ImmediateConstant(self, value):
    return '#%d' % value

  def AllLanes(self, value):
    return '%s[]' % value

  def Lane(self, value, lane):
    return '%s[%d]' % (value, lane)
