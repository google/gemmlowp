# Copyright 2016 The Gemmlowp Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""."""

import common


def _DuplicateGeneralRegister(size, emitter, registers, value, min_register):
  register = registers.QuadRegister(min_register)
  emitter.EmitVDup(size, register, value)
  return register


def _DuplicateGeneralMemoryRegister(size, emitter, registers, value,
                                    min_register):
  register = registers.QuadRegister(min_register)
  general = registers.GeneralRegister()
  emitter.EmitLdr(general, value)
  emitter.EmitVDup(size, register, general)
  registers.FreeRegister(general)
  return register


class _Indent(object):
  def __init__(self, emitter, comment=None, label=None):
    self._emitter = emitter
    self._comment = comment
    self._label = label

  def __enter__(self):
    self._emitter.EmitNewline()

    if self._comment is not None:
      self._emitter.EmitComment(self._comment)

    if self._label is not None:
      self._emitter.EmitNumericalLabel(self._label)

    self._emitter.PushIndent()

  def __exit__(self, *args, **kwargs):
    self._emitter.PopIndent()


class MinMaxTransformation(object):
  """."""

  def Check(self, in_type, out_type, kernel_size, leftovers):
    assert in_type is 'uint8_t'
    assert out_type is 'uint8_t'
    assert kernel_size is 16
    assert leftovers < 16

  def Prepare(self, emitter, registers, unused_kernel_size):
    emitter.EmitNewline()
    emitter.EmitComment('MinMax::Prepare')

    self.min = _DuplicateGeneralRegister(8, emitter, registers,
                                         registers.MapParameter('min',
                                                                'params.min'),
                                         4)
    self.max = _DuplicateGeneralRegister(8, emitter, registers,
                                         registers.MapParameter('max',
                                                                'params.max'),
                                         4)

  def Transform(self, emitter, registers, input_address, elements,
                output_address):
    """Generate the MinMax transform inner loop code."""
    emitter.EmitNewline()
    emitter.EmitComment('MinMax::Transform')
    register_count = (elements + 15) / 16
    load = [registers.QuadRegister() for unused_i in range(register_count)]
    emitter.EmitVLoadAE(8, elements, load, input_address, None)
    emitter.EmitPldOffset(input_address, emitter.ImmediateConstant(16))

    for register in load:
      emitter.EmitVMax('u8', register, register, self.min)

    for register in load:
      emitter.EmitVMin('u8', register, register, self.max)

    emitter.EmitNewline()
    emitter.EmitVStoreAE(8, elements, load, output_address, None)
    emitter.EmitPld(output_address)
    registers.FreeRegisters(load)


class DequantizeTransformation(object):
  """."""

  def Check(self, in_type, out_type, kernel_size, leftovers):
    assert in_type is 'uint8_t'
    assert out_type is 'float'
    assert kernel_size is 16
    assert leftovers < 16

  def Prepare(self, emitter, registers, unused_kernel_size):
    """Duplicate quantization offsets to vector registers."""
    emitter.EmitNewline()
    emitter.EmitComment('Dequantize::Prepare')

    self.range_min = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_min', 'params.range_min'), 4)
    self.range_offset = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_offset', 'params.range_offset'), 4)
    self.range_scale = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_scale', 'params.range_scale'), 4)

  def Transform(self, emitter, registers, input_address, elements,
                output_address):
    """Emit the dequantization inner loop."""
    emitter.EmitNewline()
    emitter.EmitComment('Dequantize::Transform')
    register_count = (elements + 3) / 4
    load = [registers.QuadRegister() for unused_i in range(register_count)]
    emitter.EmitVLoadAE(8, elements, load, input_address, None)
    emitter.EmitPldOffset(input_address, emitter.ImmediateConstant(32))

    if len(load) is 1:
      emitter.EmitVMovl('u8', load[0], load[0])
      emitter.EmitVMovl('s16', load[0], load[0])
    elif len(load) is 2:
      emitter.EmitVMovl('u8', load[0], load[0])
      emitter.EmitVMovl2('s16', load[0], load[1], load[0])
    elif len(load) is 3:
      emitter.EmitVMovl2('u8', load[0], load[1], load[0])
      emitter.EmitVMovl('s16', load[2], load[1])
      emitter.EmitVMovl2('s16', load[0], load[1], load[0])
    elif len(load) is 4:
      emitter.EmitVMovl2('u8', load[0], load[1], load[0])
      emitter.EmitVMovl2('s16', load[2], load[3], load[1])
      emitter.EmitVMovl2('s16', load[0], load[1], load[0])
    else:
      assert False

    for register in load:
      emitter.EmitVCvt('f32', 's32', register, register)

    for register in load:
      emitter.EmitVSub('f32', register, register, self.range_offset)

    for register in load:
      emitter.EmitVMul('f32', register, register, self.range_scale)

    for register in load:
      emitter.EmitVAdd('f32', register, register, self.range_min)

    emitter.EmitNewline()
    emitter.EmitVStoreAE(32, elements, load, output_address, None)
    emitter.EmitPld(output_address)
    registers.FreeRegisters(load)


class QuantizeTransformation(object):
  """."""

  def Check(self, in_type, out_type, kernel_size, leftovers):
    assert in_type is 'float'
    assert out_type is 'uint8_t'
    assert kernel_size is 16
    assert leftovers < 16

  def Prepare(self, emitter, registers, unused_kernel_size):
    """Duplicate quantization offsets to vector registers."""
    emitter.EmitNewline()
    emitter.EmitComment('Quantize::Prepare')

    self.range_min = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_min', 'params.range_min'), 4)
    self.range_offset = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_offset', 'params.range_offset'), 4)
    self.range_scale = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_scale', 'params.range_scale'), 4)

  def Transform(self, emitter, registers, input_address, elements,
                output_address):
    """Emit quantization inner loop code."""
    emitter.EmitNewline()
    emitter.EmitComment('Quantize::Transform')
    register_count = (elements + 3) / 4
    load = [registers.QuadRegister() for unused_i in range(register_count)]
    emitter.EmitVLoadAE(32, elements, load, input_address, None)
    emitter.EmitPldOffset(input_address, emitter.ImmediateConstant(64))

    for register in load:
      emitter.EmitVSub('f32', register, register, self.range_min)

    for register in load:
      emitter.EmitVMul('f32', register, register, self.range_scale)

    for register in load:
      emitter.EmitVAdd('f32', register, register, self.range_offset)

    for register in load:
      emitter.EmitVCvt('s32', 'f32', register, register)

    if len(load) is 1:
      emitter.EmitVQmovn('s32', load[0], load[0])
      emitter.EmitVQmovun('s16', load[0], load[0])
    elif len(load) is 2:
      emitter.EmitVQmovn2('s32', load[0], load[0], load[1])
      emitter.EmitVQmovun('s16', load[0], load[0])
    elif len(load) is 3:
      emitter.EmitVQmovn2('s32', load[0], load[0], load[1])
      emitter.EmitVQmovn('s32', load[2], load[2])
      emitter.EmitVQmovun2('s16', load[0], load[0], load[2])
    elif len(load) is 4:
      emitter.EmitVQmovn2('s32', load[0], load[0], load[1])
      emitter.EmitVQmovn2('s32', load[2], load[2], load[3])
      emitter.EmitVQmovun2('s16', load[0], load[0], load[2])
    else:
      assert False

    emitter.EmitNewline()
    emitter.EmitVStoreAE(8, elements, load, output_address, None)
    emitter.EmitPld(output_address)
    registers.FreeRegisters(load)


class BaseTransform(common.Transform1DKernelGenerator):
  """."""

  def __init__(self, cc_emitter, kernel_name, asm_emitter, transformation):
    common.Transform1DKernelGenerator.__init__(self, cc_emitter, kernel_name)
    self.asm_emitter = asm_emitter
    self.transformation = transformation

  def EmitTransform(self, in_type, out_type, kernel_size, leftovers):
    """."""
    self.transformation.Check(in_type, out_type, kernel_size, leftovers)

    registers = self.asm_emitter.CreateRegisters()

    self.emitter.EmitDeclare('int', 'params_count_copy', 'params.count')

    if hasattr(self.transformation, "declarations"):
        for dtype, dname, dexpr in self.transformation.declarations:
            self.emitter.EmitDeclare(dtype, dname, dexpr)

    self.asm_emitter.PushIndent(self.emitter.indent)
    self.asm_emitter.EmitAsmBegin()

    count = registers.MapOutputParameter('count', 'params_count_copy')
    input_address = registers.MapOutputParameter('input')
    output_address = registers.MapOutputParameter('output')

    self.transformation.Prepare(self.asm_emitter, registers, kernel_size)

    if leftovers:
      self.asm_emitter.EmitNewline()
      self.asm_emitter.EmitComment('Reduce count by leftovers.')
      self.asm_emitter.EmitSubs(count, count,
                                self.asm_emitter.ImmediateConstant(leftovers))
      self.asm_emitter.EmitBeqFront(2)

    self.asm_emitter.EmitNewline()
    self.asm_emitter.EmitNumericalLabel(1)
    self.asm_emitter.EmitSubs(count, count,
                              self.asm_emitter.ImmediateConstant(kernel_size))

    self.transformation.Transform(self.asm_emitter, registers, input_address,
                                  kernel_size, output_address)

    self.asm_emitter.EmitNewline()
    self.asm_emitter.EmitBneBack(1)

    if leftovers:
      self.asm_emitter.EmitNumericalLabel(2)
      self.asm_emitter.EmitNewline()
      self.asm_emitter.EmitComment('Handle leftovers.')
      self.transformation.Transform(self.asm_emitter, registers, input_address,
                                    leftovers, output_address)

    self.asm_emitter.EmitAsmEnd(registers)
    self.asm_emitter.PopIndent(len(self.emitter.indent))


class Quantize(BaseTransform):
  """."""

  def __init__(self, cc_emitter, asm_emitter):
    BaseTransform.__init__(self, cc_emitter, 'Quantize', asm_emitter,
                           QuantizeTransformation())


class Dequantize(BaseTransform):
  """."""

  def __init__(self, cc_emitter, asm_emitter):
    BaseTransform.__init__(self, cc_emitter, 'Dequantize', asm_emitter,
                           DequantizeTransformation())


class MinMax(BaseTransform):
  """."""

  def __init__(self, numerical_type, cc_emitter, asm_emitter):
    BaseTransform.__init__(self, cc_emitter, 'MinMax<%s>' % numerical_type,
                           asm_emitter, MinMaxTransformation())


class Requantize(common.Transform1DKernelGenerator):
  def __init__(self, cc_emitter, asm_emitter):
    super(Requantize, self).__init__(cc_emitter, "Requantize")
    self.asm_emitter = asm_emitter

  def EmitTransform(self, in_type, out_type, kernel_size, leftovers):
    # Check arguments are appropriate
    assert in_type == 'int32_t'
    assert out_type == 'uint8_t'
    assert kernel_size == 16
    assert leftovers < 16

    # Emit declarations required for the later assembly.
    self.emitter.EmitDeclare('int', 'params_count_copy', 'params.count')
    self.emitter.EmitDeclare(
      "const float", "coefficient",
      "params.one_over_output_range_scale * params.input_range_scale")
    self.emitter.EmitDeclare(
      "const float", "offset",
      """params.one_over_output_range_scale * (
       params.input_range_min - params.output_range_min -
       params.input_range_scale*params.input_range_offset
     )""")

    # Assign registers, we use 4 registers as "load" registers and two banks of
    # 4 as "output" registers. The two output banks are labelled "A" and "B".
    registers = self.asm_emitter.CreateRegisters()
    load_registers = [registers.QuadRegister() for _ in range(4)]
    registers_a = [registers.QuadRegister() for _ in range(4)]
    registers_b = [registers.QuadRegister() for _ in range(4)]

    # Modified registers
    self.count_register = registers.MapOutputParameter('count', 'params_count_copy')
    self.input_address = registers.MapOutputParameter('input')
    self.output_address = registers.MapOutputParameter('output')

    # Begin assembly
    self.asm_emitter.PushIndent(self.emitter.indent)
    self.asm_emitter.EmitAsmBegin()
    self._Prepare(registers)
    self.asm_emitter.EmitNewline()

    # Begin loop
    self.asm_emitter.EmitComment("Reduce count by leftovers")
    self.asm_emitter.EmitSubs(self.count_register, self.count_register, "#{:d}".format(leftovers))
    self.asm_emitter.EmitBeqFront(4)
    self.asm_emitter.EmitNewline()

    # Load A before entering loop
    self.asm_emitter.EmitComment("Prepare initial values")
    self.asm_emitter.EmitSubs(self.count_register, self.count_register, "#16")
    self._EmitCode(load_registers, registers_b, registers_a, 0, 16)
    self.asm_emitter.EmitBeqFront(2)
    self.asm_emitter.EmitNewline()

    # Looped portion of the code, process A while loading B then process B
    # while loading A.
    with _Indent(self.asm_emitter, "Requantize::Transform", label=1):
      self.asm_emitter.EmitComment("Requantize::Transform::Loop part A")
      self.asm_emitter.EmitSubs(self.count_register, self.count_register, "#16")
      self._EmitCode(load_registers, registers_a, registers_b)

      # If there are no more blocks of 16-values to load then jump to the tail to
      # finish processing the last loaded block of 16 and to load and process the
      # tail.
      self.asm_emitter.EmitNewline()
      self.asm_emitter.EmitBeqFront(3)
      self.asm_emitter.EmitNewline()

      self.asm_emitter.EmitComment("Requantize::Transform::Loop part B")
      self.asm_emitter.EmitSubs(self.count_register, self.count_register, "#16")
      self._EmitCode(load_registers, registers_b, registers_a)

      # If there are no more blocks of 16-values to load then fall through to
      # finish processing the last loaded block of 16 and to load and process the
      # tail - otherwise loop.
      self.asm_emitter.EmitNewline()
      self.asm_emitter.EmitBneBack(1)  # Loop

    # Tails
    # First tail: process remaining A while loading and processing tail in B
    # Second tail: process remaining B while loading and processing tail in A
    tails = (("A", registers_a, registers_b), ("B", registers_b, registers_a))
    for i, (label, first, second) in enumerate(tails, 2):
        comment = "Requantize::Transform::Tail {}".format(label)
        with _Indent(self.asm_emitter, comment, label=i):
          self._EmitCode(load_registers, first, second, load_number=leftovers)
          self.asm_emitter.EmitNewline()

          self._EmitCode(load_registers, second, first,
                         process_number=leftovers, load_number=0)

          self.asm_emitter.EmitNewline()
          self.asm_emitter.EmitBFront(5)

    # Third tail, process nothing, load and then process leftovers in A
    with _Indent(self.asm_emitter, "Requantize::Transform::Tail C", 4):
      self._EmitCode(load_registers, registers_b, registers_a, 0, leftovers)
      self._EmitCode(load_registers, registers_a, registers_b, leftovers, 0)

    # Explicit end of function, jumped to after end of first tail to avoid
    # running into second tail.
    self.asm_emitter.EmitNewline()
    self.asm_emitter.EmitComment("Requantize::Return")
    self.asm_emitter.EmitNumericalLabel(5)  # End of function

    registers.FreeRegisters(load_registers + registers_a + registers_b)
    self.asm_emitter.EmitAsmEnd(registers)
    self.asm_emitter.PopIndent(len(self.emitter.indent))

  def _Prepare(self, registers):
    """Prepare by duplicating parameters for later use."""
    self.asm_emitter.EmitComment("Requantize::Prepare")

    # Duplicate the constants required for the transformation
    self.coefficient = _DuplicateGeneralRegister(
        32, self.asm_emitter, registers,
        registers.MapParameter('coefficient'), 0
    )
    self.offset = _DuplicateGeneralRegister(
        32, self.asm_emitter, registers,
        registers.MapParameter('offset'), 0
    )

  def _EmitCode(self, load_registers, process_registers, next_registers,
                process_number=16, load_number=16):
    # Compute the number of registers to load and to process
    n_loads = (load_number + 3) / 4
    n_procs = (process_number + 3) / 4
    assert n_loads <= 4
    assert n_procs <= 4

    # Start by processing loaded and prepared values, while simultaneously
    # loading new values and copying the offset into the next set of registers
    # to process.
    loads_remaining = load_number
    for i in range(max(n_loads, n_procs)):
      # Extract registers relevant to this loop
      ra = process_registers[i]
      rb = next_registers[i]
      rl = load_registers[i]

      if i < n_procs:
        self.asm_emitter.EmitVMulAcc('f32', ra, rl, self.coefficient)

      if i < n_loads:
        # Update the number of loads to perform
        to_load = min(loads_remaining, 4)
        loads_remaining -= to_load

        # Perform a copy and a load of the appropriate width
        self.asm_emitter.EmitVMov('f32', rb, self.offset)
        self.asm_emitter.EmitVLoadAE(32, to_load, [rl], self.input_address, None)

      self.asm_emitter.EmitNewline()

    # Perform some housekeeping while processing (A) and preparing (B) for use.
    for ra in process_registers[:n_procs]:
      self.asm_emitter.EmitVCvt('s32', 'f32', ra, ra)

    for rl in load_registers[:n_loads]:
      self.asm_emitter.EmitVCvt('f32', 's32', rl, rl)

    # Store (A)
    if n_procs == 1:
      self.asm_emitter.EmitVQmovn('s32', process_registers[0], process_registers[0])
      self.asm_emitter.EmitVQmovun('s16', process_registers[0], process_registers[0])
    elif n_procs == 2:
      self.asm_emitter.EmitVQmovn2('s32', process_registers[0], process_registers[0], process_registers[1])
      self.asm_emitter.EmitVQmovun('s16', process_registers[0], process_registers[0])
    elif n_procs == 3:
      self.asm_emitter.EmitVQmovn2('s32', process_registers[0], process_registers[0], process_registers[1])
      self.asm_emitter.EmitVQmovn('s32', process_registers[2], process_registers[2])
      self.asm_emitter.EmitVQmovun2('s16', process_registers[0], process_registers[0], process_registers[2])
    elif n_procs == 4:
      self.asm_emitter.EmitVQmovn2('s32', process_registers[0], process_registers[0], process_registers[1])
      self.asm_emitter.EmitVQmovn2('s32', process_registers[2], process_registers[2], process_registers[3])
      self.asm_emitter.EmitVQmovun2('s16', process_registers[0], process_registers[0], process_registers[2])

    self.asm_emitter.EmitNewline()
    self.asm_emitter.EmitVStoreAE(8, process_number, process_registers, self.output_address, None)


class BiasAdd(common.Transform1DKernelGenerator):
  """."""

  def __init__(self, bias_type, cc_emitter, asm_emitter):
    common.Transform1DKernelGenerator.__init__(self, cc_emitter,
                                               'BiasAdd<%s>' % bias_type)
    self.asm_emitter = asm_emitter

  def EmitTransform(self, in_type, out_type, kernel_size, leftovers):
    """."""
    assert in_type is 'uint8_t'
    assert out_type is 'int32_t'
    assert kernel_size is 16
    assert leftovers < 16

    registers = self.asm_emitter.CreateRegisters()

    self.emitter.EmitDeclare('int', 'params_rows_copy', 'params.rows')
    self.emitter.EmitDeclare(
      "const float", "coeff_input",
      "params.input_range_scale * params.one_over_output_range_scale"
    )
    self.emitter.EmitDeclare(
      "const float", "coeff_bias",
      "params.bias_range_scale * params.one_over_output_range_scale"
    )
    self.emitter.EmitDeclare(
      "const float", "offset",
      """params.output_range_offset + params.one_over_output_range_scale * (
        params.input_range_min + params.bias_range_min - params.output_range_min
      )"""
    )

    self.asm_emitter.PushIndent(self.emitter.indent)
    self.asm_emitter.EmitAsmBegin()

    self._Prepare(self.asm_emitter, registers)

    rows = registers.MapParameter('rows', 'params_rows_copy')

    self.asm_emitter.EmitNumericalLabel(1)

    self._ProcessRow(self.asm_emitter, registers, kernel_size, leftovers)

    self.asm_emitter.EmitSubs(rows, rows, self.asm_emitter.ImmediateConstant(1))
    self.asm_emitter.EmitBneBack(1)

    self.asm_emitter.EmitAsmEnd(registers)
    self.asm_emitter.PopIndent(len(self.emitter.indent))

  def _Prepare(self, emitter, registers):
    # Non-duplicated parameters (to be duplicated below)
    nd_offset = registers.MapParameter("offset", "offset")
    nd_coeff_input = registers.MapParameter("coeff_input", "coeff_input")
    nd_coeff_bias = registers.MapParameter("coeff_bias", "coeff_bias")

    # Duplicate to fill lanes
    self.offset = _DuplicateGeneralRegister(
      32, emitter, registers, nd_offset, 12
    )
    self.coeff_input = _DuplicateGeneralRegister(
      32, emitter, registers, nd_coeff_input, 12
    )
    self.coeff_bias = _DuplicateGeneralRegister(
      32, emitter, registers, nd_coeff_bias, 12
    )

  def _ProcessRow(self, emitter, registers, kernel_size, leftovers):
    const_count = registers.MapParameter('count', 'params.count')
    const_bias = registers.MapParameter('bias', 'params.bias')

    count = registers.GeneralRegister()
    bias = registers.GeneralRegister()

    input_address = registers.MapOutputParameter('input')
    output_address = registers.MapOutputParameter('output')

    emitter.EmitMov(count, const_count)
    emitter.EmitMov(bias, const_bias)

    if leftovers:
      emitter.EmitSubs(count, count, emitter.ImmediateConstant(leftovers))
      emitter.EmitBeqFront(3)

    emitter.EmitNumericalLabel(2)
    emitter.EmitSubs(count, count, emitter.ImmediateConstant(kernel_size))

    self._BiasAdd(emitter, registers, kernel_size, input_address, bias,
                  output_address)

    emitter.EmitBneBack(2)

    if leftovers:
      emitter.EmitNumericalLabel(3)
      self._BiasAdd(emitter, registers, leftovers, input_address, bias,
                    output_address)

  def _BiasAdd(self, emitter, registers, elements, input_address, bias,
               output_address):
    emitter.EmitNewline()
    emitter.EmitComment('BiasAdd::Transform')
    register_count = (elements + 3) / 4

    load_input = [
        registers.QuadRegister() for unused_i in range(register_count)
    ]
    load_bias = [registers.QuadRegister() for unused_i in range(register_count)]
    outputs = [
        registers.QuadRegister() for unsused_i in range(register_count)
    ]

    emitter.EmitVLoadAE(8, elements, load_input, input_address, None)
    emitter.EmitVLoadAE(8, elements, load_bias, bias, None)

    # Extend the UINT8s to INT16 while copying the offset into the accumulators
    if len(load_input) is 1:
      emitter.EmitVMovl('u8', load_input[0], load_input[0])
      emitter.EmitVMov('f32', outputs[0], self.offset)  # Offset
      emitter.EmitVMovl('u8', load_bias[0], load_bias[0])
      emitter.EmitVMovl('s16', load_input[0], load_input[0])
      emitter.EmitVMovl('s16', load_bias[0], load_bias[0])
    elif len(load_input) is 2:
      emitter.EmitVMovl('u8', load_input[0], load_input[0])
      emitter.EmitVMov('f32', outputs[0], self.offset)  # Offset
      emitter.EmitVMovl('u8', load_bias[0], load_bias[0])
      emitter.EmitVMov('f32', outputs[1], self.offset)  # Offset
      emitter.EmitVMovl2('s16', load_input[0], load_input[1], load_input[0])
      emitter.EmitVMovl2('s16', load_bias[0], load_bias[1], load_bias[0])
    elif len(load_input) is 3:
      emitter.EmitVMovl2('u8', load_input[0], load_input[1], load_input[0])
      emitter.EmitVMov('f32', outputs[0], self.offset)  # Offset
      emitter.EmitVMovl2('u8', load_bias[0], load_bias[1], load_bias[0])
      emitter.EmitVMov('f32', outputs[1], self.offset)  # Offset
      emitter.EmitVMovl('s16', load_input[2], load_input[1])
      emitter.EmitVMov('f32', outputs[2], self.offset)  # Offset
      emitter.EmitVMovl('s16', load_bias[2], load_bias[1])
      emitter.EmitVMovl2('s16', load_input[0], load_input[1], load_input[0])
      emitter.EmitVMovl2('s16', load_bias[0], load_bias[1], load_bias[0])
    elif len(load_input) is 4:
      emitter.EmitVMovl2('u8', load_input[0], load_input[1], load_input[0])
      emitter.EmitVMov('f32', outputs[0], self.offset)  # Offset
      emitter.EmitVMovl2('u8', load_bias[0], load_bias[1], load_bias[0])
      emitter.EmitVMov('f32', outputs[1], self.offset)  # Offset
      emitter.EmitVMovl2('s16', load_input[2], load_input[3], load_input[1])
      emitter.EmitVMov('f32', outputs[2], self.offset)  # Offset
      emitter.EmitVMovl2('s16', load_bias[2], load_bias[3], load_bias[1])
      emitter.EmitVMov('f32', outputs[3], self.offset)  # Offset
      emitter.EmitVMovl2('s16', load_input[0], load_input[1], load_input[0])
      emitter.EmitVMovl2('s16', load_bias[0], load_bias[1], load_bias[0])
    else:
      assert False

    # Convert read values into appropriate format
    for register in load_input + load_bias:
      emitter.EmitVCvt('f32', 's32', register, register)

    # Perform the transform (two multiply-accumulates per output)
    for acc_reg, input_reg in zip(outputs, load_input):
      emitter.EmitVMulAcc('f32', acc_reg, input_reg, self.coeff_input)

    for acc_reg, bias_reg in zip(outputs, load_bias):
      emitter.EmitVMulAcc('f32', acc_reg, bias_reg, self.coeff_bias)

    # Convert the outputs back to an appropriate format
    for register in outputs:
      emitter.EmitVCvt('s32', 'f32', register, register)

    emitter.EmitNewline()
    emitter.EmitVStoreAE(32, elements, outputs, output_address, None)
    registers.FreeRegisters(outputs + load_input + load_bias)


def GenerateKernels(cc_emitter, asm_emitter, shapes):
  """Generate the quantization/dequantization/requantization kernels."""
  requantize = Requantize(cc_emitter, asm_emitter)
  quantize = Quantize(cc_emitter, asm_emitter)
  dequantize = Dequantize(cc_emitter, asm_emitter)
  minmax = MinMax('uint8_t', cc_emitter, asm_emitter)
  biasadd = BiasAdd('uint8_t', cc_emitter, asm_emitter)

  for shape in shapes:
    requantize.SpecializeTransform1DKernel('int32_t', 'uint8_t', shape[0],
                                           shape[1])

  for shape in shapes:
    quantize.SpecializeTransform1DKernel('float', 'uint8_t', shape[0], shape[1])

  for shape in shapes:
    dequantize.SpecializeTransform1DKernel('uint8_t', 'float', shape[0],
                                           shape[1])

  for shape in shapes:
    minmax.SpecializeTransform1DKernel('uint8_t', 'uint8_t', shape[0], shape[1])

  for shape in shapes:
    biasadd.SpecializeTransform1DKernel('uint8_t', 'int32_t', shape[0],
                                        shape[1])
