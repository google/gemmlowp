"""Qnt primitive used by the GEMM function.

"""

import neon_emitter


class Error(Exception):
  """Module level error."""


class ConfigurationError(Error):
  """Unsupported configuration."""


class QntLane(object):

  def __init__(self, source, output, offset, load_1, load_2):
    self.source = source
    self.output = output
    self.offset = offset
    self.load_1 = load_1
    self.load_2 = load_2


def BuildName(lanes, leftovers, aligned):
  name = 'qnt_%dx8' % lanes
  if leftovers:
    name += '_%d' % leftovers
  if aligned:
    name += '_aligned'
  return name


def LoadAndDuplicateOffsets(emitter, registers, lanes, offsets):
  if lanes == 1 or lanes == 2 or lanes == 3:
    offset_registers = []
    for unused_i in range(0, lanes):
      register = registers.QuadRegister()
      emitter.EmitVLoadA('1.32', [emitter.AllLanes(registers.Low(register)),
                                  emitter.AllLanes(registers.High(register))],
                         emitter.DereferenceIncrement(offsets, 32))
      offset_registers.append(register)
    return offset_registers
  else:
    raise ConfigurationError('Unsupported number of lanes: %d' % lanes)


def GenerateQntLanes(emitter, registers, qnt_lanes, source, stride, destination,
                     destination_stride, offsets):
  """Prepare lanes for reading unquantized multiplication results."""
  offset_registers = LoadAndDuplicateOffsets(emitter, registers, qnt_lanes,
                                             offsets)

  lanes = []
  last_input_register = source
  last_output_register = destination
  for i in range(0, qnt_lanes):
    if not i:
      lanes.append(QntLane(source,
                           destination,
                           offset_registers[i],
                           registers.QuadRegister(),  # load 1
                           registers.QuadRegister()))  # load 2
    else:
      input_register = registers.GeneralRegister()
      output_register = registers.GeneralRegister()
      lanes.append(QntLane(input_register,
                           output_register,
                           offset_registers[i],
                           registers.QuadRegister(),  # load 1
                           registers.QuadRegister()))  # load 2
      emitter.EmitAdd(input_register, last_input_register, stride)
      emitter.EmitAdd(output_register, last_output_register, destination_stride)
      last_input_register = input_register
      last_output_register = output_register
  return lanes


def DuplicateRegister(emitter, registers, value):
  register = registers.QuadRegister()
  emitter.EmitVDup('32', register, value)
  return register


def GenerateQuantize(emitter, registers, lanes, lane_temps,
                     multiplicative_offset, rounding_offset, shift):
  """Inner loop for quantization: add offsets, multiply, round, shift."""
  for lane in lanes:
    emitter.EmitVAdd('i32', lane[0], lane[0], lane[1])

  for lane in lanes:
    emitter.EmitVMul('i32', lane[0], lane[0], multiplicative_offset)

  for lane in lanes:
    emitter.EmitVAdd('i32', lane[0], lane[0], rounding_offset)

  for lane in lanes:
    emitter.EmitVShl('s32', lane[0], lane[0], shift)

  for lane in lanes:
    emitter.EmitVQmovn('s32', lane[2], lane[0])

  for lane_temp in lane_temps:
    emitter.EmitVQmovun('s16', registers.Low(lane_temp), lane_temp)


def GenerateLoadQuantizeStore(emitter, registers, lanes, multiplicative_offset,
                              rounding_offset, shift, alignment):
  """Load unquantized data from lanes, quantize, store final result."""
  lane_temps = []
  for lane in lanes:
    lane_temps.append(registers.QuadRegister())

  for lane in lanes:
    emitter.EmitVLoadA(
        '1.32', [registers.Low(lane.load_1), registers.High(lane.load_1),
                 registers.Low(lane.load_2), registers.High(lane.load_2)],
        emitter.DereferenceIncrement(lane.source, 64))

  for lane in lanes:
    emitter.EmitPld(lane.source)

  quantize_setup = []
  for (lane_temp, lane) in zip(lane_temps, lanes):
    quantize_setup.append([lane.load_1, lane.offset, registers.Low(lane_temp)])
    quantize_setup.append([lane.load_2, lane.offset, registers.High(lane_temp)])

  GenerateQuantize(emitter, registers, quantize_setup, lane_temps,
                   multiplicative_offset, rounding_offset, shift)

  for (lane_temp, lane) in zip(lane_temps, lanes):
    emitter.EmitVStore('1.8', registers.Low(lane_temp),
                       emitter.DereferenceIncrement(lane.output, alignment))

  for lane_temp in lane_temps:
    registers.FreeRegister(lane_temp)


def GenerateLoadLeftovers(emitter, registers, leftovers, lanes):
  """Handle non multiply of 8 leftover loading."""
  if leftovers == 1:
    for lane in lanes:
      emitter.EmitVLoad('1.32', emitter.Lane(
          registers.Low(lane.load_1), 0),
                        emitter.Dereference(lane.source, None))
  elif leftovers == 2:
    for lane in lanes:
      emitter.EmitVLoad('1.32', registers.Low(lane.load_1),
                        emitter.Dereference(lane.source, 64))
  elif leftovers == 3:
    for lane in lanes:
      emitter.EmitVLoad('1.32', registers.Low(lane.load_1),
                        emitter.DereferenceIncrement(lane.source, 64))
    for lane in lanes:
      emitter.EmitVLoad('1.32', emitter.Lane(
          registers.High(lane.load_1), 0),
                        emitter.Dereference(lane.source, None))
  elif leftovers == 4:
    for lane in lanes:
      emitter.EmitVLoadA('1.32', [registers.Low(lane.load_1),
                                  registers.High(lane.load_1)],
                         emitter.Dereference(lane.source, 64))
  elif leftovers == 5:
    for lane in lanes:
      emitter.EmitVLoadA('1.32', [registers.Low(lane.load_1),
                                  registers.High(lane.load_1)],
                         emitter.DereferenceIncrement(lane.source, 64))
    for lane in lanes:
      emitter.EmitVLoad('1.32', emitter.Lane(
          registers.Low(lane.load_2), 0),
                        emitter.Dereference(lane.source, None))
  elif leftovers == 6:
    for lane in lanes:
      emitter.EmitVLoadA('1.32', [registers.Low(lane.load_1),
                                  registers.High(lane.load_1),
                                  registers.Low(lane.load_2)],
                         emitter.Dereference(lane.source, 64))
  elif leftovers == 7:
    for lane in lanes:
      emitter.EmitVLoadA('1.32', [registers.Low(lane.load_1),
                                  registers.High(lane.load_1),
                                  registers.Low(lane.load_2)],
                         emitter.DereferenceIncrement(lane.source, 64))
    for lane in lanes:
      emitter.EmitVLoad('1.32', emitter.Lane(
          registers.High(lane.load_2), 0),
                        emitter.Dereference(lane.source, None))
  else:
    raise ConfigurationError('Unsuported leftover count: %d' % leftovers)


def GenerateStoreLeftovers(emitter, registers, leftovers, lane_temps, lanes):
  """Handle non multiply of 8 leftover storing."""
  setup = []
  for (temp, lane) in zip(lane_temps, lanes):
    setup.append([registers.Low(temp), lane.output])

  if leftovers == 1:
    for lane in setup:
      emitter.EmitVStore('1.8', emitter.Lane(lane[0], 0),
                         emitter.Dereference(lane[1], None))
  elif leftovers == 2:
    for lane in setup:
      emitter.EmitVStore('1.16', emitter.Lane(lane[0], 0),
                         emitter.Dereference(lane[1], None))
  elif leftovers == 3:
    for lane in setup:
      emitter.EmitVStore('1.16', emitter.Lane(lane[0], 0),
                         emitter.DereferenceIncrement(lane[1], None))
    for lane in setup:
      emitter.EmitVStore('1.8', emitter.Lane(lane[0], 2),
                         emitter.Dereference(lane[1], None))
  elif leftovers == 4:
    for lane in setup:
      emitter.EmitVStore('1.32', emitter.Lane(lane[0], 0),
                         emitter.Dereference(lane[1], None))
  elif leftovers == 5:
    for lane in setup:
      emitter.EmitVStore('1.32', emitter.Lane(lane[0], 0),
                         emitter.DereferenceIncrement(lane[1], None))
    for lane in setup:
      emitter.EmitVStore('1.8', emitter.Lane(lane[0], 4),
                         emitter.Dereference(lane[1], None))
  elif leftovers == 6:
    for lane in setup:
      emitter.EmitVStore('1.32', emitter.Lane(lane[0], 0),
                         emitter.DereferenceIncrement(lane[1], None))
    for lane in setup:
      emitter.EmitVStore('1.16', emitter.Lane(lane[0], 2),
                         emitter.Dereference(lane[1], None))
  elif leftovers == 7:
    for lane in setup:
      emitter.EmitVStore('1.32', emitter.Lane(lane[0], 0),
                         emitter.DereferenceIncrement(lane[1], None))
    for lane in setup:
      emitter.EmitVStore('1.16', emitter.Lane(lane[0], 2),
                         emitter.DereferenceIncrement(lane[1], None))
    for lane in setup:
      emitter.EmitVStore('1.8', emitter.Lane(lane[0], 6),
                         emitter.DereferenceIncrement(lane[1], None))
  else:
    raise ConfigurationError('Unsupported leftovers count: %d' % leftovers)


def GenerateLeftoverLoadQuantizeStore(emitter, registers, leftovers, lanes,
                                      multiplicative_offset, rounding_offset,
                                      shift):
  """Handle leftovers if row size not a multiply of 8."""
  lane_temps = []
  for lane in lanes:
    lane_temps.append(registers.QuadRegister())

  GenerateLoadLeftovers(emitter, registers, leftovers, lanes)

  quantize_setup = []
  for (lane_temp, lane) in zip(lane_temps, lanes):
    quantize_setup.append([lane.load_1, lane.offset, registers.Low(lane_temp)])
    if leftovers > 4:
      quantize_setup.append([lane.load_2, lane.offset, registers.High(lane_temp)
                            ])

  GenerateQuantize(emitter, registers, quantize_setup, lane_temps,
                   multiplicative_offset, rounding_offset, shift)

  GenerateStoreLeftovers(emitter, registers, leftovers, lane_temps, lanes)


def GenerateQntNx8(emitter, qnt_lanes, leftovers, aligned):
  """Emits optimized quantization code for given lanes and row size."""
  if leftovers < 0 or leftovers > 7:
    raise ConfigurationError('Leftovers should be between 0 and 7 inclusive.')
  if qnt_lanes < 1 or qnt_lanes > 3:
    raise ConfigurationError('Qnt_lanes should should be 1, 2 or 3.')

  name = BuildName(qnt_lanes, leftovers, aligned)

  emitter.EmitFunctionBeginA(
      name,
      [['const std::int32_t*', 'source'], ['std::int32_t', 'count'],
       ['std::int32_t', 'stride'], ['const std::int32_t*', 'offsets'],
       ['std::uint8_t*', 'destination'], ['std::int32_t', 'destination_stride'],
       ['std::int32_t', 'multiplicative_offset'],
       ['std::int32_t', 'rounding_offset'], ['std::int32_t', 'shift']], 'void')
  emitter.EmitAssert('count %% 8 == %d' % leftovers)
  emitter.EmitAssert('count >= 8')
  emitter.EmitAssert('reinterpret_cast<std::uintptr_t>(source) % 8 == 0')
  if aligned:
    emitter.EmitAssert('reinterpret_cast<std::uintptr_t>(destination) % 8 == 0')
    if qnt_lanes > 1:
      emitter.EmitAssert('destination_stride % 8 == 0')
  emitter.EmitAsmBegin()

  registers = neon_emitter.NeonRegisters()

  count = registers.MapParameter('count')

  multiplicative_offset = DuplicateRegister(
      emitter, registers, registers.MapParameter('multiplicative_offset'))
  rounding_offset = DuplicateRegister(emitter, registers,
                                      registers.MapParameter('rounding_offset'))
  shift = DuplicateRegister(emitter, registers, registers.MapParameter('shift'))

  lanes = GenerateQntLanes(
      emitter, registers, qnt_lanes, registers.MapParameter('source'),
      registers.MapParameter('stride'), registers.MapParameter('destination'),
      registers.MapParameter('destination_stride'),
      registers.MapParameter('offsets'))

  if leftovers:
    emitter.EmitSubs(count, count, emitter.ImmediateConstant(leftovers))
    emitter.EmitBeqFront(2)

  emitter.EmitNewline()
  emitter.EmitNumericalLabel(1)
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))

  GenerateLoadQuantizeStore(emitter, registers, lanes, multiplicative_offset,
                            rounding_offset, shift, 64 if aligned else None)

  emitter.EmitNewline()
  emitter.EmitBneBack(1)

  if leftovers:
    emitter.EmitNumericalLabel(2)
    GenerateLeftoverLoadQuantizeStore(emitter, registers, leftovers, lanes,
                                      multiplicative_offset, rounding_offset,
                                      shift)

  emitter.EmitAsmEnd(registers.MappedParameters(), [],
                     registers.Clobbers() + ['cc', 'memory'])
  emitter.EmitFunctionEnd()


def BuildMultiQuantizeName(aligned, rows):
  name = 'multi_qnt_%dx8' % rows
  if aligned:
    name = '%s_aligned' % name
  return name


def GenerateMultiQuantize(emitter, aligned, rows):
  """Emit main quantization code that switches between optimized versions."""
  name = BuildMultiQuantizeName(aligned, rows)
  emitter.EmitFunctionBeginA(
      name,
      [['const std::int32_t*', 'source'], ['std::int32_t', 'count'],
       ['std::int32_t', 'stride'], ['const std::int32_t*', 'offsets'],
       ['std::uint8_t*', 'destination'], ['std::int32_t', 'destination_stride'],
       ['std::int32_t', 'multiplicative_offset'],
       ['std::int32_t', 'rounding_offset'], ['std::int32_t', 'shift']], 'void')
  emitter.EmitSwitch('count % 8')

  for leftovers in range(0, 8):
    emitter.EmitCase(leftovers)
    emitter.PushIndent()
    emitter.EmitCall(
        BuildName(rows, leftovers, aligned),
        ['source', 'count', 'stride', 'offsets', 'destination',
         'destination_stride', 'multiplicative_offset', 'rounding_offset',
         'shift'])
    emitter.EmitBreak()
    emitter.PopIndent()

  emitter.EmitSwitchEnd()
  emitter.EmitFunctionEnd()


def GenerateFunctions(neon, cc):
  for aligned in [True, False]:
    for lanes in range(1, 4):
      for leftovers in range(0, 8):
        GenerateQntNx8(neon, lanes, leftovers, aligned)
        neon.EmitNewline()

  for aligned in [True, False]:
    for rows in range(1, 4):
      GenerateMultiQuantize(cc, aligned, rows)
      cc.EmitNewline()
