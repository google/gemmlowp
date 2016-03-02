"""Mul primitive used by the GEMM function.

The Mul primitive takes 1-3 zipped rows and 1-3 zipped columns and performs
matrix multiplication on those resulting in a small 1x1 to 3x3 block of results.
"""

import neon_emitter


class Error(Exception):
  """Module level error."""


class ConfigurationError(Error):
  """Unsupported configuration."""


class MulLanes(object):

  def __init__(self, input_address):
    self.input_address = input_address
    self.lanes = []

  def AddLane(self, lane):
    self.lanes.append(lane)

  def FreeRegisters(self, registers):
    for i in range(0, len(self.lanes)):
      registers.FreeRegister(self.lanes[i])
      self.lanes[i] = None


def GenerateMulLanes(registers, lane_count, address):
  lanes = MulLanes(address)
  for unused_i in range(0, lane_count):
    lanes.AddLane(registers.DoubleRegister())
  return lanes


def Generate3MulLanes(quad_register, registers, address):
  lanes = MulLanes(address)
  lanes.AddLane(registers.Low(quad_register))
  lanes.AddLane(registers.High(quad_register))
  lanes.AddLane(registers.DoubleRegister())
  return lanes


def GenerateAndClearAggregators(emitter, registers, aggregator_count):
  """Prepare aggregators and emit aggregator clear code."""
  emitter.EmitComment('Clear aggregators.')
  aggregators = []
  for i in range(0, aggregator_count):
    aggregator = registers.QuadRegister()
    aggregators.append(aggregator)
    if i < 3:
      emitter.EmitVMov('i32', aggregator, emitter.ImmediateConstant(0))
    else:
      emitter.EmitVMov('i32', aggregator, aggregators[i - 3])
  emitter.EmitNewline()
  return aggregators


def GenerateNxMLoadMultiplyAggregate(emitter, registers, left_lanes,
                                     right_lanes, aggregators, count):
  """Emit inner loop for N rows x M cols multiplication."""
  emitter.EmitComment('General NxM lanes loop.')
  emitter.EmitNumericalLabel(1)
  emitter.EmitNewline()
  emitter.EmitComment('Subtract counter.')
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))
  emitter.EmitNewline()

  emitter.EmitVLoadA('1.8', left_lanes.lanes,
                     emitter.DereferenceIncrement(left_lanes.input_address, 64))
  emitter.EmitVLoadA(
      '1.8', right_lanes.lanes,
      emitter.DereferenceIncrement(right_lanes.input_address, 64))

  emitter.EmitPldOffset(left_lanes.input_address, emitter.ImmediateConstant(64))
  emitter.EmitPldOffset(right_lanes.input_address,
                        emitter.ImmediateConstant(64))

  rows = len(left_lanes.lanes)
  cols = len(right_lanes.lanes)

  multiply_results = []
  for i in range(0, rows * cols):
    multiply_results.append(registers.QuadRegister())

  for row in range(0, rows):
    for col in range(0, cols):
      index = row * cols + col
      emitter.EmitVMull('u8', multiply_results[index], right_lanes.lanes[col],
                        left_lanes.lanes[row])

  for i in range(0, rows * cols):
    emitter.EmitVPadal('u16', aggregators[i], multiply_results[i])

  emitter.EmitNewline()
  emitter.EmitComment('Loop break.')
  emitter.EmitBneBack(1)
  emitter.EmitNewline()

  for register in multiply_results:
    registers.FreeRegister(register)


def Generate3x3LoadMultiplyAggregate(emitter, registers, left_lanes,
                                     right_lanes, aggregators, count,
                                     backup_register):
  """Emit inner loop for 3 rows x 3 cols multiplication (register trick)."""
  emitter.EmitComment('3x3 lanes loop.')
  emitter.EmitNumericalLabel(1)
  emitter.EmitNewline()
  emitter.EmitComment('Subtract counter.')
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))
  emitter.EmitNewline()

  emitter.EmitVLoadA('1.8', left_lanes.lanes,
                     emitter.DereferenceIncrement(left_lanes.input_address, 64))
  emitter.EmitVLoadA(
      '1.8', right_lanes.lanes,
      emitter.DereferenceIncrement(right_lanes.input_address, 64))

  emitter.EmitPldOffset(left_lanes.input_address, emitter.ImmediateConstant(64))
  emitter.EmitPldOffset(right_lanes.input_address,
                        emitter.ImmediateConstant(64))

  temp = []
  for unused_i in range(0, 4):
    temp.append(registers.QuadRegister())

  emitter.EmitVMull('u8', temp[0], left_lanes.lanes[0], right_lanes.lanes[0])
  emitter.EmitVMull('u8', temp[1], left_lanes.lanes[0], right_lanes.lanes[1])
  emitter.EmitVMull('u8', temp[2], left_lanes.lanes[0], right_lanes.lanes[2])
  emitter.EmitVMull('u8', temp[3], left_lanes.lanes[1], right_lanes.lanes[0])

  emitter.EmitVPadal('u16', aggregators[0], temp[0])
  emitter.EmitVPadal('u16', aggregators[1], temp[1])
  emitter.EmitVPadal('u16', aggregators[2], temp[2])
  emitter.EmitVPadal('u16', aggregators[3], temp[3])

  emitter.EmitVMull('u8', temp[0], left_lanes.lanes[1], right_lanes.lanes[1])
  emitter.EmitVMull('u8', temp[1], left_lanes.lanes[1], right_lanes.lanes[2])
  emitter.EmitVMull('u8', temp[2], left_lanes.lanes[2], right_lanes.lanes[0])
  emitter.EmitVMull('u8', temp[3], left_lanes.lanes[2], right_lanes.lanes[1])
  emitter.EmitVMull('u8', backup_register, left_lanes.lanes[2],
                    right_lanes.lanes[2])

  emitter.EmitVPadal('u16', aggregators[4], temp[0])
  emitter.EmitVPadal('u16', aggregators[5], temp[1])
  emitter.EmitVPadal('u16', aggregators[6], temp[2])
  emitter.EmitVPadal('u16', aggregators[7], temp[3])
  emitter.EmitVPadal('u16', aggregators[8], backup_register)

  emitter.EmitNewline()
  emitter.EmitComment('Loop break.')
  emitter.EmitBneBack(1)
  emitter.EmitNewline()

  for register in temp:
    registers.FreeRegister(register)


def ReadParams(emitter, registers, input_address, elements, min_reg):
  if elements == 1 or elements == 2:
    register = registers.DoubleRegister(min_reg * 2)
    emitter.EmitVLoad('1.32', register, emitter.Dereference(input_address, 64))
    return register
  elif elements == 3 or elements == 4:
    register = registers.QuadRegister(min_reg)
    emitter.EmitVLoad('1.32', register, emitter.Dereference(input_address, 64))
    return register
  else:
    raise ConfigurationError('Unsupported elements no: %d' % elements)


def Duplicate(emitter, registers, rows, cols, min_register, values):
  """Populate a grid of registers duplicating provided values."""
  duplicated = []
  if cols == 1 or cols == 2:
    for unused_i in range(0, rows):
      duplicated.append(registers.DoubleRegister(min_register))
  elif cols == 3 or cols == 4:
    for unused_i in range(0, rows):
      duplicated.append(registers.QuadRegister(min_register))
  else:
    raise ConfigurationError('Unsupported duplicate amount: %d' % cols)

  if rows == 1:
    emitter.EmitVDup('32', duplicated[0], emitter.Lane(values, 0))
  elif rows == 2:
    emitter.EmitVDup('32', duplicated[0], emitter.Lane(values, 0))
    emitter.EmitVDup('32', duplicated[1], emitter.Lane(values, 1))
  elif rows == 3:
    emitter.EmitVDup('32', duplicated[0], emitter.Lane(
        registers.Low(values), 0))
    emitter.EmitVDup('32', duplicated[1], emitter.Lane(
        registers.Low(values), 1))
    emitter.EmitVDup('32', duplicated[2], emitter.Lane(
        registers.High(values), 0))
  elif rows == 4:
    emitter.EmitVDup('32', duplicated[0], emitter.Lane(
        registers.Low(values), 0))
    emitter.EmitVDup('32', duplicated[1], emitter.Lane(
        registers.Low(values), 1))
    emitter.EmitVDup('32', duplicated[2], emitter.Lane(
        registers.High(values), 0))
    emitter.EmitVDup('32', duplicated[3], emitter.Lane(
        registers.High(values), 1))

  return duplicated


def DuplicateGeneralRegister(emitter, registers, cols, general_register,
                             min_register):
  if cols == 1 or cols == 2:
    duplicated = registers.DoubleRegister(min_register)
  elif cols == 3 or cols == 4:
    duplicated = registers.QuadRegister(min_register)
  else:
    raise ConfigurationError('Unsupported duplicate amount: %d' % cols)

  emitter.EmitVDup('32', duplicated, general_register)
  return duplicated


def ReduceAggregator(emitter, registers, aggregators, row, cols):
  if cols == 1:
    register = registers.Low(aggregators[row])
    emitter.EmitVPadd('u32', register, register, register)
    return register
  elif cols == 2:
    register = registers.Low(aggregators[row * 2])
    emitter.EmitVPadd('u32', register, register,
                      registers.Low(aggregators[row * 2 + 1]))
    return register
  elif cols == 3:
    register = aggregators[row * 3]
    emitter.EmitVPadd('u32', registers.Low(register), registers.Low(register),
                      registers.Low(aggregators[row * 3 + 1]))
    emitter.EmitVPadd('u32', registers.High(register),
                      registers.Low(aggregators[row * 3 + 2]),
                      registers.Low(aggregators[row * 3 + 2]))
    return register
  elif cols == 4:
    register = aggregators[row * 3]
    emitter.EmitVPadd('u32', registers.Low(register), registers.Low(register),
                      registers.Low(aggregators[row * 3 + 1]))
    emitter.EmitVPadd('u32', registers.High(register),
                      registers.Low(aggregators[row * 3 + 2]),
                      registers.Low(aggregators[row * 3 + 3]))
    return register
  else:
    raise ConfigurationError('Unsupported columns no: %d' % cols)


def StoreAggregator(emitter, registers, aggregator, cols, result_address,
                    result_stride):
  if cols == 1:
    emitter.EmitVStoreOffset('1.32', emitter.Lane(aggregator, 0),
                             emitter.Dereference(result_address, None),
                             result_stride)
  elif cols == 2:
    emitter.EmitVStoreOffset('1.32', aggregator,
                             emitter.Dereference(result_address, None),
                             result_stride)
  elif cols == 3:
    emitter.EmitVStore('1.32', registers.Low(aggregator),
                       emitter.DereferenceIncrement(result_address, None))
    emitter.EmitVStoreOffset('1.32', emitter.Lane(
        registers.High(aggregator),
        0), emitter.Dereference(result_address, None), result_stride)
    emitter.EmitNewline()
  elif cols == 4:
    emitter.EmitVStoreOffsetA(
        '1.32', [registers.Low(aggregator), registers.High(aggregator)],
        emitter.Dereference(result_address, None), result_stride)
  else:
    raise ConfigurationError('Unsupported columns no: %d' % cols)


def GenerateAggregatorReduceStore(emitter, registers, aggregators, result_type,
                                  lhs_add, rhs_add, left_lanes, right_lanes,
                                  results, results_stride):
  """Emit code that reduces 4 lane aggregators to 1 value, and stores them."""
  rows = len(left_lanes.lanes)
  cols = len(right_lanes.lanes)

  if lhs_add:
    left_offset = ReadParams(emitter, registers, left_lanes.input_address, rows,
                             4)
    left_offsets = Duplicate(emitter, registers, rows, cols, 4, left_offset)
  else:
    left_offsets = None

  if rhs_add:
    right_offset = ReadParams(emitter, registers, right_lanes.input_address,
                              cols, 4)
  else:
    right_offset = None

  if result_type is 'float':
    result_scale = DuplicateGeneralRegister(
        emitter, registers, cols, registers.MapParameter('result_scale'), 4)
  else:
    result_scale = None

  if cols == 3:
    emitter.EmitNewline()
    emitter.EmitComment('Change stride because storing in two ops.')
    emitter.EmitSub(results_stride, results_stride,
                    emitter.ImmediateConstant(8))

  emitter.EmitNewline()
  emitter.EmitComment('Horizontal reduce aggregators.')
  for aggregator in aggregators:
    emitter.EmitVPadd('u32', registers.Low(aggregator),
                      registers.Low(aggregator), registers.High(aggregator))

  emitter.EmitNewline()
  emitter.EmitComment('Reduce rows.')
  row_temps = []
  for i in range(0, rows):
    row_temps.append(ReduceAggregator(emitter, registers, aggregators, i, cols))

  if lhs_add:
    emitter.EmitNewline()
    emitter.EmitComment('Add lhs offsets to aggregated rows.')
    for (row_temp, left_offset) in zip(row_temps, left_offsets):
      emitter.EmitVAdd('s32', row_temp, row_temp, left_offset)

  if rhs_add:
    emitter.EmitNewline()
    emitter.EmitComment('Add rhs offset to aggregated rows.')
    for row_temp in row_temps:
      emitter.EmitVAdd('s32', row_temp, row_temp, right_offset)

  if result_type is 'float':
    emitter.EmitNewline()
    emitter.EmitComment('Convert to float. Multiply by result scale.')
    for row_temp in row_temps:
      emitter.EmitVCvt('f32', 's32', row_temp, row_temp)
    for row_temp in row_temps:
      emitter.EmitVMul('f32', row_temp, row_temp, result_scale)

  emitter.EmitNewline()
  emitter.EmitComment('Store reduced rows.')
  for row_temp in row_temps:
    StoreAggregator(emitter, registers, row_temp, cols, results, results_stride)


def BuildName(result_type, lhs_add, rhs_add, left, right):
  name = 'mul_%dx8_%dx8_%s' % (left, right, result_type)
  if lhs_add:
    name += '_lhsadd'
  if rhs_add:
    name += '_rhsadd'
  return name


def CppResultType(result_type):
  if result_type is 'int32':
    return 'std::int32_t*'
  elif result_type is 'float':
    return 'float*'
  else:
    raise ConfigurationError('Unsupported result type: %s' % result_type)


def GetParameters(result_type):
  params = [['const std::uint8_t*', 'lhs'], ['const std::uint8_t*', 'rhs'],
            ['std::int32_t', 'count'], [CppResultType(result_type), 'result'],
            ['std::int32_t', 'result_stride']]
  if result_type is 'float':
    params.append(['float', 'result_scale'])
  return params


def GenerateMulNx8Mx8(emitter, result_type, lhs_add, rhs_add, left_lanes_count,
                      right_lanes_count):
  """Emit the multiply code for given rows and cols counts."""
  if left_lanes_count < 1 or left_lanes_count > 4:
    raise ConfigurationError('Left_lanes should be: 1, 2, 3 or 4.')
  if right_lanes_count < 1 or right_lanes_count > 4:
    raise ConfigurationError('Right_lanes should be: 1, 2, 3 or 4.')

  emitter.EmitFunctionBeginA(
      BuildName(result_type, lhs_add, rhs_add, left_lanes_count,
                right_lanes_count), GetParameters(result_type), 'inline void')

  emitter.EmitAssert('count % 8 == 0')
  emitter.EmitAssert('count >= 8')
  emitter.EmitAsmBegin()

  registers = neon_emitter.NeonRegisters()

  count = registers.MapParameter('count')

  size = left_lanes_count * right_lanes_count

  lhs = registers.MapParameter('lhs')
  rhs = registers.MapParameter('rhs')

  emitter.EmitPld(lhs)
  emitter.EmitPld(rhs)

  aggregators = GenerateAndClearAggregators(emitter, registers, size)

  if size < 9:
    left_lanes = GenerateMulLanes(registers, left_lanes_count, lhs)
    right_lanes = GenerateMulLanes(registers, right_lanes_count, rhs)

    GenerateNxMLoadMultiplyAggregate(emitter, registers, left_lanes,
                                     right_lanes, aggregators, count)

  else:  # left == 3 and right == 3
    backup_register = registers.QuadRegister()
    left_lanes = Generate3MulLanes(backup_register, registers, lhs)
    right_lanes = GenerateMulLanes(registers, right_lanes_count, rhs)

    Generate3x3LoadMultiplyAggregate(emitter, registers, left_lanes,
                                     right_lanes, aggregators, count,
                                     backup_register)
  left_lanes.FreeRegisters(registers)
  right_lanes.FreeRegisters(registers)

  GenerateAggregatorReduceStore(emitter, registers, aggregators, result_type,
                                lhs_add, rhs_add, left_lanes, right_lanes,
                                registers.MapParameter('result'),
                                registers.MapParameter('result_stride'))

  emitter.EmitAsmEnd(registers.MappedParameters(), [],
                     registers.Clobbers() + ['cc', 'memory'])
  emitter.EmitFunctionEnd()


def GenerateFunctions(emitter, result_type, lhs_add, rhs_add):
  for left_lanes in range(1, 4):
    for right_lanes in range(1, 4):
      GenerateMulNx8Mx8(emitter, result_type, lhs_add, rhs_add, left_lanes,
                        right_lanes)
      emitter.EmitNewline()

  GenerateMulNx8Mx8(emitter, result_type, lhs_add, rhs_add, 1, 4)
  emitter.EmitNewline()


if __name__ == '__main__':
  GenerateFunctions(neon_emitter.NeonEmitter(), 'int32', True, True)
