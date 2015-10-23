"""Zip primitive used by the GEMM function.

Takes 1 to 3 rows of data and interleaves them in 8 byte chunks. Pads to
multiply of 8 length with zeros. Calculates row sums and appends those at the
end.
"""


import neon_emitter


class Error(Exception):
  """Module level error."""


class ConfigurationError(Error):
  """Unsupported configuration."""


class ZipLane(object):

  def __init__(self, input_address, load, aggregator):
    self.input_address = input_address
    self.load = load
    self.aggregator = aggregator


def GenerateZipLanes(emitter, registers, zip_lanes, input_address, stride):
  """Prepares read lanes for the zip operation.

  Args:
    emitter: ARM/NEON emitter.
    registers: ARM/NEON registers state.
    zip_lanes: number of lanes to prepare.
    input_address: register that contains the input address for the first lane.
    stride: memory stride for lane inputs.

  Returns:
    Array of ZipLane objects.
  """
  lanes = []
  last_address_register = input_address
  for i in range(0, zip_lanes):
    if not i:
      lanes.append(ZipLane(input_address,
                           registers.DoubleRegister(),
                           registers.QuadRegister(2)))
    else:
      address_register = registers.GeneralRegister()
      lanes.append(ZipLane(address_register,
                           registers.DoubleRegister(),
                           registers.QuadRegister(2)))
      emitter.EmitAdd(address_register, last_address_register, stride)
      last_address_register = address_register
  return lanes


def BuildName(zip_lanes, leftovers, aligned):
  name = 'zip_%dx8' % zip_lanes
  if leftovers:
    name += '_%d' % leftovers
  if aligned:
    name += '_aligned'
  return name


def GenerateClearAggregators(emitter, lanes):
  for lane in lanes:
    emitter.EmitVMov('i16', lane.aggregator, emitter.ImmediateConstant(0))


def GenerateLoadAggregateStore(emitter, lanes, output_address, alignment):
  """Emit inner loop code for reading N lanes and interweaving them."""
  emitter.EmitNewline()
  emitter.EmitComment('Load Aggregate Store.')

  for lane in lanes:
    emitter.EmitVLoad(
        '1.8', lane.load,
        emitter.DereferenceIncrement(lane.input_address, alignment))

  store_registers = []
  for lane in lanes:
    emitter.EmitVAddw('u8', lane.aggregator, lane.aggregator, lane.load)
    store_registers.append(lane.load)

  emitter.EmitVStoreA('1.8', store_registers,
                      emitter.DereferenceIncrement(output_address, 64))


def GenerateLeftoverLoadAggregateStore(
    emitter, leftovers, lanes, output_address):
  """Handle leftovers when count is not a multiply of 8."""
  emitter.EmitNewline()
  emitter.EmitComment('Leftover Load Aggregate Store.')

  # Clear load registers.
  for lane in lanes:
    emitter.EmitVMov('i8', lane.load, emitter.ImmediateConstant(0))

  if leftovers == 1:
    # Load 8 bits.
    for lane in lanes:
      emitter.EmitVLoad('1.8', emitter.Lane(lane.load, 0),
                        emitter.Dereference(lane.input_address, None))
  elif leftovers == 2:
    # Load 16 bits.
    for lane in lanes:
      emitter.EmitVLoad('1.16', emitter.Lane(lane.load, 0),
                        emitter.Dereference(lane.input_address, None))
  elif leftovers == 3:
    # Load 16 bits.
    for lane in lanes:
      emitter.EmitVLoad(
          '1.16', emitter.Lane(lane.load, 0),
          emitter.DereferenceIncrement(lane.input_address, None))
    # Load 8 bits.
    for lane in lanes:
      emitter.EmitVLoad('1.8', emitter.Lane(lane.load, 2),
                        emitter.Dereference(lane.input_address, None))
  elif leftovers == 4:
    # Load 32 bits.
    for lane in lanes:
      emitter.EmitVLoad('1.32', emitter.Lane(lane.load, 0),
                        emitter.Dereference(lane.input_address, None))
  elif leftovers == 5:
    # Load 32 bits..
    for lane in lanes:
      emitter.EmitVLoad(
          '1.32', emitter.Lane(lane.load, 0),
          emitter.DereferenceIncrement(lane.input_address, None))
    # Load 8 bits.
    for lane in lanes:
      emitter.EmitVLoad('1.8', emitter.Lane(lane.load, 4),
                        emitter.Dereference(lane.input_address, None))
  elif leftovers == 6:
    # Load 32 bits..
    for lane in lanes:
      emitter.EmitVLoad(
          '1.32', emitter.Lane(lane.load, 0),
          emitter.DereferenceIncrement(lane.input_address, None))
    # Load 16 bits.
    for lane in lanes:
      emitter.EmitVLoad('1.16', emitter.Lane(lane.load, 2),
                        emitter.Dereference(lane.input_address, None))
  elif leftovers == 7:
    # Load 32 bits..
    for lane in lanes:
      emitter.EmitVLoad(
          '1.32', emitter.Lane(lane.load, 0),
          emitter.DereferenceIncrement(lane.input_address, None))
    # Load 16 bits.
    for lane in lanes:
      emitter.EmitVLoad(
          '1.16', emitter.Lane(lane.load, 2),
          emitter.DereferenceIncrement(lane.input_address, None))
    # Load 8 bits.
    for lane in lanes:
      emitter.EmitVLoad('1.8', emitter.Lane(lane.load, 6),
                        emitter.Dereference(lane.input_address, None))
  else:
    raise ConfigurationError('Unsupported leftover num: %d' % leftovers)

  # Aggregate.
  store_registers = []
  for lane in lanes:
    emitter.EmitVAddw('u8', lane.aggregator, lane.aggregator, lane.load)
    store_registers.append(lane.load)

  # Store.
  emitter.EmitVStoreA('1.8', store_registers,
                      emitter.DereferenceIncrement(output_address, 64))


def GenerateAggregatorReduction(emitter,
                                registers,
                                lanes,
                                output_address,
                                multiplicative_offset,
                                additive_offset):
  """Reduce 4 lane sum aggregators to 1 value and store the sums."""
  emitter.EmitNewline()
  emitter.EmitComment('Aggregator Reduction.')

  multiplier = registers.DoubleRegister()
  emitter.EmitVMov('32', emitter.Lane(multiplier, 0), multiplicative_offset)
  offset = registers.QuadRegister()
  emitter.EmitVDup('32', offset, additive_offset)

  lane_temps = []
  for lane in lanes:
    emitter.EmitVPaddl('u16', lane.aggregator, lane.aggregator)

  for lane in lanes:
    lane_temp = registers.DoubleRegister()
    lane_temps.append(lane_temp)
    emitter.EmitVPadd('u32',
                      lane_temp,
                      registers.Low(lane.aggregator),
                      registers.High(lane.aggregator))

  temp = registers.QuadRegister()
  low = registers.Low(temp)
  high = registers.High(temp)

  if len(lanes) == 1:
    emitter.EmitVPadd('u32', low, lane_temps[0], lane_temps[0])
  elif len(lanes) == 2:
    emitter.EmitVPadd('u32', low, lane_temps[0], lane_temps[1])
  elif len(lanes) == 3:
    emitter.EmitVPadd('u32', low, lane_temps[0], lane_temps[1])
    emitter.EmitVPadd('u32', high, lane_temps[2], lane_temps[2])
  elif len(lanes) == 4:
    emitter.EmitVPadd('u32', low, lane_temps[0], lane_temps[1])
    emitter.EmitVPadd('u32', high, lane_temps[2], lane_temps[3])
  else:
    raise ConfigurationError(
        'Unexpected number of aggregators to reduce: %d' % len(lanes))

  emitter.EmitVMul('i32', temp, temp, emitter.Lane(multiplier, 0))
  emitter.EmitVAdd('i32', temp, temp, offset)

  if len(lanes) == 1:
    emitter.EmitVStore(
        '1.32', emitter.Lane(low, 0), emitter.Dereference(output_address, None))
  elif len(lanes) == 2:
    emitter.EmitVStore('1.32', low, emitter.Dereference(output_address, 64))
  elif len(lanes) == 3:
    emitter.EmitVStore(
        '1.32', low, emitter.DereferenceIncrement(output_address, 64))
    emitter.EmitVStore(
        '1.32', emitter.Lane(high, 0),
        emitter.Dereference(output_address, None))
  elif len(lanes) == 4:
    emitter.EmitVStore(
        '1.32', low, emitter.DereferenceIncrement(output_address, 64))
    emitter.EmitVStore('1.32', high, emitter.Dereference(output_address, 64))


def GenerateZipNx8(emitter, zip_lanes, leftovers, aligned):
  """Emit the zip function for a given number of rows and row size leftovers."""
  if leftovers < 0 or leftovers > 7:
    raise ConfigurationError('Leftovers should be between 0 and 7 inclusive.')
  if zip_lanes < 1 or zip_lanes > 3:
    raise ConfigurationError('Zip_lanes should should be 1, 2 or 3.')

  name = BuildName(zip_lanes, leftovers, aligned)

  emitter.EmitFunctionBeginA(name,
                             [['const std::uint8_t*', 'source'],
                              ['std::int32_t', 'count'],
                              ['std::int32_t', 'stride'],
                              ['std::uint8_t*', 'destination'],
                              ['std::int32_t', 'multiplicative_offset'],
                              ['std::int32_t', 'additive_offset']],
                             'void')
  emitter.EmitAssert('count %% 8 == %d' % leftovers)
  emitter.EmitAssert('count <= 2048')
  emitter.EmitAssert('count >= 8')
  emitter.EmitAssert('reinterpret_cast<std::uintptr_t>(destination) % 8 == 0')
  if aligned:
    emitter.EmitAssert('reinterpret_cast<std::uintptr_t>(source) % 8 == 0')
    if zip_lanes > 1:
      emitter.EmitAssert('stride % 8 == 0')
  emitter.EmitAsmBegin()

  registers = neon_emitter.NeonRegisters()

  count = registers.MapParameter('count')
  output_address = registers.MapParameter('destination')

  lanes = GenerateZipLanes(emitter,
                           registers,
                           zip_lanes,
                           registers.MapParameter('source'),
                           registers.MapParameter('stride'))

  if leftovers:
    emitter.EmitSub(count, count, emitter.ImmediateConstant(leftovers))

  GenerateClearAggregators(emitter, lanes)

  emitter.EmitNewline()
  emitter.EmitNumericalLabel(1)
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))

  GenerateLoadAggregateStore(
      emitter, lanes, output_address, 64 if aligned else None)

  emitter.EmitNewline()
  emitter.EmitBneBack(1)

  if leftovers:
    GenerateLeftoverLoadAggregateStore(
        emitter, leftovers, lanes, output_address)

  GenerateAggregatorReduction(emitter,
                              registers,
                              lanes,
                              output_address,
                              registers.MapParameter('multiplicative_offset'),
                              registers.MapParameter('additive_offset'))

  emitter.EmitAsmEnd(registers.MappedParameters(),
                     [],
                     registers.Clobbers() + ['cc', 'memory'])
  emitter.EmitFunctionEnd()


def GenerateFunctions(emitter):
  for aligned in [True, False]:
    for lanes in range(1, 4):
      for leftovers in range(0, 8):
        GenerateZipNx8(emitter, lanes, leftovers, aligned)
        emitter.EmitNewline()
