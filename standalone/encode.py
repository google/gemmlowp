# Copyright 2018 The gemmlowp Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Encodes ARM asm code for certain instructions into the corresponding machine code encoding, as a .word directive in the asm code, preserving the original code in a comment.

Reads from stdin, writes to stdout.

Example diff:
-        "udot v16.4s, v4.16b, v0.16b\n"
+        ".word 0x6e809490  // udot v16.4s, v4.16b, v0.16b\n"

The intended use case is to make asm code easier to compile on toolchains that
do not support certain new instructions.
"""

import sys
import re
import argparse


def encode_udot_sdot_vector(line):
  m = re.search(
      r'\b([us])dot[ ]+v([0-9]+)[ ]*\.[ ]*4s[ ]*\,[ ]*v([0-9]+)[ ]*\.[ ]*16b[ ]*\,[ ]*v([0-9]+)[ ]*\.[ ]*16b',
      line)
  if not m:
    return 0, line

  match = m.group(0)
  unsigned = 1 if m.group(1) == 'u' else 0
  accum = int(m.group(2))
  lhs = int(m.group(3))
  rhs = int(m.group(4))
  assert accum >= 0 and accum <= 31
  assert lhs >= 0 and lhs <= 31
  assert rhs >= 0 and rhs <= 31
  mcode = 0x4e809400 | (accum << 0) | (lhs << 5) | (rhs << 16) | (
      unsigned << 29)
  return mcode, match


def encode_udot_sdot_element(line):
  m = re.search(
      r'\b([us])dot[ ]+v([0-9]+)[ ]*\.[ ]*4s[ ]*\,[ ]*v([0-9]+)[ ]*\.[ ]*16b[ ]*\,[ ]*v([0-9]+)[ ]*\.[ ]*4b[ ]*\[([0-9])\]',
      line)
  if not m:
    return 0, line

  match = m.group(0)
  unsigned = 1 if m.group(1) == 'u' else 0
  accum = int(m.group(2))
  lhs = int(m.group(3))
  rhs = int(m.group(4))
  lanegroup = int(m.group(5))
  assert accum >= 0 and accum <= 31
  assert lhs >= 0 and lhs <= 31
  assert rhs >= 0 and rhs <= 31
  assert lanegroup >= 0 and lanegroup <= 3
  l = 1 if lanegroup & 1 else 0
  h = 1 if lanegroup & 2 else 0
  mcode = 0x4f80e000 | (accum << 0) | (lhs << 5) | (rhs << 16) | (l << 21) | (
      h << 11) | (
          unsigned << 29)
  return mcode, match


def encode(line):
  for encode_func in [encode_udot_sdot_vector, encode_udot_sdot_element]:
    mcode, match = encode_func(line)
    if mcode:
      return mcode, match
  return 0, line


def read_existing_encoding(line):
  m = re.search(r'\.word\ (0x[0-9a-f]+)', line)
  if m:
    return int(m.group(1), 16)
  return 0


parser = argparse.ArgumentParser(description='Encode some A64 instructions.')
parser.add_argument(
    '-f',
    '--fix',
    help='fix existing wrong encodings in-place and continue',
    action='store_true')
args = parser.parse_args()

lineno = 0
found_existing_encodings = False
found_error = False
found_fixes = False
for line in sys.stdin:
  lineno = lineno + 1
  mcode, match = encode(line)
  if mcode:
    existing_encoding = read_existing_encoding(line)
    if existing_encoding:
      found_existing_encodings = True
      if mcode != existing_encoding:
        if args.fix:
          line = line.replace('.word 0x%x  // %s' % (existing_encoding, match),
                              '.word 0x%x  // %s' % (mcode, match))
          found_fixes = True
        else:
          sys.stderr.write(
              "Error at line %d: existing encoding 0x%x differs from encoding 0x%x for instruction '%s':\n\n%s\n\n"
              % (lineno, existing_encoding, mcode, match, line))
          found_error = True
    else:
      line = line.replace(match, '.word 0x%x  // %s' % (mcode, match))
  sys.stdout.write(line)
if found_error:
  sys.exit(1)
if found_existing_encodings:
  if found_fixes:
    sys.stderr.write(
        'Note: some instructions that this program is able to encode, were already encoded and their existing encodings didn\'t match the specified asm instructions. Since --fix was passed, these were fixed in-place.\n'
    )
  else:
    sys.stderr.write(
        'Note: some instructions that this program is able to encode, were already encoded. These encodings have been checked.\n'
    )
