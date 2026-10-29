from typing import assert_type

from unitx import Quantity, Unit
from unitx.information import information
from unitx.si import si

assert_type(information.bit, Unit)
assert_type(information.B, Unit)
assert_type(information.MiB, Unit)
assert_type(information.QiB, Unit)
assert_type(information.unit('MiB/s'), Unit)
assert_type(3 * information.MiB, Quantity[int])
assert_type(si.turn, Unit)
assert_type(si.cycle, Unit)
assert_type(si.ppm, Unit)
assert_type(si.Jy, Unit)
