"""Concrete logarithmic types preserve numerical-domain annotations."""

from decimal import Decimal
from fractions import Fraction
from typing import assert_type

from unitx import AffineQuantity, LogQuantity, LogUnit, Quantity
from unitx._numerics import Magnitude
from unitx.si import si

assert_type(si.dBm, LogUnit)
assert_type(si.dBm.reference, Quantity[Fraction])
assert_type(si.dBm.at, Fraction)
assert_type(30 @ si.dBm, LogQuantity[int])
assert_type(Decimal(30) @ si.dBm, LogQuantity[Decimal])
level = LogQuantity(Decimal(30), si.dBm)
assert_type(level.unit, LogUnit)
assert_type(level.to(si.dBW), LogQuantity[Decimal])
assert_type(level.value_in(si.dBW), Decimal)
assert_type(level.to_quantity(si.W), Quantity[Decimal])
assert_type(level - LogQuantity(0, si.dBW), Quantity[Magnitude])
assert_type(level + 1 * si.dB, LogQuantity[Magnitude])
assert_type(1 * si.dB + level, LogQuantity[Magnitude])
assert_type(LogQuantity.from_quantity(Decimal(1) * si.W, si.dBm), LogQuantity[Decimal])
assert_type((Decimal(20) @ si.degC).to(si.K), AffineQuantity[Decimal])
