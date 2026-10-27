"""Magnitude-specific result types for explicit representation APIs."""

from decimal import Decimal
from fractions import Fraction
from typing import assert_type

from unitx import Quantity
from unitx._numerics import Real
from unitx.si import si

assert_type(Quantity(Decimal('25'), si.percent).n, Decimal)
assert_type(Quantity(Fraction(1), si.percent).n, Fraction | float)
assert_type(Quantity(1.0, si.percent).n, Real)
assert_type(Quantity(1, si.percent).n, Real)
assert_type(Quantity(1j, si.percent).n, complex)
