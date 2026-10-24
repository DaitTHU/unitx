"""The extension preserves its type for its specialized network operations."""

from fractions import Fraction
from typing import assert_type

from examples.impedance import Impedance, Number
from unitx import Quantity
from unitx._numerics import Scalar
from unitx.si import si

first, second = Impedance(Fraction(100)), Impedance(200j)
assert_type(first.value, Number)
assert_type(first.value_in(si.kohm), Number)
assert_type(first + second, Impedance)
assert_type(first // second, Impedance)
assert_type((first + second).to(si.kohm) // second, Impedance)
assert_type(first * (2 * si.A), Quantity[Scalar])
