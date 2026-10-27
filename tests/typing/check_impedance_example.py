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
assert_type(first * (2 * si.A), Quantity[Number])
ordinary = 100 * si.ohm
assert_type(first + ordinary, Quantity[Scalar])
assert_type(ordinary + first, Quantity[Number])
assert_type(first - second, Quantity[Number])
assert_type(-first, Quantity[Scalar])
series: Impedance = first
series += second
assert_type(series, Impedance)
mixed = first
mixed += ordinary
assert_type(mixed, Quantity[Scalar])
parallel = first
parallel //= second
assert_type(parallel, Impedance)
