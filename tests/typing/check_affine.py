"""Points and differences retain distinct public classes and construction types."""

from decimal import Decimal
from fractions import Fraction
from typing import assert_type

from unitx import AffineQuantity, AffineUnit, Quantity, Unit
from unitx._numerics import Magnitude
from unitx.si import si

point = AffineQuantity(Decimal('20'), 'degC')
assert_type(point, AffineQuantity[Decimal])
assert_type(point.unit, AffineUnit)
assert_type(point.unit.delta_unit, Unit)
assert_type(point.to('K'), AffineQuantity[Decimal])
assert_type(point.to_quantity('K'), Quantity[Decimal])
assert_type(Quantity(1, 'K').unit, Unit)
assert_type(Unit(), Unit)
assert_type(Unit('m'), Unit)
assert_type(Unit('degC'), Unit)
assert_type(si.degC, AffineUnit)
assert_type(2 * Unit('degC'), Quantity[int])
assert_type(Unit('m') * Unit('s'), Unit)

celsius = AffineUnit(Decimal('273.15') * si.K)
fahrenheit = AffineUnit(AffineQuantity(0, celsius), at=32, delta_unit=si.degR)
assert_type(celsius, AffineUnit)
assert_type(fahrenheit, AffineUnit)
assert_type(fahrenheit.origin.unit, Unit)
assert_type(32 @ fahrenheit, AffineQuantity[int])
assert_type(Fraction(32) @ fahrenheit, AffineQuantity[Fraction])
assert_type(AffineQuantity(32, fahrenheit) - AffineQuantity(0, celsius), Quantity[Magnitude])

assert_type(0 @ si.degC, AffineQuantity[int])
assert_type(Decimal('20') @ si.degC, AffineQuantity[Decimal])
assert_type(Fraction(20) @ Unit('degC'), AffineQuantity[Fraction])
assert_type(0 @ Unit('K'), AffineQuantity[int])

assert_type(2 * si.degC, Quantity[int])
assert_type(si.degC * Decimal('2'), Quantity[Decimal])
assert_type(Quantity(2, si.degC), Quantity[int])
assert_type(Quantity(2, si.degC).unit, Unit)
assert_type(Quantity(Decimal('2'), 'K').to(si.degC), Quantity[Decimal])
assert_type(Quantity(2, 'm') * si.degC, Quantity[int])
assert_type(si.degC * Quantity(2, 'm'), Quantity[int])
assert_type(Unit('m') * si.degC, Unit)
assert_type(si.degC / Unit('m'), Unit)
assert_type(Quantity(Decimal('2'), si.K) + point, AffineQuantity[Magnitude])
assert_type(si.degC * si.dBm, Unit)
