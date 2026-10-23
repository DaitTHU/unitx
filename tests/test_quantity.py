import math
import operator
import pickle
import warnings
from copy import copy, deepcopy
from decimal import (
    ROUND_DOWN,
    ROUND_UP,
    Decimal,
    FloatOperation,
    Inexact,
    InvalidOperation,
    Rounded,
    localcontext,
)
from fractions import Fraction

import pytest

from unitx.dimension import DIMENSIONLESS, Dimension
from unitx.exceptions import DimensionError, InexactFloatWarning, UnitSymbolError
from unitx.factor import Factor, one, pi
from unitx.quantity import Quantity
from unitx.unit import Unit


class _IntMagnitude(int):
    pass


class _FloatMagnitude(float):
    pass


class _DecimalMagnitude(Decimal):
    pass


class _FractionMagnitude(Fraction):
    pass


class _ComplexMagnitude(complex):
    pass


def test_quantity_creation() -> None:
    value = Decimal('1.25')
    unit = Unit('cm/s')
    quantity = Quantity(value, unit)
    assert quantity.value is value
    assert quantity.unit is unit
    assert quantity.dimension == Dimension(L=1, T=-1)
    assert Quantity(value, 'cm/s').unit == unit
    assert not quantity.is_dimensionless()
    assert Quantity(1, 'km/m').is_dimensionless()
    assert Quantity(1, '').dimension is DIMENSIONLESS


@pytest.mark.parametrize(
    ('unit', 'expected'),
    [('', '2'), ('m', '2 m'), ('1/s', '2/s'), ('1/(m*s)', '2/(m⋅s)')],
)
def test_quantity_str(unit: str, expected: str) -> None:
    assert str(Quantity(2, unit)) == expected


@pytest.mark.parametrize(
    ('unit', 'expected'),
    [
        (Factor(2) * Unit(), '3⋅2'),
        (Factor(2) / Unit('s'), '3⋅2/s'),
        (Factor(60) * Unit('s'), '3⋅60⋅s'),
        (Unit('J*s') / (2 * pi), '3 J⋅s/(2⋅π)'),
        (Unit('1/s') / (2 * pi), '3/(2⋅π⋅s)'),
        (Unit() / pi, '3/π'),
        (pi * Unit('s'), '3 π⋅s'),
        (Factor(12) * Unit('s'), '3⋅12⋅s'),
        (Unit(Unit(), scale=one), '3'),
        (Unit(Unit('1/s'), scale=one), '3/s'),
    ],
)
def test_quantity_scaled_str(unit: Unit, expected: str) -> None:
    quantity = Quantity(3, unit)
    before = (quantity.value, quantity.unit, quantity.unit.factor)
    assert str(quantity) == expected
    assert (quantity.value, quantity.unit, quantity.unit.factor) == before


@pytest.mark.parametrize('value', [0, -2, 2**60 + 1, 0.1, Fraction(3, 7), Decimal('1.2345'), 2j])
@pytest.mark.parametrize('unit', [Unit('cm'), Unit('cm', scale=pi), Unit('km/m')])
@pytest.mark.parametrize('factor', [one, Factor(2), pi, 1 / (2 * pi), Factor(2) ** Fraction(1, 2)])
def test_quantity_factor_operations(value, unit: Unit, factor) -> None:
    quantity = Quantity(value, unit)
    before = (repr(quantity), hash(quantity))
    with localcontext() as context:
        context.prec = 2
        products = (quantity * factor, factor * quantity)
        quotient = quantity / factor
    for result in products:
        assert result.value is value
        assert result.unit == unit * factor
        assert (result / factor).unit == unit
        assert result == value * (unit * factor)
    assert quotient.value is value
    assert quotient.unit == unit / factor
    assert (quotient * factor) == quantity
    assert hash(quotient * factor) == hash(quantity)
    assert {quantity: 'found'}[quotient * factor] == 'found'
    assert (repr(quantity), hash(quantity)) == before


@pytest.mark.parametrize('value', [2, Fraction(3, 7), Decimal('1.2345'), 2j])
@pytest.mark.parametrize('factor', [Factor(2), pi])
def test_quantity_factor_reciprocal(value, factor) -> None:
    quantity = Quantity(value, Unit('cm', scale=pi))
    result = factor / quantity
    reciprocal = 1 / quantity
    assert result.value == reciprocal.value
    assert type(result.value) is type(reciprocal.value)
    assert result.unit == factor / quantity.unit
    assert result == reciprocal * factor


def test_quantity_factor_scaling() -> None:
    quantity = Quantity(3, 'cm')
    numeric = quantity * 2
    exact = quantity * Factor(2)
    assert numeric.value == 6
    assert numeric.unit == quantity.unit
    assert exact.value == 3
    assert exact.unit == Unit('cm', scale=Factor(2))
    assert exact == numeric
    assert hash(exact) == hash(numeric)
    assert (exact.to('m').to(exact.unit)).value == exact.value


def test_quantity_scaled_operations() -> None:
    minute = Factor(60) * Unit('s')
    quantity = Quantity(1, minute)
    for product in (quantity * Unit(), Unit() * quantity):
        assert product.to('s').value == 60
    assert (quantity / Unit()).to('s').value == 60
    assert (quantity**2).to('s2').value == 3600
    assert (quantity / (2 * Unit('m'))).to('s/m').value == 30
    assert (Quantity(Fraction(2), 'm') / quantity).to('m/s').value == Fraction(1, 30)
    assert (1 / minute).to('1/s').value == Fraction(1, 60)
    assert Quantity(2, 's').to(minute).to('s') == Quantity(2, 's')
    assert eval(repr(quantity)) == quantity


def test_quantity_scaled_copy() -> None:
    unit = Unit('cm', scale=pi / 180)
    quantity = Quantity(Decimal('1.25'), unit)
    for rebuilt in (copy(quantity), deepcopy(quantity), pickle.loads(pickle.dumps(quantity))):
        assert rebuilt == quantity
        assert type(rebuilt.value) is Decimal
        assert rebuilt.value == quantity.value
        assert type(rebuilt.unit) is Unit
        assert rebuilt.unit.scale == unit.scale
        assert rebuilt.unit.factor == unit.factor
        assert {quantity: 'found'}[rebuilt] == 'found'


@pytest.mark.parametrize('unit', ['', 'm*s', 'm2', 'm^(1/2)', '1/(m*s)', 'µm/ms'])
def test_quantity_repr_roundtrip(unit: str) -> None:
    quantity = Quantity(Fraction(3, 2), unit)
    rebuilt = eval(repr(quantity))
    assert rebuilt == quantity
    assert rebuilt.value == quantity.value
    assert rebuilt.unit == quantity.unit
    assert hash(rebuilt) == hash(quantity)


@pytest.mark.parametrize('as_unit', [False, True])
@pytest.mark.parametrize(
    ('quantity', 'target', 'expected', 'expected_type'),
    [
        (Quantity(1, 'cm'), 'm', Fraction(1, 100), Fraction),
        (Quantity(Fraction(1, 3), 'm'), 'cm', Fraction(100, 3), Fraction),
        (Quantity(Decimal('1.25'), 'm'), 'm', Decimal('1.25'), Decimal),
        (Quantity(Decimal('1'), 'm'), 'm', Decimal(1), Decimal),
        (Quantity(Decimal('1.2345'), 'g/cm3'), 'kg/m3', Decimal('1.2e3'), Decimal),
        (Quantity(0.5, 'm'), 'cm', 50.0, float),
        (Quantity(2**53 + 1, 'm'), 'cm', (2**53 + 1) * 100, int),
        (Quantity(1, 'km/m'), '', 1000, int),
        (Quantity(1, Unit('cm', scale=Factor(2))), 'cm', 2, int),
        (Quantity(1, 'm'), Unit('cm', scale=Factor(2)), 50, int),
        (Quantity(2j, 'cm'), 'm', 0.02j, complex),
        (Quantity(1, 'mm^(1/2)'), 'm^(1/2)', math.sqrt(0.001), float),
        (Quantity(1, pi * Unit('m')), 'm', math.pi, float),
    ],
)
def test_quantity_value_in(
    quantity: Quantity, target, expected, expected_type, as_unit: bool
) -> None:
    unit = Unit(target) if as_unit and isinstance(target, str) else target
    before = (quantity.value, quantity.unit, repr(quantity), hash(quantity))
    with localcontext() as context:
        context.prec = 2
        result = quantity.value_in(unit)
        converted = quantity.to(unit)
    assert result == (pytest.approx(expected) if expected_type in (float, complex) else expected)
    assert type(result) is expected_type
    assert result == converted.value
    assert type(result) is type(converted.value)
    assert (quantity.value, quantity.unit, repr(quantity), hash(quantity)) == before
    if isinstance(unit, Unit):
        assert converted.unit is unit


@pytest.mark.parametrize('unit', ['s', Unit('s')])
def test_quantity_value_in_invalid_dimension(unit) -> None:
    with pytest.raises(DimensionError, match='Cannot convert quantity with dimension'):
        Quantity(1, 'm').value_in(unit)


def test_quantity_value_in_invalid_symbol() -> None:
    with pytest.raises(UnitSymbolError, match='is not a valid unit symbol'):
        Quantity(1, 'm').value_in('invalid_unit')


def test_quantity_conversion() -> None:
    quantity = Quantity[int](1, 'cm')
    converted = quantity.to('m')
    assert converted.value == Fraction(1, 100)
    assert type(converted.value) is Fraction
    assert converted.unit == Unit('m')
    assert converted.to('cm') == quantity
    assert quantity.value == 1
    assert quantity.unit == Unit('cm')


def test_quantity_float_annotation() -> None:
    quantity = Quantity[float](1, 'cm')
    converted = quantity.to('m')
    assert converted.value == Fraction(1, 100)
    assert type(converted.value) is Fraction


def test_quantity_decimal_roundtrip() -> None:
    quantity = Quantity(Decimal('1.2345'), 'g/cm3')
    with localcontext() as context:
        context.prec = 2
        converted = quantity.to('kg/m3')
        restored = converted.to(quantity.unit)
    assert converted.value == Decimal('1.2e3')
    assert type(converted.value) is Decimal
    assert restored.value == Decimal('1.2')
    assert type(restored.value) is Decimal
    assert restored != quantity
    # Use Fraction explicitly when a context-independent exact round trip is needed.
    exact = Quantity(Fraction(quantity.value), quantity.unit)
    assert exact.to('kg/m3').to(quantity.unit) == exact


def test_quantity_irrational_conversion() -> None:
    quantity = Quantity(1, 'mm^(1/2)')
    converted = quantity.to('m^(1/2)')
    assert converted.value == pytest.approx(math.sqrt(0.001))
    assert type(converted.value) is float
    assert converted.to(quantity.unit).value == pytest.approx(quantity.value)


@pytest.mark.parametrize('value', [0, -2, Fraction(3, 2), Decimal('1.25'), 1.5])
def test_quantity_equality(value) -> None:
    quantity = Quantity(value, 'm')
    equivalent = Quantity(value * 100, 'cm')
    assert quantity == equivalent
    assert hash(quantity) == hash(equivalent)
    assert {quantity: 'found'}[equivalent] == 'found'
    assert quantity != Quantity(value, 's')
    assert quantity != object()


@pytest.mark.parametrize(
    ('quantity', 'number'),
    [
        (Quantity(1, ''), 1),
        (Quantity(1, 'km/m'), 1000),
        (Quantity(1, 'm/km'), Fraction(1, 1000)),
        (Quantity(Decimal('0.1'), ''), Decimal('0.1')),
        (Quantity(0, '(km/m)^(1/2)'), 0),
        (Quantity(-2, 'km/m'), -2000),
        (Quantity(0.5, ''), 0.5),
    ],
)
def test_quantity_scalar_hash(quantity: Quantity, number) -> None:
    assert quantity == number
    assert number == quantity
    assert hash(quantity) == hash(number)
    assert {quantity: 'found'}[number] == 'found'
    assert {number: 'found'}[quantity] == 'found'
    assert len({quantity, number}) == 1


def test_quantity_float_equality() -> None:
    quantity = Quantity(0.1, '')
    assert quantity == 0.1
    assert quantity != Decimal('0.1')
    assert quantity != Fraction(1, 10)
    assert hash(quantity) == hash(0.1)


@pytest.mark.parametrize(
    ('left', 'right', 'expected'),
    [
        (Quantity(0.1 + 0.2, 'm'), Quantity(0.3, 'm'), True),
        (Quantity(1, 'm'), Quantity(100 + 1e-8, 'cm'), True),
        (Quantity(1, 'm'), Quantity(101, 'cm'), False),
        (Quantity(-1, 'm'), Quantity(-100 - 1e-8, 'cm'), True),
        (Quantity(0, 'm'), Quantity(1e-20, 'cm'), False),
        (Quantity(1, 'km/m'), Quantity(1000, ''), True),
        (Quantity(1, pi * Unit('m')), Quantity(math.pi, 'm'), True),
        (Quantity(1000, 'mm^(1/2)'), Quantity(1, 'km^(1/2)'), True),
        (Quantity(math.inf, 'm'), Quantity(math.inf, 'cm'), True),
        (Quantity(-math.inf, 'm'), Quantity(-math.inf, 'cm'), True),
        (Quantity(math.inf, 'm'), Quantity(-math.inf, 'cm'), False),
        (Quantity(math.inf, 'm'), Quantity(10**400, 'cm'), False),
        (Quantity(math.nan, 'm'), Quantity(math.nan, 'cm'), False),
        (Quantity(Decimal('Infinity'), 'm'), Quantity(math.inf, 'cm'), True),
        (Quantity(Decimal('NaN'), 'm'), Quantity(0, 'cm'), False),
    ],
)
def test_quantity_isclose(left: Quantity, right: Quantity, expected: bool) -> None:
    before = (repr(left), repr(right))
    assert left.isclose(right) is expected
    assert right.isclose(left) is expected
    assert (repr(left), repr(right)) == before


@pytest.mark.parametrize('sign', [-1, 1])
def test_quantity_isclose_relative(sign: int) -> None:
    left, right = Quantity(sign, 'm'), Quantity(90 * sign, 'cm')
    # Only a symmetric threshold based on the larger magnitude includes this boundary.
    for a, b in ((left, right), (right, left)):
        assert a.isclose(b, rel_tol=Fraction(1, 10))
        assert not a.isclose(b, rel_tol=Fraction(99, 1000))


@pytest.mark.parametrize('tolerance', [Quantity(1, 'mm'), Quantity(Fraction(1, 10), 'cm')])
def test_quantity_isclose_absolute(tolerance: Quantity) -> None:
    left, right = Quantity(0, 'm'), Quantity(Decimal('0.1'), 'cm')
    for a, b in ((left, right), (right, left)):
        assert a.isclose(b, rel_tol=0, abs_tol=tolerance)
        assert not a.isclose(b, rel_tol=0, abs_tol=Quantity(Decimal('0.999'), 'mm'))


def test_quantity_isclose_scaled() -> None:
    left, right = Quantity(1, 'km/m'), Quantity(Decimal('1000.25'), '')
    for a, b in ((left, right), (right, left)):
        assert a.isclose(b, rel_tol=0, abs_tol=Decimal('0.25'))
        assert not a.isclose(b, rel_tol=0, abs_tol=Decimal('0.249'))
    assert left.isclose(1000)
    assert not left.isclose(1)
    assert left.isclose(Decimal('1000.25'), rel_tol=0, abs_tol=Fraction(1, 4))


@pytest.mark.parametrize('magnitude', [2**53, 10**400, Fraction(1, 10**400)])
@pytest.mark.parametrize('unit', ['m', 'mm^(1/2)'])
def test_quantity_isclose_precision(magnitude, unit: str) -> None:
    difference = Fraction(1, 10**410) if magnitude < 1 else 1
    left, right = Quantity(magnitude, unit), Quantity(magnitude + difference, unit)
    relative = Fraction(difference) / (2 * abs(magnitude))
    for a, b in ((left, right), (right, left)):
        assert not a.isclose(b, rel_tol=0)
        assert not a.isclose(b, rel_tol=relative)
        assert a.isclose(b, rel_tol=relative * 2)


def test_quantity_isclose_decimal() -> None:
    left, right = Quantity(Decimal('1.2345'), 'm'), Quantity(Decimal('123.46'), 'cm')
    with localcontext() as context:
        context.prec = 2
        for a, b in ((left, right), (right, left)):
            assert a.isclose(b, rel_tol=Decimal('0.0001'))
            assert not a.isclose(b, rel_tol=Decimal('0.00001'))
            assert a.isclose(b, rel_tol=0, abs_tol=Quantity(Decimal('0.0001'), 'm'))


def test_quantity_isclose_exact() -> None:
    left, right = Quantity(0.1, ''), Quantity(Fraction(1, 10), '')
    assert left.isclose(right)
    assert not left.isclose(right, rel_tol=0)
    assert not right.isclose(left, rel_tol=0)
    large_scale = pi**10000 * Unit('m')
    assert Quantity(1, large_scale).isclose(Quantity(1, large_scale), rel_tol=0)
    assert not Quantity(1, large_scale).isclose(Quantity(2, large_scale), rel_tol=0)


@pytest.mark.parametrize('other', [Quantity(1, 's'), 1])
def test_quantity_isclose_invalid_dimension(other) -> None:
    with pytest.raises(DimensionError, match='Cannot compare quantities with different dimensions'):
        Quantity(1, 'm').isclose(other)


@pytest.mark.parametrize('tolerance', [Quantity(1, 's'), Quantity(0, 's'), 1])
def test_quantity_isclose_invalid_tolerance_dimension(tolerance) -> None:
    with pytest.raises(DimensionError, match='Absolute tolerance must have the same dimension'):
        Quantity(1, 'm').isclose(Quantity(1, 'm'), abs_tol=tolerance)


@pytest.mark.parametrize('tolerance', [-1, math.inf, math.nan, Decimal('-Infinity')])
def test_quantity_isclose_invalid_relative(tolerance) -> None:
    with pytest.raises(ValueError, match='rel_tol must be finite and nonnegative'):
        Quantity(1, 'm').isclose(Quantity(1, 'm'), rel_tol=tolerance)


@pytest.mark.parametrize('tolerance', [-1, math.inf, math.nan, Decimal('-Infinity')])
def test_quantity_isclose_invalid_absolute(tolerance) -> None:
    for absolute in (tolerance, Quantity(tolerance, '')):
        with pytest.raises(ValueError, match='abs_tol must be finite and nonnegative'):
            Quantity(1, '').isclose(1, abs_tol=absolute)


@pytest.mark.parametrize('value', ['foo', 1j, object()])
def test_quantity_isclose_invalid_other(value) -> None:
    with pytest.raises(TypeError, match='other must be a Quantity or real scalar'):
        Quantity(1, '').isclose(value)


@pytest.mark.parametrize('left', [Quantity(1j, 'm'), Quantity(1, 'm')])
def test_quantity_isclose_invalid_magnitude(left: Quantity) -> None:
    with pytest.raises(TypeError, match=r'isclose\(\) requires real scalar magnitudes'):
        left.isclose(Quantity(1j, 'm'))


def test_quantity_isclose_invalid_relative_type() -> None:
    with pytest.raises(TypeError, match='rel_tol must be a real scalar'):
        Quantity(1, '').isclose(1, rel_tol='foo')


def test_quantity_isclose_invalid_absolute_type() -> None:
    with pytest.raises(TypeError, match='abs_tol must be a Quantity or real scalar'):
        Quantity(1, '').isclose(1, abs_tol='foo')


def test_quantity_exact_comparison() -> None:
    smaller = Quantity(10**16, 'mm^(1/2)')
    larger = Quantity(10**16 + 1, 'mm^(1/2)')
    assert smaller != larger
    assert smaller < larger
    assert smaller <= larger
    assert larger > smaller
    assert larger >= smaller
    assert (smaller - larger).value == -1
    assert len({smaller, larger}) == 2


def test_quantity_irrational_hash() -> None:
    quantity = Quantity(1000, 'mm^(1/2)')
    equivalent = Quantity(1, 'km^(1/2)')
    assert quantity == equivalent
    assert not quantity < equivalent
    assert quantity <= equivalent
    assert hash(quantity) == hash(equivalent)
    assert {quantity: 'found'}[equivalent] == 'found'
    assert quantity != Quantity(quantity.to('m^(1/2)').value, 'm^(1/2)')
    ratio = Quantity(1, '(km/m)^(1/2)')
    assert ratio != math.sqrt(1000)
    assert math.sqrt(1000) != ratio


@pytest.mark.parametrize('value', [0, -2, Fraction(3, 2), Decimal('0.1')])
def test_quantity_mixed_comparison(value) -> None:
    quantity = Quantity(value, 'm')
    assert quantity == Quantity(Fraction(value) * 100, 'cm')
    assert quantity < Quantity(Fraction(value) * 100 + 1, 'cm')
    assert quantity > Quantity(Fraction(value) * 100 - 1, 'cm')


def test_quantity_irrational_ordering() -> None:
    quantity = Quantity(1, 'mm^(1/2)')
    assert quantity < Quantity(1, 'm^(1/2)')
    assert quantity > Quantity(Fraction(1, 100), 'm^(1/2)')
    dimensionless = Quantity(1, '(km/m)^(1/2)')
    assert 31 < dimensionless < 32
    assert dimensionless >= 31
    assert dimensionless <= 32


@pytest.mark.parametrize('value', [0, -1, 1, Fraction(3, 2), Decimal('1.25'), 0.5])
def test_quantity_dimensionless_arithmetic(value) -> None:
    quantity = Quantity(value, 'km/m')
    base_value = Fraction(value) * 1000
    for result in (quantity + 1, 1 + quantity, quantity + Quantity(1, '')):
        assert result.unit == quantity.unit
        assert result.to('').value == base_value + 1
    assert (quantity - 1).to('').value == base_value - 1
    assert (1 - quantity).to('').value == 1 - base_value


def test_quantity_add_subtract() -> None:
    quantity = Quantity(1, 'm')
    other = Quantity(50, 'cm')
    assert (quantity + other).value == Fraction(3, 2)
    assert (quantity + other).unit == quantity.unit
    assert (other + quantity).value == 150
    assert (other + quantity).unit == other.unit
    assert quantity + other == other + quantity
    assert (quantity - other).value == Fraction(1, 2)
    assert (other - quantity).value == -50


def test_quantity_decimal_arithmetic() -> None:
    quantity = Quantity(Decimal('1.2345'), 'm')
    other = Quantity(Decimal('12.345'), 'cm')
    with localcontext() as context:
        context.prec = 2
        total = quantity + other
        difference = quantity - other
        assert (quantity + quantity).value == Decimal('2.5')
        assert (quantity - quantity).value == Decimal(0)
    assert total.value == Decimal('1.4')
    assert difference.value == Decimal('1.1')
    assert type(total.value) is type(difference.value) is Decimal
    assert total.unit is difference.unit is quantity.unit


def test_quantity_decimal_alignment() -> None:
    quantity = Quantity(Decimal('1.00'), 'm')
    other = Quantity(50, 'cm')
    assert (quantity + other).value == Decimal('1.50')
    assert type((quantity + other).value) is Decimal
    assert (other + quantity).value == Decimal('150')
    assert type((other + quantity).value) is Decimal
    assert (quantity - other).value == Decimal('0.50')
    assert (other - quantity).value == Decimal('-50')
    assert quantity + other == other + quantity


def test_quantity_decimal_scalar() -> None:
    quantity = Quantity(Decimal('0.012345'), 'km/m')
    scalar = Decimal('0.12345')
    with localcontext() as context:
        context.prec = 2
        total = quantity + scalar
        difference = scalar - quantity
        assert total == scalar + quantity
    assert total.value == Decimal('0.012')
    assert difference.value == Decimal('-0.012')
    assert type(total.value) is type(difference.value) is Decimal
    assert (Quantity(Decimal(1), 'km/m') + 1).value == Decimal('1.001')
    assert (1 - Quantity(Decimal(1), 'km/m')).value == Decimal('-0.999')


@pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize('value', [2, Fraction(2), Decimal('2'), 2.0, 2j])
@pytest.mark.parametrize('scalar', [3, Fraction(1, 3), Decimal('0.3'), 0.3, 0.3j])
def test_quantity_native_arithmetic(op, value, scalar) -> None:
    quantity = Quantity(value, '')
    for left, right in ((value, scalar), (scalar, value)):
        operands = (quantity, scalar) if left is value else (scalar, quantity)
        try:
            expected = op(left, right)
        except TypeError:
            with pytest.raises(TypeError):
                op(*operands)
            continue
        result = op(*operands)
        assert result.value == expected
        assert type(result.value) is type(expected)
        assert result.unit == Unit()


@pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize('value', [Fraction(1, 3), 0.1, 0.1j])
def test_quantity_decimal_invalid_mixed(op, value) -> None:
    decimal = Quantity(Decimal('1.25'), 'm')
    other = Quantity(value, 'm')
    with pytest.raises(TypeError):
        op(decimal, other)
    with pytest.raises(TypeError):
        op(other, decimal)


def test_quantity_fraction_scalar() -> None:
    converted = Quantity(1, 'cm').to('m')
    result = converted * converted.value
    assert result.value == Fraction(1, 10000)
    assert type(result.value) is Fraction


def test_quantity_decimal_multiply() -> None:
    quantity = Quantity(Decimal('1.2345'), 'm')
    other = Quantity(Decimal('0.5'), 's')
    with localcontext() as context:
        context.prec = 2
        product = quantity * other
        quotient = quantity / other
        scalar_product = quantity * Decimal('0.5')
    assert product.value == scalar_product.value == Decimal('0.62')
    assert quotient.value == Decimal('2.5')
    assert type(product.value) is type(quotient.value) is Decimal
    assert product.unit == Unit('m*s')
    assert quotient.unit == Unit('m/s')


@pytest.mark.parametrize('value', [2**53 + 1, 10**400, Fraction(3, 2), Decimal('1.2345')])
def test_quantity_unit_operations(value) -> None:
    quantity = Quantity(value, 'm')
    unit = Unit('s')
    product = quantity * unit
    quotient = quantity / unit
    assert product.value is value
    assert quotient.value is value
    assert product.unit == Unit('m*s')
    assert quotient.unit == Unit('m/s')
    assert unit * quantity == product


def test_quantity_reflected_division() -> None:
    result = Unit('m') / Quantity(Fraction(2), 's')
    assert result.value == Fraction(1, 2)
    assert type(result.value) is Fraction
    assert result.unit == Unit('m/s')


@pytest.mark.parametrize(
    ('rounding', 'positive_value', 'negative_value'),
    [(ROUND_DOWN, Decimal('1.2'), Decimal('-1.2')), (ROUND_UP, Decimal('1.3'), Decimal('-1.3'))],
)
def test_quantity_unary(rounding: str, positive_value: Decimal, negative_value: Decimal) -> None:
    quantity = Quantity(Decimal('1.2345'), 'm')
    with localcontext() as context:
        context.prec = 2
        context.rounding = rounding
        context.clear_flags()
        positive, negative = +quantity, -quantity
        assert context.flags[Inexact]
        assert context.flags[Rounded]
    assert positive.value == positive_value
    assert negative.value == negative_value
    assert type(positive.value) is type(negative.value) is Decimal
    assert positive.unit is negative.unit is quantity.unit
    assert quantity.value == Decimal('1.2345')


@pytest.mark.parametrize('op', [operator.pos, operator.neg])
@pytest.mark.parametrize('signal', [Inexact, Rounded])
def test_quantity_unary_traps(op, signal) -> None:
    quantity = Quantity(Decimal('1.2345'), 'm')
    with localcontext() as context:
        context.prec = 2
        context.traps[signal] = True
        with pytest.raises(signal):
            op(quantity)


@pytest.mark.parametrize('op', [operator.pos, operator.neg])
@pytest.mark.parametrize('value', [Decimal('-0.00'), Decimal('Infinity'), Decimal('NaN')])
def test_quantity_unary_special(op, value: Decimal) -> None:
    quantity = Quantity(value, 'm')
    result = op(quantity)
    assert type(result.value) is Decimal
    assert result.value.as_tuple() == op(value).as_tuple()
    assert result.unit is quantity.unit
    assert quantity.value is value


@pytest.mark.parametrize('exponent', [Fraction(1, 2), Decimal('0.5'), 0.5])
def test_quantity_rational_power(exponent) -> None:
    quantity = Quantity(4, 'cm2')
    root = quantity**exponent
    assert root.value == pytest.approx(2)
    assert root.unit == Unit('cm')
    assert root**2 == quantity


def test_quantity_decimal_power() -> None:
    quantity = Quantity(Decimal('1.2345'), 'm')
    with localcontext() as context:
        context.prec = 2
        squared = quantity ** Decimal(2)
        root = Quantity(Decimal(4), 'm2') ** Decimal('0.5')
    assert squared.value == Decimal('1.5')
    assert type(squared.value) is Decimal
    assert squared.unit == Unit('m2')
    assert root.value == pytest.approx(2)
    assert root.unit == Unit('m')


def test_quantity_complex_power() -> None:
    root = Quantity(-4, 'm2') ** Fraction(1, 2)
    assert type(root.value) is complex
    assert root.value == pytest.approx(2j)
    assert root.unit == Unit('m')


@pytest.mark.parametrize('exponent', [0.1, 1e-10, 1.000000000001])
def test_quantity_float_power(exponent: float) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        result = Quantity(2, 'm') ** exponent
    assert result.value == 2**exponent
    assert type(result.value) is float
    assert result.unit == Unit('m') ** exponent
    assert result.dimension.L == Fraction(str(exponent))


def test_quantity_state_preservation() -> None:
    quantity = Quantity(Fraction(3, 2), 'cm/s')
    before = (quantity.value, quantity.unit, repr(quantity), hash(quantity))
    lookup = {quantity: 'found'}
    product = quantity
    product *= Quantity(2, 's')
    converted = quantity.to('m/s')
    assert product.unit == Unit('cm')
    assert converted.unit == Unit('m/s')
    assert (quantity.value, quantity.unit, repr(quantity), hash(quantity)) == before
    assert lookup[quantity] == 'found'


@pytest.mark.parametrize('value', [math.inf, -math.inf, math.nan])
def test_quantity_nonfinite_comparison(value: float) -> None:
    quantity = Quantity(value, 'mm^(1/2)')
    equivalent = Quantity(value, 'km^(1/2)')
    if math.isnan(value):
        assert quantity != equivalent
        assert quantity != quantity
    else:
        assert quantity == equivalent
        assert hash(quantity) == hash(equivalent)
        assert (quantity > Quantity(1, 'm^(1/2)')) == (value > 1)


def test_quantity_invalid_conversion() -> None:
    with pytest.raises(DimensionError, match='Cannot convert quantity with dimension'):
        Quantity(1, 'm').to('s')


@pytest.mark.parametrize('op', [operator.add, operator.sub])
def test_quantity_invalid_addition(op) -> None:
    with pytest.raises(
        DimensionError, match='Cannot operate on quantities with different dimensions'
    ):
        op(Quantity(1, 'm'), Quantity(1, 's'))


@pytest.mark.parametrize('op', [operator.lt, operator.le, operator.gt, operator.ge])
def test_quantity_invalid_comparison(op) -> None:
    with pytest.raises(DimensionError, match='Cannot compare quantities with different dimensions'):
        op(Quantity(1, 'm'), Quantity(1, 's'))


@pytest.mark.parametrize('op', [operator.add, operator.sub, operator.lt])
def test_quantity_invalid_scalar(op) -> None:
    quantity = Quantity(1, 'm')
    with pytest.raises(TypeError, match='not supported|unsupported operand'):
        op(quantity, 1)
    with pytest.raises(TypeError, match='not supported|unsupported operand'):
        op(1, quantity)


@pytest.mark.parametrize('op', [operator.mul, operator.truediv])
def test_quantity_invalid_operand(op) -> None:
    quantity = Quantity(1, 'm')
    with pytest.raises(TypeError, match='unsupported operand'):
        op(quantity, object())
    with pytest.raises(TypeError, match='unsupported operand'):
        op(object(), quantity)


def test_quantity_invalid_power() -> None:
    with pytest.raises(TypeError, match='unsupported operand'):
        Quantity(1, 'm') ** '1/2'


def test_quantity_invalid_boolean_power() -> None:
    with pytest.raises(TypeError, match='Cannot convert bool to Fraction'):
        Quantity(1, 'm') ** True


def test_quantity_invalid_symbol() -> None:
    with pytest.raises(UnitSymbolError, match='is not a valid unit symbol'):
        Quantity(1, 'invalid_unit')


def test_quantity_zero_division() -> None:
    with pytest.raises(ZeroDivisionError):
        1 / Quantity(0, 'm')
    with pytest.raises(ZeroDivisionError):
        Quantity(1, 'm') / 0


@pytest.mark.parametrize('value', [Decimal('1.2345'), Decimal('-0.00'), Decimal('sNaN')])
def test_quantity_conversion_identity(value: Decimal) -> None:
    quantity = Quantity(value, 'rad')
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = True
        context.clear_flags()
        assert quantity.value_in('') is value
        assert quantity.to('').value is value
        assert not any(context.flags.values())


def test_quantity_decimal_conversion() -> None:
    unit = Unit('m', scale=Factor(Fraction(9, 10)))
    with localcontext() as context:
        context.prec = 2
        context.Emax = 100
        context.clear_flags()
        result = Quantity(Decimal('9e100'), unit).to('m')
        assert result.value == Decimal('8.1e100')
        assert not context.flags[Inexact]
        context.traps[Inexact] = True
        with pytest.raises(Inexact):
            Quantity(Decimal('1.2345'), 'm').to('cm')


@pytest.mark.parametrize('value', [Decimal('-0.00'), Decimal('Infinity'), Decimal('NaN')])
def test_quantity_decimal_special(value: Decimal) -> None:
    converted = Quantity(value, 'm').to('cm')
    assert type(converted.value) is Decimal
    assert converted.value.as_tuple() == value.as_tuple()


def test_quantity_decimal_irrational() -> None:
    quantity = Quantity(Decimal(1), pi * Unit('m'))
    result = quantity.to('m')
    assert type(result.value) is Decimal
    assert result.value == Decimal(str(math.pi))
    assert quantity.to('m').to(quantity.unit).isclose(quantity)


@pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv])
def test_quantity_decimal_traps(op) -> None:
    left = Quantity(Decimal('1.2345'), '')
    right = Quantity(Decimal(3), '')
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = True
        for a, b in ((left, right), (left, right.value), (right.value, left)):
            with pytest.raises(Inexact):
                op(a, b)


class _OrderedMagnitude:
    """Protocol substitute recording operand order, without assuming commutativity."""

    def __init__(self, expression: str) -> None:
        self.expression = expression

    def __str__(self) -> str:
        return self.expression

    def _binary(self, other, symbol: str, reverse: bool = False):
        if isinstance(other, (Quantity, Unit)):
            return NotImplemented
        left, right = (other, self) if reverse else (self, other)
        return _OrderedMagnitude(f'({left}{symbol}{right})')

    def __add__(self, other):
        return self._binary(other, '+')

    def __radd__(self, other):
        return self._binary(other, '+', True)

    def __sub__(self, other):
        return self._binary(other, '-')

    def __rsub__(self, other):
        return self._binary(other, '-', True)

    def __mul__(self, other):
        return self._binary(other, '*')

    def __rmul__(self, other):
        return self._binary(other, '*', True)

    def __truediv__(self, other):
        return self._binary(other, '/')

    def __rtruediv__(self, other):
        return self._binary(other, '/', True)

    def __pos__(self):
        return _OrderedMagnitude(f'(+{self})')

    def __neg__(self):
        return _OrderedMagnitude(f'(-{self})')

    def __pow__(self, other):
        return self._binary(other, '**')

    def __eq__(self, other):
        if not isinstance(other, _OrderedMagnitude):
            return NotImplemented
        return self.expression == other.expression


@pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv])
def test_quantity_custom_order(op) -> None:
    left, right = _OrderedMagnitude('left'), _OrderedMagnitude('right')
    quantity = Quantity(right, '')
    assert op(quantity, left).value == op(right, left)
    assert op(left, quantity).value == op(left, right)
    assert op(Quantity(left, ''), quantity).value == op(left, right)


def test_quantity_custom_conversion() -> None:
    value = _OrderedMagnitude('x')
    quantity = Quantity(value, 'm')
    assert quantity.to('m').value is value
    assert quantity.to('cm').value == value * 100
    assert (quantity * Unit('s')).value is value
    assert (Unit('s') * quantity).value is value


@pytest.mark.parametrize(
    'op', [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]
)
@pytest.mark.parametrize('unit', ['m', 'cm'])
@pytest.mark.parametrize('trapped', [False, True])
@pytest.mark.parametrize('decimal_type', [Decimal, _DecimalMagnitude])
@pytest.mark.parametrize('float_type', [float, _FloatMagnitude])
def test_quantity_decimal_comparison(
    op, unit: str, trapped: bool, decimal_type, float_type
) -> None:
    left, right = Quantity(decimal_type(1), 'm'), Quantity(float_type(100), unit)
    with localcontext() as context:
        context.traps[FloatOperation] = trapped
        context.clear_flags()
        if trapped and unit == 'm' and op not in (operator.eq, operator.ne):
            with pytest.raises(FloatOperation):
                op(left, right)
            with pytest.raises(FloatOperation):
                op(right, left)
        else:
            expected = 100 if unit == 'm' else 1
            assert op(left, right) is op(1, expected)
            assert op(right, left) is op(expected, 1)
        # Decimal flags can be bools or integer 0/1 depending on Python version.
        assert bool(context.flags[FloatOperation]) is (unit == 'm')


@pytest.mark.parametrize(
    'op', [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]
)
@pytest.mark.parametrize('value', [Decimal('NaN'), Decimal('sNaN'), Decimal('Infinity')])
@pytest.mark.parametrize('trapped', [False, True])
def test_quantity_comparison_special(op, value: Decimal, trapped: bool) -> None:
    with localcontext() as context:
        context.traps[InvalidOperation] = trapped
        context.clear_flags()
        try:
            expected = op(value, Decimal(0))
        except InvalidOperation:
            with pytest.raises(InvalidOperation):
                op(Quantity(value, pi * Unit('m')), Quantity(Decimal(0), 'm'))
        else:
            flags = context.flags.copy()
            context.clear_flags()
            result = op(Quantity(value, pi * Unit('m')), Quantity(Decimal(0), 'm'))
            assert result is expected
            assert dict(context.flags) == flags


@pytest.mark.parametrize('unit', ['', 'm', 'cm', 'km/m'])
@pytest.mark.parametrize('value', [0.1, 2**53, 0.0, -0.5])
def test_quantity_complex_real(unit: str, value: float) -> None:
    real, other = Quantity(value, unit), Quantity(complex(value), unit)
    assert real == other
    assert other == real
    assert not real != other
    assert hash(real) == hash(other)
    assert {real: 'found'}[other] == 'found'
    assert len({real, other}) == 1


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (Quantity(1 + 2j, 'm'), Quantity(100 + 200j, 'cm')),
        (Quantity(1 + 2j, 'km/m'), 1000 + 2000j),
        (Quantity(complex(math.inf, 1), 'm'), Quantity(complex(math.inf, 100), 'cm')),
        (Quantity(complex(0, math.inf), pi * Unit('m')), Quantity(complex(0, math.inf), 'm')),
    ],
)
def test_quantity_complex_hash(left: Quantity, right) -> None:
    assert left == right
    assert right == left
    assert hash(left) == hash(right)
    assert {left: 'found'}[right] == 'found'
    assert len({left, right}) == 1


@pytest.mark.parametrize('unit', ['', 'm', 'cm', 'km/m'])
@pytest.mark.parametrize('value', [complex(math.nan, 1), complex(1, math.nan)])
def test_quantity_complex_nan(unit: str, value: complex) -> None:
    quantity = Quantity(value, unit)
    assert quantity != quantity
    assert hash(quantity) == hash(quantity)
    assert len({quantity, quantity}) == 1


@pytest.mark.parametrize('op', [operator.lt, operator.le, operator.gt, operator.ge])
def test_quantity_complex_invalid_order(op) -> None:
    with pytest.raises(TypeError):
        op(Quantity(0j, 'm'), Quantity(1, 'cm'))


def test_quantity_irrational_range() -> None:
    large = Quantity(10**400, pi * Unit('m'))
    assert large < Quantity(4 * 10**400, 'm')
    assert large > Quantity(3 * 10**400, 'm')
    assert Quantity(0, 'm') < Quantity(1, pi**10000 * Unit('m'))
    assert Quantity(-1, pi**10000 * Unit('m')) < Quantity(0, 'm')


class _ComparisonMagnitude(_OrderedMagnitude):
    """Protocol substitute whose comparison needs no numeric scaling at scale one."""

    def __mul__(self, other):
        raise AssertionError('Unexpected magnitude multiplication')

    __rmul__ = __mul__

    def __ne__(self, other):
        # Deliberately independent of __eq__, to check native __ne__ dispatch.
        return True

    def __lt__(self, other):
        return self.expression < other.expression

    def __le__(self, other):
        return self.expression <= other.expression

    def __gt__(self, other):
        return self.expression > other.expression

    def __ge__(self, other):
        return self.expression >= other.expression

    def __hash__(self):
        return hash(self.expression)


@pytest.mark.parametrize('unit', ['', 'm', 'cm'])
def test_quantity_custom_comparison(unit: str) -> None:
    a, b = _ComparisonMagnitude('a'), _ComparisonMagnitude('b')
    left, right = Quantity(a, unit), Quantity(b, unit)
    for op in (operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge):
        assert op(left, right) is op(a, b)
    assert left == left
    assert left != left  # The magnitude's deliberately independent __ne__ is respected.
    if unit == '':
        assert left == a
        assert left != a
        assert hash(left) == hash(a)


class _UnhashableMagnitude(int):
    __hash__ = None


def test_quantity_custom_invalid_hash() -> None:
    with pytest.raises(TypeError, match='unhashable type'):
        hash(Quantity(_UnhashableMagnitude(2), 'cm'))


@pytest.mark.parametrize(
    'value', [0, 2, -2, Fraction(0), Fraction(1, 3), Decimal('-0.00'), Decimal('NaN'), 0j, 1j]
)
@pytest.mark.parametrize('unit', ['', 'm', 'km/m'])
def test_quantity_bool(value, unit: str) -> None:
    assert bool(Quantity(value, unit)) is bool(value)


@pytest.mark.parametrize('value', [2, 2.0, Fraction(2), Decimal(2), 2j])
@pytest.mark.parametrize(
    'exponent', [2, 2.0, Fraction(2), Decimal(2), Fraction(1, 2), Decimal('0.5')]
)
def test_quantity_native_power(value, exponent) -> None:
    quantity = Quantity(value, 'm')
    try:
        expected = value**exponent
    except TypeError:
        with pytest.raises(TypeError):
            quantity**exponent
    else:
        result = quantity**exponent
        assert result.value == expected
        assert type(result.value) is type(expected)
        assert result.unit == quantity.unit**exponent


@pytest.mark.parametrize('exponent', [2.0, Decimal(2), Fraction(1, 3)])
def test_quantity_custom_power(exponent) -> None:
    value = _OrderedMagnitude('x')
    quantity = Quantity(value, 'm')
    assert (quantity**exponent).value == value**exponent
    assert (quantity**exponent).unit == quantity.unit**exponent


def test_quantity_comparison_precision() -> None:
    left = Quantity(Decimal('1.2345'), 'm')
    equivalent = Quantity(Decimal('123.45'), 'cm')
    larger = Quantity(Decimal('123.46'), 'cm')
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = context.traps[Rounded] = True
        context.clear_flags()
        assert left == equivalent
        assert equivalent == left
        assert left != larger
        assert larger != left
        assert left < larger
        assert larger > left
        assert hash(left) == hash(equivalent)
        assert {left: 'found'}[equivalent] == 'found'
        assert not any(context.flags.values())


@pytest.mark.parametrize(
    ('value', 'subclass'),
    [
        (1, _IntMagnitude),
        (0.1, _FloatMagnitude),
        (Decimal('0.1'), _DecimalMagnitude),
        (Fraction(1, 10), _FractionMagnitude),
        (0.1 + 0.2j, _ComplexMagnitude),
    ],
)
@pytest.mark.parametrize('unit', [Unit('m'), Unit('cm'), Unit('km/m'), pi * Unit('m')])
def test_quantity_subclass_hash(value, subclass, unit: Unit) -> None:
    plain, derived = Quantity(value, unit), Quantity(subclass(value), unit)
    assert plain == derived
    assert derived == plain
    assert not plain != derived
    assert hash(plain) == hash(derived)
    assert {plain: 'found'}[derived] == 'found'
    assert {derived: 'found'}[plain] == 'found'
    assert len({plain, derived}) == 1


@pytest.mark.parametrize('op', [operator.add, operator.sub])
def test_quantity_subclass_alignment(op) -> None:
    left = Quantity(Decimal('1'), 'm')
    plain = Quantity(50, 'cm')
    derived = Quantity(_IntMagnitude(50), 'cm')
    assert op(left, plain) == op(left, derived)
    assert op(plain, left) == op(derived, left)
    assert type(op(left, derived).value) is Decimal
    assert type(op(derived, left).value) is Decimal


@pytest.mark.parametrize(
    'subclass', [_IntMagnitude, _FloatMagnitude, _DecimalMagnitude, _FractionMagnitude]
)
def test_quantity_subclass_conversion(subclass) -> None:
    value = subclass(1)
    quantity = Quantity(value, 'cm')
    exact = Quantity(Fraction(value, 100) if isinstance(value, int) else Fraction(value) / 100, 'm')
    assert quantity == exact
    assert exact == quantity
    assert hash(quantity) == hash(exact)
    assert {quantity: 'found'}[exact] == 'found'
