import math
import operator
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx.dimension import DIMENSIONLESS, Dimension
from unitx.exceptions import DimensionError, InexactFloatWarning, UnitSymbolError
from unitx.quantity import Quantity
from unitx.unit import Unit


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


@pytest.mark.parametrize('unit', ['', 'm*s', 'm2', 'm^(1/2)', '1/(m*s)', 'µm/ms'])
def test_quantity_repr_roundtrip(unit: str) -> None:
    quantity = Quantity(Fraction(3, 2), unit)
    rebuilt = eval(repr(quantity))
    assert rebuilt == quantity
    assert rebuilt.value == quantity.value
    assert rebuilt.unit == quantity.unit
    assert hash(rebuilt) == hash(quantity)


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
    assert converted.value == Fraction(2469, 2)
    assert type(converted.value) is Fraction
    assert restored.value == Fraction(2469, 2000)
    assert restored == quantity


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
        assert total == other + quantity
        assert difference == -(other - quantity)
        assert (quantity + quantity).value == Fraction(2469, 1000)
        assert (quantity - quantity).value == 0
    assert total.value == Fraction(27159, 20000)
    assert difference.value == Fraction(22221, 20000)
    assert type(total.value) is Fraction
    assert type(difference.value) is Fraction


def test_quantity_decimal_scalar() -> None:
    quantity = Quantity(Decimal('0.012345'), 'km/m')
    scalar = Decimal('0.12345')
    with localcontext() as context:
        context.prec = 2
        total = quantity + scalar
        difference = scalar - quantity
        assert total == scalar + quantity
    assert total.to('').value == Fraction(249369, 20000)
    assert difference.to('').value == Fraction(-244431, 20000)


@pytest.mark.parametrize('value', [2, Fraction(2), Decimal('2'), 2.0])
@pytest.mark.parametrize('scalar', [2, Fraction(1, 2), Decimal('0.5'), 0.5])
def test_quantity_scalar_operations(value, scalar) -> None:
    quantity = Quantity(value, 'm')
    magnitude = Fraction(value)
    multiplier = Fraction(scalar)
    assert quantity * scalar == scalar * quantity
    assert (quantity * scalar).value == magnitude * multiplier
    assert (quantity / scalar).value == magnitude / multiplier
    reciprocal = scalar / quantity
    assert reciprocal.value == multiplier / magnitude
    assert reciprocal.unit == Unit('m^-1')


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
    assert product.value == scalar_product.value == Fraction(2469, 4000)
    assert quotient.value == Fraction(2469, 1000)
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


def test_quantity_unary() -> None:
    quantity = Quantity(Decimal('1.2345'), 'm')
    with localcontext() as context:
        context.prec = 2
        positive, negative = +quantity, -quantity
    assert positive == quantity
    assert negative.value == Fraction(-2469, 2000)
    assert positive.unit == negative.unit == quantity.unit


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
        root = Quantity(Decimal(4), 'm2') ** Fraction(1, 2)
    assert squared.value == Fraction(2469, 2000) ** 2
    assert type(squared.value) is Fraction
    assert squared.unit == Unit('m2')
    assert root.value == pytest.approx(2)
    assert root.unit == Unit('m')


def test_quantity_complex_power() -> None:
    root = Quantity(-4, 'm2') ** Fraction(1, 2)
    assert type(root.value) is complex
    assert root.value == pytest.approx(2j)
    assert root.unit == Unit('m')


@pytest.mark.parametrize('exponent', [1e-10, 1.000000000001])
def test_quantity_float_power(exponent: float) -> None:
    with pytest.warns(InexactFloatWarning) as caught:
        result = Quantity(2, 'm') ** exponent
    assert len(caught) == 1
    normalized = 0 if exponent < 1 else 1
    assert result.value == 2**normalized
    assert result.unit == Unit('m') ** normalized


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
