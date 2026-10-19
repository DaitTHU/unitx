"""Static API contracts, checked by Pyright rather than executed by pytest."""

from decimal import Decimal
from fractions import Fraction
from typing import Any, assert_type

from unitx._numerics import Magnitude, Number, Real
from unitx.factor import PI, Factor
from unitx.quantity import Quantity
from unitx.unit import ScaledUnit, Unit
from unitx.unitregistry import PrefixPolicy


class _Magnitude(int):
    pass


def test_quantity_construction() -> None:
    assert_type(Quantity(1, 'm'), Quantity[int])
    assert_type(Quantity(1.0, 'm'), Quantity[float])
    assert_type(Quantity(Fraction(1), 'm'), Quantity[Fraction])
    assert_type(Quantity(Decimal(1), 'm'), Quantity[Decimal])
    assert_type(Quantity(1j, 'm'), Quantity[complex])
    assert_type(Quantity(1, 'm').value, int)


def test_quantity_isclose() -> None:
    left, right = Quantity(Decimal('1'), 'm'), Quantity(100.0, 'cm')
    assert_type(left.isclose(right), bool)
    assert_type(right.isclose(left, rel_tol=Fraction(1, 1000)), bool)
    assert_type(left.isclose(right, abs_tol=Quantity(1, 'mm')), bool)
    assert_type(Quantity(Fraction(1), '').isclose(Decimal('1'), abs_tol=Decimal('0.1')), bool)


def test_unit_value_operations[T: Magnitude](
    value: T, unit: Unit
) -> tuple[Quantity[T], Quantity[T]]:
    product = value * unit
    quotient = value / unit
    assert_type(product, Quantity[T])
    assert_type(quotient, Quantity[T])
    return product, quotient


def test_unit_numeric_construction() -> None:
    unit = Unit('m')
    assert_type(2 * unit, Quantity[int])
    assert_type(2 / unit, Quantity[int])
    assert_type(2.0 * unit, Quantity[float])
    assert_type(2.0 / unit, Quantity[float])
    assert_type(Decimal(2) * unit, Quantity[Decimal])
    assert_type(Decimal(2) / unit, Quantity[Decimal])
    assert_type(Fraction(2) * unit, Quantity[Fraction])
    assert_type(Fraction(2) / unit, Quantity[Fraction])
    assert_type(2j * unit, Quantity[complex])
    assert_type(2j / unit, Quantity[complex])
    assert_type(_Magnitude(2) * unit, Quantity[_Magnitude])
    assert_type(_Magnitude(2) / unit, Quantity[_Magnitude])
    assert_type(test_unit_value_operations(2, unit), tuple[Quantity[int], Quantity[int]])
    assert_type(
        test_unit_value_operations(Decimal(2), unit),
        tuple[Quantity[Decimal], Quantity[Decimal]],
    )
    assert_type(
        test_unit_value_operations(_Magnitude(2), unit),
        tuple[Quantity[_Magnitude], Quantity[_Magnitude]],
    )


def test_scaled_unit_construction() -> None:
    unit = Unit('m')
    scaled = Factor(2) * unit
    assert_type(scaled, ScaledUnit)
    assert_type(PI * unit, ScaledUnit)
    assert_type(Factor(2) / unit, ScaledUnit)
    assert_type(PI / scaled, ScaledUnit)
    assert_type(3 * scaled, Quantity[int])
    assert_type(Fraction(3) / scaled, Quantity[Fraction])
    assert_type(scaled * unit, ScaledUnit)
    # Base signatures keep mixed operations conservative; runtime dispatch promotes.
    assert_type(unit * scaled, ScaledUnit)
    assert_type(scaled / unit, ScaledUnit)
    assert_type(unit / scaled, ScaledUnit)
    assert_type(scaled * scaled, ScaledUnit)
    assert_type(scaled / scaled, ScaledUnit)
    assert_type(scaled**2, ScaledUnit)
    assert_type(scaled**0, ScaledUnit)
    assert_type(unit * unit, Unit)
    assert_type(unit / unit, Unit)
    assert_type(unit**2, Unit)
    assert_type(scaled.scale, Factor)
    assert_type(ScaledUnit('cm', scale=Factor(2)), ScaledUnit)


def test_unit_numeric_scaling(unit: Unit, scaled: ScaledUnit) -> None:
    for value in (Factor(2), PI):
        assert_type(unit * value, ScaledUnit)
        assert_type(unit / value, ScaledUnit)
        assert_type(scaled * value, ScaledUnit)
        assert_type(scaled / value, ScaledUnit)
    assert_type(unit * 2, Quantity[int])
    assert_type(unit * 0.1, Quantity[float])
    assert_type(unit * Fraction(1, 3), Quantity[Fraction])
    assert_type(unit * Decimal('0.1'), Quantity[Decimal])
    assert_type(scaled * 2, Quantity[int])
    assert_type(scaled * Decimal('0.1'), Quantity[Decimal])
    assert_type(unit / 60, Quantity[Fraction])
    assert_type(unit / Fraction(3, 2), Quantity[Fraction])
    assert_type(unit / Decimal('0.1'), Quantity[Fraction])
    assert_type(unit / 0.1, Quantity[float])
    assert_type(scaled / 60, Quantity[Fraction])
    assert_type(scaled / 0.1, Quantity[float])
    assert_type(unit * 2 / 3, Quantity[float])
    assert_type(3 * (unit * 2), Quantity[int])
    assert_type(Fraction(3) * (unit / 2), Quantity[Real])
    assert_type(Factor(2) * unit, ScaledUnit)
    assert_type(PI * scaled, ScaledUnit)
    assert_type(unit * Quantity(2, 's'), Quantity[int])
    assert_type(scaled * Quantity(2, 's'), Quantity[int])
    assert_type(unit / Quantity(2, 's'), Quantity[float])
    assert_type(scaled / Quantity(2, 's'), Quantity[float])


def test_unit_define() -> None:
    policy: PrefixPolicy = ('rvpolicy',)
    assert_type(Unit('m').define('rvpolicy', 'review policy', prefixable=policy), Unit)
    single_policy: PrefixPolicy = 'rvpolicy'
    assert_type(Unit('m').define('rvpolicy', 'review policy', prefixable=single_policy), Unit)
    assert_type((PI * Unit('m')).define('rvpolicy', 'review policy', prefixable='rvpolicy'), Unit)
    assert_type(
        Quantity(2, 'm').define('rvpolicy', 'review policy', prefixable='review policy'), Unit
    )
    assert_type(
        Unit('m').define(
            'rvbound',
            'review bound',
            prefixable=('rvbound', 'review bound'),
        ),
        Unit,
    )
    assert_type(
        (PI * Unit('m')).define('rvbound', 'review bound', prefixable=()),
        Unit,
    )
    assert_type(
        Quantity(2, 'm').define('rvbound', 'review bound', prefixable=('review bound',)),
        Unit,
    )
    assert_type(Unit('m').define('rvlen', name='reviewlength'), Unit)
    assert_type(Unit('m').define('rvlen', 'reviewlength'), Unit)
    assert_type(Unit('m').define('rvlen', 'reviewlength', 'reviewdistance', 'reviewspan'), Unit)
    assert_type((PI * Unit('m')).define('rvpi', name='reviewpilength'), Unit)
    assert_type((PI * Unit('m')).define('rvpi', 'reviewpilength', 'reviewpispan'), Unit)
    assert_type((60 * Unit('s')).define('rvmin', name='reviewminute'), Unit)
    assert_type((60 * Unit('s')).define('rvmin', 'reviewminute', 'reviewduration'), Unit)
    assert_type(Quantity(Decimal('0.3048'), 'm').define('rvft', name='reviewfoot'), Unit)


def test_quantity_value_in(quantity: Quantity[Magnitude]) -> None:
    assert_type(Quantity(1, 'm').value_in('cm'), Real)
    assert_type(Quantity(Fraction(1), 'm').value_in('cm'), Fraction | float)
    assert_type(Quantity(Decimal(1), 'm').value_in('cm'), Decimal)
    assert_type(Quantity(1.0, 'm').value_in('cm'), Real)
    assert_type(Quantity[float](1, 'cm').value_in('m'), Real)
    assert_type(Quantity(1, 'm').value_in(ScaledUnit('cm', Factor(2))), Real)
    assert_type(Quantity(1j, 'cm').value_in('m'), Magnitude)
    assert_type(quantity.value_in('m'), Magnitude)


def test_quantity_conversion() -> None:
    assert_type(Quantity(1, 'm').to('cm'), Quantity[Real])
    assert_type(Quantity(Fraction(1), 'm').to('cm'), Quantity[Fraction | float])
    assert_type(Quantity(Decimal(1), 'm').to('cm'), Quantity[Decimal])
    assert_type(Quantity(1.0, 'm').to('cm'), Quantity[Real])
    # A float annotation accepts int values too; scaling can then yield Fraction.
    assert_type(Quantity[float](1, 'cm').to('m'), Quantity[Real])


def test_quantity_unit_operations[T: Magnitude](value: T) -> Quantity[T]:
    quantity = Quantity(value, 'm')
    unit = Unit('s')
    assert_type(quantity, Quantity[T])
    assert_type(quantity.value, T)
    assert_type(quantity * unit, Quantity[T])
    assert_type(unit * quantity, Quantity[T])
    assert_type(quantity / unit, Quantity[T])
    return quantity


def test_quantity_factor_operations[T: Magnitude](value: T) -> Quantity[T]:
    quantity = Quantity(value, 'm')
    for factor in (Factor(2), PI):
        assert_type(quantity * factor, Quantity[T])
        assert_type(factor * quantity, Quantity[T])
        assert_type(quantity / factor, Quantity[T])
        assert_type(factor / quantity, Quantity[Magnitude])
        assert_type(Factor(2) / Quantity(2, 's'), Quantity[float])
        assert_type(PI / Quantity(Decimal(2), 's'), Quantity[Decimal])
    assert_type(Quantity(Decimal(2), 's') / (2 * PI), Quantity[Decimal])
    return quantity


def test_quantity_numeric_operations() -> None:
    integer = Quantity(1, 'm')
    rational = Quantity(Fraction(1), 'm')
    decimal = Quantity(Decimal(1), 'm')
    assert_type(integer + integer, Quantity[Real])
    assert_type(integer + rational, Quantity[Real])
    assert_type(decimal + integer, Quantity[Decimal])
    assert_type(integer - decimal, Quantity[Decimal])
    assert_type(integer * integer, Quantity[int])
    assert_type(integer * 2, Quantity[int])
    assert_type(2 * integer, Quantity[int])
    assert_type(integer * decimal, Quantity[Decimal])
    assert_type(integer / integer, Quantity[float])
    assert_type(integer / 2, Quantity[float])
    assert_type(2 / integer, Quantity[float])
    assert_type(Unit('s') / integer, Quantity[float])
    assert_type(decimal / integer, Quantity[Decimal])
    assert_type(decimal * Decimal(2), Quantity[Decimal])
    assert_type(Decimal(2) * decimal, Quantity[Decimal])
    assert_type(Decimal(2) / integer, Quantity[Decimal])


def test_quantity_unary() -> None:
    assert_type(+Quantity(1, 'm'), Quantity[int])
    assert_type(-Quantity(1.0, 'm'), Quantity[float])
    assert_type(-Quantity(Fraction(1), 'm'), Quantity[Fraction])
    assert_type(+Quantity(Decimal(1), 'm'), Quantity[Decimal])
    assert_type(-Quantity(Decimal(1), 'm'), Quantity[Decimal])
    assert_type(-Quantity(1j, 'm'), Quantity[complex])


def test_quantity_unary_generic(quantity: Quantity[Magnitude]) -> None:
    assert_type(+quantity, Quantity[Magnitude])
    assert_type(-quantity, Quantity[Magnitude])


def test_quantity_dimensionless() -> None:
    quantity = Quantity(1, 'km/m')
    assert_type(quantity + 1, Quantity[Real])
    assert_type(1 + quantity, Quantity[Real])
    assert_type(quantity - Decimal(1), Quantity[Decimal])
    assert_type(Decimal(1) - quantity, Quantity[Decimal])
    assert_type(quantity == 1000, bool)
    assert_type(quantity < 1001, bool)


def test_quantity_powers() -> None:
    assert_type(Quantity(2, 'm') ** 2, Quantity[Number])
    assert_type(Quantity(-2, 'm') ** Fraction(1, 2), Quantity[Number])
    assert_type(Quantity(Decimal(2), 'm') ** Decimal('0.5'), Quantity[Decimal])


def test_quantity_custom(value: Magnitude) -> None:
    quantity = Quantity(value, 'm')
    assert_type(quantity, Quantity[Magnitude])
    assert_type(quantity.to('cm'), Quantity[Magnitude])
    assert_type(quantity + quantity, Quantity[Magnitude])
    assert_type(quantity * 2, Quantity[Magnitude])
    assert_type(quantity / 2, Quantity[Magnitude])
    assert_type(-quantity, Quantity[Magnitude])
    assert_type(quantity**2, Quantity[Magnitude])
    assert_type(value == value, bool)
    assert_type(value != value, bool)
    assert_type(quantity == quantity, bool)
    assert_type(quantity != quantity, bool)


def test_factor_scaling(factor: Factor, value: Magnitude) -> None:
    assert_type(factor(1), Real)
    assert_type(factor(1.0), Real)
    assert_type(factor(Decimal(1)), Real)
    assert_type(factor(Fraction(1)), Fraction | float)
    assert_type(factor(value), Magnitude)
    # Factor also permits native multiplication of nonnumeric magnitudes.
    assert_type(factor([1, 2]), Any)


def test_quantity_decimal_types() -> None:
    decimal = Quantity(Decimal('1.25'), 'm')
    integer = Quantity(2, 'cm')
    assert_type(decimal + integer, Quantity[Decimal])
    assert_type(integer + decimal, Quantity[Decimal])
    assert_type(decimal - integer, Quantity[Decimal])
    assert_type(integer - decimal, Quantity[Decimal])
    assert_type(decimal * integer, Quantity[Decimal])
    assert_type(integer * decimal, Quantity[Decimal])
    assert_type(decimal / integer, Quantity[Decimal])
    assert_type(integer / decimal, Quantity[Decimal])
    assert_type(decimal.value_in('cm'), Decimal)
    assert_type(decimal.to('cm'), Quantity[Decimal])
    assert_type(decimal ** Decimal('0.5'), Quantity[Decimal])


def test_quantity_custom_scalars(value: Magnitude) -> None:
    quantity = Quantity(value, '')
    assert_type(quantity + value, Quantity[Magnitude])
    assert_type(value + quantity, Quantity[Magnitude])
    assert_type(quantity * value, Quantity[Magnitude])
    assert_type(value * quantity, Quantity[Magnitude])
    assert_type(quantity / value, Quantity[Magnitude])
    assert_type(value / quantity, Quantity[Magnitude])


def test_quantity_power_types() -> None:
    assert_type(Quantity(2, 'm') ** Decimal('0.5'), Quantity[Decimal])
    assert_type(Quantity(2, 'm') ** 2.0, Quantity[Number])
    assert_type(Quantity(Decimal(2), 'm') ** Decimal('0.5'), Quantity[Decimal])
    assert_type(Quantity(Decimal(2), 'm') ** 2, Quantity[Decimal])
