"""Static API contracts, checked by Pyright rather than executed by pytest."""

from decimal import Decimal
from fractions import Fraction
from typing import Any, assert_type

from unitx._numerics import Number, Real, Scalar
from unitx.factor import Factor, pi
from unitx.quantity import Quantity
from unitx.si import si
from unitx.unit import Unit
from unitx.unitdefinition import PrefixPolicy


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


def test_unit_value_operations[T: Scalar](value: T, unit: Unit) -> tuple[Quantity[T], Quantity[T]]:
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


def test_unit_scale_construction() -> None:
    unit = Unit('m')
    scaled = Factor(2) * unit
    assert_type(scaled, Unit)
    assert_type(pi * unit, Unit)
    assert_type(Factor(2) / unit, Unit)
    assert_type(pi / scaled, Unit)
    assert_type(3 * scaled, Quantity[int])
    assert_type(Fraction(3) / scaled, Quantity[Fraction])
    assert_type(scaled * unit, Unit)
    # Base signatures keep mixed operations conservative; runtime dispatch promotes.
    assert_type(unit * scaled, Unit)
    assert_type(scaled / unit, Unit)
    assert_type(unit / scaled, Unit)
    assert_type(scaled * scaled, Unit)
    assert_type(scaled / scaled, Unit)
    assert_type(scaled**2, Unit)
    assert_type(scaled**0, Unit)
    assert_type(unit * unit, Unit)
    assert_type(unit / unit, Unit)
    assert_type(unit**2, Unit)
    assert_type(scaled.scale, Factor)
    assert_type(Unit('cm', scale=Factor(2)), Unit)


def test_unit_numeric_scaling(unit: Unit, scaled: Unit) -> None:
    for value in (Factor(2), pi):
        assert_type(unit * value, Unit)
        assert_type(unit / value, Unit)
        assert_type(scaled * value, Unit)
        assert_type(scaled / value, Unit)
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
    assert_type(Fraction(3) * (unit / 2), Quantity[Fraction])
    assert_type(Factor(2) * unit, Unit)
    assert_type(pi * scaled, Unit)
    assert_type(unit * Quantity(2, 's'), Quantity[int])
    assert_type(scaled * Quantity(2, 's'), Quantity[int])
    assert_type(unit / Quantity(2, 's'), Quantity[float])
    assert_type(scaled / Quantity(2, 's'), Quantity[float])


def test_unit_define() -> None:
    policy: PrefixPolicy = ('rvpolicy',)
    assert_type(si.m.define('rvpolicy', 'review policy', prefixable=policy), Unit)
    single_policy: PrefixPolicy = 'rvpolicy'
    assert_type(si.m.define('rvpolicy', 'review policy', prefixable=single_policy), Unit)
    assert_type((pi * Unit('m')).define('rvpolicy', 'review policy', prefixable='rvpolicy'), Unit)
    assert_type(
        Quantity(2, 'm').define('rvpolicy', 'review policy', prefixable='review policy'), Unit
    )
    assert_type(
        si.m.define(
            'rvbound',
            'review bound',
            prefixable=('rvbound', 'review bound'),
        ),
        Unit,
    )
    assert_type(
        (pi * Unit('m')).define('rvbound', 'review bound', prefixable=()),
        Unit,
    )
    assert_type(
        Quantity(2, 'm').define('rvbound', 'review bound', prefixable=('review bound',)),
        Unit,
    )
    assert_type(si.m.define('rvlen', name='reviewlength'), Unit)
    assert_type(si.m.define('rvlen', 'reviewlength'), Unit)
    assert_type(si.m.define('rvlen', 'reviewlength', 'reviewdistance', 'reviewspan'), Unit)
    assert_type((pi * Unit('m')).define('rvpi', name='reviewpilength'), Unit)
    assert_type((pi * Unit('m')).define('rvpi', 'reviewpilength', 'reviewpispan'), Unit)
    assert_type((60 * Unit('s')).define('rvmin', name='reviewminute'), Unit)
    assert_type((60 * Unit('s')).define('rvmin', 'reviewminute', 'reviewduration'), Unit)
    assert_type(Quantity(Decimal('0.3048'), 'm').define('rvft', name='reviewfoot'), Unit)


def test_quantity_value_in(quantity: Quantity[Scalar]) -> None:
    assert_type(Quantity(1, 'm').value_in('cm'), Real)
    assert_type(Quantity(Fraction(1), 'm').value_in('cm'), Fraction | float)
    assert_type(Quantity(Decimal(1), 'm').value_in('cm'), Decimal)
    assert_type(Quantity(1.0, 'm').value_in('cm'), Real)
    assert_type(Quantity[float](1, 'cm').value_in('m'), Real)
    assert_type(Quantity(1, 'm').value_in(Unit('cm', scale=Factor(2))), Real)
    assert_type(Quantity(1j, 'cm').value_in('m'), Number)
    assert_type(quantity.value_in('m'), Scalar)


def test_quantity_conversion() -> None:
    assert_type(Quantity(1, 'm').to('cm'), Quantity[Real])
    assert_type(Quantity(Fraction(1), 'm').to('cm'), Quantity[Fraction | float])
    assert_type(Quantity(Decimal(1), 'm').to('cm'), Quantity[Decimal])
    assert_type(Quantity(1.0, 'm').to('cm'), Quantity[Real])
    # A float annotation accepts int values too; scaling can then yield Fraction.
    assert_type(Quantity[float](1, 'cm').to('m'), Quantity[Real])


def test_quantity_unit_operations[T: Scalar](value: T) -> Quantity[T]:
    quantity = Quantity(value, 'm')
    unit = Unit('s')
    assert_type(quantity, Quantity[T])
    assert_type(quantity.value, T)
    assert_type(quantity * unit, Quantity[T])
    assert_type(unit * quantity, Quantity[T])
    assert_type(quantity / unit, Quantity[T])
    return quantity


def test_quantity_factor_operations[T: Scalar](value: T) -> Quantity[T]:
    quantity = Quantity(value, 'm')
    for factor in (Factor(2), pi):
        assert_type(quantity * factor, Quantity[T])
        assert_type(factor * quantity, Quantity[T])
        assert_type(quantity / factor, Quantity[T])
        assert_type(factor / quantity, Quantity[Scalar])
        assert_type(Factor(2) / Quantity(2, 's'), Quantity[float])
        assert_type(pi / Quantity(Decimal(2), 's'), Quantity[Decimal])
    assert_type(Quantity(Decimal(2), 's') / (2 * pi), Quantity[Decimal])
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


def test_quantity_unary_generic(quantity: Quantity[Scalar]) -> None:
    assert_type(+quantity, Quantity[Scalar])
    assert_type(-quantity, Quantity[Scalar])


def test_quantity_dimensionless() -> None:
    quantity = Quantity(1, 'km/m')
    assert_type(quantity + 1, Quantity[Real])
    assert_type(1 + quantity, Quantity[Real])
    assert_type(quantity - Decimal(1), Quantity[Decimal])
    assert_type(Decimal(1) - quantity, Quantity[Decimal])
    assert_type(quantity == 1000, bool)
    assert_type(quantity < 1001, bool)


def test_quantity_powers() -> None:
    assert_type(Quantity(2, 'm') ** 2, Quantity[int])
    assert_type(Quantity(-2, 'm') ** Fraction(1, 2), Quantity[Number])
    assert_type(Quantity(Decimal(2), 'm') ** Decimal('0.5'), Quantity[Decimal])


def test_quantity_custom(value: Scalar) -> None:
    quantity = Quantity(value, 'm')
    assert_type(quantity, Quantity[Scalar])
    assert_type(quantity.to('cm'), Quantity[Scalar])
    assert_type(quantity + quantity, Quantity[Scalar])
    assert_type(quantity * 2, Quantity[Scalar])
    assert_type(quantity / 2, Quantity[Scalar])
    assert_type(-quantity, Quantity[Scalar])
    assert_type(quantity**2, Quantity[Scalar])
    assert_type(value == value, bool)
    assert_type(value != value, bool)
    assert_type(quantity == quantity, bool)
    assert_type(quantity != quantity, bool)


def test_factor_scaling(factor: Factor, value: Scalar) -> None:
    assert_type(factor(1), Real)
    assert_type(factor(1.0), Real)
    assert_type(factor(Decimal(1)), Real)
    assert_type(factor(Fraction(1)), Fraction | float)
    assert_type(factor(value), Scalar)
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


def test_quantity_custom_scalars(value: Scalar) -> None:
    quantity = Quantity(value, '')
    assert_type(quantity + value, Quantity[Scalar])
    assert_type(value + quantity, Quantity[Scalar])
    assert_type(quantity * value, Quantity[Scalar])
    assert_type(value * quantity, Quantity[Scalar])
    assert_type(quantity / value, Quantity[Scalar])
    assert_type(value / quantity, Quantity[Scalar])


def test_quantity_power_types() -> None:
    assert_type(Quantity(2, 'm') ** Decimal('0.5'), Quantity[Decimal])
    assert_type(Quantity(2, 'm') ** 2.0, Quantity[Number])
    assert_type(Quantity(Decimal(2), 'm') ** Decimal('0.5'), Quantity[Decimal])
    assert_type(Quantity(Decimal(2), 'm') ** 2, Quantity[Decimal])


def test_scalar_augmented_types(value: Scalar) -> None:
    decimal = Quantity(Decimal('1.25'), 'm')
    decimal += Quantity(1, 'cm')
    assert_type(decimal, Quantity[Decimal])
    decimal -= Quantity(Decimal('0.25'), 'm')
    decimal *= 2
    decimal /= 2
    decimal **= 2
    assert_type(decimal, Quantity[Decimal])
    integer = Quantity(2, 'm')
    integer *= 2
    integer *= Unit('s')
    assert_type(integer, Quantity[int])
    integer /= 2
    assert_type(integer, Quantity[float])
    real = Quantity(2, 'm')
    real += real
    assert_type(real, Quantity[Real])
    custom = Quantity(value, 'm')
    custom += custom
    custom *= Unit('s')
    assert_type(custom, Quantity[Scalar])


def test_display_types() -> None:
    q = Quantity(Decimal('1.25'), 'm')
    assert_type(format(q, '.2f;latex'), str)
    assert_type(q.to_latex(), str)
    assert_type(q.to_latex('.2e', fraction=False), str)
    assert_type(q.unit.to_latex(fraction=False), str)
    assert_type(format(q.unit, 'latex'), str)
    assert_type(q.to_siunitx('.2f'), str)
    assert_type(q.unit.to_siunitx(), str)
    assert_type(format(q, '.2f;siunitx'), str)
    assert_type(q.to_html('.2f'), str)
    assert_type(q.unit.to_latex(), str)
    assert_type(q.unit.to_html(), str)
    assert_type(Unit('m', scale=Factor(2)).to_latex(), str)


def test_quantity_integer_power_literals(q: Quantity[int], exponent: int) -> None:
    assert_type(q**0, Quantity[int])
    assert_type(q**4, Quantity[int])
    assert_type(q**25, Quantity[int])
    assert_type(q**-1, Quantity[float])
    assert_type(q**-20, Quantity[float])
    assert_type(q**exponent, Quantity[Real])
    squared = q
    squared **= 2
    assert_type(squared, Quantity[int])
