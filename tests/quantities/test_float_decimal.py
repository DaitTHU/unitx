"""The explicit precision boundary used by float scientific formulas."""

import operator
from collections.abc import Callable
from decimal import Decimal, FloatOperation, Inexact, localcontext
from fractions import Fraction

import pytest

from unitx import Quantity, Unit
from unitx.factor import Factor
from unitx.si import SI, si


@pytest.mark.parametrize('operation', [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize('reverse', [False, True])
def test_quantity_float_decimal_context(operation: Callable[..., Quantity], reverse: bool) -> None:
    a, b = Decimal('1.23456789'), 0.123456781
    left, right = (b, a) if reverse else (a, b)
    with localcontext() as context:
        context.prec = 5
        context.traps[FloatOperation] = True
        expected = float(
            operation(
                Decimal(repr(left)) if type(left) is float else left,
                Decimal(repr(right)) if type(right) is float else right,
            )
        )
        result = operation(Quantity(left, ''), Quantity(right, ''))
        assert type(result.value) is float and result.value == expected
        context.traps[Inexact] = True
        with pytest.raises(Inexact):
            operation(Quantity(left, ''), Quantity(right, ''))


def test_quantity_float_decimal_alignment() -> None:
    with localcontext() as context:
        context.prec = 7
        result = Quantity(1e-300, 'm') + Quantity(Decimal('1e400'), 'cm')
        assert result.value == float(Decimal('1e-300') + Decimal('1e398'))
        # Normalize in Decimal before scaling: converting the float first in
        # the float domain would overflow even though the final sum is finite.
        result = Quantity(Decimal('-1e400'), 'm') + Quantity(
            1e300, Unit('m', scale=Factor(10**100))
        )
        assert result.value == 0.0
        result = 0.1 + Quantity(Decimal('1.23'), 'km/m')
        assert result.value == float(Decimal('1.23') + Decimal('0.0001'))


def test_quantity_float_decimal_range() -> None:
    assert (1e-300 * Quantity(Decimal('1e400'), 'm')).value == 1e100
    assert (Quantity(Decimal('1e400'), 'm') / 1e300).value == 1e100
    assert (1e300 / Quantity(Decimal('1e400'), 'm')).value == 1e-100
    assert (0.1 * Quantity(Decimal('1e400'), 'm')).value == float('inf')
    assert (Quantity(Decimal('1e-400'), 'm') * -0.1).value.hex() == '-0x0.0p+0'


def test_quantity_float_decimal_augmented() -> None:
    q = Decimal('1.25') * si.m
    q *= 0.1
    assert type(q.value) is float and q.value == 0.125
    q += Decimal('0.25') * si.m
    q -= Decimal('0.125') * si.m
    q /= Decimal('0.25')
    assert type(q.value) is float and q.value == 1


def test_quantity_float_constant() -> None:
    energy = (2.5 * si.V * SI.e).to(si.J)
    assert type(energy.value) is float
    assert energy.value == 4.005441585e-19
    assert (0.5 * SI.e).value == 8.01088317e-20
    assert type(SI.e.value) is Decimal


def test_quantity_float_decimal_exact_comparison() -> None:
    decimal = Decimal('0.1') * si.m
    binary = 0.1 * si.m
    assert decimal != binary
    assert hash(decimal) == hash(Fraction(1, 10) * si.m)
    assert (2 * decimal).value == Decimal('0.2')


def test_quantity_float_addition() -> None:
    left, right = 1.1 * si.m, 50 * si.cm
    total, difference = left + right, right - left
    assert type(total.value) is type(difference.value) is float
    assert total.value == 1.6 and difference.value == pytest.approx(-60)
    assert type((1 - 0.1 * si.one).value) is float
    # Numeric annotations do not coerce construction at runtime.
    annotated = Quantity[float](1, 'cm')
    assert (annotated + Quantity[float](1, 'mm')).value == Fraction(11, 10)
