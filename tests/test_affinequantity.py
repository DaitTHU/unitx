"""Point identity belongs to the quantity class, including at zero-origin coordinates."""

from __future__ import annotations

import copy
import operator
import pickle
from collections.abc import Callable
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx import AffineQuantity, AffineUnit, LogQuantity, Quantity, Unit
from unitx._numerics import Magnitude
from unitx.exceptions import DimensionError
from unitx.factor import Factor
from unitx.imperial import imperial
from unitx.si import si


def test_unit_linear_temperature() -> None:
    assert type(Unit('degC')) is Unit
    assert Unit('degC').factor == Unit('K').factor
    assert Quantity(1, 'K').to('degC') == Quantity(1, 'degC')
    assert Quantity(9, imperial.degF).to('degC') == Quantity(5, 'K')
    assert Unit('degC/degC') == Unit()
    assert Unit('degC**0') == Unit()
    assert Quantity(1, 'degC/m').to('K/m').value == 1
    assert isinstance(Unit('degC', scale=Factor(2)), Unit)


def test_affine_quantity_zero_origin() -> None:
    zero = AffineQuantity(0, 'K')
    celsius = zero.to('degC')
    assert type(zero) is AffineQuantity
    assert type(celsius) is AffineQuantity
    assert celsius == AffineQuantity(Decimal('-273.15'), 'degC')
    assert celsius.value == Fraction(-5463, 20)
    assert celsius.to('K') == zero
    assert zero.to('K').value is zero.value
    assert repr(zero) == "AffineQuantity(0, 'K')"
    assert str(zero) == '0 K'
    # Finite float literals retain their binary rational value, without a tolerance.
    approximate = AffineQuantity(-273.15, 'degC')
    assert celsius != approximate
    assert celsius.isclose(approximate, abs_tol=Quantity(1e-12, 'K'))


def test_affine_quantity_point_delta() -> None:
    hot = AffineQuantity(300, 'K')
    cold = AffineQuantity(20, 'degC')
    difference = hot - cold
    assert type(difference) is Quantity
    assert difference == Quantity(Fraction(137, 20), 'K')
    assert cold + difference == hot
    assert difference + cold == hot
    assert (cold + difference).to('K') == hot
    assert hot.to('degC') - cold == difference
    assert hot != Quantity(300, 'K')
    assert Quantity(300, 'K') != hot
    with pytest.raises(TypeError):
        hot + cold
    with pytest.raises(TypeError):
        Quantity(300, 'K') - cold
    with pytest.raises(DimensionError):
        hot < Quantity(300, 'K')


def test_affine_quantity_physical_exit() -> None:
    point = AffineQuantity(0, 'degC')
    physical = point.to_quantity('K')
    assert type(physical) is Quantity
    assert physical == Quantity(Fraction(5463, 20), 'K')
    assert type(point.to('K')) is AffineQuantity
    assert physical * 2 == Quantity(Fraction(5463, 10), 'K')
    assert LogQuantity(30, 'dBm').to_quantity('W').value == 1
    assert LogQuantity.from_quantity(Quantity(1, 'W'), 'dBm').value == 30
    assert LogQuantity(0, 'dBm').to_quantity('mW') == Quantity(1, 'mW')
    with pytest.raises(TypeError):
        point.to_quantity('dB')


def test_affine_quantity_scalar_rebinding() -> None:
    point = AffineQuantity(20, 'degC')
    original = point
    point += Quantity(1, 'K')
    assert point == AffineQuantity(21, 'degC')
    assert point is not original
    point -= AffineQuantity(20, 'degC')
    assert type(point) is Quantity
    assert point == Quantity(1, 'K')


def test_affine_quantity_reference_exactness() -> None:
    origin = 2**53 + 1
    coordinate = AffineUnit(origin * si.deg)
    point = AffineQuantity(0, coordinate)
    assert point.value_in(si.deg) == origin
    assert point == AffineQuantity(origin, si.deg)
    assert point != AffineQuantity(origin - 1, si.deg)
    assert point.to_quantity(si.deg) == Quantity(origin, si.deg)
    assert point.to(AffineUnit((origin - 1) * si.deg)).value == 1


@pytest.mark.parametrize('precision', [2, 8, 28])
def test_affine_quantity_decimal_cancellation(precision: int) -> None:
    with localcontext() as ctx:
        ctx.prec = precision
        difference = AffineQuantity(Decimal('1000'), 'degC') - AffineQuantity(
            Decimal('1831.982'), imperial.degF
        )
        assert difference.value == Decimal('0.01')
        assert isinstance(difference.value, Decimal)
        shifted = AffineQuantity(Decimal('1000'), 'K') - Quantity(Decimal('999.99'), 'degC')
        assert shifted.value == Decimal('0.01')


@pytest.mark.parametrize('unit', ['K', 'degC', imperial.degF, 'dBm'])
def test_affine_quantity_reconstruction(unit: str | AffineUnit) -> None:
    quantity_type = LogQuantity if unit == 'dBm' else AffineQuantity
    point = quantity_type(Decimal('20'), unit)
    assert not isinstance(point, Quantity)
    assert not issubclass(AffineQuantity, Quantity)
    assert eval(repr(point)) == point
    assert pickle.loads(pickle.dumps(point)) == point
    assert copy.copy(point) == point
    assert copy.deepcopy(point) == point
    with pytest.raises(TypeError):
        hash(point)


@pytest.mark.parametrize('value', [True, 1j, [1, 2], (1, 2), object()])
def test_affine_quantity_invalid_magnitude(value: object) -> None:
    with pytest.raises(TypeError):
        AffineQuantity(value, 'K')


@pytest.mark.parametrize(
    'operation', [operator.neg, operator.abs, lambda p: p * 2, lambda p: 2 / p, lambda p: p**2]
)
def test_affine_quantity_invalid_arithmetic(operation: Callable[..., object]) -> None:
    with pytest.raises(TypeError):
        operation(AffineQuantity(1, 'K'))


def test_affine_quantity_invalid_cross_domain() -> None:
    point = AffineQuantity(0, 'degC')
    assert Quantity(1, point.unit) == Quantity(1, 'degC')
    assert Quantity(1, point.unit).unit == Unit('degC')
    with pytest.raises(TypeError):
        point.isclose(Quantity(0, 'degC'))
    with pytest.raises(TypeError):
        point.isclose(point, abs_tol=0j)
    assert point.isclose(point)
    assert not (point == Quantity(0, 'K dB'))
    assert not (Quantity(0, 'K dB') == point)


def test_affine_quantity_scaled_coordinate() -> None:
    scaled = Factor(2) * si.K
    point = AffineQuantity(1, scaled)
    assert point.to('K') == AffineQuantity(2, 'K')
    assert eval(repr(point)) == point
    scaled_celsius = Factor(2) * si.degC
    assert AffineQuantity(1, scaled_celsius).value_in('K') == 2


@pytest.mark.parametrize('value', [0, -20, Fraction(1, 3), Decimal('1.25')])
def test_unit_at_construction(value: int | Fraction | Decimal) -> None:
    point = value @ si.degC
    assert type(point) is AffineQuantity
    assert point.value is value
    assert point == AffineQuantity(value, 'degC')
    assert value * si.degC == si.degC * value == Quantity(value, 'degC')
    assert (value @ si.K).unit.delta_unit == si.K
    custom = AffineUnit(100 * si.K)
    assert (value @ custom).unit is custom
    assert (value @ custom).value is value


@pytest.mark.parametrize('value', [True, 1j, [1, 2], (1, 2), Quantity(1, 'K')])
def test_unit_at_invalid_magnitude(
    value: bool | complex | list[int] | tuple[int, ...] | Quantity[Magnitude],
) -> None:
    with pytest.raises(TypeError):
        value @ si.degC


def test_affine_unit_linear_projection() -> None:
    descriptor = si.degC
    assert isinstance(descriptor, AffineUnit)
    assert not isinstance(descriptor, Unit)
    linear = Unit('degC')
    for quantity in (Quantity(2, descriptor), 2 * descriptor, descriptor * 2):
        assert type(quantity) is Quantity
        assert quantity.unit == linear
        assert str(quantity) == '2 °C'
        assert quantity.value == 2
    assert Quantity(2, 'K').to(descriptor).unit == linear
    assert Quantity(2, 'K').to(descriptor).value == 2
    assert Quantity(2, 'K').value_in(imperial.degF) == Fraction(18, 5)
    assert descriptor * si.m == linear * si.m
    assert si.m * descriptor == si.m * linear
    assert descriptor / si.m == linear / si.m
    assert si.m / descriptor == si.m / linear
    assert descriptor**2 == linear**2
    assert descriptor * imperial.degF == linear * Quantity(1, imperial.degF).unit
    assert (Quantity(2, 'm') * descriptor).unit == si.m * linear
    assert (descriptor * Quantity(2, 'm')).unit == linear * si.m
    assert (Quantity(2, 'm') / descriptor).unit == si.m / linear
    assert (descriptor / 2).value == Fraction(1, 2)
    assert (2 / descriptor).unit == linear**-1
    assert (Factor(2) * descriptor).scale == Factor(2)
    assert (descriptor * Factor(2)).scale == Factor(2)


def test_affine_unit_anonymous_projection() -> None:
    custom = AffineUnit(100 * si.K, at=10, delta_unit=imperial.degF.delta_unit)
    difference = Quantity(2, custom)
    assert difference.unit is custom.delta_unit
    assert 2 * custom == custom * 2 == difference
    assert (2 @ custom).unit is custom
    assert (2 @ custom).value_in('K') != difference.value_in('K')


def test_affine_quantity_symbolic_pi() -> None:
    import math

    from unitx.factor import pi

    exact = AffineQuantity(180, 'deg')
    approximate = AffineQuantity(math.pi, 'rad')
    # math.pi is a finite binary rational; the degree factor retains symbolic pi.
    assert Quantity(180, 'deg') != Quantity(math.pi, 'rad')
    assert exact != approximate
    assert approximate != exact
    assert not exact.isclose(approximate)
    assert exact.isclose(approximate, abs_tol=Quantity(1e-12, 'rad'))
    symbolic = AffineQuantity(1, pi * si.rad)
    assert exact == symbolic
    assert symbolic == exact
    assert exact.isclose(symbolic)


def test_affine_quantity_symbolic_origin() -> None:
    from unitx.factor import pi

    degrees = AffineUnit(Quantity(180, 'deg'))
    radians = AffineUnit(Quantity(1, pi * si.rad))
    # Equal physical origins cancel even though their formal units differ.
    assert AffineQuantity(180, degrees) == AffineQuantity(1, radians)
    assert AffineQuantity(180, degrees).isclose(AffineQuantity(1, radians))
    shifted = AffineUnit(Quantity(90, 'deg'))
    assert AffineQuantity(90, shifted) == AffineQuantity(180, 'deg')
    assert AffineQuantity(89, shifted) < AffineQuantity(180, 'deg')
    assert AffineQuantity(91, shifted) > AffineQuantity(180, 'deg')


@pytest.mark.parametrize('operation', [operator.eq, operator.lt, lambda a, b: a.isclose(b)])
def test_affine_quantity_invalid_symbolic_sum(operation: Callable[..., object]) -> None:
    coordinate = AffineUnit(Quantity(1, 'rad'), delta_unit='deg')
    point = AffineQuantity(1, coordinate)
    with pytest.raises(TypeError, match='symbolic sum'):
        operation(point, AffineQuantity(1, 'rad'))
    assert point.isclose(point.to('rad'), abs_tol=Quantity(1e-12, 'rad'))
