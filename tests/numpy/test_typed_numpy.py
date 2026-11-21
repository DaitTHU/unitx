"""Finite typed entry points share ordinary NumPy runtime semantics."""

from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.factor import Factor

np = pytest.importorskip('numpy')
unp = pytest.importorskip('unitx.numpy')


@pytest.mark.parametrize(
    'family, unit', [(Quantity, 'cm'), (AffineQuantity, 'degC'), (LogQuantity, 'dBm')]
)
@pytest.mark.parametrize('name', ['zeros_like', 'ones_like', 'empty_like', 'full_like'])
def test_typed_numpy_like(family: Any, unit: str, name: str) -> None:
    q = family(np.arange(6.0).reshape(2, 3)[:, ::2], unit)
    args = (2,) if name == 'full_like' else ()
    result = getattr(unp, name)(q, *args)
    assert type(result) is family and result.unit is q.unit
    assert result.dtype == q.dtype and result.shape == q.shape
    assert not np.shares_memory(result.value, q.value)
    if name != 'empty_like':
        np.testing.assert_array_equal(result.value, getattr(np, name)(q, *args).value)
    changed = getattr(unp, name)(q, *args, dtype=np.float32, shape=(4,))
    assert changed.unit is q.unit and changed.shape == (4,)
    assert changed.dtype == np.dtype('f4')


def test_typed_numpy_full_like_conversion() -> None:
    q = Quantity(np.zeros(2), Unit('cm', scale=Factor(3)))
    np.testing.assert_array_equal(unp.full_like(q, 6 * Unit('cm')).value, 2)
    with pytest.raises(DimensionError):
        unp.full_like(q, 1 * Unit('s'))
    with pytest.raises(TypeError):
        unp.zeros_like(np.zeros(2))


def test_typed_numpy_calculus() -> None:
    q = Quantity(np.arange(12.0).reshape(3, 4), 'm')
    x = Quantity(np.linspace(0.0, 2.0, 4), 's')
    integral = unp.trapezoid(q, x, axis=1)
    assert integral.unit == Unit('m*s')
    np.testing.assert_array_equal(integral.value, np.trapezoid(q, x, axis=1).value)
    np.testing.assert_array_equal(
        unp.trapezoid(q, dx=2 * Unit('s')).value, np.trapezoid(q, dx=2 * Unit('s')).value
    )
    derivative = unp.gradient(q, x, axis=1)
    assert derivative.unit == Unit('m/s')
    np.testing.assert_array_equal(derivative.value, np.gradient(q, x, axis=1).value)
    assert isinstance(unp.gradient(q), tuple)
    points = AffineQuantity(np.array([0.0, 10.0, 30.0]), 'degC')
    assert unp.gradient(points, axis=0).unit == points.unit.delta_unit
    with pytest.raises(TypeError):
        unp.trapezoid(points)
    with pytest.raises(TypeError):
        unp.gradient(np.ones(3))


def test_typed_numpy_scalar_templates() -> None:
    q = Quantity(2, 'm')
    points = AffineQuantity(20, 'degC')
    levels = LogQuantity(3, 'dBm')
    for template, function, dtype in (
        (q, unp.zeros_like, int),
        (points, unp.ones_like, np.float64),
        (levels, unp.empty_like, np.dtype(np.float32)),
    ):
        result = function(template, dtype)
        assert type(result) is type(template) and result.unit is template.unit
        assert result.shape == () and result.dtype == np.dtype(dtype)
    assert unp.zeros_like(q).dtype == np.zeros_like(2).dtype
    filled = unp.full_like(q, Quantity(30, 'cm'), np.float32)
    assert filled.dtype == np.dtype(np.float32) and filled.unit is q.unit
    np.testing.assert_allclose(filled.value, 0.3)


def test_typed_numpy_calculus_steps() -> None:
    y = Quantity(np.array([1, 3, 5], dtype=np.float32), 'm')
    times = AffineQuantity(np.array([3.0, 4.0, 5.0]), 's')
    for step, dtype in (
        (Decimal('0.5'), np.float32),
        (Quantity(Fraction(1, 2), 's'), np.float32),
        (np.float64(0.5), np.float64),
    ):
        area = unp.trapezoid(y, dx=step)
        assert area.dtype == np.dtype(dtype) and area.value == 3
    slope = unp.gradient(y, times, axis=(0,))
    assert slope.unit == Unit('m/s') and slope.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(slope.value, [2, 2, 2])
    assert unp.gradient(y, axis=()) == ()
    rows, cols = unp.gradient(Quantity(np.ones((2, 3), dtype=np.float32), 'm'), axis=(0, 1))
    assert rows.dtype == cols.dtype == np.dtype(np.float32)
