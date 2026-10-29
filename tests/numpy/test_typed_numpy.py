"""Finite typed entry points share ordinary NumPy runtime semantics."""

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
