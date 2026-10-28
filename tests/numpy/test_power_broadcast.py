"""A single formal exponent can accompany a broadcast numerical array."""

from typing import Any

import pytest

from unitx import Quantity
from unitx.exceptions import DimensionError
from unitx.factor import Factor
from unitx.si import si

np = pytest.importorskip('numpy')


@pytest.mark.parametrize('operation', [np.power, np.float_power])
@pytest.mark.parametrize('scaled', [False, True])
def test_quantity_power_uniform(operation: Any, scaled: bool) -> None:
    q = Quantity(np.array([[1.0], [4.0]]), Factor(4) * si.m**2)
    exponent = np.full((1, 3), 0.5)
    if scaled:
        exponent = Quantity(exponent * 100, si.percent)
    result = operation(q, exponent)
    assert result.unit == Factor(2) * si.m
    np.testing.assert_array_equal(result.value, [[1, 1, 1], [2, 2, 2]])
    target = Quantity(np.zeros((2, 3)), si.cm)
    assert operation(q, exponent, out=target) is target
    np.testing.assert_array_equal(target.value, [[200] * 3, [400] * 3])


@pytest.mark.parametrize(
    'exponent',
    [
        np.array([]),
        np.array([1, 2]),
        np.array([1, np.nan]),
        np.array([1, np.inf]),
        np.array([1j, 1j]),
        np.array([1.0, np.nextafter(1.0, 2.0)]),
    ],
)
def test_quantity_power_exponent_invalid(exponent: Any) -> None:
    target = Quantity(np.full(2, -1.0), si.m)
    with pytest.raises(TypeError):
        np.power(Quantity(np.ones(2), si.m), exponent, out=target, where=False)
    np.testing.assert_array_equal(target.value, [-1, -1])


def test_quantity_power_log_exponent_invalid() -> None:
    with pytest.raises(DimensionError):
        np.power(Quantity(np.ones(2), si.m), Quantity(np.ones(2), si.dB))


def test_quantity_power_broadcast_invalid() -> None:
    out = Quantity(np.full(2, -1.0), si.m**2)
    with pytest.raises(ValueError):
        np.power(Quantity(np.ones(2), si.m), np.full(3, 2), out=out)
    np.testing.assert_array_equal(out.value, [-1, -1])


def test_quantity_power_empty_base() -> None:
    result = np.power(Quantity(np.empty((0, 3)), si.m), np.full((1, 3), 2))
    assert result.shape == (0, 3) and result.unit == si.m**2
