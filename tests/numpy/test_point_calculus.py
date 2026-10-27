"""Coordinate calculus returns differences and ignores arbitrary origins."""

from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity
from unitx.exceptions import DimensionError, UnitOperationError
from unitx.factor import Factor
from unitx.si import si

np = pytest.importorskip('numpy')


@pytest.mark.parametrize('cls,unit', [(AffineQuantity, si.degC), (LogQuantity, si.dBm)])
@pytest.mark.parametrize('n', [1, 2, 5])
def test_point_diff_orders(cls: Any, unit: Any, n: int) -> None:
    readings = np.array([[0.0, 1, 3, 6], [4, 7, 9, 12]])
    points = cls(readings, unit)
    result = np.diff(points, n=n, axis=1)
    assert type(result) is Quantity and result.unit == unit.delta_unit
    np.testing.assert_array_equal(result.value, np.diff(readings, n=n, axis=1))
    assert np.diff(points, n=0, axis=99, prepend=None) is points


def test_point_diff_boundaries() -> None:
    points = AffineQuantity(np.array([0.0, 10, 30]), si.degC)
    result = np.diff(points, prepend=273.15 @ si.K, append=313.15 @ si.K)
    assert result.unit == si.degC.delta_unit
    np.testing.assert_allclose(result.value, [0, 10, 20, 10], atol=1e-13)
    levels = LogQuantity(np.array([0.0, 10]), si.dBm)
    np.testing.assert_allclose(np.diff(levels, prepend=-30 @ si.dBW).value, [0, 10])


@pytest.mark.parametrize('boundary', [0, 0 * si.K])
def test_point_diff_invalid_boundary(boundary: object) -> None:
    points = AffineQuantity(np.array([0.0, 10]), si.degC)
    with pytest.raises(UnitOperationError, match='points and differences'):
        np.diff(points, prepend=boundary)


def test_point_diff_invalid_convention() -> None:
    with pytest.raises(DimensionError, match='compatible points'):
        np.diff(LogQuantity(np.arange(3.0), si.dBm), prepend=0 @ si.degC)


@pytest.mark.parametrize('cls,unit', [(AffineQuantity, si.degC), (LogQuantity, si.dBm)])
def test_point_gradient_origins(cls: Any, unit: Any) -> None:
    coordinate = AffineQuantity(np.array([0.0, 1, 3, 6]), si.s)
    points = cls(np.array([0.0, 2, 6, 12]), unit)
    result = np.gradient(points, coordinate, edge_order=2)
    assert type(result) is Quantity and result.unit == unit.delta_unit / si.s
    np.testing.assert_allclose(result.value, 2)
    shifted = coordinate + Quantity(100, si.s)
    np.testing.assert_allclose(np.gradient(points, shifted).value, result.value)


def test_point_gradient_mixed_axes() -> None:
    t = AffineQuantity(np.array([10.0, 11, 13, 16]), si.s)
    x = AffineQuantity(np.array([0.0, 1, 4]), si.cm)
    values = Quantity(t.value[:, None] ** 2 + x.value[None, :] ** 2, si.V)
    dt, dx = np.gradient(values, t, x, edge_order=2)
    assert dt.unit == si.V / si.s and dx.unit == si.V / si.cm
    np.testing.assert_allclose(dt.value, np.broadcast_to(2 * t.value[:, None], values.shape))
    np.testing.assert_allclose(dx.value, np.broadcast_to(2 * x.value, values.shape), atol=1e-12)
    raw_result = np.gradient(values.value, t, x, edge_order=2)
    assert raw_result[0].unit == si.s**-1
    assert raw_result[1].unit == si.cm**-1


def test_point_gradient_invalid_step() -> None:
    points = AffineQuantity(np.arange(4.0), si.degC)
    with pytest.raises(UnitOperationError, match='scalar point is not a spacing step'):
        np.gradient(points, 2 @ si.s)


def test_point_gradient_invalid_coordinate_shape() -> None:
    points = AffineQuantity(np.arange(4.0), si.degC)
    coordinates = AffineQuantity(np.arange(4.0).reshape(2, 2), si.s)
    with pytest.raises(ValueError, match='scalars or 1d'):
        np.gradient(points, coordinates)


@pytest.mark.parametrize('values', [[0.0, 1], [0.0, 1, 3, 6], [6.0, 3, 1, 0], [0.0, 2, 1, 4]])
@pytest.mark.parametrize(
    'cls,unit', [(Quantity, si.s), (AffineQuantity, si.s), (LogQuantity, si.dBm)]
)
def test_point_gradient_quadrature(values: list[float], cls: Any, unit: Any) -> None:
    coordinates = cls(np.array(values), unit)
    before = coordinates.value.copy()
    weights = np.gradient(coordinates, axis=0)
    weights[[0, -1]] /= 2
    delta = coordinates.unit if cls is Quantity else coordinates.unit.delta_unit
    assert weights.unit == delta
    signal = np.arange(len(values)) ** 2 + 1j
    np.testing.assert_allclose(
        np.sum(signal * weights.value), np.trapezoid(signal, coordinates.value)
    )
    np.testing.assert_allclose(weights.value.sum(), values[-1] - values[0])
    np.testing.assert_array_equal(coordinates.value, before)


def test_quantity_gradient_scaled_quadrature() -> None:
    coordinates = Quantity(np.array([0.0, 1, 4]), Factor(2) * si.ms)
    weights = np.gradient(coordinates, axis=0)
    weights[[0, -1]] /= 2
    np.testing.assert_allclose(weights.value, [0.5, 2, 1.5])
    values = Quantity(np.array([2.0, 4, 8]), si.A)
    assert np.sum(values * weights).isclose(np.trapezoid(values, coordinates))
    assert np.sum(values * weights).isclose(np.trapezoid(values, coordinates.to(si.s)))
