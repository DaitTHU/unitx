"""Histograms retain coordinate and weight units; density integrates to one."""

from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi
from unitx.si import si

np = pytest.importorskip('numpy')


@pytest.mark.parametrize(
    'constructor,unit', [(Quantity, 'm'), (AffineQuantity, 'degC'), (LogQuantity, 'dBm')]
)
@pytest.mark.parametrize('density', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_numpy_histogram(constructor: Any, unit: str, density: bool, weighted: bool) -> None:
    q = constructor(np.array([0.0, 1, 2, 3]), unit)
    weights = Quantity(np.array([1.0, 2, 3, 4]), Factor(2) * pi * si.kg) if weighted else None
    hist, edges = np.histogram(q, bins=2, density=density, weights=weights)
    expected, expected_edges = np.histogram(
        q.value, bins=2, density=density, weights=None if weights is None else weights.value
    )
    assert type(edges) is constructor and edges.unit is q.unit
    np.testing.assert_allclose(edges.value, expected_edges)
    if density:
        delta = q.unit.delta_unit if constructor is not Quantity else q.unit
        assert hist.unit == delta**-1
        np.testing.assert_allclose(np.sum(hist * np.diff(Quantity(edges.value, delta))).value, 1)
    elif weighted:
        assert hist.unit is weights.unit
    else:
        assert type(hist) is np.ndarray
    np.testing.assert_allclose(hist.value if isinstance(hist, Quantity) else hist, expected)


def test_numpy_histogram_edges_and_range() -> None:
    q = Quantity(np.array([0.0, 1, 2, 3]), 'm')
    bins = Quantity(np.array([0.0, 200, 400]), 'cm')
    hist, edges = np.histogram(q, bins=bins)
    np.testing.assert_array_equal(hist, [2, 2])
    np.testing.assert_array_equal(edges.value, [0, 2, 4])
    generated = np.histogram_bin_edges(q, bins=2, range=(0 * si.cm, 400 * si.cm))
    np.testing.assert_allclose(generated.value, [0, 2, 4])
    np.testing.assert_array_equal(np.digitize(q, bins), [1, 1, 2, 2])
    np.testing.assert_array_equal(np.digitize(q, bins, right=True), [0, 1, 1, 2])
    _, sequence = np.histogram(q, bins=[0 * si.m, 200 * si.cm, 4 * si.m])
    np.testing.assert_array_equal(sequence.value, [0, 2, 4])


@pytest.mark.parametrize('operation', [np.histogram, np.histogram_bin_edges, np.digitize])
def test_numpy_histogram_invalid_edges(operation: Any) -> None:
    q = Quantity(np.arange(4.0), 'm')
    saved = q.value.copy()
    with pytest.raises(DimensionError):
        operation(q, bins=np.array([0.0, 2, 4]))
    with pytest.raises(DimensionError):
        operation(q, bins=Quantity(np.array([0.0, 2, 4]), 's'))
    np.testing.assert_array_equal(q.value, saved)


@pytest.mark.parametrize('density', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_numpy_histogram_multidimensional(density: bool, weighted: bool) -> None:
    x = Quantity(np.array([0.0, 1, 2, 3]), Factor(2) * si.m)
    y = AffineQuantity(np.array([10.0, 20, 30, 40]), 'degC')
    weights = Quantity(np.array([1.0, 2, 3, 4]), 'kg') if weighted else None
    bins = [
        Quantity(np.array([0.0, 400, 800]), 'cm'),
        AffineQuantity(np.array([273.15, 298.15, 323.15]), 'K'),
    ]
    hist, xe, ye = np.histogram2d(x, y, bins=bins, density=density, weights=weights)
    nd, edges = np.histogramdd((x, y), bins=bins, density=density, weights=weights)
    assert xe.unit is x.unit and ye.unit is y.unit
    np.testing.assert_array_equal(edges[0].value, xe.value)
    np.testing.assert_array_equal(edges[1].value, ye.value)
    if density:
        assert hist.unit == (x.unit * y.unit.delta_unit) ** -1
        area = Quantity(np.diff(xe.value)[:, None], x.unit) * Quantity(
            np.diff(ye.value)[None, :], y.unit.delta_unit
        )
        np.testing.assert_allclose(np.sum(hist * area).value, 1)
    elif weighted:
        assert hist.unit is weights.unit
    actual = hist.value if isinstance(hist, Quantity) else hist
    np.testing.assert_array_equal(actual, nd.value if isinstance(nd, Quantity) else nd)


def test_numpy_histogram_matrix() -> None:
    q = Quantity(np.arange(12.0).reshape(6, 2), 'cm')
    result, edges = np.histogramdd(q, bins=[2, 3], density=True)
    expected, raw_edges = np.histogramdd(q.value, bins=[2, 3], density=True)
    assert result.unit == si.cm**-2
    np.testing.assert_allclose(result.value, expected)
    for edge, raw in zip(edges, raw_edges):
        assert edge.unit is q.unit
        np.testing.assert_array_equal(edge.value, raw)


def test_numpy_histogram_shared_edges() -> None:
    x = Quantity(np.array([0.0, 1, 2]), 'm')
    y = Quantity(np.array([0.0, 100, 200]), 'cm')
    bins = Quantity(np.array([0.0, 1, 2, 3]), 'm')
    result, xe, ye = np.histogram2d(x, y, bins=bins)
    expected, _, _ = np.histogram2d(x.value, y.value / 100, bins=bins.value)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(xe.value, [0, 1, 2, 3])
    np.testing.assert_array_equal(ye.value, [0, 100, 200, 300])


def test_numpy_bincount_weights() -> None:
    weights = Quantity(np.array([1.0, 2, 3]), Factor(2) * pi * si.kg)
    result = np.bincount(np.array([0, 1, 0]), weights=weights, minlength=3)
    assert result.unit is weights.unit
    np.testing.assert_array_equal(result.value, [4, 2, 0])
    with pytest.raises(TypeError):
        np.bincount(Quantity(np.array([0, 1, 0]), ''), weights=weights)


def test_numpy_histogram_invalid_weights() -> None:
    q = Quantity(np.arange(4.0), 'm')
    with pytest.raises(TypeError):
        np.histogram(q, weights=AffineQuantity(np.ones(4), 'degC'))
    with pytest.raises(ValueError):
        np.histogram(q, weights=Quantity(np.ones(2), 'kg'))


def test_numpy_histogram_empty_coordinates() -> None:
    x, y = Quantity(np.empty(0), 'm'), Quantity(np.empty(0), 's')
    hist, edges = np.histogramdd((x, y), bins=2)
    np.testing.assert_array_equal(hist, np.zeros((2, 2)))
    assert edges[0].unit == si.m and edges[1].unit == si.s
    matrix, matrix_edges = np.histogramdd(Quantity(np.empty((0, 2)), 'm'), bins=2)
    np.testing.assert_array_equal(matrix, hist)
    assert all(edge.unit == si.m for edge in matrix_edges)


def test_numpy_histogram_warning_location() -> None:
    with pytest.warns(RuntimeWarning) as caught:
        np.histogram(Quantity(np.empty(0), 'm'), bins=2, density=True)
    assert caught[0].filename == __file__
