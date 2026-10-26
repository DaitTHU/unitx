"""Allocation and sample construction preserve full units and coordinates."""

from collections.abc import Callable
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi

from .conftest import ArrayQuantity

np = pytest.importorskip('numpy')


def test_full_like_readings(array_sample: ArrayQuantity) -> None:
    result = np.full_like(array_sample, 3, dtype=np.float32, shape=(2, 3))
    assert type(result) is type(array_sample) and result.unit == array_sample.unit
    assert result.value.dtype == np.dtype('f4') and result.shape == (2, 3)
    np.testing.assert_array_equal(result.value, 3)
    assert not np.shares_memory(result.value, array_sample.value)
    converted = np.full_like(array_sample, array_sample[0, 0, 0])
    np.testing.assert_array_equal(converted.value, array_sample.value[0, 0, 0])


def test_full_like_conversion() -> None:
    q = Quantity(np.zeros(3), 'm')
    np.testing.assert_array_equal(np.full_like(q, Quantity(250, 'cm')).value, 2.5)
    p = np.full_like(AffineQuantity(np.zeros(3), 'degC'), AffineQuantity(273.15, 'K'))
    np.testing.assert_allclose(p.value, 0, atol=1e-12)


@pytest.mark.parametrize('fill', [Quantity(1, 's'), AffineQuantity(1, 'K')])
def test_full_like_invalid(fill: Any) -> None:
    with pytest.raises((DimensionError, TypeError)):
        np.full_like(Quantity(np.zeros(3), 'm'), fill)
    with pytest.raises(TypeError):
        np.full_like(Quantity(np.zeros(3), 'm'), 0, dtype=object)


def test_linspace_scaled_unit() -> None:
    unit = Factor(2) * pi * Unit('cm')
    start = Quantity(np.array([1.0, 2.0]), unit)
    stop = Quantity(np.array([3.0, 4.0]), unit).to('m')
    values, step = np.linspace(start, stop, 4, endpoint=False, retstep=True, axis=-1)
    assert values.unit == step.unit == unit
    expected, spacing = np.linspace(
        start.value, np.array([3.0, 4.0]), 4, endpoint=False, retstep=True, axis=-1
    )
    np.testing.assert_allclose(values.value, expected)
    np.testing.assert_allclose(step.value, spacing)


@pytest.mark.parametrize('family,coordinate', [(AffineQuantity, 'degC'), (LogQuantity, 'dBm')])
@pytest.mark.parametrize('num', [0, 1, 5])
def test_linspace_points(family: Any, coordinate: str, num: int) -> None:
    start, stop = family(0.0, coordinate), family(20.0, coordinate)
    values, step = np.linspace(start, stop, num, retstep=True)
    assert type(values) is family and values.unit == start.unit
    assert type(step) is Quantity and step.unit == start.unit.delta_unit
    expected, spacing = np.linspace(0.0, 20.0, num, retstep=True)
    np.testing.assert_allclose(values.value, expected)
    np.testing.assert_allclose(step.value, spacing)


def test_linspace_invalid() -> None:
    for end in (1, Quantity(1, 's'), AffineQuantity(1, 'K')):
        with pytest.raises((DimensionError, TypeError)):
            np.linspace(Quantity(0, 'm'), end)
    with pytest.raises(TypeError):
        np.linspace(Quantity(0, 'm'), Quantity(1, 'm'), dtype=bool)


def test_geomspace_units() -> None:
    start, stop = Quantity(1, 'm'), Quantity(10000, 'cm')
    values = np.geomspace(start, stop, 3)
    assert values.unit == start.unit
    np.testing.assert_allclose(values.value, [1, 10, 100])
    values = np.geomspace(Quantity(1j, 'm'), Quantity(100j, 'm'), 3)
    np.testing.assert_allclose(values.value, [1j, 10j, 100j])
    with pytest.raises(TypeError):
        np.geomspace(AffineQuantity(1, 'K'), AffineQuantity(10, 'K'))


def test_logspace_dimensionless() -> None:
    values = np.logspace(Quantity(0, 'cm/m'), Quantity(200, 'cm/m'), 3, base=Quantity(1000, 'cm/m'))
    assert values.unit == Unit()
    np.testing.assert_allclose(values.value, [1, 10, 100])
    with pytest.raises(DimensionError):
        np.logspace(Quantity(1, 'm'), Quantity(2, 'm'))


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('indexing', ['ij', 'xy'])
def test_meshgrid_independent_units(sparse: bool, indexing: str) -> None:
    x = Quantity(np.arange(3.0), 'm')
    t = AffineQuantity(np.arange(2.0), 'K')
    y = np.arange(4.0)
    grids = np.meshgrid(x, t, y, sparse=sparse, indexing=indexing, copy=False)
    expected = np.meshgrid(x.value, t.value, y, sparse=sparse, indexing=indexing, copy=False)
    assert grids[0].unit == x.unit and grids[1].unit == t.unit
    assert type(grids[2]) is np.ndarray
    for actual, raw in zip(grids, expected):
        np.testing.assert_array_equal(actual.value if hasattr(actual, 'unit') else actual, raw)
    assert np.shares_memory(grids[0].value, x.value)


@pytest.mark.parametrize(
    'mode',
    [
        'constant',
        'edge',
        'linear_ramp',
        'maximum',
        'mean',
        'median',
        'minimum',
        'reflect',
        'symmetric',
        'wrap',
        'empty',
    ],
)
def test_pad_modes(array_sample: ArrayQuantity, mode: str) -> None:
    result = np.pad(array_sample, ((0, 0), (1, 2), (0, 0)), mode=mode)
    assert type(result) is type(array_sample) and result.unit == array_sample.unit
    if mode != 'empty':
        np.testing.assert_array_equal(
            result.value, np.pad(array_sample.value, ((0, 0), (1, 2), (0, 0)), mode=mode)
        )
    assert not np.shares_memory(result.value, array_sample.value)


def test_pad_converts_constants() -> None:
    q = Quantity(np.array([1.0, 2.0]), 'm')
    np.testing.assert_array_equal(
        np.pad(q, 1, constant_values=(Quantity(50, 'cm'), 3)).value, [0.5, 1, 2, 3]
    )
    np.testing.assert_array_equal(
        np.pad(q, 1, mode='linear_ramp', end_values=Quantity(50, 'cm')).value, [0.5, 1, 2, 0.5]
    )
    p = AffineQuantity(np.array([10.0, 20.0]), 'degC')
    np.testing.assert_allclose(
        np.pad(p, 1, constant_values=AffineQuantity(273.15, 'K')).value, [0, 10, 20, 0], atol=1e-12
    )
    with pytest.raises(DimensionError):
        np.pad(q, 1, constant_values=Quantity(1, 's'))
    with pytest.raises(TypeError):
        np.pad(q, 1, mode=lambda *args: None)


def test_sliding_window_view_storage(array_sample: ArrayQuantity) -> None:
    windows = np.lib.stride_tricks.sliding_window_view(array_sample, 2, axis=1)
    assert type(windows) is type(array_sample) and windows.unit == array_sample.unit
    assert not windows.value.flags.writeable
    assert np.shares_memory(windows.value, array_sample.value)
    np.testing.assert_array_equal(
        windows.value, np.lib.stride_tricks.sliding_window_view(array_sample.value, 2, axis=1)
    )
    writable = np.lib.stride_tricks.sliding_window_view(array_sample, 2, axis=1, writeable=True)
    writable.value[...] = 5
    np.testing.assert_array_equal(array_sample.value, 5)


@pytest.mark.parametrize('function', [np.zeros_like, np.ones_like, np.empty_like, np.full_like])
def test_allocation_invalid_dtype(function: Callable[..., Any]) -> None:
    kwargs = {'fill_value': 1} if function is np.full_like else {}
    with pytest.raises(TypeError):
        function(Quantity(np.zeros(2), 'm'), dtype=bool, **kwargs)
