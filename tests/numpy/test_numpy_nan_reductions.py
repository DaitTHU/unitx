"""NaN extrema, indices, sums and products retain their unit contracts."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

np = pytest.importorskip('numpy')

from unitx import AffineQuantity, LogQuantity, Quantity, Unit  # noqa: E402
from unitx.exceptions import DimensionError  # noqa: E402
from unitx.factor import Factor  # noqa: E402
from unitx.imperial import imperial  # noqa: E402
from unitx.si import si  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


EXTREMA = [np.nanmin, np.nanmax]
INDICES = [np.nanargmin, np.nanargmax]


@pytest.mark.parametrize('operation', [*EXTREMA, np.nansum])
@pytest.mark.parametrize('axis', [None, 0, -1, (), (0, -1)])
@pytest.mark.parametrize('dtype', [np.int32, np.float32, np.longdouble, np.complex128])
@pytest.mark.parametrize('unit', [Unit('cm', scale=Factor(2)), Unit('km/m')])
def test_numpy_nan_reduction_axes(
    operation: Callable[..., Any],
    axis: int | tuple[int, ...] | None,
    dtype: DTypeLike,
    unit: Unit,
) -> None:
    values = np.arange(48, dtype=dtype).reshape(2, 3, 8)[..., ::2]
    if values.dtype.kind in 'fc':
        values[0, :, 0] = np.nan
    saved = values.copy()
    values.flags.writeable = False
    samples = Quantity(values, unit)
    for keepdims in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # axis=() retains all-NaN slices.
            result = operation(samples, axis=axis, keepdims=keepdims)
            expected = operation(values, axis=axis, keepdims=keepdims)
        assert result.unit == unit
        np.testing.assert_array_equal(result.value, expected)
        assert np.asarray(result.value).dtype == np.asarray(expected).dtype
    np.testing.assert_array_equal(values, saved)


@pytest.mark.parametrize('operation', EXTREMA)
def test_numpy_nan_extremum_initial(operation: Callable[..., Any]) -> None:
    samples = Quantity(np.array([[np.nan, 2, 3], [4, np.nan, 6]]), 'm')
    mask = np.array([True, True, False])
    target = Quantity(np.empty((2, 1)), 'cm')
    expected = operation(samples.value, 1, None, True, 2.5, mask)
    assert operation(samples, 1, target, True, Quantity(250, 'cm'), mask) is target
    np.testing.assert_allclose(target.value, expected * 100)
    empty = Quantity(np.empty((2, 0)), 'm')
    np.testing.assert_array_equal(operation(empty, axis=1, initial=Quantity(2, 'm')).value, 2)


@pytest.mark.parametrize('operation', EXTREMA)
def test_numpy_nan_extremum_warning(operation: Callable[..., Any]) -> None:
    samples = Quantity(np.full((2, 3), np.nan), 'm')
    with pytest.warns(RuntimeWarning, match='All-NaN'):
        result = operation(samples, axis=1)
    assert result.unit == si.m
    np.testing.assert_array_equal(result.value, np.nan)
    target = Quantity(np.full(2, -123.0), 'cm')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(RuntimeWarning, match='All-NaN'):
            operation(samples, axis=1, out=target)
    np.testing.assert_array_equal(target.value, -123)
    with pytest.raises(ValueError, match='zero-size'):
        operation(Quantity(np.empty((2, 0)), 'm'), axis=1, out=target)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('operation', [*EXTREMA, *INDICES])
@pytest.mark.parametrize(
    'point_type,coordinate,target_coordinate',
    [
        (AffineQuantity, si.degC, imperial.degF),
        (LogQuantity, si.dBm, si.dBW),
    ],
)
def test_numpy_nan_selection_points(
    operation: Callable[..., Any], point_type: type, coordinate: Any, target_coordinate: Any
) -> None:
    samples = point_type(np.array([[np.nan, 0, 20], [30, 50, np.nan]]), coordinate)
    expected = operation(samples.value, axis=1)
    if operation in INDICES:
        target = np.full(2, -123, dtype=np.intp)
        result = operation(samples, axis=1)
        np.testing.assert_array_equal(result, expected)
        assert operation(samples, 1, target) is target
        np.testing.assert_array_equal(target, expected)
    else:
        result = operation(samples, axis=1)
        assert type(result) is point_type and result.unit is coordinate
        np.testing.assert_array_equal(result.value, expected)
        target = point_type(np.empty(2), target_coordinate)
        assert operation(samples, axis=1, out=target) is target
        np.testing.assert_allclose(target.value, result.value_in(target_coordinate))
        initial = point_type(10, coordinate).to(target_coordinate)
        bounded = operation(samples, axis=1, initial=initial, where=True)
        np.testing.assert_allclose(bounded.value, operation(samples.value, axis=1, initial=10))


@pytest.mark.parametrize('operation', INDICES)
@pytest.mark.parametrize('axis', [None, 0, -1])
def test_numpy_nan_index_axes(operation: Callable[..., Any], axis: int | None) -> None:
    values = np.array([[np.nan, 2, np.inf, 2], [-np.inf, 5, 5, np.nan]])[:, ::-1]
    samples = Quantity(values, 'km/m')
    for keepdims in (False, True):
        expected = operation(values, axis=axis, keepdims=keepdims)
        target = np.full(np.shape(expected), -123, dtype=np.intp)
        assert operation(samples, axis, target, keepdims=keepdims) is target
        np.testing.assert_array_equal(target, expected)


@pytest.mark.parametrize('operation', INDICES)
@pytest.mark.parametrize('values', [np.full((2, 3), np.nan), np.empty((2, 0))])
def test_numpy_nan_index_invalid_slice(operation: Callable[..., Any], values: Any) -> None:
    target = np.full(2, -123, dtype=np.intp)
    with pytest.raises(ValueError):
        operation(Quantity(values, 'm'), axis=1, out=target)
    np.testing.assert_array_equal(target, -123)


@pytest.mark.parametrize('operation', EXTREMA)
@pytest.mark.parametrize('initial', [0, Quantity(0, 's'), 0 @ si.degC])
def test_numpy_nan_extremum_invalid_initial(operation: Callable[..., Any], initial: Any) -> None:
    target = Quantity(np.array(-123.0), 'm')
    with pytest.raises((TypeError, DimensionError)):
        operation(Quantity(np.array([1.0, np.nan]), 'm'), initial=initial, out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', EXTREMA)
def test_numpy_nan_extremum_invalid_convention(operation: Callable[..., Any]) -> None:
    samples = LogQuantity(np.array([np.nan, 10.0]), si.dBW)
    target = LogQuantity(np.array(-123.0), si.dBm)
    with pytest.raises(DimensionError, match='same convention'):
        operation(samples, initial=0 @ si.dB(si.W, kind='amplitude'), out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', [*EXTREMA, *INDICES, np.nansum, np.nanprod])
@pytest.mark.parametrize('invalid', ['shape', 'dtype', 'readonly', 'wrapper'])
def test_numpy_nan_reduction_invalid_out(operation: Callable[..., Any], invalid: str) -> None:
    samples = Quantity(np.array([[np.nan, 1, 3], [4, np.nan, 6]]), 'm')
    indices = operation in INDICES
    dtype = np.intp if indices else np.float64
    unit = si.m**2 if operation is np.nanprod else si.m
    storage = np.full(3 if invalid == 'shape' else 2, -123, dtype=dtype)
    if invalid == 'dtype':
        storage = storage.astype(float if indices else int)
    if invalid == 'readonly':
        storage.flags.writeable = False
    if (invalid == 'wrapper') == indices:
        target = Quantity(storage, unit)
    else:
        target = storage
    with pytest.raises((TypeError, ValueError)):
        operation(samples, axis=1, out=target)
    np.testing.assert_array_equal(storage, -123)


@pytest.mark.parametrize('operation', [*EXTREMA, *INDICES, np.nansum, np.nanprod])
def test_numpy_nan_reduction_out_alias(operation: Callable[..., Any]) -> None:
    dtype = np.intp if operation in INDICES else np.float64
    storage = np.array([[1, 3, 2], [4, 6, 5]], dtype=dtype)
    if operation not in INDICES:
        storage[:, -1] = np.nan
    saved = storage.copy()
    samples = Quantity(storage, Unit('m', scale=Factor(2)))
    unit = samples.unit**2 if operation is np.nanprod else samples.unit
    target = storage[:, 0] if operation in INDICES else Quantity(storage[:, 0], unit)
    expected = operation(saved, axis=1)
    assert operation(samples, axis=1, out=target) is target
    saved[:, 0] = expected
    np.testing.assert_array_equal(storage, saved)


def test_numpy_nansum_mask_and_empty() -> None:
    samples = Quantity(np.array([[1.0, np.nan, 3], [np.nan, 5, 7]]), 'm')
    mask = np.array([[True, True, False], [False, True, True]])
    target = Quantity(np.empty(2), 'cm')
    assert np.nansum(samples, 1, np.float32, target, False, Quantity(50, 'cm'), mask) is target
    np.testing.assert_array_equal(target.value, [150, 1250])
    for values in (np.full((2, 3), np.nan), np.empty((2, 0))):
        result = np.nansum(Quantity(values, 'm'), axis=1)
        assert result.unit == si.m
        np.testing.assert_array_equal(result.value, 0)


@pytest.mark.parametrize('axis,count', [(None, 4), (0, 1), (-1, 2), ((0, 1), 4)])
def test_numpy_nanprod_counts(axis: int | tuple[int, ...] | None, count: int) -> None:
    values = np.array([[2.0, np.nan, 3, np.nan], [np.nan, 4, np.nan, 5]])
    samples = Quantity(values, Unit('cm', scale=Factor(2)))
    for keepdims in (False, True):
        result = np.nanprod(samples, axis=axis, keepdims=keepdims, initial=Quantity(2, 's'))
        assert result.unit == samples.unit**count * si.s
        np.testing.assert_array_equal(
            result.value, np.nanprod(values, axis=axis, keepdims=keepdims, initial=2)
        )
    target = Quantity(np.empty(2), 'm2 s')
    assert np.nanprod(samples, 1, None, target, False, Quantity(2, 's'), True) is target
    np.testing.assert_allclose(target.value, np.nanprod(values, axis=1) * 0.02**2 * 2)


def test_numpy_nanprod_mask() -> None:
    samples = Quantity(np.array([[2.0, np.nan, 3], [np.nan, 4, 5]]), 'm')
    mask = np.array([True, True, False])
    result = np.nanprod(samples, axis=1, where=mask)
    assert result.unit == si.m
    np.testing.assert_array_equal(result.value, [2, 4])
    samples = Quantity(np.array([[0.0, np.nan, 3], [np.inf, 4, np.nan]]), 'm')
    assert np.nanprod(samples, axis=1).unit == si.m**2  # Zero and infinity still count.


@pytest.mark.parametrize(
    'values,axis',
    [
        (np.full((2, 3), np.nan), 1),
        (np.array(np.nan), None),
        (np.array(np.nan), 0),
        (np.array(np.nan), -1),
        (np.array(np.nan), ()),
        (np.empty((2, 0)), 1),
    ],
)
def test_numpy_nanprod_identity(values: Any, axis: int | tuple[int, ...] | None) -> None:
    result = np.nanprod(Quantity(values, 'm'), axis=axis, initial=Quantity(2, 's'))
    assert result.unit == si.s
    np.testing.assert_array_equal(result.value, 2)


@pytest.mark.parametrize('unit', ['m', 'dB'])
@pytest.mark.parametrize('axis', [1, ()])
def test_numpy_nanprod_invalid_counts(unit: str, axis: int | tuple[int, ...]) -> None:
    samples = Quantity(np.array([[2.0, np.nan, 3], [4, 5, 6]]), unit)
    target = Quantity(np.full((2,) if axis == 1 else (2, 3), -123.0), unit)
    with pytest.raises(DimensionError, match='equal element counts'):
        np.nanprod(samples, axis=axis, initial=0, out=target)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('unit', ['km/m', 'deg', ''])
def test_numpy_nanprod_dimensionless(unit: str) -> None:
    samples = Quantity(np.array([[2.0, np.nan, 3], [4, 5, 6]]), unit)
    result = np.nanprod(samples, axis=1, initial=Quantity(2, 's'))
    assert result.unit == si.s
    np.testing.assert_allclose(result.value, np.nanprod(samples.value_in(''), axis=1, initial=2))
