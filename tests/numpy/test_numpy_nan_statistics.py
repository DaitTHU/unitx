"""NaN omission preserves statistical units and staged output guarantees."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

np = pytest.importorskip('numpy')

from unitx import AffineQuantity, LogQuantity, Quantity, Unit  # noqa: E402
from unitx.exceptions import DimensionError  # noqa: E402
from unitx.factor import Factor  # noqa: E402
from unitx.imperial import imperial  # noqa: E402
from unitx.si import si  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


STATISTICS = [np.nanmean, np.nanmedian, np.nanstd, np.nanvar, np.nanquantile, np.nanpercentile]
MOMENTS = [np.nanmean, np.nanstd, np.nanvar]


@pytest.mark.parametrize('operation', STATISTICS)
@pytest.mark.parametrize('axis', [None, 0, -1, (0, -1)])
@pytest.mark.parametrize('unit', [Unit('cm', scale=Factor(2)), Unit('km/m')])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_numpy_nan_statistics_axes(
    operation: Callable[..., object],
    axis: int | tuple[int, ...] | None,
    unit: Unit,
    dtype: DTypeLike,
) -> None:
    values = np.arange(48, dtype=dtype).reshape(2, 3, 8)[..., ::2]
    values[0, :, 0] = np.nan
    saved = values.copy()
    values.flags.writeable = False
    samples = Quantity(values, unit)
    args = (0.5,) if operation is np.nanquantile else (50,) if operation is np.nanpercentile else ()
    for keepdims in (False, True):
        result = operation(samples, *args, axis=axis, keepdims=keepdims)
        expected = operation(values, *args, axis=axis, keepdims=keepdims)
        assert result.unit == (unit**2 if operation is np.nanvar else unit)
        np.testing.assert_allclose(result.value, expected)
        assert np.asarray(result.value).dtype == np.asarray(expected).dtype
    np.testing.assert_array_equal(values, saved)


@pytest.mark.parametrize('operation', STATISTICS)
@pytest.mark.parametrize('empty', [False, True])
def test_numpy_nan_statistics_empty(operation: Callable[..., object], empty: bool) -> None:
    samples = Quantity(np.empty((2, 0)) if empty else np.full((2, 3), np.nan), 'm')
    args = (
        ([0.25, 0.75],)
        if operation is np.nanquantile
        else (([25, 75],) if operation is np.nanpercentile else ())
    )
    with pytest.warns(RuntimeWarning) as actual_warnings:
        result = operation(samples, *args, axis=1, keepdims=True)
    with pytest.warns(RuntimeWarning) as expected_warnings:
        expected = operation(samples.value, *args, axis=1, keepdims=True)
    assert result.unit == Unit('m2' if operation is np.nanvar else 'm')
    np.testing.assert_array_equal(result.value, expected)
    assert [str(w.message) for w in actual_warnings] == [str(w.message) for w in expected_warnings]


@pytest.mark.parametrize('operation', MOMENTS)
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex128])
def test_numpy_nan_statistics_where(operation: Callable[..., object], dtype: DTypeLike) -> None:
    values = np.array([[1, np.nan, 3, 20], [4, 6, np.nan, 8]], dtype=dtype)
    if dtype is np.complex128:
        values += 1j
    samples = Quantity(values, 'm')
    mask = np.array([[True, True, True, False], [True, False, True, True]])
    ddofs = (None,) if operation is np.nanmean else (0, 0.5, 1)
    for ddof in ddofs:
        options = {} if ddof is None else {'ddof': ddof}
        result = operation(samples, axis=1, where=mask, keepdims=True, **options)
        expected = operation(values, axis=1, where=mask, keepdims=True, **options)
        np.testing.assert_allclose(result.value, expected)
        assert result.unit == Unit('m2' if operation is np.nanvar else 'm')
        assert np.asarray(result.value).dtype == np.asarray(expected).dtype


@pytest.mark.parametrize('operation', STATISTICS)
def test_numpy_nan_statistics_out_alias(operation: Callable[..., object]) -> None:
    storage = np.array([[np.nan, 1, 3, 5], [2, 4, np.nan, 6]])
    saved = storage.copy()
    samples = Quantity(storage, Unit('m', scale=Factor(2)))
    target = Quantity(storage[:, 0], 'cm2' if operation is np.nanvar else 'cm')
    args = (0.5,) if operation is np.nanquantile else (50,) if operation is np.nanpercentile else ()
    source_unit = samples.unit**2 if operation is np.nanvar else samples.unit
    expected = Quantity(operation(saved, *args, axis=1), source_unit).value_in(target.unit)
    buffer, unit, dtype = target.value, target.unit, target.value.dtype
    assert operation(samples, *args, axis=1, out=target) is target
    assert target.value is buffer and target.unit is unit and target.value.dtype == dtype
    saved[:, 0] = expected
    np.testing.assert_allclose(storage, saved)


@pytest.mark.parametrize('operation', STATISTICS)
def test_numpy_nan_statistics_invalid_warning(operation: Callable[..., object]) -> None:
    samples = Quantity(np.full(3, np.nan), 'm')
    target = Quantity(np.array(-123.0), 'm2' if operation is np.nanvar else 'm')
    args = (0.5,) if operation is np.nanquantile else (50,) if operation is np.nanpercentile else ()
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(RuntimeWarning):
            operation(samples, *args, out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', [np.nanmedian, np.nanmean, np.nanstd, np.nanvar])
@pytest.mark.parametrize(
    'point_type,coordinate', [(AffineQuantity, si.degC), (LogQuantity, si.dBm)]
)
def test_numpy_nan_statistics_points(
    operation: Callable[..., object], point_type: type, coordinate: object
) -> None:
    samples = point_type(np.array([[0.0, np.nan, 20], [30, 50, np.nan]]), coordinate)
    result = operation(samples, axis=1)
    np.testing.assert_allclose(result.value, operation(samples.value, axis=1))
    if operation in (np.nanmean, np.nanmedian):
        assert type(result) is point_type and result.unit is samples.unit
        target = point_type(np.empty(2), imperial.degF if point_type is AffineQuantity else si.dBW)
    else:
        unit = samples.unit.delta_unit
        assert type(result) is Quantity
        assert result.unit == (unit**2 if operation is np.nanvar else unit)
        target = Quantity(np.empty(2), result.unit)
        with pytest.raises(DimensionError):
            operation(samples, axis=1, out=samples)
    assert operation(samples, axis=1, out=target) is target
    np.testing.assert_allclose(target.value, result.value_in(target.unit))


@pytest.mark.parametrize('operation', [np.nanquantile, np.nanpercentile])
@pytest.mark.parametrize('axis', [None, 0, 1, -1])
def test_numpy_nan_quantile_weights(operation: Callable[..., object], axis: int | None) -> None:
    values = np.arange(24.0).reshape(2, 3, 4)
    values[0, 0, 0] = np.nan
    values[1, 2, 1] = np.nan
    shape = values.shape
    weights = np.arange(1.0, np.prod(shape) + 1).reshape(shape)
    samples = Quantity(values, 'm')
    levels = [0.25, 0.75] if operation is np.nanquantile else [25, 75]
    probabilities = Quantity(np.array(levels) / 100, Unit('m/cm'))
    result = operation(
        samples, probabilities, axis=axis, weights=Quantity(weights, 'm/cm'), method='inverted_cdf'
    )
    expected = operation(values, levels, axis=axis, weights=weights * 100, method='inverted_cdf')
    assert result.unit is samples.unit
    np.testing.assert_allclose(result.value, expected)


@pytest.mark.parametrize('operation', [np.nanquantile, np.nanpercentile])
def test_numpy_nan_quantile_omitted_weights(operation: Callable[..., object]) -> None:
    samples = Quantity(np.array([1.0, np.nan, 3, 10]), 'm')
    weights = np.array([1, 10000, 3, 1])
    levels = [0.25, 0.75] if operation is np.nanquantile else [25, 75]
    result = operation(samples, levels, weights=weights, method='inverted_cdf')
    expected = operation(
        np.array([1.0, 3, 10]), levels, weights=np.array([1, 3, 1]), method='inverted_cdf'
    )
    np.testing.assert_array_equal(result.value, expected)


@pytest.mark.parametrize('operation', [np.nanquantile, np.nanpercentile])
@pytest.mark.parametrize('axis', [(0, 1), (1,)])
def test_numpy_nan_quantile_invalid_weighted_axis(
    operation: Callable[..., object], axis: tuple[int, ...]
) -> None:
    values = np.array([[3.0, np.nan, 1], [4, 5, 6]])
    samples = Quantity(values, 'm')
    saved = values.copy()
    target = Quantity(values[:1, :1].reshape(()), 'cm')
    with pytest.raises(TypeError, match='one integer axis'):
        operation(samples, 0.5, axis=axis, weights=np.ones(values.shape), out=target)
    np.testing.assert_array_equal(values, saved)


@pytest.mark.parametrize('operation', [np.nanmedian, np.nanquantile, np.nanpercentile])
def test_numpy_nan_statistics_invalid_overwrite(operation: Callable[..., object]) -> None:
    values = np.array([3.0, np.nan, 1, 2])
    saved = values.copy()
    samples = Quantity(values, 'm')
    target = Quantity(values[:1].reshape(()), 'cm')
    args = () if operation is np.nanmedian else (0.5,)
    with pytest.raises(TypeError, match='overwrite_input'):
        operation(samples, *args, overwrite_input=True, out=target)
    np.testing.assert_array_equal(values, saved)
