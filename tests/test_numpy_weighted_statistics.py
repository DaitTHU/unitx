"""Dimensionless statistical parameters and fixed-unit statistical outputs."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING

import pytest

from unitx._numerics import Magnitude

np = pytest.importorskip('numpy')

from unitx import AffineQuantity, LogQuantity, Quantity, Unit  # noqa: E402
from unitx.exceptions import DimensionError  # noqa: E402
from unitx.factor import Factor  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


QUANTILES = [np.quantile, np.percentile]
STATISTICS = [np.average, *QUANTILES]
METHODS = [
    'inverted_cdf',
    'averaged_inverted_cdf',
    'closest_observation',
    'interpolated_inverted_cdf',
    'hazen',
    'weibull',
    'linear',
    'median_unbiased',
    'normal_unbiased',
    'lower',
    'higher',
    'midpoint',
    'nearest',
]


@pytest.mark.parametrize('operation', STATISTICS)
@pytest.mark.parametrize('axis', [None, 0, -1, (2, 0)])
@pytest.mark.parametrize('unit', [Unit('cm', scale=Factor(2)), Unit('km/m')])
def test_numpy_weighted_statistics_axes(
    operation: Callable[..., object], axis: int | tuple[int, ...] | None, unit: Unit
) -> None:
    values = np.arange(48.0).reshape(2, 3, 8)[..., ::2]
    values.flags.writeable = False
    samples = Quantity(values, unit)
    levels = [0.25, 0.75] if operation is np.quantile else [25, 75]
    args = () if operation is np.average else (levels,)
    for keepdims in (False, True):
        result = operation(samples, *args, axis=axis, keepdims=keepdims)
        expected = operation(values, *args, axis=axis, keepdims=keepdims)
        assert result.unit is unit
        np.testing.assert_allclose(result.value, expected)
        assert np.asarray(result.value).dtype == np.asarray(expected).dtype


@pytest.mark.parametrize('operation', STATISTICS)
@pytest.mark.parametrize('dtype', [np.int32, np.float32, np.longdouble])
@pytest.mark.parametrize(
    'axis,shape', [(None, (2, 3, 4)), (1, (3,)), ((2, 0), (4, 2)), (-1, (2, 3, 4))]
)
def test_numpy_weighted_statistics_weights(
    operation: Callable[..., object],
    dtype: DTypeLike,
    axis: int | tuple[int, ...] | None,
    shape: tuple[int, ...],
) -> None:
    values = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    weights = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
    weights.flags.writeable = False
    samples = Quantity(values, 'm')
    scaled_weights = Quantity(weights, Unit('m/cm'))
    args = (
        () if operation is np.average else ([0.2, 0.8] if operation is np.quantile else [20, 80],)
    )
    options = {} if operation is np.average else {'method': 'inverted_cdf'}
    result = operation(samples, *args, axis=axis, weights=scaled_weights, keepdims=True, **options)
    expected = operation(values, *args, axis=axis, weights=weights * 100, keepdims=True, **options)
    assert result.unit == samples.unit
    np.testing.assert_allclose(result.value, expected)
    assert np.asarray(result.value).dtype == np.asarray(expected).dtype


@pytest.mark.parametrize('weights', [None, [1, 2, 1], Quantity(np.array([1, 2, 1]), 'm/cm')])
@pytest.mark.parametrize('keepdims', [False, True])
def test_numpy_average_returned(
    weights: list[int] | Quantity[Magnitude] | None, keepdims: bool
) -> None:
    samples = Quantity(np.arange(6, dtype=np.float32).reshape(2, 3), 'm')
    actual, weight_sum = np.average(samples, 1, weights, True, keepdims=keepdims)
    numeric_weights = weights.value_in('') if isinstance(weights, Quantity) else weights
    expected, expected_sum = np.average(samples.value, 1, numeric_weights, True, keepdims=keepdims)
    assert actual.unit == samples.unit and weight_sum.unit == Unit()
    assert np.shape(weight_sum.value) == np.shape(actual.value)
    np.testing.assert_array_equal(actual.value, expected)
    np.testing.assert_array_equal(weight_sum.value, expected_sum)
    assert actual.value.dtype == expected.dtype and weight_sum.value.dtype == expected_sum.dtype


def test_numpy_average_complex_and_negative_weights() -> None:
    samples = Quantity(np.array([1 + 2j, 4 - 1j]), 'm')
    for weights in ([-1, 2], [1j, 2]):
        result = np.average(samples, weights=weights)
        np.testing.assert_allclose(result.value, np.average(samples.value, weights=weights))


@pytest.mark.parametrize('operation', STATISTICS)
@pytest.mark.parametrize('value', [2, Fraction(2), Decimal(2), np.float32(2), np.array(2.0)])
def test_numpy_weighted_statistics_scalar(
    operation: Callable[..., object],
    value: int | Fraction | Decimal | np.float32 | NDArray[np.generic],
) -> None:
    samples = Quantity(value, 'm')
    args = () if operation is np.average else (0.5 if operation is np.quantile else 50,)
    result = operation(samples, *args)
    assert result.unit == samples.unit and result.value == 2
    assert isinstance(result.value, np.number)
    if operation is np.average:
        average, weight_sum = operation(samples, returned=True)
        assert average.value == 2 and weight_sum.value == 1 and weight_sum.unit == Unit()


@pytest.mark.parametrize('operation', QUANTILES)
@pytest.mark.parametrize('method', METHODS)
def test_numpy_quantile_methods(operation: Callable[..., object], method: str) -> None:
    samples = Quantity(np.array([5, 1, 9, 3]), 'm')
    levels = [0, 0.25, 0.6, 1] if operation is np.quantile else [0, 25, 60, 100]
    result = operation(samples, levels, method=method)
    np.testing.assert_allclose(result.value, operation(samples.value, levels, method=method))
    assert result.unit == samples.unit
    np.testing.assert_array_equal(samples.value, [5, 1, 9, 3])


@pytest.mark.parametrize('operation', QUANTILES)
def test_numpy_quantile_probabilities(operation: Callable[..., object]) -> None:
    samples = Quantity(np.array([0.0, 10]), 'km/m')
    # Normalize first; percentile still interprets the normalized value as a percentage.
    levels = np.array([0.0, 0.5, 1.0]) if operation is np.quantile else np.array([0, 50, 100.0])
    q = Quantity(levels / 100, Unit('m/cm'))
    result = operation(samples, q)
    assert result.unit == samples.unit
    np.testing.assert_allclose(result.value, [0, 5, 10])
    # Scalar, 0-D and ordinary sequence spellings are accepted as parameters.
    level = 0.5 if operation is np.quantile else 50
    for q in (level, np.array(level), Fraction(level), Decimal(level), Quantity(level, '')):
        assert operation(samples, q).value == 5
    np.testing.assert_array_equal(operation(samples, (level,)).value, [5])


@pytest.mark.parametrize('operation', QUANTILES)
@pytest.mark.parametrize('alias', ['samples', 'q', 'weights'])
def test_numpy_quantile_out_alias(operation: Callable[..., object], alias: str) -> None:
    storage = np.array([3.0, 1, 2, 4, 0.25, 0.75, 1, 2, 1, 1])
    if operation is np.percentile:
        storage[4:6] *= 100
    samples = Quantity(storage[:4], Factor(2) * Unit('m'))
    q = Quantity(storage[4:6], '')
    weights = Quantity(storage[6:], '')
    offset = {'samples': 0, 'q': 4, 'weights': 6}[alias]
    target = Quantity(storage[offset : offset + 2], 'cm')
    original = storage.copy()
    expected = (
        operation(
            samples.value.copy(),
            q.value.copy(),
            weights=weights.value.copy(),
            method='inverted_cdf',
        )
        * 200
    )
    buffer, unit, dtype = target.value, target.unit, target.value.dtype
    assert operation(samples, q, None, target, False, 'inverted_cdf', weights=weights) is target
    assert target.value is buffer and target.unit is unit and target.value.dtype == dtype
    original[offset : offset + 2] = expected
    np.testing.assert_array_equal(storage, original)


@pytest.mark.parametrize('operation', QUANTILES)
def test_numpy_quantile_out_axes(operation: Callable[..., object]) -> None:
    samples = Quantity(np.arange(24.0).reshape(2, 3, 4), 'm')
    levels = [0.25, 0.75] if operation is np.quantile else [25, 75]
    storage = np.full((2, 1, 3, 1, 2), -123.0)
    target = Quantity(storage[..., 0], 'cm')
    result = operation(samples, levels, (2, 0), target, False, 'linear', True)
    assert result is target
    np.testing.assert_allclose(
        target.value, operation(samples.value, levels, axis=(2, 0), keepdims=True) * 100
    )
    np.testing.assert_array_equal(storage[..., 1], -123)


@pytest.mark.parametrize('operation', STATISTICS)
@pytest.mark.parametrize('weights', [Quantity(1, 's'), Quantity(1, 'dB')])
def test_numpy_weighted_statistics_invalid_dimension(
    operation: Callable[..., object], weights: Quantity[Magnitude]
) -> None:
    args = () if operation is np.average else (0.5,)
    with pytest.raises(DimensionError, match='weights must be dimensionless'):
        operation(Quantity(np.arange(3.0), 'm'), *args, weights=weights)


@pytest.mark.parametrize('operation', QUANTILES)
@pytest.mark.parametrize('q', [Quantity(0.5, 's'), Quantity(0.5, 'dB')])
def test_numpy_quantile_invalid_dimension(
    operation: Callable[..., object], q: Quantity[Magnitude]
) -> None:
    target = Quantity(np.array(-123.0), 'm')
    with pytest.raises(DimensionError, match='q must be dimensionless'):
        operation(Quantity(np.arange(3.0), 'm'), q, out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', STATISTICS)
@pytest.mark.parametrize(
    'point', [AffineQuantity(np.arange(3.0), 'degC'), LogQuantity(np.arange(3.0), 'dBm')]
)
def test_numpy_weighted_statistics_invalid_point(
    operation: Callable[..., object], point: AffineQuantity[Magnitude] | LogQuantity[Magnitude]
) -> None:
    args = () if operation is np.average else (0.5,)
    with pytest.raises(TypeError, match='convert explicitly'):
        operation(point, *args)


@pytest.mark.parametrize('operation', QUANTILES)
@pytest.mark.parametrize('parameter', ['q', 'weights'])
@pytest.mark.parametrize(
    'value',
    [
        True,
        [True],
        np.array([True]),
        ['0.5'],
        np.array([0.5], dtype=object),
        np.ma.array([0.5]),
        np.array([0.5]).view(np.recarray),
        AffineQuantity(0.5, 'K'),
        LogQuantity(0.5, 'dBm'),
    ],
)
def test_numpy_quantile_invalid_parameter(
    operation: Callable[..., object], parameter: str, value: object
) -> None:
    options = {'q': 0.5, 'weights': None, parameter: value}
    with pytest.raises(TypeError):
        operation(Quantity(np.arange(3.0), 'm'), **options)


@pytest.mark.parametrize('operation', STATISTICS)
@pytest.mark.parametrize(
    'axis,weights,error',
    [(None, [1, 2], TypeError), (1, [1, 2], ValueError), (1, [[1, 2, 3]], ValueError)],
)
def test_numpy_weighted_statistics_invalid_shape(
    operation: Callable[..., object], axis: int | None, weights: object, error: type[Exception]
) -> None:
    samples = Quantity(np.arange(6.0).reshape(2, 3), 'm')
    args = () if operation is np.average else (0.5,)
    options = {} if operation is np.average else {'method': 'inverted_cdf'}
    with pytest.raises(error):
        operation(samples, *args, axis=axis, weights=weights, **options)


def test_numpy_average_invalid_zero_weight_sum() -> None:
    with pytest.raises(ZeroDivisionError):
        np.average(Quantity(np.array([1.0, 2]), 'm'), weights=[-1, 1])


@pytest.mark.parametrize('operation', QUANTILES)
@pytest.mark.parametrize(
    'options',
    [
        {'q': -1},
        {'q': 101},
        {'q': np.nan},
        {'axis': 2},
        {'method': 'invalid'},
        {'weights': [1, 2, 1]},
        {'weights': [1, -1, 1], 'method': 'inverted_cdf'},
        {'overwrite_input': True},
    ],
)
def test_numpy_quantile_invalid_kernel(
    operation: Callable[..., object], options: dict[str, object]
) -> None:
    values = np.array([3.0, 1, 2])
    target = Quantity(values[:1].reshape(()), 'cm')
    with pytest.raises((TypeError, ValueError)):
        operation(Quantity(values, 'm'), out=target, **{'q': 0.5, **options})
    np.testing.assert_array_equal(values, [3, 1, 2])


@pytest.mark.parametrize('operation', QUANTILES)
def test_numpy_quantile_invalid_output(operation: Callable[..., object]) -> None:
    samples = Quantity(np.array([3.0, 1, 2]), 'm')
    for target, error in (
        (Quantity(np.array(-123.0), 's'), DimensionError),
        (Quantity(np.full(2, -123.0), 'm'), ValueError),
        (Quantity(np.array(-123), 'm'), TypeError),
        (np.array(-123.0), TypeError),
        (AffineQuantity(np.array(-123.0), 'K'), DimensionError),
    ):
        with pytest.raises(error):
            operation(samples, 0.5, out=target)
        np.testing.assert_array_equal(target if type(target) is np.ndarray else target.value, -123)
    target = Quantity(np.array(-123.0), 'm')
    target.value.flags.writeable = False
    with pytest.raises(ValueError, match='read-only'):
        operation(samples, 0.5, out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', QUANTILES)
def test_numpy_quantile_invalid_conversion(operation: Callable[..., object]) -> None:
    target = Quantity(np.array(-123.0), 'm')
    with np.errstate(over='raise'):
        with pytest.raises(FloatingPointError):
            operation(Quantity(np.array([1e308, 1e308]), 'km'), 0.5, out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', QUANTILES)
def test_numpy_quantile_invalid_complex(operation: Callable[..., object]) -> None:
    with pytest.raises(TypeError, match='real'):
        operation(Quantity(np.array([1j, 2j]), 'm'), 0.5)


@pytest.mark.parametrize('operation', QUANTILES)
def test_numpy_quantile_nan(operation: Callable[..., object]) -> None:
    result = operation(Quantity(np.array([1.0, np.nan, 3]), 'm'), 0.5)
    assert result.unit == Unit('m') and np.isnan(result.value)


@pytest.mark.parametrize('operation', [np.nanmean, np.nanquantile, np.nanpercentile])
def test_numpy_statistics_invalid_nan_function(operation: Callable[..., object]) -> None:
    args = () if operation is np.nanmean else (0.5,)
    with pytest.raises(TypeError):
        operation(Quantity(np.array([1.0, np.nan, 3]), 'm'), *args)


@pytest.mark.parametrize('operation', STATISTICS)
@pytest.mark.parametrize(
    'options', [{'dtype': float}, {'where': True}, {'initial': 0}, {'interpolation': 'linear'}]
)
def test_numpy_weighted_statistics_invalid_options(
    operation: Callable[..., object], options: dict[str, object]
) -> None:
    args = () if operation is np.average else (0.5,)
    with pytest.raises(TypeError):
        operation(Quantity(np.arange(3.0), 'm'), *args, **options)


def test_numpy_average_invalid_out() -> None:
    with pytest.raises(TypeError):
        np.average(Quantity(np.arange(3.0), 'm'), out=Quantity(np.array(-123.0), 'm'))
