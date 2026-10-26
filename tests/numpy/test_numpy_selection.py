"""Unit-aware data cleaning, coordinate selection and extremum boundaries."""

from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity
from unitx.exceptions import DimensionError
from unitx.si import si

np = pytest.importorskip('numpy')


def test_numpy_where_units_and_broadcasting() -> None:
    left = Quantity(np.array([[1.0], [2.0]]), 'm')
    right = Quantity(np.array([50.0, 150, 250]), 'cm')
    mask = np.array([[True, False, True], [False, True, False]])
    result = np.where(mask, left, right)
    assert result.unit == si.m
    np.testing.assert_allclose(result.value, [[1, 1.5, 1], [0.5, 2, 2.5]])
    other_units = np.where(mask, left.to('cm'), right.to('m')).to('m')
    np.testing.assert_allclose(other_units.value, result.value)
    np.testing.assert_array_equal(left.value, [[1], [2]])


def test_numpy_where_dimensionless() -> None:
    scaled = Quantity(np.array([1.0, 2]), 'km/m')
    for left, right, expected in ((scaled, 0, [1, 0]), (0, scaled, [0, 2])):
        result = np.where(np.array([True, False]), left, right)
        assert result.unit == scaled.unit
        np.testing.assert_array_equal(result.value, expected)
    assert np.where(np.bool_(True), Quantity(Decimal('1.5'), ''), Fraction(1, 2)).value == 1.5
    np.testing.assert_array_equal(np.where(scaled > 1500)[0], [1])


@pytest.mark.parametrize('condition', [1, [True, False], np.array([1, 0]), Quantity(1, '')])
def test_numpy_where_invalid_condition(condition: Any) -> None:
    with pytest.raises(TypeError, match='boolean mask'):
        np.where(condition, Quantity(np.array([1, 2]), 'm'), Quantity(0, 'm'))


@pytest.mark.parametrize('other', [0, Quantity(1, 's'), Quantity(0, 'dB')])
def test_numpy_where_invalid_branch(other: Any) -> None:
    with pytest.raises(DimensionError):
        np.where(True, Quantity(np.array([1, 2]), 'm'), other)


@pytest.mark.parametrize('operation', [np.minimum, np.maximum])
def test_numpy_selection_units(operation: Callable[..., Any]) -> None:
    left = Quantity(np.array([1.0, np.nan, np.inf]), 'm')
    right = Quantity(np.array([150.0, 200, 250]), 'cm')
    result = operation(left, right)
    assert result.unit == si.m
    np.testing.assert_equal(result.value, operation(left.value, right.value / 100))
    np.testing.assert_equal(operation(left.to('cm'), right).value / 100, result.value)
    scaled = Quantity(np.array([1, 2]), 'km/m')
    np.testing.assert_equal(operation(1500, scaled).value, operation(1.5, scaled.value))


@pytest.mark.parametrize('operation', [np.minimum, np.maximum])
def test_numpy_selection_masked_out(operation: Callable[..., Any]) -> None:
    storage = np.array([1.0, 2, 3, 4])
    source = Quantity(storage[:3], 'm')
    target = Quantity(storage[1:], 'cm')
    mask = np.array([True, False, True])
    expected = operation(source.value.copy(), 2.5) * 100
    assert operation(source, Quantity(250, 'cm'), out=target, where=mask) is target
    np.testing.assert_array_equal(storage, [1, expected[0], 3, expected[2]])


@pytest.mark.parametrize('operation', [np.minimum, np.maximum])
@pytest.mark.parametrize('other', [0, Quantity(1, 's'), Quantity(0, 'dB')])
def test_numpy_selection_invalid_dimension(operation: Callable[..., Any], other: Any) -> None:
    target = Quantity(np.full(2, -123.0), 'm')
    with pytest.raises(DimensionError):
        operation(Quantity(np.arange(2.0), 'm'), other, out=target)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('operation', [np.minimum, np.maximum])
def test_numpy_selection_methods(operation: Callable[..., Any]) -> None:
    q = Quantity(np.arange(3.0), 'm')
    reduced = operation.reduce(q)
    accumulated = operation.accumulate(q)
    assert reduced.unit == accumulated.unit == q.unit
    np.testing.assert_array_equal(reduced.value, operation.reduce(q.value))
    np.testing.assert_array_equal(accumulated.value, operation.accumulate(q.value))


def test_numpy_clip_bounds_and_options() -> None:
    q = Quantity(np.array([[-1.0, 1, 3], [2, 4, 6]]), 'm')
    lower = Quantity(np.array([0.0, 150, 200]), 'cm')
    upper = Quantity(5, 'm')
    for result in (np.clip(q, lower, upper), np.clip(q, min=lower, max=upper)):
        assert result.unit == q.unit
        np.testing.assert_allclose(result.value, [[0, 1.5, 3], [2, 4, 5]])
    np.testing.assert_allclose(np.clip(q.to('cm'), lower, upper).value / 100, result.value)
    np.testing.assert_array_equal(np.clip(q, None, upper).value, np.minimum(q.value, 5))
    np.testing.assert_array_equal(
        np.clip(q, min=lower).value, np.maximum(q.value, lower.value / 100)
    )
    np.testing.assert_array_equal(np.clip(q, max=upper).value, np.minimum(q.value, 5))
    np.testing.assert_array_equal(np.clip(q, None, None).value, q.value)
    np.testing.assert_array_equal(np.clip(q).value, q.value)
    np.testing.assert_array_equal(np.clip(q, upper, Quantity(0, 'm')).value, 0)
    assert np.clip(q, None, upper, dtype=np.float32).value.dtype == np.dtype(np.float32)


def test_numpy_clip_scaled_dimensionless() -> None:
    q = Quantity(np.array([-1.0, 1, 3]), 'km/m')
    result = np.clip(q, 0, 2000)
    assert result.unit == q.unit
    np.testing.assert_allclose(result.value, [0, 1, 2])


def test_numpy_clip_masked_out_alias() -> None:
    storage = np.array([-1.0, 1, 3, 4])
    source = Quantity(storage[:3], 'm')
    target = Quantity(storage[1:], 'cm')
    assert (
        np.clip(
            source,
            Quantity(0, 'm'),
            Quantity(200, 'cm'),
            target,
            where=np.array([True, False, True]),
        )
        is target
    )
    np.testing.assert_array_equal(storage, [-1, 0, 3, 200])
    q = Quantity(np.array([-1.0, 1, 3]), 'm')
    buffer = q.value
    assert np.clip(q, min=Quantity(0, 'm'), max=Quantity(2, 'm'), out=q) is q
    assert q.value is buffer
    np.testing.assert_array_equal(buffer, [0, 1, 2])


@pytest.mark.parametrize(
    'options',
    [
        {'a_min': Quantity(0, 'm')},
        {'a_min': None, 'a_max': None, 'min': None},
    ],
)
def test_numpy_clip_invalid_bound_names(options: dict[str, Any]) -> None:
    target = Quantity(np.full(2, -123.0), 'm')
    with pytest.raises(TypeError, match='Supply both a_min/a_max'):
        np.clip(Quantity(np.array([1.0, 2]), 'm'), out=target, **options)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('bound', [0, Quantity(1, 's')])
def test_numpy_clip_invalid_bound(bound: Any) -> None:
    target = Quantity(np.full(2, -123.0), 'm')
    with pytest.raises(DimensionError):
        np.clip(Quantity(np.array([1.0, 2]), 'm'), min=bound, out=target)
    np.testing.assert_array_equal(target.value, -123)


def test_numpy_clip_invalid_dtype() -> None:
    with pytest.raises(TypeError, match='numeric output dtype'):
        np.clip(Quantity(np.array([1.0, 2]), 'm'), dtype=object)


def test_numpy_clip_invalid_mask() -> None:
    with pytest.raises(TypeError, match='boolean mask'):
        np.clip(Quantity(np.array([1.0, 2]), 'm'), where=[True, False])


def test_numpy_clip_invalid_keyword() -> None:
    with pytest.raises(TypeError, match='Unsupported clip keyword'):
        np.clip(Quantity(np.array([1.0, 2]), 'm'), unknown=True)


@pytest.mark.parametrize('operation', [np.minimum, np.maximum, np.clip])
def test_numpy_selection_invalid_out_dimension(operation: Callable[..., Any]) -> None:
    q = Quantity(np.array([1.0, 2]), 'm')
    target = Quantity(np.full(2, -123.0), 's')
    args = (q, Quantity(150, 'cm'), None) if operation is np.clip else (q, Quantity(150, 'cm'))
    with pytest.raises(DimensionError, match='out has an incompatible dimension'):
        operation(*args, out=target)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('operation', [np.minimum, np.maximum, np.clip])
def test_numpy_selection_invalid_out_shape(operation: Callable[..., Any]) -> None:
    q = Quantity(np.array([1.0, 2]), 'm')
    target = Quantity(np.full(3, -123.0), 'cm')
    args = (q, Quantity(150, 'cm'), None) if operation is np.clip else (q, Quantity(150, 'cm'))
    with pytest.raises(ValueError):
        operation(*args, out=target)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('operation', [np.minimum, np.maximum, np.clip])
def test_numpy_selection_invalid_out_cast(operation: Callable[..., Any]) -> None:
    q = Quantity(np.array([1.0, 2]), 'm')
    target = Quantity(np.full(2, -123), 'cm')
    args = (q, Quantity(150, 'cm'), None) if operation is np.clip else (q, Quantity(150, 'cm'))
    with pytest.raises(TypeError):
        operation(*args, out=target)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('operation', [np.minimum, np.maximum, np.clip])
def test_numpy_selection_invalid_out_readonly(operation: Callable[..., Any]) -> None:
    q = Quantity(np.array([1.0, 2]), 'm')
    target = Quantity(np.full(2, -123.0), 'cm')
    target.value.flags.writeable = False
    args = (q, Quantity(150, 'cm'), None) if operation is np.clip else (q, Quantity(150, 'cm'))
    with pytest.raises(ValueError, match='read-only'):
        operation(*args, out=target)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('operation', [np.minimum, np.maximum, np.clip])
def test_numpy_selection_invalid_out_buffer(operation: Callable[..., Any]) -> None:
    q = Quantity(np.array([1.0, 2]), 'm')
    storage = np.full(2, -123.0)
    args = (q, Quantity(150, 'cm'), None) if operation is np.clip else (q, Quantity(150, 'cm'))
    with pytest.raises(TypeError):
        operation(*args, out=storage)
    np.testing.assert_array_equal(storage, -123)


@pytest.mark.parametrize('family', ['affine', 'logarithmic'])
def test_numpy_selection_points(family: str) -> None:
    if family == 'affine':
        q = AffineQuantity(np.array([-10.0, 10, 30]), si.degC)
        lower, upper = 273.15 @ si.K, 293.15 @ si.K
    else:
        q = LogQuantity(np.array([-10.0, 10, 30]), si.dBm)
        lower, upper = -30 @ si.dBW, -10 @ si.dBW
    clipped = np.clip(q, lower, upper)
    assert type(clipped) is type(q) and clipped.unit == q.unit
    np.testing.assert_allclose(clipped.value, [0, 10, 20], atol=1e-12)
    selected = np.where(np.array([True, False, True]), q, lower)
    assert type(selected) is type(q) and selected.unit == q.unit
    np.testing.assert_allclose(selected.value, [-10, 0, 30], atol=1e-12)
    target = q.to(upper.unit)
    assert np.clip(q, lower, upper, out=target) is target
    np.testing.assert_allclose(target.to(q.unit).value, clipped.value, atol=1e-12)
    for operation in (np.minimum, np.maximum):
        expected = operation(q.value, 20)
        result = operation(q, upper)
        assert type(result) is type(q) and result.unit == q.unit
        np.testing.assert_allclose(result.value, expected, atol=1e-12)


@pytest.mark.parametrize('operation', [np.minimum, np.maximum, np.clip, np.where])
def test_numpy_selection_invalid_points(operation: Callable[..., Any]) -> None:
    q = AffineQuantity(np.array([0.0, 20]), si.degC)
    args = (True, q, Quantity(0, 'K')) if operation is np.where else (q, Quantity(0, 'K'))
    if operation is np.clip:
        args = (*args, None)
    with pytest.raises(TypeError, match='points and differences'):
        operation(*args)


@pytest.mark.parametrize('operation', [np.minimum, np.maximum, np.clip, np.where, np.min])
def test_numpy_selection_invalid_convention(operation: Callable[..., Any]) -> None:
    q = LogQuantity(np.array([0.0, 20]), si.dBW)
    amplitude = 0 @ si.dB(si.W, kind='amplitude')
    if operation is np.min:
        with pytest.raises(DimensionError, match='same convention'):
            operation(q, initial=amplitude)
    else:
        args = (True, q, amplitude) if operation is np.where else (q, amplitude)
        if operation is np.clip:
            args = (*args, None)
        with pytest.raises(DimensionError, match='same convention'):
            operation(*args)


@pytest.mark.parametrize('operation', [np.min, np.max, np.amin, np.amax])
def test_numpy_extremum_options(operation: Callable[..., Any]) -> None:
    q = Quantity(np.arange(6.0).reshape(2, 3), 'm')
    mask = np.array([[True, False, True], [False, True, False]])
    target = Quantity(np.full((2, 1), -123.0), 'cm')
    expected = operation(q.value, axis=1, keepdims=True, where=mask, initial=2.5)
    assert (
        operation(q, axis=1, keepdims=True, where=mask, initial=Quantity(250, 'cm'), out=target)
        is target
    )
    np.testing.assert_allclose(target.value, expected * 100)
    empty = Quantity(np.empty((0, 3)), 'm')
    np.testing.assert_allclose(operation(empty, axis=0, initial=Quantity(250, 'cm')).value, 2.5)
    with pytest.raises(ValueError):
        operation(empty, axis=0)
    assert np.isnan(operation(Quantity(np.array([1.0, np.nan]), 'm')).value)


@pytest.mark.parametrize('operation', [np.min, np.max])
def test_numpy_extremum_invalid_initial(operation: Callable[..., Any]) -> None:
    target = Quantity(np.array(-123.0), 'm')
    with pytest.raises(TypeError, match='initial must carry'):
        operation(Quantity(np.array([1.0, 2]), 'm'), initial=0, out=target)
    np.testing.assert_array_equal(target.value, -123)


def test_numpy_selection_empty_and_scalar() -> None:
    empty = Quantity(np.empty((0, 2)), 'm')
    assert np.clip(empty, Quantity(0, 'm'), None).shape == (0, 2)
    assert np.where(np.empty((0, 2), dtype=bool), empty, Quantity(0, 'm')).shape == (0, 2)
    q = Quantity(np.array(2.0), 'm')
    assert np.where(True, q, Quantity(100, 'cm')).shape == ()
    assert np.minimum(q, Quantity(100, 'cm')).value == 1
    assert np.clip(q, None, Quantity(100, 'cm')).value == 1


@pytest.mark.parametrize('operation', [np.minimum, np.maximum, np.clip])
def test_numpy_selection_out_broadcast(operation: Callable[..., Any]) -> None:
    q = Quantity(np.array(2.0), 'm')
    target = Quantity(np.empty((2, 3)), 'cm')
    args = (q, Quantity(100, 'cm'), None) if operation is np.clip else (q, Quantity(100, 'cm'))
    expected = operation(2.0, 1.0, None) if operation is np.clip else operation(2.0, 1.0)
    assert operation(*args, out=target) is target
    np.testing.assert_array_equal(target.value, expected * 100)


@pytest.mark.parametrize('operation', [np.minimum, np.maximum, np.clip])
def test_numpy_selection_invalid_out_overflow(operation: Callable[..., Any]) -> None:
    q = Quantity(np.array([1e100]), 'm')
    target = Quantity(np.array([-123], dtype=np.float32), 'm')
    args = (q, Quantity(1e200, 'm'), None) if operation is np.clip else (q, Quantity(1e200, 'm'))
    with np.errstate(over='raise'):
        with pytest.raises(FloatingPointError):
            operation(*args, out=target)
    np.testing.assert_array_equal(target.value, -123)
