"""Ordering, searching and sets align units without stripping result values."""

from collections.abc import Callable
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx._numerics import Array
from unitx.exceptions import DimensionError
from unitx.si import si

np = pytest.importorskip('numpy')

ArrayQuantity = Quantity[Array] | AffineQuantity[Array] | LogQuantity[Array]
SETS = [np.intersect1d, np.union1d, np.setdiff1d, np.setxor1d, np.isin]


def _check_wrapped(result: Any, expected: Any, source: ArrayQuantity) -> None:
    assert type(result) is type(source) and result.unit == source.unit
    assert type(result.value) is type(expected) and result.value.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected)


@pytest.mark.parametrize('operation', [np.sort, np.argsort])
@pytest.mark.parametrize('axis', [None, 0, -1])
@pytest.mark.parametrize('kind', [None, 'stable', 'heapsort'])
def test_numpy_sort(
    array_sample: ArrayQuantity, operation: Callable[..., Any], axis: int | None, kind: str | None
) -> None:
    array_sample = type(array_sample)(array_sample.value[::-1, ::-1, ::-1], array_sample.unit)
    array_sample.value.flags.writeable = False
    expected = operation(array_sample.value, axis=axis, kind=kind)
    result = operation(array_sample, axis=axis, kind=kind)
    if operation is np.sort:
        _check_wrapped(result, expected, array_sample)
        assert not np.shares_memory(result.value, array_sample.value)
    else:
        np.testing.assert_array_equal(result, expected)
        assert type(result) is np.ndarray and result.dtype.kind in 'iu'


@pytest.mark.parametrize('operation', [np.sort, np.argsort])
def test_numpy_sort_stable(array_sample: ArrayQuantity, operation: Callable[..., Any]) -> None:
    expected = operation(array_sample.value, stable=True)
    result = operation(array_sample, stable=True)
    np.testing.assert_array_equal(result.value if operation is np.sort else result, expected)


@pytest.mark.parametrize('operation', [np.partition, np.argpartition])
@pytest.mark.parametrize('kth', [0, -1, [0, 1], np.array([1, 0])])
def test_numpy_partition(
    array_sample: ArrayQuantity, operation: Callable[..., Any], kth: Any
) -> None:
    array_sample = type(array_sample)(array_sample.value[::-1, ::-1, ::-1], array_sample.unit)
    expected = operation(array_sample.value, kth, axis=-1)
    result = operation(array_sample, kth, axis=-1)
    if operation is np.partition:
        _check_wrapped(result, expected, array_sample)
    else:
        assert type(result) is np.ndarray and result.dtype.kind in 'iu'
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('operation', [np.partition, np.argpartition])
def test_numpy_partition_invalid_kth(
    array_sample: ArrayQuantity, operation: Callable[..., Any]
) -> None:
    with pytest.raises(TypeError):
        operation(array_sample, Quantity(0, Unit()))


def test_numpy_lexsort_independent_units() -> None:
    distance = Quantity(np.array([2, 1, 2, 1]), 'm')
    time = Quantity(np.array([1, 2, 1, 2]), 's')
    point = AffineQuantity(np.array([1.0, 2, 1, 2]), 'degC')
    for keys in [(distance, time), (distance, point), (time.value, distance)]:
        values = tuple(
            key.value if isinstance(key, (Quantity, AffineQuantity)) else key for key in keys
        )
        result = np.lexsort(keys)
        assert type(result) is np.ndarray and result.dtype.kind in 'iu'
        np.testing.assert_array_equal(result, np.lexsort(values))


@pytest.mark.parametrize('side', ['left', 'right'])
def test_numpy_searchsorted_alignment(side: str) -> None:
    q = Quantity(np.array([3.0, 1, 2, 2]), 'm')
    values = Quantity(np.array([100.0, 200, 250, 400]), 'cm')
    sorter = np.argsort(q)
    result = np.searchsorted(q, values, side, sorter)
    np.testing.assert_array_equal(result, np.searchsorted(q.value, [1, 2, 2.5, 4], side, sorter))
    assert isinstance(np.searchsorted(np.sort(q), Quantity(200, 'cm')), np.integer)


@pytest.mark.parametrize('operation', SETS)
def test_numpy_set_alignment(operation: Callable[..., Any]) -> None:
    q = Quantity(np.array([1.0, 2, 2, 3]), 'm')
    other = Quantity(np.array([200.0, 400]), 'cm')
    result = operation(q, other)
    expected = operation(q.value, np.array([2.0, 4]))
    if operation is np.isin:
        assert type(result) is np.ndarray and result.dtype == bool
        np.testing.assert_array_equal(result, expected)
    else:
        _check_wrapped(result, expected, q)


@pytest.mark.parametrize('operation', SETS)
@pytest.mark.parametrize('family', [AffineQuantity, LogQuantity])
def test_numpy_set_coordinate(operation: Callable[..., Any], family: Any) -> None:
    q = family(np.array([0.0, 10, 30]), 'degC' if family is AffineQuantity else 'dBm')
    other = family(
        np.array([273.15, 283.15]) if family is AffineQuantity else np.array([0.0, 1.0]),
        'K' if family is AffineQuantity else 'dBW',
    )
    aligned = np.array([0.0, 10]) if family is AffineQuantity else np.array([30.0, 31])
    expected = operation(q.value, aligned)
    result = operation(q, other)
    if operation is np.isin:
        np.testing.assert_array_equal(result, expected)
    else:
        _check_wrapped(result, expected, q)


def test_numpy_searchsorted_coordinate() -> None:
    q = AffineQuantity(np.array([0.0, 10, 20]), 'degC')
    np.testing.assert_array_equal(
        np.searchsorted(q, AffineQuantity(np.array([273.15, 288.15]), 'K')),
        [0, 2],
    )


@pytest.mark.parametrize('operation', SETS + [np.searchsorted])
@pytest.mark.parametrize('bare_first', [False, True])
def test_numpy_set_dimensionless(operation: Callable[..., Any], bare_first: bool) -> None:
    q = Quantity(np.array([1, 2, 3]), Unit('m/cm'))
    bare = np.array([200, 400])
    a, b = (bare, q) if bare_first else (q, bare)
    raw_a, raw_b = (bare / 100, q.value) if bare_first else (q.value, bare / 100)
    result = operation(a, b)
    expected = operation(raw_a, raw_b)
    if operation in (np.isin, np.searchsorted):
        np.testing.assert_array_equal(result, expected)
    else:
        _check_wrapped(result, expected, q)


def test_numpy_intersect_indices() -> None:
    q = Quantity(np.array([3.0, 1, 2, 2]), 'm')
    other = Quantity(np.array([200.0, 300]), 'cm')
    result = np.intersect1d(q, other, return_indices=True)
    expected = np.intersect1d(q.value, np.array([2.0, 3]), return_indices=True)
    _check_wrapped(result[0], expected[0], q)
    for actual, raw in zip(result[1:], expected[1:]):
        assert type(actual) is np.ndarray and actual.dtype.kind in 'iu'
        np.testing.assert_array_equal(actual, raw)


@pytest.mark.parametrize('operation', SETS + [np.searchsorted])
@pytest.mark.parametrize(
    'other', [Quantity(np.array([1]), 's'), np.array([1]), AffineQuantity(np.array([1.0]), 'K')]
)
def test_numpy_set_invalid_dimension(operation: Callable[..., Any], other: Any) -> None:
    with pytest.raises((DimensionError, TypeError)):
        operation(Quantity(np.array([1]), 'm'), other)


@pytest.mark.parametrize('operation', SETS + [np.searchsorted])
def test_numpy_set_invalid_convention(operation: Callable[..., Any]) -> None:
    a = LogQuantity(np.array([0.0]), si.dB(si.mW))
    b = LogQuantity(np.array([0.0]), si.dB(si.mW, kind='amplitude'))
    with pytest.raises(DimensionError, match='same convention'):
        operation(a, b)


@pytest.mark.parametrize('axis', [None, 0, -1])
@pytest.mark.parametrize('flags', [(True, False, False), (False, True, True), (True, True, True)])
def test_numpy_unique_outputs(
    array_sample: ArrayQuantity, axis: int | None, flags: tuple[bool, bool, bool]
) -> None:
    q = type(array_sample)(np.array([[2.0, 1, 2], [2, 1, 2]]), array_sample.unit)
    expected = np.unique(q.value, *flags, axis)
    result = np.unique(q, *flags, axis)
    assert type(result) is tuple
    _check_wrapped(result[0], expected[0], q)
    for actual, raw in zip(result[1:], expected[1:]):
        assert type(actual) is np.ndarray and actual.dtype.kind in 'iu'
        np.testing.assert_array_equal(actual, raw)


@pytest.mark.parametrize('equal_nan', [False, True])
def test_numpy_unique_nan(array_sample: ArrayQuantity, equal_nan: bool) -> None:
    q = type(array_sample)(np.array([1.0, np.nan, 1, np.nan]), array_sample.unit)
    _check_wrapped(np.unique(q, equal_nan=equal_nan), np.unique(q.value, equal_nan=equal_nan), q)


@pytest.mark.parametrize(
    'operation', [np.unique_all, np.unique_counts, np.unique_inverse, np.unique_values]
)
def test_numpy_unique_variants(array_sample: ArrayQuantity, operation: Callable[..., Any]) -> None:
    q = type(array_sample)(np.array([[2.0, 1, 2], [2, 3, 1]]), array_sample.unit)
    expected = operation(q.value)
    result = operation(q)
    if operation is np.unique_values:
        _check_wrapped(result, expected, q)
    else:
        assert type(result) is type(expected) and result._fields == expected._fields
        _check_wrapped(result.values, expected.values, q)
        for actual, raw in zip(result[1:], expected[1:]):
            assert type(actual) is np.ndarray
            np.testing.assert_array_equal(actual, raw)


def test_numpy_unique_sorted_keyword(array_sample: ArrayQuantity) -> None:
    if np.lib.NumpyVersion(np.__version__) < '2.3.0':
        with pytest.raises(TypeError, match='sorted'):
            np.unique(array_sample, sorted=False)
    else:
        _check_wrapped(
            np.unique(array_sample, sorted=False),
            np.unique(array_sample.value, sorted=False),
            array_sample,
        )


@pytest.mark.parametrize('operation', [np.nonzero, np.flatnonzero, np.argwhere, np.count_nonzero])
def test_numpy_nonzero(array_sample: ArrayQuantity, operation: Callable[..., Any]) -> None:
    expected = operation(array_sample.value)
    result = operation(array_sample)
    if isinstance(expected, tuple):
        assert type(result) is tuple
        for actual, raw in zip(result, expected):
            np.testing.assert_array_equal(actual, raw)
    else:
        assert type(result) is type(expected)
        np.testing.assert_array_equal(result, expected)


def test_numpy_count_nonzero_axes(array_sample: ArrayQuantity) -> None:
    np.testing.assert_array_equal(
        np.count_nonzero(array_sample, axis=(0, -1), keepdims=True),
        np.count_nonzero(array_sample.value, axis=(0, -1), keepdims=True),
    )


@pytest.mark.parametrize(
    'operation', [np.sort, np.partition, np.unique, np.unique_all, np.nonzero, np.count_nonzero]
)
def test_numpy_ordering_invalid_keyword(
    array_sample: ArrayQuantity, operation: Callable[..., Any]
) -> None:
    with pytest.raises(TypeError, match='unexpected keyword'):
        if operation is np.partition:
            operation(array_sample, 0, out=array_sample)
        else:
            operation(array_sample, out=array_sample)
