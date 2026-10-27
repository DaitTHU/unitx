"""Runtime boundaries underlying the dtype and indexing overloads."""

import operator
from collections.abc import Callable
from typing import Any

import pytest

from unitx import Quantity
from unitx.si import si

np = pytest.importorskip('numpy')


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('shape', [(), (3,)])
@pytest.mark.parametrize('operation', [operator.mul, operator.truediv])
def test_quantity_complex_promotion(
    dtype: str, shape: tuple[int, ...], operation: Callable[..., Any]
) -> None:
    values = np.full(shape, 2, dtype=dtype)
    for value in (values, values.dtype.type(2)):
        q = Quantity(value, si.m)
        for reverse in (False, True):
            result = operation(1j, q) if reverse else operation(q, 1j)
            ufunc = np.multiply if operation is operator.mul else np.true_divide
            expected = ufunc(1j, value) if reverse else ufunc(value, 1j)
            assert type(result.value) is type(expected)
            assert result.value.dtype == expected.dtype
            assert result.unit == (si.m**-1 if reverse and operation is operator.truediv else si.m)
            np.testing.assert_array_equal(result.value, expected)


@pytest.mark.parametrize(
    'key',
    [
        slice(None),
        (slice(None), 0, 0),
        np.array([True, False]),
        (np.array([True, False]), 0, 0),
        np.array(True),
        True,
        None,
        Ellipsis,
    ],
)
def test_quantity_index_array(key: Any) -> None:
    values = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    q = Quantity(values, si.m)
    result = q[key]
    assert isinstance(result.value, np.ndarray)
    assert result.value.dtype == values.dtype and result.unit is q.unit
    np.testing.assert_array_equal(result.value, values[key])


@pytest.mark.parametrize('key', [(), np.array(0), (0,)])
def test_quantity_index_scalar(key: Any) -> None:
    values = np.array(2.0) if isinstance(key, tuple) and not key else np.array([2.0])
    result = Quantity(values, si.m)[key]
    assert isinstance(result.value, np.float64)
    assert result == 2 * si.m


@pytest.mark.parametrize('key', [Ellipsis, None, np.array(True)])
def test_quantity_index_zero_dimensional(key: Any) -> None:
    for value in (np.array(2.0), np.float64(2)):
        result = Quantity(value, si.m)[key]
        assert isinstance(result.value, np.ndarray)
        np.testing.assert_array_equal(result.value, value[key])
